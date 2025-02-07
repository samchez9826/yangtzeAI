from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from torch.cuda import amp

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """损失函数配置"""
    name: str
    alpha: float = 0.25  # Focal Loss的alpha参数
    gamma: float = 2.0  # Focal Loss的gamma参数
    smooth: float = 1.0  # Dice Loss的平滑参数
    eps: float = 1e-7  # 数值稳定性参数
    temperature: float = 0.5  # 温度参数
    edge_width: int = 3  # 边缘宽度
    weights: Optional[Dict[str, float]] = None  # 多任务损失权重

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LossConfig':
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


class LossBase(nn.Module, ABC):
    """损失函数基类"""

    @abstractmethod
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass

    def _check_inputs(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """输入验证"""
        assert inputs.device == targets.device, "输入和目标必须在同一设备上"
        assert inputs.size(0) == targets.size(0), "输入和目标的批次大小必须相同"


class FocalLoss(LossBase):
    """优化的Focal Loss实现"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    @amp.autocast()
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._check_inputs(inputs, targets)

        # 高效的维度处理
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N, C, HW]
            inputs = inputs.transpose(1, 2)  # [N, HW, C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # [NHW, C]
        targets = targets.view(-1, 1)

        # 使用log_softmax提高数值稳定性
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets).view(-1)
        pt = logpt.exp()

        # 向量化计算focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * logpt

        # 降维
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(LossBase):
    """优化的Dice Loss实现"""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    @amp.autocast()
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._check_inputs(inputs, targets)
        num_classes = inputs.size(1)

        if num_classes == 1:
            return self._binary_dice_loss(self.sigmoid(inputs), targets)

        # 多类别情况下的并行计算
        inputs = F.softmax(inputs, dim=1)
        dice_losses = torch.stack([
            self._binary_dice_loss(inputs[:, i], targets == i)
            for i in range(num_classes)
        ])

        return dice_losses.mean()

    def _binary_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 展平并计算交集
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()

        # 计算并集
        union = inputs.sum() + targets.sum()

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1. - dice


class EdgeConsistencyLoss(LossBase):
    """优化的边缘一致性损失"""

    def __init__(self, edge_width: int = 3):
        super().__init__()
        self.edge_width = edge_width
        self.register_buffer('sobel_x', torch.FloatTensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.FloatTensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).view(1, 1, 3, 3))

    @amp.autocast()
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._check_inputs(inputs, targets)

        # 批量计算梯度
        grads = self._compute_gradients(inputs, targets)
        pred_edge, target_edge = self._compute_edge_maps(grads)

        # 扩展边缘区域
        target_edge = F.max_pool2d(
            target_edge,
            kernel_size=self.edge_width,
            stride=1,
            padding=self.edge_width // 2
        )

        return F.mse_loss(pred_edge * target_edge, target_edge)

    def _compute_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'pred_x': F.conv2d(inputs, self.sobel_x, padding=1),
            'pred_y': F.conv2d(inputs, self.sobel_y, padding=1),
            'target_x': F.conv2d(targets, self.sobel_x, padding=1),
            'target_y': F.conv2d(targets, self.sobel_y, padding=1)
        }

    def _compute_edge_maps(self, grads: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_edge = torch.sqrt(grads['pred_x'].pow(2) + grads['pred_y'].pow(2))
        target_edge = torch.sqrt(grads['target_x'].pow(2) + grads['target_y'].pow(2))
        return pred_edge, target_edge


class ConsistencyLoss(LossBase):
    """优化的特征一致性损失"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    @amp.autocast()
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        self._check_inputs(features1, features2)

        # 特征归一化
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # 高效的批量相似度计算
        batch_size = features1.size(0)
        features1_flat = features1.view(batch_size, -1)
        features2_flat = features2.view(batch_size, -1)

        # 计算相似度矩阵并应用温度缩放
        similarity = torch.matmul(features1_flat, features2_flat.t()) / self.temperature
        labels = torch.arange(batch_size, device=similarity.device)

        return self.criterion(similarity, labels)


class MultiTaskLoss(nn.Module):
    """优化的多任务损失"""

    def __init__(self, tasks: Dict[str, LossBase], weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.tasks = nn.ModuleDict(tasks)
        self.weights = weights or {task: 1.0 for task in tasks.keys()}

        # 验证权重配置
        self._validate_weights()

    def _validate_weights(self) -> None:
        """验证权重配置"""
        missing_tasks = set(self.tasks.keys()) - set(self.weights.keys())
        if missing_tasks:
            raise ValueError(f"Missing weights for tasks: {missing_tasks}")

    @amp.autocast()
    def forward(self, inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = 0.

        # 并行计算各任务损失
        for task_name, criterion in self.tasks.items():
            if task_name in inputs and task_name in targets:
                try:
                    loss = criterion(inputs[task_name], targets[task_name])
                    weighted_loss = self.weights[task_name] * loss
                    losses[task_name] = loss.detach()  # 分离用于监控的loss
                    total_loss += weighted_loss
                except Exception as e:
                    logger.error(f"Error computing loss for task {task_name}: {e}")
                    continue

        losses['total'] = total_loss
        return losses


def create_criterion(config: Dict[str, Any]) -> Dict[str, nn.Module]:
    """工厂函数:创建损失函数"""
    loss_config = LossConfig.from_dict(config)
    criterion = {}

    try:
        # 分类损失
        if config.get('classification'):
            criterion['cls'] = FocalLoss(
                alpha=loss_config.alpha,
                gamma=loss_config.gamma
            ) if loss_config.name == 'focal' else nn.CrossEntropyLoss()

        # 分割损失
        if config.get('segmentation'):
            criterion['seg'] = DiceLoss(
                smooth=loss_config.smooth,
                eps=loss_config.eps
            ) if loss_config.name == 'dice' else nn.BCEWithLogitsLoss()

        # 边缘一致性损失
        if config.get('edge'):
            criterion['edge'] = EdgeConsistencyLoss(
                edge_width=loss_config.edge_width
            )

        # 特征一致性损失
        if config.get('consistency'):
            criterion['consistency'] = ConsistencyLoss(
                temperature=loss_config.temperature
            )

        # 多任务整合
        if len(criterion) > 1 and loss_config.weights:
            return MultiTaskLoss(criterion, loss_config.weights)

        return criterion

    except Exception as e:
        logger.error(f"Error creating criterion: {e}")
        raise