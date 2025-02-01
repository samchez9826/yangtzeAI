from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LossConfig:
    """损失函数配置"""
    name: str  # 损失函数名称
    alpha: float = 0.25  # Focal Loss的alpha参数
    gamma: float = 2.0  # Focal Loss的gamma参数
    smooth: float = 1.0  # Dice Loss的平滑参数
    eps: float = 1e-7  # 数值稳定性参数
    temperature: float = 0.5  # 温度参数
    edge_width: int = 3  # 边缘宽度
    weights: Optional[Dict[str, float]] = None  # 多任务损失的权重


class LossBase(nn.Module, ABC):
    """损失函数基类"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError


class FocalLoss(LossBase):
    """Focal Loss"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    @torch.cuda.amp.autocast()
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Focal Loss"""
        # 处理输入维度
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N, C, HW]
            inputs = inputs.transpose(1, 2)  # [N, HW, C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # [NHW, C]

        targets = targets.view(-1, 1)

        # 计算log概率
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        # 计算focal loss
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        # 降维
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(LossBase):
    """Dice Loss"""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    @torch.cuda.amp.autocast()
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Dice Loss"""
        num_classes = inputs.size(1)

        if num_classes == 1:
            return self._binary_dice_loss(torch.sigmoid(inputs), targets)

        # 多类别
        dice = 0.
        for i in range(num_classes):
            dice += self._binary_dice_loss(inputs[:, i, ...], targets == i)

        return dice / num_classes

    def _binary_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算二分类的Dice Loss"""
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1. - dice


class EdgeConsistencyLoss(LossBase):
    """边缘一致性损失"""

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

    @torch.cuda.amp.autocast()
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算边缘一致性损失"""
        # 计算梯度
        pred_grad_x = F.conv2d(inputs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(inputs, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(targets, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(targets, self.sobel_y, padding=1)

        # 计算边缘强度
        pred_edge = torch.sqrt(pred_grad_x.pow(2) + pred_grad_y.pow(2))
        target_edge = torch.sqrt(target_grad_x.pow(2) + target_grad_y.pow(2))

        # 扩展边缘区域
        target_edge = F.max_pool2d(
            target_edge,
            kernel_size=self.edge_width,
            stride=1,
            padding=self.edge_width // 2
        )

        # 计算边缘区域的损失
        return F.mse_loss(pred_edge * target_edge, target_edge)


class ConsistencyLoss(LossBase):
    """特征一致性损失"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    @torch.cuda.amp.autocast()
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """计算特征一致性损失"""
        # 归一化特征
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # 计算相似度矩阵
        batch_size = features1.size(0)
        features1_flat = features1.view(batch_size, -1)
        features2_flat = features2.view(batch_size, -1)

        similarity = torch.matmul(features1_flat, features2_flat.t()) / self.temperature
        labels = torch.arange(batch_size, device=similarity.device)

        return F.cross_entropy(similarity, labels)


class MultiTaskLoss(nn.Module):
    """多任务损失"""

    def __init__(self, tasks: Dict[str, LossBase],
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.tasks = nn.ModuleDict(tasks)
        self.weights = weights or {task: 1.0 for task in tasks.keys()}

    @torch.cuda.amp.autocast()
    def forward(self, inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多任务损失"""
        losses = {}
        total_loss = 0.

        # 计算每个任务的损失
        for task_name, criterion in self.tasks.items():
            if task_name in inputs and task_name in targets:
                try:
                    loss = criterion(inputs[task_name], targets[task_name])
                    weighted_loss = self.weights[task_name] * loss
                    losses[task_name] = loss
                    total_loss += weighted_loss
                except Exception as e:
                    print(f"Error computing loss for task {task_name}: {e}")
                    continue

        losses['total'] = total_loss
        return losses


def create_criterion(config: LossConfig) -> Dict[str, nn.Module]:
    """
    创建损失函数
    Args:
        config: 损失函数配置
    Returns:
        损失函数字典
    """
    criterion = {}

    # 分类损失
    if config.name == 'focal':
        criterion['cls'] = FocalLoss(
            alpha=config.alpha,
            gamma=config.gamma
        )
    elif config.name == 'cross_entropy':
        criterion['cls'] = nn.CrossEntropyLoss()

    # 分割损失
    if config.name == 'dice':
        criterion['seg'] = DiceLoss(
            smooth=config.smooth,
            eps=config.eps
        )
    elif config.name == 'bce':
        criterion['seg'] = nn.BCEWithLogitsLoss()

    # 边缘一致性损失
    if config.name == 'edge':
        criterion['edge'] = EdgeConsistencyLoss(
            edge_width=config.edge_width
        )

    # 特征一致性损失
    if config.name == 'consistency':
        criterion['consistency'] = ConsistencyLoss(
            temperature=config.temperature
        )

    # 如果有多个任务且指定了权重
    if len(criterion) > 1 and config.weights:
        return MultiTaskLoss(criterion, config.weights)

    return criterion