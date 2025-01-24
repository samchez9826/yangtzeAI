import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for 图像篡改检测"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        初始化Focal Loss
        Args:
            alpha: 平衡正负样本的权重
            gamma: 聚焦参数,用于降低易分样本的权重
            reduction: 损失计算方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        Args:
            inputs: 预测值 (N, C) 或 (N, C, H, W)
            targets: 目标值 (N) 或 (N, H, W)
        Returns:
            计算的损失值
        """
        # 确保输入是概率分布
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)  # (N, H*W, C)
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # (N*H*W, C)
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        # 计算focal loss
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for 分割任务"""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        """
        初始化Dice Loss
        Args:
            smooth: 平滑参数,防止分母为0
            eps: 极小值,防止数值不稳定
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss
        Args:
            inputs: 预测的概率图 (N, C, H, W)
            targets: 目标掩码 (N, H, W)
        Returns:
            计算的损失值
        """
        num_classes = inputs.size(1)
        if num_classes == 1:
            inputs = torch.sigmoid(inputs)
            targets = targets.view(-1)
            inputs = inputs.view(-1)

            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()

            dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
            return 1. - dice

        # 多类别
        dice = 0.
        for i in range(num_classes):
            dice += self._binary_dice_loss(inputs[:, i, ...],
                                           targets == i)
        return dice / num_classes

    def _binary_dice_loss(self, inputs: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """计算二分类的Dice Loss"""
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1. - dice


class EdgeConsistencyLoss(nn.Module):
    """边缘一致性损失"""

    def __init__(self, edge_width: int = 3):
        """
        初始化边缘一致性损失
        Args:
            edge_width: 边缘宽度
        """
        super().__init__()
        self.edge_width = edge_width

        # Sobel算子
        self.sobel_x = torch.FloatTensor([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.FloatTensor([[-1, -2, -1],
                                          [0, 0, 0],
                                          [1, 2, 1]]).view(1, 1, 3, 3)

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算边缘一致性损失
        Args:
            inputs: 预测图 (N, C, H, W)
            targets: 目标图 (N, C, H, W)
        Returns:
            边缘一致性损失
        """
        if not hasattr(self, 'sobel_x') or self.sobel_x.device != inputs.device:
            self.sobel_x = self.sobel_x.to(inputs.device)
            self.sobel_y = self.sobel_y.to(inputs.device)

        # 计算梯度
        pred_grad_x = F.conv2d(inputs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(inputs, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(targets, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(targets, self.sobel_y, padding=1)

        # 计算边缘强度
        pred_edge = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
        target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)

        # 扩展边缘区域
        target_edge = F.max_pool2d(target_edge, kernel_size=self.edge_width,
                                   stride=1, padding=self.edge_width // 2)

        # 计算边缘区域的损失
        loss = F.mse_loss(pred_edge * target_edge, target_edge)

        return loss


class ConsistencyLoss(nn.Module):
    """特征一致性损失"""

    def __init__(self, temperature: float = 0.5):
        """
        初始化一致性损失
        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features1: torch.Tensor,
                features2: torch.Tensor) -> torch.Tensor:
        """
        计算特征一致性损失
        Args:
            features1: 特征1 (N, C, H, W)
            features2: 特征2 (N, C, H, W)
        Returns:
            一致性损失
        """
        # 归一化特征
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(features1.view(features1.size(0), -1),
                                  features2.view(features2.size(0), -1).t())

        # 应用温度缩放
        similarity = similarity / self.temperature

        # 对角线元素应该最大(相同样本的特征应该最相似)
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        loss = F.cross_entropy(similarity, labels)

        return loss


class MultiTaskLoss(nn.Module):
    """多任务损失"""

    def __init__(self, tasks: Dict[str, nn.Module],
                 weights: Optional[Dict[str, float]] = None):
        """
        初始化多任务损失
        Args:
            tasks: 任务名称到损失函数的映射
            weights: 各任务的权重
        """
        super().__init__()
        self.tasks = nn.ModuleDict(tasks)
        self.weights = weights or {task: 1.0 for task in tasks.keys()}

    def forward(self, inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        Args:
            inputs: 各任务的预测值
            targets: 各任务的目标值
        Returns:
            各任务的损失值及总损失
        """
        losses = {}
        total_loss = 0.

        for task_name, criterion in self.tasks.items():
            if task_name in inputs and task_name in targets:
                loss = criterion(inputs[task_name], targets[task_name])
                weighted_loss = self.weights[task_name] * loss
                losses[task_name] = loss
                total_loss += weighted_loss

        losses['total'] = total_loss
        return losses


def build_criterion(config: Dict) -> Dict[str, nn.Module]:
    """
    构建损失函数
    Args:
        config: 损失函数配置
    Returns:
        损失函数字典
    """
    criterion = {}

    # 分类损失
    if 'classification' in config:
        cls_cfg = config['classification']
        if cls_cfg['name'] == 'focal':
            criterion['cls'] = FocalLoss(
                alpha=cls_cfg['params']['alpha'],
                gamma=cls_cfg['params']['gamma']
            )
        elif cls_cfg['name'] == 'cross_entropy':
            criterion['cls'] = nn.CrossEntropyLoss()

    # 分割损失
    if 'segmentation' in config:
        seg_cfg = config['segmentation']
        if seg_cfg['name'] == 'dice':
            criterion['seg'] = DiceLoss()
        elif seg_cfg['name'] == 'bce':
            criterion['seg'] = nn.BCEWithLogitsLoss()

    # 边缘一致性损失
    if 'edge' in config:
        criterion['edge'] = EdgeConsistencyLoss(
            edge_width=config['edge'].get('edge_width', 3)
        )

    # 特征一致性损失
    if 'consistency' in config:
        criterion['consistency'] = ConsistencyLoss(
            temperature=config['consistency'].get('temperature', 0.5)
        )

    return criterion