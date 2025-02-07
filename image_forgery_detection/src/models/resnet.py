from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Type
from dataclasses import dataclass
import logging
from torch.cuda import amp
from .base_model import BaseForgeryDetector
from .attention import MultiHeadAttention, CBAM

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResNetConfig:
    """ResNet模型配置"""
    in_channels: List[int]
    out_channels: int
    scales: List[float] = (1.0, 0.5, 0.25)
    reduction_ratio: int = 16
    dilations: Tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    use_attention: bool = True
    attention_heads: int = 8

    @classmethod
    def from_dict(cls, config: Dict) -> 'ResNetConfig':
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


class ResNetDetector(BaseForgeryDetector):
    """优化的ResNet篡改检测器"""

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.model_config = ResNetConfig.from_dict(self.config)

        # 构建增强模块
        self.residual_connections = self._build_residual_connections()
        self.multi_scale_fusion = self._build_multi_scale_fusion()
        self.feature_enhancement = self._build_feature_enhancement()

        # 初始化权重
        self._init_weights()

        # 模型统计信息
        self._count_parameters()

    def _init_weights(self) -> None:
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _count_parameters(self) -> None:
        """统计模型参数"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _build_residual_connections(self) -> nn.ModuleList:
        """构建优化的残差连接"""
        return nn.ModuleList([
            ResidualConnection(
                in_channel,
                self.model_config.out_channels,
                self.model_config.dropout
            ) for in_channel in self.model_config.in_channels[:-1]
        ])

    def _build_multi_scale_fusion(self) -> nn.Module:
        """构建多尺度特征融合"""
        return MultiScaleFusion(
            self.model_config.out_channels,
            scales=self.model_config.scales,
            dropout=self.model_config.dropout
        )

    def _build_feature_enhancement(self) -> nn.Module:
        """构建特征增强模块"""
        return FeatureEnhancement(
            self.model_config.out_channels,
            reduction=self.model_config.reduction_ratio,
            use_attention=self.model_config.use_attention,
            num_heads=self.model_config.attention_heads
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        outputs = {}

        try:
            # 特征提取与增强
            features = self._extract_features(x)
            enhanced = self._enhance_features(features)

            # 主任务预测
            outputs.update(self._compute_main_tasks(enhanced))

            # 辅助任务预测
            outputs.update(self._compute_auxiliary_tasks(enhanced))

            return outputs

        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取并增强特征"""
        # 基础特征提取
        features = self.backbone(x)

        # 应用残差连接
        enhanced_features = []
        for i, feature in enumerate(features[:-1]):
            enhanced = self.residual_connections[i](feature, features[i + 1])
            enhanced_features.append(enhanced)
        enhanced_features.append(features[-1])

        return enhanced_features

    def _enhance_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """特征增强流程"""
        # 特征融合
        neck_features = self.neck(features)
        fused_features = self.multi_scale_fusion(neck_features)

        # 最终增强
        return self.feature_enhancement(fused_features)

    def _compute_main_tasks(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算主任务输出"""
        outputs = {}
        if 'cls' in self.head:
            outputs['cls'] = self.head['cls'](features)
        if 'seg' in self.head:
            outputs['seg'] = self.head['seg'](features)
        return outputs

    def _compute_auxiliary_tasks(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算辅助任务输出"""
        return {name: head(features) for name, head in self.auxiliary_heads.items()}


class ResidualConnection(nn.Module):
    """优化的残差连接模块"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        # 主路径
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

        # 快捷连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 尺寸对齐
        if x.shape[2:] != residual.shape[2:]:
            x = F.interpolate(x, size=residual.shape[2:],
                              mode='bilinear', align_corners=False)

        # 主路径
        out = self.main_path(x)

        # 快捷连接
        identity = self.shortcut(x)

        # 融合
        out = out + identity + residual
        return self.relu(out)


class MultiScaleFusion(nn.Module):
    """优化的多尺度特征融合模块"""

    def __init__(self, channels: int, scales: List[float] = (1.0, 0.5, 0.25),
                 dropout: float = 0.1):
        super().__init__()

        self.scales = scales

        # 多尺度分支
        self.branches = nn.ModuleList([
            DilatedConvBlock(channels, channels, dropout)
            for _ in scales
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行处理多尺度特征
        multi_scale_features = []

        for scale, branch in zip(self.scales, self.branches):
            # 动态缩放
            if scale != 1.0:
                scaled = F.interpolate(x, scale_factor=scale,
                                       mode='bilinear', align_corners=False)
            else:
                scaled = x

            feat = branch(scaled)

            # 恢复原始尺寸
            if scale != 1.0:
                feat = F.interpolate(feat, size=x.shape[2:],
                                     mode='bilinear', align_corners=False)

            multi_scale_features.append(feat)

        # 融合特征
        return self.fusion(torch.cat(multi_scale_features, dim=1))


class FeatureEnhancement(nn.Module):
    """优化的特征增强模块"""

    def __init__(self, channels: int, reduction: int = 16,
                 use_attention: bool = True, num_heads: int = 8):
        super().__init__()

        # 基于CBAM的注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 可选的多头自注意力
        self.self_attention = (
            MultiHeadAttention(channels, num_heads)
            if use_attention else None
        )

        # 特征增强
        self.enhancement = nn.Sequential(
            DilatedConvBlock(channels, channels),
            DilatedConvBlock(channels, channels)
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        x = x * self.channel_attention(x)

        # 空间注意力
        spatial_mask = self.spatial_attention(
            torch.cat([
                torch.mean(x, dim=1, keepdim=True),
                torch.max(x, dim=1, keepdim=True)[0]
            ], dim=1)
        )
        x = x * spatial_mask

        # 自注意力
        if self.self_attention is not None:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1).transpose(1, 2)
            x_flat = self.self_attention(x_flat)[0]
            x = x_flat.transpose(1, 2).view(b, c, h, w)

        # 特征增强
        return x + self.enhancement(x)


class DilatedConvBlock(nn.Module):
    """优化的空洞卷积块"""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.1,
                 dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(dilations),
                          3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels // len(dilations)),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行处理多尺度特征
        branch_outputs = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(branch_outputs, dim=1))