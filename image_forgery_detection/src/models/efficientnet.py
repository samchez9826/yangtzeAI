from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from .base_model import BaseForgeryDetector
from .attention import MultiHeadAttention, CBAM
from .decoder import DecoderConfig, create_decoder


@dataclass
class AttentionConfig:
    """注意力模块配置"""
    enabled: bool = False
    num_heads: int = 8
    dropout: float = 0.1
    reduction_ratio: int = 16


@dataclass
class FusionConfig:
    """特征融合配置"""
    type: str = 'concat'  # ['concat', 'sum', 'attention']
    dropout: float = 0.1


class EfficientNetDetector(BaseForgeryDetector):
    """基于EfficientNet的篡改检测器"""

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # 注意力模块
        self.attention = self._build_attention()

        # 特征融合模块
        self.fusion = self._build_fusion()

    def _build_attention(self) -> nn.ModuleDict:
        """构建注意力模块"""
        attention_modules = {}

        if self.config.attention.get('self_attention', AttentionConfig()).enabled:
            attn_cfg = self.config.attention.self_attention
            attention_modules['self'] = MultiHeadAttention(
                dim=self.config.neck.out_channels,
                num_heads=attn_cfg.num_heads,
                dropout=attn_cfg.dropout
            )

        if self.config.attention.get('cbam_attention', AttentionConfig()).enabled:
            attn_cfg = self.config.attention.cbam_attention
            attention_modules['cbam'] = CBAM(
                channels=self.config.neck.out_channels,
                reduction=attn_cfg.reduction_ratio
            )

        return nn.ModuleDict(attention_modules)

    def _build_fusion(self) -> nn.Module:
        """构建特征融合模块"""
        fusion_cfg = self.config.get('fusion', FusionConfig())
        return FeatureFusionModule(
            in_channels=self.config.neck.out_channels,
            fusion_type=fusion_cfg.type,
            dropout=fusion_cfg.dropout
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入图像张量 [B, C, H, W]
        Returns:
            dict: 模型输出字典
        """
        # 特征提取
        features = self.backbone(x)

        # 颈部特征融合
        neck_features = self.neck(features)

        # 应用注意力机制
        attended_features = neck_features
        for name, attention in self.attention.items():
            attended_features = attention(attended_features)

        # 特征融合
        fused_features = self.fusion(attended_features)

        outputs = {}

        # 主任务预测
        if 'cls' in self.head:
            outputs['cls'] = self.head['cls'](fused_features)
        if 'seg' in self.head:
            outputs['seg'] = self.head['seg'](fused_features)

        # 辅助任务预测
        for name, head in self.auxiliary_heads.items():
            outputs[name] = head(fused_features)

        return outputs


class FeatureFusionModule(nn.Module):
    """特征融合模块"""

    def __init__(self, in_channels: int,
                 fusion_type: str = 'concat',
                 dropout: float = 0.1):
        """
        初始化
        Args:
            in_channels: 输入通道数
            fusion_type: 融合类型,支持 concat/sum/attention
            dropout: Dropout率
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.dropout = dropout

        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        elif fusion_type == 'sum':
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        elif fusion_type == 'attention':
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid(),
                nn.Dropout2d(dropout)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            融合后的特征
        """
        if self.fusion_type == 'concat':
            # 拼接 + 平均池化特征
            pooled = F.avg_pool2d(x, 2)
            x = torch.cat([x, pooled], dim=1)
            return self.fusion(x)
        elif self.fusion_type == 'sum':
            # 相加
            return self.fusion(x + F.avg_pool2d(x, 2))
        else:  # attention
            # 注意力加权
            weights = self.fusion(x)
            return x * weights


class ClassificationHead(nn.Module):
    """分类头"""

    def __init__(self, in_channels: int, hidden_dims: List[int],
                 num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        layers = []
        curr_channels = in_channels

        # MLP层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(curr_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ])
            curr_channels = hidden_dim

        # 最终分类
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(curr_channels, num_classes, 1),
            nn.Flatten()
        ])

        self.classifier = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            分类得分 [B, num_classes]
        """
        return self.classifier(x)


class SegmentationHead(nn.Module):
    """分割头"""

    def __init__(self, in_channels: int, out_channels: int,
                 decoder_channels: List[int]):
        """
        初始化
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            decoder_channels: 解码器通道数列表
        """
        super().__init__()

        # 使用统一的decoder配置
        decoder_config = DecoderConfig(
            in_channels=[in_channels],
            out_channels=out_channels,
            skip_channels=decoder_channels
        )

        self.decoder = create_decoder('unet', decoder_config)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            分割掩码 [B, out_channels, H, W]
        """
        return self.decoder([x])