import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base_model import BaseForgeryDetector
import timm


class ResNetDetector(BaseForgeryDetector):
    """基于ResNet的篡改检测器"""

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # 构建残差连接
        self.residual_connections = self._build_residual_connections()

        # 构建多尺度特征融合
        self.multi_scale_fusion = self._build_multi_scale_fusion()

        # 构建深度特征增强模块
        self.feature_enhancement = self._build_feature_enhancement()

    def _build_residual_connections(self) -> nn.ModuleList:
        """构建残差连接"""
        connections = []
        in_channels = self.config['backbone']['in_channels']

        for i in range(len(in_channels) - 1):
            connections.append(
                ResidualConnection(
                    in_channels[i],
                    in_channels[i + 1]
                )
            )

        return nn.ModuleList(connections)

    def _build_multi_scale_fusion(self) -> nn.Module:
        """构建多尺度特征融合"""
        return MultiScaleFusion(
            in_channels=self.config['neck']['in_channels'],
            out_channels=self.config['neck']['out_channels'],
            scales=self.config.get('fusion_scales', [1, 0.5, 0.25])
        )

    def _build_feature_enhancement(self) -> nn.Module:
        """构建特征增强模块"""
        return FeatureEnhancement(
            in_channels=self.config['neck']['out_channels'],
            reduction=16
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取基础特征
        features = self.backbone(x)

        # 应用残差连接
        enhanced_features = []
        for i, feature in enumerate(features[:-1]):
            enhanced = self.residual_connections[i](
                feature,
                features[i + 1]
            )
            enhanced_features.append(enhanced)
        enhanced_features.append(features[-1])

        # 特征融合
        neck_features = self.neck(enhanced_features)

        # 多尺度特征融合
        fused_features = self.multi_scale_fusion(neck_features)

        # 特征增强
        enhanced_features = self.feature_enhancement(fused_features)

        outputs = {}

        # 主任务预测
        if 'cls' in self.head:
            outputs['cls'] = self.head['cls'](enhanced_features)

        if 'seg' in self.head:
            outputs['seg'] = self.head['seg'](enhanced_features)

        # 辅助任务预测
        for name, head in self.auxiliary_heads.items():
            outputs[name] = head(enhanced_features)

        return outputs


class ResidualConnection(nn.Module):
    """残差连接模块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 短连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 调整x的大小以匹配residual
        if x.shape[2:] != residual.shape[2:]:
            x = F.interpolate(
                x,
                size=residual.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 短连接
        identity = self.shortcut(x)

        # 残差连接
        out = out + identity + residual
        out = self.relu(out)

        return out


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, in_channels: int, out_channels: int,
                 scales: List[float] = [1, 0.5, 0.25]):
        super().__init__()

        self.scales = scales

        # 多尺度处理分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(scales), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        multi_scale_features = []

        for scale, branch in zip(self.scales, self.branches):
            # 缩放输入
            if scale != 1:
                h = int(x.shape[2] * scale)
                w = int(x.shape[3] * scale)
                scaled = F.interpolate(
                    x,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                scaled = x

            # 处理特征
            feat = branch(scaled)

            # 恢复原始尺寸
            if scale != 1:
                feat = F.interpolate(
                    feat,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            multi_scale_features.append(feat)

        # 拼接并融合特征
        out = torch.cat(multi_scale_features, dim=1)
        out = self.fusion(out)

        return out


class FeatureEnhancement(nn.Module):
    """特征增强模块"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 特征增强
        self.enhancement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        # 特征增强
        out = self.enhancement(x)

        # 残差连接
        out = out + x

        return out


class DilatedEncoder(nn.Module):
    """带空洞卷积的编码器"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          dilation=2 ** i, padding=2 ** i),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(4)  # 空洞率：1,2,4,8
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))

        # 特征融合
        out = torch.cat(features, dim=1)
        out = self.fusion(out)

        return out