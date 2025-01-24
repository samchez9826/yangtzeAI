import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base_model import BaseForgeryDetector


class EfficientNetDetector(BaseForgeryDetector):
    """基于EfficientNet的篡改检测器"""

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # 添加注意力机制
        self.attention = self._build_attention()

        # 构建特征融合模块
        self.fusion = self._build_fusion()

    def _build_attention(self) -> nn.ModuleDict:
        """构建注意力模块"""
        attention_modules = {}

        if self.config.get('self_attention', {}).get('enabled', False):
            # 构建自注意力
            attention_modules['self'] = SelfAttentionModule(
                in_channels=self.config['neck']['out_channels'],
                num_heads=self.config['self_attention']['num_heads'],
                dropout=self.config['self_attention']['dropout']
            )

        if self.config.get('cbam_attention', {}).get('enabled', False):
            # 构建CBAM注意力
            attention_modules['cbam'] = CBAM(
                in_channels=self.config['neck']['out_channels'],
                reduction_ratio=self.config['cbam_attention']['reduction_ratio']
            )

        return nn.ModuleDict(attention_modules)

    def _build_fusion(self) -> nn.Module:
        """构建特征融合模块"""
        return FeatureFusionModule(
            in_channels=self.config['neck']['out_channels'],
            fusion_type=self.config.get('fusion_type', 'concat')
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入图像张量 (B, C, H, W)
        Returns:
            模型输出字典
        """
        # 特征提取
        features = self.backbone(x)

        # 颈部特征融合
        neck_features = self.neck(features)

        # 应用注意力机制
        attended_features = neck_features
        for name, attention in self.attention.items():
            if name == 'self':
                attended_features = attention(attended_features)
            elif name == 'cbam':
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

    class SelfAttentionModule(nn.Module):
        """多头自注意力模块"""

        def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()

            self.num_heads = num_heads
            self.head_dim = in_channels // num_heads
            self.scale = self.head_dim ** -0.5

            # 定义注意力层
            self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
            self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm([in_channels, 1, 1])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape

            # 生成Q、K、V
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
            qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 计算注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            # 应用注意力
            x = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
            x = self.proj(x)

            # Layer Normalization
            x = x + self.norm(x)

            return x

    class CBAM(nn.Module):
        """CBAM: Convolutional Block Attention Module"""

        def __init__(self, in_channels: int, reduction_ratio: int = 16):
            super().__init__()

            # 通道注意力
            self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
            # 空间注意力
            self.spatial_attention = SpatialAttention()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 应用通道注意力
            x = self.channel_attention(x)
            # 应用空间注意力
            x = self.spatial_attention(x)
            return x

    class ChannelAttention(nn.Module):
        """通道注意力模块"""

        def __init__(self, in_channels: int, reduction_ratio: int = 16):
            super().__init__()

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            # 共享MLP
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 平均池化分支
            avg_out = self.mlp(self.avg_pool(x))
            # 最大池化分支
            max_out = self.mlp(self.max_pool(x))

            # 融合并应用sigmoid激活
            out = torch.sigmoid(avg_out + max_out)

            return x * out

    class SpatialAttention(nn.Module):
        """空间注意力模块"""

        def __init__(self, kernel_size: int = 7):
            super().__init__()

            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 生成空间注意力图
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)

            out = self.conv(out)
            out = self.sigmoid(out)

            return x * out

    class FeatureFusionModule(nn.Module):
        """特征融合模块"""

        def __init__(self, in_channels: int, fusion_type: str = 'concat'):
            super().__init__()

            self.fusion_type = fusion_type

            if fusion_type == 'concat':
                self.fusion = nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            elif fusion_type == 'sum':
                self.fusion = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            elif fusion_type == 'attention':
                self.fusion = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    nn.BatchNorm2d(in_channels),
                    nn.Sigmoid()
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.fusion_type == 'concat':
                # 将特征在通道维度上拼接
                x = torch.cat([x, F.avg_pool2d(x, 2)], dim=1)
                return self.fusion(x)
            elif self.fusion_type == 'sum':
                # 直接相加
                return self.fusion(x + F.avg_pool2d(x, 2))
            elif self.fusion_type == 'attention':
                # 注意力加权
                attention = self.fusion(x)
                return x * attention

    class FeaturePyramidNetwork(nn.Module):
        """特征金字塔网络"""

        def __init__(self, in_channels: List[int], out_channels: int):
            super().__init__()

            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(in_channel, out_channels, kernel_size=1)
                for in_channel in in_channels
            ])

            self.fpn_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in range(len(in_channels))
            ])

        def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
            # 侧向连接
            laterals = [
                lateral_conv(x[i])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]

            # 自顶向下的路径
            for i in range(len(laterals) - 1, 0, -1):
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i],
                    size=laterals[i - 1].shape[-2:],
                    mode='nearest'
                )

            # 最终卷积
            outs = [
                fpn_conv(lateral)
                for fpn_conv, lateral in zip(self.fpn_convs, laterals)
            ]

            return outs

    class AtrousSpatialPyramidPooling(nn.Module):
        """ASPP: 空洞空间金字塔池化"""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            rates = [6, 12, 18]

            self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=rates[0], dilation=rates[0])
            self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=rates[1], dilation=rates[1])
            self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=rates[2], dilation=rates[2])

            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            aspp1 = self.aspp1(x)
            aspp2 = self.aspp2(x)
            aspp3 = self.aspp3(x)
            aspp4 = self.aspp4(x)

            global_feat = self.global_avg_pool(x)
            global_feat = F.interpolate(
                global_feat,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

            out = torch.cat([aspp1, aspp2, aspp3, aspp4, global_feat], dim=1)
            out = self.fusion(out)

            return out

    class ClassificationHead(nn.Module):
        """分类头"""

        def __init__(self, in_channels: int, hidden_dims: List[int],
                     num_classes: int, dropout: float = 0.5):
            super().__init__()

            layers = []
            curr_channels = in_channels

            # 构建MLP层
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Conv2d(curr_channels, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout)
                ])
                curr_channels = hidden_dim

            # 全局平均池化
            layers.append(nn.AdaptiveAvgPool2d(1))

            # 分类器
            layers.extend([
                nn.Conv2d(curr_channels, num_classes, kernel_size=1),
                nn.Flatten()
            ])

            self.classifier = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(x)

    class SegmentationHead(nn.Module):
        """分割头"""

        def __init__(self, in_channels: int, out_channels: int,
                     decoder_channels: List[int]):
            super().__init__()

            self.decoder = nn.ModuleList()
            curr_channels = in_channels

            # 构建解码器
            for decoder_channel in decoder_channels:
                self.decoder.append(nn.Sequential(
                    nn.Conv2d(curr_channels, decoder_channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(decoder_channel),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                ))
                curr_channels = decoder_channel

            # 最后的预测层
            self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for decoder_layer in self.decoder:
                x = decoder_layer(x)
            return self.final_conv(x)