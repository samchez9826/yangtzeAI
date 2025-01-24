import torch
import torch.nn as nn
import timm
from typing import Dict, List, Optional, Tuple, Union
import logging


class EncoderBase(nn.Module):
    """编码器基类"""

    def __init__(self, name: str, pretrained: bool = True,
                 features_only: bool = True):
        """
        初始化编码器
        Args:
            name: 编码器名称
            pretrained: 是否使用预训练权重
            features_only: 是否只返回特征
        """
        super().__init__()
        self.name = name
        self.features_only = features_only

        # 加载预训练模型
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=features_only
        )

        # 获取特征维度
        self.feature_info = self.model.feature_info

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """前向传播"""
        return self.model(x)


class PyramidEncoder(EncoderBase):
    """金字塔特征编码器"""

    def __init__(self, name: str, pretrained: bool = True,
                 features_only: bool = True, out_indices: Tuple = (1, 2, 3, 4)):
        """
        初始化金字塔编码器
        Args:
            name: 编码器名称
            pretrained: 是否使用预训练权重
            features_only: 是否只返回特征
            out_indices: 输出特征的层索引
        """
        super().__init__(name, pretrained, features_only)

        # 设置输出层
        self.model.out_indices = out_indices

        # 获取每层特征维度
        self.channels = [
            self.feature_info.channels()[i]
            for i in range(len(out_indices))
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            多尺度特征列表
        """
        return self.model(x)


class MultiScaleEncoder(EncoderBase):
    """多尺度特征编码器"""

    def __init__(self, name: str, pretrained: bool = True,
                 scales: List[float] = [1.0, 0.75, 0.5]):
        """
        初始化多尺度编码器
        Args:
            name: 编码器名称
            pretrained: 是否使用预训练权重
            scales: 特征尺度列表
        """
        super().__init__(name, pretrained, features_only=True)
        self.scales = scales

        # 特征融合层
        in_channels = self.feature_info.channels()[-1]
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            融合后的特征
        """
        features = []

        # 多尺度特征提取
        for scale in self.scales:
            if scale != 1.0:
                size = (int(x.shape[2] * scale), int(x.shape[3] * scale))
                scaled = nn.functional.interpolate(
                    x, size=size, mode='bilinear', align_corners=True
                )
            else:
                scaled = x

            feat = self.model(scaled)[-1]  # 取最后一层特征

            # 调整特征大小
            if scale != 1.0:
                feat = nn.functional.interpolate(
                    feat,
                    size=(x.shape[2] // 32, x.shape[3] // 32),  # 假设下采样率为32
                    mode='bilinear',
                    align_corners=True
                )

            features.append(feat)

        # 特征融合
        multi_scale_features = torch.cat(features, dim=1)
        fused_features = self.fusion(multi_scale_features)

        return fused_features


class AttentionEncoder(EncoderBase):
    """带注意力机制的编码器"""

    def __init__(self, name: str, pretrained: bool = True,
                 attention_type: str = 'self'):
        """
        初始化注意力编码器
        Args:
            name: 编码器名称
            pretrained: 是否使用预训练权重
            attention_type: 注意力类型 ['self', 'cbam']
        """
        super().__init__(name, pretrained, features_only=True)

        in_channels = self.feature_info.channels()[-1]

        # 选择注意力机制
        if attention_type == 'self':
            self.attention = SelfAttention(in_channels)
        elif attention_type == 'cbam':
            self.attention = CBAM(in_channels)
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            特征和注意力图
        """
        features = self.model(x)[-1]  # 取最后一层特征
        features, attention_map = self.attention(features)
        return features, attention_map

    class DilatedEncoder(EncoderBase):
        """空洞卷积编码器"""

        def __init__(self, name: str, pretrained: bool = True,
                     dilations: List[int] = [6, 12, 18]):
            """
            初始化空洞卷积编码器
            Args:
                name: 编码器名称
                pretrained: 是否使用预训练权重
                dilations: 空洞率列表
            """
            super().__init__(name, pretrained, features_only=True)

            in_channels = self.feature_info.channels()[-1]

            # 空洞卷积分支
            self.aspp_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3,
                              padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                ) for dilation in dilations
            ])

            # 1x1卷积分支
            self.aspp_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))

            # 全局上下文分支
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

            # 特征融合
            total_channels = in_channels * (len(dilations) + 2)
            self.fusion = nn.Sequential(
                nn.Conv2d(total_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向传播
            Args:
                x: 输入张量
            Returns:
                融合后的特征
            """
            features = self.model(x)[-1]  # 取最后一层特征

            # ASPP分支
            aspp_outputs = []
            for branch in self.aspp_branches:
                aspp_outputs.append(branch(features))

            # 全局上下文分支
            global_context = self.global_branch(features)
            global_context = nn.functional.interpolate(
                global_context,
                size=features.shape[2:],
                mode='bilinear',
                align_corners=True
            )
            aspp_outputs.append(global_context)

            # 特征融合
            concat_features = torch.cat(aspp_outputs, dim=1)
            fused_features = self.fusion(concat_features)

            return fused_features

    class SelfAttention(nn.Module):
        """自注意力模块"""

        def __init__(self, in_channels: int, heads: int = 8):
            """
            初始化自注意力模块
            Args:
                in_channels: 输入通道数
                heads: 注意力头数
            """
            super().__init__()
            self.heads = heads
            self.scale = (in_channels // heads) ** -0.5

            self.query = nn.Conv2d(in_channels, in_channels, 1)
            self.key = nn.Conv2d(in_channels, in_channels, 1)
            self.value = nn.Conv2d(in_channels, in_channels, 1)

            self.proj = nn.Conv2d(in_channels, in_channels, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            前向传播
            Args:
                x: 输入特征 [B, C, H, W]
            Returns:
                注意力特征和注意力图
            """
            B, C, H, W = x.shape

            # 生成Q、K、V
            q = self.query(x).view(B, self.heads, C // self.heads, -1)
            k = self.key(x).view(B, self.heads, C // self.heads, -1)
            v = self.value(x).view(B, self.heads, C // self.heads, -1)

            # 计算注意力
            attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
            attn = attn.softmax(dim=-1)

            # 应用注意力
            out = torch.matmul(attn, v.transpose(-2, -1))
            out = out.view(B, C, H, W)

            # 投影
            out = self.proj(out)

            return out, attn.mean(dim=1)  # 返回特征和注意力图

    class CBAM(nn.Module):
        """CBAM注意力模块"""

        def __init__(self, in_channels: int, reduction: int = 16):
            """
            初始化CBAM模块
            Args:
                in_channels: 输入通道数
                reduction: 通道降维比例
            """
            super().__init__()

            # 通道注意力
            self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.channel_max_pool = nn.AdaptiveMaxPool2d(1)

            self.channel_mlp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1)
            )

            # 空间注意力
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            前向传播
            Args:
                x: 输入特征
            Returns:
                注意力特征和注意力图
            """
            # 通道注意力
            avg_out = self.channel_mlp(self.channel_avg_pool(x))
            max_out = self.channel_mlp(self.channel_max_pool(x))
            channel_attn = torch.sigmoid(avg_out + max_out)

            x = x * channel_attn

            # 空间注意力
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_feat = torch.cat([avg_out, max_out], dim=1)
            spatial_attn = torch.sigmoid(self.spatial_conv(spatial_feat))

            out = x * spatial_attn

            # 返回特征和组合注意力图
            attn = channel_attn * spatial_attn
            return out, attn

    class EncoderFactory:
        """编码器工厂类"""

        @staticmethod
        def create(encoder_type: str, **kwargs) -> EncoderBase:
            """
            创建编码器实例
            Args:
                encoder_type: 编码器类型
                **kwargs: 其他参数
            Returns:
                编码器实例
            """
            encoders = {
                'pyramid': PyramidEncoder,
                'multiscale': MultiScaleEncoder,
                'attention': AttentionEncoder,
                'dilated': DilatedEncoder
            }

            if encoder_type not in encoders:
                raise ValueError(f"不支持的编码器类型: {encoder_type}")

            return encoders[encoder_type](**kwargs)