import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.shape[0]

        # 线性投影
        q = self.q_proj(q).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权特征
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim)
        out = self.out_proj(out)

        return out, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        out, attn = self.mha(x_norm, x_norm, x_norm)
        return x + out, attn


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        context_norm = self.norm2(context)
        out, attn = self.mha(x_norm, context_norm, context_norm)
        return x + out, attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        # 平均池化
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        # 最大池化
        max_out = self.mlp(self.max_pool(x).view(b, c))

        # 注意力加权
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention))

        return x * attention


class PyramidAttention(nn.Module):
    def __init__(self, channels: int, scales: list = [1, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.attention_blocks = nn.ModuleList([
            SelfAttention(channels) for _ in scales
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attentions = []

        for scale, attn in zip(self.scales, self.attention_blocks):
            # 缩放输入
            if scale != 1:
                h = int(x.shape[2] * scale)
                w = int(x.shape[3] * scale)
                scaled = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            else:
                scaled = x

            # 重塑为序列
            b, c, h, w = scaled.shape
            scaled = scaled.reshape(b, c, -1).transpose(-2, -1)

            # 应用注意力
            attended, _ = attn(scaled)

            # 恢复空间维度
            attended = attended.transpose(-2, -1).reshape(b, c, h, w)

            # 上采样回原始尺寸
            if scale != 1:
                attended = F.interpolate(
                    attended,
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=True
                )

            attentions.append(attended)

        # 特征融合
        out = torch.cat(attentions, dim=1)
        out = self.fusion(out)

        return out


class DeformableAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.channels = channels

        # 偏移预测
        self.offset_conv = nn.Conv2d(channels, num_heads * num_points * 2, 1)

        # 注意力权重
        self.attention_conv = nn.Conv2d(channels, num_heads * num_points, 1)

        # 特征变换
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.output_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # 预测采样点偏移
        offsets = self.offset_conv(x)
        offsets = offsets.reshape(
            batch_size, self.num_heads, self.num_points, 2,
            x.shape[2], x.shape[3]
        )

        # 注意力权重
        attention = self.attention_conv(x)
        attention = attention.reshape(
            batch_size, self.num_heads, self.num_points,
            x.shape[2], x.shape[3]
        )
        attention = torch.sigmoid(attention)

        # 变换特征
        value = self.value_conv(x)

        # 双线性插值采样
        out = self._deformable_sampling(value, offsets, attention)
        out = self.output_conv(out)

        return out + x

    def _deformable_sampling(self, x: torch.Tensor,
                             offsets: torch.Tensor,
                             attention: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        height, width = x.shape[2:]

        # 生成参考点
        ref_y, ref_x = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref = ref.reshape(1, 1, 1, 2, height, width)

        # 采样坐标
        sampling_locations = ref + offsets
        sampling_locations = sampling_locations.reshape(
            batch_size, self.num_heads * self.num_points, 2, height, width
        )

        # 归一化坐标
        sampling_locations[..., 0, :, :] /= (width - 1)
        sampling_locations[..., 1, :, :] /= (height - 1)
        sampling_locations = sampling_locations * 2 - 1

        # 重塑特征和注意力权重
        x = x.reshape(batch_size, channels, height * width)
        sampling_locations = sampling_locations.reshape(
            batch_size, self.num_heads * self.num_points, 2, height * width
        )
        attention = attention.reshape(
            batch_size, self.num_heads * self.num_points, height * width
        )

        # 双线性插值采样
        sampled_features = F.grid_sample(
            x.reshape(batch_size, channels, height, width),
            sampling_locations.permute(0, 2, 1, 3).reshape(
                batch_size, 2, height, width
            ),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # 加权求和
        output = (sampled_features * attention.unsqueeze(1)).sum(dim=2)
        return output.reshape(batch_size, channels, height, width)