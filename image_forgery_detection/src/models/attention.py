import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union, Dict
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """优化的多头注意力"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 合并QKV投影
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)

        if bias:
            nn.init.constant_(self.qkv_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                cache: Optional[Dict] = None) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape

        # QKV投影
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 高效的注意力计算
        attn_weights = torch.empty(batch_size, self.num_heads, seq_len, seq_len,
                                   dtype=q.dtype, device=q.device)
        attn_weights = torch.baddbmm(attn_weights, q, k.transpose(-2, -1),
                                     beta=0.0, alpha=self.scale)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.type_as(q)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return out, attn_weights


class SelfAttention(nn.Module):
    """优化的自注意力"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        shortcut = x
        x = self.norm(x)
        out, attn = self.mha(x, mask=mask)
        return shortcut + self.gate * out, attn


class CrossAttention(nn.Module):
    """优化的交叉注意力"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, context: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        shortcut = x
        x = self.norm1(x)
        context = self.norm2(context)
        out, attn = self.mha(x, mask=mask)
        return shortcut + self.gate * out, attn


class CBAM(nn.Module):
    """优化的CBAM注意力"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()

        # 通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.BatchNorm2d(channels),
        )

        # 空间注意力 - 使用深度可分离卷积
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, groups=1),
            nn.BatchNorm2d(1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # 通道注意力
        channel_attn = torch.sigmoid(self.channel_gate(x))
        x = x * channel_attn

        # 空间注意力
        spatial = torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_gate(spatial))

        return x * spatial_attn


class PyramidAttention(nn.Module):
    """优化的金字塔注意力"""

    def __init__(self, channels: int, scales: List[float] = [1, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            SelfAttention(channels) for _ in scales
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.size_cache = {}

    def _get_size(self, size: Tuple[int, int], scale: float) -> Tuple[int, int]:
        cache_key = (size, scale)
        if cache_key not in self.size_cache:
            h = int(size[0] * scale)
            w = int(size[1] * scale)
            self.size_cache[cache_key] = (h, w)
        return self.size_cache[cache_key]

    @torch.cuda.amp.autocast()
    def forward(self, x: Tensor) -> Tensor:
        results = []
        size = (x.shape[2], x.shape[3])

        for scale, attn in zip(self.scales, self.attentions):
            if scale != 1:
                scaled_size = self._get_size(size, scale)
                scaled = F.interpolate(x, size=scaled_size, mode='bilinear',
                                       align_corners=False)
            else:
                scaled = x

            b, c, h, w = scaled.shape
            feat = scaled.reshape(b, c, -1).transpose(-2, -1)
            feat, _ = attn(feat)
            feat = feat.transpose(-2, -1).reshape(b, c, h, w)

            if scale != 1:
                feat = F.interpolate(feat, size=size, mode='bilinear',
                                     align_corners=False)

            results.append(feat)

        out = torch.cat(results, dim=1)
        out = self.fusion(out)

        return out


class DeformableAttention(nn.Module):
    """优化的可变形注意力"""

    def __init__(self, channels: int, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.channels = channels

        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_heads * num_points * 2, 1)
        )

        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_heads * num_points, 1)
        )

        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Conv2d(channels, channels, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_offset_reference(self, spatial_shapes: Tuple[int, int],
                              device: torch.device) -> Tensor:
        """获取参考点坐标"""
        h, w = spatial_shapes
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, device=device),
            torch.linspace(0.5, w - 0.5, w, device=device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref[..., 0] = ref[..., 0] / max(w - 1, 1)
        ref[..., 1] = ref[..., 1] / max(h - 1, 1)
        ref = ref * 2 - 1
        return ref

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        b, c, h, w = x.shape

        # 计算采样偏移
        offset = self.offset_conv(x)
        offset = offset.view(b, self.num_heads, self.num_points, 2, h, w)

        # 计算注意力权重
        attention = self.attention_conv(x)
        attention = attention.view(b, self.num_heads, self.num_points, h, w)
        attention = torch.sigmoid(attention)

        # 特征变换
        value = self.value_proj(x)

        # 获取参考点
        reference = self._get_offset_reference((h, w), x.device)

        # 采样坐标
        if self.num_points > 1:
            reference = reference.repeat(1, 1, self.num_points, 1)
            offset = offset.reshape(b, self.num_heads * self.num_points, 2, h, w)

        sampling_locations = reference + offset
        sampling_locations = sampling_locations.reshape(b, -1, h, w, 2)

        # 双线性采样
        value = value.reshape(b, self.num_heads, -1, h, w)
        output = F.grid_sample(
            value, sampling_locations,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # 注意力加权
        attention = attention.reshape(b, self.num_heads, -1, 1, h, w)
        output = (output * attention).sum(dim=2)

        output = output.reshape(b, c, h, w)
        output = self.output_proj(output)

        return output + identity


def create_attention_mask(seq_len: int, device: torch.device,
                          is_causal: bool = False) -> Tensor:
    """创建注意力掩码"""
    mask = torch.ones(seq_len, seq_len, device=device)
    if is_causal:
        mask = torch.triu(mask, diagonal=1)
    return mask


def relative_position_bucket(relative_position: Tensor,
                             num_buckets: int = 32,
                             max_distance: int = 128) -> Tensor:
    """计算相对位置编码的bucket"""
    ret = 0
    n = -relative_position
    n = torch.max(n, torch.zeros_like(n))

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
    ).long()
    val_if_large = torch.min(val_if_large,
                             torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret