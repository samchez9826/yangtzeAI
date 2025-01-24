import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base_model import BaseForgeryDetector
from einops import rearrange, repeat
import math


class ViTDetector(BaseForgeryDetector):
    """基于Vision Transformer的篡改检测器"""

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # 构建Transformer编码器
        self.transformer = self._build_transformer()

        # 构建多尺度特征提取器
        self.multi_scale = self._build_multi_scale()

        # 构建特征解码器
        self.decoder = self._build_decoder()

    def _build_transformer(self) -> nn.Module:
        """构建Transformer编码器"""
        config = self.config['transformer']
        return TransformerEncoder(
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            dropout=config.get('dropout', 0.1)
        )

    def _build_multi_scale(self) -> nn.Module:
        """构建多尺度特征提取器"""
        config = self.config['multi_scale']
        return MultiScaleViT(
            dim=config['dim'],
            patch_sizes=config['patch_sizes'],
            channels=config['channels']
        )

    def _build_decoder(self) -> nn.Module:
        """构建特征解码器"""
        config = self.config['decoder']
        return TransformerDecoder(
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim']
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        B, C, H, W = x.shape

        # 多尺度特征提取
        multi_scale_tokens = self.multi_scale(x)

        # Transformer编码
        encoded_features = self.transformer(multi_scale_tokens)

        # 特征解码
        decoded_features = self.decoder(encoded_features)

        outputs = {}

        # 主任务预测
        if 'cls' in self.head:
            outputs['cls'] = self.head['cls'](decoded_features)

        if 'seg' in self.head:
            # 重构空间维度
            features_2d = rearrange(
                decoded_features,
                'b (h w) d -> b d h w',
                h=H // 16, w=W // 16
            )
            outputs['seg'] = self.head['seg'](features_2d)

        # 辅助任务预测
        for name, head in self.auxiliary_heads.items():
            if name == 'attention':
                # 注意力图可视化
                attn_map = self.transformer.get_attention_maps()[-1]
                outputs[name] = rearrange(
                    attn_map,
                    'b h (p1 p2) n -> b (h p1) p2 n',
                    p1=int(math.sqrt(attn_map.shape[2]))
                )
            else:
                outputs[name] = head(decoded_features)

        return outputs


class MultiScaleViT(nn.Module):
    """多尺度Vision Transformer"""

    def __init__(self, dim: int, patch_sizes: List[int], channels: List[int]):
        super().__init__()

        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                patch_size=p,
                in_channels=c,
                embed_dim=dim
            ) for p, c in zip(patch_sizes, channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度patch embedding
        tokens = []
        for embedding in self.patch_embeddings:
            tokens.append(embedding(x))

        # 合并不同尺度的token
        return torch.cat(tokens, dim=1)

    class PatchEmbedding(nn.Module):
        """Patch Embedding层"""

        def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
            super().__init__()

            self.patch_size = patch_size
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )

            # Position Embedding
            self.pos_embed = nn.Parameter(
                torch.randn(1, (224 // patch_size) ** 2, embed_dim)
            )

            # CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape

            # Patch projection
            x = self.proj(x)  # (B, E, H/P, W/P)
            x = rearrange(x, 'b e h w -> b (h w) e')  # (B, N, E)

            # Add CLS token
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([cls_tokens, x], dim=1)

            # Add position embedding
            x = x + self.pos_embed[:, :(x.shape[1])]

            return x

    class TransformerEncoder(nn.Module):
        """Transformer编码器"""

        def __init__(self, dim: int, depth: int, heads: int,
                     mlp_dim: int, dropout: float = 0.1):
            super().__init__()

            self.layers = nn.ModuleList([])
            self.attention_maps = []

            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.attention_maps = []  # 清除之前的注意力图

            for attn, ff in self.layers:
                attn_output, attn_map = attn(x)
                x = attn_output + x
                x = ff(x) + x
                self.attention_maps.append(attn_map)

            return x

        def get_attention_maps(self) -> List[torch.Tensor]:
            """获取所有层的注意力图"""
            return self.attention_maps

    class TransformerDecoder(nn.Module):
        """Transformer解码器"""

        def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int):
            super().__init__()

            self.layers = nn.ModuleList([])

            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, CrossAttention(dim, heads=heads)),
                    PreNorm(dim, FeedForward(dim, mlp_dim))
                ]))

        def forward(self, x: torch.Tensor,
                    memory: Optional[torch.Tensor] = None) -> torch.Tensor:
            for cross_attn, ff in self.layers:
                if memory is not None:
                    x = cross_attn(x, memory) + x
                x = ff(x) + x
            return x

    class PreNorm(nn.Module):
        """Layer Normalization"""

        def __init__(self, dim: int, fn: nn.Module):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fn = fn

        def forward(self, x: torch.Tensor,
                    **kwargs) -> torch.Tensor:
            return self.fn(self.norm(x), **kwargs)

    class FeedForward(nn.Module):
        """前馈网络"""

        def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class Attention(nn.Module):
        """多头自注意力"""

        def __init__(self, dim: int, heads: int = 8, dropout: float = 0.):
            super().__init__()
            self.heads = heads
            self.scale = dim ** -0.5

            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h=self.heads
            ), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)

            return out, attn

    class CrossAttention(nn.Module):
        """交叉注意力"""

        def __init__(self, dim: int, heads: int = 8, dropout: float = 0.):
            super().__init__()
            self.heads = heads
            self.scale = dim ** -0.5

            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim, dim * 2, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor,
                    context: torch.Tensor) -> torch.Tensor:
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)

            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)

            return out

    class SpatialTransformer(nn.Module):
        """空间Transformer"""

        def __init__(self, dim: int, depth: int, heads: int,
                     window_size: int, mlp_dim: int):
            super().__init__()

            self.window_size = window_size
            self.pos_embed = nn.Parameter(
                torch.randn(1, window_size ** 2, dim)
            )

            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, WindowAttention(
                        dim, heads=heads,
                        window_size=window_size
                    )),
                    PreNorm(dim, FeedForward(dim, mlp_dim))
                ]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))

            # 分割成窗口
            x = rearrange(
                x,
                'b (h w) c -> b (h w) c',
                h=H // self.window_size
            )

            # 添加位置编码
            x = x + self.pos_embed

            # 应用注意力
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x

            # 恢复原始形状
            x = rearrange(
                x,
                'b (h w) c -> b (h w) c',
                h=H
            )

            return x

    class WindowAttention(nn.Module):
        """窗口注意力"""

        def __init__(self, dim: int, heads: int,
                     window_size: int, dropout: float = 0.):
            super().__init__()
            self.heads = heads
            self.scale = dim ** -0.5
            self.window_size = window_size

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size - 1) * (2 * window_size - 1),
                    heads
                )
            )

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid(
                [coords_h, coords_w]
            ))
            coords_flatten = torch.flatten(coords, 1)

            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)

            self.register_buffer("relative_position_index", relative_position_index)

            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h=self.heads
            ), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size ** 2,
                self.window_size ** 2,
                -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots = dots + relative_position_bias.unsqueeze(0)

            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')

            return self.to_out(out)