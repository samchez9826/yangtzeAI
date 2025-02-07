from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Type, Any
from dataclasses import dataclass
from einops import rearrange, repeat
import math
import logging
from torch.cuda import amp
from abc import ABC, abstractmethod

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ViTConfig:
    """Vision Transformer配置"""
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    patch_size: int
    channels: int = 3
    dropout: float = 0.1
    emb_dropout: float = 0.1
    window_size: Optional[int] = None
    multi_scale: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ViTConfig':
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


class ViTDetector(nn.Module):
    """优化的Vision Transformer篡改检测器"""

    def __init__(self, config_path: str):
        super().__init__()
        self.config = ViTConfig.from_dict(self.config)

        # 构建主要组件
        self.patch_embed = PatchEmbedding(self.config)
        self.transformer = TransformerEncoder(self.config)
        self.decoder = TransformerDecoder(self.config)
        self.multi_scale = MultiScaleViT(self.config) if self.config.multi_scale else None

        # 初始化权重
        self._init_weights()

        # 性能优化
        self._setup_checkpointing()

        # 统计参数
        self._count_parameters()

    def _init_weights(self) -> None:
        """初始化权重"""
        def _init_layer(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init_layer)

    def _setup_checkpointing(self) -> None:
        """设置梯度检查点"""
        if hasattr(torch, 'utils') and hasattr(torch.utils, 'checkpoint'):
            self.use_checkpointing = True
            for block in self.transformer.layers:
                block.grad_checkpointing = True

    def _count_parameters(self) -> None:
        """统计模型参数"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        outputs = {}
        try:
            # 主干特征提取
            features = self._extract_features(x)

            # 解码与预测
            outputs.update(self._decode_and_predict(features))

            # 辅助任务
            if hasattr(self, 'auxiliary_heads'):
                outputs.update(self._compute_auxiliary_tasks(features))

            return outputs

        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        # Patch embedding
        x = self.patch_embed(x)

        # 多尺度处理
        if self.multi_scale is not None:
            x = self.multi_scale(x)

        # Transformer编码
        return self.transformer(x)

    def _decode_and_predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """解码与预测"""
        outputs = {}

        # 解码
        decoded = self.decoder(features)

        # 分类预测
        if hasattr(self, 'cls_head'):
            outputs['cls'] = self.cls_head(decoded[:, 0])

        # 分割预测
        if hasattr(self, 'seg_head'):
            B, N, C = decoded.shape
            H = W = int(math.sqrt(N - 1))  # 减去CLS token
            spatial_features = rearrange(decoded[:, 1:], 'b (h w) c -> b c h w', h=H)
            outputs['seg'] = self.seg_head(spatial_features)

        return outputs

    def _compute_auxiliary_tasks(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算辅助任务"""
        return {name: head(features) for name, head in self.auxiliary_heads.items()}


class PatchEmbedding(nn.Module):
    """优化的Patch Embedding"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        image_height, image_width = 224, 224  # 标准输入尺寸
        patch_height, patch_width = config.patch_size, config.patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.channels * patch_height * patch_width

        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(config.channels, config.dim,
                              kernel_size=config.patch_size,
                              stride=config.patch_size)

        # Position embedding 与 CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        self.dropout = nn.Dropout(config.emb_dropout)

        # 动态位置编码缓存
        self.register_buffer('interpolated_pos_embed', None, persistent=False)

        # 初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if isinstance(self.proj, nn.Conv2d):
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """动态调整位置编码尺寸"""
        npatch = height * width
        N = pos_embed.shape[1] - 1

        if npatch == N:
            return pos_embed

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        # 计算原始图像尺寸
        h = w = int(math.sqrt(N))
        patch_pos_embed = patch_pos_embed.reshape(1, h, w, -1).permute(0, 3, 1, 2)

        # 双线性插值调整尺寸
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, height * width, -1)
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 如果输入尺寸与预期不同,动态调整位置编码
        if H != 224 or W != 224:
            pos_embed = self._interpolate_pos_embed(self.pos_embed, H // self.patch_size, W // self.patch_size)
        else:
            pos_embed = self.pos_embed

        # Patch projection
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding and dropout
        x = x + pos_embed
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """优化的Transformer编码器"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(config.dim)
        self.grad_checkpointing = False

        for _ in range(config.depth):
            self.layers.append(nn.ModuleList([
                PreNorm(config.dim,
                        WindowAttention(config.dim, config.heads, config.window_size)
                        if config.window_size
                        else Attention(config.dim, config.heads)),
                PreNorm(config.dim,
                        FeedForward(config.dim, config.mlp_dim))
            ]))

        # 性能优化
        self._optimize_memory_efficiency()

    def _optimize_memory_efficiency(self) -> None:
        """优化内存使用效率"""
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # 使用 Flash Attention 如果可用
            self.use_flash_attention = True
        else:
            self.use_flash_attention = False

        # 设置梯度检查点
        if torch.cuda.is_available():
            self.grad_checkpointing = True

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        attention_maps = [] if return_attention else None

        for attn, ff in self.layers:
            # 使用梯度检查点以节省显存
            if self.grad_checkpointing and self.training:
                if return_attention:
                    x_attn, attn_map = torch.utils.checkpoint.checkpoint(attn, x, return_attention)
                    attention_maps.append(attn_map)
                else:
                    x = torch.utils.checkpoint.checkpoint(attn, x)
                x = torch.utils.checkpoint.checkpoint(ff, x)
            else:
                if return_attention:
                    x_attn, attn_map = attn(x, return_attention=True)
                    attention_maps.append(attn_map)
                    x = x_attn
                else:
                    x = attn(x)
                x = ff(x)

        x = self.norm(x)
        return (x, attention_maps) if return_attention else x

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """设置是否启用梯度检查点"""
        self.grad_checkpointing = enable


# 辅助的配置和性能监控装饰器
def record_function(name: str):
    """用于性能分析的装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    with torch.profiler.record_function(name):
                        return func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class Attention(nn.Module):
    """优化的自注意力机制"""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.dim = dim
        head_dim = dim // heads

        # QKV投影
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        # Flash Attention 支持
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # 初始化
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 使用 Flash Attention 如果可用
        if self.use_flash_attention and not return_attention:
            out = F.scaled_dot_product_attention(q, k, v)
            attn = None
        else:
            # 传统注意力计算
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return (out, attn) if return_attention else out


class WindowAttention(nn.Module):
    """优化的窗口注意力"""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = dim ** -0.5
        head_dim = dim // num_heads

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # 相对位置编码
        self.relative_position_bias = self._get_relative_position_bias()
        self.register_buffer("relative_position_index", self._get_relative_position_index())

        # Flash Attention 支持
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def _get_relative_position_bias(self) -> nn.Parameter:
        """获取相对位置编码"""
        num_relative_distance = (2 * self.window_size - 1) ** 2
        relative_position_bias = nn.Parameter(
            torch.zeros(num_relative_distance, self.num_heads))
        nn.init.trunc_normal_(relative_position_bias, std=0.02)
        return relative_position_bias

    def _get_relative_position_index(self) -> torch.Tensor:
        """计算相对位置索引"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1

        return relative_coords.sum(-1)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 使用 Flash Attention 如果可用
        if self.use_flash_attention and not return_attention:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            attn = None
        else:
            # 传统注意力计算
            dots = (q @ k.transpose(-2, -1)) * self.scale

            # 添加相对位置偏置
            relative_position_bias = self.relative_position_bias[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size ** 2,
                self.window_size ** 2,
                -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots = dots + relative_position_bias.unsqueeze(0)

            attn = dots.softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return (out, attn) if return_attention else out


class CrossAttention(nn.Module):
    """优化的交叉注意力"""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        head_dim = dim // heads

        # Query 和 Key-Value 投影
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        # Flash Attention 支持
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # 初始化
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.to_q, self.to_kv]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # 使用 Flash Attention 如果可用
        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # 传统注意力计算
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    """Layer Normalization封装"""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """优化的前馈网络"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleViT(nn.Module):
    """优化的多尺度Vision Transformer"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # 多尺度编码器
        self.levels = nn.ModuleList([
            TransformerEncoder(config)
            for _ in range(3)  # 3个尺度层级
        ])

        # 下采样层
        self.pool = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Conv2d(config.dim, config.dim, kernel_size=3,
                          stride=2, padding=1),
                nn.GELU()
            ) for _ in range(2)  # 2个下采样层
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(config.dim * 3, config.dim),
            nn.LayerNorm(config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        B = x.shape[0]

        # 多尺度特征提取
        for i, (level, pool) in enumerate(zip(self.levels[:-1], self.pool)):
            # 特征提取
            x = level(x)
            features.append(x)

            # 下采样
            if i < len(self.pool):
                cls_token, spatial_tokens = x[:, 0:1], x[:, 1:]
                H = W = int(math.sqrt(spatial_tokens.shape[1]))

                # 处理空间tokens
                spatial_tokens = rearrange(spatial_tokens, 'b (h w) c -> b c h w', h=H)
                spatial_tokens = pool(spatial_tokens)
                spatial_tokens = rearrange(spatial_tokens, 'b c h w -> b (h w) c')

                # 重组tokens
                x = torch.cat([cls_token, spatial_tokens], dim=1)

        # 最后一层特征
        x = self.levels[-1](x)
        features.append(x)

        # 特征融合
        return self._fuse_features(features)

    def _fuse_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """特征融合"""
        # 调整特征尺寸
        resized_features = []
        target_size = features[0].shape[1]

        for feat in features:
            if feat.shape[1] != target_size:
                feat = self._resize_feature(feat, target_size)
            resized_features.append(feat)

        # 注意力加权融合
        concat_features = torch.cat(resized_features, dim=-1)
        return self.fusion(concat_features)

    @staticmethod
    def _resize_feature(x: torch.Tensor, target_size: int) -> torch.Tensor:
        """特征尺寸调整"""
        cls_token, spatial_tokens = x[:, 0:1], x[:, 1:]

        # 调整空间特征尺寸
        H = W = int(math.sqrt(spatial_tokens.shape[1]))
        spatial_tokens = rearrange(spatial_tokens, 'b (h w) c -> b c h w', h=H)

        target_hw = int(math.sqrt(target_size - 1))
        spatial_tokens = F.interpolate(
            spatial_tokens,
            size=(target_hw, target_hw),
            mode='bilinear',
            align_corners=False
        )

        spatial_tokens = rearrange(spatial_tokens, 'b c h w -> b (h w) c')
        return torch.cat([cls_token, spatial_tokens], dim=1)


class TransformerDecoder(nn.Module):
    """优化的Transformer解码器"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(config.dim)

        for _ in range(config.depth):
            self.layers.append(nn.ModuleList([
                PreNorm(config.dim, CrossAttention(config.dim, config.heads)),
                PreNorm(config.dim, FeedForward(config.dim, config.mlp_dim))
            ]))

        # 特征重建
        self.reconstruction = nn.Sequential(
            nn.Linear(config.dim, config.channels * 16 * 16),
            nn.GELU(),
            Rearrange('b n (c h w) -> b c (n h) w', c=config.channels, h=4)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor,
                memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Transformer解码
        for cross_attn, ff in self.layers:
            if memory is not None:
                x = cross_attn(x, memory) + x
            x = ff(x) + x

        x = self.norm(x)

        # 特征重建
        if hasattr(self, 'reconstruction'):
            B, N, C = x.shape
            if N > 1:  # 跳过CLS token
                spatial_tokens = x[:, 1:]
                x = self.reconstruction(spatial_tokens)

        return x


def create_vit_detector(config_path: str,
                        pretrained: bool = True,
                        device: Optional[torch.device] = None) -> ViTDetector:
    """创建ViT检测器的工厂函数

    Args:
        config_path: 配置文件路径
        pretrained: 是否加载预训练权重
        device: 运行设备

    Returns:
        初始化好的ViT检测器实例
    """
    try:
        # 创建模型
        model = ViTDetector(config_path)

        # 加载预训练权重
        if pretrained:
            try:
                state_dict = torch.load(f'pretrained/vit_{model.config.dim}.pth',
                                        map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded pretrained weights")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {str(e)}")

        # 移动到指定设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 打印模型信息
        logger.info(f"Created ViT detector - {model.config}")
        logger.info(f"Device: {device}")

        return model

    except Exception as e:
        logger.error(f"Failed to create ViT detector: {str(e)}")
        raise


class Rearrange(nn.Module):
    """张量重排模块"""

    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, self.pattern)