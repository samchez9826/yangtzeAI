from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DecoderConfig:
    """配置数据类"""
    in_channels: List[int]
    out_channels: int
    skip_channels: Optional[List[int]] = None
    dropout: float = 0.1
    use_bias: bool = False
    attention_heads: int = 8
    dim_feedforward: int = 2048
    max_len: int = 10000
    pos_encoding_scale: float = 1.0


class DecoderBase(nn.Module, ABC):
    """解码器基类"""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.initialize_layers()
        self._init_weights()

    @abstractmethod
    def initialize_layers(self) -> None:
        raise NotImplementedError

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @abstractmethod
    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]],
                skip_features: Optional[List[torch.Tensor]] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError


class FPNDecoder(DecoderBase):
    """特征金字塔解码器"""

    def initialize_layers(self) -> None:
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, self.config.out_channels, 1,
                          bias=self.config.use_bias),
                nn.BatchNorm2d(self.config.out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in self.config.in_channels
        ])

        # 特征融合
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.config.out_channels, self.config.out_channels, 3,
                          padding=1, bias=self.config.use_bias),
                nn.BatchNorm2d(self.config.out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.config.dropout)
            ) for _ in range(len(self.config.in_channels))
        ])

    @torch.cuda.amp.autocast()
    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        # 横向连接
        laterals = [
            lateral_conv(feat) for feat, lateral_conv in zip(features, self.lateral_convs)
        ]

        # 自顶向下特征融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )

        # 特征精炼
        return [
            fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)
        ]


class UNetDecoder(DecoderBase):
    """UNet解码器"""

    def initialize_layers(self) -> None:
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(self.config.in_channels)):
            skip_channel = self.config.skip_channels[i] if self.config.skip_channels else 0
            in_channel = self.config.in_channels[i] + skip_channel

            block = nn.Sequential(
                nn.Conv2d(in_channel, self.config.out_channels, 3, padding=1, bias=self.config.use_bias),
                nn.BatchNorm2d(self.config.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.config.out_channels, self.config.out_channels, 3, padding=1, bias=self.config.use_bias),
                nn.BatchNorm2d(self.config.out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.config.dropout)
            )
            self.decoder_blocks.append(block)

    @torch.cuda.amp.autocast()
    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        x = features[-1]

        for i, block in enumerate(self.decoder_blocks):
            # 上采样
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            # 跳跃连接
            if skip_features is not None:
                x = torch.cat([x, skip_features[-(i + 1)]], dim=1)

            x = block(x)

        return x


class ASPPDecoder(DecoderBase):
    """ASPP解码器"""

    def initialize_layers(self) -> None:
        in_channels = self.config.in_channels[0]
        out_channels = self.config.out_channels

        # ASPP分支
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=self.config.use_bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.config.dropout)
            ) for d in [6, 12, 18]
        ])

        # 1x1 卷积
        self.aspp_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=self.config.use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.config.dropout)
        ))

        # 全局上下文
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=self.config.use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=self.config.use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.config.dropout)
        )

    @torch.cuda.amp.autocast()
    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        x = features[-1]
        size = x.shape[-2:]

        # 并行处理ASPP分支
        aspp_outs = [block(x) for block in self.aspp_blocks]

        # 全局上下文
        global_context = self.global_pool(x)
        global_context = F.interpolate(global_context, size=size, mode='bilinear', align_corners=False)

        # 特征融合
        aspp_outs.append(global_context)
        x = self.fusion(torch.cat(aspp_outs, dim=1))

        return x


class TransformerDecoder(DecoderBase):
    """Transformer解码器"""

    def initialize_layers(self) -> None:
        d_model = self.config.out_channels

        # 特征投影
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_channels, d_model, 1, bias=self.config.use_bias)
            for in_channels in self.config.in_channels
        ])

        # Transformer解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=self.config.attention_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
                batch_first=True
            ) for _ in range(len(self.config.in_channels))
        ])

        # 位置编码
        self.pos_embedding = PositionalEncoding(
            d_model=d_model,
            max_len=self.config.max_len,
            dropout=self.config.dropout,
            scale=self.config.pos_encoding_scale
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model * 2, d_model)
        )

    @torch.cuda.amp.autocast()
    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        B = features[0].shape[0]
        H, W = features[-1].shape[-2:]

        # 特征投影和位置编码
        memory = []
        for feat, proj in zip(features, self.input_proj):
            m = proj(feat).flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            m = self.pos_embedding(m)
            memory.append(m)

        # 级联解码
        tgt = torch.zeros_like(memory[0])
        for layer, mem in zip(self.layers, memory):
            tgt = layer(tgt, mem)

        # 输出投影
        out = self.output_proj(tgt)

        # 重塑输出
        return out.permute(0, 2, 1).reshape(B, -1, H, W)


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first

        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self._get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor,
                memory: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                query_padding_mask: Optional[torch.Tensor] = None,
                memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        query2 = self.norm1(query)
        query = query + self.dropout(
            self.self_attn(
                query2, query2, query2,
                attn_mask=query_mask,
                key_padding_mask=query_padding_mask
            )[0]
        )

        # 交叉注意力
        query2 = self.norm2(query)
        query = query + self.dropout(
            self.cross_attn(
                query2, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_padding_mask
            )[0]
        )

        # 前馈网络
        query2 = self.norm3(query)
        query = query + self.dropout(self.feed_forward(query2))

        return query

    @staticmethod
    def _get_activation_fn(activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 10000,
                 dropout: float = 0.1, scale: float = 1.0):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        pe = self._get_positional_encoding(d_model, max_len)
        self.register_buffer('pe', pe)

    def _get_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x * math.sqrt(self.scale)
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    def create_decoder(decoder_type: str, config: DecoderConfig) -> DecoderBase:
        """
        创建解码器
        Args:
            decoder_type: 解码器类型 ['fpn', 'unet', 'aspp', 'transformer']
            config: 解码器配置
        Returns:
            解码器实例
        """
        decoders: Dict[str, Type[DecoderBase]] = {
            'fpn': FPNDecoder,
            'unet': UNetDecoder,
            'aspp': ASPPDecoder,
            'transformer': TransformerDecoder
        }

        if decoder_type not in decoders:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

        return decoders[decoder_type](config)