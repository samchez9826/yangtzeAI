import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F


class DecoderBase(nn.Module):
    """解码器基类"""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 skip_channels: Optional[List[int]] = None):
        """
        初始化解码器
        Args:
            in_channels: 输入通道数列表
            out_channels: 输出通道数
            skip_channels: 跳跃连接通道数列表
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

        self.initialize_layers()

    def initialize_layers(self):
        """初始化层"""
        raise NotImplementedError

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]],
                skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError


class FPNDecoder(DecoderBase):
    """特征金字塔解码器"""

    def initialize_layers(self):
        """初始化层"""
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            for in_channels in self.in_channels
        ])

        # 特征融合
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(self.in_channels))
        ])

    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """
        前向传播
        Args:
            features: 编码器特征列表
            skip_features: 跳跃连接特征列表
        Returns:
            解码器特征列表
        """
        # 横向连接
        laterals = [
            conv(feat) for feat, conv in zip(features, self.lateral_convs)
        ]

        # 自顶向下的路径
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            # 特征融合
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 最终卷积
        outs = [
            conv(lateral) for lateral, conv in zip(laterals, self.fpn_convs)
        ]

        return outs


class UNetDecoder(DecoderBase):
    """UNet解码器"""

    def initialize_layers(self):
        """初始化层"""
        self.decoder_blocks = nn.ModuleList()

        # 创建解码块
        for i in range(len(self.in_channels)):
            skip_channel = self.skip_channels[i] if self.skip_channels else 0
            in_channel = self.in_channels[i] + skip_channel

            block = nn.Sequential(
                nn.Conv2d(in_channel, self.out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(block)

    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            features: 编码器特征列表
            skip_features: 跳跃连接特征列表
        Returns:
            解码器特征
        """
        x = features[-1]

        for i, block in enumerate(self.decoder_blocks):
            # 上采样
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=True)

            # 跳跃连接
            if skip_features is not None:
                skip = skip_features[-(i + 1)]
                x = torch.cat([x, skip], dim=1)

            # 解码块
            x = block(x)

        return x


class ASPPDecoder(DecoderBase):
    """带ASPP模块的解码器"""

    def initialize_layers(self):
        """初始化层"""
        dilations = [6, 12, 18]

        # ASPP分支
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels[0], self.out_channels,
                          kernel_size=3, dilation=d, padding=d),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])

        # 1x1卷积分支
        self.aspp_blocks.append(nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        ))

        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels[0], self.out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(self.out_channels * (len(self.aspp_blocks) + 1),
                      self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: List[torch.Tensor],
                skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            features: 编码器特征列表
            skip_features: 跳跃连接特征列表
        Returns:
            解码器特征
        """
        x = features[-1]

        # ASPP分支
        aspp_outs = []
        for block in self.aspp_blocks:
            aspp_outs.append(block(x))

        # 全局上下文
        global_context = self.global_branch(x)
        global_context = F.interpolate(
            global_context,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        aspp_outs.append(global_context)

        # 特征融合
        x = torch.cat(aspp_outs, dim=1)
        x = self.fusion(x)

        # 最终处理
        x = self.final_conv(x)

        return x

    class TransformerDecoder(DecoderBase):
        """Transformer解码器"""

        def initialize_layers(self):
            """初始化层"""
            self.num_layers = len(self.in_channels)

            # 位置编码
            self.pos_encoder = PositionalEncoding(self.out_channels)

            # 解码器层
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(
                    self.out_channels,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1
                ) for _ in range(self.num_layers)
            ])

            # 特征投影
            self.proj = nn.ModuleList([
                nn.Conv2d(in_channel, self.out_channels, kernel_size=1)
                for in_channel in self.in_channels
            ])

        def forward(self, features: List[torch.Tensor],
                    skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
            """
            前向传播
            Args:
                features: 编码器特征列表
                skip_features: 跳跃连接特征列表
            Returns:
                解码器特征
            """
            B = features[0].shape[0]

            # 投影特征
            proj_features = [
                proj(feat) for proj, feat in zip(self.proj, features)
            ]

            # 调整形状并添加位置编码
            memory = []
            for feat in proj_features:
                # [B, C, H, W] -> [B, H*W, C]
                feat = feat.flatten(2).permute(0, 2, 1)
                feat = self.pos_encoder(feat)
                memory.append(feat)

            # 初始化查询向量
            query = torch.zeros(
                B,
                memory[0].shape[1],
                self.out_channels,
                device=features[0].device
            )

            # 依次通过解码器层
            for layer, mem in zip(self.layers, memory):
                query = layer(query, mem)

            # 恢复空间维度
            H = int(math.sqrt(query.shape[1]))
            x = query.permute(0, 2, 1).reshape(B, self.out_channels, H, H)

            return x

    class PositionalEncoding(nn.Module):
        """位置编码"""

        def __init__(self, d_model: int, max_len: int = 10000):
            """
            初始化位置编码
            Args:
                d_model: 特征维度
                max_len: 最大序列长度
            """
            super().__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() *
                (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            添加位置编码
            Args:
                x: 输入特征 [B, L, D]
            Returns:
                添加位置编码后的特征
            """
            return x + self.pe[:, :x.size(1)]

    class TransformerDecoderLayer(nn.Module):
        """Transformer解码器层"""

        def __init__(self, d_model: int, nhead: int,
                     dim_feedforward: int = 2048, dropout: float = 0.1):
            """
            初始化解码器层
            Args:
                d_model: 特征维度
                nhead: 注意力头数
                dim_feedforward: 前馈网络维度
                dropout: Dropout率
            """
            super().__init__()

            # 自注意力
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            # 交叉注意力
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            # 前馈网络
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )

            # 层归一化
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

            self.dropout = nn.Dropout(dropout)

        def forward(self, query: torch.Tensor,
                    memory: torch.Tensor) -> torch.Tensor:
            """
            前向传播
            Args:
                query: 查询特征 [B, L, D]
                memory: 记忆特征 [B, L, D]
            Returns:
                解码器特征
            """
            # 自注意力
            query2 = self.norm1(query)
            query = query + self.dropout(
                self.self_attn(query2, query2, query2)[0]
            )

            # 交叉注意力
            query2 = self.norm2(query)
            query = query + self.dropout(
                self.cross_attn(query2, memory, memory)[0]
            )

            # 前馈网络
            query2 = self.norm3(query)
            query = query + self.dropout(self.feed_forward(query2))

            return query

    class DecoderFactory:
        """解码器工厂类"""

        @staticmethod
        def create(decoder_type: str, **kwargs) -> DecoderBase:
            """
            创建解码器实例
            Args:
                decoder_type: 解码器类型
                **kwargs: 其他参数
            Returns:
                解码器实例
            """
            decoders = {
                'fpn': FPNDecoder,
                'unet': UNetDecoder,
                'aspp': ASPPDecoder,
                'transformer': TransformerDecoder
            }

            if decoder_type not in decoders:
                raise ValueError(f"不支持的解码器类型: {decoder_type}")

            return decoders[decoder_type](**kwargs)