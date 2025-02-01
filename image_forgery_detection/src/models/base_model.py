from __future__ import annotations

import torch
import torch.nn as nn
import timm
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Type, TypeVar
from abc import ABC, abstractmethod
import yaml
from pathlib import Path
import time
from contextlib import contextmanager
from dataclasses import dataclass
from torch.cuda import amp
from .networks import FeaturePyramidNetwork, AtrousSpatialPyramidPooling

try:
    import thop

    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    import torchinfo

    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

T = TypeVar('T', bound='BaseForgeryDetector')


@dataclass
class ModelConfig:
    """模型配置数据类"""
    backbone: Dict[str, Any]
    neck: Optional[Dict[str, Any]]
    head: Dict[str, Any]
    auxiliary_tasks: Optional[Dict[str, Any]]
    loss_weights: Dict[str, float]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ModelConfig':
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['model']
        return cls(**config)


class BaseForgeryDetector(nn.Module, ABC):
    """优化的图像篡改检测模型基类"""

    def __init__(self, config_path: str):
        """
        初始化模型
        Args:
            config_path: 配置文件路径
        """
        super().__init__()

        # 加载配置
        self.config = ModelConfig.from_yaml(config_path)

        # 设置设备和混合精度训练
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = amp.GradScaler(enabled=torch.cuda.is_available())

        # 设置日志
        self._setup_logger()

        # 构建模型
        with self._init_context():
            self.backbone = self._build_backbone()
            self.neck = self._build_neck()
            self.head = self._build_head()
            self.auxiliary_heads = self._build_auxiliary_heads()

            # 初始化权重
            self._init_weights()

    @contextmanager
    def _init_context(self):
        """模型初始化上下文管理器"""
        try:
            yield
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    def _setup_logger(self):
        """设置日志系统"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _build_backbone(self) -> nn.Module:
        """构建主干网络"""
        try:
            backbone_config = self.config.backbone
            model_name = backbone_config['name']
            pretrained = backbone_config.get('pretrained', True)

            # 使用timm加载预训练模型
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                drop_path_rate=backbone_config.get('drop_path_rate', 0.2)
            )

            # 添加激活检查点以节省显存
            if hasattr(backbone, 'set_grad_checkpointing'):
                backbone.set_grad_checkpointing(True)

            # 冻结指定层
            if 'frozen_stages' in backbone_config:
                self._freeze_stages(backbone, backbone_config['frozen_stages'])

            return backbone

        except Exception as e:
            self.logger.error(f"Backbone creation failed: {e}")
            raise

    def _build_neck(self) -> nn.Module:
        """构建特征融合网络"""
        if not self.config.neck:
            return nn.Identity()

        try:
            neck_config = self.config.neck
            neck_type = neck_config['type'].lower()

            if neck_type == 'fpn':
                return FeaturePyramidNetwork(
                    in_channels=neck_config['in_channels'],
                    out_channels=neck_config['out_channels'],
                    use_p6p7=neck_config.get('use_p6p7', False)
                )
            elif neck_type == 'aspp':
                return AtrousSpatialPyramidPooling(
                    in_channels=neck_config['in_channels'],
                    out_channels=neck_config['out_channels'],
                    atrous_rates=neck_config.get('atrous_rates', [6, 12, 18])
                )
            else:
                raise ValueError(f"Unsupported neck type: {neck_type}")

        except Exception as e:
            self.logger.error(f"Neck creation failed: {e}")
            raise

    def _build_head(self) -> nn.ModuleDict:
        """构建检测头"""
        heads = {}
        try:
            head_config = self.config.head

            # 分类头
            if 'classification' in head_config:
                heads['cls'] = self._create_classification_head(
                    head_config['classification']
                )

            # 分割头
            if 'segmentation' in head_config:
                heads['seg'] = self._create_segmentation_head(
                    head_config['segmentation']
                )

            return nn.ModuleDict(heads)

        except Exception as e:
            self.logger.error(f"Head creation failed: {e}")
            raise

    def _build_auxiliary_heads(self) -> nn.ModuleDict:
        """构建辅助任务头"""
        aux_heads = {}

        if not self.config.auxiliary_tasks:
            return nn.ModuleDict(aux_heads)

        try:
            aux_config = self.config.auxiliary_tasks

            # 噪声估计
            if aux_config.get('noise_level', {}).get('enabled', False):
                aux_heads['noise'] = self._create_noise_head(
                    aux_config['noise_level']
                )

            # 边缘一致性
            if aux_config.get('edge_consistency', {}).get('enabled', False):
                aux_heads['edge'] = self._create_edge_head(
                    aux_config['edge_consistency']
                )

            return nn.ModuleDict(aux_heads)

        except Exception as e:
            self.logger.error(f"Auxiliary head creation failed: {e}")
            raise

    def _create_classification_head(self, config: Dict[str, Any]) -> nn.Module:
        """创建分类头"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config['in_channels'], config['hidden_dims'][0]),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(config['hidden_dims'][0], config['num_classes'])
        )

    def _create_segmentation_head(self, config: Dict[str, Any]) -> nn.Module:
        """创建分割头"""
        in_channels = config['in_channels']
        decoder_channels = config['decoder_channels']

        decoder = []
        for out_channels in decoder_channels:
            decoder.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ])
            in_channels = out_channels

        decoder.append(
            nn.Conv2d(in_channels, config['out_channels'], 1)
        )

        return nn.Sequential(*decoder)

    def _create_noise_head(self, config: Dict[str, Any]) -> nn.Module:
        """创建噪声估计头"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config['in_channels'], config['hidden_dims'][0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(config['hidden_dims'][0], 1),
            nn.Sigmoid()
        )

    def _create_edge_head(self, config: Dict[str, Any]) -> nn.Module:
        """创建边缘一致性头"""
        return nn.Sequential(
            nn.Conv2d(config['in_channels'], config['hidden_dims'][0], 3, padding=1),
            nn.BatchNorm2d(config['hidden_dims'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['hidden_dims'][0], 1, 1),
            nn.Sigmoid()
        )

    @torch.cuda.amp.autocast()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        neck_feat = self.neck(features)

        outputs = {}
        # 主任务输出
        if 'cls' in self.head:
            outputs['cls'] = self.head['cls'](neck_feat)
        if 'seg' in self.head:
            outputs['seg'] = self.head['seg'](neck_feat)

        # 辅助任务输出
        for name, head in self.auxiliary_heads.items():
            outputs[name] = head(neck_feat)

        return outputs

    @torch.cuda.amp.autocast()
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """训练步骤"""
        try:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 特征提取
            features = self.backbone(images)
            neck_feat = self.neck(features)

            outputs = {}
            losses = {}

            # 主任务
            if 'cls' in self.head:
                cls_out = self.head['cls'](neck_feat)
                outputs['cls'] = cls_out
                losses['cls'] = self.criterion['cls'](cls_out, labels)

            if 'seg' in self.head and 'mask' in batch:
                masks = batch['mask'].to(self.device)
                seg_out = self.head['seg'](neck_feat)
                outputs['seg'] = seg_out
                losses['seg'] = self.criterion['seg'](seg_out, masks)

            # 辅助任务
            for name, head in self.auxiliary_heads.items():
                aux_out = head(neck_feat)
                outputs[name] = aux_out
                if f'{name}_target' in batch:
                    target = batch[f'{name}_target'].to(self.device)
                    losses[name] = self.criterion[name](aux_out, target)

            # 总损失
            total_loss = sum(
                loss * self.config.loss_weights.get(name, 1.0)
                for name, loss in losses.items()
            )
            losses['total'] = total_loss

            return {'outputs': outputs, 'losses': losses}

        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            raise

    @torch.no_grad()
    def get_features(self, batch: Dict[str, torch.Tensor],
                     layers: List[str]) -> Dict[str, torch.Tensor]:
        """提取特征"""
        try:
            images = batch['image'].to(self.device)
            features = {}

            # 提取backbone特征
            if any(layer.startswith('backbone') for layer in layers):
                backbone_feat = self.backbone(images)
                features.update({
                    f'backbone_layer{i + 1}': feat
                    for i, feat in enumerate(backbone_feat)
                })

            # 提取neck特征
            if 'neck' in layers:
                if 'backbone_feat' not in locals():
                    backbone_feat = self.backbone(images)
                neck_feat = self.neck(backbone_feat)
                features['neck'] = neck_feat

            # 提取head特征
            if any(layer.startswith('head_') for layer in layers):
                if 'neck_feat' not in locals():
                    if 'backbone_feat' not in locals():
                        backbone_feat = self.backbone(images)
                    neck_feat = self.neck(backbone_feat)
                for layer in layers:
                    if layer.startswith('head_'):
                        head_name = layer.split('_')[1]
                        if head_name in self.head:
                            features[layer] = self.head[head_name](neck_feat)

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise

    def analyze_predictions(self, outputs: Dict[str, torch.Tensor],
                            threshold: float = 0.5) -> Dict[str, Any]:
        """分析预测结果"""
        try:
            results = {}

            # 分类预测分析
            if 'cls' in outputs:
                probs = outputs['cls'].cpu().numpy()
                preds = (probs[:, 1] > threshold).astype(int)
                results['classification'] = {
                    'predictions': preds,
                    'probabilities': probs,
                    'confidence': np.max(probs, axis=1)
                }

            # 分割预测分析
            if 'seg' in outputs:
                masks = outputs['seg'].cpu().numpy()
                binary_masks = (masks > threshold).astype(int)

                results['segmentation'] = {
                    'masks': binary_masks,
                    'area_ratio': np.mean(binary_masks),
                    'confidence_map': masks,
                    'regions': self._analyze_regions(binary_masks, masks)
                }

            # 辅助任务分析
            for name in ['noise', 'edge']:
                if name in outputs:
                    pred = outputs[name].cpu().numpy()
                    results[f'{name}_estimation'] = {
                        'predictions': pred,
                        'mean': float(np.mean(pred)),
                        'std': float(np.std(pred))
                    }

            return results

        except Exception as e:
            self.logger.error(f"Prediction analysis failed: {e}")
            raise

    def _analyze_regions(self, binary_masks: np.ndarray,
                         confidence_masks: np.ndarray) -> List[Dict[str, Any]]:
        """分析篡改区域"""
        from scipy import ndimage
        regions_info = []

        try:
            for i in range(len(binary_masks)):
                mask = binary_masks[i]
                conf_mask = confidence_masks[i]

                # 连通区域分析
                labeled, num_features = ndimage.label(mask)
                if num_features > 0:
                    for j in range(1, num_features + 1):
                        region = (labeled == j)
                        bbox = self._get_bbox(region)
                        area = np.sum(region)
                        confidence = float(np.mean(conf_mask[region]))

                        regions_info.append({
                            'image_idx': i,
                            'region_idx': j,
                            'area': int(area),
                            'bbox': tuple(int(x) for x in bbox),
                            'confidence': confidence,
                            'perimeter': self._calculate_perimeter(region)
                        })

            return regions_info

        except Exception as e:
            self.logger.error(f"Region analysis failed: {e}")
            return []

    def _freeze_stages(self, model: nn.Module, num_stages: int) -> None:
        """冻结指定层的参数"""
        if num_stages <= 0:
            return

        for name, param in model.named_parameters():
            if any(f'layer{i}' in name for i in range(num_stages)):
                param.requires_grad = False

    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """获取边界框"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (x1, y1, x2, y2)

    @staticmethod
    def _calculate_perimeter(mask: np.ndarray) -> int:
        """计算区域周长"""
        from scipy import ndimage
        structure = np.ones((3, 3))
        eroded = ndimage.binary_erosion(mask, structure=structure)
        perimeter = mask & ~eroded
        return int(np.sum(perimeter))

    def profile(self, input_shape: Tuple[int, ...],
                batch_size: int = 1) -> Dict[str, Any]:
        """分析模型性能"""
        if not HAS_THOP:
            self.logger.warning("thop not installed, skipping FLOPs calculation")
            return {}

        try:
            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
            flops, params = thop.profile(self, (dummy_input,))

            # 测试推理时间
            times = []
            with torch.cuda.amp.autocast():
                for _ in range(100):
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = self(dummy_input)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

            avg_time = np.mean(times[10:])  # 去掉预热时间
            fps = batch_size / avg_time

            return {
                'flops': flops,
                'params': params,
                'inference_time': avg_time,
                'fps': fps,
                'model_size_mb': sum(p.numel() * p.element_size()
                                     for p in self.parameters()) / 1024 / 1024
            }

        except Exception as e:
            self.logger.error(f"Profiling failed: {e}")
            return {}

    def load_checkpoint(self, ckpt_path: Union[str, Path]) -> Dict[str, Any]:
        """加载检查点"""
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(ckpt['model_state_dict'])
            return ckpt
        except Exception as e:
            self.logger.error(f"Checkpoint loading failed: {e}")
            raise

    def export(self, save_path: Union[str, Path], export_format: str = 'onnx',
               input_shape: Optional[Tuple[int, ...]] = None,
               dynamic_axes: bool = True) -> None:
        """导出模型"""
        save_path = Path(save_path)
        if input_shape is None:
            input_shape = (1, 3, 224, 224)

        try:
            dummy_input = torch.randn(input_shape).to(self.device)

            if export_format.lower() == 'onnx':
                dynamic_axes_dict = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                } if dynamic_axes else None

                torch.onnx.export(
                    self, dummy_input, save_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes_dict
                )

            elif export_format.lower() == 'torchscript':
                script_model = torch.jit.trace(self, dummy_input)
                script_model.save(str(save_path))

            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            self.logger.info(f"Model exported to {save_path}")

        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            raise

    def summary(self) -> None:
        """打印模型概要"""
        if not HAS_TORCHINFO:
            self.logger.warning("torchinfo not installed, skipping model summary")
            return

        try:
            input_size = getattr(self, 'input_shape', (1, 3, 224, 224))
            torchinfo.summary(self, input_size=input_size)

            # 打印组件参数统计
            print("\nComponent Parameters:")
            print("-" * 50)
            components = {
                'Backbone': self.backbone,
                'Neck': self.neck,
                'Heads': self.head,
                'Auxiliary Heads': self.auxiliary_heads
            }

            total_params = 0
            for name, module in components.items():
                params = sum(p.numel() for p in module.parameters())
                print(f"{name:15s}: {params:,} parameters")
                total_params += params

            print("-" * 50)
            print(f"Total Parameters: {total_params:,}")

            # 显存使用统计
            if torch.cuda.is_available():
                print("\nGPU Memory Usage:")
                print("-" * 50)
                print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
                print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

        except Exception as e:
            self.logger.error(f"Model summary failed: {e}")
            raise

    @classmethod
    def create(cls: Type[T], model_type: str, config_path: str) -> T:
        """模型工厂方法"""
        # 导入所有可用模型
        from .efficientnet import EfficientNetDetector
        from .resnet import ResNetDetector
        from .vision_transformer import ViTDetector

        model_classes = {
            'efficientnet': EfficientNetDetector,
            'resnet': ResNetDetector,
            'vit': ViTDetector
        }

        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_classes[model_type](config_path)