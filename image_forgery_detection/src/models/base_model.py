import torch
import torch.nn as nn
import timm
import logging
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import yaml
from pathlib import Path


class BaseForgeryDetector(nn.Module, ABC):
    """图像篡改检测模型基类"""

    def __init__(self, config_path: str):
        """
        初始化模型
        Args:
            config_path: 模型配置文件路径
        """
        super().__init__()

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化日志
        self.logger = logging.getLogger(__name__)

        # 构建模型组件
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = self._build_head()
        self.auxiliary_heads = self._build_auxiliary_heads()

        # 初始化权重
        self._init_weights()

    def _build_backbone(self) -> nn.Module:
        """构建主干网络"""
        try:
            backbone_config = self.config['backbone']
            model_name = backbone_config['name']
            pretrained = backbone_config.get('pretrained', True)

            # 使用timm加载预训练模型
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4)
            )

            # 冻结指定层
            if 'frozen_stages' in backbone_config:
                self._freeze_stages(backbone, backbone_config['frozen_stages'])

            return backbone

        except Exception as e:
            self.logger.error(f"构建主干网络失败: {str(e)}")
            raise

    def _build_neck(self) -> nn.Module:
        """构建颈部网络(特征融合)"""
        try:
            if 'neck' not in self.config:
                return nn.Identity()

            neck_config = self.config['neck']
            neck_type = neck_config['type']

            if neck_type == 'FPN':
                return FeaturePyramidNetwork(
                    in_channels=neck_config['in_channels'],
                    out_channels=neck_config['out_channels']
                )
            elif neck_type == 'ASPP':
                return AtrousSpatialPyramidPooling(
                    in_channels=neck_config['in_channels'],
                    out_channels=neck_config['out_channels']
                )
            else:
                raise ValueError(f"不支持的颈部网络类型: {neck_type}")

        except Exception as e:
            self.logger.error(f"构建颈部网络失败: {str(e)}")
            raise

    def _build_head(self) -> Dict[str, nn.Module]:
        """构建检测头"""
        try:
            heads = {}
            head_config = self.config['head']

            # 分类头
            if 'classification' in head_config:
                cls_config = head_config['classification']
                heads['cls'] = ClassificationHead(
                    in_channels=cls_config['in_channels'],
                    hidden_dims=cls_config['hidden_dims'],
                    num_classes=cls_config['num_classes'],
                    dropout=cls_config.get('dropout', 0.5)
                )

            # 分割头
            if 'segmentation' in head_config:
                seg_config = head_config['segmentation']
                heads['seg'] = SegmentationHead(
                    in_channels=seg_config['in_channels'],
                    out_channels=seg_config['out_channels'],
                    decoder_channels=seg_config['decoder_channels']
                )

            return nn.ModuleDict(heads)

        except Exception as e:
            self.logger.error(f"构建检测头失败: {str(e)}")
            raise

    def _build_auxiliary_heads(self) -> Dict[str, nn.Module]:
        """构建辅助任务头"""
        try:
            aux_heads = {}

            if 'auxiliary_tasks' not in self.config:
                return nn.ModuleDict(aux_heads)

            aux_config = self.config['auxiliary_tasks']

            # 噪声等级估计
            if aux_config.get('noise_level', {}).get('enabled', False):
                aux_heads['noise'] = NoiseEstimationHead(
                    in_channels=aux_config['noise_level']['in_channels'],
                    hidden_dims=aux_config['noise_level']['hidden_dims']
                )

            # 边缘一致性预测
            if aux_config.get('edge_consistency', {}).get('enabled', False):
                aux_heads['edge'] = EdgeConsistencyHead(
                    in_channels=aux_config['edge_consistency']['in_channels'],
                    hidden_dims=aux_config['edge_consistency']['hidden_dims']
                )

            return nn.ModuleDict(aux_heads)

        except Exception as e:
            self.logger.error(f"构建辅助任务头失败: {str(e)}")
            raise

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_stages(self, model: nn.Module, num_stages: int):
        """冻结指定层的参数"""
        if num_stages <= 0:
            return

        for i in range(num_stages):
            for name, param in model.named_parameters():
                if f'layer{i}' in name:
                    param.requires_grad = False

    def _channel_squeeze(self, x: List[torch.Tensor],
                         squeeze_factor: int = 4) -> List[torch.Tensor]:
        """通道压缩"""
        squeezed = []
        for feat in x:
            # 使用1x1卷积压缩通道
            squeeze = nn.Conv2d(
                feat.size(1),
                feat.size(1) // squeeze_factor,
                kernel_size=1
            ).to(feat.device)
            squeezed.append(squeeze(feat))
        return squeezed

    def _spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """空间注意力"""
        # 计算通道维度的最大值和平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # 拼接并通过卷积得到注意力图
        attention = torch.cat([max_pool, avg_pool], dim=1)
        conv = nn.Conv2d(2, 1, kernel_size=7, padding=3).to(x.device)
        attention = torch.sigmoid(conv(attention))

        return x * attention

    def _channel_attention(self, x: torch.Tensor,
                           reduction: int = 16) -> torch.Tensor:
        """通道注意力"""
        b, c, h, w = x.size()

        # 全局平均池化
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)

        # MLP
        fc1 = nn.Linear(c, c // reduction).to(x.device)
        fc2 = nn.Linear(c // reduction, c).to(x.device)
        y = torch.relu(fc1(y))
        y = torch.sigmoid(fc2(y)).view(b, c, 1, 1)

        return x * y

    def _pyramid_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """金字塔池化"""
        b, c, h, w = x.size()
        pool_sizes = [1, 2, 3, 6]
        pool_outs = [x]

        for size in pool_sizes:
            # 自适应平均池化
            pool = nn.AdaptiveAvgPool2d(size)(x)

            # 上采样回原始尺寸
            up = nn.Upsample(
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )(pool)

            pool_outs.append(up)

        return torch.cat(pool_outs, dim=1)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤
        Args:
            batch: 包含图像和标签的批次数据
        Returns:
            损失和预测结果
        """
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        # 特征提取
        features = self.backbone(images)
        neck_feat = self.neck(features)

        outputs = {}
        losses = {}

        # 主任务前向传播
        if 'cls' in self.head:
            cls_out = self.head['cls'](neck_feat)
            outputs['cls'] = cls_out
            losses['cls'] = self.criterion['cls'](cls_out, labels)

        if 'seg' in self.head and 'mask' in batch:
            masks = batch['mask'].to(self.device)
            seg_out = self.head['seg'](neck_feat)
            outputs['seg'] = seg_out
            losses['seg'] = self.criterion['seg'](seg_out, masks)

        # 辅助任务前向传播
        for name, head in self.auxiliary_heads.items():
            aux_out = head(neck_feat)
            outputs[name] = aux_out
            if f'{name}_target' in batch:
                target = batch[f'{name}_target'].to(self.device)
                losses[name] = self.criterion[name](aux_out, target)

        # 计算总损失
        total_loss = sum(
            loss * self.config['loss_weights'].get(name, 1.0)
            for name, loss in losses.items()
        )
        losses['total'] = total_loss

        return {'outputs': outputs, 'losses': losses}

    def inference_step(self,
                       batch: Dict[str, torch.Tensor],
                       return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        推理步骤
        Args:
            batch: 输入数据批次
            return_features: 是否返回特征
        Returns:
            预测结果
        """
        images = batch['image'].to(self.device)

        with torch.no_grad():
            # 特征提取
            features = self.backbone(images)
            neck_feat = self.neck(features)

            outputs = {}

            # 主任务预测
            if 'cls' in self.head:
                outputs['cls'] = torch.softmax(
                    self.head['cls'](neck_feat),
                    dim=1
                )

            if 'seg' in self.head:
                outputs['seg'] = torch.sigmoid(
                    self.head['seg'](neck_feat)
                )

            # 辅助任务预测
            for name, head in self.auxiliary_heads.items():
                outputs[name] = head(neck_feat)

            # 返回特征
            if return_features:
                outputs['features'] = {
                    'backbone': features,
                    'neck': neck_feat
                }

            return outputs

    def get_features(self,
                     batch: Dict[str, torch.Tensor],
                     layers: List[str]) -> Dict[str, torch.Tensor]:
        """
        提取指定层的特征
        Args:
            batch: 输入数据
            layers: 需要提取特征的层名列表
        Returns:
            特征字典
        """
        images = batch['image'].to(self.device)
        features = {}

        with torch.no_grad():
            # 主干网络特征
            if any(layer.startswith('backbone') for layer in layers):
                backbone_feat = self.backbone(images)
                for i, feat in enumerate(backbone_feat):
                    features[f'backbone_layer{i + 1}'] = feat

            # 颈部特征
            if 'neck' in layers:
                neck_feat = self.neck(backbone_feat)
                features['neck'] = neck_feat

            # 头部特征
            if any(layer.startswith('head') for layer in layers):
                for name, head in self.head.items():
                    if f'head_{name}' in layers:
                        features[f'head_{name}'] = head(neck_feat)

        return features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        raise NotImplementedError

    def save_features(self,
                      features: Dict[str, torch.Tensor],
                      save_dir: Union[str, Path],
                      prefix: str = ''):
        """
        保存特征图
        Args:
            features: 特征字典
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, feat in features.items():
            # 转换为numpy数组
            feat_np = feat.cpu().numpy()

            # 保存为npy文件
            save_path = save_dir / f"{prefix}_{name}.npy"
            np.save(str(save_path), feat_np)

    def load_checkpoint(self, ckpt_path: Union[str, Path]):
        """加载检查点"""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt['model_state_dict'])
        return ckpt

    def analyze_predictions(self,
                            outputs: Dict[str, torch.Tensor],
                            threshold: float = 0.5) -> Dict[str, Any]:
        """
        分析模型预测结果
        Args:
            outputs: 模型输出
            threshold: 预测阈值
        Returns:
            分析结果
        """
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

            # 计算篡改区域特征
            results['segmentation'] = {
                'masks': binary_masks,
                'area_ratio': np.mean(binary_masks),
                'confidence_map': masks
            }

            # 分析篡改区域
            for i in range(len(binary_masks)):
                mask = binary_masks[i]

                # 连通区域分析
                from scipy import ndimage
                labeled, num_features = ndimage.label(mask)
                if num_features > 0:
                    # 计算每个篡改区域的属性
                    regions = []
                    for j in range(1, num_features + 1):
                        region = (labeled == j)
                        area = np.sum(region)
                        centroid = ndimage.center_of_mass(region)
                        bbox = self._get_bbox(region)

                        regions.append({
                            'area': area,
                            'centroid': centroid,
                            'bbox': bbox,
                            'confidence': float(np.mean(masks[i][region]))
                        })
                    results['segmentation'][f'regions_{i}'] = regions

        # 噪声等级预测分析
        if 'noise' in outputs:
            noise_levels = outputs['noise'].cpu().numpy()
            results['noise_estimation'] = {
                'levels': noise_levels,
                'mean_level': float(np.mean(noise_levels))
            }

        # 边缘一致性预测分析
        if 'edge' in outputs:
            edge_scores = outputs['edge'].cpu().numpy()
            results['edge_consistency'] = {
                'scores': edge_scores,
                'mean_score': float(np.mean(edge_scores))
            }

        return results

    def _get_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """获取掩码的边界框"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (x1, y1, x2, y2)

    def profile(self,
                input_shape: Tuple[int, ...],
                batch_size: int = 1) -> Dict[str, Any]:
        """
        分析模型性能
        Args:
            input_shape: 输入张量形状 (C, H, W)
            batch_size: 批次大小
        Returns:
            性能统计信息
        """
        from thop import profile as count_ops

        # 创建示例输入
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

        # 计算FLOPs和参数量
        flops, params = count_ops(self, dummy_input)

        # 统计每层参数量
        layer_params = {}
        for name, module in self.named_modules():
            if any(isinstance(module, t) for t in [nn.Conv2d, nn.Linear, nn.BatchNorm2d]):
                layer_params[name] = sum(p.numel() for p in module.parameters())

        # 测试推理速度
        import time
        times = []
        for _ in range(100):  # 进行100次测试取平均
            start = time.time()
            with torch.no_grad():
                _ = self(dummy_input)
            times.append(time.time() - start)

        avg_time = np.mean(times[10:])  # 去掉前10次预热
        fps = batch_size / avg_time

        # 计算模型大小
        model_size = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 / 1024  # MB

        return {
            'flops': flops,
            'params': params,
            'layer_params': layer_params,
            'inference_time': avg_time,
            'fps': fps,
            'model_size': model_size
        }

    def visualize_features(self, features: Dict[str, torch.Tensor],
                           save_dir: Optional[Union[str, Path]] = None):
        """
        可视化特征图
        Args:
            features: 特征字典
            save_dir: 保存目录
        """
        import matplotlib.pyplot as plt

        for name, feat in features.items():
            # 将特征图转换为CPU numpy数组
            feat_np = feat.detach().cpu().numpy()

            if len(feat_np.shape) == 4:  # (B, C, H, W)
                # 计算特征图的平均值和最大值响应
                mean_response = np.mean(feat_np, axis=1)  # (B, H, W)
                max_response = np.max(feat_np, axis=1)  # (B, H, W)

                for i in range(feat_np.shape[0]):  # 遍历批次
                    plt.figure(figsize=(12, 6))

                    # 显示平均响应
                    plt.subplot(121)
                    plt.imshow(mean_response[i], cmap='jet')
                    plt.colorbar()
                    plt.title(f'{name} - Mean Response (Sample {i})')

                    # 显示最大响应
                    plt.subplot(122)
                    plt.imshow(max_response[i], cmap='jet')
                    plt.colorbar()
                    plt.title(f'{name} - Max Response (Sample {i})')

                    if save_dir:
                        save_dir = Path(save_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        plt.savefig(save_dir / f'{name}_sample{i}.png')
                    else:
                        plt.show()
                    plt.close()

    def export_onnx(self, save_path: Union[str, Path],
                    input_shape: Tuple[int, ...]):
        """
        导出ONNX模型
        Args:
            save_path: 保存路径
            input_shape: 输入形状 (B, C, H, W)
        """
        import onnx
        import onnxruntime

        # 创建示例输入
        dummy_input = torch.randn(*input_shape).to(self.device)

        # 导出模型
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # 验证导出的模型
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        # 测试推理
        ort_session = onnxruntime.InferenceSession(save_path)
        ort_inputs = {
            'input': dummy_input.cpu().numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)

        # 比较PyTorch和ONNX输出
        torch_output = self(dummy_input).detach().cpu().numpy()
        np.testing.assert_allclose(
            torch_output,
            ort_outputs[0],
            rtol=1e-03,
            atol=1e-05
        )

        self.logger.info(f"模型已成功导出到 {save_path}")

    def export_torchscript(self, save_path: Union[str, Path],
                           input_shape: Tuple[int, ...]):
        """
        导出TorchScript模型
        Args:
            save_path: 保存路径
            input_shape: 输入形状
        """
        # 创建示例输入
        dummy_input = torch.randn(*input_shape).to(self.device)

        # 转换为TorchScript
        traced_script_module = torch.jit.trace(self, dummy_input)

        # 保存模型
        traced_script_module.save(str(save_path))

        # 验证导出的模型
        loaded_model = torch.jit.load(str(save_path))
        test_output = loaded_model(dummy_input)
        torch_output = self(dummy_input)

        assert torch.allclose(test_output, torch_output, rtol=1e-03, atol=1e-05)

        self.logger.info(f"模型已成功导出到 {save_path}")

    def summary(self):
        """打印模型概要"""
        from torchsummary import summary

        # 确定输入大小
        if hasattr(self, 'input_shape'):
            input_size = self.input_shape
        else:
            input_size = (3, 224, 224)  # 默认输入大小

        print("\n模型概要:")
        print("=" * 50)
        summary(self, input_size)
        print("=" * 50)

        # 打印每个组件的参数量
        total_params = 0
        print("\n各组件参数统计:")
        print("-" * 50)
        for name, module in [
            ('Backbone', self.backbone),
            ('Neck', self.neck),
            ('Heads', self.head),
            ('Auxiliary Heads', self.auxiliary_heads)
        ]:
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {params:,} parameters")
            total_params += params

        print("-" * 50)
        print(f"Total: {total_params:,} parameters")
        print("\n")

        # 打印配置信息
        print("模型配置:")
        print("-" * 50)
        for key, value in self.config.items():
            print(f"{key}: {value}")

    @staticmethod
    def create(model_type: str, config_path: str) -> 'BaseForgeryDetector':
        """
        模型工厂方法
        Args:
            model_type: 模型类型
            config_path: 配置文件路径
        Returns:
            模型实例
        """
        # 导入所有可用的模型
        from .efficientnet import EfficientNetDetector
        from .resnet import ResNetDetector
        from .vision_transformer import ViTDetector

        model_classes = {
            'efficientnet': EfficientNetDetector,
            'resnet': ResNetDetector,
            'vit': ViTDetector
        }

        if model_type not in model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")

        return model_classes[model_type](config_path)

