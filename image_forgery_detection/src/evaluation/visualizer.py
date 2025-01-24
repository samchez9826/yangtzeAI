import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class VisualizationManager:
    """可视化管理器"""

    def __init__(self, save_dir: Union[str, Path]):
        """
        初始化可视化管理器
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 设置样式
        plt.style.use('seaborn')

    def plot_training_curves(self, history: Dict[str, List[float]],
                             title: str = 'Training History'):
        """
        绘制训练曲线
        Args:
            history: 训练历史记录
            title: 图表标题
        """
        plt.figure(figsize=(12, 6))

        for metric_name, values in history.items():
            if 'val_' in metric_name:
                continue
            plt.plot(values, label=f'Train {metric_name}')

            # 绘制验证集曲线
            val_metric = f'val_{metric_name}'
            if val_metric in history:
                plt.plot(history[val_metric], '--',
                         label=f'Val {metric_name}')

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = 'Confusion Matrix'):
        """
        绘制混淆矩阵
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签
            title: 图表标题
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)

        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                       title: str = 'ROC Curve'):
        """
        绘制ROC曲线
        Args:
            y_true: 真实标签
            y_score: 预测概率
            title: 图表标题
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', label=f'ROC')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')

        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def plot_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                      title: str = 'Precision-Recall Curve'):
        """
        绘制PR曲线
        Args:
            y_true: 真实标签
            y_score: 预测概率
            title: 图表标题
        """
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, 'b-', label='PR Curve')

        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def visualize_predictions(self, images: List[np.ndarray],
                              masks: List[np.ndarray],
                              pred_masks: List[np.ndarray],
                              max_samples: int = 16):
        """
        可视化分割预测结果
        Args:
            images: 原始图像列表
            masks: 真实掩码列表
            pred_masks: 预测掩码列表
            max_samples: 最大样本数
        """
        n_samples = min(len(images), max_samples)
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for idx in range(n_samples):
            # 原始图像
            axes[idx, 0].imshow(images[idx])
            axes[idx, 0].imshow(images[idx])
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')

            # 真实掩码
            axes[idx, 1].imshow(masks[idx], cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')

            # 预测掩码
            axes[idx, 2].imshow(pred_masks[idx], cmap='gray')
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')

        plt.tight_layout()
        save_path = self.save_dir / 'segmentation_results.png'
        plt.savefig(save_path)
        plt.close()

    def visualize_attention_maps(self, image: np.ndarray,
                                 attention_maps: List[np.ndarray],
                                 title: str = 'Attention Maps'):
        """
        可视化注意力图
        Args:
            image: 输入图像
            attention_maps: 注意力图列表
            title: 图表标题
        """
        n_maps = len(attention_maps)
        fig, axes = plt.subplots(1, n_maps + 1, figsize=(5 * (n_maps + 1), 5))

        # 显示原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 显示注意力图
        for idx, attn_map in enumerate(attention_maps):
            # 归一化注意力图
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

            # 调整大小以匹配原始图像
            attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))

            # 使用热力图显示
            axes[idx + 1].imshow(image)
            axes[idx + 1].imshow(attn_map, cmap='jet', alpha=0.5)
            axes[idx + 1].set_title(f'Attention Map {idx + 1}')
            axes[idx + 1].axis('off')

        plt.tight_layout()
        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def visualize_feature_maps(self, feature_maps: List[torch.Tensor],
                               max_features: int = 16,
                               title: str = 'Feature Maps'):
        """
        可视化特征图
        Args:
            feature_maps: 特征图列表
            max_features: 每层显示的最大特征数
            title: 图表标题
        """
        for layer_idx, features in enumerate(feature_maps):
            features = features.detach().cpu().numpy()

            # 选择要显示的特征图
            n_features = min(features.shape[1], max_features)
            selected_features = features[0, :n_features]

            # 计算网格大小
            grid_size = int(np.ceil(np.sqrt(n_features)))

            fig, axes = plt.subplots(grid_size, grid_size,
                                     figsize=(2 * grid_size, 2 * grid_size))

            for idx, feat in enumerate(selected_features):
                if idx >= grid_size * grid_size:
                    break

                i, j = idx // grid_size, idx % grid_size

                # 归一化特征图
                feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

                axes[i, j].imshow(feat, cmap='viridis')
                axes[i, j].axis('off')

            # 隐藏多余的子图
            for idx in range(n_features, grid_size * grid_size):
                i, j = idx // grid_size, idx % grid_size
                axes[i, j].axis('off')

            plt.suptitle(f'Layer {layer_idx + 1} Feature Maps')
            plt.tight_layout()

            save_path = self.save_dir / f"feature_maps_layer{layer_idx + 1}.png"
            plt.savefig(save_path)
            plt.close()

    def visualize_gradients(self, gradients: torch.Tensor,
                            title: str = 'Gradient Analysis'):
        """
        可视化梯度分布
        Args:
            gradients: 梯度张量
            title: 图表标题
        """
        grads = gradients.detach().cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))

        # 绘制梯度直方图
        plt.hist(grads, bins=50, alpha=0.7)
        plt.title(f'{title}\nMean: {grads.mean():.2e}, Std: {grads.std():.2e}')
        plt.xlabel('Gradient Value')
        plt.ylabel('Count')
        plt.grid(True)

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def plot_learning_rate(self, learning_rates: List[float],
                           title: str = 'Learning Rate Schedule'):
        """
        绘制学习率变化曲线
        Args:
            learning_rates: 学习率列表
            title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()

    def visualize_model_architecture(self, model: nn.Module,
                                     input_size: Tuple[int, ...] = (3, 224, 224),
                                     title: str = 'Model Architecture'):
        """
        可视化模型架构
        Args:
            model: PyTorch模型
            input_size: 输入大小
            title: 图表标题
        """
        try:
            from torchviz import make_dot

            # 创建示例输入
            x = torch.randn(1, *input_size)
            y = model(x)

            # 创建计算图可视化
            dot = make_dot(y, params=dict(model.named_parameters()))

            # 保存图像
            save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
            dot.render(str(save_path), format='png', cleanup=True)

        except ImportError:
            logger.warning("torchviz not installed. Skip architecture visualization.")

    def visualize_class_distribution(self, labels: np.ndarray,
                                     class_names: Optional[List[str]] = None,
                                     title: str = 'Class Distribution'):
        """
        可视化类别分布
        Args:
            labels: 标签数组
            class_names: 类别名称列表
            title: 图表标题
        """
        classes, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=classes if class_names is None else class_names,
                    y=counts)

        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        save_path = self.save_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def save_metrics_table(self, metrics: Dict[str, float],
                           filename: str = 'metrics_summary.csv'):
        """
        保存评估指标表格
        Args:
            metrics: 指标字典
            filename: 保存文件名
        """
        df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        save_path = self.save_dir / filename
        df.to_csv(save_path, index=False)
        logger.info(f"Metrics saved to {save_path}")