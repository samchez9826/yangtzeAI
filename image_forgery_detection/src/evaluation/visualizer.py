import string

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import cv2
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
from datetime import datetime
import weakref
from PIL import Image
import io
import gc

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncPlotManager:
    """异步绘图管理器"""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = weakref.WeakSet()
        self._lock = Lock()

    def submit(self, func, *args, **kwargs):
        """提交绘图任务"""
        future = self._executor.submit(func, *args, **kwargs)
        with self._lock:
            self._futures.add(future)
        return future

    def wait_all(self, timeout: Optional[float] = None):
        """等待所有绘图任务完成"""
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Plot task failed: {e}")


class VisualizationManager:
    """可视化管理器"""

    def __init__(self,
                 save_dir: Union[str, Path],
                 max_workers: int = 4,
                 style: str = 'seaborn-darkgrid',
                 dpi: int = 300,
                 fig_format: str = 'png',
                 max_memory_gb: float = 4.0):
        """
        初始化可视化管理器
        Args:
            save_dir: 保存目录
            max_workers: 最大工作线程数
            style: matplotlib样式
            dpi: 图像DPI
            fig_format: 图像保存格式
            max_memory_gb: 最大内存使用量(GB)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.plot_dir = self.save_dir / 'plots'
        self.metrics_dir = self.save_dir / 'metrics'
        self.predictions_dir = self.save_dir / 'predictions'
        for dir_path in [self.plot_dir, self.metrics_dir, self.predictions_dir]:
            dir_path.mkdir(exist_ok=True)

        self.dpi = dpi
        self.fig_format = fig_format
        self.max_memory_gb = max_memory_gb

        # 初始化异步绘图管理器
        self._plot_manager = AsyncPlotManager(max_workers=max_workers)

        # 设置matplotlib样式
        plt.style.use(style)
        sns.set_context("paper", font_scale=1.2)

        # 状态跟踪
        self._plot_count = 0
        self._memory_tracker = self._track_memory()

    def _track_memory(self):
        """内存使用跟踪器"""
        while True:
            memory_gb = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
            if memory_gb > self.max_memory_gb:
                logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeds limit ({self.max_memory_gb}GB)")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            yield

    def plot_training_curves(self,
                             history: Dict[str, List[float]],
                             title: str = 'Training History',
                             smooth_factor: float = 0.6,
                             show_std: bool = True) -> None:
        """
        绘制训练曲线
        Args:
            history: 训练历史记录
            title: 图表标题
            smooth_factor: 平滑因子
            show_std: 是否显示标准差区域
        """

        def plot_func():
            plt.figure(figsize=(12, 6))

            for metric_name, values in history.items():
                if 'val_' in metric_name:
                    continue

                # 应用平滑
                values = np.array(values)
                smoothed_values = self._smooth_curve(values, smooth_factor)

                # 计算标准差区域
                if show_std and len(values) > 1:
                    std = np.std(values)
                    plt.fill_between(range(len(values)),
                                     smoothed_values - std,
                                     smoothed_values + std,
                                     alpha=0.2)

                plt.plot(smoothed_values, label=f'Train {metric_name}',
                         linewidth=2)

                # 绘制验证集曲线
                val_metric = f'val_{metric_name}'
                if val_metric in history:
                    val_values = np.array(history[val_metric])
                    smoothed_val = self._smooth_curve(val_values, smooth_factor)
                    plt.plot(smoothed_val, '--',
                             label=f'Val {metric_name}',
                             linewidth=2)

            plt.title(title, pad=20)
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            save_path = self.plot_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存原始数据
            data_path = self.plot_dir / f"{self._sanitize_filename(title)}.json"
            with open(data_path, 'w') as f:
                json.dump({
                    'history': history,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'smooth_factor': smooth_factor
                    }
                }, f, indent=4)

        self._plot_manager.submit(plot_func)

    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = 'Confusion Matrix',
                              normalize: bool = True,
                              cmap: str = 'Blues') -> None:
        """
        绘制混淆矩阵
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签
            title: 图表标题
            normalize: 是否归一化
            cmap: 颜色主题
        """

        def plot_func():
            cm = confusion_matrix(y_true, y_pred)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                        cmap=cmap, xticklabels=labels, yticklabels=labels)

            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('True')

            save_path = self.metrics_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存混淆矩阵数据
            data_path = self.metrics_dir / f"{self._sanitize_filename(title)}.csv"
            df = pd.DataFrame(cm, index=labels, columns=labels)
            df.to_csv(data_path)

        self._plot_manager.submit(plot_func)

    def visualize_predictions(self,
                              images: List[np.ndarray],
                              masks: List[np.ndarray],
                              pred_masks: List[np.ndarray],
                              max_samples: int = 16,
                              overlay_alpha: float = 0.5) -> None:
        """
        可视化分割预测结果
        Args:
            images: 原始图像列表
            masks: 真实掩码列表
            pred_masks: 预测掩码列表
            max_samples: 最大样本数
            overlay_alpha: 叠加透明度
        """

        def plot_func():
            n_samples = min(len(images), max_samples)
            fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))

            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for idx in range(n_samples):
                # 原始图像
                axes[idx, 0].imshow(self._normalize_image(images[idx]))
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

                # 叠加显示
                overlay = self._create_overlay(images[idx], pred_masks[idx], overlay_alpha)
                axes[idx, 3].imshow(overlay)
                axes[idx, 3].set_title('Overlay')
                axes[idx, 3].axis('off')

            plt.tight_layout()
            save_path = self.predictions_dir / f'predictions_batch_{self._plot_count}.{self.fig_format}'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            self._plot_count += 1

        self._plot_manager.submit(plot_func)

    def visualize_attention_maps(self,
                                 image: np.ndarray,
                                 attention_maps: List[np.ndarray],
                                 title: str = 'Attention Maps',
                                 colormap: str = 'jet') -> None:
        """
        可视化注意力图
        Args:
            image: 输入图像
            attention_maps: 注意力图列表
            title: 图表标题
            colormap: 颜色主题
        """

        def plot_func():
            n_maps = len(attention_maps)
            fig, axes = plt.subplots(1, n_maps + 1, figsize=(5 * (n_maps + 1), 5))

            # 显示原始图像
            axes[0].imshow(self._normalize_image(image))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # 显示注意力图
            for idx, attn_map in enumerate(attention_maps):
                # 归一化注意力图
                attn_map = self._normalize_attention_map(attn_map)

                # 调整大小以匹配原始图像
                attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))

                # 创建叠加图
                axes[idx + 1].imshow(image)
                im = axes[idx + 1].imshow(attn_map, cmap=colormap, alpha=0.7)
                axes[idx + 1].set_title(f'Attention Map {idx + 1}')
                axes[idx + 1].axis('off')

                # 添加颜色条
                plt.colorbar(im, ax=axes[idx + 1])

            plt.tight_layout()
            save_path = self.plot_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        self._plot_manager.submit(plot_func)

    def plot_metrics_comparison(self,
                                metrics_list: List[Dict[str, float]],
                                labels: List[str],
                                title: str = 'Metrics Comparison') -> None:
        """
        绘制指标对比图
        Args:
            metrics_list: 多组指标数据
            labels: 每组指标的标签
            title: 图表标题
        """

        def plot_func():
            # 准备数据
            metrics = pd.DataFrame(metrics_list)
            metrics.index = labels

            # 创建热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(metrics, annot=True, fmt='.3f', cmap='RdYlBu_r',
                        center=0, vmin=-1, vmax=1)

            plt.title(title)
            plt.ylabel('Model / Method')
            plt.xlabel('Metric')

            save_path = self.metrics_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存原始数据
            data_path = self.metrics_dir / f"{self._sanitize_filename(title)}.csv"
            metrics.to_csv(data_path)

        self._plot_manager.submit(plot_func)


    def generate_report(self, filename: str = 'visualization_report.html') -> None:
        """
        生成可视化报告
        Args:
            filename: 报告文件名
        """
        try:
            # 等待所有绘图任务完成
            self._plot_manager.wait_all(timeout=300)

            # 生成HTML报告
            report_html = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '<meta charset="utf-8">',
                '<style>',
                'body { font-family: Arial, sans-serif; margin: 20px; }',
                '.plot-container { margin: 20px 0; }',
                '.plot-title { font-size: 1.2em; color: #333; }',
                '.metrics-table { border-collapse: collapse; width: 100%; }',
                '.metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; }',
                '.metrics-table th { background-color: #f5f5f5; }',
                '</style>',
                '</head>',
                '<body>',
                f'<h1>Visualization Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h1>'
            ]

            # 添加训练曲线
            training_plots = sorted(self.plot_dir.glob('training_*.png'))
            if training_plots:
                report_html.extend([
                    '<h2>Training Progress</h2>',
                    '<div class="plot-container">',
                    *[f'<img src="{plot.relative_to(self.save_dir)}" />' for plot in training_plots],
                    '</div>'
                ])

            # 添加评估指标
            metrics_files = sorted(self.metrics_dir.glob('*.csv'))
            if metrics_files:
                report_html.append('<h2>Evaluation Metrics</h2>')
                for metrics_file in metrics_files:
                    df = pd.read_csv(metrics_file)
                    report_html.extend([
                        f'<h3>{metrics_file.stem}</h3>',
                        df.to_html(classes='metrics-table')
                    ])

            # 添加预测结果可视化
            prediction_plots = sorted(self.predictions_dir.glob('*.png'))
            if prediction_plots:
                report_html.extend([
                    '<h2>Prediction Results</h2>',
                    '<div class="plot-container">',
                    *[f'<img src="{plot.relative_to(self.save_dir)}" />' for plot in prediction_plots],
                    '</div>'
                ])

            # 添加其他可视化结果
            other_plots = [p for p in self.plot_dir.glob('*.png')
                           if not p.name.startswith('training_')]
            if other_plots:
                report_html.extend([
                    '<h2>Other Visualizations</h2>',
                    '<div class="plot-container">',
                    *[f'<img src="{plot.relative_to(self.save_dir)}" />' for plot in other_plots],
                    '</div>'
                ])

            report_html.extend(['</body>', '</html>'])

            # 保存报告
            report_path = self.save_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_html))

            logger.info(f"Report generated successfully: {report_path}")

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def _smooth_curve(self, values: np.ndarray, weight: float) -> np.ndarray:
        """
        使用指数移动平均平滑曲线
        Args:
            values: 输入数据
            weight: 平滑权重
        Returns:
            平滑后的数据
        """
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
        return smoothed

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像
        Args:
            image: 输入图像
        Returns:
            归一化后的图像
        """
        if image.dtype == np.uint8:
            return image

        image = image.astype(np.float32)
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _normalize_attention_map(self, attention_map: np.ndarray) -> np.ndarray:
        """
        归一化注意力图
        Args:
            attention_map: 注意力图
        Returns:
            归一化后的注意力图
        """
        if len(attention_map.shape) == 3:
            attention_map = attention_map.mean(axis=-1)

        attention_map = attention_map.astype(np.float32)
        attention_map = (attention_map - attention_map.min()) / \
                        (attention_map.max() - attention_map.min() + 1e-8)
        return attention_map

    def _create_overlay(self,
                        image: np.ndarray,
                        mask: np.ndarray,
                        alpha: float = 0.5,
                        color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        创建叠加显示图像
        Args:
            image: 原始图像
            mask: 掩码
            alpha: 透明度
            color: 掩码颜色
        Returns:
            叠加后的图像
        """
        image = self._normalize_image(image)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0.5] = color
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名
        Args:
            filename: 原始文件名
        Returns:
            清理后的文件名
        """
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        return ''.join(c for c in filename if c in valid_chars).replace(' ', '_')

    def _save_plot_data(self, data: Dict, metadata: Dict, filename: str) -> None:
        """
        保存绘图数据
        Args:
            data: 绘图数据
            metadata: 元数据
            filename: 文件名
        """
        save_path = self.plot_dir / f"{self._sanitize_filename(filename)}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'data': data,
                'metadata': {
                    **metadata,
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }, f, indent=4)

    def __del__(self):
        """清理资源"""
        try:
            self._plot_manager.wait_all(timeout=60)
            plt.close('all')
            gc.collect()
        except Exception as e:
            logger.error(f"Error cleaning up visualizer: {e}")

def create_visualizer(save_dir: Union[str, Path], **kwargs) -> VisualizationManager:
    """
    创建可视化管理器实例
    Args:
        save_dir: 保存目录
        **kwargs: 其他参数
    Returns:
        VisualizationManager实例
    """
    return VisualizationManager(save_dir, **kwargs)


class AdvancedVisualizationManager(VisualizationManager):
    def __init__(self,
                 save_dir: Union[str, Path],
                 max_workers: int = 4,
                 style: str = 'seaborn-darkgrid',
                 dpi: int = 300,
                 fig_format: str = 'png',
                 max_memory_gb: float = 4.0):
        # 正确调用父类初始化
        super().__init__(save_dir, max_workers, style, dpi, fig_format, max_memory_gb)


    """高级可视化管理器"""

    def plot_feature_importance(self,
                                feature_names: List[str],
                                importance_scores: np.ndarray,
                                title: str = 'Feature Importance Analysis',
                                top_k: Optional[int] = None) -> None:
        """
        绘制特征重要性分析图
        Args:
            feature_names: 特征名列表
            importance_scores: 重要性分数
            title: 图表标题
            top_k: 显示前k个重要特征
        """

        def plot_func():
            # 排序特征
            indices = np.argsort(importance_scores)
            if top_k:
                indices = indices[-top_k:]

            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(indices)), importance_scores[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', va='center')

            plt.title(title)
            plt.xlabel('Importance Score')
            plt.tight_layout()

            save_path = self.plot_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存数据
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            })
            df.to_csv(self.plot_dir / f"{self._sanitize_filename(title)}.csv", index=False)

        self._plot_manager.submit(plot_func)

    def plot_correlation_matrix(self,
                                data: pd.DataFrame,
                                title: str = 'Feature Correlation Matrix',
                                method: str = 'pearson',
                                annotate: bool = True) -> None:
        """
        绘制特征相关性矩阵
        Args:
            data: 特征数据
            title: 图表标题
            method: 相关系数计算方法
            annotate: 是否显示数值标注
        """

        def plot_func():
            # 计算相关性矩阵
            corr_matrix = data.corr(method=method)

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix), k=1)

            # 使用seaborn绘制热力图
            sns.heatmap(corr_matrix, mask=mask, annot=annotate, fmt='.2f',
                        cmap='coolwarm', center=0, square=True)

            plt.title(title)
            plt.tight_layout()

            save_path = self.plot_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存相关性数据
            corr_matrix.to_csv(self.plot_dir / f"{self._sanitize_filename(title)}.csv")

        self._plot_manager.submit(plot_func)

    def plot_model_comparison(self,
                              model_metrics: Dict[str, Dict[str, float]],
                              title: str = 'Model Performance Comparison') -> None:
        """
        绘制模型性能对比图
        Args:
            model_metrics: 模型指标字典 {model_name: {metric_name: value}}
            title: 图表标题
        """

        def plot_func():
            # 转换数据格式
            df = pd.DataFrame(model_metrics).T

            # 设置图表样式
            plt.style.use('seaborn')
            fig, ax = plt.subplots(figsize=(12, 6))

            # 创建分组柱状图
            x = np.arange(len(df.index))
            width = 0.8 / len(df.columns)

            for i, col in enumerate(df.columns):
                ax.bar(x + i * width, df[col], width, label=col)

            ax.set_title(title)
            ax.set_xticks(x + width * len(df.columns) / 2)
            ax.set_xticklabels(df.index, rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.metrics_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存比较数据
            df.to_csv(self.metrics_dir / f"{self._sanitize_filename(title)}.csv")

        self._plot_manager.submit(plot_func)

    def plot_class_distribution(self,
                                labels: np.ndarray,
                                class_names: Optional[List[str]] = None,
                                title: str = 'Class Distribution',
                                normalize: bool = True) -> None:
        """
        绘制类别分布图
        Args:
            labels: 标签数组
            class_names: 类别名称
            title: 图表标题
            normalize: 是否归一化
        """

        def plot_func():
            plt.figure(figsize=(10, 6))

            counts = np.bincount(labels)
            if normalize:
                counts = counts / len(labels)

            x = range(len(counts))
            bars = plt.bar(x, counts)

            if class_names:
                plt.xticks(x, class_names, rotation=45)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height,
                         f'{height:.2%}' if normalize else f'{int(height)}',
                         ha='center', va='bottom')

            plt.title(title)
            plt.ylabel('Percentage' if normalize else 'Count')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.plot_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        self._plot_manager.submit(plot_func)

    def plot_prediction_error_analysis(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       feature_values: np.ndarray,
                                       feature_names: List[str],
                                       title: str = 'Prediction Error Analysis') -> None:
        """
        绘制预测错误分析图
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            feature_values: 特征值
            feature_names: 特征名称
            title: 图表标题
        """

        def plot_func():
            # 计算错误
            errors = (y_true != y_pred)
            error_indices = np.where(errors)[0]

            if len(error_indices) == 0:
                logger.warning("No prediction errors found.")
                return

            # 创建错误样本的特征分布图
            n_features = len(feature_names)
            fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))

            for i, (name, ax) in enumerate(zip(feature_names, axes)):
                # 正确预测的分布
                sns.kdeplot(data=feature_values[~errors, i],
                            ax=ax, label='Correct', color='blue')
                # 错误预测的分布
                sns.kdeplot(data=feature_values[errors, i],
                            ax=ax, label='Error', color='red')

                ax.set_title(f'Feature: {name}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle(title)
            plt.tight_layout()

            save_path = self.plot_dir / f"{self._sanitize_filename(title)}.{self.fig_format}"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # 保存错误分析数据
            error_df = pd.DataFrame({
                'Feature': feature_names,
                'Error_Mean': [feature_values[errors, i].mean() for i in range(n_features)],
                'Error_Std': [feature_values[errors, i].std() for i in range(n_features)],
                'Correct_Mean': [feature_values[~errors, i].mean() for i in range(n_features)],
                'Correct_Std': [feature_values[~errors, i].std() for i in range(n_features)]
            })
            error_df.to_csv(self.plot_dir / f"{self._sanitize_filename(title)}.csv", index=False)

        self._plot_manager.submit(plot_func)

    def create_interactive_report(self,
                                  filename: str = 'interactive_report.html',
                                  include_plotly: bool = True) -> None:
        """
        创建交互式可视化报告
        Args:
            filename: 报告文件名
            include_plotly: 是否包含plotly交互图表
        """
        try:
            # 等待所有绘图任务完成
            self._plot_manager.wait_all(timeout=300)

            # 创建基础HTML
            report_html = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '<meta charset="utf-8">',
                '<title>Interactive Visualization Report</title>'
            ]

            # 添加必要的JavaScript库
            if include_plotly:
                report_html.extend([
                    '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
                ])

            # 添加CSS样式
            report_html.extend([
                '<style>',
                'body { font-family: Arial, sans-serif; margin: 20px; }',
                '.plot-container { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }',
                '.interactive-plot { width: 100%; height: 500px; }',
                '</style>',
                '</head>',
                '<body>',
                f'<h1>Interactive Visualization Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h1>'
            ])

            # 添加交互式内容
            if include_plotly:
                self._add_interactive_plots(report_html)

            # 添加静态图表
            self._add_static_plots(report_html)

            report_html.extend(['</body>', '</html>'])

            # 保存报告
            report_path = self.save_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_html))

            logger.info(f"Interactive report generated successfully: {report_path}")

        except Exception as e:
            logger.error(f"Error generating interactive report: {e}")
            raise

    def _add_interactive_plots(self, report_html: List[str]) -> None:
        """添加交互式图表"""
        # TODO: Implement interactive plot generation
        pass

    def _add_static_plots(self, report_html: List[str]) -> None:
        """添加静态图表"""
        # TODO: Implement static plot embedding
        pass