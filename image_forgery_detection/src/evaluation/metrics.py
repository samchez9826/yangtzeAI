import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import logging
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricTracker:
    """指标跟踪器,线程安全"""

    def __init__(self, metrics: List[str], window_size: int = 100):
        """
        初始化指标跟踪器
        Args:
            metrics: 需要跟踪的指标名列表
            window_size: 滑动窗口大小,用于计算移动平均
        """
        self._data = defaultdict(lambda: {'total': 0., 'count': 0, 'history': []})
        self.metrics = metrics
        self.window_size = window_size
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            for metric in self.metrics:
                self._data[metric] = {
                    'total': 0.,
                    'count': 0,
                    'history': []
                }

    def update(self, metric: str, value: float, n: int = 1) -> None:
        """
        更新指标值,线程安全
        Args:
            metric: 指标名
            value: 指标值
            n: 样本数量
        """
        with self._lock:
            if metric not in self._data:
                self._data[metric] = {'total': 0., 'count': 0, 'history': []}

            self._data[metric]['total'] += value * n
            self._data[metric]['count'] += n
            self._data[metric]['history'].append(value)

            # 维护滑动窗口
            if len(self._data[metric]['history']) > self.window_size:
                self._data[metric]['history'] = self._data[metric]['history'][-self.window_size:]

    @lru_cache(maxsize=32)
    def avg(self, metric: str) -> float:
        """获取指标平均值"""
        with self._lock:
            if self._data[metric]['count'] == 0:
                return 0.
            return self._data[metric]['total'] / self._data[metric]['count']

    def moving_avg(self, metric: str) -> float:
        """获取移动平均值"""
        with self._lock:
            history = self._data[metric]['history']
            if not history:
                return 0.
            return sum(history) / len(history)

    def result(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的统计结果"""
        with self._lock:
            results = {}
            for metric in self.metrics:
                results[metric] = {
                    'avg': self.avg(metric),
                    'moving_avg': self.moving_avg(metric),
                    'current': self._data[metric]['history'][-1] if self._data[metric]['history'] else 0.
                }
            return results


class BinaryClassificationMetrics:
    """二分类评估指标,支持并发评估"""

    def __init__(self, threshold: float = 0.5, num_workers: int = 4):
        """
        初始化二分类评估器
        Args:
            threshold: 分类阈值
            num_workers: 工作线程数
        """
        self.threshold = threshold
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self.reset()

    def reset(self) -> None:
        """重置状态"""
        with self._lock:
            self.predictions = []
            self.targets = []
            self.probabilities = []

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新预测和目标值
        Args:
            preds: 预测值 (N, C) or (N,)
            targets: 目标值 (N,)
        """
        try:
            # 转移到CPU
            if preds.device != torch.device('cpu'):
                preds = preds.cpu()
            if targets.device != torch.device('cpu'):
                targets = targets.cpu()

            # 计算概率
            if preds.dim() > 1:
                probs = F.softmax(preds, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(preds)

            # 线程安全更新
            with self._lock:
                self.probabilities.extend(probs.numpy())
                self.predictions.extend((probs > self.threshold).numpy())
                self.targets.extend(targets.numpy())

        except Exception as e:
            logger.error(f"更新评估指标失败: {str(e)}")
            raise

    def compute(self) -> Dict[str, float]:
        """计算所有指标,支持并发计算"""
        try:
            predictions = np.array(self.predictions)
            targets = np.array(self.targets)
            probabilities = np.array(self.probabilities)

            # 提交评估任务
            metric_futures = {
                'basic_metrics': self._executor.submit(self._compute_basic_metrics, predictions, targets),
                'auc_metrics': self._executor.submit(self._compute_auc_metrics, targets, probabilities),
                'pr_metrics': self._executor.submit(self._compute_pr_metrics, targets, probabilities)
            }

            # 收集结果
            metrics = {}
            for name, future in metric_futures.items():
                try:
                    metrics.update(future.result(timeout=30))
                except Exception as e:
                    logger.error(f"计算{name}失败: {str(e)}")

            # 添加置信区间
            metrics.update(self._compute_confidence_intervals(metrics))

            return metrics

        except Exception as e:
            logger.error(f"计算评估指标失败: {str(e)}")
            return {'error': str(e)}

    def _compute_basic_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算基础指标"""
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        tn = np.sum((predictions == 0) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))

        eps = 1e-7  # 避免除零
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn + eps),
            'precision': tp / (tp + fp + eps),
            'recall': tp / (tp + fn + eps),
            'f1': 2 * tp / (2 * tp + fp + fn + eps),
            'specificity': tn / (tn + fp + eps)
        }
        return metrics

    def _compute_auc_metrics(self, targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """计算AUC相关指标"""
        metrics = {}
        if len(np.unique(targets)) > 1:
            metrics['auc'] = roc_auc_score(targets, probabilities)
        return metrics

    def _compute_pr_metrics(self, targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """计算PR相关指标"""
        metrics = {}
        if len(np.unique(targets)) > 1:
            metrics['pr_auc'] = average_precision_score(targets, probabilities)
        return metrics

    def _compute_confidence_intervals(self, metrics: Dict[str, float], confidence: float = 0.95) -> Dict[str, float]:
        """计算置信区间"""
        ci_metrics = {}
        n_samples = len(self.targets)

        for metric_name, value in metrics.items():
            if metric_name in ['auc', 'pr_auc']:
                continue
            # 使用Wilson score interval
            z = 1.96  # 95% confidence
            denominator = 1 + z * z / n_samples
            centre_adj = value + z * z / (2 * n_samples)
            adj_interval = z * np.sqrt(value * (1 - value) / n_samples + z * z / (4 * n_samples * n_samples))

            lower = (centre_adj - adj_interval) / denominator
            upper = (centre_adj + adj_interval) / denominator

            ci_metrics[f'{metric_name}_ci_lower'] = float(lower)
            ci_metrics[f'{metric_name}_ci_upper'] = float(upper)

        return ci_metrics


class SegmentationMetrics:
    """图像分割评估指标,支持并发评估"""

    def __init__(self, num_classes: int, ignore_index: int = 255, num_workers: int = 4):
        """
        初始化分割评估器
        Args:
            num_classes: 类别数
            ignore_index: 忽略的类别索引
            num_workers: 工作线程数
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self.reset()

    def reset(self) -> None:
        """重置状态"""
        with self._lock:
            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    @torch.no_grad()
    def update(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        """
        更新混淆矩阵
        Args:
            pred_mask: 预测掩码 (N, H, W)
            target_mask: 目标掩码 (N, H, W)
        """
        try:
            if pred_mask.device != torch.device('cpu'):
                pred_mask = pred_mask.cpu()
            if target_mask.device != torch.device('cpu'):
                target_mask = target_mask.cpu()

            pred_mask = pred_mask.numpy()
            target_mask = target_mask.numpy()

            mask = (target_mask != self.ignore_index)

            # 并发更新混淆矩阵
            futures = []
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    futures.append(
                        self._executor.submit(
                            self._update_confusion_matrix,
                            i, j, pred_mask, target_mask, mask
                        )
                    )

            # 等待所有更新完成
            for future in futures:
                future.result()

        except Exception as e:
            logger.error(f"更新混淆矩阵失败: {str(e)}")
            raise

    def _update_confusion_matrix(self, i: int, j: int, pred_mask: np.ndarray,
                                 target_mask: np.ndarray, mask: np.ndarray) -> None:
        """更新混淆矩阵的单个元素"""
        count = np.sum((pred_mask[mask] == i) & (target_mask[mask] == j))
        with self._lock:
            self.confusion_matrix[i, j] += count

    def compute(self) -> Dict[str, float]:
        """计算所有分割指标"""
        try:
            metrics = {}

            # 计算IoU
            intersection = np.diag(self.confusion_matrix)
            union = (np.sum(self.confusion_matrix, axis=1) +
                     np.sum(self.confusion_matrix, axis=0) -
                     np.diag(self.confusion_matrix))
            iou = intersection / (union + 1e-7)

            # 基础指标
            metrics['mIoU'] = float(np.nanmean(iou))
            for i in range(self.num_classes):
                metrics[f'IoU_class{i}'] = float(iou[i])

            # 像素准确率
            metrics['pixel_accuracy'] = float(
                np.sum(np.diag(self.confusion_matrix)) /
                (np.sum(self.confusion_matrix) + 1e-7)
            )

            # 类别准确率
            metrics['mean_pixel_accuracy'] = float(
                np.nanmean(
                    np.diag(self.confusion_matrix) /
                    (np.sum(self.confusion_matrix, axis=1) + 1e-7)
                )
            )

            # Dice系数
            dice = 2 * intersection / (
                    np.sum(self.confusion_matrix, axis=1) +
                    np.sum(self.confusion_matrix, axis=0) +
                    1e-7
            )
            metrics['mean_dice'] = float(np.nanmean(dice))

            # 添加置信区间
            metrics.update(self._compute_confidence_intervals(metrics))

            return metrics

        except Exception as e:
            logger.error(f"计算分割指标失败: {str(e)}")
            return {'error': str(e)}

    def _compute_confidence_intervals(self, metrics: Dict[str, float],
                                      confidence: float = 0.95) -> Dict[str, float]:
        """计算置信区间"""
        ci_metrics = {}
        total_pixels = np.sum(self.confusion_matrix)

        for metric_name, value in metrics.items():
            if 'class' in metric_name:  # 跳过类别特定的指标
                continue

            # 使用Delta方法估计标准误差
            se = np.sqrt(value * (1 - value) / total_pixels)
            z = 1.96  # 95% confidence

            ci_metrics[f'{metric_name}_ci_lower'] = float(max(0, value - z * se))
            ci_metrics[f'{metric_name}_ci_upper'] = float(min(1, value + z * se))

        return ci_metrics