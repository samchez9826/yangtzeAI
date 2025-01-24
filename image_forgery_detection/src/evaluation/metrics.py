import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import logging
from collections import defaultdict


class MetricTracker:
    """指标跟踪器"""

    def __init__(self, metrics: List[str]):
        """
        初始化指标跟踪器
        Args:
            metrics: 需要跟踪的指标名列表
        """
        self._data = defaultdict(lambda: {'total': 0., 'count': 0})
        self.metrics = metrics
        self.reset()

    def reset(self):
        """重置所有指标"""
        for metric in self.metrics:
            self._data[metric] = {'total': 0., 'count': 0}

    def update(self, metric: str, value: float, n: int = 1):
        """
        更新指标值
        Args:
            metric: 指标名
            value: 指标值
            n: 样本数量
        """
        if metric not in self._data:
            self._data[metric] = {'total': 0., 'count': 0}
        self._data[metric]['total'] += value * n
        self._data[metric]['count'] += n

    def avg(self, metric: str) -> float:
        """获取指标平均值"""
        if self._data[metric]['count'] == 0:
            return 0.
        return self._data[metric]['total'] / self._data[metric]['count']

    def result(self) -> Dict[str, float]:
        """获取所有指标的平均值"""
        return {metric: self.avg(metric) for metric in self.metrics}


class BinaryClassificationMetrics:
    """二分类评估指标"""

    def __init__(self, threshold: float = 0.5):
        """
        初始化二分类评估器
        Args:
            threshold: 分类阈值
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """重置状态"""
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        更新预测和目标值
        Args:
            preds: 预测值 (N, C) or (N,)
            targets: 目标值 (N,)
        """
        if preds.dim() > 1:
            probs = F.softmax(preds, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(preds)

        self.probabilities.extend(probs.cpu().numpy())
        self.predictions.extend((probs > self.threshold).cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)

        # 基础指标
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        tn = np.sum((predictions == 0) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))

        # 计算各项指标
        metrics = {}

        # 准确率
        metrics['accuracy'] = (tp + tn) / len(targets)

        # 精确率
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0

        # 召回率
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1分数
        metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # ROC-AUC
        if len(np.unique(targets)) > 1:
            metrics['auc'] = roc_auc_score(targets, probabilities)

        # PR-AUC
        metrics['pr_auc'] = average_precision_score(targets, probabilities)

        # 特异度
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics


class SegmentationMetrics:
    """图像分割评估指标"""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        """
        初始化分割评估器
        Args:
            num_classes: 类别数
            ignore_index: 忽略的类别索引
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """重置状态"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        更新混淆矩阵
        Args:
            pred_mask: 预测掩码 (N, H, W)
            target_mask: 目标掩码 (N, H, W)
        """
        pred_mask = pred_mask.cpu().numpy()
        target_mask = target_mask.cpu().numpy()

        mask = (target_mask != self.ignore_index)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.sum(
                    (pred_mask[mask] == i) & (target_mask[mask] == j)
                )

    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}

        # 计算每个类别的IoU
        iou = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix)
        )

        # 平均IoU
        metrics['mIoU'] = np.nanmean(iou)

        # 每个类别的IoU
        for i in range(self.num_classes):
            metrics[f'IoU_class{i}'] = iou[i]

        # 像素准确率
        metrics['pixel_accuracy'] = np.sum(np.diag(self.confusion_matrix)) / (
                np.sum(self.confusion_matrix) + 1e-10
        )

        # 平均像素准确率
        metrics['mean_pixel_accuracy'] = np.nanmean(
            np.diag(self.confusion_matrix) /
            (np.sum(self.confusion_matrix, axis=1) + 1e-10)
        )

        # Dice系数
        dice = 2 * np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) + 1e-10
        )
        metrics['mean_dice'] = np.nanmean(dice)

        return metrics


class DetectionMetrics:
    """篡改检测评估指标"""

    def __init__(self, iou_threshold: float = 0.5):
        """
        初始化检测评估器
        Args:
            iou_threshold: IoU阈值
        """
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """重置状态"""
        self.image_predictions = []  # 图像级预测
        self.image_targets = []  # 图像级标签
        self.mask_predictions = []  # 像素级预测
        self.mask_targets = []  # 像素级标签

    def update(self, pred: Dict[str, torch.Tensor],
               target: Dict[str, torch.Tensor]):
        """
        更新预测和目标
        Args:
            pred: 预测字典,包含'cls'和'seg'
            target: 目标字典,包含'cls_target'和'seg_target'
        """
        # 图像级预测
        if 'cls' in pred:
            self.image_predictions.extend(
                F.softmax(pred['cls'], dim=1)[:, 1].cpu().numpy()
            )
            self.image_targets.extend(
                target['cls_target'].cpu().numpy()
            )

        # 像素级预测
        if 'seg' in pred:
            self.mask_predictions.extend(
                pred['seg'].cpu().numpy()
            )
            self.mask_targets.extend(
                target['seg_target'].cpu().numpy()
            )

    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}

        # 图像级指标
        if self.image_predictions:
            binary_metrics = BinaryClassificationMetrics()
            for pred, target in zip(self.image_predictions, self.image_targets):
                binary_metrics.update(
                    torch.tensor([pred]),
                    torch.tensor([target])
                )
            metrics.update(binary_metrics.compute())

        # 像素级指标
        if self.mask_predictions:
            mask_metrics = SegmentationMetrics(num_classes=2)
            for pred, target in zip(self.mask_predictions, self.mask_targets):
                mask_metrics.update(
                    torch.tensor(pred > 0.5),
                    torch.tensor(target)
                )
            metrics.update(mask_metrics.compute())

        return metrics


class QualityMetrics:
    """图像质量评估指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态"""
        self.sharpness_scores = []
        self.noise_scores = []
        self.compression_scores = []
        self.contrast_scores = []

    def update(self, image: np.ndarray):
        """
        更新图像质量指标
        Args:
            image: BGR图像
        """
        # 锐度评估
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.sharpness_scores.append(sharpness)

        # 噪声评估
        noise = self._estimate_noise(gray)
        self.noise_scores.append(noise)

        # 压缩评估
        compression = self._estimate_compression(gray)
        self.compression_scores.append(compression)

        # 对比度评估
        contrast = self._estimate_contrast(gray)
        self.contrast_scores.append(contrast)

    def compute(self) -> Dict[str, float]:
        """计算所有质量指标"""
        return {
            'sharpness': np.mean(self.sharpness_scores),
            'noise_level': np.mean(self.noise_scores),
            'compression_quality': np.mean(self.compression_scores),
            'contrast': np.mean(self.contrast_scores)
        }

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """估计图像噪声水平"""
        noise_sigma = cv2.fastNlMeansDenoising(gray)
        return np.std(gray - noise_sigma)

    def _estimate_compression(self, gray: np.ndarray) -> float:
        """估计压缩质量"""
        # 计算DCT系数
        dct = cv2.dct(gray.astype(np.float32))
        # 分析高频系数
        return np.mean(np.abs(dct[5:, 5:]))

    def _estimate_contrast(self, gray: np.ndarray) -> float:
        """估计图像对比度"""
        return (np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-6)