from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from threading import Lock
import logging
from dataclasses import dataclass, field

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """指标数据存储"""
    total: float = 0.0
    count: int = 0
    history: List[float] = field(default_factory=list)
    avg_window: int = 100  # 移动平均窗口大小

    def update(self, value: float, n: int = 1) -> None:
        """更新指标值"""
        self.total += value * n
        self.count += n
        self.history.append(value)

        # 维持固定大小的历史记录
        if len(self.history) > self.avg_window:
            self.history = self.history[-self.avg_window:]

    def reset(self) -> None:
        """重置指标"""
        self.total = 0.0
        self.count = 0
        self.history.clear()

    @property
    def avg(self) -> float:
        """计算平均值"""
        return self.total / max(1, self.count)

    @property
    def moving_avg(self) -> float:
        """计算移动平均"""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    @property
    def current(self) -> float:
        """获取最新值"""
        return self.history[-1] if self.history else 0.0


class MetricTracker:
    """线程安全的指标跟踪器

    功能:
    - 支持多种指标类型
    - 线程安全的更新
    - 移动平均计算
    - 指标历史记录
    - 内存优化的存储
    """

    def __init__(self, metric_names: List[str], avg_window: int = 100):
        """
        初始化跟踪器

        Args:
            metric_names: 需要跟踪的指标名列表
            avg_window: 移动平均窗口大小
        """
        self._data = {}
        self._lock = Lock()
        self.avg_window = avg_window

        # 初始化指标
        for name in metric_names:
            self._data[name] = MetricData(avg_window=avg_window)

    def update(self, name: str, value: float, n: int = 1) -> None:
        """
        更新指标值

        Args:
            name: 指标名
            value: 指标值
            n: 样本数量
        """
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Invalid value for metric {name}: {value}")
            return

        with self._lock:
            if name not in self._data:
                self._data[name] = MetricData(avg_window=self.avg_window)
            self._data[name].update(value, n)

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            for metric in self._data.values():
                metric.reset()

    def avg(self, name: str) -> float:
        """
        获取指标平均值

        Args:
            name: 指标名

        Returns:
            指标平均值
        """
        with self._lock:
            if name not in self._data:
                logger.warning(f"Metric {name} not found")
                return 0.0
            return self._data[name].avg

    def moving_avg(self, name: str) -> float:
        """
        获取指标移动平均值

        Args:
            name: 指标名

        Returns:
            移动平均值
        """
        with self._lock:
            if name not in self._data:
                logger.warning(f"Metric {name} not found")
                return 0.0
            return self._data[name].moving_avg

    def result(self) -> Dict[str, float]:
        """
        获取所有指标的结果

        Returns:
            指标名到平均值的映射
        """
        with self._lock:
            return {name: data.avg for name, data in self._data.items()}

    def current(self) -> Dict[str, float]:
        """
        获取所有指标的当前值

        Returns:
            指标名到当前值的映射
        """
        with self._lock:
            return {name: data.current for name, data in self._data.items()}

    def history(self, name: str) -> List[float]:
        """
        获取指标的历史记录

        Args:
            name: 指标名

        Returns:
            历史值列表
        """
        with self._lock:
            if name not in self._data:
                logger.warning(f"Metric {name} not found")
                return []
            return self._data[name].history.copy()

    def add_metric(self, name: str) -> None:
        """
        添加新的指标

        Args:
            name: 指标名
        """
        with self._lock:
            if name not in self._data:
                self._data[name] = MetricData(avg_window=self.avg_window)

    def remove_metric(self, name: str) -> None:
        """
        移除指标

        Args:
            name: 指标名
        """
        with self._lock:
            self._data.pop(name, None)

    def __str__(self) -> str:
        """获取可读的字符串表示"""
        results = self.result()
        return " - ".join(f"{k}: {v:.4f}" for k, v in results.items())