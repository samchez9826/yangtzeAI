from __future__ import annotations

import math
import logging
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial
import numpy as np

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """学习率调度器配置

    Attributes:
        name: 调度器名称
        warmup_epochs: 预热周期数
        total_epochs: 总周期数
        min_lr: 最小学习率
        max_lr: 最大学习率
        base_lr: 基础学习率
        step_size: 步长
        mode: 调度模式
        gamma: 学习率衰减因子
        pct_start: 上升阶段占比
        div_factor: 学习率除数
        final_div_factor: 最终学习率除数
        cycle_momentum: 是否使用动量循环
        total_steps: 总步数
    """
    name: str
    warmup_epochs: int = 0
    total_epochs: int = 100
    min_lr: float = 1e-6
    max_lr: float = 1.0
    base_lr: float = 1e-3
    step_size: int = 1000
    mode: str = 'cos'
    gamma: float = 1.0
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    cycle_momentum: bool = True
    total_steps: Optional[int] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SchedulerConfig':
        """从字典创建配置

        Args:
            config: 配置字典

        Returns:
            配置实例
        """
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

    def validate(self) -> None:
        """验证配置参数有效性

        Raises:
            ValueError: 参数无效时抛出
        """
        if self.warmup_epochs < 0:
            raise ValueError(f"Invalid warmup epochs: {self.warmup_epochs}")
        if self.total_epochs <= 0:
            raise ValueError(f"Invalid total epochs: {self.total_epochs}")
        if not 0.0 <= self.min_lr <= self.max_lr:
            raise ValueError(f"Invalid learning rate range: [{self.min_lr}, {self.max_lr}]")
        if not 0.0 <= self.pct_start <= 1.0:
            raise ValueError(f"Invalid pct_start: {self.pct_start}")
        if self.mode not in ['cos', 'linear', 'triangular', 'triangular2', 'exp_range']:
            raise ValueError(f"Invalid mode: {self.mode}")


class LRSchedulerBase(_LRScheduler, ABC):
    """学习率调度器基类

    提供了基本的学习率调度功能,包括:
    - 学习率上下限管理
    - 状态保存和加载
    - 预热支持
    - 动量调整
    """

    def __init__(self,
                 optimizer: Optimizer,
                 last_epoch: int = -1,
                 min_lr: float = 0.0,
                 warmup_epochs: int = 0,
                 verbose: bool = False):
        """
        Args:
            optimizer: 优化器实例
            last_epoch: 上一轮epoch索引
            min_lr: 最小学习率
            warmup_epochs: 预热周期数
            verbose: 是否打印学习率变化
        """
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        # 保存初始学习率
        self.initial_lrs = []
        for group in optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self) -> List[float]:
        """获取当前学习率列表"""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            'min_lr': self.min_lr,
            'warmup_epochs': self.warmup_epochs,
            '_step_count': self._step_count,
            'verbose': self.verbose,
            'initial_lrs': self.initial_lrs
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载调度器状态

        Args:
            state_dict: 状态字典
        """
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.min_lr = state_dict['min_lr']
        self.warmup_epochs = state_dict['warmup_epochs']
        self._step_count = state_dict['_step_count']
        self.verbose = state_dict['verbose']
        self.initial_lrs = state_dict['initial_lrs']

    def _warmup_lr(self, progress: float) -> List[float]:
        """计算预热阶段的学习率

        Args:
            progress: 预热进度 [0,1]

        Returns:
            预热阶段的学习率列表
        """
        return [
            lr * progress for lr in self.initial_lrs
        ]

    def _clip_lr(self, lr: float) -> float:
        """裁剪学习率到有效范围

        Args:
            lr: 输入学习率

        Returns:
            裁剪后的学习率
        """
        return max(self.min_lr, lr)

    def print_lr(self, is_verbose: Optional[bool] = None) -> None:
        """打印当前学习率

        Args:
            is_verbose: 是否打印详细信息
        """
        if is_verbose or (is_verbose is None and self.verbose):
            lrs = self.get_last_lr()
            logger.info(f'Epoch {self.last_epoch}: learning rates = {lrs}')

    def _validate_param_groups(self) -> None:
        """验证参数组有效性"""
        for param_group in self.optimizer.param_groups:
            if 'lr' not in param_group:
                raise KeyError('param "lr" is not specified in param_groups[0]')


class WarmupLRScheduler(LRSchedulerBase):
    """带预热的学习率调度器

    特点:
    - 支持线性和指数预热
    - 预热后使用余弦退火
    - 支持最小学习率限制
    - 内存高效的实现
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 min_lr: float = 1e-6,
                 warmup_method: str = 'linear',
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Args:
            optimizer: 优化器实例
            warmup_epochs: 预热周期数
            total_epochs: 总周期数
            min_lr: 最小学习率
            warmup_method: 预热方法 ['linear', 'exp']
            last_epoch: 上一个epoch的索引
            verbose: 是否打印学习率变化
        """
        self.total_epochs = total_epochs
        self.warmup_method = warmup_method

        # 预计算一些常量
        self._warmup_factor = 1.0 / warmup_epochs if warmup_epochs > 0 else 1.0
        self._cos_scaling = 0.5 * (1 + math.cos(math.pi))

        # 验证参数
        self._validate_params()

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            verbose=verbose
        )

    def _validate_params(self) -> None:
        """验证参数有效性"""
        if self.warmup_method not in ['linear', 'exp']:
            raise ValueError(f"Unsupported warmup method: {self.warmup_method}")
        if self.total_epochs <= self.warmup_epochs:
            raise ValueError("total_epochs must be greater than warmup_epochs")

    @torch.no_grad()
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                           "please use `get_last_lr()`.")

        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            return self._get_warmup_lr()

        # 余弦退火阶段
        return self._get_cosine_lr()

    def _get_warmup_lr(self) -> List[float]:
        """计算预热阶段学习率"""
        if self.warmup_method == 'linear':
            alpha = self.last_epoch * self._warmup_factor
        else:  # exp
            alpha = math.pow(self.last_epoch / self.warmup_epochs, 2)

        return [self._clip_lr(base_lr * alpha) for base_lr in self.base_lrs]

    def _get_cosine_lr(self) -> List[float]:
        """计算余弦退火阶段学习率"""
        # 计算退火进度
        progress = float(self.last_epoch - self.warmup_epochs) / \
                   float(max(1, self.total_epochs - self.warmup_epochs))

        # 计算余弦衰减因子
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 应用到所有参数组
        return [self._clip_lr(base_lr * cos_factor) for base_lr in self.base_lrs]

    def _format_lr(self, lrs: List[float]) -> str:
        """格式化学习率输出"""
        if len(lrs) == 1:
            return f"{lrs[0]:.3e}"
        return "[" + ", ".join(f"{lr:.3e}" for lr in lrs) + "]"


class CosineAnnealingLR(LRSchedulerBase):
    """优化的余弦退火学习率调度器

    特点:
    - 支持预热
    - 支持最小学习率限制
    - 支持重启策略
    - 高效的学习率计算
    """

    def __init__(self,
                 optimizer: Optimizer,
                 T_max: int,
                 eta_min: float = 0,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 warmup_epochs: int = 0,
                 warmup_method: str = 'linear',
                 cycle_momentum: bool = True,
                 min_momentum: float = 0.85,
                 max_momentum: float = 0.95):
        """
        Args:
            optimizer: 优化器实例
            T_max: 最大周期数
            eta_min: 最小学习率
            last_epoch: 上一个epoch的索引
            verbose: 是否打印学习率变化
            warmup_epochs: 预热周期数
            warmup_method: 预热方法
            cycle_momentum: 是否调整动量
            min_momentum: 最小动量
            max_momentum: 最大动量
        """
        self.T_max = T_max
        self.cycle_momentum = cycle_momentum

        if cycle_momentum:
            self.min_momentum = min_momentum
            self.max_momentum = max_momentum

        # 预计算常量
        self._init_cos_scale()

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
            min_lr=eta_min,
            warmup_epochs=warmup_epochs,
            verbose=verbose
        )

    def _init_cos_scale(self) -> None:
        """初始化余弦缩放因子"""
        self._cos_factor = math.pi / self.T_max
        if self.cycle_momentum:
            self._momentum_diff = self.max_momentum - self.min_momentum

    @torch.no_grad()
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                           "please use `get_last_lr()`.")

        # 预热阶段
        if self.last_epoch < self.warmup_epochs:
            return self._warmup_lr(self.last_epoch / max(1, self.warmup_epochs))

        # 计算余弦退火
        effective_epoch = self.last_epoch - self.warmup_epochs
        cos_scale = 0.5 * (1 + math.cos(effective_epoch * self._cos_factor))

        # 应用到所有参数组
        lrs = [self._clip_lr(
            self.min_lr + (base_lr - self.min_lr) * cos_scale
        ) for base_lr in self.base_lrs]

        # 更新动量
        if self.cycle_momentum:
            self._adjust_momentum(cos_scale)

        return lrs

    def _adjust_momentum(self, cos_scale: float) -> None:
        """调整动量参数"""
        if not hasattr(self.optimizer, 'momentum'):
            return

        momentum = self.min_momentum + self._momentum_diff * cos_scale
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                param_group['momentum'] = momentum

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = super().state_dict()
        if self.cycle_momentum:
            state.update({
                'min_momentum': self.min_momentum,
                'max_momentum': self.max_momentum
            })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        if self.cycle_momentum:
            self.min_momentum = state_dict.pop('min_momentum')
            self.max_momentum = state_dict.pop('max_momentum')
        super().load_state_dict(state_dict)


class CyclicLRScheduler(LRSchedulerBase):
    """高效的循环学习率调度器

    特点:
    - 支持多种循环模式
    - 动态动量调整
    - 内存高效的缓存策略
    - 支持自定义缩放函数
    """

    def __init__(self,
                 optimizer: Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size: int,
                 mode: str = 'triangular',
                 gamma: float = 1.0,
                 scale_mode: str = 'cycle',
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.8,
                 max_momentum: float = 0.9,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Args:
            optimizer: 优化器实例
            base_lr: 基础学习率
            max_lr: 最大学习率
            step_size: 步长(半个周期的步数)
            mode: 学习率变化模式 ['triangular', 'triangular2', 'exp_range']
            gamma: exp_range模式的衰减率
            scale_mode: 缩放模式 ['cycle', 'iterations']
            cycle_momentum: 是否调整动量
            base_momentum: 基础动量
            max_momentum: 最大动量
            last_epoch: 上一个epoch索引
            verbose: 是否打印学习率变化
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum

        if cycle_momentum:
            self.base_momentum = base_momentum
            self.max_momentum = max_momentum

        # 验证参数
        self._validate_params()

        # 预计算常量
        self._step_ratio = float(step_size)
        self._lr_diff = float(max_lr - base_lr)

        if cycle_momentum:
            self._momentum_diff = max_momentum - base_momentum

        # 初始化缓存
        self._last_scale_mode_cycle = 0
        self._scale_fn = self._get_scale_fn()

        super().__init__(optimizer, last_epoch, verbose=verbose)

    def _validate_params(self) -> None:
        """验证参数有效性"""
        if self.mode not in ['triangular', 'triangular2', 'exp_range']:
            raise ValueError(f"mode must be one of ['triangular', 'triangular2', 'exp_range'], got {self.mode}")
        if self.scale_mode not in ['cycle', 'iterations']:
            raise ValueError(f"scale_mode must be one of ['cycle', 'iterations'], got {self.scale_mode}")
        if self.base_lr >= self.max_lr:
            raise ValueError(f"base_lr must be less than max_lr, got {self.base_lr} >= {self.max_lr}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")

    def _get_scale_fn(self) -> Callable[[int], float]:
        """获取缩放函数"""
        if self.mode == 'triangular':
            return lambda _: 1.0
        elif self.mode == 'triangular2':
            return lambda cycle: 1.0 / (2.0 ** cycle)
        elif self.mode == 'exp_range':
            return lambda cycle: self.gamma ** cycle
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @torch.no_grad()
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                           "please use `get_last_lr()`.")

        # 计算周期和位置
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        # 计算缩放因子
        scale_factor = self._scale_fn(cycle if self.scale_mode == 'cycle' else self.last_epoch)

        # 计算学习率
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr + (self.max_lr - base_lr) * max(0, (1 - x)) * scale_factor
            lrs.append(self._clip_lr(lr))

        # 调整动量
        if self.cycle_momentum:
            self._adjust_momentum(x)

        return lrs

    def _adjust_momentum(self, x: float) -> None:
        """调整动量参数

        动量与学习率反向变化,学习率高时动量低
        """
        if not hasattr(self.optimizer, 'momentum'):
            return

        momentum = self.max_momentum - self._momentum_diff * max(0, (1 - x))
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                param_group['momentum'] = momentum


class OneCycleLRScheduler(LRSchedulerBase):
    """高效的 One Cycle 学习率调度器

    特点:
    - 三阶段学习率调整
    - 动态动量调整
    - 支持多种退火策略
    - 内存高效实现
    """

    def __init__(self,
                 optimizer: Optimizer,
                 max_lr: float,
                 total_steps: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.85,
                 max_momentum: float = 0.95,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Args:
            optimizer: 优化器实例
            max_lr: 最大学习率
            total_steps: 总步数
            pct_start: 上升阶段占比
            anneal_strategy: 退火策略 ['cos', 'linear']
            cycle_momentum: 是否调整动量
            base_momentum: 基础动量
            max_momentum: 最大动量
            div_factor: 初始学习率除数
            final_div_factor: 最终学习率除数
            last_epoch: 上一个epoch索引
            verbose: 是否打印学习率变化
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # 验证参数
        self._validate_params()

        # 计算阶段步数
        self._step_size_up = float(pct_start * total_steps) - 1
        self._step_size_down = float(total_steps - self._step_size_up) - 1

        # 预计算常量
        self._init_lr = float(max_lr) / self.div_factor
        self._final_lr = self._init_lr / self.final_div_factor

        if cycle_momentum:
            self._momentum_diff = max_momentum - base_momentum

        super().__init__(optimizer, last_epoch, verbose=verbose)

    def _validate_params(self) -> None:
        """验证参数有效性"""
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if not 0 < self.pct_start < 1:
            raise ValueError(f"pct_start must be between 0 and 1, got {self.pct_start}")
        if self.anneal_strategy not in ['cos', 'linear']:
            raise ValueError(f"anneal_strategy must be one of ['cos', 'linear'], got {self.anneal_strategy}")

    def _annealing_cos(self, start: float, end: float, pct: float) -> float:
        """余弦退火"""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start: float, end: float, pct: float) -> float:
        """线性退火"""
        return end + (start - end) * (1 - pct)

    @torch.no_grad()
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                           "please use `get_last_lr()`.")

        # 超出总步数
        if self.last_epoch >= self.total_steps:
            return [self._final_lr for _ in self.base_lrs]

        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch <= self._step_size_up:
                # 上升阶段
                lr = self._annealing_func(
                    self._init_lr,
                    self.max_lr,
                    self.last_epoch / self._step_size_up
                )
            else:
                # 下降阶段
                down_step = self.last_epoch - self._step_size_up
                lr = self._annealing_func(
                    self.max_lr,
                    self._final_lr,
                    down_step / self._step_size_down
                )
            lrs.append(self._clip_lr(lr))

        # 调整动量
        if self.cycle_momentum:
            self._adjust_momentum()

        return lrs

    def _adjust_momentum(self) -> None:
        """调整动量参数"""
        if not hasattr(self.optimizer, 'momentum'):
            return

        if self.last_epoch <= self._step_size_up:
            # 上升阶段动量下降
            momentum = self._annealing_func(
                self.max_momentum,
                self.base_momentum,
                self.last_epoch / self._step_size_up
            )
        else:
            # 下降阶段动量上升
            down_step = self.last_epoch - self._step_size_up
            momentum = self._annealing_func(
                self.base_momentum,
                self.max_momentum,
                down_step / self._step_size_down
            )

        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                param_group['momentum'] = momentum

    @property
    def _annealing_func(self) -> Callable[[float, float, float], float]:
        """获取退火函数"""
        return self._annealing_cos if self.anneal_strategy == 'cos' else self._annealing_linear