import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from typing import List, Dict


class WarmupLRScheduler(_LRScheduler):
    """带预热的学习率调度器"""

    def __init__(self, optimizer: Optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            alpha = float(self.last_epoch) / float(self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = float(self.last_epoch - self.warmup_epochs) / \
                       float(max(1, self.total_epochs - self.warmup_epochs))
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


class CyclicLRScheduler(_LRScheduler):
    """周期性学习率调度器"""

    def __init__(self, optimizer: Optimizer, base_lr: float,
                 max_lr: float, step_size: int, mode: str = 'triangular',
                 gamma: float = 1.0, last_epoch: int = -1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        if self.mode == 'triangular':
            factor = 1.0
        elif self.mode == 'triangular2':
            factor = 1.0 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            factor = self.gamma ** self.last_epoch

        return [self.base_lr + (self.max_lr - self.base_lr) *
                max(0, (1 - x)) * factor for _ in self.base_lrs]


class OneCycleLRScheduler(_LRScheduler):
    """One Cycle学习率调度器"""

    def __init__(self, optimizer: Optimizer, max_lr: float,
                 total_steps: int, pct_start: float = 0.3,
                 anneal_strategy: str = 'cos', div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch >= self.total_steps:
            return [self.max_lr / self.final_div_factor for _ in self.base_lrs]

        init_lr = self.max_lr / self.div_factor
        final_lr = self.max_lr / self.final_div_factor

        # 计算当前阶段
        if self.last_epoch <= self.total_steps * self.pct_start:
            # 上升阶段
            pct = self.last_epoch / (self.total_steps * self.pct_start)
            scale_factor = self._annealing_func(pct)
            return [(self.max_lr - init_lr) * scale_factor + init_lr
                    for _ in self.base_lrs]
        else:
            # 下降阶段
            pct = (self.last_epoch - self.total_steps * self.pct_start) / \
                  (self.total_steps * (1 - self.pct_start))
            scale_factor = self._annealing_func(pct)
            return [(self.max_lr - final_lr) * (1 - scale_factor) + final_lr
                    for _ in self.base_lrs]

    def _annealing_func(self, x: float) -> float:
        if self.anneal_strategy == 'cos':
            return 0.5 * (1 + math.cos(math.pi * x + math.pi))
        elif self.anneal_strategy == 'linear':
            return 1 - x
        else:
            raise ValueError(f"Unknown anneal strategy: {self.anneal_strategy}")


def create_scheduler(optimizer: Optimizer, config: Dict) -> _LRScheduler:
    """创建学习率调度器"""
    name = config['name'].lower()

    if name == 'warmup':
        return WarmupLRScheduler(
            optimizer,
            warmup_epochs=config['warmup_epochs'],
            total_epochs=config['total_epochs'],
            min_lr=config.get('min_lr', 1e-6)
        )
    elif name == 'cyclic':
        return CyclicLRScheduler(
            optimizer,
            base_lr=config['base_lr'],
            max_lr=config['max_lr'],
            step_size=config['step_size'],
            mode=config.get('mode', 'triangular'),
            gamma=config.get('gamma', 1.0)
        )
    elif name == 'one_cycle':
        return OneCycleLRScheduler(
            optimizer,
            max_lr=config['max_lr'],
            total_steps=config['total_steps'],
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 1e4)
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")