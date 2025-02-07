from __future__ import annotations

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, List, Optional, Union, Tuple, Any, Type, Callable
from collections import defaultdict
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """优化器配置"""
    name: str
    lr: float
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    layer_lrs: Optional[Dict[str, float]] = None
    lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'OptimizerConfig':
        """从字典创建配置"""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

    def validate(self) -> None:
        """验证配置参数"""
        if not 0.0 <= self.lr:
            raise ValueError(f"Invalid learning rate: {self.lr}")
        if not 0.0 <= self.eps:
            raise ValueError(f"Invalid epsilon value: {self.eps}")
        if not 0.0 <= self.betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {self.betas[0]}")
        if not 0.0 <= self.betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {self.betas[1]}")
        if self.lookahead:
            if not 1 <= self.lookahead_k:
                raise ValueError(f"Invalid lookahead steps: {self.lookahead_k}")
            if not 0.0 <= self.lookahead_alpha <= 1.0:
                raise ValueError(f"Invalid lookahead alpha: {self.lookahead_alpha}")


class OptimizerBase(Optimizer, ABC):
    """优化器基类"""

    def __init__(self, params, defaults: Dict[str, Any]):
        super().__init__(params, defaults)
        self._step_count: int = 0
        self._initialized: bool = False

    @abstractmethod
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行优化步骤"""
        pass

    def zero_grad(self, set_to_none: bool = True) -> None:
        """高效地清空梯度

        Args:
            set_to_none: 如果为True,则将梯度设置为None而不是0
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state_dict = super().state_dict()
        state_dict.update({
            'step_count': self._step_count,
            'initialized': self._initialized
        })
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self._step_count = state_dict.pop('step_count', 0)
        self._initialized = state_dict.pop('initialized', False)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def _init_state(self, p: torch.Tensor) -> None:
        """初始化参数状态"""
        state = self.state[p]
        if len(state) != 0:  # 已经初始化
            return

        state['step'] = 0
        for key, init_value in self.defaults.get('init_values', {}).items():
            if callable(init_value):
                state[key] = init_value(p)
            else:
                state[key] = torch.clone(init_value).detach()

    @staticmethod
    def _true_device(p: torch.Tensor) -> torch.device:
        """获取参数的真实设备"""
        if p.device.type == 'cuda':
            return torch.device(f'cuda:{p.get_device()}')
        return p.device

    def _handle_sparse_grad(self, grad: torch.Tensor) -> bool:
        """处理稀疏梯度"""
        return grad.is_sparse

    def _clip_grad_norm(self, parameters, max_norm: float, norm_type: float = 2.0) -> None:
        """梯度裁剪"""
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    def _memory_efficient_update(self, p: torch.Tensor,
                                 grad: torch.Tensor,
                                 state: Dict[str, Any]) -> None:
        """内存高效的参数更新"""
        with torch.cuda.amp.autocast():
            if p.grad is None:
                return

            if grad.is_sparse:
                raise RuntimeError('Sparse gradients are not supported')

            grad = grad.data
            if grad.is_sparse:
                raise RuntimeError('Sparse gradients are not supported')

            state['step'] += 1

            # 具体的更新逻辑在子类中实现
            self._update_param(p, grad, state)

    @abstractmethod
    def _update_param(self, p: torch.Tensor,
                      grad: torch.Tensor,
                      state: Dict[str, Any]) -> None:
        """参数更新的具体实现"""
        pass


class RAdam(OptimizerBase):
    """内存优化的 Rectified Adam 实现

    Args:
        params: 迭代优化的参数
        lr: 学习率
        betas: 用于计算梯度和其平方的移动平均系数
        eps: 添加到分母以提高数值稳定性的值
        weight_decay: 权重衰减系数
        degenerated_to_sgd: 当修正项小于阈值时是否退化为SGD
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0,
                 degenerated_to_sgd: bool = True):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        degenerated_to_sgd=degenerated_to_sgd,
                        init_values={
                            'exp_avg': lambda p: torch.zeros_like(p, memory_format=torch.preserve_format),
                            'exp_avg_sq': lambda p: torch.zeros_like(p, memory_format=torch.preserve_format)
                        })

        super().__init__(params, defaults)

        # 预计算缓冲区
        self.buffer = [[None, None, None] for _ in range(10)]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行单步优化

        Args:
            closure: 重新评估模型并返回loss的闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]
                if not state:
                    self._init_state(p)

                self._memory_efficient_update(p, grad, state)

        return loss

    def _update_param(self, p: torch.Tensor,
                      grad: torch.Tensor,
                      state: Dict[str, Any]) -> None:
        """实现参数更新逻辑"""
        group = self.param_groups[0]
        beta1, beta2 = group['betas']

        # 获取状态变量
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        # 应用权重衰减
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])

        # 更新移动平均
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # 计算偏差修正
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # 获取步长
        buffered = self.buffer[int(state['step'] % 10)]
        if state['step'] == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = state['step']
            beta2_t = beta2 ** state['step']
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma

            if N_sma >= 5:
                step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                                      (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (
                                    1 - beta1 ** state['step'])
            elif group['degenerated_to_sgd']:
                step_size = 1.0 / (1 - beta1 ** state['step'])
            else:
                step_size = -1
            buffered[2] = step_size

        if N_sma >= 5:
            denom = exp_avg_sq.sqrt().add_(group['eps'])
            p.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
        elif step_size > 0:
            p.add_(exp_avg, alpha=-step_size * group['lr'])


class AdaBelief(OptimizerBase):
    """内存优化的 AdaBelief 实现

    Args:
        params: 迭代优化的参数
        lr: 学习率
        betas: 用于计算梯度和其平方的移动平均系数
        eps: 添加到分母以提高数值稳定性的值
        weight_decay: 权重衰减系数
        amsgrad: 是否使用 AMSGrad 变体
        weight_decouple: 是否解耦权重衰减
        fixed_decay: 使用固定的权重衰减
        rectify: 是否使用修正项
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0,
                 amsgrad: bool = False, weight_decouple: bool = True,
                 fixed_decay: bool = False, rectify: bool = True):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        weight_decouple=weight_decouple,
                        fixed_decay=fixed_decay, rectify=rectify,
                        init_values={
                            'exp_avg': lambda p: torch.zeros_like(p, memory_format=torch.preserve_format),
                            'exp_avg_sq': lambda p: torch.zeros_like(p, memory_format=torch.preserve_format),
                            'max_exp_avg_sq': lambda p: torch.zeros_like(p, memory_format=torch.preserve_format)
                            if amsgrad else None
                        })

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]
                if not state:
                    self._init_state(p)

                self._memory_efficient_update(p, grad, state)

        return loss

    def _update_param(self, p: torch.Tensor,
                      grad: torch.Tensor,
                      state: Dict[str, Any]) -> None:
        """实现参数更新逻辑"""
        group = self.param_groups[0]
        beta1, beta2 = group['betas']

        # 获取状态变量
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if group['amsgrad']:
            max_exp_avg_sq = state['max_exp_avg_sq']

        # 更新偏置修正
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # 权重衰减
        if group['weight_decouple']:
            if group['weight_decay'] != 0:
                if group['fixed_decay']:
                    p.data.mul_(1 - group['weight_decay'])
                else:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

        # 计算梯度估计误差
        grad_residual = grad - exp_avg
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # 更新方差估计
        exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

        if group['amsgrad']:
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        # 更新参数
        step_size = group['lr'] / bias_correction1
        p.data.addcdiv_(exp_avg, denom, value=-step_size)


class Lookahead(OptimizerBase):
    """内存优化的 Lookahead 优化器包装器

    Args:
        optimizer: 基础优化器
        k: 快速权重更新次数
        alpha: 慢速权重更新步长
    """

    def __init__(self, base_optimizer: Optimizer,
                 k: int = 5, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        # 初始化基础优化器
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.k = k
        self.alpha = alpha
        self.state = defaultdict(dict)

        # 保存快速权重状态
        self.fast_state = self.optimizer.state

        # 初始化计数器
        for group in self.optimizer.param_groups:
            group['counter'] = 0

    @torch.no_grad()
    def update_slow_weights(self, group: Dict[str, Any]) -> None:
        """更新慢速权重

        Args:
            group: 参数组
        """
        for fast_p in group['params']:
            if fast_p.grad is None:
                continue

            param_state = self.state[fast_p]
            if 'slow_param' not in param_state:
                # 初始化慢速权重
                param_state['slow_param'] = torch.clone(fast_p.data).detach()
            else:
                # 更新慢速权重
                slow_p = param_state['slow_param']
                slow_p.add_(fast_p.data - slow_p, alpha=self.alpha)
            # 将慢速权重拷贝到快速权重
            fast_p.data.copy_(param_state['slow_param'])

    @torch.no_grad()
    def sync_lookahead(self) -> None:
        """同步所有组的Lookahead状态"""
        for group in self.param_groups:
            self.update_slow_weights(group)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行优化步骤"""
        # 更新快速权重
        loss = self.optimizer.step(closure)

        # 更新慢速权重
        for group in self.optimizer.param_groups:
            group['counter'] += 1
            if group['counter'] >= self.k:
                self.update_slow_weights(group)
                group['counter'] = 0

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super().load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state


def _build_param_groups(model: torch.nn.Module,
                        config: OptimizerConfig) -> List[Dict[str, Any]]:
    """构建参数组

    Args:
        model: 模型
        config: 优化器配置

    Returns:
        参数组列表
    """
    if not config.layer_lrs:
        # 所有参数使用相同学习率
        return [{'params': model.parameters(), 'lr': config.lr}]

    param_groups = []
    remaining_params = set(model.parameters())

    # 为指定层设置不同学习率
    for layer_name, lr in config.layer_lrs.items():
        layer_params = [p for n, p in model.named_parameters()
                        if layer_name in n]
        if layer_params:
            param_groups.append({
                'params': layer_params,
                'lr': lr
            })
            remaining_params.difference_update(set(layer_params))

    # 剩余参数使用基础学习率
    if remaining_params:
        param_groups.append({
            'params': list(remaining_params),
            'lr': config.lr
        })

    return param_groups


def _create_base_optimizer(param_groups: List[Dict[str, Any]],
                           config: OptimizerConfig) -> Optimizer:
    """创建基础优化器

    Args:
        param_groups: 参数组列表
        config: 优化器配置

    Returns:
        优化器实例

    Raises:
        ValueError: 不支持的优化器类型
    """
    name = config.name.lower()

    if name == 'adam':
        return optim.Adam(
            param_groups,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    elif name == 'radam':
        return RAdam(
            param_groups,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    elif name == 'adamw':
        return optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    elif name == 'adabelief':
        return AdaBelief(
            param_groups,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {name}")


def build_optimizer(model: torch.nn.Module,
                    config: Union[Dict[str, Any], OptimizerConfig]) -> Optimizer:
    """构建优化器的工厂函数

    Args:
        model: 需要优化的模型
        config: 优化器配置

    Returns:
        配置好的优化器实例
    """
    try:
        # 转换配置
        if isinstance(config, dict):
            config = OptimizerConfig.from_dict(config)

        # 验证配置
        config.validate()

        # 构建参数组
        param_groups = _build_param_groups(model, config)

        # 创建基础优化器
        optimizer = _create_base_optimizer(param_groups, config)

        # 添加Lookahead包装
        if config.lookahead:
            optimizer = Lookahead(
                optimizer,
                k=config.lookahead_k,
                alpha=config.lookahead_alpha
            )

        return optimizer

    except Exception as e:
        logger.error(f"Failed to build optimizer: {str(e)}")
        raise


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    name: str
    T_max: Optional[int] = None  # cosine调度器的最大周期
    eta_min: float = 0  # 最小学习率
    step_size: Optional[int] = None  # step调度器的步长
    gamma: Optional[float] = None  # 学习率衰减因子
    milestones: Optional[List[int]] = None  # multistep调度器的里程碑
    mode: str = 'min'  # plateau调度器的模式
    factor: float = 0.1  # plateau调度器的衰减因子
    patience: int = 10  # plateau调度器的耐心值
    threshold: float = 1e-4  # plateau调度器的阈值
    min_lr: float = 0  # 最小学习率
    eps: float = 1e-8  # 数值稳定性
    verbose: bool = False

    # 预热相关配置
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-6
    warmup_method: str = 'linear'

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SchedulerConfig':
        """从字典创建配置"""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

    def validate(self) -> None:
        """验证配置参数"""
        if self.name not in ['cosine', 'step', 'multistep', 'plateau', 'warmup']:
            raise ValueError(f"Invalid scheduler type: {self.name}")

        if self.name == 'cosine' and self.T_max is None:
            raise ValueError("T_max must be specified for cosine scheduler")

        if self.name == 'step' and (self.step_size is None or self.gamma is None):
            raise ValueError("step_size and gamma must be specified for step scheduler")

        if self.name == 'multistep' and (not self.milestones or self.gamma is None):
            raise ValueError("milestones and gamma must be specified for multistep scheduler")


class SchedulerBase(_LRScheduler, ABC):
    """学习率调度器基类"""

    def __init__(self, optimizer: Optimizer,
                 warmup_epochs: int = 0,
                 warmup_start_lr: float = 1e-6,
                 warmup_method: str = 'linear',
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_method = warmup_method
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """获取学习率"""
        if self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lr()
        return self._get_scheduled_lr()

    @abstractmethod
    def _get_scheduled_lr(self) -> List[float]:
        """获取调度后的学习率"""
        pass

    def _get_warmup_lr(self) -> List[float]:
        """获取预热阶段的学习率"""
        if self.warmup_method == 'linear':
            alpha = self.last_epoch / self.warmup_epochs
            factor = alpha
        elif self.warmup_method == 'exp':
            alpha = self.last_epoch / self.warmup_epochs
            factor = alpha * alpha
        else:
            raise ValueError(f"Invalid warmup method: {self.warmup_method}")

        return [self.warmup_start_lr + (lr - self.warmup_start_lr) * factor
                for lr in self.initial_lrs]


class CosineAnnealingScheduler(SchedulerBase):
    """Cosine退火学习率调度器"""

    def __init__(self, optimizer: Optimizer,
                 T_max: int,
                 eta_min: float = 0,
                 **kwargs):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, **kwargs)

    def _get_scheduled_lr(self) -> List[float]:
        """计算Cosine退火学习率"""
        effective_epoch = self.last_epoch - self.warmup_epochs
        return [self.eta_min + (lr - self.eta_min) *
                (1 + math.cos(math.pi * effective_epoch / self.T_max)) / 2
                for lr in self.initial_lrs]


class StepScheduler(SchedulerBase):
    """步进式学习率调度器"""

    def __init__(self, optimizer: Optimizer,
                 step_size: int,
                 gamma: float = 0.1,
                 **kwargs):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    def _get_scheduled_lr(self) -> List[float]:
        """计算步进式学习率"""
        effective_epoch = self.last_epoch - self.warmup_epochs
        return [lr * self.gamma ** (effective_epoch // self.step_size)
                for lr in self.initial_lrs]


class MultiStepScheduler(SchedulerBase):
    """多步进式学习率调度器"""

    def __init__(self, optimizer: Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 **kwargs):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, **kwargs)

    def _get_scheduled_lr(self) -> List[float]:
        """计算多步进式学习率"""
        effective_epoch = self.last_epoch - self.warmup_epochs
        return [lr * self.gamma ** len([m for m in self.milestones if m <= effective_epoch])
                for lr in self.initial_lrs]


class PlateauScheduler(SchedulerBase):
    """自适应学习率调度器"""

    def __init__(self, optimizer: Optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 min_lr: float = 0,
                 eps: float = 1e-8,
                 **kwargs):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.eps = eps
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        super().__init__(optimizer, **kwargs)

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        """根据指标更新学习率"""
        current = float(metrics)

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch: int) -> None:
        """降低学习率"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr

            if self.verbose:
                logger.info(f'Epoch {epoch}: reducing learning rate '
                            f'of group {i} to {new_lr:.4e}.')

    def is_better(self, current: float, best: float) -> bool:
        """判断当前指标是否更好"""
        if self.mode == 'min':
            return current < best - self.threshold
        return current > best + self.threshold


def build_scheduler(optimizer: Optimizer,
                    config: Union[Dict[str, Any], SchedulerConfig]) -> _LRScheduler:
    """构建学习率调度器的工厂函数

    Args:
        optimizer: 优化器
        config: 调度器配置

    Returns:
        配置好的学习率调度器
    """
    try:
        # 转换配置
        if isinstance(config, dict):
            config = SchedulerConfig.from_dict(config)

        # 验证配置
        config.validate()

        # 通用参数
        scheduler_kwargs = {
            'optimizer': optimizer,
            'warmup_epochs': config.warmup_epochs,
            'warmup_start_lr': config.warmup_start_lr,
            'warmup_method': config.warmup_method,
            'verbose': config.verbose
        }

        # 构建调度器
        if config.name == 'cosine':
            return CosineAnnealingScheduler(
                T_max=config.T_max,
                eta_min=config.eta_min,
                **scheduler_kwargs
            )
        elif config.name == 'step':
            return StepScheduler(
                step_size=config.step_size,
                gamma=config.gamma,
                **scheduler_kwargs
            )
        elif config.name == 'multistep':
            return MultiStepScheduler(
                milestones=config.milestones,
                gamma=config.gamma,
                **scheduler_kwargs
            )
        elif config.name == 'plateau':
            return PlateauScheduler(
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                threshold=config.threshold,
                min_lr=config.min_lr,
                eps=config.eps,
                **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.name}")

    except Exception as e:
        logger.error(f"Failed to build scheduler: {str(e)}")
        raise