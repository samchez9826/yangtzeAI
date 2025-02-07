from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Union, Any, Protocol, Callable
import logging
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from ..utils.metrics import MetricTracker
from ..utils.logger import setup_logger
from ..utils.checkpoint import save_checkpoint, load_checkpoint
import psutil
import gc
from contextlib import contextmanager


@dataclass
class TrainerConfig:
    """训练器配置"""
    # 基本训练参数
    epochs: int
    save_period: int
    checkpoint_dir: str

    # 优化器配置
    accumulation_steps: int = 1
    fp16: bool = False
    grad_clip: Optional[float] = None

    # 早停配置
    early_stopping: Optional[dict] = None
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # 内存管理
    cache_size: int = 1000
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2

    # 日志配置
    log_interval: int = 100
    metrics: List[str] = None
    wandb: Optional[dict] = None

    # 性能监控
    monitor_memory: bool = True
    profile_batch: Optional[int] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainerConfig':
        """从字典创建配置"""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

    def validate(self) -> None:
        """验证配置有效性"""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps must be positive, got {self.accumulation_steps}")
        if self.save_period <= 0:
            raise ValueError(f"save_period must be positive, got {self.save_period}")


class TrainingState:
    """训练状态管理"""

    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_value: float = float('inf')
        self.early_stopping_counter: int = 0
        self.early_stopping_best: float = float('inf')

    def update(self, **kwargs):
        """更新状态"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_value': self.best_value,
            'early_stopping_counter': self.early_stopping_counter,
            'early_stopping_best': self.early_stopping_best
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


class TrainingCallback(Protocol):
    """训练回调接口"""

    def on_train_begin(self, trainer: 'Trainer') -> None:
        """训练开始时调用"""
        pass

    def on_train_end(self, trainer: 'Trainer') -> None:
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None:
        """每个epoch开始时调用"""
        pass

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, float]) -> None:
        """每个epoch结束时调用"""
        pass

    def on_batch_begin(self, trainer: 'Trainer', batch: int) -> None:
        """每个batch开始时调用"""
        pass

    def on_batch_end(self, trainer: 'Trainer', batch: int, logs: Dict[str, float]) -> None:
        """每个batch结束时调用"""
        pass


class MemoryManager:
    """内存管理器"""

    def __init__(self, monitor: bool = True):
        self.monitor = monitor
        self.peak_memory = 0

    @contextmanager
    def track(self):
        """跟踪内存使用"""
        try:
            if self.monitor:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            yield
        finally:
            if self.monitor:
                memory_stats = torch.cuda.memory_stats()
                peak = memory_stats["allocated_bytes.all.peak"] / 1024 / 1024
                self.peak_memory = max(self.peak_memory, peak)

    def report(self) -> Dict[str, float]:
        """报告内存使用情况"""
        if not self.monitor:
            return {}

        stats = {
            'peak_gpu_memory_mb': self.peak_memory,
            'current_gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'current_gpu_cache_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'cpu_memory_percent': psutil.Process().memory_percent()
        }
        return stats

    def clear(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BaseTrainer:
    """训练器基类"""

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: Union[Dict, TrainerConfig],
                 device: torch.device):
        """
        初始化训练器

        Args:
            model: 模型
            optimizer: 优化器
            config: 训练配置
            device: 训练设备
        """
        # 转换配置
        if isinstance(config, dict):
            self.config = TrainerConfig.from_dict(config)
        else:
            self.config = config

        # 验证配置
        self.config.validate()

        self.model = model
        self.optimizer = optimizer
        self.device = device

        # 初始化组件
        self.logger = setup_logger(__name__)
        self.memory = MemoryManager(self.config.monitor_memory)
        self.state = TrainingState()
        self.metrics = MetricTracker(self.config.metrics or [])

        # 混合精度训练
        self.scaler = GradScaler() if self.config.fp16 else None

        # 检查点目录
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 回调列表
        self.callbacks: List[TrainingCallback] = []

        # wandb初始化
        if self.config.wandb and self.config.wandb.get('enabled', False):
            self._init_wandb(self.config.wandb)

    def _init_wandb(self, config: Dict) -> None:
        """初始化wandb"""
        wandb.init(
            project=config['project'],
            config=self.config.__dict__,
            name=config.get('run_name'),
            group=config.get('group'),
            tags=config.get('tags', [])
        )

    def add_callback(self, callback: TrainingCallback) -> None:
        """添加训练回调"""
        self.callbacks.append(callback)

    def _call_callbacks(self, hook: str, *args, **kwargs) -> None:
        """调用回调函数"""
        for callback in self.callbacks:
            getattr(callback, hook)(self, *args, **kwargs)

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """训练步骤"""
        pass

    @abstractmethod
    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """验证步骤"""
        pass

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """准备批次数据"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _handle_early_stopping(self, val_loss: float) -> bool:
        """处理早停"""
        if not self.config.early_stopping:
            return False

        if val_loss < self.state.early_stopping_best - self.config.early_stopping_min_delta:
            self.state.early_stopping_best = val_loss
            self.state.early_stopping_counter = 0
        else:
            self.state.early_stopping_counter += 1

        return self.state.early_stopping_counter >= self.config.early_stopping_patience


class Trainer(BaseTrainer):
    """优化的训练器实现"""

    def __init__(self,
                 model: nn.Module,
                 criterion: Dict[str, nn.Module],
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 config: Union[Dict, TrainerConfig],
                 device: torch.device):
        """
        初始化训练器

        Args:
            model: 模型
            criterion: 损失函数字典
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 训练配置
            device: 训练设备
        """
        super().__init__(model, optimizer, config, device)
        self.criterion = criterion
        self.scheduler = scheduler

        # 训练和验证指标
        self.train_metrics = MetricTracker(self.config.metrics)
        self.val_metrics = MetricTracker(self.config.metrics)

        # 性能分析器
        self.profiler = None
        if self.config.profile_batch is not None:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=6,
                    repeat=1
                ),
                with_stack=True
            )

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              resume_path: Optional[str] = None) -> None:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_path: 恢复训练的检查点路径
        """
        try:
            # 恢复训练
            if resume_path:
                self._resume_checkpoint(resume_path)

            # 调用训练开始回调
            self._call_callbacks('on_train_begin')

            self.logger.info("开始训练...")

            # 主训练循环
            for epoch in range(self.state.epoch + 1, self.config.epochs + 1):
                try:
                    self.state.epoch = epoch

                    # 训练一个epoch
                    train_logs = self._train_epoch(train_loader)

                    # 验证
                    val_logs = {}
                    if val_loader:
                        val_logs = self._validate_epoch(val_loader)

                    # 更新学习率
                    self._update_learning_rate(val_logs.get('loss'))

                    # 检查早停
                    if self._handle_early_stopping(val_logs.get('loss', float('inf'))):
                        self.logger.info("Early stopping triggered")
                        break

                    # 保存检查点
                    if epoch % self.config.save_period == 0:
                        self._save_checkpoint(epoch, val_logs.get('loss', None))

                    # 记录日志
                    self._log_epoch(epoch, train_logs, val_logs)

                except Exception as e:
                    self.logger.error(f"Epoch {epoch} failed: {str(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # 清理资源
            self.memory.clear()
            if self.profiler:
                self.profiler.stop()
            # 调用训练结束回调
            self._call_callbacks('on_train_end')

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()

        # 调用epoch开始回调
        self._call_callbacks('on_epoch_begin', self.state.epoch)

        pbar = tqdm(total=len(train_loader), desc=f'Epoch {self.state.epoch}')

        for batch_idx, batch in enumerate(train_loader):
            try:
                with self.memory.track():
                    # 调用batch开始回调
                    self._call_callbacks('on_batch_begin', batch_idx)

                    # 训练步骤
                    batch_logs = self.train_step(batch)

                    # 更新指标
                    for name, value in batch_logs.items():
                        self.train_metrics.update(name, value)

                    # 更新进度条
                    pbar.update()
                    if batch_idx % self.config.log_interval == 0:
                        pbar.set_postfix(self.train_metrics.current())

                    # 调用batch结束回调
                    self._call_callbacks('on_batch_end', batch_idx, batch_logs)

                    # 性能分析
                    if self.profiler and batch_idx == self.config.profile_batch:
                        self.profiler.step()

            except Exception as e:
                self.logger.error(f"Batch {batch_idx} failed: {str(e)}")
                raise

        pbar.close()

        # 调用epoch结束回调
        epoch_logs = self.train_metrics.result()
        self._call_callbacks('on_epoch_end', self.state.epoch, epoch_logs)

        return epoch_logs

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一个训练步骤"""
        # 准备数据
        batch = self._prepare_batch(batch)

        # 混合精度训练
        with autocast(enabled=self.config.fp16):
            # 前向传播
            outputs = self.model(batch['image'])

            # 计算损失
            losses = self._compute_losses(outputs, batch)
            loss = losses['total'] / self.config.accumulation_steps

        # 反向传播
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积
        if (self.state.global_step + 1) % self.config.accumulation_steps == 0:
            # 梯度裁剪
            if self.config.grad_clip is not None:
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            # 更新参数
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        # 更新全局步数
        self.state.global_step += 1

        # 返回损失字典
        return {name: loss.item() for name, loss in losses.items()}

    def _compute_losses(self, outputs: Dict[str, torch.Tensor],
                        batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        total_loss = 0

        # 计算各任务损失
        for task_name, criterion in self.criterion.items():
            if f'{task_name}_target' in batch:
                task_loss = criterion(
                    outputs[task_name],
                    batch[f'{task_name}_target']
                )
                losses[task_name] = task_loss
                # 应用损失权重
                if 'loss_weights' in self.config:
                    task_loss = task_loss * self.config.loss_weights.get(task_name, 1.0)
                total_loss += task_loss

        losses['total'] = total_loss
        return losses


class Trainer(BaseTrainer):  # continued

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证指标字典
        """
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(total=len(val_loader), desc=f'Epoch {self.state.epoch} Validation')

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    with self.memory.track():
                        # 验证步骤
                        batch_logs = self.validate_step(batch)

                        # 更新指标
                        for name, value in batch_logs.items():
                            self.val_metrics.update(name, value)

                        # 更新进度条
                        pbar.update()
                        if batch_idx % self.config.log_interval == 0:
                            pbar.set_postfix(self.val_metrics.current())

                except Exception as e:
                    self.logger.error(f"Validation batch {batch_idx} failed: {str(e)}")
                    raise

        pbar.close()
        return self.val_metrics.result()

    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一个验证步骤

        Args:
            batch: 批次数据

        Returns:
            验证指标字典
        """
        # 准备数据
        batch = self._prepare_batch(batch)

        # 前向传播
        outputs = self.model(batch['image'])

        # 计算损失
        losses = self._compute_losses(outputs, batch)

        # 计算其他指标
        metrics = self._compute_metrics(outputs, batch)

        return {**losses, **metrics}

    def _compute_metrics(self, outputs: Dict[str, torch.Tensor],
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算评估指标

        Args:
            outputs: 模型输出
            batch: 数据批次

        Returns:
            指标字典
        """
        metrics = {}

        # 分类指标
        if 'cls' in outputs and 'cls_target' in batch:
            preds = torch.argmax(outputs['cls'], dim=1)
            targets = batch['cls_target']

            # 准确率
            acc = (preds == targets).float().mean().item()
            metrics['accuracy'] = acc

            # 分类具体类别的准确率
            for i in range(outputs['cls'].size(1)):
                mask = targets == i
                if mask.sum() > 0:
                    class_acc = (preds[mask] == targets[mask]).float().mean().item()
                    metrics[f'class{i}_accuracy'] = class_acc

        # 分割指标
        if 'seg' in outputs and 'seg_target' in batch:
            preds = (outputs['seg'] > 0.5).float()
            targets = batch['seg_target']

            # IoU
            intersection = (preds * targets).sum().item()
            union = (preds + targets).sum().item() - intersection
            iou = intersection / (union + 1e-8)
            metrics['iou'] = iou

            # Dice系数
            dice = (2 * intersection) / (preds.sum().item() + targets.sum().item() + 1e-8)
            metrics['dice'] = dice

        return metrics

    def _update_learning_rate(self, val_loss: Optional[float] = None) -> None:
        """更新学习率

        Args:
            val_loss: 验证损失
        """
        if self.scheduler is None:
            return

        try:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is None:
                    self.logger.warning("No validation loss provided for ReduceLROnPlateau")
                    return
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning rate updated to {current_lr:.2e}")

        except Exception as e:
            self.logger.error(f"Failed to update learning rate: {str(e)}")

    def _save_checkpoint(self, epoch: int, val_loss: Optional[float] = None) -> None:
        """保存检查点

        Args:
            epoch: 当前epoch
            val_loss: 验证损失
        """
        try:
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.config.__dict__,
                'state': self.state.state_dict()
            }

            if self.scheduler is not None:
                state['scheduler'] = self.scheduler.state_dict()

            # 保存最新检查点
            latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
            torch.save(state, latest_path)

            # 保存当前epoch检查点
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
            torch.save(state, epoch_path)

            # 保存最佳模型
            if val_loss is not None and val_loss < self.state.best_value:
                self.state.best_value = val_loss
                best_path = self.checkpoint_dir / 'checkpoint_best.pth'
                torch.save(state, best_path)

            self.logger.info(f"Checkpoint saved at epoch {epoch}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")

    def _resume_checkpoint(self, resume_path: str) -> None:
        """恢复检查点

        Args:
            resume_path: 检查点路径
        """
        try:
            self.logger.info(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)

            # 加载模型权重
            self.model.load_state_dict(checkpoint['state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # 加载调度器状态
            if self.scheduler is not None and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

            # 加载训练状态
            self.state.load_state_dict(checkpoint['state'])

            # 尝试加载配置
            if 'config' in checkpoint:
                # 只更新非关键配置
                for k, v in checkpoint['config'].items():
                    if k not in ['model', 'optimizer', 'scheduler']:
                        setattr(self.config, k, v)

            self.logger.info(f"Checkpoint loaded (epoch {checkpoint['epoch']})")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    def _log_epoch(self, epoch: int,
                   train_logs: Dict[str, float],
                   val_logs: Dict[str, float]) -> None:
        """记录epoch日志

        Args:
            epoch: 当前epoch
            train_logs: 训练指标
            val_logs: 验证指标
        """
        # 1. 命令行日志
        log_str = f"\nEpoch {epoch} - "
        # 训练指标
        train_str = " - ".join([
            f"train_{k}: {v:.4f}" for k, v in train_logs.items()
        ])
        log_str += train_str

        # 验证指标
        if val_logs:
            val_str = " - ".join([
                f"val_{k}: {v:.4f}" for k, v in val_logs.items()
            ])
            log_str += " - " + val_str

        # 学习率
        lr = self.optimizer.param_groups[0]['lr']
        log_str += f" - lr: {lr:.2e}"

        # 内存使用
        memory_stats = self.memory.report()
        if memory_stats:
            mem_str = " - ".join([
                f"{k}: {v:.1f}" for k, v in memory_stats.items()
            ])
            log_str += f"\nMemory Usage: {mem_str}"

        self.logger.info(log_str)

        # 2. Wandb日志
        if wandb.run:
            wandb_logs = {
                'epoch': epoch,
                'lr': lr,
                **{f"train/{k}": v for k, v in train_logs.items()},
                **{f"val/{k}": v for k, v in val_logs.items()},
                **{f"memory/{k}": v for k, v in memory_stats.items()}
            }
            wandb.log(wandb_logs)