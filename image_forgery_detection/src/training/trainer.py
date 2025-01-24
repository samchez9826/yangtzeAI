import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Union, Any
import logging
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from ..utils.metrics import MetricTracker
from ..utils.logger import setup_logger
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """训练器类"""

    def __init__(self,
                 model: nn.Module,
                 criterion: Dict[str, nn.Module],
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 config: Dict[str, Any],
                 device: torch.device):
        """
        初始化训练器
        Args:
            model: 模型
            criterion: 损失函数字典
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 配置字典
            device: 设备
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # 设置日志
        self.logger = setup_logger(__name__)

        # 训练配置
        self.epochs = config['training']['epochs']
        self.save_period = config['training']['save_period']

        # 初始化指标跟踪器
        self.train_metrics = MetricTracker(config['metrics'])
        self.val_metrics = MetricTracker(config['metrics'])

        # 梯度累积步数
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)

        # 混合精度训练
        self.fp16 = config['training'].get('fp16', False)
        if self.fp16:
            self.scaler = GradScaler()

        # 梯度裁剪
        self.grad_clip = config['training'].get('grad_clip', None)

        # Early Stopping
        self.early_stopping = config['training'].get('early_stopping', None)
        if self.early_stopping:
            self.early_stopping_counter = 0
            self.early_stopping_best = float('inf')

        # 检查点保存配置
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # wandb记录
        if config.get('wandb', {}).get('enabled', False):
            self._init_wandb(config['wandb'])

    def _init_wandb(self, config: Dict):
        """初始化wandb"""
        wandb.init(
            project=config['project'],
            config=self.config,
            name=config.get('run_name', None),
            group=config.get('group', None)
        )

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              resume: Optional[str] = None):
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume: 恢复训练的检查点路径
        """
        # 恢复检查点
        start_epoch = 1
        if resume:
            start_epoch = load_checkpoint(
                resume, self.model, self.optimizer,
                self.scheduler, self.device
            )['epoch'] + 1

        self.logger.info("开始训练...")

        for epoch in range(start_epoch, self.epochs + 1):
            # 训练一个epoch
            train_logs = self._train_epoch(train_loader, epoch)

            # 验证
            val_logs = {}
            if val_loader:
                val_logs = self._valid_epoch(val_loader, epoch)

            # 打印日志
            self._print_epoch_logs(epoch, train_logs, val_logs)

            # 学习率调整
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_logs['loss'])
                else:
                    self.scheduler.step()

            # Early Stopping检查
            if self.early_stopping:
                val_loss = val_logs.get('loss', float('inf'))
                if val_loss < self.early_stopping_best:
                    self.early_stopping_best = val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping:
                        self.logger.info("Early Stopping触发，停止训练")
                        break

            # 保存检查点
            if epoch % self.save_period == 0:
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                if self.scheduler:
                    state['scheduler_state_dict'] = self.scheduler.state_dict()

                save_path = self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth'
                save_checkpoint(state, save_path)

            # 更新wandb
            if wandb.run:
                wandb.log({
                    **{f'train/{k}': v for k, v in train_logs.items()},
                    **{f'val/{k}': v for k, v in val_logs.items()},
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """
        训练一个epoch
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
        Returns:
            训练日志
        """
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch} Training')

        for batch_idx, batch in enumerate(train_loader):
            try:
                # 准备数据
                batch = self._prepare_batch(batch)

                # 混合精度训练
                with autocast(enabled=self.fp16):
                    # 前向传播
                    outputs = self.model(batch['image'])

                    # 计算损失
                    losses = self._compute_losses(outputs, batch)
                    loss = losses['total'] / self.accumulation_steps

                # 反向传播
                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 梯度累积
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    if self.grad_clip:
                        if self.fp16:
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip
                        )

                    # 更新参数
                    if self.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                # 更新指标
                self.train_metrics.update('loss', losses['total'].item())
                for task, task_loss in losses.items():
                    if task != 'total':
                        self.train_metrics.update(f'{task}_loss', task_loss.item())

                # 计算其他指标
                self._compute_metrics(outputs, batch, self.train_metrics)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{self.train_metrics.avg('loss'):.4f}"
                })

            except Exception as e:
                self.logger.error(f"训练批次{batch_idx}出错: {str(e)}")
                raise e

        pbar.close()
        return self.train_metrics.result()

    def _valid_epoch(self, val_loader: DataLoader, epoch: int) -> Dict:
        """
        验证一个epoch
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
        Returns:
            验证日志
        """
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(total=len(val_loader), desc=f'Epoch {epoch} Validation')

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # 准备数据
                    batch = self._prepare_batch(batch)

                    # 前向传播
                    outputs = self.model(batch['image'])

                    # 计算损失
                    losses = self._compute_losses(outputs, batch)

                    # 更新指标
                    self.val_metrics.update('loss', losses['total'].item())
                    for task, task_loss in losses.items():
                        if task != 'total':
                            self.val_metrics.update(f'{task}_loss', task_loss.item())

                    # 计算其他指标
                    self._compute_metrics(outputs, batch, self.val_metrics)

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{self.val_metrics.avg('loss'):.4f}"
                    })

                except Exception as e:
                    self.logger.error(f"验证批次{batch_idx}出错: {str(e)}")
                    raise e

        pbar.close()
        return self.val_metrics.result()

    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        准备批次数据
        Args:
            batch: 原始批次数据
        Returns:
            处理后的数据
        """
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _compute_losses(self, outputs: Dict, batch: Dict) -> Dict:
        """
        计算所有损失
        Args:
            outputs: 模型输出
            batch: 批次数据
        Returns:
            损失字典
        """
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
                total_loss += task_loss * self.config['loss_weights'].get(task_name, 1.0)

        losses['total'] = total_loss
        return losses

    def _compute_metrics(self, outputs: Dict, batch: Dict,
                         metrics: MetricTracker):
        """
        计算评估指标
        Args:
            outputs: 模型输出
            batch: 批次数据
            metrics: 指标跟踪器
        """
        # 分类指标
        if 'cls' in outputs and 'cls_target' in batch:
            preds = torch.argmax(outputs['cls'], dim=1)
            targets = batch['cls_target']

            metrics.update('accuracy',
                           (preds == targets).float().mean().item())

            # 计算每个类别的准确率
            for i in range(outputs['cls'].size(1)):
                mask = targets == i
                if mask.sum() > 0:
                    metrics.update(f'class{i}_accuracy',
                                   (preds[mask] == targets[mask]).float().mean().item())

        # 分割指标
        if 'seg' in outputs and 'seg_target' in batch:
            preds = (outputs['seg'] > 0.5).float()
            targets = batch['seg_target']

            # IoU
            intersection = (preds * targets).sum()
            union = preds.sum() + targets.sum() - intersection
            metrics.update('iou', (intersection / (union + 1e-6)).item())

            # Dice系数
            dice = 2 * intersection / (preds.sum() + targets.sum() + 1e-6)
            metrics.update('dice', dice.item())

    def _print_epoch_logs(self, epoch: int, train_logs: Dict,
                          val_logs: Optional[Dict] = None):
        """
        打印epoch日志
        Args:
            epoch: 当前epoch
            train_logs: 训练日志
            val_logs: 验证日志
        """
        msg = f"\nEpoch {epoch} - "
        msg += " - ".join([f"train_{k}: {v:.4f}" for k, v in train_logs.items()])

        if val_logs:
            msg += " - " + " - ".join([f"val_{k}: {v:.4f}" for k, v in val_logs.items()])

        self.logger.info(msg)