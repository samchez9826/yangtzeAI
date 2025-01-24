import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from pathlib import Path
import os
import json
import time
import logging
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""

    def __init__(self,
                 save_dir: Union[str, Path],
                 max_checkpoints: int = 5,
                 mode: str = 'min',
                 save_best_only: bool = True):
        """
        初始化检查点管理器
        Args:
            save_dir: 保存目录
            max_checkpoints: 最大保存数量
            mode: 'min' 或 'max'，用于确定最优模型
            save_best_only: 是否只保存最优模型
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.mode = mode
        self.save_best_only = save_best_only

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化最优指标
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints_list = []

        # 加载检查点历史记录
        self._load_checkpoints_history()

    def _load_checkpoints_history(self):
        """加载检查点历史记录"""
        history_file = self.save_dir / 'checkpoints_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.best_metric = history.get('best_metric', self.best_metric)
                self.checkpoints_list = history.get('checkpoints_list', [])

    def _save_checkpoints_history(self):
        """保存检查点历史记录"""
        history_file = self.save_dir / 'checkpoints_history.json'
        history = {
            'best_metric': self.best_metric,
            'checkpoints_list': self.checkpoints_list
        }
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

    def save(self,
             epoch: int,
             metric: float,
             model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             extras: Optional[Dict] = None) -> str:
        """
        保存检查点
        Args:
            epoch: 当前轮次
            metric: 评估指标
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            extras: 额外需要保存的内容
        Returns:
            检查点文件路径
        """
        # 检查是否为最优模型
        is_best = False
        if self.mode == 'min':
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric

        if is_best:
            self.best_metric = metric

        # 如果只保存最优模型且当前不是最优
        if self.save_best_only and not is_best:
            return None

        # 构建检查点内容
        checkpoint = {
            'epoch': epoch,
            'metric': metric,
            'model_state_dict': model.state_dict(),
            'is_best': is_best,
            'timestamp': time.strftime('%Y%m%d_%H%M%S')
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if extras is not None:
            checkpoint['extras'] = extras

        # 生成检查点文件名
        filename = f"checkpoint_epoch{epoch}_{metric:.4f}.pth"
        save_path = str(self.save_dir / filename)

        # 保存检查点
        torch.save(checkpoint, save_path)

        # 更新检查点列表
        self.checkpoints_list.append({
            'path': save_path,
            'epoch': epoch,
            'metric': metric,
            'is_best': is_best
        })

        # 如果超过最大保存数量，删除旧检查点
        if len(self.checkpoints_list) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoints_list.pop(0)
            if os.path.exists(oldest_checkpoint['path']):
                os.remove(oldest_checkpoint['path'])

        # 保存历史记录
        self._save_checkpoints_history()

        # 如果是最优模型，复制一份best模型
        if is_best:
            best_path = str(self.save_dir / 'best_model.pth')
            shutil.copy2(save_path, best_path)

        logger.info(f"保存检查点: {filename}")
        return save_path

    def load(self,
             checkpoint_path: Union[str, Path],
             model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             strict: bool = True) -> dict:
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径，可以是具体路径或'best'或'latest'
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            strict: 是否严格加载模型参数
        Returns:
            检查点内容
        """
        if isinstance(checkpoint_path, str) and checkpoint_path in ['best', 'latest']:
            if checkpoint_path == 'best':
                checkpoint_path = self.save_dir / 'best_model.pth'
            else:  # latest
                if not self.checkpoints_list:
                    raise ValueError("没有可用的检查点")
                checkpoint_path = self.checkpoints_list[-1]['path']

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # 加载优化器参数
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器参数
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"加载检查点: {checkpoint_path}")
        return checkpoint

    def get_best_checkpoint(self) -> Optional[dict]:
        """获取最优检查点信息"""
        for checkpoint in reversed(self.checkpoints_list):
            if checkpoint['is_best']:
                return checkpoint
        return None

    def get_latest_checkpoint(self) -> Optional[dict]:
        """获取最新检查点信息"""
        return self.checkpoints_list[-1] if self.checkpoints_list else None


def save_checkpoint(state: dict,
                    save_path: Union[str, Path],
                    is_best: bool = False):
    """
    保存检查点的快捷函数
    Args:
        state: 要保存的状态字典
        save_path: 保存路径
        is_best: 是否为最优模型
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, save_path)
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        shutil.copy2(save_path, best_path)


def load_checkpoint(checkpoint_path: Union[str, Path],
                    model: Optional[nn.Module] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    strict: bool = True) -> dict:
    """
    加载检查点的快捷函数
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        strict: 是否严格加载模型参数
    Returns:
        检查点内容
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint