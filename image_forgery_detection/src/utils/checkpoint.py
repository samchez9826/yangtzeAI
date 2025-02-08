from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, Protocol, List
from pathlib import Path
import os
import json
import time
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from contextlib import contextmanager
import tempfile
import weakref

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    path: str
    epoch: int
    metric: float
    is_best: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'path': self.path,
            'epoch': self.epoch,
            'metric': self.metric,
            'is_best': self.is_best,
            'timestamp': self.timestamp,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """从字典创建元数据"""
        return cls(**data)


@dataclass
class CheckpointConfig:
    """检查点配置"""
    save_dir: Union[str, Path]
    max_checkpoints: int = 5
    mode: str = 'min'
    save_best_only: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    validate_hash: bool = True
    use_compression: bool = True
    temp_backup: bool = True

    def __post_init__(self):
        """验证配置"""
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.save_dir = Path(self.save_dir)


class CheckpointInterface(Protocol):
    """检查点接口协议"""

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        ...


class CheckpointError(Exception):
    """检查点相关错误"""
    pass


class HashMismatchError(CheckpointError):
    """哈希不匹配错误"""
    pass


class CheckpointIO:
    """检查点IO管理器"""

    def __init__(self, use_compression: bool = True):
        self.use_compression = use_compression

    @contextmanager
    def safe_save(self, filepath: Union[str, Path]):
        """安全保存检查点

        使用临时文件先保存,成功后再重命名,避免保存失败导致文件损坏
        """
        filepath = Path(filepath)
        tmp_path = filepath.parent / f".tmp_{filepath.name}"
        try:
            yield tmp_path
            # 如果成功,重命名临时文件
            if tmp_path.exists():
                tmp_path.rename(filepath)
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()

    def save(self, state: Dict[str, Any], filepath: Union[str, Path]) -> str:
        """保存检查点

        Args:
            state: 状态字典
            filepath: 保存路径

        Returns:
            文件hash值
        """
        with self.safe_save(filepath) as tmp_path:
            if self.use_compression:
                torch.save(state, tmp_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(state, tmp_path)
            # 计算文件hash
            return self._compute_file_hash(tmp_path)

    def load(self, filepath: Union[str, Path],
             expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """加载检查点

        Args:
            filepath: 文件路径
            expected_hash: 预期的hash值

        Returns:
            加载的状态字典
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        # 验证hash
        if expected_hash:
            file_hash = self._compute_file_hash(filepath)
            if file_hash != expected_hash:
                raise HashMismatchError(f"Hash mismatch for {filepath}")

        return torch.load(filepath, map_location='cpu')

    @staticmethod
    def _compute_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
        """计算文件hash值"""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


class ModelCheckpoint:
    """模型检查点基类"""

    def get_state(self) -> Dict[str, Any]:
        """获取模型状态"""
        raise NotImplementedError

    def load_state(self, state: Dict[str, Any]) -> None:
        """加载模型状态"""
        raise NotImplementedError

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """验证状态有效性"""
        raise NotImplementedError


class CheckpointManager:
    """优化的检查点管理器

    特点:
    - 安全的文件操作
    - 状态验证和恢复
    - 自动清理机制
    - 内存使用优化
    - 性能监控支持
    """

    def __init__(self, config: Union[CheckpointConfig, Dict[str, Any]]):
        """
        初始化检查点管理器

        Args:
            config: 检查点配置
        """
        # 初始化配置
        self.config = (config if isinstance(config, CheckpointConfig)
                       else CheckpointConfig(**config))

        # 创建保存目录
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.io = CheckpointIO(use_compression=self.config.use_compression)

        # 加载状态
        self._checkpoints: List[CheckpointMetadata] = []
        self._best_metric = float('inf') if self.config.mode == 'min' else float('-inf')
        self._load_state()

        # 性能监控
        self._save_times: List[float] = []
        self._load_times: List[float] = []

    def _load_state(self) -> None:
        """加载管理器状态"""
        state_file = self.config.save_dir / "checkpoint_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                self._best_metric = state.get('best_metric', self._best_metric)
                checkpoints = state.get('checkpoints', [])
                self._checkpoints = [
                    CheckpointMetadata.from_dict(cp) for cp in checkpoints
                ]

            # 验证现有检查点
            self._validate_checkpoints()

        except Exception as e:
            logger.error(f"Failed to load checkpoint state: {e}")
            # 状态文件损坏时重置
            self._reset_state()

    def _save_state(self) -> None:
        """保存管理器状态"""
        state = {
            'best_metric': self._best_metric,
            'checkpoints': [cp.to_dict() for cp in self._checkpoints]
        }

        state_file = self.config.save_dir / "checkpoint_state.json"
        tmp_file = state_file.with_suffix('.tmp')

        try:
            with open(tmp_file, 'w') as f:
                json.dump(state, f, indent=2)
            tmp_file.rename(state_file)
        except Exception as e:
            logger.error(f"Failed to save checkpoint state: {e}")
            if tmp_file.exists():
                tmp_file.unlink()

    def _validate_checkpoints(self) -> None:
        """验证所有检查点的完整性"""
        valid_checkpoints = []
        for cp in self._checkpoints:
            cp_path = Path(cp.path)
            if not cp_path.exists():
                logger.warning(f"Checkpoint file missing: {cp_path}")
                continue

            if self.config.validate_hash and cp.hash:
                try:
                    actual_hash = self.io._compute_file_hash(cp_path)
                    if actual_hash != cp.hash:
                        logger.warning(f"Hash mismatch for checkpoint: {cp_path}")
                        continue
                except Exception as e:
                    logger.error(f"Failed to validate checkpoint hash: {e}")
                    continue

            valid_checkpoints.append(cp)

        self._checkpoints = valid_checkpoints

    def _reset_state(self) -> None:
        """重置管理器状态"""
        self._checkpoints.clear()
        self._best_metric = float('inf') if self.config.mode == 'min' else float('-inf')
        self._save_state()

    def _is_better(self, metric: float) -> bool:
        """检查是否为更好的指标"""
        if self.config.mode == 'min':
            return metric < self._best_metric
        return metric > self._best_metric

    def _update_best_metric(self, metric: float) -> None:
        """更新最优指标"""
        if self._is_better(metric):
            self._best_metric = metric

    def _clean_old_checkpoints(self) -> None:
        """清理旧的检查点"""
        while len(self._checkpoints) > self.config.max_checkpoints:
            cp = self._checkpoints.pop(0)
            try:
                if Path(cp.path).exists():
                    os.remove(cp.path)
            except Exception as e:
                logger.error(f"Failed to remove old checkpoint: {e}")

    @contextmanager
    def _timing(self, operation: str = 'save'):
        """计时上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation == 'save':
                self._save_times.append(duration)
            else:
                self._load_times.append(duration)

    def save(self,
             model: nn.Module,
             epoch: int,
             metric: float,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             extras: Optional[Dict] = None) -> Optional[str]:
        """
        保存检查点

        Args:
            model: 模型
            epoch: 当前epoch
            metric: 评估指标
            optimizer: 优化器
            scheduler: 学习率调度器
            extras: 额外数据

        Returns:
            保存的文件路径
        """
        with self._timing('save'):
            try:
                # 检查是否需要保存
                is_best = self._is_better(metric)
                if self.config.save_best_only and not is_best:
                    return None

                # 更新最优指标
                self._update_best_metric(metric)

                # 构建检查点内容
                state = {
                    'epoch': epoch,
                    'metric': metric,
                    'model_state_dict': model.state_dict(),
                    'is_best': is_best
                }

                if self.config.save_optimizer and optimizer is not None:
                    state['optimizer_state_dict'] = optimizer.state_dict()
                if self.config.save_scheduler and scheduler is not None:
                    state['scheduler_state_dict'] = scheduler.state_dict()
                if extras:
                    state['extras'] = extras

                # 生成文件名和路径
                filename = f"checkpoint_epoch{epoch}_{metric:.4f}.pth"
                save_path = self.config.save_dir / filename

                # 保存检查点
                file_hash = self.io.save(state, save_path)

                # 创建元数据
                metadata = CheckpointMetadata(
                    path=str(save_path),
                    epoch=epoch,
                    metric=metric,
                    is_best=is_best,
                    hash=file_hash
                )

                # 更新检查点列表
                self._checkpoints.append(metadata)
                self._clean_old_checkpoints()

                # 更新状态文件
                self._save_state()

                # 处理最优模型
                if is_best:
                    best_path = self.config.save_dir / 'best_model.pth'
                    shutil.copy2(save_path, best_path)

                logger.info(f"Saved checkpoint: {filename}")
                return str(save_path)

            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                raise CheckpointError(f"Failed to save checkpoint: {e}")

    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """获取最新的检查点"""
        return self._checkpoints[-1] if self._checkpoints else None

    def get_best_checkpoint(self) -> Optional[CheckpointMetadata]:
        """获取最优的检查点"""
        for cp in reversed(self._checkpoints):
            if cp.is_best:
                return cp
        return None

    def load(self,
             checkpoint_path: Union[str, Path, str],
             model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
             strict: bool = True,
             device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径,可以是具体路径或'best'/'latest'
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            strict: 是否严格加载模型参数
            device: 目标设备

        Returns:
            检查点内容
        """
        with self._timing('load'):
            try:
                # 处理特殊路径
                if isinstance(checkpoint_path, str) and checkpoint_path in ['best', 'latest']:
                    metadata = (self.get_best_checkpoint() if checkpoint_path == 'best'
                                else self.get_latest_checkpoint())
                    if metadata is None:
                        raise CheckpointError(f"No {checkpoint_path} checkpoint available")
                    checkpoint_path = metadata.path
                    expected_hash = metadata.hash
                else:
                    expected_hash = None

                checkpoint_path = Path(checkpoint_path)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

                # 加载检查点
                state = self.io.load(checkpoint_path, expected_hash)

                try:
                    # 加载到设备
                    if device is not None:
                        state = self._to_device(state, device)

                    # 加载模型参数
                    model.load_state_dict(state['model_state_dict'], strict=strict)

                    # 加载优化器参数
                    if optimizer is not None and 'optimizer_state_dict' in state:
                        optimizer.load_state_dict(state['optimizer_state_dict'])

                    # 加载调度器参数
                    if scheduler is not None and 'scheduler_state_dict' in state:
                        scheduler.load_state_dict(state['scheduler_state_dict'])

                    logger.info(f"Loaded checkpoint: {checkpoint_path}")
                    return state

                except Exception as e:
                    # 加载失败的详细日志
                    logger.error(f"Failed to restore state: {str(e)}")
                    # 获取更多错误信息
                    if isinstance(e, RuntimeError):
                        logger.error(f"State keys: {state.keys()}")
                        if 'model_state_dict' in state:
                            logger.error(f"Model state keys: {state['model_state_dict'].keys()}")
                    raise CheckpointError(f"Failed to restore state: {str(e)}")

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise CheckpointError(f"Failed to load checkpoint: {e}")

    def load_partial(self,
                     checkpoint_path: Union[str, Path],
                     model: nn.Module,
                     keys: List[str],
                     strict: bool = False) -> Dict[str, Any]:
        """
        部分加载检查点

        Args:
            checkpoint_path: 检查点路径
            model: 模型
            keys: 需要加载的键列表
            strict: 是否严格检查键存在
        """
        try:
            state = self.io.load(checkpoint_path)
            model_state = state['model_state_dict']

            # 筛选指定的键
            partial_state = {
                k: v for k, v in model_state.items()
                if any(key in k for key in keys)
            }

            # 检查是否找到所有请求的键
            if strict:
                found_keys = set(k for k in partial_state.keys()
                                 for key in keys if key in k)
                missing = set(keys) - found_keys
                if missing:
                    raise KeyError(f"Keys not found: {missing}")

            # 加载部分权重
            model.load_state_dict(partial_state, strict=False)
            logger.info(f"Loaded {len(partial_state)} parameters from {checkpoint_path}")

            return state

        except Exception as e:
            logger.error(f"Failed to load partial checkpoint: {e}")
            raise CheckpointError(f"Failed to load partial checkpoint: {e}")

    def _to_device(self, state_dict: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """将状态字典移动到指定设备"""
        result = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, dict):
                result[k] = self._to_device(v, device)
            else:
                result[k] = v
        return result

    def get_performance_info(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'save_times': {
                'mean': np.mean(self._save_times) if self._save_times else 0,
                'std': np.std(self._save_times) if self._save_times else 0,
                'min': np.min(self._save_times) if self._save_times else 0,
                'max': np.max(self._save_times) if self._save_times else 0,
                'count': len(self._save_times)
            },
            'load_times': {
                'mean': np.mean(self._load_times) if self._load_times else 0,
                'std': np.std(self._load_times) if self._load_times else 0,
                'min': np.min(self._load_times) if self._load_times else 0,
                'max': np.max(self._load_times) if self._load_times else 0,
                'count': len(self._load_times)
            }
        }

    def verify_checkpoints(self) -> Dict[str, bool]:
        """验证所有检查点的完整性"""
        results = {}
        for cp in self._checkpoints:
            try:
                # 检查文件存在
                path = Path(cp.path)
                if not path.exists():
                    results[cp.path] = False
                    continue

                # 验证hash
                if self.config.validate_hash and cp.hash:
                    current_hash = self.io._compute_file_hash(path)
                    results[cp.path] = current_hash == cp.hash
                else:
                    # 尝试加载验证
                    _ = self.io.load(path)
                    results[cp.path] = True

            except Exception as e:
                logger.error(f"Failed to verify checkpoint {cp.path}: {e}")
                results[cp.path] = False

        return results

    def cleanup(self, keep_best: bool = True) -> None:
        """清理检查点文件"""
        try:
            best_cp = self.get_best_checkpoint() if keep_best else None
            removed = []

            for cp in self._checkpoints[:]:
                if best_cp and cp.path == best_cp.path:
                    continue

                try:
                    path = Path(cp.path)
                    if path.exists():
                        path.unlink()
                    removed.append(cp)
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {cp.path}: {e}")

            # 更新检查点列表
            for cp in removed:
                self._checkpoints.remove(cp)
            self._save_state()
            logger.info(f"Removed {len(removed)} checkpoints")

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")

    def __str__(self) -> str:
        """可读的字符串表示"""
        return (f"CheckpointManager(save_dir={self.config.save_dir}, "
                f"checkpoints={len(self._checkpoints)}, "
                f"best_metric={self._best_metric:.4f})")

    def __repr__(self) -> str:
        """详细的字符串表示"""
        return (f"CheckpointManager(config={self.config}, "
                f"checkpoints={self._checkpoints}, "
                f"best_metric={self._best_metric})")