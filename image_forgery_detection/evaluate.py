from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from dataclasses import dataclass
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import psutil
import gc
from contextlib import contextmanager

from src.data_processing.dataset import (
    CASIADataset,
    FaceForensicsDataset,
    DatasetError
)
from src.models.base_model import BaseForgeryDetector
from src.evaluation.metrics import (
    BinaryClassificationMetrics,
    SegmentationMetrics
)
from src.evaluation.visualizer import VisualizationManager
from src.utils.logger import setup_logger
from src.utils.checkpoint import CheckpointError

# 配置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置类"""
    # 基本配置
    config_path: Path
    checkpoint_path: Path
    output_dir: Path
    device: str = 'cuda'

    # 数据加载配置
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # 评估配置
    save_predictions: bool = True
    visualize_samples: int = 16
    metrics: List[str] = ('accuracy', 'precision', 'recall', 'f1', 'auc')

    # 可视化配置
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_pr_curve: bool = True

    # 性能监控配置
    monitor_memory: bool = True
    profile_batch: Optional[int] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'EvaluationConfig':
        """从命令行参数创建配置"""
        return cls(
            config_path=Path(args.config),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            device=args.device
        )

    def validate(self) -> None:
        """验证配置有效性"""
        if not self.config_path.exists():
            raise ValueError(f"Config file not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {self.checkpoint_path}")
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"Invalid number of workers: {self.num_workers}")

        # 检查设备可用性
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'

        # 根据系统资源调整参数
        if self.num_workers > psutil.cpu_count():
            logger.warning(
                f"Reducing number of workers from {self.num_workers} to {psutil.cpu_count()}"
            )
            self.num_workers = psutil.cpu_count()


class MemoryTracker:
    """内存使用跟踪器"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.peak_memory = 0

    @contextmanager
    def track(self):
        """跟踪内存使用上下文"""
        if not self.enabled:
            yield
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            yield
        finally:
            if torch.cuda.is_available():
                memory_stats = torch.cuda.memory_stats()
                peak = memory_stats["allocated_bytes.all.peak"] / 1024 / 1024
                self.peak_memory = max(self.peak_memory, peak)

    def report(self) -> Dict[str, float]:
        """生成内存使用报告"""
        if not self.enabled:
            return {}

        return {
            'peak_gpu_memory_mb': self.peak_memory,
            'current_gpu_memory_mb': (
                torch.cuda.memory_allocated() / 1024 / 1024
                if torch.cuda.is_available() else 0
            ),
            'current_cpu_percent': psutil.Process().memory_percent()
        }

    def clear(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Evaluate forgery detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run evaluation on'
    )

    return parser.parse_args()


class Evaluator:
    """优化的评估器实现"""

    def __init__(self, config: EvaluationConfig):
        """
        初始化评估器

        Args:
            config: 评估配置
        """
        self.config = config
        self.device = torch.device(config.device)

        # 初始化组件
        self.memory_tracker = MemoryTracker(config.monitor_memory)
        self.logger = setup_logger(
            'evaluator',
            config.output_dir / 'logs',
            level='INFO'
        )

        # 加载模型配置
        with open(config.config_path) as f:
            self.model_config = yaml.safe_load(f)

        # 创建输出目录
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化评估组件
        self._setup_components()

        # 性能统计
        self.eval_times: List[float] = []

    def _setup_components(self) -> None:
        """初始化评估组件"""
        # 初始化模型
        try:
            self.model = self._build_model()
        except Exception as e:
            raise RuntimeError(f"Failed to build model: {e}")

        # 初始化数据加载器
        try:
            self.data_loader = self._build_dataloader()
        except Exception as e:
            raise RuntimeError(f"Failed to build data loader: {e}")

        # 初始化评估指标
        self.metrics = {
            'classification': BinaryClassificationMetrics(),
            'segmentation': SegmentationMetrics(num_classes=2)
        }

        # 初始化可视化管理器
        self.visualizer = VisualizationManager(
            self.output_dir / 'visualizations'
        )

    def _build_model(self) -> nn.Module:
        """构建和加载模型"""
        # 创建模型
        model = BaseForgeryDetector.create(
            self.model_config['model']['name'],
            self.model_config
        ).to(self.device)

        # 加载检查点
        try:
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(
                f"Loaded checkpoint from {self.config.checkpoint_path}"
            )
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")

        return model

    def _build_dataloader(self) -> DataLoader:
        """构建数据加载器"""
        # 创建数据集
        try:
            if 'casia' in self.model_config['datasets']:
                dataset = CASIADataset(
                    self.model_config['datasets']['casia']['test_path']
                )
            elif 'faceforensics' in self.model_config['datasets']:
                dataset = FaceForensicsDataset(
                    self.model_config['datasets']['faceforensics']['test_path']
                )
            else:
                raise ValueError("No supported dataset found in config")
        except Exception as e:
            raise DatasetError(f"Failed to create dataset: {e}")

        # 创建数据加载器
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False
        )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        执行评估流程

        Returns:
            评估结果字典
        """
        self.model.eval()
        start_time = time.time()

        predictions = []
        targets = []
        seg_outputs = []
        seg_targets = []

        try:
            # 评估循环
            for batch_idx, batch in enumerate(tqdm(self.data_loader, desc='Evaluating')):
                with self.memory_tracker.track():
                    batch_results = self._evaluate_batch(batch, batch_idx)

                    # 收集结果
                    predictions.extend(batch_results['predictions'])
                    targets.extend(batch_results['targets'])
                    if 'seg_outputs' in batch_results:
                        seg_outputs.extend(batch_results['seg_outputs'])
                        seg_targets.extend(batch_results['seg_targets'])

            # 计算指标
            results = self._compute_metrics(
                predictions,
                targets,
                seg_outputs,
                seg_targets
            )

            # 生成可视化
            self._generate_visualizations(
                predictions,
                targets,
                seg_outputs,
                seg_targets
            )

            # 保存结果
            self._save_results(results)

            # 记录评估时间
            eval_time = time.time() - start_time
            self.eval_times.append(eval_time)
            results['evaluation_time'] = eval_time

            # 添加内存使用信息
            results['memory_stats'] = self.memory_tracker.report()

            return results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            # 清理资源
            self.memory_tracker.clear()

    def _evaluate_batch(self, batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> Dict[str, List]:
        """评估单个批次"""
        # 准备数据
        images = batch['image'].to(self.device)
        cls_targets = batch['cls_target'].to(self.device)
        seg_targets = batch.get('seg_target')
        if seg_targets is not None:
            seg_targets = seg_targets.to(self.device)

        # 前向传播
        outputs = self.model(images)

        # 更新指标
        self.metrics['classification'].update(outputs['cls'], cls_targets)
        if seg_targets is not None and 'seg' in outputs:
            self.metrics['segmentation'].update(outputs['seg'], seg_targets)

        # 收集结果
        batch_results = {
            'predictions': outputs['cls'].argmax(1).cpu().numpy(),
            'targets': cls_targets.cpu().numpy()
        }

        # 收集分割结果
        if seg_targets is not None and 'seg' in outputs:
            batch_results.update({
                'seg_outputs': outputs['seg'].cpu().numpy(),
                'seg_targets': seg_targets.cpu().numpy()
            })

        # 可视化样本
        if batch_idx == 0 and self.config.visualize_samples > 0:
            self._visualize_batch(
                images.cpu().numpy(),
                outputs,
                seg_targets.cpu().numpy() if seg_targets is not None else None
            )

        return batch_results


class Evaluator:  # continued
    def _compute_metrics(
            self,
            predictions: List[np.ndarray],
            targets: List[np.ndarray],
            seg_outputs: Optional[List[np.ndarray]] = None,
            seg_targets: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        计算评估指标

        Args:
            predictions: 分类预测结果
            targets: 真实标签
            seg_outputs: 分割预测结果
            seg_targets: 分割真实标签

        Returns:
            包含所有指标的字典
        """
        try:
            results = {}

            # 计算分类指标
            cls_metrics = self.metrics['classification'].compute()
            results['classification'] = {
                k: float(v) for k, v in cls_metrics.items()
            }

            # 计算分割指标
            if seg_outputs and seg_targets:
                seg_metrics = self.metrics['segmentation'].compute()
                results['segmentation'] = {
                    k: float(v) for k, v in seg_metrics.items()
                }

            # 计算每个类别的指标
            results['per_class'] = self._compute_per_class_metrics(
                predictions,
                targets
            )

            # 添加样本统计
            results['statistics'] = {
                'total_samples': len(targets),
                'positive_samples': int(sum(targets)),
                'negative_samples': int(len(targets) - sum(targets))
            }

            return results

        except Exception as e:
            self.logger.error(f"Failed to compute metrics: {e}")
            raise

    def _compute_per_class_metrics(
            self,
            predictions: List[np.ndarray],
            targets: List[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """计算每个类别的详细指标"""
        predictions = np.array(predictions)
        targets = np.array(targets)

        per_class_metrics = {}
        for cls_idx in range(2):  # 二分类问题
            mask = targets == cls_idx
            if not np.any(mask):
                continue

            cls_preds = predictions[mask]
            cls_targets = targets[mask]

            per_class_metrics[f'class_{cls_idx}'] = {
                'accuracy': float(np.mean(cls_preds == cls_targets)),
                'support': int(np.sum(mask))
            }

        return per_class_metrics

    def _visualize_batch(
            self,
            images: np.ndarray,
            outputs: Dict[str, torch.Tensor],
            seg_targets: Optional[np.ndarray] = None
    ) -> None:
        """
        可视化批次结果

        Args:
            images: 输入图像
            outputs: 模型输出
            seg_targets: 分割真实标签
        """
        try:
            n_samples = min(
                images.shape[0],
                self.config.visualize_samples
            )

            # 可视化预测结果
            self.visualizer.visualize_predictions(
                images[:n_samples],
                seg_targets[:n_samples] if seg_targets is not None else None,
                outputs['seg'].cpu().numpy()[:n_samples] if 'seg' in outputs else None,
                max_samples=n_samples
            )

            # 可视化注意力图(如果可用)
            if 'attention_maps' in outputs:
                self.visualizer.visualize_attention_maps(
                    images[:n_samples],
                    outputs['attention_maps'].cpu().numpy()[:n_samples]
                )

        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {e}")

    def _generate_visualizations(
            self,
            predictions: List[np.ndarray],
            targets: List[np.ndarray],
            seg_outputs: Optional[List[np.ndarray]] = None,
            seg_targets: Optional[List[np.ndarray]] = None
    ) -> None:
        """生成评估可视化"""
        try:
            if self.config.plot_confusion_matrix:
                self.visualizer.plot_confusion_matrix(
                    targets,
                    predictions,
                    labels=['Authentic', 'Tampered'],
                    title='Confusion Matrix'
                )

            if self.config.plot_roc_curve:
                self.visualizer.plot_roc_curve(
                    targets,
                    predictions,
                    title='ROC Curve'
                )

            if self.config.plot_pr_curve:
                self.visualizer.plot_pr_curve(
                    targets,
                    predictions,
                    title='Precision-Recall Curve'
                )

            # 生成分割结果可视化
            if seg_outputs and seg_targets:
                self.visualizer.plot_segmentation_results(
                    seg_outputs,
                    seg_targets,
                    title='Segmentation Results'
                )

        except Exception as e:
            self.logger.warning(f"Failed to generate result plots: {e}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        保存评估结果

        Args:
            results: 评估结果字典
        """
        try:
            # 保存详细结果
            results_file = self.output_dir / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)

            # 保存摘要结果
            summary = self._generate_summary(results)
            summary_file = self.output_dir / 'evaluation_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(summary)

            self.logger.info(f"Results saved to {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """生成评估摘要"""
        lines = [
            "Evaluation Summary",
            "=" * 50,
            f"\nDataset: {self.model_config['datasets']['name']}",
            f"Model: {self.model_config['model']['name']}",
            f"Checkpoint: {self.config.checkpoint_path}",
            f"\nEvaluation Time: {results['evaluation_time']:.2f}s",
            "\nClassification Metrics:"
        ]

        # 添加分类指标
        for name, value in results['classification'].items():
            lines.append(f"{name}: {value:.4f}")

        # 添加分割指标
        if 'segmentation' in results:
            lines.extend([
                "\nSegmentation Metrics:"
            ])
            for name, value in results['segmentation'].items():
                lines.append(f"{name}: {value:.4f}")

        # 添加性能统计
        if 'memory_stats' in results:
            lines.extend([
                "\nPerformance Statistics:",
                f"Peak GPU Memory: {results['memory_stats'].get('peak_gpu_memory_mb', 0):.1f}MB",
                f"CPU Memory Usage: {results['memory_stats'].get('current_cpu_percent', 0):.1f}%"
            ])

        return "\n".join(lines)


class NumpyEncoder(json.JSONEncoder):
    """用于JSON序列化numpy数据类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_environment(config: EvaluationConfig) -> None:
    """
    配置运行环境

    Args:
        config: 评估配置
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 配置CUDA
    if config.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # 设置线程数
    if config.num_workers > 0:
        torch.set_num_threads(config.num_workers)


def run_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    """
    运行评估流程

    Args:
        config: 评估配置

    Returns:
        评估结果
    """
    try:
        # 创建评估器
        evaluator = Evaluator(config)

        # 运行评估
        results = evaluator.evaluate()

        # 打印摘要
        print("\nEvaluation Results:")
        print("=" * 50)

        # 分类结果
        print("\nClassification Metrics:")
        for name, value in results['classification'].items():
            print(f"{name:15s}: {value:.4f}")

        # 分割结果(如果有)
        if 'segmentation' in results:
            print("\nSegmentation Metrics:")
            for name, value in results['segmentation'].items():
                print(f"{name:15s}: {value:.4f}")

        # 性能统计
        print("\nPerformance Statistics:")
        if 'memory_stats' in results:
            for name, value in results['memory_stats'].items():
                print(f"{name:20s}: {value:.1f}")

        print(f"\nResults saved to: {config.output_dir}")

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()

        # 创建配置
        config = EvaluationConfig.from_args(args)

        # 验证配置
        config.validate()

        # 配置环境
        setup_environment(config)

        # 打印配置信息
        logger.info("Starting evaluation with configuration:")
        logger.info(f"Config file: {config.config_path}")
        logger.info(f"Checkpoint: {config.checkpoint_path}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Device: {config.device}")

        # 记录开始时间
        start_time = time.time()

        # 运行评估
        results = run_evaluation(config)

        # 记录总用时
        total_time = time.time() - start_time
        logger.info(f"\nTotal evaluation time: {total_time:.2f}s")

        # 返回成功状态码
        return 0

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 130  # 标准的SIGINT返回码

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if isinstance(e, (ValueError, ConfigError)):
            return 1  # 配置错误
        if isinstance(e, (DatasetError, CheckpointError)):
            return 2  # 数据或模型错误
        return 3  # 其他错误

    finally:
        # 清理资源
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # 设置更好的异常展示
    import sys
    import traceback


    def excepthook(exc_type, exc_value, exc_traceback):
        """自定义异常处理"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("An unhandled exception occurred:")
        logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))


    sys.excepthook = excepthook

    # 运行主函数
    exit_code = main()
    sys.exit(exit_code)