import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml
import random
from functools import lru_cache
from queue import Queue
from threading import Thread
import time
import psutil
from .preprocessor import ImagePreprocessor
from .augmentation import AugmentationPipeline


class DatasetError(Exception):
    """数据集相关错误"""
    pass


class DatasetConfig:
    """数据集配置类"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['datasets']

        # 验证配置
        self.validate_config()

    def validate_config(self):
        """验证配置有效性"""
        required_fields = ['casia', 'faceforensics']
        for field in required_fields:
            if field not in self.config:
                raise DatasetError(f"配置缺少必需字段: {field}")


class ForgeryDataset(Dataset):
    """图像篡改检测数据集基类"""

    def __init__(self,
                 config_path: str,
                 split: str = 'train',
                 transform_config: Optional[str] = None,
                 cache_size: int = 1000,
                 preload_size: int = 100):
        """
        初始化数据集
        Args:
            config_path: 数据集配置文件路径
            split: 数据集分割 ['train', 'val', 'test']
            transform_config: 数据增强配置文件路径
            cache_size: 样本缓存大小
            preload_size: 预加载队列大小
        """
        # 加载配置
        self.config = DatasetConfig(config_path).config

        self.split = split
        self.samples = []  # 存储样本路径
        self.labels = []  # 存储标签
        self.masks = []  # 存储掩码路径

        # 性能监控
        self.load_times = []
        self._monitor_memory()

        # 初始化预处理器和增强器
        self.preprocessor = ImagePreprocessor(config_path)
        if transform_config and split == 'train':
            self.augmentor = AugmentationPipeline(transform_config)
        else:
            self.augmentor = None

        # 设置日志
        self._setup_logger()

        # 初始化缓存
        self.cache_size = cache_size
        self.sample_cache = {}

        # 初始化预加载
        self.preload_size = preload_size
        self.preload_queue = Queue(maxsize=preload_size)
        self._start_preload()

        # 加载数据
        self._load_data()

        # 验证数据集
        if not self.validate_dataset():
            raise DatasetError("数据集验证失败")

    def _setup_logger(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # 文件处理器
            fh = logging.FileHandler('dataset.log')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def _monitor_memory(self):
        """监控内存使用"""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        self.logger.info(f"当前内存使用: {mem:.2f}MB")

    def _start_preload(self):
        """启动预加载线程"""
        self.preload_thread = Thread(target=self._preload_samples)
        self.preload_thread.daemon = True
        self.preload_thread.start()

    def _preload_samples(self):
        """异步预加载样本"""
        try:
            while True:
                if self.preload_queue.qsize() < self.preload_size:
                    # 随机选择未缓存的样本进行预加载
                    idx = random.randint(0, len(self) - 1)
                    if idx not in self.sample_cache:
                        sample = self._load_sample(idx)
                        self.preload_queue.put((idx, sample))
                time.sleep(0.1)  # 避免过度占用CPU
        except Exception as e:
            self.logger.error(f"预加载失败: {str(e)}")

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """加载单个样本"""
        try:
            # 读取图像
            image_path = self.samples[idx]
            image = cv2.imread(str(image_path))
            if image is None:
                raise DatasetError(f"无法读取图像: {image_path}")

            # 读取掩码(如果存在)
            mask = None
            if self.masks:
                mask_path = self.masks[idx]
                if os.path.exists(str(mask_path)):
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # 预处理
            processed = self.preprocessor.preprocess_image(image)

            # 数据增强
            if self.augmentor and self.split == 'train':
                augmented = self.augmentor.augment_batch(
                    [processed['resized']],
                    [mask] if mask is not None else None
                )
                processed['resized'] = augmented['images'][0]
                if mask is not None:
                    mask = augmented['masks'][0]

            # 准备输出
            sample = {
                'image': torch.from_numpy(processed['resized']).float(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }

            # 添加掩码(如果存在)
            if mask is not None:
                sample['mask'] = torch.from_numpy(mask).float()

            # 添加图像质量特征
            if 'quality_metrics' in processed:
                quality_features = []
                for metric_name in ['sharpness', 'noise_level', 'jpeg_quality']:
                    if metric_name in processed['quality_metrics']:
                        quality_features.append(
                            processed['quality_metrics'][metric_name]
                        )
                sample['quality_features'] = torch.tensor(
                    quality_features,
                    dtype=torch.float32
                )

            # 添加纹理特征
            if 'lbp' in processed:
                sample['texture_features'] = torch.from_numpy(
                    processed['lbp']
                ).float()

            return sample

        except Exception as e:
            self.logger.error(f"样本加载失败: {str(e)}")
            raise

    def _load_data(self):
        """加载数据集"""
        raise NotImplementedError("子类必须实现此方法")

    def validate_dataset(self) -> bool:
        """验证数据集完整性"""
        try:
            # 检查文件是否存在
            for path in self.samples:
                if not os.path.exists(path):
                    self.logger.error(f"找不到文件: {path}")
                    return False

            if self.masks:
                for path in self.masks:
                    if not os.path.exists(path):
                        self.logger.error(f"找不到掩码: {path}")
                        return False

            # 检查样本数
            if len(self.samples) == 0:
                self.logger.error("数据集为空")
                return False

            if len(self.samples) != len(self.labels):
                self.logger.error("样本数量与标签数量不匹配")
                return False

            if self.masks and len(self.samples) != len(self.masks):
                self.logger.error("样本数量与掩码数量不匹配")
                return False

            # 检查类别平衡性
            class_counts = np.bincount(self.labels)
            if len(class_counts) < 2:
                self.logger.error("数据集缺少某些类别")
                return False

            imbalance_ratio = np.max(class_counts) / np.min(class_counts)
            if imbalance_ratio > 10:
                self.logger.warning(f"数据集类别严重不平衡, 比例 {imbalance_ratio:.2f}")

            return True

        except Exception as e:
            self.logger.error(f"数据集验证失败: {str(e)}")
            return False

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本,带缓存功能
        Args:
            idx: 样本索引
        Returns:
            包含图像、标签和掩码的字典
        """
        try:
            start_time = time.time()

            # 尝试从缓存获取
            if idx in self.sample_cache:
                sample = self.sample_cache[idx]
            else:
                # 尝试从预加载队列获取
                for _ in range(self.preload_queue.qsize()):
                    cache_idx, cache_sample = self.preload_queue.get()
                    self.sample_cache[cache_idx] = cache_sample
                    if cache_idx == idx:
                        sample = cache_sample
                        break
                else:
                    # 如果没有预加载,直接加载
                    sample = self._load_sample(idx)

                    # 更新缓存
                    if len(self.sample_cache) >= self.cache_size:
                        # LRU策略: 移除最早的缓存
                        oldest_key = next(iter(self.sample_cache))
                        del self.sample_cache[oldest_key]
                    self.sample_cache[idx] = sample

            # 记录加载时间
            load_time = time.time() - start_time
            self.load_times.append(load_time)

            # 监控加载性能
            if len(self.load_times) >= 100:
                avg_time = np.mean(self.load_times)
                if avg_time > 0.1:  # 100ms
                    self.logger.warning(f"样本加载性能警告: 平均耗时 {avg_time * 1000:.2f}ms")
                self.load_times = []

            return sample

        except Exception as e:
            self.logger.error(f"获取样本失败: {str(e)}")
            raise DatasetError(f"获取样本失败: {str(e)}")

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重"""
        try:
            class_counts = np.bincount(self.labels)
            total = len(self.labels)
            weights = torch.FloatTensor(total / (len(class_counts) * class_counts))

            # 检查权重是否合理
            if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
                self.logger.warning("类别权重包含无效值,使用均匀权重")
                return torch.ones(len(class_counts), dtype=torch.float)

            return weights

        except Exception as e:
            self.logger.error(f"计算类别权重失败: {str(e)}")
            return torch.ones(len(np.unique(self.labels)), dtype=torch.float)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.samples),
            'class_distribution': np.bincount(self.labels).tolist(),
            'has_masks': bool(self.masks),
        }

        # 计算图像尺寸统计
        if len(self.samples) > 0:
            sample_image = cv2.imread(str(self.samples[0]))
            if sample_image is not None:
                stats['image_height'] = sample_image.shape[0]
                stats['image_width'] = sample_image.shape[1]
                stats['image_channels'] = sample_image.shape[2]

        return stats


class CASIADataset(ForgeryDataset):
    """CASIA 2.0数据集加载器"""

    def _load_data(self):
        """加载CASIA 2.0数据集"""
        try:
            base_path = Path(self.config['casia'][f'{self.split}_path'])
            if not base_path.exists():
                raise DatasetError(f"数据集路径不存在: {base_path}")

            # 加载真实图像
            authentic_dir = base_path / self.config['casia']['authentic_dir']
            if not authentic_dir.exists():
                raise DatasetError(f"真实图像目录不存在: {authentic_dir}")

            for img_path in authentic_dir.glob('*.jpg'):
                self.samples.append(img_path)
                self.labels.append(0)  # 0表示真实图像

                # 对应的掩码(全黑图像)
                mask_path = Path(str(img_path).replace(
                    self.config['casia']['authentic_dir'],
                    self.config['casia']['mask_dir']
                )).with_suffix('.png')
                self.masks.append(mask_path)

            # 加载篡改图像
            tampered_dir = base_path / self.config['casia']['tampered_dir']
            if not tampered_dir.exists():
                raise DatasetError(f"篡改图像目录不存在: {tampered_dir}")

            for img_path in tampered_dir.glob('*.jpg'):
                self.samples.append(img_path)
                self.labels.append(1)  # 1表示篡改图像

                # 对应的掩码
                mask_path = Path(str(img_path).replace(
                    self.config['casia']['tampered_dir'],
                    self.config['casia']['mask_dir']
                )).with_suffix('.png')
                self.masks.append(mask_path)

            # 数据集统计
            self.logger.info(f"加载CASIA {self.split}集:")
            self.logger.info(f"真实图像: {self.labels.count(0)}张")
            self.logger.info(f"篡改图像: {self.labels.count(1)}张")

            # 检查数据集大小是否合理
            if len(self.samples) < 100:
                self.logger.warning(f"数据集样本数量过少: {len(self.samples)}")

        except Exception as e:
            self.logger.error(f"加载CASIA数据集失败: {str(e)}")
            raise DatasetError(f"加载CASIA数据集失败: {str(e)}")


class FaceForensicsDataset(ForgeryDataset):
    """FaceForensics++数据集加载器"""

    def _load_data(self):
        """加载FaceForensics++数据集"""
        try:
            base_path = Path(self.config['faceforensics'][f'{self.split}_path'])
            if not base_path.exists():
                raise DatasetError(f"数据集路径不存在: {base_path}")

            # 加载真实视频帧
            original_dir = base_path / 'original'
            if not original_dir.exists():
                raise DatasetError(f"真实帧目录不存在: {original_dir}")

            for img_path in original_dir.glob('*.png'):
                self.samples.append(img_path)
                self.labels.append(0)

                # 对应的掩码(全黑图像)
                mask_path = Path(str(img_path).replace(
                    'original',
                    'masks'
                ))
                self.masks.append(mask_path)

            # 加载各种篡改方法的视频帧
            for method in self.config['faceforensics']['methods']:
                method_dir = base_path / method
                if not method_dir.exists():
                    self.logger.warning(f"篡改方法目录不存在: {method_dir}")
                    continue

                for compression in self.config['faceforensics']['compression_levels']:
                    compression_dir = method_dir / compression
                    if not compression_dir.exists():
                        self.logger.warning(f"压缩等级目录不存在: {compression_dir}")
                        continue

                    for img_path in compression_dir.glob('*.png'):
                        self.samples.append(img_path)
                        self.labels.append(1)

                        # 对应的掩码
                        mask_path = Path(str(img_path).replace(
                            method,
                            'masks'
                        ))
                        self.masks.append(mask_path)

            # 数据集统计
            self.logger.info(f"加载FaceForensics++ {self.split}集:")
            self.logger.info(f"真实帧: {self.labels.count(0)}张")
            self.logger.info(f"篡改帧: {self.labels.count(1)}张")

            # 检查是否所有篡改方法都有样本
            method_counts = {}
            for path in self.samples:
                for method in self.config['faceforensics']['methods']:
                    if method in str(path):
                        method_counts[method] = method_counts.get(method, 0) + 1
                        break

            for method in self.config['faceforensics']['methods']:
                if method not in method_counts:
                    self.logger.warning(f"篡改方法 {method} 没有样本")
                else:
                    self.logger.info(f"{method}: {method_counts[method]}张")

        except Exception as e:
            self.logger.error(f"加载FaceForensics++数据集失败: {str(e)}")
            raise DatasetError(f"加载FaceForensics++数据集失败: {str(e)}")


class MixedDataset(ForgeryDataset):
    """混合数据集加载器"""

    def __init__(self,
                 config_path: str,
                 datasets: List[str],
                 split: str = 'train',
                 transform_config: Optional[str] = None,
                 cache_size: int = 1000,
                 preload_size: int = 100):
        """
        初始化混合数据集
        Args:
            config_path: 配置文件路径
            datasets: 要混合的数据集列表
            split: 数据集分割
            transform_config: 数据增强配置
            cache_size: 缓存大小
            preload_size: 预加载大小
        """
        self.dataset_names = datasets
        super().__init__(config_path, split, transform_config,
                         cache_size, preload_size)

    def _load_data(self):
        """加载多个数据集"""
        try:
            dataset_loaders = {
                'casia': CASIADataset,
                'faceforensics': FaceForensicsDataset
            }

            total_samples = {name: 0 for name in self.dataset_names}

            for dataset_name in self.dataset_names:
                dataset_name = dataset_name.lower()
                if dataset_name not in dataset_loaders:
                    raise DatasetError(f"不支持的数据集: {dataset_name}")

                # 创建数据集实例
                dataset = dataset_loaders[dataset_name](
                    self.config_path,
                    self.split,
                    self.transform_config
                )

                # 合并数据
                self.samples.extend(dataset.samples)
                self.labels.extend(dataset.labels)
                self.masks.extend(dataset.masks)

                total_samples[dataset_name] = len(dataset.samples)

            # 打乱数据
            if self.split == 'train':
                combined = list(zip(self.samples, self.labels, self.masks))
                random.shuffle(combined)
                self.samples, self.labels, self.masks = zip(*combined)
                self.samples = list(self.samples)
                self.labels = list(self.labels)
                self.masks = list(self.masks)

            # 数据集统计
            self.logger.info(f"加载混合数据集 {self.split}集:")
            for name, count in total_samples.items():
                self.logger.info(f"{name}: {count}张")
            self.logger.info(f"真实样本: {self.labels.count(0)}张")
            self.logger.info(f"篡改样本: {self.labels.count(1)}张")

            # 检查数据比例
            for name, count in total_samples.items():
                ratio = count / len(self.samples)
                if ratio < 0.1:
                    self.logger.warning(
                        f"数据集 {name} 占比过小: {ratio:.2%}"
                    )
                elif ratio > 0.8:
                    self.logger.warning(
                        f"数据集 {name} 占比过大: {ratio:.2%}"
                    )

        except Exception as e:
            self.logger.error(f"加载混合数据集失败: {str(e)}")
            raise DatasetError(f"加载混合数据集失败: {str(e)}")


def create_dataloader(dataset: ForgeryDataset,
                      batch_size: int,
                      num_workers: int = 4,
                      shuffle: bool = None,
                      pin_memory: bool = True,
                      persistent_workers: bool = True) -> DataLoader:
    """
    创建数据加载器
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
        pin_memory: 是否将数据固定在内存中
        persistent_workers: 是否使用持久化工作进程
    Returns:
        DataLoader实例
    """
    try:
        # 默认训练集打乱、验证和测试集不打乱
        if shuffle is None:
            shuffle = dataset.split == 'train'

        # 检查batch_size是否合理
        if batch_size > len(dataset):
            raise DatasetError(f"batch_size({batch_size})大于数据集大小({len(dataset)})")

        # 检查num_workers是否合理
        cpu_count = psutil.cpu_count()
        if num_workers > cpu_count:
            dataset.logger.warning(
                f"num_workers({num_workers})大于CPU核心数({cpu_count}), "
                f"已自动调整为{cpu_count}"
            )
            num_workers = cpu_count

        # 创建dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=dataset.split == 'train'
        )

        # 记录配置
        dataset.logger.info(f"创建DataLoader:")
        dataset.logger.info(f"batch_size: {batch_size}")
        dataset.logger.info(f"num_workers: {num_workers}")
        dataset.logger.info(f"shuffle: {shuffle}")
        dataset.logger.info(f"pin_memory: {pin_memory}")
        dataset.logger.info(f"persistent_workers: {persistent_workers}")

        return loader

    except Exception as e:
        dataset.logger.error(f"创建DataLoader失败: {str(e)}")
        raise DatasetError(f"创建DataLoader失败: {str(e)}")