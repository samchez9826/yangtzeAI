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
from .preprocessor import ImagePreprocessor
from .augmentation import AugmentationPipeline


class ForgeryDataset(Dataset):
    """图像篡改检测数据集基类"""

    def __init__(self,
                 config_path: str,
                 split: str = 'train',
                 transform_config: Optional[str] = None):
        """
        初始化数据集
        Args:
            config_path: 数据集配置文件路径
            split: 数据集分割 ['train', 'val', 'test']
            transform_config: 数据增强配置文件路径
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['datasets']

        self.split = split
        self.samples = []  # 存储样本路径
        self.labels = []  # 存储标签
        self.masks = []  # 存储掩码路径

        # 初始化预处理器和增强器
        self.preprocessor = ImagePreprocessor(config_path)
        if transform_config and split == 'train':
            self.augmentor = AugmentationPipeline(transform_config)
        else:
            self.augmentor = None

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载数据集"""
        raise NotImplementedError("子类必须实现此方法")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        Args:
            idx: 样本索引
        Returns:
            包含图像、标签和掩码的字典
        """
        try:
            # 读取图像
            image_path = self.samples[idx]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

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
            self.logger.error(f"加载样本失败: {str(e)}")
            raise

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重"""
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
        return weights


class CASIADataset(ForgeryDataset):
    """CASIA 2.0数据集加载器"""

    def _load_data(self):
        """加载CASIA 2.0数据集"""
        base_path = Path(self.config['casia'][f'{self.split}_path'])

        # 加载真实图像
        authentic_dir = base_path / self.config['casia']['authentic_dir']
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


class FaceForensicsDataset(ForgeryDataset):
    """FaceForensics++数据集加载器"""

    def _load_data(self):
        """加载FaceForensics++数据集"""
        base_path = Path(self.config['faceforensics'][f'{self.split}_path'])

        # 加载真实视频帧
        original_dir = base_path / 'original'
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
            for compression in self.config['faceforensics']['compression_levels']:
                compression_dir = method_dir / compression
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


class MixedDataset(ForgeryDataset):
    """混合数据集加载器"""

    def __init__(self,
                 config_path: str,
                 datasets: List[str],
                 split: str = 'train',
                 transform_config: Optional[str] = None):
        """
        初始化混合数据集
        Args:
            config_path: 配置文件路径
            datasets: 要混合的数据集列表
            split: 数据集分割
            transform_config: 数据增强配置
        """
        self.dataset_names = datasets
        super().__init__(config_path, split, transform_config)

    def _load_data(self):
        """加载多个数据集"""
        for dataset_name in self.dataset_names:
            if dataset_name.lower() == 'casia':
                dataset = CASIADataset(
                    self.config_path,
                    self.split,
                    self.transform_config
                )
            elif dataset_name.lower() == 'faceforensics':
                dataset = FaceForensicsDataset(
                    self.config_path,
                    self.split,
                    self.transform_config
                )
            else:
                raise ValueError(f"不支持的数据集: {dataset_name}")

            # 合并数据
            self.samples.extend(dataset.samples)
            self.labels.extend(dataset.labels)
            self.masks.extend(dataset.masks)

        # 打乱数据
        if self.split == 'train':
            combined = list(zip(self.samples, self.labels, self.masks))
            random.shuffle(combined)
            self.samples, self.labels, self.masks = zip(*combined)

        # 数据集统计
        self.logger.info(f"加载混合数据集 {self.split}集:")
        self.logger.info(f"真实样本: {self.labels.count(0)}张")
        self.logger.info(f"篡改样本: {self.labels.count(1)}张")


def create_dataloader(dataset: ForgeryDataset,
                      batch_size: int,
                      num_workers: int = 4,
                      shuffle: bool = None) -> DataLoader:
    """
    创建数据加载器
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
    Returns:
        DataLoader实例
    """
    # 默认训练集打乱、验证和测试集不打乱
    if shuffle is None:
        shuffle = dataset.split == 'train'

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=dataset.split == 'train'
    )