from __future__ import annotations

import cv2
import numpy as np
import torch
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import albumentations as A
from scipy import ndimage
import kornia
from PIL import Image
import numba
from functools import lru_cache


# 自定义异常类
class AugmentationError(Exception):
    """增强处理基础异常类"""
    pass


class ConfigurationError(AugmentationError):
    """配置相关错误"""
    pass


class QualityError(AugmentationError):
    """质量控制相关错误"""
    pass


# 配置数据类
@dataclass
class AugmentationConfig:
    """增强配置数据类"""
    geometric: Dict[str, Any]
    photometric: Dict[str, Any]
    noise: Dict[str, Any]
    blur: Dict[str, Any]
    compression: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'AugmentationConfig':
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['augmentation']
        return cls(**config)


# 增强策略接口
class AugmentationStrategy(Protocol):
    """增强策略接口"""
    def augment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        ...

class ImageAugmentor:
    """高级图像数据增强器"""

    __slots__ = ('config', '_device', '_logger', '_basic_transform',
                 '_advanced_transform', '_expert_transform',
                 '_quality_evaluator', '_noise_generator')

    def __init__(self, config_path: str):
        """
        初始化数据增强器
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = AugmentationConfig.from_yaml(config_path)

        # 初始化设备
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化日志
        self._logger = self._setup_logger()

        # 延迟初始化的组件
        self._basic_transform = None
        self._advanced_transform = None
        self._expert_transform = None
        self._quality_evaluator = None
        self._noise_generator = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    @lru_cache(maxsize=None)
    def basic_transform(self) -> A.Compose:
        if self._basic_transform is None:
            self._basic_transform = self._build_basic_transform()
        return self._basic_transform

    @property
    @lru_cache(maxsize=None)
    def advanced_transform(self) -> A.Compose:
        if self._advanced_transform is None:
            self._advanced_transform = self._build_advanced_transform()
        return self._advanced_transform

    @property
    @lru_cache(maxsize=None)
    def expert_transform(self) -> A.Compose:
        if self._expert_transform is None:
            self._expert_transform = self._build_expert_transform()
        return self._expert_transform

    @property
    def quality_evaluator(self) -> ImageQualityEvaluator:
        if self._quality_evaluator is None:
            self._quality_evaluator = ImageQualityEvaluator()
        return self._quality_evaluator

    @property
    def noise_generator(self) -> torch.nn.Module:
        if self._noise_generator is None:
            self._noise_generator = self._load_noise_generator()
            if hasattr(torch, 'compile'):
                self._noise_generator = torch.compile(
                    self._noise_generator,
                    mode='reduce-overhead'
                )
        return self._noise_generator

    def _load_noise_generator(self) -> torch.nn.Module:
        """加载噪声生成器"""
        model = NoiseGenerator().to(self.device)
        return model

    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # 文件处理器
            fh = logging.FileHandler('augmentation.log')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    @contextmanager
    def _gpu_memory_manager(self):
        """GPU内存管理器"""
        try:
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()

            yield

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated()
                if current_memory > initial_memory:
                    self.logger.warning(
                        f"GPU内存泄漏: {(current_memory - initial_memory) / 1024 ** 2:.2f}MB"
                    )

    def _validate_input(self, image: np.ndarray) -> None:
        """验证输入图像"""
        if not isinstance(image, np.ndarray):
            raise ValueError("输入必须是numpy数组")
        if image.ndim != 3:
            raise ValueError("图像必须是3通道")
        if image.dtype != np.uint8:
            raise ValueError("图像必须是uint8类型")
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            raise ValueError("图像包含无效值(NaN/Inf)")

    def _validate_output(self, image: np.ndarray) -> None:
        """验证输出图像"""
        self._validate_input(image)  # 基本验证
        if image.shape[2] != 3:
            raise ValueError("输出必须是3通道RGB图像")

    @torch.no_grad()
    def augment(self, image: np.ndarray,
                mask: Optional[np.ndarray] = None,
                level: str = 'basic',
                preserve_quality: bool = True) -> Dict[str, np.ndarray]:
        """
        执行数据增强
        Args:
            image: 输入图像
            mask: 可选掩码
            level: 增强级别 ['basic', 'advanced', 'expert']
            preserve_quality: 是否保持图像质量
        Returns:
            增强后的图像和掩码
        """
        try:
            # 输入验证
            self._validate_input(image)

            # 质量评估
            if preserve_quality:
                initial_quality = self.quality_evaluator.evaluate(image)

            # GPU内存管理
            with self._gpu_memory_manager():
                # 选择增强流水线
                if level == 'basic':
                    transform = self.basic_transform
                elif level == 'advanced':
                    transform = A.Compose([
                        self.basic_transform,
                        self.advanced_transform
                    ])
                elif level == 'expert':
                    transform = A.Compose([
                        self.basic_transform,
                        self.advanced_transform,
                        self.expert_transform
                    ])
                else:
                    raise ConfigurationError(f"不支持的增强级别: {level}")

                # 应用增强
                transformed = transform(image=image, mask=mask)
                result = {
                    'image': transformed['image']
                }

                if mask is not None:
                    result['mask'] = transformed['mask']

                # 质量控制
                if preserve_quality:
                    result['image'] = self._quality_control(
                        image,
                        result['image'],
                        initial_quality
                    )

                # 输出验证
                self._validate_output(result['image'])

                return result

        except Exception as e:
            self.logger.error(f"数据增强失败: {str(e)}")
            raise AugmentationError(f"增强处理失败: {str(e)}")

    def _build_basic_transform(self) -> A.Compose:
        """构建基础数据增强流水线"""
        transforms_list = []

        # 1. 几何变换
        if self.config.geometric['rotate']['enable']:
            transforms_list.extend([
                A.OneOf([
                    A.Rotate(
                        limit=self.config.geometric['rotate']['angle_range'],
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.7
                    ),
                    A.SafeRotate(
                        limit=self.config.geometric['rotate']['angle_range'],
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.3
                    )
                ], p=0.5)
            ])

        if self.config.geometric['flip']['enable']:
            transforms_list.append(
                A.OneOf([
                    A.HorizontalFlip(p=0.7),
                    A.VerticalFlip(p=0.2),
                    A.RandomRotate90(p=0.1)
                ], p=0.5)
            )

        if self.config.geometric['scale']['enable']:
            transforms_list.append(
                A.OneOf([
                    A.RandomScale(
                        scale_limit=self.config.geometric['scale']['range'],
                        interpolation=cv2.INTER_CUBIC,
                        p=0.6
                    ),
                    A.RandomResizedCrop(
                        height=self.config.geometric['scale']['target_size'],
                        width=self.config.geometric['scale']['target_size'],
                        scale=(0.8, 1.2),
                        ratio=(0.9, 1.1),
                        interpolation=cv2.INTER_CUBIC,
                        p=0.4
                    )
                ], p=0.5)
            )

        # 2. 光度变换
        if self.config.photometric['color_transforms']['enable']:
            transforms_list.append(
                A.OneOf([
                    A.ColorJitter(
                        brightness=self.config.photometric['brightness']['range'],
                        contrast=self.config.photometric['contrast']['range'],
                        saturation=self.config.photometric['saturation']['range'],
                        hue=self.config.photometric['hue']['range'],
                        p=0.4
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        brightness_by_max=True,
                        p=0.3
                    )
                ], p=0.5)
            )

        return A.Compose(transforms_list, p=1.0)

    def _build_advanced_transform(self) -> A.Compose:
        """构建高级数据增强流水线"""
        transforms_list = []

        # 1. 高级噪声
        if self.config.noise['advanced']['enable']:
            transforms_list.append(
                A.OneOf([
                    A.GaussNoise(
                        var_limit=self.config.noise['gaussian']['std'],
                        mean=self.config.noise['gaussian']['mean'],
                        per_channel=True,
                        p=0.4
                    ),
                    A.MultiplicativeNoise(
                        multiplier=[0.9, 1.1],
                        per_channel=True,
                        elementwise=True,
                        p=0.3
                    ),
                    A.Lambda(
                        image=lambda x, **kwargs: self._add_sensor_noise(x),
                        p=0.3
                    )
                ], p=0.5)
            )

        # 2. 高级模糊
        if self.config.blur['advanced']['enable']:
            transforms_list.append(
                A.OneOf([
                    A.GaussianBlur(
                        blur_limit=self.config.blur['gaussian']['kernel_range'],
                        p=0.3
                    ),
                    A.MotionBlur(
                        blur_limit=self.config.blur['motion']['kernel_range'],
                        p=0.3
                    ),
                    A.GlassBlur(
                        sigma=0.7,
                        max_delta=4,
                        iterations=2,
                        p=0.2
                    ),
                    A.Lambda(
                        image=lambda x, **kwargs: self._simulate_sensor_effects(x),
                        p=0.2
                    )
                ], p=0.5)
            )

        # 3. 高级压缩和伪影
        if self.config.compression['advanced']['enable']:
            transforms_list.append(
                A.OneOf([
                    A.ImageCompression(
                        quality_lower=self.config.compression['jpeg']['quality_range'][0],
                        quality_upper=self.config.compression['jpeg']['quality_range'][1],
                        compression_type=A.ImageCompression.JPEG,
                        p=0.4
                    ),
                    A.Lambda(
                        image=lambda x, **kwargs: self._add_compression_artifacts(x),
                        p=0.3
                    ),
                    A.Downscale(
                        scale_min=0.25,
                        scale_max=0.5,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.3
                    )
                ], p=0.5)
            )

        return A.Compose(transforms_list, p=0.7)

    def _build_expert_transform(self) -> A.Compose:
        """构建专家级数据增强流水线"""
        transforms_list = []

        # 1. 风格迁移增强
        transforms_list.append(
            A.OneOf([
                A.Lambda(
                    image=lambda x, **kwargs: self._apply_style_transfer(x),
                    p=0.5
                ),
                A.Lambda(
                    image=lambda x, **kwargs: self._apply_neural_filters(x),
                    p=0.5
                )
            ], p=0.3)
        )

        # 2. 对抗性噪声
        transforms_list.append(
            A.OneOf([
                A.Lambda(
                    image=lambda x, **kwargs: self._add_adversarial_noise(x),
                    p=0.5
                ),
                A.Lambda(
                    image=lambda x, **kwargs: self.noise_generator(
                        torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                    ).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255,
                    p=0.5
                )
            ], p=0.3)
        )

        # 3. 真实环境模拟
        transforms_list.append(
            A.OneOf([
                A.Lambda(
                    image=lambda x, **kwargs: self._simulate_low_light(x),
                    p=0.4
                ),
                A.Lambda(
                    image=lambda x, **kwargs: self._simulate_weather_effects(x),
                    p=0.3
                ),
                A.Lambda(
                    image=lambda x, **kwargs: self._simulate_sensor_effects(x),
                    p=0.3
                )
            ], p=0.3)
        )

        return A.Compose(transforms_list, p=0.5)

    def _quality_control(self, original: np.ndarray,
                         augmented: np.ndarray,
                         target_quality: float) -> np.ndarray:
        """质量控制"""
        try:
            current_quality = self.quality_evaluator.evaluate(augmented)

            if current_quality < target_quality * 0.8:  # 允许20%的质量下降
                self.logger.warning(
                    f"质量低于阈值: {current_quality:.3f} < {target_quality:.3f}"
                )
                return self._restore_quality(augmented, target_quality)

            return augmented
        except Exception as e:
            self.logger.error(f"质量控制失败: {str(e)}")
            return original

    @torch.no_grad()
    def _apply_style_transfer(self, image: np.ndarray) -> np.ndarray:
        """应用风格迁移"""
        try:
            x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(self.device)

            adain = kornia.enhance.AdaptiveInstanceNormalization()
            styled = adain(x)

            styled = (styled.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
            return np.clip(styled, 0, 255).astype(np.uint8)

        except Exception as e:
            self.logger.error(f"风格迁移失败: {str(e)}")
            return image

    @torch.no_grad()
    def _apply_neural_filters(self, image: np.ndarray) -> np.ndarray:
        """应用神经网络滤镜效果"""
        try:
            x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(self.device)

            filters = [
                kornia.filters.GaussianBlur2d((5, 5), (1.5, 1.5)),
                kornia.filters.Laplacian(3),
                kornia.filters.Sobel(),
                kornia.filters.UnsharpMask()
            ]

            filter_idx = np.random.randint(len(filters))
            filtered = filters[filter_idx](x)

            filtered = (filtered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
            return np.clip(filtered, 0, 255).astype(np.uint8)

        except Exception as e:
            self.logger.error(f"神经滤镜应用失败: {str(e)}")
            return image

    def _add_adversarial_noise(self, image: np.ndarray) -> np.ndarray:
        """添加对抗性噪声"""
        try:
            x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(self.device)
            x.requires_grad = True

            epsilon = 2.0 / 255.0
            alpha = 0.5 / 255.0

            with torch.enable_grad():
                noise = torch.zeros_like(x)
                for _ in range(3):  # 减少迭代次数以提高性能
                    x_adv = x + noise
                    loss = torch.nn.functional.mse_loss(x_adv, x)
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    noise = noise + alpha * grad.sign()
                    noise = torch.clamp(noise, -epsilon, epsilon)

            x_adv = torch.clamp(x + noise, 0, 1)
            x_adv = (x_adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255)
            return np.clip(x_adv, 0, 255).astype(np.uint8)

        except Exception as e:
            self.logger.error(f"对抗性噪声添加失败: {str(e)}")
            return image

    def _simulate_low_light(self, image: np.ndarray) -> np.ndarray:
        """模拟低光照条件"""
        try:
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycbcr)

            gamma = np.random.uniform(2.0, 3.0)
            y_dark = ((y / 255.0) ** gamma * 255.0).astype(np.uint8)

            noise_sigma = np.random.uniform(5, 15)
            noise = np.random.normal(0, noise_sigma, y_dark.shape).astype(np.uint8)
            y_noisy = np.clip(y_dark + noise, 0, 255).astype(np.uint8)

            low_light = cv2.merge([y_noisy, cr, cb])
            return cv2.cvtColor(low_light, cv2.COLOR_YCrCb2BGR)

        except Exception as e:
            self.logger.error(f"低光照模拟失败: {str(e)}")
            return image

    def _simulate_weather_effects(self, image: np.ndarray) -> np.ndarray:
        """模拟天气效果"""
        effects = {
            'rain': self._add_rain,
            'snow': self._add_snow,
            'fog': self._add_fog
        }
        effect = np.random.choice(list(effects.keys()))
        return effects[effect](image)

    def _add_rain(self, image: np.ndarray) -> np.ndarray:
        """添加雨滴效果"""
        try:
            h, w = image.shape[:2]
            rain_layer = np.zeros_like(image)

            # 使用numpy向量化操作生成雨滴参数
            n_drops = np.random.randint(100, 1000)
            x = np.random.randint(0, w, n_drops)
            y = np.random.randint(0, h, n_drops)
            lengths = np.random.randint(5, 15, n_drops)
            angles = np.random.uniform(70, 110, n_drops)

            radians = np.deg2rad(angles)
            x2 = (x + lengths * np.cos(radians)).astype(int)
            y2 = (y + lengths * np.sin(radians)).astype(int)

            # 批量绘制雨滴
            for i in range(n_drops):
                cv2.line(rain_layer, (x[i], y[i]), (x2[i], y2[i]), (255, 255, 255), 1)

            rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
            alpha = np.random.uniform(0.6, 0.8)
            return cv2.addWeighted(image, 1, rain_layer, alpha, 0)

        except Exception as e:
            self.logger.error(f"雨滴效果添加失败: {str(e)}")
            return image

    def _add_snow(self, image: np.ndarray) -> np.ndarray:
        """添加雪花效果"""
        try:
            h, w = image.shape[:2]
            snow_layer = np.zeros_like(image)

            # 使用numpy向量化操作生成雪花参数
            n_flakes = np.random.randint(300, 1500)
            x = np.random.randint(0, w, n_flakes)
            y = np.random.randint(0, h, n_flakes)
            sizes = np.random.randint(1, 4, n_flakes)

            # 批量绘制雪花
            for i in range(n_flakes):
                cv2.circle(snow_layer, (x[i], y[i]), sizes[i], (255, 255, 255), -1)

            snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)
            alpha = np.random.uniform(0.4, 0.6)
            return cv2.addWeighted(image, 1, snow_layer, alpha, 0)

        except Exception as e:
            self.logger.error(f"雪花效果添加失败: {str(e)}")
            return image

    @staticmethod
    @numba.jit(nopython=True)
    def _create_depth_map(h: int, w: int) -> np.ndarray:
        """使用numba加速深度图生成"""
        depth_map = np.zeros((h, w))
        center_y, center_x = h // 2, w // 2
        max_dist = np.sqrt(h ** 2 + w ** 2) / 2

        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                depth_map[i, j] = np.clip(distance / max_dist, 0, 1)

        return depth_map

    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        """添加雾效果"""
        try:
            h, w = image.shape[:2]
            fog_effect = np.full_like(image, 255)

            depth_map = self._create_depth_map(h, w)
            depth_map = np.dstack([depth_map] * 3)

            alpha = np.random.uniform(0.4, 0.7)
            foggy = image * (1 - depth_map * alpha) + fog_effect * (depth_map * alpha)
            return np.clip(foggy, 0, 255).astype(np.uint8)

        except Exception as e:
            self.logger.error(f"雾效果添加失败: {str(e)}")
            return image

    def _simulate_sensor_effects(self, image: np.ndarray) -> np.ndarray:
        """模拟相机传感器效果"""
        try:
            # 色温调整
            temperature = np.random.uniform(3000, 7000)
            image = self._adjust_color_temperature(image, temperature)

            # 镜头畸变
            image = self._add_lens_distortion(image)

            # 色差
            image = self._add_chromatic_aberration(image)

            # 传感器噪声
            image = self._add_sensor_noise(image)

            return image

        except Exception as e:
            self.logger.error(f"传感器效果模拟失败: {str(e)}")
            return image

    def _add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """添加压缩伪影"""
        try:
            # 转换到YCrCb空间
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # 对色度通道进行下采样
            cr_down = cv2.resize(cr, None, fx=0.5, fy=0.5)
            cb_down = cv2.resize(cb, None, fx=0.5, fy=0.5)

            # 使用加速的DCT量化
            quality = np.random.randint(60, 90)
            y_compressed = self._fast_dct_quantize(y, quality)
            cr_compressed = self._fast_dct_quantize(cr_down, quality)
            cb_compressed = self._fast_dct_quantize(cb_down, quality)

            # 色度上采样
            cr_up = cv2.resize(cr_compressed, (cr.shape[1], cr.shape[0]))
            cb_up = cv2.resize(cb_compressed, (cb.shape[1], cb.shape[0]))

            # 合并通道并添加伪影
            ycrcb_compressed = cv2.merge([y_compressed, cr_up, cb_up])
            bgr_compressed = cv2.cvtColor(ycrcb_compressed, cv2.COLOR_YCrCb2BGR)

            return bgr_compressed

        except Exception as e:
            self.logger.error(f"压缩伪影添加失败: {str(e)}")
            return image

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _fast_dct_quantize(channel: np.ndarray, quality: int) -> np.ndarray:
        """使用numba加速的DCT量化计算"""
        h, w = channel.shape
        block_size = 8
        quantized = np.zeros_like(channel)

        # 标准JPEG量化表
        q_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)

        # 根据质量调整量化表
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        q_table = np.clip(q_table * scale / 100, 1, 255)

        for i in numba.prange(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = channel[i:i + block_size, j:j + block_size].astype(np.float32)
                # 使用FFT实现DCT
                dct_block = np.fft.fft2(block)[:block_size, :block_size]
                quantized_block = np.round(dct_block / q_table) * q_table
                idct_block = np.real(np.fft.ifft2(quantized_block))
                quantized[i:i + block_size, j:j + block_size] = idct_block

        return quantized.astype(np.uint8)


class NoiseGenerator(torch.nn.Module):
    """生成逼真的图像噪声的神经网络"""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 3, 3, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        noise = self.decoder(features)
        return noise


class ImageQualityEvaluator:
    """评估图像质量的评估器"""

    def __init__(self):
        self.metrics = {
            'sharpness': self._evaluate_sharpness,
            'noise': self._evaluate_noise,
            'compression': self._evaluate_compression,
            'contrast': self._evaluate_contrast,
            'color': self._evaluate_color
        }

    def evaluate(self, image: np.ndarray) -> float:
        """评估图像整体质量"""
        scores = {}
        for name, metric in self.metrics.items():
            try:
                scores[name] = metric(image)
            except Exception as e:
                logging.error(f"{name}评估失败: {str(e)}")
                scores[name] = 0.0

        weights = {
            'sharpness': 0.3,
            'noise': 0.2,
            'compression': 0.2,
            'contrast': 0.15,
            'color': 0.15
        }

        weighted_score = sum(scores[k] * weights[k] for k in scores)
        return np.clip(weighted_score, 0, 1)

    @staticmethod
    def _evaluate_sharpness(image: np.ndarray) -> float:
        """评估图像清晰度"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx ** 2 + sobely ** 2).mean()

        score = (laplacian + gradient) / 2
        return np.clip(score / 1000, 0, 1)

    @staticmethod
    def _evaluate_noise(image: np.ndarray) -> float:
        """评估图像噪声水平"""
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y = ycbcr[..., 0]

        kernel_size = 5
        local_mean = cv2.blur(y.astype(float), (kernel_size, kernel_size))
        local_var = cv2.blur(y.astype(float) ** 2, (kernel_size, kernel_size)) - local_mean ** 2

        noise_level = np.sqrt(np.mean(local_var))
        return 1 - np.clip(noise_level / 50, 0, 1)

    @staticmethod
    def _evaluate_compression(image: np.ndarray) -> float:
        """评估压缩质量"""

        def block_effect(channel):
            h, w = channel.shape
            block_size = 8

            h_diff = np.abs(channel[:, block_size - 1:-1:block_size] -
                            channel[:, block_size:-1:block_size]).mean()
            v_diff = np.abs(channel[block_size - 1:-1:block_size, :] -
                            channel[block_size:-1:block_size, :]).mean()

            return (h_diff + v_diff) / 2

        b, g, r = cv2.split(image)
        block_scores = [block_effect(c) for c in [b, g, r]]
        compression_score = 1 - np.mean(block_scores) / 20

        return np.clip(compression_score, 0, 1)

    @staticmethod
    def _evaluate_contrast(image: np.ndarray) -> float:
        """评估对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l = lab[..., 0]

        p1, p99 = np.percentile(l, (1, 99))
        contrast_range = p99 - p1
        score = contrast_range / 255

        optimal_contrast = 0.6
        penalty = 1 - abs(score - optimal_contrast)

        return np.clip(penalty, 0, 1)

    @staticmethod
    def _evaluate_color(image: np.ndarray) -> float:
        """评估色彩质量"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1]
        v = hsv[..., 2]

        s_mean = np.mean(s)
        v_mean = np.mean(v)
        s_std = np.std(s)
        v_std = np.std(v)

        optimal_s_mean = 128
        optimal_v_mean = 128
        optimal_std = 50

        s_score = 1 - abs(s_mean - optimal_s_mean) / optimal_s_mean
        v_score = 1 - abs(v_mean - optimal_v_mean) / optimal_v_mean
        std_score = 1 - abs(np.mean([s_std, v_std]) - optimal_std) / optimal_std

        return np.clip((s_score + v_score + std_score) / 3, 0, 1)


def create_augmentation_pipeline(config_path: str) -> ImageAugmentor:
    """创建数据增强器实例"""
    return ImageAugmentor(config_path)