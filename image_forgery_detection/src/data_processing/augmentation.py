import cv2
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import yaml
import albumentations as A
from scipy import ndimage, signal
import torch
from PIL import Image, ImageEnhance, ImageFilter
import kornia
from skimage import exposure, restoration
import tensorflow as tf


class ImageAugmentor:
    """高级图像数据增强器,实现复杂的数据增强策略"""

    def __init__(self, config_path: str):
        """
        初始化数据增强器
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['augmentation']

        self.geometric_config = self.config['geometric']
        self.photometric_config = self.config['photometric']
        self.noise_config = self.config['noise']
        self.blur_config = self.config['blur']
        self.compression_config = self.config['compression']

        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化日志
        self._setup_logger()

        # 构建增强流水线
        self.basic_transform = self._build_basic_transform()
        self.advanced_transform = self._build_advanced_transform()
        self.expert_transform = self._build_expert_transform()

        # 初始化质量评估器
        self.quality_evaluator = self._init_quality_evaluator()

        # 加载预训练噪声模型
        self.noise_generator = self._load_noise_generator()

    def _setup_logger(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

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

        self.logger.addHandlers([fh, ch])

    def _build_basic_transform(self) -> A.Compose:
        """构建基础数据增强流水线"""
        transforms_list = []

        # 1. 几何变换
        if self.geometric_config['rotate']['enable']:
            transforms_list.extend([
                A.Rotate(limit=self.geometric_config['rotate']['angle_range'],
                         interpolation=cv2.INTER_CUBIC,
                         border_mode=cv2.BORDER_REFLECT_101,
                         p=0.5),

                A.SafeRotate(limit=self.geometric_config['rotate']['angle_range'],
                             interpolation=cv2.INTER_CUBIC,
                             border_mode=cv2.BORDER_REFLECT_101,
                             p=0.3)
            ])

        if self.geometric_config['flip']:
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3)
            ])

        if self.geometric_config['scale']['enable']:
            transforms_list.extend([
                A.RandomScale(scale_limit=self.geometric_config['scale']['range'],
                              interpolation=cv2.INTER_CUBIC,
                              p=0.5),

                A.RandomResizedCrop(
                    height=self.geometric_config['scale']['target_size'],
                    width=self.geometric_config['scale']['target_size'],
                    scale=(0.8, 1.2),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_CUBIC,
                    p=0.3
                )
            ])

        if self.geometric_config['shear']['enable']:
            transforms_list.extend([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                ),

                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.3
                )
            ])

        # 2. 光度变换
        if self.photometric_config['color_transforms']['enable']:
            transforms_list.extend([
                A.ColorJitter(
                    brightness=self.photometric_config['brightness']['range'],
                    contrast=self.photometric_config['contrast']['range'],
                    saturation=self.photometric_config['saturation']['range'],
                    hue=self.photometric_config['hue']['range'],
                    p=0.5
                ),

                A.CLAHE(
                    clip_limit=4.0,
                    tile_grid_size=(8, 8),
                    p=0.3
                ),

                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.3
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    brightness_by_max=True,
                    p=0.3
                ),

                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                )
            ])

        # 3. 色彩空间变换
        if self.photometric_config['color_space']['enable']:
            transforms_list.extend([
                A.ToGray(p=0.2),
                A.ToSepia(p=0.2),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=0.3
                ),
                A.ChannelShuffle(p=0.2)
            ])

        return A.Compose(transforms_list)

    def _build_advanced_transform(self) -> A.Compose:
        """构建高级数据增强流水线"""
        transforms_list = []

        # 1. 高级噪声
        if self.noise_config['advanced']['enable']:
            transforms_list.extend([
                A.GaussNoise(
                    var_limit=self.noise_config['gaussian']['std'],
                    mean=self.noise_config['gaussian']['mean'],
                    per_channel=True,
                    p=0.3
                ),

                A.MultiplicativeNoise(
                    multiplier=[0.9, 1.1],
                    per_channel=True,
                    elementwise=True,
                    p=0.2
                ),

                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=0.3
                )
            ])

        # 2. 高级模糊
        if self.blur_config['advanced']['enable']:
            transforms_list.extend([
                A.GaussianBlur(
                    blur_limit=self.blur_config['gaussian']['kernel_range'],
                    p=0.3
                ),

                A.MotionBlur(
                    blur_limit=self.blur_config['motion']['kernel_range'],
                    p=0.3
                ),

                A.MedianBlur(
                    blur_limit=self.blur_config['median']['kernel_range'],
                    p=0.2
                ),

                A.GlassBlur(
                    sigma=0.7,
                    max_delta=4,
                    iterations=2,
                    p=0.2
                ),

                A.ZoomBlur(
                    max_factor=1.5,
                    step_factor=0.01,
                    p=0.2
                )
            ])

        # 3. 高级压缩和伪影
        if self.compression_config['advanced']['enable']:
            transforms_list.extend([
                A.ImageCompression(
                    quality_lower=self.compression_config['jpeg']['quality_range'][0],
                    quality_upper=self.compression_config['jpeg']['quality_range'][1],
                    compression_type=A.ImageCompression.JPEG,
                    p=0.5
                ),

                A.Downscale(
                    scale_min=0.25,
                    scale_max=0.5,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.3
                ),

                A.Lambda(
                    image=lambda x, **kwargs: self._add_compression_artifacts(x),
                    p=0.3
                )
            ])

        return A.Compose(transforms_list)

    def _build_expert_transform(self) -> A.Compose:
        """构建专家级数据增强流水线"""
        transforms_list = []

        # 1. 风格迁移增强
        transforms_list.extend([
            A.Lambda(
                image=lambda x, **kwargs: self._apply_style_transfer(x),
                p=0.2
            ),

            A.Lambda(
                image=lambda x, **kwargs: self._apply_neural_filters(x),
                p=0.2
            )
        ])

        # 2. 对抗性噪声
        transforms_list.extend([
            A.Lambda(
                image=lambda x, **kwargs: self._add_adversarial_noise(x),
                p=0.2
            ),

            A.Lambda(
                image=lambda x, **kwargs: self._apply_perceptual_attack(x),
                p=0.2
            )
        ])

        # 3. 真实环境模拟
        transforms_list.extend([
            A.Lambda(
                image=lambda x, **kwargs: self._simulate_low_light(x),
                p=0.2
            ),

            A.Lambda(
                image=lambda x, **kwargs: self._simulate_weather_effects(x),
                p=0.2
            ),

            A.Lambda(
                image=lambda x, **kwargs: self._simulate_sensor_effects(x),
                p=0.2
            )
        ])

        return A.Compose(transforms_list)

    def _init_quality_evaluator(self) -> Any:
        """初始化图像质量评估器"""
        return ImageQualityEvaluator()

    def _load_noise_generator(self) -> torch.nn.Module:
        """加载预训练的噪声生成模型"""
        model = NoiseGenerator().to(self.device)
        return model

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
            # 质量评估
            if preserve_quality:
                initial_quality = self.quality_evaluator.evaluate(image)

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
                raise ValueError(f"不支持的增强级别: {level}")

            # 应用增强
            transformed = transform(image=image, mask=mask)
            result = {
                'image': transformed['image']
            }

            if mask is not None:
                result['mask'] = transformed['mask']

            # 质量控制
            if preserve_quality:
                current_quality = self.quality_evaluator.evaluate(result['image'])
                if current_quality < initial_quality * 0.8:  # 允许20%的质量下降
                    result['image'] = self._restore_quality(
                        result['image'],
                        target_quality=initial_quality
                    )

            return result

        except Exception as e:
            self.logger.error(f"数据增强失败: {str(e)}")
            raise

    def _add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """添加真实的压缩伪影"""
        # 色度下采样
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 对色度通道进行下采样和量化
        cr_down = cv2.resize(cr, None, fx=0.5, fy=0.5)
        cb_down = cv2.resize(cb, None, fx=0.5, fy=0.5)

        # DCT变换和量化
        def dct_quantize(channel: np.ndarray, quality: int) -> np.ndarray:
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

            for i in range(0, h - block_size + 1, block_size):
                for j in range(0, w - block_size + 1, block_size):
                    block = channel[i:i + block_size, j:j + block_size].astype(np.float32)
                    dct_block = cv2.dct(block)
                    quantized_block = np.round(dct_block / q_table) * q_table
                    idct_block = cv2.idct(quantized_block)
                    quantized[i:i + block_size, j:j + block_size] = idct_block

            return quantized

        # 对亮度和色度通道进行DCT量化
        quality = np.random.randint(60, 90)  # 随机质量因子
        y_compressed = dct_quantize(y, quality)
        cr_compressed = dct_quantize(cr_down, quality)
        cb_compressed = dct_quantize(cb_down, quality)

        # 色度上采样
        cr_up = cv2.resize(cr_compressed, (cr.shape[1], cr.shape[0]))
        cb_up = cv2.resize(cb_compressed, (cb.shape[1], cb.shape[0]))

        # 合并通道
        ycrcb_compressed = cv2.merge([y_compressed, cr_up, cb_up])
        bgr_compressed = cv2.cvtColor(ycrcb_compressed, cv2.COLOR_YCrCb2BGR)

        # 添加块效应和振铃伪影
        kernel = np.ones((8, 8), np.float32) / 64
        block_effect = cv2.filter2D(bgr_compressed, -1, kernel)

        # 添加振铃效应
        kernel_size = np.random.choice([3, 5])
        sigma = np.random.uniform(0.5, 1.5)
        blurred = cv2.GaussianBlur(block_effect, (kernel_size, kernel_size), sigma)
        ringing = cv2.addWeighted(block_effect, 1.5, blurred, -0.5, 0)

        return np.clip(ringing, 0, 255).astype(np.uint8)

    def _apply_style_transfer(self, image: np.ndarray) -> np.ndarray:
        """应用风格迁移"""
        # 转换为PyTorch张量
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(self.device)

        # 应用自适应实例归一化
        adain = kornia.enhance.AdaptiveInstanceNormalization()
        styled = adain(x)

        # 转回numpy数组
        styled = (styled.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return styled

    def _apply_neural_filters(self, image: np.ndarray) -> np.ndarray:
        """应用神经网络滤镜效果"""
        # 转换为PyTorch张量
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(self.device)

        # 应用一系列神经网络滤镜
        filters = [
            kornia.filters.GaussianBlur2d((5, 5), (1.5, 1.5)),
            kornia.filters.Laplacian(3),
            kornia.filters.Sobel(),
            kornia.filters.UnsharpMask()
        ]

        # 随机选择并应用滤镜
        filter_idx = np.random.randint(len(filters))
        filtered = filters[filter_idx](x)

        # 转回numpy数组
        filtered = (filtered.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return filtered

    def _add_adversarial_noise(self, image: np.ndarray) -> np.ndarray:
        """添加对抗性噪声"""
        # 转换为PyTorch张量
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(self.device)
        x.requires_grad = True

        # 生成对抗性扰动
        epsilon = 2.0 / 255.0  # 扰动大小
        alpha = 0.5 / 255.0  # 步长

        # FGSM攻击
        with torch.enable_grad():
            noise = torch.zeros_like(x)
            for _ in range(5):  # 迭代5次
                # 前向传播
                x_adv = x + noise

                # 计算损失
                loss = torch.nn.functional.mse_loss(x_adv, x)

                # 反向传播
                grad = torch.autograd.grad(loss, x_adv)[0]

                # 更新噪声
                noise = noise + alpha * grad.sign()
                noise = torch.clamp(noise, -epsilon, epsilon)

        # 添加扰动
        x_adv = x + noise
        x_adv = torch.clamp(x_adv, 0, 1)

        # 转回numpy数组
        x_adv = (x_adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        return x_adv

    def _apply_perceptual_attack(self, image: np.ndarray) -> np.ndarray:
        """应用感知攻击"""
        # 转换为PyTorch张量
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(self.device)

        # 使用预训练的VGG特征提取器
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg = vgg.features.to(self.device).eval()

        # 提取特征
        with torch.no_grad():
            features = vgg(x)

        # 生成对抗样本
        x_adv = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_adv], lr=0.01)

        for _ in range(5):  # 迭代5次
            optimizer.zero_grad()

            # 计算特征差异
            adv_features = vgg(x_adv)
            loss = torch.nn.functional.mse_loss(adv_features, features)

            # 添加视觉相似性约束
            perceptual_loss = torch.nn.functional.mse_loss(x_adv, x)
            total_loss = loss + 0.1 * perceptual_loss

            total_loss.backward()
            optimizer.step()

            # 裁剪到有效范围
            x_adv.data = torch.clamp(x_adv.data, 0, 1)

        # 转回numpy数组
        x_adv = (x_adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        return x_adv

    def _simulate_low_light(self, image: np.ndarray) -> np.ndarray:
        """模拟低光照条件"""
        # 转换到YCbCr空间
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)

        # 降低亮度
        gamma = np.random.uniform(2.0, 3.0)
        y_dark = ((y / 255.0) ** gamma * 255.0).astype(np.uint8)

        # 添加噪声
        noise_sigma = np.random.uniform(5, 15)
        noise = np.random.normal(0, noise_sigma, y_dark.shape).astype(np.uint8)
        y_noisy = np.clip(y_dark + noise, 0, 255).astype(np.uint8)

        # 合并通道
        low_light = cv2.merge([y_noisy, cr, cb])
        low_light = cv2.cvtColor(low_light, cv2.COLOR_YCrCb2BGR)

        return low_light

    def _simulate_weather_effects(self, image: np.ndarray) -> np.ndarray:
        """模拟天气效果"""
        effects = ['rain', 'snow', 'fog']
        effect = np.random.choice(effects)

        if effect == 'rain':
            return self._add_rain(image)
        elif effect == 'snow':
            return self._add_snow(image)
        else:  # fog
            return self._add_fog(image)

    def _add_rain(self, image: np.ndarray) -> np.ndarray:
        """添加雨滴效果"""
        h, w = image.shape[:2]
        rain_layer = np.zeros_like(image)

        # 生成雨滴
        n_drops = np.random.randint(100, 1000)
        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(5, 15)
            angle = np.random.uniform(70, 110)

            # 计算雨滴终点
            radian = np.deg2rad(angle)
            x2 = int(x + length * np.cos(radian))
            y2 = int(y + length * np.sin(radian))

            # 绘制雨滴
            cv2.line(rain_layer, (x, y), (x2, y2), (255, 255, 255), 1)

        # 模糊雨滴
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

        # 合并图层
        alpha = np.random.uniform(0.6, 0.8)
        rainy = cv2.addWeighted(image, 1, rain_layer, alpha, 0)

        # 调整对比度
        rainy = exposure.adjust_gamma(rainy, 0.9)

        return rainy

    def _add_snow(self, image: np.ndarray) -> np.ndarray:
        """添加雪花效果"""
        h, w = image.shape[:2]
        snow_layer = np.zeros_like(image)

        # 生成雪花
        n_flakes = np.random.randint(300, 1500)
        for _ in range(n_flakes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(1, 4)

            # 绘制雪花
            cv2.circle(snow_layer, (x, y), size, (255, 255, 255), -1)

        # 模糊雪花
        snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)

        # 合并图层
        alpha = np.random.uniform(0.4, 0.6)
        snowy = cv2.addWeighted(image, 1, snow_layer, alpha, 0)

        # 增加亮度
        snowy = exposure.adjust_gamma(snowy, 1.1)

        return snowy

    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        """添加雾效果"""
        h, w = image.shape[:2]

        # 创建雾效果
        fog_effect = np.zeros_like(image)
        fog_effect.fill(255)  # 白色雾

        # 创建深度图
        depth_map = np.zeros((h, w))
        center_y, center_x = h // 2, w // 2

        for i in range(h):
            for j in range(w):
                # 计算到中心的距离
                distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                # 归一化距离
                depth_map[i, j] = np.clip(distance / (np.sqrt(h ** 2 + w ** 2) / 2), 0, 1)

        # 根据深度图混合原图和雾效果
        alpha = np.random.uniform(0.4, 0.7)  # 雾的浓度
        depth_map = np.dstack([depth_map] * 3)  # 扩展到3通道
        foggy = image * (1 - depth_map * alpha) + fog_effect * (depth_map * alpha)

        return foggy.astype(np.uint8)

    def _simulate_sensor_effects(self, image: np.ndarray) -> np.ndarray:
        """模拟相机传感器效果"""
        # 色温调整
        temperature = np.random.uniform(3000, 7000)  # 色温范围
        image = self._adjust_color_temperature(image, temperature)

        # 镜头畸变
        image = self._add_lens_distortion(image)

        # 色差
        image = self._add_chromatic_aberration(image)

        # 噪声
        image = self._add_sensor_noise(image)

        return image

    def _adjust_color_temperature(self, image: np.ndarray,
                                  temperature: float) -> np.ndarray:
        """调整色温"""

        # 基于普朗克黑体辐射计算RGB系数
        def planckian_locus(T):
            if T <= 6500:
                r = 1.0
                g = 0.39 * np.log(T/100 - 10) + 0.39
                b = 0.76 * np.log(T / 100 - 10) + 0.16
                else:
                r = 0.95 * np.log(T / 100 - 10) + 0.18
                g = 1.0
                b = 1.2
            return r, g, b

            # 计算色温系数

        r_coef, g_coef, b_coef = planckian_locus(temperature)

        # 分离通道并应用系数
        b, g, r = cv2.split(image)
        r = np.clip(r * r_coef, 0, 255).astype(np.uint8)
        g = np.clip(g * g_coef, 0, 255).astype(np.uint8)
        b = np.clip(b * b_coef, 0, 255).astype(np.uint8)

        return cv2.merge([b, g, r])

    def _add_lens_distortion(self, image: np.ndarray) -> np.ndarray:
        """添加镜头畸变效果"""
        h, w = image.shape[:2]

        # 畸变参数
        k1 = np.random.uniform(-0.1, 0.1)  # 径向畸变系数
        k2 = np.random.uniform(-0.05, 0.05)
        p1 = np.random.uniform(-0.0001, 0.0001)  # 切向畸变系数
        p2 = np.random.uniform(-0.0001, 0.0001)

        # 相机内参矩阵
        focal_length = max(h, w)
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # 畸变系数
        dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float32)

        # 应用畸变
        distorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        return distorted

    def _add_chromatic_aberration(self, image: np.ndarray) -> np.ndarray:
        """添加色差效果"""
        # 分离RGB通道
        b, g, r = cv2.split(image)
        h, w = b.shape[:2]

        # 随机偏移量
        offset_x = int(w * np.random.uniform(0.005, 0.015))
        offset_y = int(h * np.random.uniform(0.005, 0.015))

        # 对红蓝通道进行偏移
        M_r = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        M_b = np.float32([[1, 0, -offset_x], [0, 1, -offset_y]])

        r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)

        return cv2.merge([b, g, r])

    def _add_sensor_noise(self, image: np.ndarray) -> np.ndarray:
        """添加传感器噪声模型"""
        # 转换为浮点型
        image = image.astype(np.float32) / 255.0

        # 1. 光子散粒噪声(泊松噪声)
        signal = np.random.poisson(image * 1000) / 1000.0

        # 2. 读出噪声(高斯噪声)
        read_noise_sigma = np.random.uniform(0.01, 0.02)
        read_noise = np.random.normal(0, read_noise_sigma, image.shape)

        # 3. 固定模式噪声
        pattern_noise = self._generate_fixed_pattern_noise(image.shape)

        # 4. 暗电流噪声
        dark_current = np.random.gamma(2, 0.01, image.shape)

        # 组合所有噪声
        noisy = signal + read_noise + pattern_noise + dark_current

        # 应用非线性响应曲线
        noisy = self._sensor_response_curve(noisy)

        return np.clip(noisy * 255, 0, 255).astype(np.uint8)

    def _generate_fixed_pattern_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """生成固定模式噪声"""
        # 使用Perlin噪声生成固定模式
        h, w = shape[:2]
        scale = 50  # 噪声尺度

        def perlin(x, y, seed=0):
            # Perlin噪声实现
            def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)

            def lerp(t, a, b): return a + t * (b - a)

            # 生成梯度向量
            np.random.seed(seed)
            p = np.arange(256, dtype=int)
            np.random.shuffle(p)
            p = np.stack([p, p]).flatten()

            # 整数部分
            xi, yi = int(x), int(y)
            # 小数部分
            xf, yf = x - xi, y - yi

            # 渐变
            u, v = fade(xf), fade(yf)

            # 计算四个角的hash值
            n00 = p[p[xi % 256] + yi % 256]
            n01 = p[p[xi % 256] + (yi + 1) % 256]
            n10 = p[p[(xi + 1) % 256] + yi % 256]
            n11 = p[p[(xi + 1) % 256] + (yi + 1) % 256]

            # 插值
            x1 = lerp(u, n00, n10)
            x2 = lerp(u, n01, n11)
            return lerp(v, x1, x2)

        noise = np.zeros(shape[:2])
        for i in range(h):
            for j in range(w):
                noise[i, j] = perlin(i / scale, j / scale)

        # 归一化到较小范围
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = noise * 0.02 - 0.01  # 缩放到±0.01范围

        # 扩展到3通道
        if len(shape) == 3:
            noise = np.stack([noise] * shape[2], axis=-1)

        return noise

    def _sensor_response_curve(self, x: np.ndarray) -> np.ndarray:
        """模拟相机传感器的非线性响应曲线"""

        # 使用修改的S型曲线
        def sigmoid(x, a=0.05):
            return 1 / (1 + np.exp(-x / a))

        # 归一化到[0,1]
        x = np.clip(x, 0, 1)

        # 应用响应曲线
        y = sigmoid(x - 0.5)
        y = (y - sigmoid(-0.5)) / (sigmoid(0.5) - sigmoid(-0.5))

        return np.clip(y, 0, 1)

    def _restore_quality(self, image: np.ndarray,
                         target_quality: float) -> np.ndarray:
        """恢复图像质量"""
        # 1. 降噪
        denoised = restoration.denoise_wavelet(
            image,
            multichannel=True,
            convert2ycbcr=True,
            method='BayesShrink'
        )

        # 2. 锐化
        sharpened = np.clip(
            denoised + 0.5 * (denoised - cv2.GaussianBlur(denoised, (0, 0), 3)),
            0, 1
        )

        # 3. 对比度增强
        p2, p98 = np.percentile(sharpened, (2, 98))
        enhanced = exposure.rescale_intensity(sharpened, in_range=(p2, p98))

        # 4. 色彩校正
        corrected = exposure.adjust_gamma(enhanced, 1.1)

        return (corrected * 255).astype(np.uint8)

    class NoiseGenerator(torch.nn.Module):
        """生成逼真的图像噪声的神经网络"""

        def __init__(self):
            super().__init__()

            # 编码器
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True)
            )

            # 噪声生成器
            self.noise_generator = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 3, 3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.encoder(x)
            noise = self.noise_generator(features)
            return noise

    class ImageQualityEvaluator:
        """评估图像质量的综合评估器"""

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
                scores[name] = metric(image)

            # 加权平均
            weights = {
                'sharpness': 0.3,
                'noise': 0.2,
                'compression': 0.2,
                'contrast': 0.15,
                'color': 0.15
            }

            weighted_score = sum(scores[k] * weights[k] for k in scores)
            return weighted_score

        def _evaluate_sharpness(self, image: np.ndarray) -> float:
            """评估图像清晰度"""
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 拉普拉斯算子
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Sobel梯度
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx ** 2 + sobely ** 2).mean()

            # 组合评分
            score = (laplacian + gradient) / 2
            return np.clip(score / 1000, 0, 1)  # 归一化

        def _evaluate_noise(self, image: np.ndarray) -> float:
            """评估图像噪声水平"""
            # 转换为YCbCr空间
            ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y = ycbcr[..., 0]

            # 计算局部方差
            kernel_size = 5
            local_mean = cv2.blur(y.astype(float), (kernel_size, kernel_size))
            local_var = cv2.blur(y.astype(float) ** 2, (kernel_size, kernel_size)) - local_mean ** 2

            # 评估噪声水平
            noise_level = np.sqrt(np.mean(local_var))
            return 1 - np.clip(noise_level / 50, 0, 1)  # 转换为质量分数

        def _evaluate_compression(self, image: np.ndarray) -> float:
            """评估压缩质量"""

            # 检测块效应
            def block_effect(channel):
                h, w = channel.shape
                block_size = 8

                # 计算块边界处的差异
                h_diff = np.abs(channel[:, block_size - 1:-1:block_size] -
                                channel[:, block_size:-1:block_size]).mean()
                v_diff = np.abs(channel[block_size - 1:-1:block_size, :] -
                                channel[block_size:-1:block_size, :]).mean()

                return (h_diff + v_diff) / 2

            # 分别计算每个通道的块效应
            b, g, r = cv2.split(image)
            block_scores = [block_effect(c) for c in [b, g, r]]

            # 评分转换
            compression_score = 1 - np.mean(block_scores) / 20
            return np.clip(compression_score, 0, 1)

        def _evaluate_contrast(self, image: np.ndarray) -> float:
            """评估对比度"""
            # 转换为LAB空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l = lab[..., 0]

            # 计算对比度
            p1, p99 = np.percentile(l, (1, 99))

            # 计算对比度得分
            contrast_range = p99 - p1
            score = contrast_range / 255

            # 惩罚过高或过低的对比度
            optimal_contrast = 0.6
            penalty = 1 - abs(score - optimal_contrast)

            return np.clip(penalty, 0, 1)

        def _evaluate_color(self, image: np.ndarray) -> float:
            """评估色彩质量"""
            # 转换到HSV空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            s = hsv[..., 1]
            v = hsv[..., 2]

            # 计算饱和度和明度的分布
            s_mean = np.mean(s)
            v_mean = np.mean(v)
            s_std = np.std(s)
            v_std = np.std(v)

            # 评估色彩平衡
            optimal_s_mean = 128
            optimal_v_mean = 128
            optimal_std = 50

            # 计算与理想值的偏差
            s_score = 1 - abs(s_mean - optimal_s_mean) / optimal_s_mean
            v_score = 1 - abs(v_mean - optimal_v_mean) / optimal_v_mean
            std_score = 1 - abs(np.mean([s_std, v_std]) - optimal_std) / optimal_std

            # 综合评分
            color_score = (s_score + v_score + std_score) / 3
            return np.clip(color_score, 0, 1)

        class AugmentationPipeline:
            """数据增强流水线管理器"""

            def __init__(self, config_path: str):
                """初始化增强流水线"""
                self.augmentor = ImageAugmentor(config_path)
                self.quality_evaluator = ImageQualityEvaluator()

                # 读取配置
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)

                # 初始化增强策略
                self.strategies = self._init_strategies()

                # 设置日志
                logging.basicConfig(level=logging.INFO)
                self.logger = logging.getLogger(__name__)

            def _init_strategies(self) -> Dict[str, Dict]:
                """初始化增强策略"""
                return {
                    'basic': {
                        'prob': 1.0,
                        'quality_threshold': 0.7
                    },
                    'advanced': {
                        'prob': 0.7,
                        'quality_threshold': 0.6
                    },
                    'expert': {
                        'prob': 0.3,
                        'quality_threshold': 0.5
                    }
                }

            def augment_batch(self, images: List[np.ndarray],
                              masks: Optional[List[np.ndarray]] = None,
                              strategy: str = 'mixed') -> Dict[str, List[np.ndarray]]:
                """
                对一批图像进行增强
                Args:
                    images: 输入图像列表
                    masks: 可选的掩码列表
                    strategy: 增强策略 ['basic', 'advanced', 'expert', 'mixed']
                Returns:
                    增强后的图像和掩码
                """
                try:
                    results = {'images': [], 'masks': []}

                    for idx, image in enumerate(images):
                        # 选择增强策略
                        if strategy == 'mixed':
                            current_strategy = self._select_strategy(image)
                        else:
                            current_strategy = strategy

                        # 获取对应的掩码
                        mask = masks[idx] if masks is not None else None

                        # 应用增强
                        if np.random.random() < self.strategies[current_strategy]['prob']:
                            augmented = self.augmentor.augment(
                                image,
                                mask=mask,
                                level=current_strategy
                            )

                            # 质量控制
                            quality = self.quality_evaluator.evaluate(augmented['image'])
                            threshold = self.strategies[current_strategy]['quality_threshold']

                            if quality >= threshold:
                                results['images'].append(augmented['image'])
                                if mask is not None:
                                    results['masks'].append(augmented['mask'])
                            else:
                                # 质量不达标，使用原图
                                results['images'].append(image)
                                if mask is not None:
                                    results['masks'].append(mask)
                        else:
                            # 不进行增强
                            results['images'].append(image)
                            if mask is not None:
                                results['masks'].append(mask)

                    return results

                except Exception as e:
                    self.logger.error(f"批量增强失败: {str(e)}")
                    raise

            def _select_strategy(self, image: np.ndarray) -> str:
                """根据图像特征选择增强策略"""
                # 评估图像质量
                quality = self.quality_evaluator.evaluate(image)

                # 评估图像复杂度
                complexity = self._evaluate_complexity(image)

                # 根据质量和复杂度选择策略
                if quality > 0.8 and complexity < 0.3:
                    # 高质量、低复杂度图像使用基础增强
                    return 'basic'
                elif 0.5 <= quality <= 0.8 and complexity < 0.7:
                    # 中等质量图像使用高级增强
                    return 'advanced'
                else:
                    # 低质量或高复杂度图像使用专家级增强
                    return 'expert'

            def _evaluate_complexity(self, image: np.ndarray) -> float:
                """评估图像复杂度"""
                # 1. 边缘复杂度
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.mean(edges > 0)

                # 2. 纹理复杂度
                glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
                contrast = greycoprops(glcm, 'contrast')[0, 0]
                entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

                # 3. 颜色复杂度
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                color_std = np.std(hsv[..., 0])

                # 综合评分
                complexity = (
                        0.4 * edge_density +
                        0.3 * (contrast / 100) +
                        0.2 * (entropy / 8) +
                        0.1 * (color_std / 180)
                )

                return np.clip(complexity, 0, 1)

            def visualize_augmentation(self, image: np.ndarray,
                                       strategy: str = 'mixed') -> None:
                """可视化增强效果"""
                import matplotlib.pyplot as plt

                # 创建子图
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.ravel()

                # 原图
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[0].set_title('Original')

                # 不同策略的增强效果
                for idx, level in enumerate(['basic', 'advanced', 'expert'], 1):
                    augmented = self.augmentor.augment(image, level=level)['image']
                    axes[idx].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
                    axes[idx].set_title(f'{level.capitalize()} Augmentation')

                # 混合策略
                if strategy == 'mixed':
                    selected = self._select_strategy(image)
                    augmented = self.augmentor.augment(image, level=selected)['image']
                    axes[4].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
                    axes[4].set_title(f'Mixed Strategy ({selected})')

                # 质量对比
                quality_orig = self.quality_evaluator.evaluate(image)
                quality_aug = self.quality_evaluator.evaluate(augmented)
                axes[5].bar(['Original', 'Augmented'], [quality_orig, quality_aug])
                axes[5].set_title('Quality Scores')

                plt.tight_layout()
                plt.show()

        def create_augmentation_pipeline(config_path: str) -> AugmentationPipeline:
            """创建数据增强流水线工厂函数"""
            return AugmentationPipeline(config_path)