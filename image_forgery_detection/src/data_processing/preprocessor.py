import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Tuple, Optional, List, Union
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage
from PIL import Image, ImageEnhance


class ImagePreprocessor:
    """图像预处理器,实现所有预处理功能"""

    def __init__(self, config_path: str):
        """初始化预处理器"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['preprocessing']

        self.resize_config = self.config['resize']
        self.color_config = self.config['color_convert']
        self.enhance_config = self.config['enhance']
        self.noise_config = self.config['noise_analysis']
        self.edge_config = self.config['edge_detection']
        self.texture_config = self.config['texture_analysis']
        self.quality_config = self.config['quality_assessment']

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        对输入图像进行全流程预处理
        Args:
            image: 输入BGR图像
        Returns:
            包含所有预处理结果的字典
        """
        try:
            results = {}

            # 1. 尺寸调整
            results['resized'] = self._resize_image(image)

            # 2. 颜色空间转换
            results.update(self._convert_color_spaces(results['resized']))

            # 3. 图像增强
            if self.enhance_config:
                results.update(self._enhance_image(results['resized']))

            # 4. 噪声分析
            if self.noise_config['estimate_noise']:
                results['noise_map'] = self._estimate_noise(results['resized'])

            # 5. 边缘检测
            results['edges'] = self._detect_edges(cv2.cvtColor(results['resized'],
                                                               cv2.COLOR_BGR2GRAY))

            # 6. 纹理分析
            if self.texture_config:
                results.update(self._analyze_texture(results['resized']))

            # 7. 质量评估
            if self.quality_config:
                results['quality_metrics'] = self._assess_quality(results['resized'])

            return results

        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            raise

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        target_size = self.resize_config['size']

        if self.resize_config['keep_aspect_ratio']:
            # 保持长宽比调整尺寸
            h, w = image.shape[:2]
            if h > w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))

            resized = cv2.resize(image, (new_w, new_h),
                                 interpolation=self._get_interpolation())

            # Padding到目标尺寸
            pad_h = target_size - new_h
            pad_w = target_size - new_w

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                         self._get_pad_mode())
        else:
            resized = cv2.resize(image, (target_size, target_size),
                                 interpolation=self._get_interpolation())

        return resized

    def _convert_color_spaces(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """转换颜色空间"""
        results = {}
        if self.color_config['to_rgb']:
            results['rgb'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.color_config['to_ycbcr']:
            results['ycbcr'] = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        if self.color_config['to_hsv']:
            results['hsv'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.color_config['to_lab']:
            results['lab'] = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return results

    def _enhance_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """图像增强"""
        results = {}
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.enhance_config['brightness']:
            enhancer = ImageEnhance.Brightness(pil_img)
            results['enhanced_brightness'] = np.array(enhancer.enhance(1.2))

        if self.enhance_config['contrast']:
            enhancer = ImageEnhance.Contrast(pil_img)
            results['enhanced_contrast'] = np.array(enhancer.enhance(1.2))

        if self.enhance_config['sharpness']:
            enhancer = ImageEnhance.Sharpness(pil_img)
            results['enhanced_sharpness'] = np.array(enhancer.enhance(1.5))

        return results

    def _estimate_noise(self, image: np.ndarray) -> np.ndarray:
        """估计图像噪声"""
        if self.noise_config['denoise_method'] == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
            noise = image - denoised
            return noise
        else:
            raise ValueError(f"不支持的降噪方法: {self.noise_config['denoise_method']}")

    def _detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """边缘检测"""
        if self.edge_config['method'] == 'canny':
            edges = cv2.Canny(gray_image,
                              self.edge_config['low_threshold'],
                              self.edge_config['high_threshold'])
            return edges
        else:
            raise ValueError(f"不支持的边缘检测方法: {self.edge_config['method']}")

    def _analyze_texture(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """纹理分析"""
        results = {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.texture_config['lbp']:
            radius = 3
            n_points = 8 * radius
            results['lbp'] = local_binary_pattern(gray, n_points, radius)

        if self.texture_config['glcm']:
            distances = [1]
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            results['glcm'] = greycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)

        if self.texture_config['gabor']:
            kernels = self._build_gabor_kernels()
            results['gabor'] = np.zeros_like(gray)
            for kernel in kernels:
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                results['gabor'] = cv2.add(results['gabor'], filtered)

        return results

    def _assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """质量评估"""
        metrics = {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.quality_config['sharpness']:
            metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()

        if self.quality_config['noise_level']:
            noise = self._estimate_noise(image)
            metrics['noise_level'] = noise.std()

        if self.quality_config['jpeg_quality']:
            metrics['jpeg_quality'] = self._estimate_jpeg_quality(image)

        return metrics

    def _get_interpolation(self) -> int:
        """获取插值方法"""
        methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        return methods.get(self.resize_config['interpolation'], cv2.INTER_LANCZOS4)

    def _get_pad_mode(self) -> int:
        """获取填充方法"""
        modes = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE
        }
        return modes.get(self.resize_config['pad_mode'], cv2.BORDER_REFLECT)

    def _build_gabor_kernels(self) -> List[np.ndarray]:
        """构建Gabor滤波器组"""
        kernels = []
        for theta in range(0, 180, 45):
            theta = theta / 180. * np.pi
            for sigma in (1, 3):
                for lambd in np.arange(np.pi / 4, np.pi, np.pi / 4):
                    for gamma in (0.3, 0.7):
                        kernel = cv2.getGaborKernel((21, 21), sigma,
                                                    theta, lambd, gamma, 0)
                        kernels.append(kernel)
        return kernels

    def _estimate_jpeg_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        全面的JPEG质量评估
        实现多种评估指标:
        1. DCT系数分析
        2. 块效应检测
        3. 振铃效应检测
        4. 色度采样分析
        5. 量化误差评估
        """
        metrics = {}

        # 1. DCT系数分析
        def analyze_dct_coefficients(channel: np.ndarray) -> Dict[str, float]:
            # 分块进行DCT变换
            h, w = channel.shape
            block_size = 8
            dct_coeffs = []

            for i in range(0, h - block_size + 1, block_size):
                for j in range(0, w - block_size + 1, block_size):
                    block = channel[i:i + block_size, j:j + block_size].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_coeffs.append(dct_block)

            dct_coeffs = np.array(dct_coeffs)

            # 分析DCT系数分布
            ac_energy = np.sum(np.abs(dct_coeffs[:, 1:, :]))
            dc_energy = np.sum(np.abs(dct_coeffs[:, 0, 0]))
            high_freq_energy = np.sum(np.abs(dct_coeffs[:, 4:, 4:]))

            return {
                'ac_energy': float(ac_energy),
                'dc_energy': float(dc_energy),
                'high_freq_ratio': float(high_freq_energy / ac_energy)
            }

        # 2. 块效应检测
        def detect_blocking_artifacts(channel: np.ndarray) -> float:
            h, w = channel.shape
            block_size = 8

            # 计算块边界处的差异
            h_diff = np.abs(channel[:, block_size - 1:-1:block_size] -
                            channel[:, block_size:-1:block_size]).mean()
            v_diff = np.abs(channel[block_size - 1:-1:block_size, :] -
                            channel[block_size:-1:block_size, :]).mean()

            # 计算非块边界处的差异
            h_diff_non_block = np.abs(np.diff(channel, axis=1))[:, ::block_size].mean()
            v_diff_non_block = np.abs(np.diff(channel, axis=0))[::block_size, :].mean()

            # 块效应强度
            blocking_score = (h_diff + v_diff) / (h_diff_non_block + v_diff_non_block)

            return float(blocking_score)

        # 3. 振铃效应检测
        def detect_ringing_artifacts(channel: np.ndarray) -> float:
            # 使用Sobel算子检测边缘
            sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx ** 2 + sobely ** 2)

            # 创建边缘掩码
            edge_mask = edges > np.percentile(edges, 90)

            # 在边缘周围检测振荡
            kernel = np.ones((3, 3), np.uint8)
            edge_region = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2)

            # 计算边缘区域的局部方差
            local_var = ndimage.variance(channel, labels=edge_region, index=1)

            return float(local_var)

        # 4. 色度采样分析
        def analyze_chroma_subsampling(image: np.ndarray) -> Dict[str, float]:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # 分析色度通道的频率特性
            cr_freq = np.fft.fft2(cr)
            cb_freq = np.fft.fft2(cb)

            cr_energy = np.abs(cr_freq).sum()
            cb_energy = np.abs(cb_freq).sum()

            # 计算色度能量比
            y_energy = np.abs(np.fft.fft2(y)).sum()
            chroma_ratio = (cr_energy + cb_energy) / (2 * y_energy)

            return {
                'chroma_energy_ratio': float(chroma_ratio),
                'cr_cb_balance': float(cr_energy / cb_energy)
            }

        # 5. 量化误差评估
        def estimate_quantization_error(channel: np.ndarray) -> float:
            # 计算像素值分布的离散程度
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()

            # 使用零交叉率评估量化程度
            diff = np.diff(hist)
            zero_crossings = np.where(np.diff(np.signbit(diff)))[0]

            # 计算量化误差得分
            quantization_score = len(zero_crossings) / 256

            return float(quantization_score)

        # 对每个通道分别进行分析
        channels = cv2.split(image)
        channel_names = ['blue', 'green', 'red']

        for channel, name in zip(channels, channel_names):
            # DCT分析
            dct_metrics = analyze_dct_coefficients(channel)
            for key, value in dct_metrics.items():
                metrics[f'{name}_dct_{key}'] = value

            # 块效应
            metrics[f'{name}_blocking_score'] = detect_blocking_artifacts(channel)

            # 振铃效应
            metrics[f'{name}_ringing_score'] = detect_ringing_artifacts(channel)

            # 量化误差
            metrics[f'{name}_quantization_score'] = estimate_quantization_error(channel)

        # 色度分析
        chroma_metrics = analyze_chroma_subsampling(image)
        metrics.update(chroma_metrics)

        # 计算综合质量得分
        quality_score = 100 * (
                1 - 0.3 * np.mean([metrics[f'{ch}_blocking_score'] for ch in channel_names])
                - 0.3 * np.mean([metrics[f'{ch}_quantization_score'] for ch in channel_names])
                - 0.2 * np.mean([metrics[f'{ch}_ringing_score'] for ch in channel_names])
                - 0.2 * (1 - metrics['chroma_energy_ratio'])
        )

        metrics['overall_quality'] = float(np.clip(quality_score, 0, 100))

        return metrics