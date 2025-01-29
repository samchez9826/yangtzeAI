import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Tuple, Optional, List, Union, Any
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage
from PIL import Image, ImageEnhance
import torch
from functools import lru_cache
import psutil
from concurrent.futures import ThreadPoolExecutor
import time
from queue import Queue
from threading import Thread


class PreprocessorError(Exception):
    """预处理器相关错误"""
    pass


class PreprocessorConfig:
    """预处理器配置类"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['preprocessing']

        # 验证配置
        self.validate_config()

    def validate_config(self):
        """验证配置有效性"""
        required_sections = [
            'resize', 'color_convert', 'enhance', 'noise_analysis',
            'edge_detection', 'texture_analysis', 'quality_assessment'
        ]

        for section in required_sections:
            if section not in self.config:
                raise PreprocessorError(f"配置缺少必需部分: {section}")

        # 验证resize配置
        resize_config = self.config['resize']
        if 'size' not in resize_config:
            raise PreprocessorError("resize配置缺少size参数")

        # 验证其他配置...


class ImagePreprocessor:
    """图像预处理器,实现所有预处理功能"""

    def __init__(self, config_path: str, num_workers: int = 4, cache_size: int = 1000):
        """
        初始化预处理器
        Args:
            config_path: 配置文件路径
            num_workers: 工作线程数
            cache_size: 缓存大小
        """
        # 加载配置
        config = PreprocessorConfig(config_path).config
        self.resize_config = config['resize']
        self.color_config = config['color_convert']
        self.enhance_config = config['enhance']
        self.noise_config = config['noise_analysis']
        self.edge_config = config['edge_detection']
        self.texture_config = config['texture_analysis']
        self.quality_config = config['quality_assessment']

        # 设置日志
        self._setup_logger()

        # 初始化缓存
        self.cache_size = cache_size
        self.preprocess_cache = {}

        # 初始化性能监控
        self.process_times = []
        self._monitor_memory()

        # 初始化GPU支持
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.logger.info("启用GPU支持")

        # 初始化线程池
        self.num_workers = min(num_workers, psutil.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        # 初始化预加载
        self.preload_queue = Queue(maxsize=100)
        self._start_preload()

        # 加载预训练模型(如果需要)
        self._load_models()

        self.logger.info("预处理器初始化完成")

    def _setup_logger(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # 文件处理器
            fh = logging.FileHandler('preprocessor.log')
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
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # MB
        self.logger.info(f"当前内存使用: {memory_usage:.2f}MB")

    def _load_models(self):
        """加载预训练模型"""
        try:
            # 如果需要加载预训练模型,在这里实现
            pass
        except Exception as e:
            self.logger.error(f"加载预训练模型失败: {str(e)}")
            raise PreprocessorError("加载预训练模型失败")

    def _start_preload(self):
        """启动预加载线程"""
        self.preload_thread = Thread(target=self._preload_worker)
        self.preload_thread.daemon = True
        self.preload_thread.start()

    def _preload_worker(self):
        """预加载工作线程"""
        try:
            while True:
                # 预加载逻辑
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"预加载线程异常: {str(e)}")

    def _get_cache_key(self, image: np.ndarray) -> str:
        """生成缓存键"""
        # 使用图像的哈希作为缓存键
        return str(hash(image.tobytes()))

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        对输入图像进行全流程预处理
        Args:
            image: 输入BGR图像
        Returns:
            包含所有预处理结果的字典
        """
        try:
            start_time = time.time()

            # 检查输入
            if not isinstance(image, np.ndarray):
                raise PreprocessorError("输入必须是numpy数组")
            if image.ndim != 3:
                raise PreprocessorError("图像必须是3通道")
            if image.dtype != np.uint8:
                raise PreprocessorError("图像必须是uint8类型")

            # 检查缓存
            cache_key = self._get_cache_key(image)
            if cache_key in self.preprocess_cache:
                return self.preprocess_cache[cache_key]

            results = {}
            futures = []

            # 1. 尺寸调整
            results['resized'] = self._resize_image(image)

            # 2. 颜色空间转换
            futures.append(
                self.executor.submit(
                    self._convert_color_spaces,
                    results['resized']
                )
            )

            # 3. 图像增强
            if self.enhance_config:
                futures.append(
                    self.executor.submit(
                        self._enhance_image,
                        results['resized']
                    )
                )

            # 4. 噪声分析
            if self.noise_config['estimate_noise']:
                futures.append(
                    self.executor.submit(
                        self._estimate_noise,
                        results['resized']
                    )
                )

            # 5. 边缘检测
            gray = cv2.cvtColor(results['resized'], cv2.COLOR_BGR2GRAY)
            futures.append(
                self.executor.submit(
                    self._detect_edges,
                    gray
                )
            )

            # 6. 纹理分析
            if self.texture_config:
                futures.append(
                    self.executor.submit(
                        self._analyze_texture,
                        results['resized']
                    )
                )

            # 7. 质量评估
            if self.quality_config:
                futures.append(
                    self.executor.submit(
                        self._assess_quality,
                        results['resized']
                    )
                )

            # 收集结果
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    if isinstance(result, dict):
                        results.update(result)
                    else:
                        self.logger.warning("处理结果格式异常")
                except Exception as e:
                    self.logger.error(f"处理任务失败: {str(e)}")

            # 更新缓存
            if len(self.preprocess_cache) >= self.cache_size:
                # LRU策略: 移除最早的缓存
                oldest_key = next(iter(self.preprocess_cache))
                del self.preprocess_cache[oldest_key]
            self.preprocess_cache[cache_key] = results

            # 记录处理时间
            process_time = time.time() - start_time
            self.process_times.append(process_time)

            # 监控处理性能
            if len(self.process_times) >= 100:
                avg_time = np.mean(self.process_times)
                if avg_time > 0.5:  # 500ms
                    self.logger.warning(f"预处理性能警告: 平均耗时 {avg_time * 1000:.2f}ms")
                self.process_times = []

            return results

        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            raise PreprocessorError(f"预处理失败: {str(e)}")

    def preprocess_batch(self,
                         images: List[np.ndarray],
                         batch_size: int = 32) -> List[Dict[str, np.ndarray]]:
        """
        批量处理图像
        Args:
            images: 图像列表
            batch_size: 批处理大小
        Returns:
            处理结果列表
        """
        try:
            results = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                futures = [
                    self.executor.submit(self.preprocess_image, img)
                    for img in batch
                ]
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"批处理任务失败: {str(e)}")
                        results.append(None)
            return results

        except Exception as e:
            self.logger.error(f"批量预处理失败: {str(e)}")
            raise PreprocessorError(f"批量预处理失败: {str(e)}")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        try:
            target_size = self.resize_config['size']

            if self.resize_config['keep_aspect_ratio']:
                h, w = image.shape[:2]
                if h > w:
                    new_h = target_size
                    new_w = int(w * (target_size / h))
                else:
                    new_w = target_size
                    new_h = int(h * (target_size / w))

                resized = cv2.resize(
                    image,
                    (new_w, new_h),
                    interpolation=self._get_interpolation()
                )

                # Padding到目标尺寸
                pad_h = target_size - new_h
                pad_w = target_size - new_w

                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left

                resized = cv2.copyMakeBorder(
                    resized,
                    top, bottom, left, right,
                    self._get_pad_mode()
                )
            else:
                resized = cv2.resize(
                    image,
                    (target_size, target_size),
                    interpolation=self._get_interpolation()
                )

            return resized

        except Exception as e:
            self.logger.error(f"调整图像尺寸失败: {str(e)}")
            raise PreprocessorError("调整图像尺寸失败")

    def _convert_color_spaces(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """转换颜色空间"""
        try:
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

        except Exception as e:
            self.logger.error(f"颜色空间转换失败: {str(e)}")
            raise PreprocessorError("颜色空间转换失败")

    def _enhance_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """图像增强"""
        try:
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

        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            raise PreprocessorError("图像增强失败")

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        对输入图像进行全流程预处理
        Args:
            image: 输入BGR图像
        Returns:
            包含所有预处理结果的字典
        """
        try:
            start_time = time.time()

            # 检查输入
            if not isinstance(image, np.ndarray):
                raise PreprocessorError("输入必须是numpy数组")
            if image.ndim != 3:
                raise PreprocessorError("图像必须是3通道")
            if image.dtype != np.uint8:
                raise PreprocessorError("图像必须是uint8类型")

            # 检查缓存
            cache_key = self._get_cache_key(image)
            if cache_key in self.preprocess_cache:
                return self.preprocess_cache[cache_key]

            results = {}
            futures = []

            # 1. 尺寸调整
            results['resized'] = self._resize_image(image)

            # 2. 颜色空间转换
            futures.append(
                self.executor.submit(
                    self._convert_color_spaces,
                    results['resized']
                )
            )

            # 3. 图像增强
            if self.enhance_config:
                futures.append(
                    self.executor.submit(
                        self._enhance_image,
                        results['resized']
                    )
                )

            # 4. 噪声分析
            if self.noise_config['estimate_noise']:
                futures.append(
                    self.executor.submit(
                        self._estimate_noise,
                        results['resized']
                    )
                )

            # 5. 边缘检测
            gray = cv2.cvtColor(results['resized'], cv2.COLOR_BGR2GRAY)
            futures.append(
                self.executor.submit(
                    self._detect_edges,
                    gray
                )
            )

            # 6. 纹理分析
            if self.texture_config:
                futures.append(
                    self.executor.submit(
                        self._analyze_texture,
                        results['resized']
                    )
                )

            # 7. 质量评估
            if self.quality_config:
                futures.append(
                    self.executor.submit(
                        self._assess_quality,
                        results['resized']
                    )
                )

            # 收集结果
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    if isinstance(result, dict):
                        results.update(result)
                    else:
                        self.logger.warning("处理结果格式异常")
                except Exception as e:
                    self.logger.error(f"处理任务失败: {str(e)}")

            # 更新缓存
            if len(self.preprocess_cache) >= self.cache_size:
                # LRU策略: 移除最早的缓存
                oldest_key = next(iter(self.preprocess_cache))
                del self.preprocess_cache[oldest_key]
            self.preprocess_cache[cache_key] = results

            # 记录处理时间
            process_time = time.time() - start_time
            self.process_times.append(process_time)

            # 监控处理性能
            if len(self.process_times) >= 100:
                avg_time = np.mean(self.process_times)
                if avg_time > 0.5:  # 500ms
                    self.logger.warning(f"预处理性能警告: 平均耗时 {avg_time * 1000:.2f}ms")
                self.process_times = []

            return results

        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            raise PreprocessorError(f"预处理失败: {str(e)}")

    def preprocess_batch(self,
                         images: List[np.ndarray],
                         batch_size: int = 32) -> List[Dict[str, np.ndarray]]:
        """
        批量处理图像
        Args:
            images: 图像列表
            batch_size: 批处理大小
        Returns:
            处理结果列表
        """
        try:
            results = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                futures = [
                    self.executor.submit(self.preprocess_image, img)
                    for img in batch
                ]
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"批处理任务失败: {str(e)}")
                        results.append(None)
            return results

        except Exception as e:
            self.logger.error(f"批量预处理失败: {str(e)}")
            raise PreprocessorError(f"批量预处理失败: {str(e)}")

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        try:
            target_size = self.resize_config['size']

            if self.resize_config['keep_aspect_ratio']:
                h, w = image.shape[:2]
                if h > w:
                    new_h = target_size
                    new_w = int(w * (target_size / h))
                else:
                    new_w = target_size
                    new_h = int(h * (target_size / w))

                resized = cv2.resize(
                    image,
                    (new_w, new_h),
                    interpolation=self._get_interpolation()
                )

                # Padding到目标尺寸
                pad_h = target_size - new_h
                pad_w = target_size - new_w

                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left

                resized = cv2.copyMakeBorder(
                    resized,
                    top, bottom, left, right,
                    self._get_pad_mode()
                )
            else:
                resized = cv2.resize(
                    image,
                    (target_size, target_size),
                    interpolation=self._get_interpolation()
                )

            return resized

        except Exception as e:
            self.logger.error(f"调整图像尺寸失败: {str(e)}")
            raise PreprocessorError("调整图像尺寸失败")

    def _convert_color_spaces(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """转换颜色空间"""
        try:
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

        except Exception as e:
            self.logger.error(f"颜色空间转换失败: {str(e)}")
            raise PreprocessorError("颜色空间转换失败")

    def _enhance_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """图像增强"""
        try:
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

        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            raise PreprocessorError("图像增强失败")

    def _estimate_noise(self, image: np.ndarray) -> Dict[str, float]:
        """估计图像噪声水平"""
        try:
            results = {}

            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 1. 方差估计
            kernel = np.ones((7, 7), np.float32) / 49
            mean = cv2.filter2D(gray, -1, kernel)
            mean_square = cv2.filter2D(np.square(gray), -1, kernel)
            var = mean_square - np.square(mean)
            noise_var = np.mean(var)
            results['noise_variance'] = float(noise_var)

            # 2. 拉普拉斯算子估计
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_lap = np.var(laplacian) / np.sqrt(2)
            results['noise_laplacian'] = float(noise_lap)

            # 3. 高频分量分析
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            high_freq_noise = np.mean(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 90)])
            results['high_freq_noise'] = float(high_freq_noise)

            if self.noise_config['denoise_method'] == 'gaussian':
                denoised = cv2.GaussianBlur(
                    gray,
                    (self.noise_config['params']['kernel_size'],
                     self.noise_config['params']['kernel_size']),
                    self.noise_config['params']['sigma']
                )
                results['denoised'] = denoised

            return results

        except Exception as e:
            self.logger.error(f"噪声估计失败: {str(e)}")
            raise PreprocessorError("噪声估计失败")

    def _detect_edges(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        """边缘检测与分析"""
        try:
            results = {}

            # 1. Canny边缘检测
            edges_canny = cv2.Canny(
                gray,
                self.edge_config['low_threshold'],
                self.edge_config['high_threshold']
            )
            results['edges_canny'] = edges_canny

            # 2. Sobel边缘检测
            if 'sobel' in self.edge_config['additional_methods']:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
                results['edges_sobel'] = np.uint8(sobel)

            # 3. Laplacian边缘检测
            if 'laplacian' in self.edge_config['additional_methods']:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                results['edges_laplacian'] = np.uint8(np.absolute(laplacian))

            # 4. 边缘强度分析
            results['edge_strength'] = {
                'mean': float(np.mean(edges_canny)),
                'std': float(np.std(edges_canny)),
                'max': float(np.max(edges_canny))
            }

            # 5. 边缘连续性分析
            if self.edge_config.get('edge_refinement', False):
                kernel = np.ones((3, 3), np.uint8)
                refined_edges = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel)
                results['edges_refined'] = refined_edges

            return results

        except Exception as e:
            self.logger.error(f"边缘检测失败: {str(e)}")
            raise PreprocessorError("边缘检测失败")

    def _analyze_texture(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """纹理特征分析"""
        try:
            results = {}
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 1. LBP特征
            if self.texture_config['lbp']['enabled']:
                lbp = local_binary_pattern(
                    gray,
                    self.texture_config['lbp']['points'],
                    self.texture_config['lbp']['radius'],
                    method='uniform'
                )
                results['lbp'] = lbp
                results['lbp_histogram'] = np.histogram(
                    lbp,
                    bins=int(lbp.max() + 1),
                    range=(0, int(lbp.max() + 1)),
                    density=True
                )[0]

            # 2. GLCM特征
            if self.texture_config['glcm']['enabled']:
                distances = self.texture_config['glcm']['distances']
                angles = self.texture_config['glcm']['angles']
                glcm = greycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)

                results['glcm_features'] = {
                    'contrast': greycoprops(glcm, 'contrast'),
                    'dissimilarity': greycoprops(glcm, 'dissimilarity'),
                    'homogeneity': greycoprops(glcm, 'homogeneity'),
                    'correlation': greycoprops(glcm, 'correlation'),
                    'energy': greycoprops(glcm, 'energy')
                }

            # 3. Gabor滤波特征
            if self.texture_config['gabor']['enabled']:
                frequencies = self.texture_config['gabor']['frequencies']
                orientations = self.texture_config['gabor']['orientations']

                gabor_features = []
                for freq in frequencies:
                    for theta in orientations:
                        kernel = cv2.getGaborKernel(
                            (21, 21), 8.0, theta, freq, 0.5, 0, ktype=cv2.CV_32F
                        )
                        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                        gabor_features.append(filtered)

                results['gabor_features'] = np.array(gabor_features)

            return results

        except Exception as e:
            self.logger.error(f"纹理分析失败: {str(e)}")
            raise PreprocessorError("纹理分析失败")

    def _assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """图像质量评估"""
        try:
            results = {}
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 1. 清晰度评估
            if self.quality_config['sharpness']['enabled']:
                if self.quality_config['sharpness']['method'] == 'laplacian_variance':
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    results['sharpness'] = float(np.var(laplacian))

            # 2. 噪声水平评估
            if self.quality_config['noise_level']['enabled']:
                if self.quality_config['noise_level']['method'] == 'standard_deviation':
                    gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
                    noise = gray.astype(np.float32) - gaussian.astype(np.float32)
                    results['noise_level'] = float(np.std(noise))

            # 3. JPEG质量评估
            if self.quality_config['jpeg_quality']['enabled']:
                if self.quality_config['jpeg_quality']['method'] == 'dct_statistics':
                    dct = cv2.dct(np.float32(gray))
                    results['jpeg_quality'] = float(np.mean(np.abs(dct)))

            # 4. 模糊检测
            if self.quality_config['blur_detection']['enabled']:
                if self.quality_config['blur_detection']['method'] == 'frequency_analysis':
                    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
                    dft_shift = np.fft.fftshift(dft)
                    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
                    results['blur_level'] = float(np.mean(magnitude))

            # 5. 压缩伪影检测
            if self.quality_config['compression_artifacts']['enabled']:
                if self.quality_config['compression_artifacts']['method'] == 'blocking_effect':
                    block_size = 8
                    h, w = gray.shape
                    results['blocking_effect'] = float(np.mean([
                        np.mean(np.abs(gray[i:i + block_size - 1, :] - gray[i + 1:i + block_size, :]))
                        for i in range(0, h - block_size, block_size)
                    ]))

            return results

        except Exception as e:
            self.logger.error(f"质量评估失败: {str(e)}")
            raise PreprocessorError("质量评估失败")

    def _get_interpolation(self) -> int:
        """获取插值方法"""
        interpolation_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        return interpolation_map.get(
            self.resize_config['interpolation'],
            cv2.INTER_LINEAR
        )

    def _get_pad_mode(self) -> int:
        """获取填充模式"""
        pad_mode_map = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE
        }
        return pad_mode_map.get(
            self.resize_config['pad_mode'],
            cv2.BORDER_REFLECT
        )

    def __del__(self):
        """清理资源"""
        self.executor.shutdown(wait=True)