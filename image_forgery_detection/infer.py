from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

from src.models.base_model import BaseForgeryDetector
from src.data_processing.preprocessor import ImagePreprocessor
from src.utils.logger import setup_logger
from src.utils.visualization import VisualizationManager

# 配置日志
logger = setup_logger('inference', level='INFO')


@dataclass
class InferenceConfig:
    """推理配置"""
    config_path: Path
    checkpoint_path: Path
    input_path: Path
    output_dir: Path
    device: str = 'cuda'
    threshold: float = 0.5
    batch_size: int = 32
    num_workers: int = 4

    # 视频相关配置
    save_frames: bool = True
    frame_interval: int = 1  # 处理帧间隔
    min_clip_duration: float = 1.0  # 最小篡改片段持续时间(秒)

    # 可视化配置
    save_visualizations: bool = True
    visualization_fps: int = 30

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'InferenceConfig':
        return cls(
            config_path=Path(args.config),
            checkpoint_path=Path(args.checkpoint),
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            device=args.device,
            threshold=args.threshold
        )

    def validate(self) -> None:
        """验证配置"""
        if not self.config_path.exists():
            raise ValueError(f"Config file not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.input_path.exists():
            raise ValueError(f"Input path not found: {self.input_path}")
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError(f"Invalid threshold: {self.threshold}")


class InferenceEngine:
    """推理引擎"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 加载模型配置
        with open(config.config_path) as f:
            self.model_config = yaml.safe_load(f)

        # 初始化组件
        self.model = self._build_model()
        self.preprocessor = ImagePreprocessor(self.model_config)
        self.visualizer = VisualizationManager(config.output_dir / 'visualizations')

        # 创建输出目录
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 性能统计
        self.inference_times: List[float] = []

        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)

    def _build_model(self) -> nn.Module:
        """构建模型"""
        try:
            model = BaseForgeryDetector.create(
                self.model_config['model']['name'],
                self.model_config
            ).to(self.device)

            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to build model: {e}")

    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """处理单张图像"""
        try:
            # 加载图像
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # 预处理
            processed = self.preprocessor.preprocess_image(image)
            tensor = torch.from_numpy(processed['resized']).unsqueeze(0).to(self.device)

            # 推理
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(tensor)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # 处理输出
            cls_pred = torch.sigmoid(outputs['cls']).cpu().numpy()[0]
            seg_pred = outputs.get('seg', None)
            if seg_pred is not None:
                seg_pred = seg_pred.cpu().numpy()[0]

            # 保存可视化结果
            if self.config.save_visualizations:
                self._save_visualization(
                    image, cls_pred, seg_pred,
                    self.output_dir / f"{Path(image_path).stem}_result.png"
                )

            return {
                'path': str(image_path),
                'prediction': 'Fake' if cls_pred > self.config.threshold else 'Real',
                'confidence': float(cls_pred),
                'inference_time': inference_time,
                'has_mask': seg_pred is not None
            }

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return {
                'path': str(image_path),
                'error': str(e)
            }


class InferenceEngine:  # continued
    def _save_visualization(
            self,
            image: np.ndarray,
            cls_pred: float,
            seg_pred: Optional[np.ndarray],
            save_path: Path
    ) -> None:
        """保存可视化结果"""
        try:
            plt.figure(figsize=(12, 4))

            # 原始图像
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Prediction: {'Fake' if cls_pred > self.config.threshold else 'Real'}\n"
                      f"Confidence: {cls_pred:.4f}")
            plt.axis('off')

            if seg_pred is not None:
                # 分割掩码
                plt.subplot(132)
                plt.imshow(seg_pred, cmap='jet')
                plt.title('Forgery Mask')
                plt.axis('off')

                # 叠加结果
                plt.subplot(133)
                overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
                mask = cv2.resize(seg_pred, (image.shape[1], image.shape[0]))
                overlay[mask > self.config.threshold] = (0, 255, 0)
                plt.imshow(overlay)
                plt.title('Overlay')
                plt.axis('off')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to save visualization: {e}")

    def process_batch(
            self,
            image_paths: List[Path],
            batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """批量处理图像"""
        results = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(image_paths), batch_size), total=total_batches):
            batch_paths = image_paths[i:i + batch_size]

            try:
                # 加载并预处理批次
                batch_images = []
                valid_indices = []

                for idx, path in enumerate(batch_paths):
                    try:
                        image = cv2.imread(str(path))
                        if image is not None:
                            processed = self.preprocessor.preprocess_image(image)
                            batch_images.append(processed['resized'])
                            valid_indices.append(idx)
                    except Exception as e:
                        logger.error(f"Failed to load image {path}: {e}")

                if not batch_images:
                    continue

                # 转换为tensor
                batch_tensor = torch.from_numpy(
                    np.stack(batch_images)
                ).to(self.device)

                # 批量推理
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                batch_time = time.time() - start_time

                # 处理输出
                cls_preds = torch.sigmoid(outputs['cls']).cpu().numpy()
                seg_preds = outputs.get('seg', None)
                if seg_preds is not None:
                    seg_preds = seg_preds.cpu().numpy()

                # 收集结果
                for idx, (valid_idx, cls_pred) in enumerate(zip(valid_indices, cls_preds)):
                    path = batch_paths[valid_idx]
                    result = {
                        'path': str(path),
                        'prediction': 'Fake' if cls_pred > self.config.threshold else 'Real',
                        'confidence': float(cls_pred),
                        'inference_time': batch_time / len(valid_indices),
                        'has_mask': seg_preds is not None
                    }

                    # 保存可视化结果
                    if self.config.save_visualizations:
                        image = cv2.imread(str(path))
                        seg_pred = seg_preds[idx] if seg_preds is not None else None
                        self._save_visualization(
                            image,
                            cls_pred,
                            seg_pred,
                            self.output_dir / f"{path.stem}_result.png"
                        )

                    results.append(result)

            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                continue

        return results

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """处理视频"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建输出视频
            output_path = self.output_dir / f"{video_path.stem}_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.config.visualization_fps,
                (frame_width * 2, frame_height)
            )

            # 创建结果缓冲区
            frame_results = []
            buffer_queue = Queue(maxsize=100)  # 帧缓冲队列
            result_queue = Queue()  # 结果队列

            # 启动处理线程
            stop_event = threading.Event()
            process_thread = threading.Thread(
                target=self._process_video_frames,
                args=(buffer_queue, result_queue, stop_event)
            )
            process_thread.start()

            # 读取帧
            frame_idx = 0
            with tqdm(total=total_frames, desc='Processing video') as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 按间隔处理帧
                    if frame_idx % self.config.frame_interval == 0:
                        buffer_queue.put((frame_idx, frame))

                    frame_idx += 1
                    pbar.update(1)

            # 停止处理
            stop_event.set()
            process_thread.join()

            # 收集结果
            while not result_queue.empty():
                frame_results.append(result_queue.get())

            # 分析结果
            results = self._analyze_video_results(
                frame_results,
                fps,
                total_frames
            )

            # 保存结果
            with open(self.output_dir / f"{video_path.stem}_analysis.json", 'w') as f:
                json.dump(results, f, indent=2)

            return results

        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {e}")
            raise
        finally:
            cap.release()
            if 'out' in locals():
                out.release()

    def _process_video_frames(
            self,
            buffer_queue: Queue,
            result_queue: Queue,
            stop_event: threading.Event
    ) -> None:
        """处理视频帧线程"""
        while not stop_event.is_set() or not buffer_queue.empty():
            try:
                frame_idx, frame = buffer_queue.get(timeout=1)
            except:
                continue

            try:
                # 处理帧
                processed = self.preprocessor.preprocess_image(frame)
                tensor = torch.from_numpy(processed['resized']).unsqueeze(0).to(self.device)

                # 推理
                with torch.no_grad():
                    outputs = self.model(tensor)

                # 处理结果
                cls_pred = torch.sigmoid(outputs['cls']).cpu().numpy()[0]
                seg_pred = outputs.get('seg', None)
                if seg_pred is not None:
                    seg_pred = seg_pred.cpu().numpy()[0]

                result_queue.put({
                    'frame_idx': frame_idx,
                    'prediction': 'Fake' if cls_pred > self.config.threshold else 'Real',
                    'confidence': float(cls_pred),
                    'has_mask': seg_pred is not None
                })

            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {e}")


class InferenceEngine:  # continued
    def _analyze_video_results(
            self,
            frame_results: List[Dict[str, Any]],
            fps: float,
            total_frames: int
    ) -> Dict[str, Any]:
        """分析视频结果"""
        # 按时间顺序排序结果
        frame_results.sort(key=lambda x: x['frame_idx'])

        # 识别篡改片段
        segments = self._identify_tampering_segments(
            frame_results,
            fps
        )

        # 统计结果
        fake_frames = sum(1 for r in frame_results if r['prediction'] == 'Fake')
        fake_ratio = fake_frames / len(frame_results)

        avg_confidence = np.mean([r['confidence'] for r in frame_results])
        std_confidence = np.std([r['confidence'] for r in frame_results])

        return {
            'summary': {
                'total_frames': total_frames,
                'processed_frames': len(frame_results),
                'fake_frames': fake_frames,
                'fake_ratio': float(fake_ratio),
                'avg_confidence': float(avg_confidence),
                'std_confidence': float(std_confidence),
                'fps': float(fps)
            },
            'tampering_segments': segments,
            'frame_results': frame_results
        }

    def _identify_tampering_segments(
            self,
            frame_results: List[Dict[str, Any]],
            fps: float
    ) -> List[Dict[str, Any]]:
        """识别视频中的篡改片段"""
        segments = []
        min_segment_frames = int(self.config.min_clip_duration * fps)

        start_idx = None
        current_segment = []

        for result in frame_results:
            if result['prediction'] == 'Fake':
                if start_idx is None:
                    start_idx = result['frame_idx']
                current_segment.append(result)
            else:
                if start_idx is not None and len(current_segment) >= min_segment_frames:
                    segments.append({
                        'start_frame': start_idx,
                        'end_frame': current_segment[-1]['frame_idx'],
                        'start_time': start_idx / fps,
                        'end_time': current_segment[-1]['frame_idx'] / fps,
                        'duration': (current_segment[-1]['frame_idx'] - start_idx) / fps,
                        'avg_confidence': float(np.mean([r['confidence'] for r in current_segment])),
                        'max_confidence': float(np.max([r['confidence'] for r in current_segment]))
                    })
                start_idx = None
                current_segment = []

        # 处理最后一个片段
        if start_idx is not None and len(current_segment) >= min_segment_frames:
            segments.append({
                'start_frame': start_idx,
                'end_frame': current_segment[-1]['frame_idx'],
                'start_time': start_idx / fps,
                'end_time': current_segment[-1]['frame_idx'] / fps,
                'duration': (current_segment[-1]['frame_idx'] - start_idx) / fps,
                'avg_confidence': float(np.mean([r['confidence'] for r in current_segment])),
                'max_confidence': float(np.max([r['confidence'] for r in current_segment]))
            })

        return segments

    def run(self) -> Dict[str, Any]:
        """运行推理"""
        results = {
            'config': {
                'threshold': self.config.threshold,
                'device': str(self.device),
                'model': self.model_config['model']['name']
            },
            'inputs': [],
            'performance': {}
        }

        try:
            input_path = self.config.input_path

            # 处理单个文件
            if input_path.is_file():
                if input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    # 处理视频
                    results['type'] = 'video'
                    results['results'] = self.process_video(input_path)
                else:
                    # 处理图像
                    results['type'] = 'image'
                    results['results'] = [self.process_image(input_path)]

            # 处理目录
            else:
                # 收集所有图像文件
                image_files = sorted(
                    [p for p in input_path.glob('**/*') if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                )
                if not image_files:
                    raise ValueError(f"No image files found in {input_path}")

                # 批量处理图像
                results['type'] = 'image_batch'
                results['results'] = self.process_batch(
                    image_files,
                    self.config.batch_size
                )

            # 计算性能统计
            results['performance'] = {
                'avg_inference_time': float(np.mean(self.inference_times)),
                'std_inference_time': float(np.std(self.inference_times)),
                'min_inference_time': float(np.min(self.inference_times)),
                'max_inference_time': float(np.max(self.inference_times)),
                'total_time': float(sum(self.inference_times))
            }

            # 保存统计信息
            self._save_summary(results)

            return results

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
        finally:
            # 清理资源
            self._executor.shutdown()

    def _save_summary(self, results: Dict[str, Any]) -> None:
        """保存结果摘要"""
        summary_path = self.output_dir / 'inference_summary.txt'

        lines = [
            "Inference Summary",
            "=" * 50,
            f"\nInput: {self.config.input_path}",
            f"Model: {self.model_config['model']['name']}",
            f"Device: {self.device}",
            f"\nResults:",
        ]

        if results['type'] == 'image_batch':
            total = len(results['results'])
            fake_count = sum(1 for r in results['results'] if r['prediction'] == 'Fake')
            lines.extend([
                f"Total images: {total}",
                f"Fake images: {fake_count}",
                f"Real images: {total - fake_count}",
                f"Average confidence: {np.mean([r['confidence'] for r in results['results']]):.4f}"
            ])
        elif results['type'] == 'video':
            lines.extend([
                f"Total frames: {results['results']['summary']['total_frames']}",
                f"Fake ratio: {results['results']['summary']['fake_ratio']:.2%}",
                f"Number of tampering segments: {len(results['results']['tampering_segments'])}"
            ])

        lines.extend([
            "\nPerformance:",
            f"Average inference time: {results['performance']['avg_inference_time']:.4f}s",
            f"Total processing time: {results['performance']['total_time']:.2f}s"
        ])

        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    """主函数"""
    try:
        # 解析参数
        parser = argparse.ArgumentParser(
            description='Run forgery detection inference',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--config', required=True, help='Path to config file')
        parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
        parser.add_argument('--input', required=True, help='Path to input image/video/directory')
        parser.add_argument('--output-dir', default='outputs/inference', help='Output directory')
        parser.add_argument('--device', default='cuda', help='Device to run inference on')
        parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')

        args = parser.parse_args()

        # 创建配置
        config = InferenceConfig.from_args(args)

        # 验证配置
        config.validate()

        # 创建推理引擎
        engine = InferenceEngine(config)

        # 运行推理
        results = engine.run()

        # 输出结果路径
        print(f"\nResults saved to: {config.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == '__main__':
    # 设置更好的异常展示
    import sys
    import traceback


    def excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("An unhandled exception occurred:")
        logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))


    sys.excepthook = excepthook

    # 运行主函数
    sys.exit(main())