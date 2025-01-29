import argparse
import torch
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.models.base_model import BaseForgeryDetector
from src.data_processing.preprocessor import ImagePreprocessor
from src.utils.logger import setup_logger
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output-dir', type=str, default='outputs/results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='检测阈值')
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """加载并预处理图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image


def visualize_results(image: np.ndarray,
                      cls_pred: float,
                      seg_pred: Optional[np.ndarray] = None,
                      save_path: str = None):
    """可视化预测结果"""
    plt.figure(figsize=(12, 4))

    # 原始图像
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {'Fake' if cls_pred > 0.5 else 'Real'}\n"
              f"Confidence: {cls_pred:.4f}")
    plt.axis('off')

    # 分割掩码
    if seg_pred is not None:
        plt.subplot(132)
        plt.imshow(seg_pred, cmap='jet')
        plt.title('Forgery Mask')
        plt.axis('off')

        # 叠加结果
        plt.subplot(133)
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        overlay[seg_pred > 0.5] = (0, 255, 0)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def process_single_image(image_path: str,
                         model: torch.nn.Module,
                         preprocessor: ImagePreprocessor,
                         device: torch.device,
                         threshold: float,
                         output_dir: Path) -> Dict[str, Any]:
    """处理单张图像"""
    # 加载并预处理图像
    image = load_image(image_path)
    processed = preprocessor.preprocess_image(image)
    tensor = torch.from_numpy(processed['resized']).unsqueeze(0).to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)

    # 处理预测结果
    cls_pred = torch.sigmoid(outputs['cls']).cpu().numpy()[0]
    seg_pred = outputs['seg'].cpu().numpy()[0] if 'seg' in outputs else None

    # 保存可视化结果
    save_path = output_dir / f"{Path(image_path).stem}_result.png"
    visualize_results(image, cls_pred, seg_pred, str(save_path))

    return {
        'path': image_path,
        'prediction': 'Fake' if cls_pred > threshold else 'Real',
        'confidence': float(cls_pred),
        'has_mask': seg_pred is not None
    }


def main():
    args = parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 创建模型并加载权重
    model = BaseForgeryDetector.create(config['model']['name'], config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建预处理器
    preprocessor = ImagePreprocessor(config)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理输入
    input_path = Path(args.input)
    results = []

    if input_path.is_file():
        # 处理单个文件
        result = process_single_image(
            str(input_path), model, preprocessor,
            device, args.threshold, output_dir
        )
        results.append(result)
    else:
        # 处理目录
        image_files = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.png'))
        for image_file in tqdm(image_files, desc='Processing'):
            try:
                result = process_single_image(
                    str(image_file), model, preprocessor,
                    device, args.threshold, output_dir
                )
                results.append(result)
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {str(e)}")

        # 保存结果
        save_path = output_dir / 'inference_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

        # 打印统计信息
        total = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'Fake')
        real_count = total - fake_count

        print("\nInference Results:")
        print(f"Total images: {total}")
        print(f"Fake images detected: {fake_count}")
        print(f"Real images detected: {real_count}")
        print(f"Average confidence: {sum(r['confidence'] for r in results) / total:.4f}")
        print(f"\nResults saved to: {save_path}")

    class VideoInference:
        """视频推理类"""

        def __init__(self, model: nn.Module, preprocessor: ImagePreprocessor,
                     device: torch.device, threshold: float = 0.5):
            self.model = model
            self.preprocessor = preprocessor
            self.device = device
            self.threshold = threshold

        def process_video(self, video_path: str, output_path: str,
                          save_frames: bool = False):
            """处理视频"""
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")

            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (frame_width * 2, frame_height))

            frame_results = []
            pbar = tqdm(total=total_frames, desc='Processing video')

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                processed = self.preprocessor.preprocess_image(frame)
                tensor = torch.from_numpy(processed['resized']).unsqueeze(0).to(self.device)

                # 模型推理
                with torch.no_grad():
                    outputs = self.model(tensor)

                # 处理预测结果
                cls_pred = torch.sigmoid(outputs['cls']).cpu().numpy()[0]
                seg_pred = outputs['seg'].cpu().numpy()[0] if 'seg' in outputs else None

                # 创建可视化结果
                vis_frame = frame.copy()
                if seg_pred is not None:
                    # 调整掩码大小
                    mask = cv2.resize(seg_pred, (frame_width, frame_height))
                    vis_frame[mask > self.threshold] = (0, 255, 0)

                # 添加预测信息
                label = f"Fake ({cls_pred:.2f})" if cls_pred > self.threshold else \
                    f"Real ({1 - cls_pred:.2f})"
                cv2.putText(vis_frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 合并原始帧和可视化结果
                combined = np.hstack((frame, vis_frame))
                out.write(combined)

                # 保存关键帧
                if save_frames and cls_pred > self.threshold:
                    save_path = Path(output_path).parent / 'frames' / \
                                f'frame_{frame_idx:04d}.png'
                    save_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(save_path), combined)

                # 收集结果
                frame_results.append({
                    'frame_idx': frame_idx,
                    'prediction': 'Fake' if cls_pred > self.threshold else 'Real',
                    'confidence': float(cls_pred)
                })

                frame_idx += 1
                pbar.update(1)

            pbar.close()
            cap.release()
            out.release()

            # 保存分析结果
            results = {
                'video_path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'frame_results': frame_results,
                'fake_frames': sum(1 for r in frame_results if r['prediction'] == 'Fake'),
                'real_frames': sum(1 for r in frame_results if r['prediction'] == 'Real')
            }

            return results

if __name__ == '__main__':
    main()