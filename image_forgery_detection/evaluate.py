import argparse
import torch
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from src.data_processing.dataset import CASIADataset, FaceForensicsDataset
from src.models.base_model import BaseForgeryDetector
from src.evaluation.metrics import BinaryClassificationMetrics, SegmentationMetrics
from src.evaluation.visualizer import VisualizationManager
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output-dir', type=str, default='outputs/results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    return parser.parse_args()


def evaluate(model, data_loader, device, output_dir):
    model.eval()

    # 初始化评估器
    cls_metrics = BinaryClassificationMetrics()
    seg_metrics = SegmentationMetrics(num_classes=2)
    vis_manager = VisualizationManager(output_dir)

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            cls_targets = batch['cls_target'].to(device)
            seg_targets = batch.get('seg_target')
            if seg_targets is not None:
                seg_targets = seg_targets.to(device)

            # 前向传播
            outputs = model(images)

            # 更新指标
            cls_metrics.update(outputs['cls'], cls_targets)
            if seg_targets is not None and 'seg' in outputs:
                seg_metrics.update(outputs['seg'], seg_targets)

            # 收集预测结果
            predictions.extend(outputs['cls'].argmax(1).cpu().numpy())
            targets.extend(cls_targets.cpu().numpy())

            # 可视化一些结果
            if len(predictions) <= 16:  # 只保存前16个样本的可视化结果
                vis_manager.visualize_predictions(
                    images.cpu().numpy(),
                    seg_targets.cpu().numpy() if seg_targets is not None else None,
                    outputs['seg'].cpu().numpy() if 'seg' in outputs else None
                )

    # 计算指标
    cls_results = cls_metrics.compute()
    seg_results = seg_metrics.compute() if seg_targets is not None else {}

    # 合并结果
    results = {
        'classification': cls_results,
        'segmentation': seg_results
    }

    # 保存结果
    save_path = Path(output_dir) / 'evaluation_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    # 绘制评估曲线
    vis_manager.plot_roc_curve(targets, predictions)
    vis_manager.plot_pr_curve(targets, predictions)
    vis_manager.plot_confusion_matrix(targets, predictions)

    return results


def main():
    args = parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 创建测试数据加载器
    test_dataset = CASIADataset(
        config['datasets']['casia']['test_path']
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['dataloader']['batch_size']['test'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    # 创建模型并加载权重
    model = BaseForgeryDetector.create(config['model']['name'], config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评估模型
    results = evaluate(model, test_loader, device, output_dir)

    # 打印结果
    print("\nEvaluation Results:")
    print("=" * 50)
    for task, metrics in results.items():
        print(f"\n{task.capitalize()} Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == '__main__':
    main()