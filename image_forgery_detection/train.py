from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch 
from torch.utils.data import DataLoader

from src.data_processing.dataset import CASIADataset, FaceForensicsDataset
from src.models.base_model import BaseForgeryDetector
from src.training.trainer import Trainer  # 使用已有的Trainer实现
from src.training.optimizer import build_optimizer
from src.training.scheduler import create_scheduler 
from src.models.losses import build_criterion
from src.utils.logger import setup_logger

logger = setup_logger('train')

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Train forgery detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--model-type', type=str, default='efficientnet', help='模型类型')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    
    return parser.parse_args()

def create_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    # 创建数据集
    if 'casia' in config['datasets']:
        train_dataset = CASIADataset(
            config['datasets']['casia']['train_path'],
            transform=config['augmentation']
        )
        val_dataset = CASIADataset(
            config['datasets']['casia']['val_path']
        )
    elif 'faceforensics' in config['datasets']:
        train_dataset = FaceForensicsDataset(
            config['datasets']['faceforensics']['train_path'],
            transform=config['augmentation']  
        )
        val_dataset = FaceForensicsDataset(
            config['datasets']['faceforensics']['val_path']
        )
    else:
        raise ValueError("No supported dataset found in config")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size']['train'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size']['val'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    return train_loader, val_loader

def main() -> None:
    """主函数"""
    try:
        # 解析参数
        args = parse_args()

        # 加载配置
        with open(args.config) as f:
            config = yaml.safe_load(f)

        # 设置设备
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')

        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(config)
        logger.info(f'Dataset size - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}')

        # 创建模型
        model = BaseForgeryDetector.create(args.model_type, config).to(device)
        logger.info(f'Created model: {args.model_type}')

        # 创建损失函数
        criterion = build_criterion(config['loss'])

        # 创建优化器
        optimizer = build_optimizer(model.parameters(), config['optimizer'])

        # 创建学习率调度器
        scheduler = create_scheduler(optimizer, config['scheduler'])

        # 创建训练器
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device
        )

        # 恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f'Resumed training from: {args.resume}')

        # 开始训练
        trainer.train(train_loader, val_loader)
        logger.info('Training completed')

    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
    except Exception as e:
        logger.error(f'Training failed: {e}')
        raise

if __name__ == '__main__':
    main()
