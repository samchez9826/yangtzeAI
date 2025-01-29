import argparse
import torch
import yaml
from pathlib import Path
from src.data_processing.dataset import CASIADataset, FaceForensicsDataset
from src.models.base_model import BaseForgeryDetector
from src.training.trainer import Trainer
from src.training.optimizer import build_optimizer
from src.training.scheduler import create_scheduler
from src.models.losses import build_criterion
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--model-type', type=str, default='efficientnet', help='模型类型')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 创建数据加载器
    train_dataset = CASIADataset(
        config['datasets']['casia']['train_path'],
        transform=config['augmentation']
    )
    val_dataset = CASIADataset(
        config['datasets']['casia']['val_path']
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size']['train'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size']['val'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )

    # 创建模型
    model = BaseForgeryDetector.create(args.model_type, config).to(device)

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

    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()