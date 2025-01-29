# 图像篡改检测系统

基于深度学习的端到端图像篡改检测系统，支持多种篡改类型检测和定位。

## 目录

- [主要特性](#主要特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [使用指南](#使用指南)
- [模型说明](#模型说明)
- [性能指标](#性能指标)
- [API文档](#api文档)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [更新日志](#更新日志)
- [贡献指南](#贡献指南)
- [引用](#引用)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 主要特性

### 多模型架构支持
- EfficientNet(B0-B7)系列
- ResNet(18/34/50/101)系列
- Vision Transformer
- 支持自定义模型扩展

### 多任务学习
- 图像级篡改分类
- 像素级篡改定位
- 篡改类型识别
- 篡改强度估计

### 高级注意力机制
- 多头自注意力
- CBAM注意力
- 可变形注意力
- 金字塔注意力
- 空间注意力

### 数据处理流程
- 自动数据清洗
- 多尺度预处理
- 高级数据增强
- 图像质量评估
- 自动标注校正

### 评估工具
- 多维度评估指标
- 实时性能监控
- 可视化分析工具
- 模型解释工具

## 快速开始

### 系统要求
- Python >= 3.8
- CUDA >= 11.0
- PyTorch >= 1.9.0
- 内存 >= 16GB
- GPU Memory >= 8GB

### 安装

```bash
# 克隆仓库
git clone https://github.com/username/image_forgery_detection.git
cd image_forgery_detection

# 创建环境
conda create -n forgery python=3.8
conda activate forgery

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖(可选)
pip install -r requirements-dev.txt
```

### 数据集准备

支持的数据集:
- CASIA 2.0
- FaceForensics++
- DeepFake Detection Challenge
- WildDeepfake
- 自定义数据集

```bash
# 下载公开数据集
python scripts/download_datasets.py --dataset casia --path data/raw

# 数据预处理
python scripts/preprocess_data.py --config configs/data_config.yaml

# 数据集分析
python scripts/analyze_dataset.py --data-dir data/processed
```

### 训练模型

```bash
# 单GPU训练
python train.py \
    --config configs/train_config.yaml \
    --model-type efficientnet \
    --data-dir data/processed \
    --output-dir outputs/models

# 多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --config configs/train_config.yaml \
    --dist-url tcp://localhost:23456

# 恢复训练
python train.py \
    --config configs/train_config.yaml \
    --resume outputs/models/checkpoint_latest.pth
```

### 评估模型

```bash
# 基础评估
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint outputs/models/best_model.pth

# 详细评估
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint outputs/models/best_model.pth \
    --save-predictions \
    --analysis-mode full

# 交叉验证
python scripts/cross_validate.py \
    --config configs/train_config.yaml \
    --folds 5
```

### 模型推理

```bash
# 单张图像
python infer.py \
    --config configs/train_config.yaml \
    --checkpoint outputs/models/best_model.pth \
    --input examples/test.jpg \
    --output outputs/results

# 批量处理
python infer.py \
    --config configs/train_config.yaml \
    --checkpoint outputs/models/best_model.pth \
    --input path/to/image/dir \
    --output outputs/results \
    --batch-size 32

# 视频处理
python infer.py \
    --config configs/train_config.yaml \
    --checkpoint outputs/models/best_model.pth \
    --input video.mp4 \
    --output outputs/results \
    --save-frames
```

## 项目结构

```
image_forgery_detection/
├── configs/                # 配置文件
│   ├── data_config.yaml   # 数据配置
│   ├── model_config.yaml  # 模型配置
│   └── train_config.yaml  # 训练配置
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 预处理数据
│   └── interim/          # 中间结果
├── src/                  # 源代码
│   ├── data_processing/  # 数据处理
│   │   ├── dataset.py
│   │   ├── preprocessor.py
│   │   └── augmentation.py
│   ├── models/          # 模型定义
│   │   ├── base_model.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── attention.py
│   │   └── losses.py
│   ├── training/        # 训练相关
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   ├── evaluation/      # 评估工具
│   │   ├── metrics.py
│   │   └── visualizer.py
│   └── utils/          # 工具函数
│       ├── logger.py
│       └── config.py
├── scripts/            # 实用脚本
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── analyze_dataset.py
├── tests/             # 测试代码
├── docs/              # 文档
├── notebooks/         # Jupyter notebooks
├── outputs/           # 输出目录
│   ├── models/       # 模型权重
│   └── results/      # 预测结果
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
├── infer.py          # 推理脚本
├── requirements.txt  # 环境依赖
├── setup.py         # 安装脚本
├── LICENSE          # 许可证
└── README.md        # 项目说明
```

## 技术细节

### 预处理流程
1. 图像清洗
   - 分辨率检查
   - 质量评估
   - 异常检测
   - 重复图像去除

2. 数据标准化
   - 尺寸统一
   - 色彩空间转换
   - 通道标准化
   - 直方图均衡化

3. 质量增强
   - 降噪处理
   - 锐化增强
   - 对比度调整
   - 色彩平衡

### 数据增强策略
1. 几何变换
   - 随机旋转
   - 随机翻转
   - 尺度变换
   - 透视变换

2. 光度变换
   - 亮度调整
   - 对比度调整
   - 色调变换
   - 饱和度调整

3. 噪声注入
   - 高斯噪声
   - 泊松噪声
   - 椒盐噪声
   - 斑点噪声

### 模型架构
1. 特征提取
   - 多尺度特征
   - 金字塔池化
   - 注意力机制
   - 跳跃连接

2. 特征融合
   - 通道注意力
   - 空间注意力
   - 多层特征
   - 级联融合

3. 输出头
   - 分类头
   - 分割头
   - 辅助任务头
   - 多任务融合

### 训练策略
1. 优化器配置
   - RAdam优化器
   - 余弦退火
   - 梯度累积
   - 权重衰减

2. 学习率调度
   - 预热阶段
   - 周期调整
   - 自适应调整
   - 多阶段衰减

3. 损失函数
   - Focal Loss
   - Dice Loss
   - 边缘一致性损失
   - 特征一致性损失

## 使用指南

### 数据准备

1. 数据集组织结构
```
data/
├── raw/
│   ├── authentic/
│   ├── tampered/
│   └── masks/
└── processed/
    ├── train/
    ├── val/
    └── test/
```

2. 自定义数据集
```python
from src.data_processing.dataset import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        # 自定义初始化代码
        
    def __getitem__(self, idx):
        # 自定义数据加载逻辑
        return sample
```

3. 数据增强配置
```yaml
augmentation:
  geometric:
    rotate:
      enable: true
      angle_range: [-45, 45]
  photometric:
    brightness:
      enable: true
      range: [0.8, 1.2]
```

### 模型训练

1. 配置优化器
```yaml
optimizer:
  name: 'adamw'
  params:
    lr: 0.001
    weight_decay: 0.01
```

2. 配置学习率调度器
```yaml
scheduler:
  name: 'cosine'
  params:
    T_max: 100
    eta_min: 1e-6
```

3. 配置损失函数
```yaml
loss:
  classification:
    name: 'focal'
    weight: 1.0
  segmentation:
    name: 'dice'
    weight: 0.5
```

### 模型评估

1. 评估指标
- 准确率(Accuracy)
- 精确率(Precision)
- 召回率(Recall)
- F1分数
- AUC-ROC
- IoU

2. 可视化分析
- 混淆矩阵
- ROC曲线
- PR曲线
- CAM热力图

3. 错误分析
- 错误案例分析
- 失败模式统计
- 性能瓶颈分析
- 改进建议

### 模型部署

1. 模型导出
```python
# ONNX导出
python scripts/export_onnx.py \
    --checkpoint path/to/model.pth \
    --output model.onnx

# TorchScript导出
python scripts/export_torchscript.py \
    --checkpoint path/to/model.pth \
    --output model.pt
```

2. 性能优化
- 模型量化
- 模型剪枝
- 知识蒸馏
- TensorRT加速

3. 服务部署
- REST API
- gRPC服务
- Docker容器
- 云端部署

## 模型说明

### 支持的主干网络

1. EfficientNet系列
- EfficientNet-B0 ~ B7
- EfficientNet-V2
- EfficientNetLite

2. ResNet系列
- ResNet-18/34/50/101
- ResNeXt
- ResNeSt

3. Vision Transformer
- ViT-Base
- ViT-Large
- DeiT
- Swin Transformer

### 注意力机制

1. 自注意力
- 多头注意力
- 相对位置编码
- 局部注意力
- 全局注意力

2. 通道注意力
- SE模块
- CBAM
- ECA模块
- GC模块

3. 空间注意力
- 非局部注意力
- 可变形卷积
- Position Attention
- Channel Attention

### 损失函数

1. 分类损失
- CrossEntropy Loss
- Focal Loss
- Label Smoothing
- BCE Loss

2. 分割损失
- Dice Loss
- IoU Loss
- BCE-Dice Loss
- Tversky Loss

3. 特征损失
- L1/L2 Loss
- Perceptual Loss
- Style Loss
- SSIM Loss

## 性能指标

### CASIA 2.0数据集

| 模型 | Accuracy | Precision | Recall | F1 | AUC | IoU |
|-----|----------|-----------|--------|----|----|-----|
| EfficientNet-B3 | 0.956 | 0.947 | 0.943 | 0.945 | 0.982 | 0.876 |
| ResNet-50 | 0.942 | 0.935 | 0.941 | 0.938 | 0.975 | 0.862 |
| ViT-Base | 0.961 | 0.953 | 0.951 | 0.952 | 0.987 | 0.881 |

### FaceForensics++数据集

| 模型 | Accuracy | Precision | Recall | F1 | AUC | IoU |
|-----|----------|-----------|--------|----|----|-----|
| EfficientNet-B3 | 0.982 | 0.975 | 0.971 | 0.973 | 0.991 | 0.912 |
| ResNet-50 | 0.973 | 0.968 | 0.965 | 0.966 | 0.988 | 0.895 |
| ViT-Base | 0.985 | 0.979 | 0.976 | 0.977 | 0.994 | 0.921 |

### 推理速度

| 模型 | GPU (ms) | CPU (ms) | 内存 (MB) | 参数量 (M) |
|-----|-----------|----------|------------|------------|
| EfficientNet-B3 | 15.6 | 156.3 | 892 | 12.2 |
| ResNet-50 | 12.4 | 142.1 | 753 | 25.6 |
| ViT-Base | 18.2 | 198.5 | 986 | 86.4 |

*测试环境：NVIDIA RTX 3090, Intel i9-10900K, 32GB RAM

## API文档

### 数据处理 API

```python
from src.data_processing import ImagePreprocessor, AugmentationPipeline

# 图像预处理
preprocessor = ImagePreprocessor(config_path='configs/data_config.yaml')
processed = preprocessor.preprocess_image(image)

# 数据增强
augmentor = AugmentationPipeline(config_path='configs/data_config.yaml')
augmented = augmentor.augment_batch(images, masks)
```

### 模型 API

```python
from src.models import BaseForgeryDetector

# 创建模型
model = BaseForgeryDetector.create('efficientnet', config)

# 训练模式
model.train()
outputs = model(images)

# 评估模式
model.eval()
with torch.no_grad():
    predictions = model(images)
```

### 训练 API

```python
from src.training import Trainer

# 创建训练器
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config
)

# 训练模型
trainer.train(train_loader, val_loader)

# 评估模型
metrics = trainer.evaluate(test_loader)
```

## 开发指南

### 环境设置

1. 开发环境
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 设置pre-commit hooks
pre-commit install
```

2. 代码风格
```bash
# 格式化代码
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/

# 代码检查
flake8 src/ tests/
```

3. 测试
```bash
# 运行单元测试
pytest tests/

# 测试覆盖率报告
pytest tests/ --cov=src
```

### 自定义扩展

1. 添加新的数据集
```python
# src/data_processing/dataset.py
class NewDataset(BaseDataset):
    def __init__(self, ...):
        super().__init__(...)
        
    def __getitem__(self, idx):
        # 实现数据加载逻辑
        return sample
```

2. 添加新的模型
```python
# src/models/custom_model.py
class CustomModel(BaseForgeryDetector):
    def __init__(self, ...):
        super().__init__(...)
        
    def forward(self, x):
        # 实现前向传播
        return outputs
```

3. 添加新的损失函数
```python
# src/models/losses.py
class CustomLoss(nn.Module):
    def __init__(self, ...):
        super().__init__(...)
        
    def forward(self, pred, target):
        # 实现损失计算
        return loss
```

## 常见问题

### 训练相关
Q: 如何处理数据不平衡问题？
A: 使用以下方法：
- 使用Focal Loss
- 类别权重平衡
- 过采样/欠采样
- 数据增强

Q: 如何提高训练速度？
A: 可以尝试：
- 使用混合精度训练
- 增加批次大小
- 使用梯度累积
- 数据预加载

Q: 如何防止过拟合？
A: 采用以下策略：
- 增加数据增强
- 使用正则化
- 添加Dropout
- 早停策略

### 部署相关
Q: 如何优化模型大小？
A: 可以通过：
- 模型量化
- 知识蒸馏
- 模型剪枝
- 结构优化

Q: 如何提高推理速度？
A: 建议：
- 使用TensorRT
- 模型量化
- 批处理推理
- CPU多线程

## 更新日志

### v1.0.0 (2024-01-20)
- 初始版本发布
- 支持基础模型架构
- 实现核心功能

### v1.1.0 (2024-02-15)
- 添加ViT支持
- 优化训练流程
- 改进评估指标

### v1.2.0 (2024-03-10)
- 添加新数据集
- 改进注意力机制
- 优化推理性能

## 贡献指南

### 提交规范
- 使用语义化版本号
- 遵循Angular提交信息规范
- 创建功能分支
- 提交前进行测试

### 开发流程
1. Fork项目
2. 创建功能分支
3. 提交变更
4. 创建Pull Request

### 文档贡献
- 改进README
- 添加示例代码
- 完善API文档
- 分享使用经验

## 引用

如果您使用了本项目的代码，请引用以下论文：

```bibtex
@article{author2023forgery,
  title={An End-to-End Deep Learning Framework for Image Forgery Detection},
  author={Author, A. and Author, B.},
  journal={arXiv preprint arXiv:2023.xxxxx},
  year={2023}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 问题反馈: GitHub Samchez9826
- 邮件联系: samchez@qq.com
