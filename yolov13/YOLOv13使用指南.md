# YOLOv13 使用指南

YOLOv13是一个基于超图增强的自适应视觉感知实时目标检测模型。本指南将详细介绍如何在自定义数据集上训练YOLOv13模型。

---

## 目录

1. [环境安装](#1-环境安装)
2. [模型简介](#2-模型简介)
3. [数据集准备](#3-数据集准备)
4. [自定义数据集配置](#4-自定义数据集配置)
5. [模型训练](#5-模型训练)
6. [训练参数详解](#6-训练参数详解)
7. [模型推理与导出](#7-模型推理与导出)
8. [模型文件结构说明](#8-模型文件结构说明)
9. [常见问题解决](#9-常见问题解决)

---

## 1. 环境安装

### 1.1 基础依赖安装

```bash
# 创建虚拟环境（推荐使用Python 3.11）
conda create -n yolov13 python=3.11
conda activate yolov13

# 进入项目目录
cd yolov13

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 1.2 完整依赖列表 (requirements.txt)

```
torch==2.2.2 
torchvision==0.17.2
flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
timm==1.0.14
albumentations==2.0.4
onnx==1.14.0
onnxruntime==1.15.1
pycocotools==2.0.7
PyYAML==6.0.1
scipy==1.13.0
onnxslim==0.1.31
onnxruntime-gpu==1.18.0
gradio==4.44.1
opencv-python==4.9.0.80
psutil==5.9.8
py-cpuinfo==9.0.0
huggingface-hub==0.23.2
safetensors==0.4.3
numpy==1.26.4
supervision==0.22.0
```

### 1.3 下载预训练权重

YOLOv13提供4种模型规模的预训练权重：

| 模型 | 下载链接 |
|------|----------|
| YOLOv13-N (Nano) | [yolov13n.pt](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt) |
| YOLOv13-S (Small) | [yolov13s.pt](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt) |
| YOLOv13-L (Large) | [yolov13l.pt](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt) |
| YOLOv13-X (X-Large) | [yolov13x.pt](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt) |

```bash
# 使用脚本下载
bash ultralytics/data/scripts/download_weights.sh

# 或手动下载权重文件，放置到项目根目录
```

---

## 2. 模型简介

### 2.1 YOLOv13 核心特性

- **HyperACE**: 基于超图的自适应关联增强模块
- **FullPAD**: 全流程聚合分发范式
- **DS-based Blocks**: 基于深度可分离卷积的轻量化模块

### 2.2 模型规模对比 (MS COCO数据集)

| 模型 | FLOPs (G) | 参数量(M) | AP<sub>50:95</sub> | 延迟(ms) |
|------|-----------|-----------|---------------------|----------|
| YOLOv13-N | 6.4 | 2.5 | 41.6 | 1.97 |
| YOLOv13-S | 20.8 | 9.0 | 48.0 | 2.98 |
| YOLOv13-L | 88.4 | 27.6 | 53.4 | 8.63 |
| YOLOv13-X | 199.2 | 64.0 | 54.8 | 14.67 |

### 2.3 模型配置文件位置

```
yolov13/ultralytics/cfg/models/v13/yolov13.yaml
```

该文件定义了4种模型规模的缩放系数：

```yaml
scales:
  n: [0.50, 0.25, 1024]   # Nano
  s: [0.50, 0.50, 1024]  # Small
  l: [1.00, 1.00, 512]   # Large
  x: [1.00, 1.50, 512]   # X-Large
```

---

## 3. 数据集准备

### 3.1 数据集目录结构

YOLOv13使用与YOLO系列相同的数据集格式。建议按照以下结构组织数据集：

```
your_dataset/
├── images/
│   ├── train/          # 训练图像
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/            # 验证图像
│   │   ├── image1.jpg
│   │   └── ...
│   └── test/            # 测试图像（可选）
│       └── ...
├── labels/
│   ├── train/          # 训练标签
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/            # 验证标签
│   │   ├── image1.txt
│   │   └── ...
│   └── test/           # 测试标签（可选）
│       └── ...
```

### 3.2 标注格式 (YOLO格式)

每个图像对应一个`.txt`文件，格式如下：

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 类别ID（从0开始）
- `x_center`, `y_center`: 归一化的中心坐标（0-1）
- `width`, `height`: 归一化的宽高（0-1）

**示例** (`image1.txt`)：
```
0 0.5 0.5 0.3 0.4    # 类别0，中心(0.5,0.5)，宽0.3高0.4
1 0.2 0.3 0.1 0.15   # 类别1，中心(0.2,0.3)，宽0.1高0.15
```

### 3.3 标注工具推荐

- [LabelImg](https://github.com/tzutalin/labelImg) - 开源图像标注工具
- [CVAT](https://github.com/opencv/cvat) - 在线标注平台
- [Labelbox](https://labelbox.com/) - 商业标注平台

---

## 4. 自定义数据集配置

### 4.1 创建数据集配置文件

在 `ultralytics/cfg/datasets/` 目录下创建你的数据集配置文件，例如 `my_dataset.yaml`：

```yaml
# Ultralytics YOLOv13 数据集配置文件

# 数据集根目录（建议使用绝对路径）
path: E:/biyesheji/algorithm/my_dataset

# 训练、验证、测试集路径（相对于path）
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 3

# 类别名称（必须与nc数量匹配）
names:
  0: person
  1: car
  2: dog
```

### 4.2 配置文件参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `path` | 数据集根目录的绝对路径 | `E:/biyesheji/algorithm/my_dataset` |
| `train` | 训练集图像相对路径 | `images/train` |
| `val` | 验证集图像相对路径 | `images/val` |
| `test` | 测试集图像相对路径（可选） | `images/test` |
| `nc` | 类别数量 | `3` |
| `names` | 类别名称字典 | `{0: person, 1: car}` |

### 4.3 路径说明

路径可以使用以下几种方式：

1. **相对路径**（相对于path）：
   ```yaml
   path: E:/datasets/my_dataset
   train: images/train
   ```

2. **绝对路径**：
   ```yaml
   path: E:/datasets/my_dataset
   train: E:/datasets/my_dataset/images/train
   ```

3. **文件列表**（使用.txt文件）：
   ```yaml
   path: E:/datasets
   train: train.txt  # 文件包含所有训练图像路径
   ```

---

## 5. 模型训练

### 5.1 基础训练命令

```python
from ultralytics import YOLO

# 加载模型（使用预训练权重）
model = YOLO('yolov13n.pt')

# 训练模型
results = model.train(
    data='my_dataset.yaml',  # 数据集配置文件
    epochs=100,              # 训练轮数
    batch=16,                # batch大小
    imgsz=640,               # 输入图像大小
    device='0',              # GPU设备编号
)
```

### 5.2 完整训练示例

```python
from ultralytics import YOLO

# 使用预训练模型训练
model = YOLO('yolov13n.pt')

# 训练参数配置
results = model.train(
    # ========== 数据配置 ==========
    data='my_dataset.yaml',        # 数据集配置路径
    epochs=600,                     # 总训练轮数
    batch=256,                     # batch大小（根据显存调整）
    imgsz=640,                     # 输入图像尺寸
    patience=100,                  # 早停耐心值
    
    # ========== 模型保存 ==========
    project='runs/detect',         # 项目保存目录
    name='yolov13_custom',          # 实验名称
    exist_ok=True,                  # 是否覆盖已有实验
    save=True,                      # 保存训练结果
    save_period=10,                 # 每10轮保存一次
    
    # ========== 训练策略 ==========
    scale=0.5,                     # 图像缩放比例 (N:0.5, S:0.9, L:0.9, X:0.9)
    mosaic=1.0,                    # mosaic数据增强概率
    mixup=0.0,                     # mixup数据增强概率 (S:0.05, L:0.15, X:0.2)
    copy_paste=0.1,                # copy_paste增强概率 (S:0.15, L:0.5, X:0.6)
    
    # ========== 硬件配置 ==========
    device='0',                    # GPU设备，可多卡: '0,1,2,3'
    workers=8,                     # 数据加载线程数
    amp=True,                      # 启用AMP混合精度训练
    
    # ========== 其他 ==========
    verbose=True,                  # 详细输出
    seed=0,                        # 随机种子
)
```

### 5.3 命令行训练方式

```bash
# 使用预训练权重训练
yolo detect train data=my_dataset.yaml model=yolov13n.pt epochs=100 batch=16 imgsz=640 device=0

# 从头训练（不使用预训练权重）
yolo detect train data=my_dataset.yaml model=yolov13n.yaml epochs=600 batch=256 device=0
```

### 5.4 不同规模模型的训练建议

| 模型 | 推荐batch | 推荐epochs | 推荐scale | 推荐mixup | 推荐copy_paste |
|------|-----------|------------|-----------|-----------|----------------|
| YOLOv13-N | 128-256 | 600 | 0.5 | 0.0 | 0.1 |
| YOLOv13-S | 64-128 | 600 | 0.9 | 0.05 | 0.15 |
| YOLOv13-L | 16-32 | 600 | 0.9 | 0.15 | 0.5 |
| YOLOv13-X | 8-16 | 600 | 0.9 | 0.2 | 0.6 |

---

## 6. 训练参数详解

### 6.1 参数文件位置

所有训练参数定义在以下文件：
- **主要参数**: `ultralytics/cfg/default.yaml`
- **模型结构**: `ultralytics/cfg/models/v13/yolov13.yaml`

### 6.2 完整训练参数说明

#### 数据配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data` | - | 数据集配置文件路径 |
| `epochs` | 100 | 训练轮数 |
| `batch` | 16 | batch大小，-1为自动调整 |
| `imgsz` | 640 | 输入图像尺寸 |
| `cache` | False | 数据缓存方式: True/ram, disk, False |
| `rect` | False | 矩形训练模式 |
| `fraction` | 1.0 | 使用数据集的比例 |

#### 模型保存参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | - | 模型配置文件或权重路径 |
| `project` | - | 项目保存目录 |
| `name` | - | 实验名称 |
| `exist_ok` | False | 是否覆盖已有实验 |
| `save` | True | 保存训练检查点 |
| `save_period` | -1 | 每N轮保存一次检查点 |
| `pretrained` | True | 是否使用预训练权重 |

#### 硬件配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | - | 设备: 0,1,2,3 或 cpu |
| `workers` | 8 | 数据加载线程数 |
| `amp` | True | 启用AMP混合精度 |
| `multi_scale` | False | 多尺度训练 |

#### 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `optimizer` | auto | 优化器: SGD, Adam, AdamW, RMSProp |
| `lr0` | 0.01 | 初始学习率 |
| `lrf` | 0.01 | 最终学习率 (lr0 * lrf) |
| `momentum` | 0.937 | SGD动量 / Adam beta1 |
| `weight_decay` | 0.0005 | 权重衰减 |
| `warmup_epochs` | 3.0 | 预热轮数 |
| `warmup_momentum` | 0.8 | 预热动量 |
| `warmup_bias_lr` | 0.0 | 预热偏置学习率 |

#### 损失函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `box` | 7.5 | 边界框损失权重 |
| `cls` | 0.5 | 分类损失权重 |
| `dfl` | 1.5 | DFL损失权重 |
| `nbs` | 64 | 标称batch大小 |

#### 数据增强参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hsv_h` | 0.015 | HSV色调增强比例 |
| `hsv_s` | 0.7 | HSV饱和度增强比例 |
| `hsv_v` | 0.4 | HSV亮度增强比例 |
| `degrees` | 0.0 | 旋转角度范围 |
| `translate` | 0.1 | 平移比例 |
| `scale` | 0.5 | 缩放比例 |
| `shear` | 0.0 | 剪切角度 |
| `perspective` | 0.0 | 透视变换 |
| `flipud` | 0.0 | 上下翻转概率 |
| `fliplr` | 0.5 | 左右翻转概率 |
| `mosaic` | 1.0 | mosaic增强概率 |
| `mixup` | 0.0 | mixup增强概率 |
| `copy_paste` | 0.1 | copy_paste增强概率 |

#### 验证参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `val` | True | 训练时验证 |
| `split` | val | 验证集划分 |
| `conf` | - | 置信度阈值 |
| `iou` | 0.7 | NMS IoU阈值 |
| `max_det` | 300 | 每张图最大检测数 |
| `plots` | True | 保存训练曲线 |

---

## 7. 模型推理与导出

### 7.1 模型验证

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/yolov13_custom/weights/best.pt')

# 在验证集上评估
metrics = model.val(data='my_dataset.yaml')

# 打印评估指标
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP75: {metrics.box.map75:.3f}")
```

### 7.2 模型预测

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/yolov13_custom/weights/best.pt')

# 预测图像
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,           # 置信度阈值
    iou=0.7,            # NMS IoU阈值
    save=True,          # 保存结果
    show=True,          # 显示结果
)

# 访问检测结果
for r in results:
    boxes = r.boxes    # 边界框
    classes = boxes.cls # 类别
    confs = boxes.conf # 置信度
    xyxy = boxes.xyxy  # 坐标 (x1, y1, x2, y2)
```

### 7.3 模型导出

支持多种导出格式：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolov13_custom/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx')

# 导出为 TensorRT (需要安装TensorRT)
model.export(format='engine', half=True)

# 导出为 TFLite
model.export(format='tflite')

# 导出为 CoreML
model.export(format='coreml')

# 导出为 TorchScript
model.export(format='torchscript')
```

### 7.4 命令行导出

```bash
# ONNX格式
yolo export model=runs/detect/yolov13_custom/weights/best.pt format=onnx

# TensorRT格式
yolo export model=runs/detect/yolov13_custom/weights/best.pt format=engine half=True

# TorchScript格式
yolo export model=runs/detect/yolov13_custom/weights/best.pt format=torchscript
```

---

## 8. 模型文件结构说明

### 8.1 项目目录结构

```
yolov13/
├── ultralytics/
│   ├── __init__.py
│   ├── cfg/                          # 配置文件目录
│   │   ├── default.yaml              # 默认训练参数
│   │   ├── models/
│   │   │   └── v13/
│   │   │       └── yolov13.yaml      # 模型结构配置
│   │   └── datasets/
│   │       ├── coco.yaml             # COCO数据集配置
│   │       └── my_dataset.yaml       # 自定义数据集配置
│   ├── data/                         # 数据处理相关
│   ├── models/                       # 模型定义
│   ├── nn/                          # 神经网络模块
│   ├── engine/                       # 训练引擎
│   ├── utils/                        # 工具函数
│   └── ...
├── examples/                         # 示例代码
├── assets/                           # 资源文件
├── requirements.txt                  # 依赖列表
└── README.md
```

### 8.2 训练输出目录结构

训练完成后，结果保存在 `project/name` 目录下：

```
runs/detect/yolov13_custom/
├── weights/
│   ├── best.pt          # 最佳模型权重
│   └── last.pt          # 最后一轮权重
├── args.yaml           # 训练参数配置
├── results.csv         # 训练结果日志
├── results.png         # 训练曲线
├── train_batch*.jpg    # 训练样本可视化
├── val_batch*_pred.jpg # 验证预测可视化
└── ...
```

---

## 9. 常见问题解决

### 9.1 显存不足 (OOM)

**问题**: 训练时出现 CUDA out of memory 错误

**解决方案**:
```python
# 1. 减小batch大小
model.train(data='my_dataset.yaml', batch=8, imgsz=640)

# 2. 减小图像尺寸
model.train(data='my_dataset.yaml', batch=16, imgsz=416)

# 3. 启用梯度累积
model.train(data='my_dataset.yaml', batch=8, imgsz=640, accumulation=4)

# 4. 关闭AMP混合精度
model.train(data='my_dataset.yaml', amp=False)
```

### 9.2 数据加载慢

**解决方案**:
```python
# 使用RAM缓存
model.train(data='my_dataset.yaml', cache='ram')

# 或使用磁盘缓存
model.train(data='my_dataset.yaml', cache='disk')

# 增加数据加载线程
model.train(data='my_dataset.yaml', workers=16)
```

### 9.3 训练不收敛

**可能原因与解决方案**:

1. **学习率不合适**
   ```python
   # 调整学习率
   model.train(data='my_dataset.yaml', lr0=0.001, lrf=0.01)
   ```

2. **数据增强过度**
   ```python
   # 减少数据增强
   model.train(data='my_dataset.yaml', mosaic=0.5, mixup=0.0)
   ```

3. **batch太小**
   ```python
   # 增加batch大小
   model.train(data='my_dataset.yaml', batch=32)
   ```

### 9.4 验证集评估指标低

**检查清单**:
1. 确认标注是否正确
2. 检查train/val数据划分是否合理
3. 增加训练轮数
4. 调整模型规模（尝试更大的模型）
5. 调整数据增强参数
6. 检查是否存在数据泄露

### 9.5 多GPU训练

```python
# 使用多GPU训练
model.train(
    data='my_dataset.yaml',
    device='0,1,2,3',  # 指定多个GPU
    batch=64           # 总batch会自动分配到各GPU
)
```

### 9.6 断点续训

```python
# 从上次中断的地方继续训练
model.train(
    data='my_dataset.yaml',
    resume=True  # 自动从last.pt恢复
)
```

---

## 10. 进阶使用技巧

### 10.1 模型微调策略

```python
from ultralytics import YOLO

# 方法1: 使用预训练权重微调
model = YOLO('yolov13n.pt')
model.train(
    data='my_dataset.yaml',
    epochs=100,
    freeze=10,  # 冻结前10层
)

# 方法2: 解冻后微调
model.train(
    data='my_dataset.yaml',
    epochs=50,
    freeze=0,   # 解冻所有层
    lr0=0.001,  # 降低学习率
)
```

### 10.2 自定义数据增强

```python
from ultralytics import YOLO

# 使用Albumentations增强
model = YOLO('yolov13n.pt')
model.train(
    data='my_dataset.yaml',
    # 自定义增强参数
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.5,
)
```

### 10.3 训练监控

使用TensorBoard监控训练过程：

```bash
# 启动TensorBoard
tensorboard --logdir runs/detect
```

或使用wandb：

```python
# 使用wandb监控
model.train(
    data='my_dataset.yaml',
    project='yolov13',
    name='my_experiment',
    logger='wandb',
)
```

---

## 附录

### A. 参考资源

- [YOLOv13论文](https://arxiv.org/abs/2506.17733)
- [Ultralytics官方文档](https://docs.ultralytics.com/)
- [HuggingFace Demo](https://huggingface.co/spaces/atalaydenknalbant/Yolov13)

### B. 快速训练命令参考

```bash
# 快速测试（使用coco8小数据集）
yolo train data=coco8.yaml model=yolov13n.pt epochs=10

# 标准训练（使用自己的数据集）
yolo train data=my_dataset.yaml model=yolov13n.pt epochs=100 batch=16 device=0

# 高性能训练（多GPU）
yolo train data=my_dataset.yaml model=yolov13s.pt epochs=300 batch=64 device=0,1,2,3
```

---

**注意**: 本指南基于YOLOv13官方代码编写，参数和行为可能随版本更新而变化。建议在使用前查阅最新的官方文档。
