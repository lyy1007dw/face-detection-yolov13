# YOLOv13 人脸检测项目

基于 YOLOv13 的 WiderFace 人脸检测训练、验证与测试完整流程。

## 项目结构

```
algorithm/
├── dataset/                    # 数据集目录
│   ├── wider_train/            # 训练集
│   │   ├── images/
│   │   └── labels/
│   ├── wider_val/              # 验证集
│   │   ├── images/
│   │   └── labels/
│   └── wider_test/             # 测试集
│       └── images/
├── yolov13/                    # YOLOv13 项目目录
│   ├── train_widerface.py      # 训练脚本
│   ├── test_widerface.py       # 测试脚本（生成评估结果）
│   ├── widerface_evaluate/     # WiderFace 官方评估工具
│   │   ├── evaluation.py       # 评估代码
│   │   └── box_overlaps.pyx    # IOU计算（Cython）
│   └── ultralytics/
│       └── cfg/datasets/
│           └── wider_face.yaml # 数据集配置
└── README.md
```

## 环境准备

### 1. 克隆项目

```bash
git clone https://github.com/lyy1007dw/face-detection-yolov13.git
cd face-detection-yolov13
```

### 2. 下载数据集

本项目使用 [WiderFace](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 数据集，需要自行下载并配置：

1. **下载图片数据**：
   - [训练集](http://shuoyang1213.me/WIDERFACE/train.html)
   - [验证集](http://shuoyang1213.me/WIDERFACE/val.html)
   - [测试集](http://shuoyang1213.me/WIDERFACE/test.html)

2. **下载标注文件**：
   - [ground truth](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation_v1.zip)

3. **下载评估所需文件**（用于官方评估）：
   - `wider_face_val.mat`
   - `wider_easy_val.mat`
   - `wider_medium_val.mat`
   - `wider_hard_val.mat`

4. **配置目录结构**：
   ```
   dataset/
   ├── wider_train/
   │   ├── images/     # 放置训练图片
   │   └── labels/    # 放置训练标注（YOLO格式）
   ├── wider_val/
   │   ├── images/    # 放置验证图片
   │   └── labels/    # 放置验证标注
   └── wider_test/
       └── images/    # 放置测试图片
   ```

5. **标注转换**：如果标注是VOC格式，需要转换为YOLO格式（每行：`class_id x_center y_center width height`，坐标归一化到0-1）

### 3. 安装依赖

```bash
cd yolov13
pip install -r requirements.txt

# 编译 WiderFace 评估工具的 Cython 扩展
cd widerface_evaluate
python setup.py build_ext --inplace
```

## 训练

### 1. 修改数据配置

确保 `yolov13/ultralytics/cfg/datasets/wider_face.yaml` 配置正确：

```yaml
path: ./dataset

train: wider_train/images
val: wider_val/images
test: wider_test/images

nc: 1

names:
  0: face
```

### 2. 开始训练

```bash
cd yolov13
python train_widerface.py
```

或使用命令行参数自定义训练：

```bash
python train_widerface.py \
    --model yolov13n.pt \
    --data ultralytics/cfg/datasets/wider_face.yaml \
    --epochs 300 \
    --batch 16 \
    --img 640 \
    --device 0
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | yolov13n.pt | 预训练模型 (n/s/m/l/x) |
| --epochs | 300 | 训练轮数 |
| --batch | 16 | 批量大小 |
| --img | 640 | 输入图片大小 |
| --device | 0 | GPU编号，多卡用 0,1,2,3 |
| --workers | 8 | 数据加载线程数 |

### 4. 训练结果

训练完成后，结果保存在 `runs/train/yolov13n_widerface/` 目录：

```
runs/train/yolov13n_widerface/
├── weights/
│   ├── best.pt   # 最佳权重
│   └── last.pt  # 最后一次权重
├── results.csv  # 训练指标
└── ...
```

## 验证

训练过程中会自动验证，可通过以下命令单独验证：

```bash
cd yolov13
python -c "from ultralytics import YOLO; model = YOLO('runs/train/yolov13n_widerface/weights/best.pt'); results = model.val(data='ultralytics/cfg/datasets/wider_face.yaml')"
```

## 测试（生成WiderFace格式结果）

### 1. 使用训练好的权重生成预测结果

```bash
cd yolov13
python test_widerface.py --weights runs/train/yolov13n_widerface/weights/best.pt
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --weights | (必填) | 权重路径 |
| --dataset_folder | ./dataset/wider_val/images | 测试图片目录 |
| --save_folder | ./widerface_evaluate/predictions | 预测结果保存目录 |
| --img-size | 640 | 输入图片大小 |
| --conf-thres | 0.001 | 置信度阈值 |
| --iou-thres | 0.5 | IOU阈值 |

### 2. 使用官方评估工具评估

```bash
cd yolov13/widerface_evaluate
python evaluation.py -p predictions -g <ground_truth_dir>
```

> **注意**：需要下载 WiderFace 官方评估所需的 `.mat` 文件（`wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`, `wider_hard_val.mat`），放在 ground_truth 目录中。

### 3. 评估结果

```
==================== Results ====================
Easy   Val AP: 0.XXX
Medium Val AP: 0.XXX
Hard   Val AP: 0.XXX
=================================================
```

## 完整流程示例

```bash
# 1. 训练
cd yolov13
python train_widerface.py

# 2. 生成预测结果
python test_widerface.py --weights runs/train/yolov13n_widerface/weights/best.pt

# 3. 评估
cd widerface_evaluate
python evaluation.py -p predictions -g ../ground_truth/
```

## 注意事项

1. 数据集路径使用相对路径，确保从 yolov13 目录执行
2. 评估工具需要先编译 Cython 扩展
3. WiderFace 官方评估需要下载对应的 mat 文件
