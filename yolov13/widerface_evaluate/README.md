# WiderFace-Evaluation

WiderFace 数据集人脸检测评估代码

## 项目简介

本项目用于评估人脸检测模型在 [Wider Face Dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 数据集上的表现。评估支持三种难度级别：

- **Easy**: 清晰可见的人脸
- **Medium**: 正常大小的人脸  
- **Hard**: 模糊或遮挡的人脸

评估指标为 **AP (Average Precision)**。

## 环境要求

- Python 3.x
- NumPy
- SciPy
- Cython
- tqdm
- ipython

安装依赖：
```bash
pip install numpy scipy cython tqdm ipython
```

## 使用方法

### 1. 编译 Cython 扩展

```bash
python3 setup.py build_ext --inplace
```

### 2. 运行评估

```bash
python3 evaluation.py -p <你的预测结果目录> -g <真实标注目录>
```

### 3. 预测结果格式

预测结果目录结构如下：
```
预测结果目录/
├── 事件名称1/
│   ├── 图片1.txt
│   ├── 图片2.txt
│   └── ...
├── 事件名称2/
│   └── ...
```

每个 `.txt` 文件格式如下：
```
图片路径
检测框数量
x1 y1 x2 y2 score
x1 y1 x2 y2 score
...
```

**注意**：坐标格式为 (x1, y1, width, height)，最后一项为置信度分数。

### 4. 真实标注文件

真实标注目录需包含以下 `.mat` 文件（可在 WiderFace 官网下载）：
- `wider_face_val.mat` - 验证集标注
- `wider_easy_val.mat` - 简单样本标注
- `wider_medium_val.mat` - 中等样本标注
- `wider_hard_val.mat` - 困难样本标注

### 5. 示例

```bash
python3 evaluation.py -p ./predictions -g ./ground_truth/
```

## 输出说明

评估完成后会输出三个类别的 AP 值：
```
==================== Results ====================
Easy   Val AP: 0.XXX
Medium Val AP: 0.XXX
Hard   Val AP: 0.XXX
=================================================
```

AP 值范围为 0-1，值越高表示检测效果越好。

## 文件说明

- `evaluation.py` - 主评估代码
- `box_overlaps.pyx` - IOU计算（Cython实现）
- `setup.py` - 编译配置

## 参考

部分代码参考了 Sergey Karayev 的工作
