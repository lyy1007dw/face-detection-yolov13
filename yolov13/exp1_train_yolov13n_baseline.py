import gc
import torch
from ultralytics import YOLO
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OMP_NUM_THREADS'] = '16'

# ==========================
# 配置参数
# ==========================
MODEL = 'yolov13n.pt'  # nano版本，速度快
DATA = 'ultralytics/cfg/datasets/wider_face.yaml'  # 数据集配置
EPOCHS = 150  # 训练轮数
BATCH_SIZE = 16
IMG_SIZE = 640  # 人脸检测推荐高分辨率
DEVICE = '0'
WORKERS = 8  # 实测workers=8稳定，16可能导致不稳定
PROJECT = 'runs/train'
NAME = 'exp1_yolov13n_baseline'  # 优化阶段
RESUME = False
FP16 = True  # Ada架构完全支持FP16

# ==========================
# 数据增强（人脸检测专用）
# ==========================
AUGMENT = {
    'mosaic': 1.0,  # 对小脸检测帮助大
    'mixup': 0.0,  # 关闭，人脸不适合
    'copy_paste': 0.2,  # 制造更多小样本
    'flipud': 0.0,  # 禁用上下翻转（人脸有方向性）
    'fliplr': 0.5,  # 水平翻转
    'scale': 0.5,  # 缩放
    'hsv_h': 0.015,  # HSV 色调增强
    'hsv_s': 0.7,  # HSV 饱和度增强
    'hsv_v': 0.4,  # HSV 亮度增强
}

# ==========================
# 优化器配置
# ==========================
OPTIMIZER = {
    'optimizer': 'SGD',  # 优化器SGD
    'lr0': 0.01,  # 初始学习率
    'lrf': 0.01,  # 学习率衰减率
    'momentum': 0.937,  # 动量
    'weight_decay': 0.0005,  # 权重衰减
    'warmup_epochs': 5,  # 预热轮数
}


# ==========================
# 显存清理 Callback
# ==========================
def clear_memory_callback(trainer):
    """每5个epoch清理一次显存碎片，防止OOM"""
    if trainer.epoch % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()


# ==========================
# 训练函数
# ==========================
def train():
    model = YOLO(MODEL)
    # 注册显存清理callback
    model.add_callback('on_train_epoch_end', clear_memory_callback)
    save_dir = os.path.join(PROJECT, NAME)
    os.makedirs(save_dir, exist_ok=True)

    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        resume=RESUME,
        half=FP16,
        save_period=10,
        val=True,
        plots=True,
        patience=50,  # 新增：50轮无提升自动早停，节省时间
        **AUGMENT,
        **OPTIMIZER,
    )

    best_weights = f'{results.save_dir}/weights/best.pt'
    print(f'\n训练完成！最优权重路径: {best_weights}')
    return best_weights


# ==========================
# 主函数
# ==========================
if __name__ == '__main__':
    train()
