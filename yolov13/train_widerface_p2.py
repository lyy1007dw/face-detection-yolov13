import gc
import torch
from ultralytics import YOLO
import os

# ============================================================================
# 环境变量设置 - 优化显存使用
# ============================================================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OMP_NUM_THREADS'] = '16'

# -------------------
# 模型配置
# -------------------
PRETRAINED_WEIGHTS = 'runs/train/yolov13n_vgpu32g2/weights/best.pt'
MODEL = 'ultralytics/cfg/models/v13/yolov13p2n.yaml'
DATA = 'ultralytics/cfg/datasets/wider_face.yaml'
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 1280
DEVICE = '0'
WORKERS = 8
PROJECT = 'runs/train'
NAME = 'yolov13n_phase2_p2_'
RESUME = False
FP16 = True

# ==========================
# 迁移学习分阶段配置
# ==========================
# 冻结backbone，只训练新增的P2 head（快速收敛）
FREEZE_BACKBONE = True
FREEZE_LAYERS = 9  # 冻结前9层backbone

# ==========================
# 数据增强配置 (人脸检测专用)
# ==========================
AUGMENT = {
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.2,
    'flipud': 0.0,
    'fliplr': 0.5,
    'scale': 0.5,
    'hsv_h': 0.015,  # 色调
    'hsv_s': 0.7,  # 饱和度
    'hsv_v': 0.4,  # 亮度
}

# ==========================
# 优化器配置
# ==========================
OPTIMIZER = {
    'optimizer': 'SGD',
    'lr0': 0.001,  # 迁移学习lr要小，原来0.01太大
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
}


# ==========================
# 显存清理 Callback
# ==========================
def clear_memory_callback(trainer):
    if trainer.epoch % 5 == 0:
        gc.collect()  # Python垃圾回收
        torch.cuda.empty_cache()  # CUDA显存回收


# ==========================
# 解冻所有层 Callback
# ==========================
def unfreeze_callback(trainer):
    UNFREEZE_EPOCH = 20  # 前20个epoch只训练P2 head，之后全部解冻
    if trainer.epoch == UNFREEZE_EPOCH:
        print(f'\n[Epoch {UNFREEZE_EPOCH}] 解冻所有层，开始全局微调...')
        for name, param in trainer.model.named_parameters():
            param.requires_grad = True
        # 解冻后适当降低lr，避免破坏已收敛的特征
        for g in trainer.optimizer.param_groups:
            g['lr'] = g['lr'] * 0.5
        print(f'  当前lr已调整为原来的0.5倍')


# ==========================
# 训练函数
# ==========================
def train():
    print(f'加载预训练权重: {PRETRAINED_WEIGHTS}')
    print(f'模型结构配置: {MODEL}')
    try:
        # 方式1：从已有.pt权重加载，能匹配的层自动加载，新增层随机初始化
        model = YOLO(MODEL)
        model.load(PRETRAINED_WEIGHTS)  # 部分权重加载，不匹配的层跳过
        print('预训练权重加载成功（新增P2层使用随机初始化）')
    except Exception as e:
        print(f'预训练权重加载失败: {e}')
        print('回退到从头训练...')
        model = YOLO(MODEL)

    # 注册显存清理callback
    model.add_callback('on_train_epoch_end', clear_memory_callback)
    # 注册解冻所有层callback
    model.add_callback('on_train_epoch_end', unfreeze_callback)

    # 创建输出目录
    save_dir = os.path.join(PROJECT, NAME)
    os.makedirs(save_dir, exist_ok=True)

    # 执行训练
    results = model.train(
        # -------------------
        # 必需参数
        # -------------------
        data=DATA,  # 数据集配置
        epochs=EPOCHS,  # 训练轮数
        batch=BATCH_SIZE,  # 批次大小 (P2版本需减小)
        imgsz=IMG_SIZE,  # 输入图像尺寸
        device=DEVICE,  # GPU设备
        workers=WORKERS,  # 数据加载线程
        project=PROJECT,  # 输出目录
        name=NAME,  # 实验名称

        # -------------------
        # 训练控制
        # -------------------
        resume=RESUME,  # 是否继续训练
        half=FP16,  # FP16混合精度
        save_period=10,  # 每10轮保存checkpoint
        val=True,  # 验证集评估
        plots=True,  # 生成训练曲线
        patience=20,  # 20轮无提升则早停
        # 冻结backbone层，只让P2新增层先学习
        freeze=FREEZE_LAYERS if FREEZE_BACKBONE else None,
        **AUGMENT,  # 解包增强配置
        **OPTIMIZER,  # 解包优化器配置
    )

    # 输出最优权重路径
    best_weights = f'{results.save_dir}/weights/best.pt'
    print(f'\n训练完成! 最优权重路径: {best_weights}')
    return best_weights


# ==========================
# 主函数
# ==========================
if __name__ == '__main__':
    train()
