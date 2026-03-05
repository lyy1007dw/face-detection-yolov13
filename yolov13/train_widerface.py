from ultralytics import YOLO

# 配置参数
MODEL = 'yolov13n.pt'         # 预训练权重，n/s/m/l/x 选一个
DATA = 'ultralytics/cfg/datasets/wider_face.yaml'  # 数据集配置
EPOCHS = 300                  #
BATCH_SIZE = 16               # 批量大小
IMG_SIZE = 640                # 输入图片大小
DEVICE = '0'                  # GPU编号，多卡写 '0,1,2,3'
WORKERS = 8                   # 数据加载线程数
PROJECT = 'runs/train'        # 训练结果保存目录
NAME = 'yolov13n_widerface'   # 训练结果保存的文件夹名
RESUME = False                # 是否从上次中断处继续训练


def train():
    model = YOLO(MODEL)

    results = model.train(
        data=DATA,           # 数据
        epochs=EPOCHS,       # 训练轮数
        batch=BATCH_SIZE,    # 批量大小
        imgsz=IMG_SIZE,      # 输入图片大小
        device=DEVICE,       # GPU编号，多卡写 '0,1,2,3'
        workers=WORKERS,     # 数据加载线程数
        project=PROJECT,     # 训练结果保存目录
        name=NAME,           # 训练结果保存的文件夹名
        resume=RESUME,       # 是否从上次中断处继续训练

        # 优化器
        optimizer='SGD',     # 优化器
        lr0=0.01,            # 初始学习率
        lrf=0.01,            # 学习率衰减率
        momentum=0.937,      # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,      # 预热轮数

        # 数据增强（人脸检测专用配置）
        mosaic=1.0,      # 马赛克增强，对小脸有帮助
        mixup=0.0,       # 关闭，人脸检测不适合
        copy_paste=0.0,  # 关闭
        fliplr=0.5,      # 水平翻转
        scale=0.5,       # 缩放范围
        hsv_h=0.015,     # HSV hue增强
        hsv_s=0.7,       # HSV saturation增强
        hsv_v=0.4,       # HSV value增强

        # 其他
        save_period=10,  # 每10轮保存一次权重
        val=True,        # 训练过程中同步跑内置验证（mAP@0.5）
        plots=True,      # 保存训练曲线图
    )

    best_weights = f'{results.save_dir}/weights/best.pt'
    print(f'\n训练完成！')
    print(f'最优权重路径: {best_weights}')
    return best_weights


if __name__ == '__main__':
    train()