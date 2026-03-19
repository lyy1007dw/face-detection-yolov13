import os
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


WEIGHTS = 'runs/train/exp2_yolov13n_p2_7/weights/best.pt'
DATASET_FOLDER = '../dataset/wider_val/images/'
SAVE_FOLDER = 'widerface_evaluate/exp2_yolov13n_p2/pred_txt/'
GT_FOLDER = 'widerface_evaluate/ground_truth'
IMG_SIZE = 640
CONF_THRES = 0.001
IOU_THRES = 0.5
DEVICE = '0'


def run_inference():
    """
    推理函数
    
    流程:
    1. 遍历val集所有图片
    2. 对每张图进行目标检测
    3. 将结果保存为txt格式
    
    输出格式 (每行):
      x1 y1 w h score
    示例:
      100.5 200.3 50.0 60.0 0.95
    """
    print('=' * 50)
    print('YOLOv13-P2 推理生成预测结果')
    print(f'权重: {WEIGHTS}')
    print(f'数据集: {DATASET_FOLDER}')
    print(f'保存路径: {SAVE_FOLDER}')
    print('=' * 50)
    
    # 检查数据集路径
    dataset_folder = Path(DATASET_FOLDER)
    if not dataset_folder.exists():
        print(f'找不到数据集目录: {DATASET_FOLDER}')
        return False
    
    # 创建输出目录
    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = YOLO(WEIGHTS)
    
    # 获取所有event目录
    event_dirs = sorted([d for d in dataset_folder.iterdir() if d.is_dir()])
    print(f'共找到 {len(event_dirs)} 个 event 目录')
    
    total_images = 0
    # 遍历每个event目录
    for event_dir in tqdm(event_dirs, desc='Processing events'):
        # 为每个event创建输出目录
        save_event_dir = Path(SAVE_FOLDER) / event_dir.name
        save_event_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取该event下所有图片
        img_paths = sorted(
            list(event_dir.glob('*.jpg')) +
            list(event_dir.glob('*.png'))
        )
        total_images += len(img_paths)
        
        # 对每张图片进行推理
        for img_path in img_paths:
            # 执行推理
            results = model.predict(
                source=str(img_path),      # 图片路径
                imgsz=IMG_SIZE,            # 输入尺寸
                conf=CONF_THRES,           # 置信度阈值
                iou=IOU_THRES,             # IOU阈值
                device=DEVICE,             # GPU设备
                verbose=False,              # 静默模式
                save=False,                # 不保存图片
                half=True,                 # FP16推理,加速
            )
            
            # 生成txt文件路径
            txt_path = save_event_dir / (img_path.stem + '.txt')
            
            # 写入预测结果
            with open(txt_path, 'w') as f:
                # 第1行: 图片路径 (相对于val根目录)
                f.write(f'{event_dir.name}/{img_path.name}\n')
                
                boxes = results[0].boxes
                
                # 无检测框
                if boxes is None or len(boxes) == 0:
                    f.write('0\n')
                    continue
                
                # 获取检测结果
                # xyxy: [x1, y1, x2, y2] 格式
                xyxy = boxes.xyxy.cpu().numpy()
                # conf: 置信度
                confs = boxes.conf.cpu().numpy()
                
                # 第2行: 检测框数量
                f.write(f'{len(xyxy)}\n')
                
                # 后续每行: x1 y1 w h score
                for (x1, y1, x2, y2), score in zip(xyxy, confs):
                    w = x2 - x1  # 计算宽度
                    h = y2 - y1  # 计算高度
                    # 写入格式: x1 y1 w h score (保留1位小数)
                    f.write(f'{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {score:.4f}\n')
    
    print(f'\n推理完成，共处理 {total_images} 张图片')
    print(f'结果保存在: {SAVE_FOLDER}')
    return True


def run_evaluation():
    """
    评估函数
    
    作用: 调用WiderFace官方评估工具计算AP
    
    评估指标:
    - Easy Val AP: 简单人脸 (尺寸较大,遮挡少)
    - Medium Val AP: 中等人脸
    - Hard Val AP: 困难人脸 (尺寸小,严重遮挡,模糊)
    
    说明:
    - Hard集包含大量<16px极小人脸
    - P2层主要提升应该在Hard集
    """
    print()
    print('=' * 50)
    print('计算 Easy / Medium / Hard AP')
    print(f'预测结果: {SAVE_FOLDER}')
    print(f'Ground Truth: {GT_FOLDER}')
    print('=' * 50)
    
    # 检查预测结果是否存在
    if not Path(SAVE_FOLDER).exists():
        print(f'找不到预测结果目录: {SAVE_FOLDER}')
        print('请先运行 run_inference()')
        return
    
    # 检查GT是否存在
    if not Path(GT_FOLDER).exists():
        print(f'找不到 ground truth 目录: {GT_FOLDER}')
        return
    
    # 添加评估工具路径
    eval_dir = 'widerface_evaluate'
    sys.path.insert(0, eval_dir)
    
    try:
        # 调用官方评估函数
        from evaluation import evaluation
        aps = evaluation(SAVE_FOLDER, GT_FOLDER)
        
        # 打印结果
        output = '\n==================== Results ====================\n'
        output += f'模型:   YOLOv13-P2\n'
        output += f'权重:   {WEIGHTS}\n'
        output += f'ImgSz:  {IMG_SIZE}\n'
        output += '-------------------------------------------------\n'
        output += f'Easy   Val AP: {aps[0]:.4f}\n'
        output += f'Medium Val AP: {aps[1]:.4f}\n'
        output += f'Hard   Val AP: {aps[2]:.4f}\n'
        output += '=================================================\n'
        
        print(output)
        
        # 保存结果到文件
        results_path = Path('widerface_evaluate/p2_results.txt')
        with open(results_path, 'w') as f:
            f.write(output)
        print(f'评估结果已保存到: {results_path}')
    
    except ImportError:
        print('评估工具未编译，请先执行:')
        print('  cd widerface_evaluate')
        print('  python setup.py build_ext --inplace')
    except Exception as e:
        print(f'评估出错: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 检查权重文件是否存在
    if not Path(WEIGHTS).exists():
        print(f'找不到权重文件: {WEIGHTS}')
        print('请确认训练已完成，或修改 WEIGHTS 路径')
    else:
        # 执行推理和评估
        run_inference()
        run_evaluation()
