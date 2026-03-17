"""
test_widerface.py
用训练好的 YOLOv13 权重对 WiderFace val 集推理，
生成评估工具所需的 txt 格式预测结果
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ==========================
# 配置参数
# ==========================
WEIGHTS = 'runs/train/yolov13n_base2/weights/best.pt'
DATASET_FOLDER = '../dataset/wider_val/images/'
SAVE_FOLDER = 'widerface_evaluate/yolov13n_base/pred_txt/'
GT_FOLDER = 'widerface_evaluate/ground_truth'
IMG_SIZE = 640        # 与训练保持一致
CONF_THRES = 0.001       # 评估时保持低阈值，不要改
IOU_THRES = 0.5
DEVICE = '0'


def run_inference():
    print('=' * 50)
    print('推理生成预测结果')
    print(f'权重: {WEIGHTS}')
    print(f'数据集: {DATASET_FOLDER}')
    print(f'保存路径: {SAVE_FOLDER}')
    print('=' * 50)

    # 检查数据集路径
    dataset_folder = Path(DATASET_FOLDER)
    if not dataset_folder.exists():
        print(f'找不到数据集目录: {DATASET_FOLDER}')
        return False

    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)

    model = YOLO(WEIGHTS)

    event_dirs = sorted([d for d in dataset_folder.iterdir() if d.is_dir()])
    print(f'共找到 {len(event_dirs)} 个 event 目录')

    total_images = 0
    for event_dir in tqdm(event_dirs, desc='Processing events'):
        save_event_dir = Path(SAVE_FOLDER) / event_dir.name
        save_event_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(
            list(event_dir.glob('*.jpg')) +
            list(event_dir.glob('*.png'))
        )
        total_images += len(img_paths)

        for img_path in img_paths:
            results = model.predict(
                source=str(img_path),
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device=DEVICE,
                verbose=False,
                save=False,
                half=True,     # FP16推理，与训练一致
            )

            txt_path = save_event_dir / (img_path.stem + '.txt')
            with open(txt_path, 'w') as f:
                f.write(f'{event_dir.name}/{img_path.name}\n')

                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    f.write('0\n')
                    continue

                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                f.write(f'{len(xyxy)}\n')
                for (x1, y1, x2, y2), score in zip(xyxy, confs):
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f'{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {score:.4f}\n')

    print(f'\n推理完成，共处理 {total_images} 张图片')
    print(f'结果保存在: {SAVE_FOLDER}')
    return True


def run_evaluation():
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

    eval_dir = 'widerface_evaluate'
    sys.path.insert(0, eval_dir)

    try:
        from evaluation import evaluation
        aps = evaluation(SAVE_FOLDER, GT_FOLDER)

        output  = '\n==================== Results ====================\n'
        output += f'权重:   {WEIGHTS}\n'
        output += f'ImgSz:  {IMG_SIZE}\n'
        output += '-------------------------------------------------\n'
        output += f'Easy   Val AP: {aps[0]:.4f}\n'
        output += f'Medium Val AP: {aps[1]:.4f}\n'
        output += f'Hard   Val AP: {aps[2]:.4f}\n'
        output += '=================================================\n'

        print(output)

        results_path = Path('widerface_evaluate/results.txt')
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
    if not Path(WEIGHTS).exists():
        print(f'找不到权重文件: {WEIGHTS}')
        print('请确认训练已完成，或修改 WEIGHTS 路径')
    else:
        run_inference()
        run_evaluation()