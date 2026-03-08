"""
test_widerface.py
用训练好的 YOLOv13 权重对 WiderFace val 集推理，
生成评估工具所需的 txt 格式预测结果
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 配置参考
WEIGHTS = r'runs\train\yolov13n_widerface5/weights/best.pt'
DATASET_FOLDER = '../dataset/wider_val/images/'
SAVE_FOLDER = 'widerface_evaluate/pred_txt/'
GT_FOLDER = 'widerface_evaluate/ground_truth'
IMG_SIZE = 640
CONF_THRES = 0.001   # 评估时保持低阈值，不要改
IOU_THRES = 0.5


def run_inference():
    print('=' * 50)
    print('推理生成预测结果')
    print('=' * 50)

    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)

    dataset_folder = Path(DATASET_FOLDER)
    event_dirs = sorted([d for d in dataset_folder.iterdir() if d.is_dir()])
    print(f'共找到 {len(event_dirs)} 个 event 目录')

    for event_dir in tqdm(event_dirs, desc='Processing'):
        save_event_dir = Path(SAVE_FOLDER) / event_dir.name
        save_event_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(list(event_dir.glob('*.jpg')) +
                           list(event_dir.glob('*.png')))

        for img_path in img_paths:
            results = model.predict(
                source=str(img_path),
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                verbose=False,
                save=False,
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

    print(f'\n推理完成，结果保存在: {SAVE_FOLDER}')


def run_evaluation():
    print()
    print('=' * 50)
    print('计算 Easy / Medium / Hard AP')
    print('=' * 50)

    eval_dir = 'widerface_evaluate'
    sys.path.insert(0, eval_dir)

    try:
        from evaluation import evaluation
        aps = evaluation(SAVE_FOLDER, GT_FOLDER)
        
        output = "==================== Results ====================\n"
        output += "Easy   Val AP: {}\n".format(aps[0])
        output += "Medium Val AP: {}\n".format(aps[1])
        output += "Hard   Val AP: {}\n".format(aps[2])
        output += "=================================================\n"
        
        results_path = Path('widerface_evaluate/results.txt')
        with open(results_path, 'w') as f:
            f.write(output)
        print(f'评估完成，结果已保存到: {results_path}')
    except Exception as e:
        print(f'评估出错: {e}')
        import traceback
        traceback.print_exc()
        print('如需重新编译评估工具，请运行:')
        print('cd widerface_evaluate')
        print('python setup.py build_ext --inplace')


if __name__ == '__main__':
    # 检查权重文件
    if not Path(WEIGHTS).exists():
        print(f'找不到权重文件: {WEIGHTS}')
        print('请先完成训练，或修改 WEIGHTS 路径')
    else:
        # run_inference()
        run_evaluation()