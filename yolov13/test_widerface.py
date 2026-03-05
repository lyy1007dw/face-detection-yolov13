"""
test_widerface.py
用训练好的 YOLOv13 权重对 WiderFace val 集推理，
生成评估工具所需的 txt 格式预测结果
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 配置参考
WEIGHTS = 'runs/detect/train5/weights/best.pt'
DATASET_FOLDER = '../datasets/wider_val/images/'
SAVE_FOLDER = 'widerface_evaluate/pred_txt/'
IMG_SIZE = 640
CONF_THRES = 0.001   # 评估时保持低阈值，不要改
IOU_THRES = 0.5


def detect_widerface(weights, dataset_folder, save_folder,
                     img_size=640, conf_thres=0.001, iou_thres=0.5):

    os.makedirs(save_folder, exist_ok=True)
    model = YOLO(weights)

    dataset_folder = Path(dataset_folder)
    event_dirs = sorted([d for d in dataset_folder.iterdir() if d.is_dir()])

    for event_dir in tqdm(event_dirs, desc='Processing events'):
        # 创建对应的输出子目录
        save_event_dir = Path(save_folder) / event_dir.name
        save_event_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(list(event_dir.glob('*.jpg')) +
                           list(event_dir.glob('*.png')))

        for img_path in img_paths:
            # 推理
            results = model.predict(
                source=str(img_path),
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
                save=False,
            )

            # 写结果 txt
            txt_path = save_event_dir / (img_path.stem + '.txt')
            with open(txt_path, 'w') as f:
                # 第一行：图片相对路径（评估工具要求）
                f.write(f'{event_dir.name}/{img_path.name}\n')

                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    f.write('0\n')
                    continue

                xyxy  = boxes.xyxy.cpu().numpy()   # [N, 4]
                confs = boxes.conf.cpu().numpy()   # [N]

                f.write(f'{len(xyxy)}\n')
                for (x1, y1, x2, y2), score in zip(xyxy, confs):
                    w = x2 - x1
                    h = y2 - y1
                    # WiderFace 格式：x y w h score
                    f.write(f'{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {score:.4f}\n')

    print(f'\n✅ 推理完成，结果保存在: {save_folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str, required=True,
                        help='训练好的权重路径')
    parser.add_argument('--dataset_folder',
                        type=str, default=r'./dataset/wider_val/images',
                        help='WiderFace val 图片目录')
    parser.add_argument('--save_folder',
                        type=str, default=r'./widerface_evaluate/predictions',
                        help='预测结果保存目录')
    parser.add_argument('--img-size',   type=int,   default=640)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres',  type=float, default=0.5)
    opt = parser.parse_args()

    detect_widerface(
        weights=opt.weights,
        dataset_folder=opt.dataset_folder,
        save_folder=opt.save_folder,
        img_size=opt.img_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
    )
