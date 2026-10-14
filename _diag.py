"""Диагностика: что модель видит на размеченных скриншотах.

Запуск:  python _diag.py
Показывает по каждой картинке все детекции (класс, уверенность, площадь в %
от кадра) и сравнивает с разметкой из labels/.
"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

DETECT_DIR = os.path.join('runs', 'detect')


def find_latest_model():
    candidates = []
    for folder in os.listdir(DETECT_DIR):
        p = os.path.join(DETECT_DIR, folder, 'weights', 'best.pt')
        if os.path.isfile(p):
            candidates.append((folder, p))
    candidates.sort(key=lambda i: int(i[0].rsplit('-', 1)[-1]) if '-' in i[0] else 0,
                    reverse=True)
    return candidates[0]


run_name, model_path = find_latest_model()
print(f"Модель: {run_name} -> {model_path}")
model = YOLO(model_path)
print(f"Классы: {model.names}")

for split in ('train', 'val'):
    print(f"\n===== {split} =====")
    for img_path in sorted(glob.glob(os.path.join('images', split, '*.png'))):
        # cv2.imread не умеет открывать пути с кириллицей в Windows —
        # читаем байтами через numpy и декодируем вручную.
        try:
            frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            frame = None
        if frame is None:
            print(f"!! не читается {img_path}")
            continue
        h, w = frame.shape[:2]
        frame_area = h * w

        # сколько классов 0 реально размечено
        lbl_path = img_path.replace('images', 'labels', 1).replace('.png', '.txt')
        n_gt0 = n_gt1 = 0
        if os.path.isfile(lbl_path):
            for line in open(lbl_path, encoding='utf-8'):
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == '0':
                    n_gt0 += 1
                elif parts[0] == '1':
                    n_gt1 += 1

        res = model(frame, conf=0.05, verbose=False)[0]
        boxes = res.boxes
        print(f"\n{os.path.basename(img_path)}  {w}x{h}  GT: game={n_gt0} upg={n_gt1}")
        if boxes is None or len(boxes) == 0:
            print("   нет ни одной детекции (conf>=0.05)")
        for b in boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            area_pct = (x2 - x1) * (y2 - y1) / frame_area * 100.0
            print(f"   cls={cls}({model.names[cls]}) conf={conf:.3f} "
                  f"box={int(x2-x1)}x{int(y2-y1)} area={area_pct:.2f}%")

        # отдельный прогон с тем же порогом, что в app.py
        res2 = model(frame, conf=0.20, verbose=False)[0]
        c0 = sum(1 for b in res2.boxes if int(b.cls[0].item()) == 0)
        c1 = sum(1 for b in res2.boxes if int(b.cls[0].item()) == 1)
        print(f"   при conf=0.20: game={c0} upg={c1}")
