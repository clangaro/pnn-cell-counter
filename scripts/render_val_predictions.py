"""
Render val-set predictions from the smoke-test best.pt alongside ground truth.
Produces side-by-side JPGs so the user can eyeball what the trained model sees.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

REPO = Path(__file__).resolve().parents[1]
YOLO_DIR = REPO / 'outputs' / 'smoke_dataset' / 'yolo'
WEIGHTS = REPO / 'outputs' / 'smoke_dataset' / 'yolo_runs' / 'smoke1' / 'weights' / 'best.pt'
OUT_DIR = REPO / 'outputs' / 'smoke_dataset' / 'val_preds'

CLASS_COLOR = {0: (255, 0, 0), 1: (255, 0, 255)}   # single_pv=red, double=magenta
CLASS_NAME = {0: 'single_pv', 1: 'double'}


def draw_yolo_boxes(im, label_path: Path, colors, width=4, dashed=False):
    from PIL import ImageDraw
    W, H = im.size
    draw = ImageDraw.Draw(im)
    if not label_path.exists():
        return
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x0 = (cx - w / 2) * W
        y0 = (cy - h / 2) * H
        x1 = (cx + w / 2) * W
        y1 = (cy + h / 2) * H
        draw.rectangle([x0, y0, x1, y1], outline=colors[cls], width=width)


def draw_pred_boxes(im, results, colors, min_conf=0.05, width=4):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(im)
    if results.boxes is None or len(results.boxes) == 0:
        return 0
    n = 0
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if float(conf) < min_conf:
            continue
        c = int(cls)
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=colors[c], width=width)
        n += 1
    return n


def main() -> None:
    if not WEIGHTS.exists():
        sys.exit(f'missing weights: {WEIGHTS}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(str(WEIGHTS))
    val_images = sorted((YOLO_DIR / 'images' / 'val').glob('*.png'))
    print(f'val images: {len(val_images)}')

    for img in val_images:
        gt_label = YOLO_DIR / 'labels' / 'val' / f'{img.stem}.txt'
        with Image.open(img) as im0:
            im_gt = im0.convert('RGB').copy()
            im_pred = im0.convert('RGB').copy()
        # GT boxes
        draw_yolo_boxes(im_gt, gt_label, CLASS_COLOR)
        # Predict at low conf so we can see what the model is producing at all
        result = model.predict(str(img), imgsz=1280, conf=0.05, device='mps', verbose=False)[0]
        n_pred = draw_pred_boxes(im_pred, result, CLASS_COLOR, min_conf=0.05)

        # Side-by-side, downsampled for viewing
        W, H = im_gt.size
        scale = 1400 / max(W, H)
        tw, th = int(W * scale), int(H * scale)
        gt_thumb = im_gt.resize((tw, th), Image.LANCZOS)
        pr_thumb = im_pred.resize((tw, th), Image.LANCZOS)
        side = Image.new('RGB', (tw * 2 + 10, th), (0, 0, 0))
        side.paste(gt_thumb, (0, 0))
        side.paste(pr_thumb, (tw + 10, 0))
        out = OUT_DIR / f'{img.stem}_gt_vs_pred.jpg'
        side.save(out, quality=85)
        # class breakdown for header
        gt_by_cls = {0: 0, 1: 0}
        for line in gt_label.read_text().splitlines() if gt_label.exists() else []:
            if line.strip():
                gt_by_cls[int(line.split()[0])] += 1
        pred_by_cls = {0: 0, 1: 0}
        if result.boxes is not None:
            for cls, conf in zip(result.boxes.cls, result.boxes.conf):
                if float(conf) >= 0.05:
                    pred_by_cls[int(cls)] += 1
        print(f'  {img.name}: GT (single_pv={gt_by_cls[0]}, double={gt_by_cls[1]})  '
              f'PRED@0.05 (single_pv={pred_by_cls[0]}, double={pred_by_cls[1]})')


if __name__ == '__main__':
    main()
