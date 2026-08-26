"""
Phase 2 val: render GT-vs-PRED overlays + per-class mAP50 from ultralytics val().
Uses the phase2_gentle run's best.pt.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

REPO = Path(__file__).resolve().parents[1]
YOLO_DIR = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo'
WEIGHTS = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo_runs' / 'phase2_gentle' / 'weights' / 'best.pt'
OUT_DIR = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'val_preds'

CLASS_COLOR = {0: (255, 0, 0), 1: (255, 0, 255)}   # red = single_pv, magenta = double
CLASS_NAME = {0: 'single_pv', 1: 'double'}


def draw_yolo_boxes(im, label_path, colors, width=3):
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
        x0, y0 = (cx - w/2) * W, (cy - h/2) * H
        x1, y1 = (cx + w/2) * W, (cy + h/2) * H
        draw.rectangle([x0, y0, x1, y1], outline=colors[cls], width=width)


def draw_pred_boxes(im, result, colors, min_conf=0.05, width=3):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(im)
    if result.boxes is None:
        return {0: 0, 1: 0}
    counts = {0: 0, 1: 0}
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        if float(conf) < min_conf:
            continue
        c = int(cls)
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=colors[c], width=width)
        counts[c] += 1
    return counts


def main() -> None:
    if not WEIGHTS.exists():
        sys.exit(f'missing weights: {WEIGHTS}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(str(WEIGHTS))

    # ---- per-class mAP via built-in val() ----
    print('=' * 60)
    print('ULTRALYTICS VAL — per-class mAP50 / mAP50-95')
    print('=' * 60)
    metrics = model.val(
        data=str(YOLO_DIR / 'data.yaml'),
        imgsz=640,
        device='mps',
        plots=False,
        verbose=False,
    )
    # metrics.box.map50 (all), metrics.box.ap50 (per-class)
    print(f'Overall  mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}')
    ap50 = metrics.box.ap50 if metrics.box.ap50 is not None else []
    ap = metrics.box.ap if metrics.box.ap is not None else []
    for c, name in CLASS_NAME.items():
        v50 = ap50[c] if c < len(ap50) else float('nan')
        v = ap[c] if c < len(ap) else float('nan')
        print(f'  class {c} ({name}):  AP50={v50:.4f}  AP50-95={v:.4f}')
    print()

    # ---- render overlays + count boxes per tile ----
    val_images = sorted((YOLO_DIR / 'images' / 'val').glob('*.png'))
    print(f'Rendering {len(val_images)} val overlays...')
    gt_totals = Counter()
    pred_totals = Counter()
    n_rendered = 0
    for img in val_images:
        gt_label = YOLO_DIR / 'labels' / 'val' / f'{img.stem}.txt'
        with Image.open(img) as im0:
            im_gt = im0.convert('RGB').copy()
            im_pred = im0.convert('RGB').copy()
        draw_yolo_boxes(im_gt, gt_label, CLASS_COLOR)
        result = model.predict(str(img), imgsz=640, conf=0.05, device='mps', verbose=False)[0]
        pred_counts = draw_pred_boxes(im_pred, result, CLASS_COLOR)
        pred_totals[0] += pred_counts[0]
        pred_totals[1] += pred_counts[1]
        for line in gt_label.read_text().splitlines() if gt_label.exists() else []:
            if line.strip():
                gt_totals[int(line.split()[0])] += 1
        # save side-by-side (640 + gap + 640)
        side = Image.new('RGB', (im_gt.size[0] * 2 + 10, im_gt.size[1]), (0, 0, 0))
        side.paste(im_gt, (0, 0))
        side.paste(im_pred, (im_gt.size[0] + 10, 0))
        side.save(OUT_DIR / f'{img.stem}_gt_vs_pred.jpg', quality=85)
        n_rendered += 1

    print(f'\nRendered {n_rendered} tiles → {OUT_DIR}')
    print('\nAggregated over ALL val tiles (conf ≥ 0.05):')
    print(f'  GT   single_pv={gt_totals[0]:5d}  double={gt_totals[1]:4d}  total={gt_totals[0]+gt_totals[1]}')
    print(f'  PRED single_pv={pred_totals[0]:5d}  double={pred_totals[1]:4d}  total={pred_totals[0]+pred_totals[1]}')

    # ---- pick 6 tiles with the most doubles for visual QC ----
    doubles_per_tile: list[tuple[Path, int]] = []
    for img in val_images:
        gt_label = YOLO_DIR / 'labels' / 'val' / f'{img.stem}.txt'
        n_double = 0
        for line in gt_label.read_text().splitlines() if gt_label.exists() else []:
            if line.strip() and int(line.split()[0]) == 1:
                n_double += 1
        doubles_per_tile.append((img, n_double))
    doubles_per_tile.sort(key=lambda x: -x[1])
    top = [p.stem for p, n in doubles_per_tile[:6] if n > 0]
    print(f'\nTop 6 val tiles by double count (open these first):')
    for stem in top:
        print(f'  {stem}_gt_vs_pred.jpg')


if __name__ == '__main__':
    main()
