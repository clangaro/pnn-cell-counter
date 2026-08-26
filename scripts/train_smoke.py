"""
Stage 4 — smoke-test YOLOv8n training on the 3-animal split.

Goal: prove the pipeline runs end-to-end. NOT a real training run.
Metrics from a single held-out animal are noisy — see DECISIONS.md.

Design choices:
  - yolov8n.pt pretrained (fast, small — fine for smoke).
  - 50 epochs, patience 20 (early stop if val plateaus).
  - imgsz=1280: our crops are 3-8k px and the boxes are 60px in native.
    At 1280 the 60px boxes downsample to ~10-30 px depending on crop size —
    the small end of what YOLO's smallest anchor handles. Real training will
    need larger imgsz or image tiling; this is documented.
  - device='mps' (Apple M-series GPU).
  - augmentation: heavy rotation + hflip + vflip (microscopy has no canonical
    orientation), disable HSV shifts (fluorescence color is semantic).

Usage:
    ~/anaconda3/bin/python scripts/train_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ultralytics can trip the libomp guard on macOS + conda envs (see homelab memory).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / 'outputs' / 'smoke_dataset' / 'yolo' / 'data.yaml'
RUNS_DIR = REPO / 'outputs' / 'smoke_dataset' / 'yolo_runs'
RUN_NAME = 'smoke1'


def main() -> None:
    if not DATA_YAML.exists():
        sys.exit(f'missing {DATA_YAML} — run build_smoke_split.py first')

    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=1280,
        batch=4,           # 1280px + big crops → keep batch small on MPS
        device='mps',
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        patience=20,
        # Augmentation
        degrees=180.0,     # arbitrary rotation
        fliplr=0.5,        # horizontal flip
        flipud=0.5,        # vertical flip (microscopy has no top/bottom)
        # Disable HSV shifts — fluorescence hue is the class signal.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        # Speed knobs
        workers=2,
        cache=False,       # 44 crops each 10-30MB — don't hold in RAM
        plots=True,
        verbose=True,
    )
    print('\nTraining finished. best weights:',
          RUNS_DIR / RUN_NAME / 'weights' / 'best.pt')


if __name__ == '__main__':
    main()
