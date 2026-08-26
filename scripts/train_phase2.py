"""
Phase 2 training — YOLOv8n on 640-px tiles at native resolution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

REPO = Path(__file__).resolve().parents[1]
DATA_YAML = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo' / 'data.yaml'
RUNS_DIR = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo_runs'
RUN_NAME = 'phase2_gentle'


def main() -> None:
    if not DATA_YAML.exists():
        sys.exit(f'missing {DATA_YAML} — run build_phase2_split.py first')

    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,           # native tile size — no downsample
        batch=16,
        device='mps',
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        patience=20,
        # Gentler regime after phase2_v1 stalled (best mAP at epoch 2,
        # train loss oscillated between 2.7-3.0 for 20 epochs then early-stopped).
        # Lower LR + narrower rotation + single flip.
        lr0=0.001,
        degrees=45.0,
        fliplr=0.5,
        flipud=0.0,
        # Disable HSV shifts — fluorescence hue is the class signal
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        workers=2,
        cache=False,
        plots=True,
        verbose=True,
    )
    print('\nTraining finished. best weights:',
          RUNS_DIR / RUN_NAME / 'weights' / 'best.pt')


if __name__ == '__main__':
    main()
