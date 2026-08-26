"""
Stage 3 — build animal-level train/val split for the smoke run.

CRITICAL: split by ANIMAL. With 3 animals:
  train = id47, id83   (both have full region coverage; different microscopy exports)
  val   = id52         (held-out; one-animal val is noisy — smoke test only)

Symlinks the crops built by build_smoke_dataset.py into the YOLO layout:

    outputs/smoke_dataset/yolo/
        images/train/*.png    labels/train/*.txt
        images/val/*.png      labels/val/*.txt
        data.yaml

Filenames start with 'idNN_' so grouping is unambiguous.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'outputs' / 'smoke_dataset'
YOLO = DATA / 'yolo'

TRAIN_IDS = {'id47', 'id83'}
VAL_IDS = {'id52'}
NAMES = ['single_pv', 'double']


def animal_id(stem: str) -> str:
    m = re.match(r'(id\d+)_', stem)
    return m.group(1) if m else ''


def main() -> None:
    # clean out prior split
    if YOLO.exists():
        shutil.rmtree(YOLO)
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (YOLO / sub).mkdir(parents=True)

    img_src = DATA / 'images'
    lbl_src = DATA / 'labels'

    train_n = val_n = 0
    for img in sorted(img_src.glob('*.png')):
        aid = animal_id(img.stem)
        if aid in TRAIN_IDS:
            bucket = 'train'
            train_n += 1
        elif aid in VAL_IDS:
            bucket = 'val'
            val_n += 1
        else:
            print(f'  [!] skipping {img.name} — animal id not recognised')
            continue
        lbl = lbl_src / f'{img.stem}.txt'
        if not lbl.exists():
            print(f'  [!] no label file for {img.name} — skipping')
            continue
        # symlink (absolute) so file locations are stable across cwd
        (YOLO / f'images/{bucket}' / img.name).symlink_to(img.resolve())
        (YOLO / f'labels/{bucket}' / lbl.name).symlink_to(lbl.resolve())

    yaml_path = YOLO / 'data.yaml'
    # Absolute path for YOLO (ultralytics resolves `train:`/`val:` under `path:`).
    yaml_path.write_text(
        f'path: {YOLO.resolve()}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'nc: {len(NAMES)}\n'
        f'names: {NAMES}\n'
    )

    print(f'wrote {yaml_path}')
    print(f'train images: {train_n}  ({sorted(TRAIN_IDS)})')
    print(f'val images:   {val_n}  ({sorted(VAL_IDS)})')

    # sanity: total label lines per bucket, per class
    for bucket in ('train', 'val'):
        cls_count = {0: 0, 1: 0}
        for lbl in (YOLO / f'labels/{bucket}').glob('*.txt'):
            for line in lbl.read_text().splitlines():
                if line.strip():
                    c = int(line.split()[0])
                    cls_count[c] = cls_count.get(c, 0) + 1
        print(f'  {bucket}: single_pv={cls_count.get(0, 0)}  double={cls_count.get(1, 0)}')


if __name__ == '__main__':
    main()
