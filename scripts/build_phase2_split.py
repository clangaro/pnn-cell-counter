"""
Phase 2 — animal-level train/val split for tiled dataset.

Same policy as Phase 1: train = id47 + id83, val = id52. No leakage.
Symlinks tile PNGs + label TXTs into a YOLO layout.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TILES = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'tiles'
YOLO = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo'

TRAIN_IDS = {'id47', 'id83'}
VAL_IDS = {'id52'}
NAMES = ['single_pv', 'double']


def animal_id(stem: str) -> str:
    m = re.match(r'(id\d+)_', stem)
    return m.group(1) if m else ''


def main() -> None:
    if YOLO.exists():
        shutil.rmtree(YOLO)
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (YOLO / sub).mkdir(parents=True)

    img_src = TILES / 'images'
    lbl_src = TILES / 'labels'

    train_n = val_n = 0
    for img in sorted(img_src.glob('*.png')):
        aid = animal_id(img.stem)
        if aid in TRAIN_IDS:
            bucket, train_n = 'train', train_n + 1
        elif aid in VAL_IDS:
            bucket, val_n = 'val', val_n + 1
        else:
            print(f'  [!] skipping {img.name} — animal id not recognised')
            continue
        lbl = lbl_src / f'{img.stem}.txt'
        if not lbl.exists():
            continue
        (YOLO / f'images/{bucket}' / img.name).symlink_to(img.resolve())
        (YOLO / f'labels/{bucket}' / lbl.name).symlink_to(lbl.resolve())

    yaml_path = YOLO / 'data.yaml'
    yaml_path.write_text(
        f'path: {YOLO.resolve()}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'nc: {len(NAMES)}\n'
        f'names: {NAMES}\n'
    )
    print(f'wrote {yaml_path}')
    print(f'train tiles: {train_n}  ({sorted(TRAIN_IDS)})')
    print(f'val tiles:   {val_n}  ({sorted(VAL_IDS)})')

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
