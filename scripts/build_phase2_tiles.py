"""
Phase 2 tiling: split each Phase-1 ROI crop into 640×640 tiles at native
resolution (no downsample). Re-derive arrow boxes in tile-local coordinates.

Consumes:
    outputs/smoke_dataset/images/*.png     (Phase-1 ROI crops, best-z sourced)
    outputs/smoke_dataset/labels/*.txt     (Phase-1 YOLO labels in crop-local norm coords)

Produces:
    outputs/smoke_dataset/phase2/tiles/images/<crop_stem>_r{R}c{C}.png
    outputs/smoke_dataset/phase2/tiles/labels/<crop_stem>_r{R}c{C}.txt
    outputs/smoke_dataset/phase2/tiles/tiling_report.txt

Policy:
    tile   = 640 px
    stride = 512 px  (20% overlap — 128 px shared with each neighbour)
    Empty tiles (no arrow tip in the tile) are DROPPED. Hard-negative sampling
    is deferred until after we see if the class-1 collapse from Phase 1 recurs
    with abundant positives.
    Arrows in the overlap zone are duplicated into both host tiles for
    training. Inference-time dedup (NMS across tile seams) is a Phase-2b task.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image

from pnn_counter.dataset_builder import (
    CLASS_IDX,
    arrows_in_tile,
    tile_grid,
    write_yolo_label,
)

Image.MAX_IMAGE_PIXELS = None

SRC_IMG = REPO / 'outputs' / 'smoke_dataset' / 'images'
SRC_LBL = REPO / 'outputs' / 'smoke_dataset' / 'labels'
OUT = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'tiles'
TILE = 640
STRIDE = 512     # 128-px overlap = 20%
BOX_SIZE = 60


CLASS_NAME = {v: k for k, v in CLASS_IDX.items()}


def parse_yolo_labels(label_path: Path, crop_w: int, crop_h: int
                      ) -> list[tuple[str, float, float]]:
    """Convert stored YOLO labels back to (label_str, crop_x_px, crop_y_px)
    tip coordinates. The stored labels are boxes centered on tips, so cx/cy
    normalized give the tip in crop-local pixel space directly."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        cx_n, cy_n = float(parts[1]), float(parts[2])
        cx_px = cx_n * crop_w
        cy_px = cy_n * crop_h
        out.append((CLASS_NAME[cls], cx_px, cy_px))
    return out


def main() -> None:
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'labels').mkdir(parents=True, exist_ok=True)

    crops = sorted(SRC_IMG.glob('*.png'))
    if not crops:
        sys.exit(f'no crops found in {SRC_IMG} — run build_smoke_dataset.py first')

    report_lines: list[str] = []
    total_tiles_generated = 0
    total_tiles_kept = 0
    total_tiles_dropped = 0
    total_arrows_kept = 0
    class_counter: Counter = Counter()
    per_animal_tile_count: dict[str, int] = {}
    per_animal_arrow_count: dict[str, int] = {}

    for crop_path in crops:
        with Image.open(crop_path) as im:
            w, h = im.size
            im_rgb = im.convert('RGB').copy()
        label_path = SRC_LBL / f'{crop_path.stem}.txt'
        arrows_crop = parse_yolo_labels(label_path, w, h)

        tiles = tile_grid(w, h, tile=TILE, stride=STRIDE)
        n_cols = len({t[0] for t in tiles})
        aid = crop_path.stem.split('_')[0]  # 'id47' etc.
        per_animal_tile_count.setdefault(aid, 0)
        per_animal_arrow_count.setdefault(aid, 0)

        for tile in tiles:
            total_tiles_generated += 1
            tile_arrows = arrows_in_tile(arrows_crop, tile)
            if not tile_arrows:
                total_tiles_dropped += 1
                continue
            x0, y0, x1, y1 = tile
            tw, th = x1 - x0, y1 - y0
            col = sorted({t[0] for t in tiles}).index(x0)
            row = sorted({t[1] for t in tiles}).index(y0)
            stem = f'{crop_path.stem}_r{row}c{col}'
            out_img = OUT / 'images' / f'{stem}.png'
            out_lbl = OUT / 'labels' / f'{stem}.txt'
            im_rgb.crop((x0, y0, x1, y1)).save(out_img, format='PNG')
            write_yolo_label(out_lbl, tile_arrows, (tw, th), BOX_SIZE)
            total_tiles_kept += 1
            total_arrows_kept += len(tile_arrows)
            per_animal_tile_count[aid] += 1
            per_animal_arrow_count[aid] += len(tile_arrows)
            for label, _, _ in tile_arrows:
                class_counter[label] += 1

        report_lines.append(
            f'  {crop_path.stem}: {w}x{h} → {len(tiles)} tiles '
            f'(cols={n_cols}, kept={sum(1 for t in tiles if arrows_in_tile(arrows_crop, t))}, '
            f'{len(arrows_crop)} arrows in source crop)'
        )

    print('=' * 70)
    print('PHASE 2 TILING SUMMARY')
    print('=' * 70)
    print(f'source crops           : {len(crops)}')
    print(f'tile size / stride     : {TILE} / {STRIDE}  (overlap {TILE-STRIDE} px)')
    print(f'tiles generated total  : {total_tiles_generated}')
    print(f'tiles kept (has arrows): {total_tiles_kept}')
    print(f'tiles dropped (empty)  : {total_tiles_dropped}  '
          f'({total_tiles_dropped/total_tiles_generated:.1%})')
    print(f'arrow-box instances    : {total_arrows_kept}  (arrows in overlap zones are duplicated)')
    print(f'class breakdown        : {dict(class_counter)}')
    print(f'\ntiles kept per animal:')
    for aid, n in sorted(per_animal_tile_count.items()):
        print(f'  {aid}: {n} tiles, {per_animal_arrow_count[aid]} arrow instances')
    print(f'\nOutput: {OUT}')

    (OUT / 'tiling_report.txt').write_text(
        f'tile={TILE} stride={STRIDE} overlap_px={TILE-STRIDE} box_size_px={BOX_SIZE}\n'
        f'source_crops={len(crops)} tiles_generated={total_tiles_generated} '
        f'tiles_kept={total_tiles_kept} tiles_dropped_empty={total_tiles_dropped}\n'
        f'arrow_instances={total_arrows_kept} class_breakdown={dict(class_counter)}\n'
        f'per_animal_tiles={per_animal_tile_count}\n'
        f'per_animal_arrows={per_animal_arrow_count}\n'
        + '\n'.join(report_lines) + '\n'
    )


if __name__ == '__main__':
    main()
