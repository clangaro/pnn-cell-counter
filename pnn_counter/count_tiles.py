"""
Count YOLO predictions on validation tiles and measure cross-tile duplicates.

Reports:
  1. Prediction counts per class at each confidence threshold.
  2. How many predictions are near-duplicate same-class pairs (within
     --dup-radius px in the parent-crop frame), and how many of those sit
     inside a tile-overlap zone.

ultralytics runs NMS per image, i.e. per tile, so a cell in the overlap band
between two tiles can be predicted twice. Each tile's predictions are
projected back into its parent crop using the same tile_grid layout as
build_phase2_tiles.py, then greedily clustered per class.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pnn_counter.dataset_builder import tile_grid

logger = logging.getLogger(__name__)

PHASE2 = Path('outputs') / 'smoke_dataset' / 'phase2'
DEFAULT_TILES = PHASE2 / 'yolo' / 'images' / 'val'
DEFAULT_PARENTS = Path('outputs') / 'smoke_dataset' / 'images'
DEFAULT_WEIGHTS = PHASE2 / 'yolo_runs' / 'phase2_gentle' / 'weights' / 'best.pt'
DEFAULT_OUT = PHASE2 / 'tile_counts'


def parse_tile_stem(stem: str):
    """Return (parent_stem, r, c) for '<parent>_r{R}c{C}'."""
    m = re.match(r'(.+)_r(\d+)c(\d+)$', stem)
    if not m:
        return None
    return (m.group(1), int(m.group(2)), int(m.group(3)))


def tile_origin_lookup(
    parent: Path, tile: int, stride: int
) -> dict[tuple[int, int], tuple[int, int]]:
    """Map (row, col) to the tile's (x0, y0) origin in the parent crop."""
    with Image.open(parent) as im:
        w, h = im.size
    rects = tile_grid(w, h, tile, stride)
    xs = sorted({r[0] for r in rects})
    ys = sorted({r[1] for r in rects})
    return {(r, c): (xs[c], ys[r]) for r in range(len(ys)) for c in range(len(xs))}


def in_overlap_zone(x: float, y: float, w: int, h: int, tile: int, stride: int) -> bool:
    """True if (x, y) lies where two adjacent tiles overlap, on either axis.

    Tile starts follow tile_grid: 0, stride, 2*stride, ... with the last tile
    snapped to the far edge. Tiles i and i+1 overlap on [start[i+1], start[i] + tile).
    """
    for coord, limit in ((x, w), (y, h)):
        starts = []
        s = 0
        while s + tile < limit:
            starts.append(s)
            s += stride
        starts.append(max(0, limit - tile))
        starts = sorted(set(starts))
        for i in range(len(starts) - 1):
            seam_lo = starts[i + 1]
            seam_hi = starts[i] + tile
            if seam_lo <= coord < seam_hi:
                return True
    return False


def cluster_greedy(
    preds: list[tuple[float, float, int, float]], radius: float
) -> list[list[tuple[float, float, int, float]]]:
    """Greedy same-class clustering of (x, y, cls, conf) predictions.

    Two predictions join a cluster if they share a class and their centres are
    within `radius` px. Order-dependent: each unassigned prediction seeds a
    cluster and absorbs every later unassigned match.
    """
    used = [False] * len(preds)
    clusters: list[list] = []
    for i, p in enumerate(preds):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j in range(i + 1, len(preds)):
            if used[j]:
                continue
            q = preds[j]
            if p[2] != q[2]:
                continue
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= radius:
                cluster.append(q)
                used[j] = True
        clusters.append(cluster)
    return clusters


def count_tiles(args: argparse.Namespace) -> list[str]:
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    thresholds = tuple(args.conf)

    val_tiles = sorted(args.tiles.glob('*.png'))
    report = [f'val tiles: {len(val_tiles)}']

    # (parent_x, parent_y, class, conf) per parent crop
    by_parent: dict[str, list[tuple[float, float, int, float]]] = defaultdict(list)
    parent_dims: dict[str, tuple[int, int]] = {}
    raw_counts = {t: {0: 0, 1: 0} for t in thresholds}

    for tile_path in val_tiles:
        parsed = parse_tile_stem(tile_path.stem)
        if parsed is None:
            continue
        parent_stem, r, c = parsed
        parent = args.parents / f'{parent_stem}.png'
        if parent_stem not in parent_dims:
            with Image.open(parent) as im:
                parent_dims[parent_stem] = im.size
        origins = tile_origin_lookup(parent, args.tile, args.stride)
        if (r, c) not in origins:
            continue
        x_orig, y_orig = origins[(r, c)]

        result = model.predict(
            str(tile_path),
            imgsz=args.imgsz,
            conf=min(thresholds),
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            cls_i = int(cls)
            conf_f = float(conf)
            px = 0.5 * (float(box[0]) + float(box[2])) + x_orig
            py = 0.5 * (float(box[1]) + float(box[3])) + y_orig
            by_parent[parent_stem].append((px, py, cls_i, conf_f))
            for t in thresholds:
                if conf_f >= t:
                    raw_counts[t][cls_i] += 1

    rule = '=' * 74
    report += [
        '',
        rule,
        'PRED COUNTS BY CONFIDENCE (raw, per-tile NMS only — no cross-tile dedup)',
        rule,
        f'{"conf ≥":<10} {"single_pv":>12} {"double":>10} {"total":>10}',
    ]
    for t in thresholds:
        c = raw_counts[t]
        report.append(f'{t:<10.2f} {c[0]:>12,} {c[1]:>10,} {c[0] + c[1]:>10,}')

    report += [
        '',
        rule,
        f'NEAR-DUPLICATE ANALYSIS  (parent-crop frame, <{args.dup_radius:g} px same-class)',
        rule,
        (
            f'{"conf ≥":<10} {"raw":>10} {"clusters":>10} {"dup %":>10}   '
            f'{"dups in overlap":>20}   {"in-overlap %":>14}'
        ),
    ]
    for t in thresholds:
        raw = clusters_total = dup_count = dup_in_overlap = 0
        for parent_stem, preds in by_parent.items():
            filt = [p for p in preds if p[3] >= t]
            raw += len(filt)
            w, h = parent_dims[parent_stem]
            clusters = cluster_greedy(filt, args.dup_radius)
            clusters_total += len(clusters)
            # every member after the first in a cluster counts as a duplicate
            for cluster in clusters:
                dup_count += len(cluster) - 1
                for member in cluster[1:]:
                    if in_overlap_zone(member[0], member[1], w, h, args.tile, args.stride):
                        dup_in_overlap += 1
        dup_pct = (dup_count / raw * 100) if raw else 0
        in_ov_pct = (dup_in_overlap / dup_count * 100) if dup_count else 0
        report.append(
            f'{t:<10.2f} {raw:>10,} {clusters_total:>10,} {dup_pct:>9.1f}%   '
            f'{dup_in_overlap:>20,}   {in_ov_pct:>13.1f}%'
        )

    report += [
        '',
        'NMS SCOPE: ultralytics.model.predict runs NMS *inside* each image',
        '  (per-tile). It does NOT dedupe across tile seams. So predictions in',
        f'  the {args.tile - args.stride}-px overlap band between adjacent tiles are counted twice.',
    ]
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--tiles', type=Path, default=DEFAULT_TILES, help='folder of <parent>_r{R}c{C}.png tiles'
    )
    ap.add_argument(
        '--parents', type=Path, default=DEFAULT_PARENTS, help='folder of the parent ROI crops'
    )
    ap.add_argument(
        '--weights', type=Path, default=DEFAULT_WEIGHTS, help='trained YOLO weights (.pt)'
    )
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='folder for tile_counts.txt')
    ap.add_argument(
        '--conf', type=float, nargs='+', default=[0.05, 0.25, 0.50], help='confidence thresholds'
    )
    ap.add_argument('--tile', type=int, default=640, help='tile size used when tiling, px')
    ap.add_argument('--stride', type=int, default=512, help='tile stride used when tiling, px')
    ap.add_argument('--imgsz', type=int, default=640, help='YOLO inference size')
    ap.add_argument('--dup-radius', type=float, default=20, help='same-class duplicate radius, px')
    ap.add_argument('--device', default='mps', help="torch device, e.g. 'mps', 'cpu', '0'")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    # torch and opencv each bundle libomp in conda envs on macOS
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    report = count_tiles(args)
    print('\n'.join(report))
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / 'tile_counts.txt'
    out_file.write_text('\n'.join(report) + '\n', encoding='utf-8')
    logger.info('wrote %s', out_file)


if __name__ == '__main__':
    main()
