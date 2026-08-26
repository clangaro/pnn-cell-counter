"""
Phase-2 over-prediction diagnostic.

Answers:
  1. How do pred counts change at conf 0.05 vs 0.25 vs 0.50?
  2. Is NMS applied across tile-overlap seams? (Short answer: NO — ultralytics
     runs NMS per image, i.e. per tile.)
  3. Of the predictions we're counting, what fraction are near-duplicate
     same-class pairs (< 20 px apart in the parent-crop frame), and how many
     of those sit inside a tile-overlap zone?

Projects each val tile's predictions back into the parent Phase-1 crop's
coordinate frame using the same tile_grid layout as build_phase2_tiles.
Then greedy-clusters within 20 px for each class.
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image

from pnn_counter.dataset_builder import tile_grid

PARENT_CROP_DIR = REPO / 'outputs' / 'smoke_dataset' / 'images'
VAL_TILE_DIR = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo' / 'images' / 'val'
WEIGHTS = REPO / 'outputs' / 'smoke_dataset' / 'phase2' / 'yolo_runs' / 'phase2_gentle' / 'weights' / 'best.pt'

TILE, STRIDE = 640, 512
OVERLAP = TILE - STRIDE          # 128 px seam width
DUP_RADIUS = 20                  # centers this close → same detection
CONF_THRESHOLDS = (0.05, 0.25, 0.50)


def parse_tile_stem(stem: str):
    """Return (parent_stem, r, c) for '<parent>_r{R}c{C}'."""
    m = re.match(r'(.+)_r(\d+)c(\d+)$', stem)
    if not m:
        return None
    return (m.group(1), int(m.group(2)), int(m.group(3)))


def tile_origin_lookup(parent_stem: str) -> dict[tuple[int, int], tuple[int, int]]:
    """Build {(r,c): (x0,y0)} for the parent's tile grid."""
    parent = PARENT_CROP_DIR / f'{parent_stem}.png'
    with Image.open(parent) as im:
        w, h = im.size
    rects = tile_grid(w, h, TILE, STRIDE)
    xs = sorted({r[0] for r in rects})
    ys = sorted({r[1] for r in rects})
    return {(r, c): (xs[c], ys[r]) for r in range(len(ys)) for c in range(len(xs))}


def in_overlap_zone(x: float, y: float, w: int, h: int) -> bool:
    """A point is in a tile-overlap zone in the parent frame if it's within
    OVERLAP px of *any* tile's inner edge (i.e. shared with a neighbour).
    Overlap seams sit at multiples of STRIDE between 0 and (w-TILE), plus the
    snapped final rows/cols. Simpler test: is the point within OVERLAP px of
    a stride-boundary that itself lies between 0 and the last-tile origin?"""
    # gather stride lines that fall inside the crop (i.e. between two tiles)
    # in x
    for boundary_axis, limit in ((x, w), (y, h)):
        # tile starts are 0, STRIDE, 2*STRIDE, ..., snap-to-(limit-TILE)
        # a "seam" exists where two adjacent tiles overlap: the seam center
        # is at tile_start + TILE/2 no — simpler: any tile boundary in the
        # inner range [0, limit) other than the outer edges.
        starts = []
        s = 0
        while s + TILE < limit:
            starts.append(s)
            s += STRIDE
        starts.append(max(0, limit - TILE))
        starts = sorted(set(starts))
        # each pair of adjacent tiles overlaps between max(next.start, cur.end-OVERLAP)
        for i in range(len(starts) - 1):
            seam_lo = starts[i + 1]
            seam_hi = starts[i] + TILE
            if seam_lo <= boundary_axis < seam_hi:
                return True
    return False


def cluster_greedy(preds: list[tuple[float, float, int, float]],
                   radius: int = DUP_RADIUS
                   ) -> list[list[tuple[float, float, int, float]]]:
    """Greedy same-class clustering: two preds join if same class and centers
    within `radius` px. Returns list of clusters (each cluster ≥ 1 pred)."""
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
            if p[2] != q[2]:  # different class
                continue
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= radius:
                cluster.append(q)
                used[j] = True
        clusters.append(cluster)
    return clusters


def main() -> None:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    val_tiles = sorted(VAL_TILE_DIR.glob('*.png'))
    print(f'val tiles: {len(val_tiles)}')

    # Predictions per parent-crop, projected to parent coords.
    # Each item: (parent_x, parent_y, class, conf)
    by_parent: dict[str, list[tuple[float, float, int, float]]] = defaultdict(list)
    parent_dims: dict[str, tuple[int, int]] = {}

    # Track raw per-conf counts (no dedup, no projection)
    raw_counts = {t: {0: 0, 1: 0} for t in CONF_THRESHOLDS}

    for tile_path in val_tiles:
        parsed = parse_tile_stem(tile_path.stem)
        if parsed is None:
            continue
        parent_stem, r, c = parsed
        if parent_stem not in parent_dims:
            with Image.open(PARENT_CROP_DIR / f'{parent_stem}.png') as im:
                parent_dims[parent_stem] = im.size
        origins = tile_origin_lookup(parent_stem)
        if (r, c) not in origins:
            continue
        x_orig, y_orig = origins[(r, c)]

        result = model.predict(str(tile_path), imgsz=640, conf=min(CONF_THRESHOLDS),
                               device='mps', verbose=False)[0]
        if result.boxes is None:
            continue
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            cls_i = int(cls)
            conf_f = float(conf)
            # tile-local center
            tx = 0.5 * (float(box[0]) + float(box[2]))
            ty = 0.5 * (float(box[1]) + float(box[3]))
            # parent-crop center
            px = tx + x_orig
            py = ty + y_orig
            by_parent[parent_stem].append((px, py, cls_i, conf_f))
            for t in CONF_THRESHOLDS:
                if conf_f >= t:
                    raw_counts[t][cls_i] += 1

    # -------- Report 1: raw counts per conf threshold --------
    print()
    print('=' * 74)
    print('PRED COUNTS BY CONFIDENCE (raw, per-tile NMS only — no cross-tile dedup)')
    print('=' * 74)
    print(f'{"conf ≥":<10} {"single_pv":>12} {"double":>10} {"total":>10}')
    for t in CONF_THRESHOLDS:
        c = raw_counts[t]
        print(f'{t:<10.2f} {c[0]:>12,} {c[1]:>10,} {c[0]+c[1]:>10,}')

    # -------- Report 2: dedup by projecting to parent + greedy cluster --------
    print()
    print('=' * 74)
    print('NEAR-DUPLICATE ANALYSIS  (parent-crop frame, <20 px same-class)')
    print('=' * 74)
    print(f'{"conf ≥":<10} {"raw":>10} {"clusters":>10} {"dup %":>10}   {"dups in overlap":>20}   {"in-overlap %":>14}')
    for t in CONF_THRESHOLDS:
        raw = 0
        clusters_total = 0
        dup_count = 0
        dup_in_overlap = 0
        for parent_stem, preds in by_parent.items():
            filt = [p for p in preds if p[3] >= t]
            raw += len(filt)
            w, h = parent_dims[parent_stem]
            clusters = cluster_greedy(filt, DUP_RADIUS)
            clusters_total += len(clusters)
            # a duplicate is any pred beyond the first in its cluster
            for cluster in clusters:
                if len(cluster) > 1:
                    n_extra = len(cluster) - 1
                    dup_count += n_extra
                    # count extras whose center is in the overlap zone
                    for member in cluster[1:]:
                        if in_overlap_zone(member[0], member[1], w, h):
                            dup_in_overlap += 1
        dup_pct = (dup_count / raw * 100) if raw else 0
        in_ov_pct = (dup_in_overlap / dup_count * 100) if dup_count else 0
        print(f'{t:<10.2f} {raw:>10,} {clusters_total:>10,} {dup_pct:>9.1f}%   '
              f'{dup_in_overlap:>20,}   {in_ov_pct:>13.1f}%')

    # Interpretation aid: NMS scope
    print()
    print('NMS SCOPE: ultralytics.model.predict runs NMS *inside* each image')
    print('  (per-tile). It does NOT dedupe across tile seams. So predictions in')
    print(f'  the 128-px overlap band between adjacent tiles are counted twice.')


if __name__ == '__main__':
    main()
