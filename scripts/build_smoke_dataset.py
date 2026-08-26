"""
Phase-1 smoke-test dataset builder.

Runs Stage 1 (ROI crop extraction) + Stage 2 (YOLO label generation) + QC
render, for a fixed 3-animal set. See DECISIONS.md 2026-08-03 for the choice
of animals, box size, and split.

Per-crop z-plane selection: for each ROI crop, every available z-plane is
scored with variance-of-Laplacian and the sharpest z is used as the source.
This replaces the earlier "always z=1" policy.

Usage:
    ~/anaconda3/bin/python scripts/build_smoke_dataset.py
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image, ImageDraw

from pnn_counter.dataset_builder import (
    Arrow,
    arrows_for_scene,
    arrows_in_crop,
    available_z_planes,
    build_crops,
    laplacian_focus_score,
    parse_arrows,
    scene_jpg_path,
    write_crop_image,
    write_yolo_label,
)
from pnn_counter.parse_metadata import _parse_scene_bounds

BOX = Path('/Users/carolinalangaro/Library/CloudStorage/Box-Box/Cell Couting AI Pictures for Carolina')
OUT_ROOT = REPO / 'outputs' / 'smoke_dataset'
QC_DIR = OUT_ROOT / 'qc'

ANIMALS = {
    'id47': '2025_03_11__2683_id47_XML',
    'id52': '2025_03_11__2685_id52_XML',
    'id83': '2025_07_15__3476_id83',
}

# QC crops the user asked to see re-rendered — animal + scene + roi.
QC_TARGETS = {
    ('2025_07_15__3476_id83',       2, 'AC'),
    ('2025_03_11__2683_id47_XML',   2, 'ILA'),
    ('2025_03_11__2685_id52_XML',   1, 'PL'),
    ('2025_03_11__2683_id47_XML',   3, 'CA1'),
}


def nn_distances_mosaic(arrows: list[Arrow], sample_n: int = 50, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    sample = arrows if len(arrows) <= sample_n else rng.sample(arrows, sample_n)
    out = []
    for a in sample:
        best = math.inf
        for b in arrows:
            if b is a:
                continue
            d = math.hypot(a.x_mosaic - b.x_mosaic, a.y_mosaic - b.y_mosaic)
            if d < best:
                best = d
        if best != math.inf:
            out.append(best)
    return out


def render_qc(image_path: Path, arrows_local: list[tuple[str, float, float]],
              box_size_px: int, out_path: Path) -> None:
    """Draw YOLO boxes over a crop and save. red=single_pv, magenta=double."""
    with Image.open(image_path) as im:
        rgb = im.convert('RGB').copy()
    draw = ImageDraw.Draw(rgb)
    for label, cx, cy in arrows_local:
        color = (255, 0, 0) if label == 'single_pv' else (255, 0, 255)
        half = box_size_px / 2
        draw.rectangle([cx - half, cy - half, cx + half, cy + half],
                       outline=color, width=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, format='PNG')


def pick_best_z_per_scene(animal_dir: Path, specs_this_scene: list) -> dict:
    """For a group of specs on the same scene, load each available z-plane
    once, score every spec's crop from that z, and return {spec: (best_z, scores)}."""
    if not specs_this_scene:
        return {}
    scene_1based = specs_this_scene[0].scene_1based
    z_planes = available_z_planes(animal_dir, scene_1based)
    scores = {s: {} for s in specs_this_scene}
    for z in z_planes:
        jpg = scene_jpg_path(animal_dir, scene_1based, z)
        if not jpg.exists():
            continue
        with Image.open(jpg) as im:
            for spec in specs_this_scene:
                crop = im.crop((spec.crop_x0, spec.crop_y0, spec.crop_x1, spec.crop_y1))
                scores[spec][z] = laplacian_focus_score(crop)
    return {s: (max(sc, key=sc.get) if sc else None, sc) for s, sc in scores.items()}


def main() -> None:
    print('=' * 70)
    print('PHASE 1 SMOKE DATASET BUILD  (best-z per crop)')
    print('=' * 70)

    # ---- pick box size from NN distances (unchanged) ----
    all_arrows = []
    for aid, folder in ANIMALS.items():
        metadata_xml = next((BOX / folder).glob('*_metadata.xml'))
        all_arrows.extend(parse_arrows(metadata_xml))
    nn = nn_distances_mosaic(all_arrows, sample_n=50, seed=42)
    nn.sort()
    median_nn = nn[len(nn) // 2]
    p25_nn = nn[len(nn) // 4]
    print(f'\nNN-distance sample (n={len(nn)}): '
          f'p25={p25_nn:.0f}px  median={median_nn:.0f}px  '
          f'min={nn[0]:.0f}  max={nn[-1]:.0f}')
    raw = int(0.6 * p25_nn)
    box_size_px = max(60, min(100, (raw // 10) * 10))
    print(f'Chosen box size: {box_size_px}px  '
          f'(raw 0.6 * p25_nn = {raw}, clamped to [60, 100])')

    all_specs = []
    total_arrows_labelled = 0
    winning_z_log: dict[tuple[str, int, str], tuple[int, dict[int, float]]] = {}

    for aid, folder in ANIMALS.items():
        animal_dir = BOX / folder
        print(f'\n--- {aid}  ({folder}) ---')
        info_xml = next(animal_dir.glob('*_info.xml'))
        metadata_xml = next(animal_dir.glob('*_metadata.xml'))
        scenes = _parse_scene_bounds(info_xml)
        arrows = parse_arrows(metadata_xml)

        specs = build_crops(animal_dir, OUT_ROOT, pad_frac=0.10)
        # group by scene so we open each z-JPG once per scene
        by_scene: dict[int, list] = defaultdict(list)
        for spec in specs:
            by_scene[spec.scene_1based].append(spec)

        for scene_1based in sorted(by_scene):
            specs_this_scene = by_scene[scene_1based]
            z_planes = available_z_planes(animal_dir, scene_1based)
            print(f'  scene {scene_1based}: {len(specs_this_scene)} crops, '
                  f'z-planes available = {z_planes}')
            best = pick_best_z_per_scene(animal_dir, specs_this_scene)

            for spec in specs_this_scene:
                best_z, scores = best[spec]
                if best_z is None:
                    print(f'    [!] no z-plane could be scored for '
                          f'{spec.roi_name} — skipping')
                    continue
                write_crop_image(spec, best_z)
                arrows_scene = arrows_for_scene(arrows, scenes, spec.scene_1based)
                sx, sy, _, _ = scenes[spec.scene_1based - 1]
                crop_rect = (spec.crop_x0, spec.crop_y0, spec.crop_x1, spec.crop_y1)
                local = arrows_in_crop(arrows_scene, (sx, sy), crop_rect)
                cw = spec.crop_x1 - spec.crop_x0
                ch = spec.crop_y1 - spec.crop_y0
                write_yolo_label(spec.out_label_path, local, (cw, ch), box_size_px)
                total_arrows_labelled += len(local)
                all_specs.append((spec, local))
                winning_z_log[(spec.animal, spec.scene_1based, spec.roi_name)] = (best_z, scores)
                score_str = ' '.join(f'z{z}={s:.0f}' for z, s in sorted(scores.items()))
                print(f'    {spec.out_image_path.name}  crop={cw}x{ch}  '
                      f'arrows={len(local)}  best_z={best_z}  [{score_str}]')

    print(f'\nTotal crops written: {len(all_specs)}')
    print(f'Total labelled arrows across crops: {total_arrows_labelled}')

    # ---- QC render: the 4 crops the user asked for ----
    print(f'\nRendering QC for user-requested crops:')
    for spec, local in all_specs:
        key = (spec.animal, spec.scene_1based, spec.roi_name)
        if key not in QC_TARGETS:
            continue
        qc_path = QC_DIR / f'{spec.out_image_path.stem}_annotated.png'
        render_qc(spec.out_image_path, local, box_size_px, qc_path)
        wz, _ = winning_z_log[key]
        print(f'  {qc_path.relative_to(REPO)}   ({len(local)} boxes, from z={wz})')

    # ---- summary ----
    from collections import Counter
    class_counter = Counter()
    for _, local in all_specs:
        for label, _, _ in local:
            class_counter[label] += 1
    print(f'\nClass distribution across all crops:')
    for cls, n in sorted(class_counter.items()):
        print(f'  {cls:12s} {n}')

    # ---- winning-z summary ----
    from collections import Counter as C
    z_dist = C()
    for (aid, s, r), (bz, _) in winning_z_log.items():
        z_dist[bz] += 1
    print(f'\nWinning z-plane distribution across {len(winning_z_log)} crops:')
    for z, n in sorted(z_dist.items()):
        print(f'  z={z}: {n} crops')


if __name__ == '__main__':
    main()
