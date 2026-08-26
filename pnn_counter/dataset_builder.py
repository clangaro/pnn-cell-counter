"""
pnn_counter.dataset_builder

Turn parsed ZEN metadata + per-scene JPGs into per-ROI training crops and
YOLO detection labels.

Per DECISIONS.md (2026-06-03): the training input is the ROI crop (one image
per named Bezier polygon) with a fixed-size box centered on each arrow tip
that falls inside the crop rectangle. Class 0 = single_pv, class 1 = double.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .parse_metadata import (
    STROKE_BLUE_EXCLUDE,
    STROKE_LABEL,
    _host_scene,
    _local_tag,
    _parse_scene_bounds,
    normalize_roi_name,
)

# Give PIL room to load the huge scene JPGs (some are ~20k × 15k).
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class Bezier:
    name: str
    points_mosaic: list[tuple[float, float]]


@dataclass(frozen=True)
class Arrow:
    label: str  # 'single_pv' | 'double'
    x_mosaic: float
    y_mosaic: float


@dataclass(frozen=True)
class CropSpec:
    animal: str
    animal_dir: Path      # source folder — z-plane selection derives image paths from this
    scene_1based: int
    roi_name: str
    crop_x0: int          # in JPG pixel space
    crop_y0: int
    crop_x1: int
    crop_y1: int
    out_image_path: Path  # written PNG
    out_label_path: Path  # written YOLO label


def parse_beziers(metadata_xml: Path) -> list[Bezier]:
    """Extract named Bezier polygons from a ZEN metadata.xml."""
    out: list[Bezier] = []
    for el in ET.parse(metadata_xml).iter():
        if _local_tag(el) != 'Bezier':
            continue
        name = None
        points: list[tuple[float, float]] = []
        for sub in el.iter():
            t = _local_tag(sub)
            if t == 'Name' and name is None:
                name = (sub.text or '').strip()
            elif t == 'Points' and sub.text:
                for pair in sub.text.split():
                    if ',' in pair:
                        x, y = pair.split(',')
                        points.append((float(x), float(y)))
        if name and points:
            out.append(Bezier(name=name, points_mosaic=points))
    return out


def parse_arrows(metadata_xml: Path) -> list[Arrow]:
    """Extract red/violet-labelled arrows from a ZEN metadata.xml.
    Blue, no-stroke, and oddball colors are excluded."""
    out: list[Arrow] = []
    for el in ET.parse(metadata_xml).iter():
        if _local_tag(el) != 'Arrow':
            continue
        stroke = None
        x2 = y2 = None
        for sub in el.iter():
            t = _local_tag(sub)
            if t == 'Stroke':
                stroke = (sub.text or '').strip()
            elif t == 'X2':
                x2 = float(sub.text) if sub.text else None
            elif t == 'Y2':
                y2 = float(sub.text) if sub.text else None
        if stroke == STROKE_BLUE_EXCLUDE or x2 is None or y2 is None:
            continue
        label = STROKE_LABEL.get(stroke or '')
        if label is None:
            continue
        out.append(Arrow(label=label, x_mosaic=x2, y_mosaic=y2))
    return out


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def polygon_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def pad_bbox(
    bbox: tuple[float, float, float, float],
    pad_frac: float,
    clamp: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Pad by pad_frac of each side then clamp to (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    px, py = w * pad_frac, h * pad_frac
    cx0, cy0, cx1, cy1 = clamp
    return (
        max(int(x0 - px), cx0),
        max(int(y0 - py), cy0),
        min(int(x1 + px), cx1),
        min(int(y1 + py), cy1),
    )


_Z_FILE_RE = re.compile(r'_s(\d+)z(\d+)c1-2\.jpg$')


def scene_jpg_path(animal_dir: Path, scene_1based: int, z_1based: int) -> Path:
    """Path to the merged-RGB scene JPG for a specific z-plane."""
    base = animal_dir.name
    return animal_dir / f'{base}_s{scene_1based}z{z_1based}c1-2.jpg'


def available_z_planes(animal_dir: Path, scene_1based: int) -> list[int]:
    """Sorted list of z-plane indices for which a JPG exists on disk."""
    zs = []
    for p in animal_dir.glob(f'*_s{scene_1based}z*c1-2.jpg'):
        m = _Z_FILE_RE.search(p.name)
        if m and int(m.group(1)) == scene_1based:
            zs.append(int(m.group(2)))
    return sorted(zs)


def laplacian_focus_score(pil_img: Image.Image) -> float:
    """Variance-of-Laplacian focus score. Higher = sharper. Grayscale input.
    Standard bench for auto-focus / best-plane picking."""
    gray = np.asarray(pil_img.convert('L'))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_crops(
    animal_dir: Path,
    out_root: Path,
    pad_frac: float = 0.10,
) -> list[CropSpec]:
    """For each named Bezier ROI, compute the crop rectangle in JPG space
    and produce a CropSpec. Does not yet write files — z-plane is chosen
    per crop at write time.

    Beziers whose <Name> is 'Area' or empty are skipped (only named ROIs
    are cropped in the smoke test)."""
    metadata_xml = next(animal_dir.glob('*_metadata.xml'))
    info_xml = next(animal_dir.glob('*_info.xml'))
    scenes = _parse_scene_bounds(info_xml)
    beziers = parse_beziers(metadata_xml)

    specs: list[CropSpec] = []
    for bez in beziers:
        canonical = normalize_roi_name(bez.name)
        if not canonical or canonical == 'Area':
            continue
        cx, cy = polygon_centroid(bez.points_mosaic)
        sidx = _host_scene(scenes, cx, cy)
        if sidx is None:
            continue
        sx, sy, szx, szy = scenes[sidx]
        jpg_points = [(p[0] - sx, p[1] - sy) for p in bez.points_mosaic]
        bbox = polygon_bbox(jpg_points)
        x0, y0, x1, y1 = pad_bbox(bbox, pad_frac, clamp=(0, 0, szx, szy))
        if x1 <= x0 or y1 <= y0:
            continue
        scene_1based = sidx + 1
        stem = f'{_animal_short(animal_dir.name)}_s{scene_1based}_{canonical}'
        specs.append(CropSpec(
            animal=animal_dir.name,
            animal_dir=animal_dir,
            scene_1based=scene_1based,
            roi_name=canonical,
            crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1,
            out_image_path=out_root / 'images' / f'{stem}.png',
            out_label_path=out_root / 'labels' / f'{stem}.txt',
        ))
    return specs


def _animal_short(folder_name: str) -> str:
    """Compact stem for filenames — the 'idNN' segment if present, else folder name."""
    m = re.search(r'id\d+', folder_name)
    return m.group(0) if m else folder_name


def write_crop_image(spec: CropSpec, z_1based: int) -> None:
    """Load the scene JPG at the given z, crop to the spec rectangle, save PNG."""
    jpg_path = scene_jpg_path(spec.animal_dir, spec.scene_1based, z_1based)
    spec.out_image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(jpg_path) as img:
        crop = img.crop((spec.crop_x0, spec.crop_y0, spec.crop_x1, spec.crop_y1))
        crop.convert('RGB').save(spec.out_image_path, format='PNG')


def arrows_for_scene(arrows: list[Arrow], scenes, scene_1based: int) -> list[Arrow]:
    """Arrows whose tip falls inside the given scene's bounds."""
    sidx = scene_1based - 1
    if sidx not in scenes:
        return []
    sx, sy, szx, szy = scenes[sidx]
    out = []
    for a in arrows:
        if sx <= a.x_mosaic < sx + szx and sy <= a.y_mosaic < sy + szy:
            out.append(a)
    return out


def arrows_in_crop(arrows_scene: list[Arrow], scene_start: tuple[int, int],
                   crop: tuple[int, int, int, int]) -> list[tuple[str, float, float]]:
    """Return [(label, crop_x, crop_y)] for arrows whose tip lies in the crop rect.
    Coordinates are crop-local pixels (origin at crop top-left)."""
    sx, sy = scene_start
    x0, y0, x1, y1 = crop
    out = []
    for a in arrows_scene:
        jx, jy = a.x_mosaic - sx, a.y_mosaic - sy
        if x0 <= jx < x1 and y0 <= jy < y1:
            out.append((a.label, jx - x0, jy - y0))
    return out


CLASS_IDX = {'single_pv': 0, 'double': 1}


def tile_grid(w: int, h: int, tile: int = 640, stride: int = 512) -> list[tuple[int, int, int, int]]:
    """Generate 640×640 tile rectangles covering a w×h image.

    Stride < tile → overlap between adjacent tiles. Last tile in each row/col
    snaps to the right/bottom edge so no black padding is needed and every
    tile is exactly (tile, tile). If w or h is < tile, one tile at (0,0)
    covering as much as fits is returned.
    """
    if w <= 0 or h <= 0:
        return []
    xs: list[int] = []
    x = 0
    while x + tile < w:
        xs.append(x)
        x += stride
    xs.append(max(0, w - tile))  # snap final tile to right edge
    ys: list[int] = []
    y = 0
    while y + tile < h:
        ys.append(y)
        y += stride
    ys.append(max(0, h - tile))
    # dedupe (small images may collapse to a single tile)
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    out = []
    for yy in ys:
        for xx in xs:
            out.append((xx, yy, min(xx + tile, w), min(yy + tile, h)))
    return out


def arrows_in_tile(arrows_crop_local: list[tuple[str, float, float]],
                   tile: tuple[int, int, int, int]) -> list[tuple[str, float, float]]:
    """Return [(label, tile_x, tile_y)] for arrows whose tip lies in the tile.
    Coordinates are tile-local pixels."""
    x0, y0, x1, y1 = tile
    out = []
    for label, cx, cy in arrows_crop_local:
        if x0 <= cx < x1 and y0 <= cy < y1:
            out.append((label, cx - x0, cy - y0))
    return out


def write_yolo_label(
    label_path: Path,
    arrows_local: list[tuple[str, float, float]],
    crop_wh: tuple[int, int],
    box_size_px: int,
) -> None:
    """Write YOLO label lines: `cls cx cy w h` normalized to crop dims.
    Each arrow becomes a fixed-size square box centered on the tip
    (crop-local pixel coords)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cw, ch = crop_wh
    with label_path.open('w') as f:
        for label, jx, jy in arrows_local:
            cls = CLASS_IDX[label]
            cx = jx / cw
            cy = jy / ch
            w = box_size_px / cw
            h = box_size_px / ch
            f.write(f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
