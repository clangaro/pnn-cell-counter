"""
pnn_counter.crop_from_raw
========================

Reads arrow coordinates exported from burned-in overlays (CSV),
and (later) will crop arrow-free raw exports at those coordinates.

Step 1: load and group CSV rows (no cropping yet).
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ArrowCoord:
    arrow_id: str
    scene_id: int 
    label: str 
    x: int
    y: int

def load_arrow_coords(csv_path: Path) -> List[ArrowCoord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rows: List[ArrowCoord] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"arrow_id", "scene_id", "label", "x", "y"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV missing required columns: {sorted(required)}; got {reader.fieldnames}")
        
        for r in reader:
            rows.append(
                ArrowCoord(
                    arrow_id=r["arrow_id"],
                    scene_id=int(r["scene_id"]),
                    label=r["label"],
                    x=int(r["x"]),
                    y=int(r["y"]),
                )
            )

    logger.info("Loaded %d arrow coordinates from %s", len(rows), csv_path.name)
    return rows

def group_coords_by_scene(rows: List[ArrowCoord]) -> Dict[Tuple[str, int], List[ArrowCoord]]:
    """
    Groups arrow coordinates by (arrow_id, scene_id) so we can open each raw scene once and crop many coordinates.
    """
    grouped: Dict[Tuple[str, int], List[ArrowCoord]] = defaultdict(list)
    for r in rows:
        grouped[(r.arrow_id, r.scene_id)].append(r)
    logger.info("Grouped into %d (arrow_id, scene_id) buckets", len(grouped))
    return grouped

def notify_done(message: str) -> None:
    logger.info("Done: %s", message)

def main() -> None:
    csv_path = Path("/Users/carolinalangaro/Desktop/pnn-cell-counter/outputs/arrow_coords.csv")
    rows = load_arrow_coords(csv_path)
    grouped = group_coords_by_scene(rows)
    # Further cropping logic would go here.
    notify_done("Arrow coordinate loading and grouping finished.")

import re
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

FOCUS_PATCH = 64  # same idea as before

def focus_score_gray(patch: np.ndarray) -> float:
    return float(cv2.Laplacian(patch, cv2.CV_32F).var())

def pick_best_raw_z(id_folder: Path, scene_id: int, x: int, y: int) -> Optional[Path]:
    """
    Given an id folder (e.g., data/id_65), pick the sharpest z-slice
    for the specified scene around (x,y).
    """
    # Match files containing _s{scene}z{z}... with .jpg/.jpeg
    candidates = sorted(id_folder.glob(f"*s{scene_id}z*.jp*g"))
    if not candidates:
        return None

    best_path = None
    best_score = -1.0
    half = FOCUS_PATCH // 2

    for p in candidates:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x + half), min(h, y + half)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        s = focus_score_gray(patch)
        if s > best_score:
            best_score = s
            best_path = p

    return best_path

if __name__ == "__main__":
    main()