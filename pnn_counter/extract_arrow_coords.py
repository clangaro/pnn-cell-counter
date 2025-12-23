"""
pnn_counter.extract_arrow_coords
===============================

Extract arrow tip coordinates and class labels from images where arrows are
burned in (overlay / base images).

Outputs a CSV:
scene_id,label,x,y
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# HSV thresholds copied from your detect_arrows.py
# -------------------------------------------------------------------------

RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "without_pnn": [  # red
        ((0, 100, 80), (10, 255, 255)),
        ((160, 100, 80), (179, 255, 255)),
    ],
    "with_pnn": [  # purple / magenta
        ((135, 80, 60), (165, 255, 255)),
    ],
}

# -------------------------------------------------------------------------
# Filtering logic copied from your detect_arrows.py
# -------------------------------------------------------------------------

def is_mixed_color(mask: np.ndarray, img_hsv: np.ndarray) -> bool:
    hues = img_hsv[mask > 0, 0]
    if hues.size == 0:
        return False
    return bool(np.var(hues) > 400)

def is_boundary_shape(contour: np.ndarray) -> bool:
    _, _, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    return area > 3000 or aspect_ratio > 10

def find_arrows_clean(img_bgr: np.ndarray) -> List[Tuple[int, int, str]]:
    """
    Returns list of (x_tip, y_tip, label).
    NOTE: input is BGR as read by cv2.imread.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    detections: List[Tuple[int, int, str]] = []

    for label, bounds in RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], np.uint8)

        for (low, high) in bounds:
            mask_total = np.bitwise_or(
                mask_total,
                cv2.inRange(hsv, np.array(low), np.array(high)),
            )

        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 25 or is_boundary_shape(c):
                continue

            mask_c = np.zeros(hsv.shape[:2], np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)

            if is_mixed_color(mask_c, hsv):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            pts = c.reshape(-1, 2)
            d2 = np.sum((pts - np.array([cx, cy])) ** 2, axis=1)
            tip_x, tip_y = pts[np.argmax(d2)]
            detections.append((int(tip_x), int(tip_y), label))

    return detections

# -------------------------------------------------------------------------
# Scene grouping (reuse your filename parsing)
# -------------------------------------------------------------------------

def group_by_scene(folder: Path) -> dict[int, list[Path]]:
    scenes = defaultdict(list)
    for img_path in sorted(folder.glob("*.jp*g")):
        m = re.search(r"[sS](\d+)[zZ](\d+)", img_path.name)
        if not m:
            continue
        s_idx = int(m.group(1))
        scenes[s_idx].append(img_path)
    return scenes

# -------------------------------------------------------------------------
# CSV writing
# -------------------------------------------------------------------------

def write_scene_rows(csv_path: Path, arrow_id: str, scene_idx: int, detections: List[Tuple[int, int, str]]) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["arrow_id", "scene_id", "label", "x", "y"])
        for (x, y, label) in detections:
            w.writerow([arrow_id, scene_idx, label, x, y])

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def extract_folder(folder_path: str, out_csv: str = "outputs/arrow_coords.csv") -> None:
    folder = Path(folder_path)
    arrow_id = folder.name
    scenes = group_by_scene(folder)
    print("DEBUG scenes keys:", sorted(scenes.keys()))
    if not scenes:
        raise RuntimeError(f"No scenes found in {folder}. Expected filenames containing s#z#.")

    out_csv_path = Path(out_csv)


    logger.info("Found %d scenes in %s", len(scenes), folder.name)

    for scene_idx, z_imgs in sorted(scenes.items()):
        # Use first z image as the “base” image to read arrows from (as you do now)
        base_img = cv2.imread(str(z_imgs[0]))
        if base_img is None:
            logger.warning("Could not read %s", z_imgs[0])
            continue

        det = find_arrows_clean(base_img)
        logger.info("[Scene %d] %d arrows detected", scene_idx, len(det))
        write_scene_rows(out_csv_path, arrow_id, scene_idx, det)

    logger.info("Saved arrow coordinates to %s", out_csv_path)

def main() -> None:
    # Change id folder as needed
    extract_folder("/Users/carolinalangaro/Desktop/pnn-cell-counter/data/id_45")

if __name__ == "__main__":
    main()
