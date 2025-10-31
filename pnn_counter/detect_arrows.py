"""
pnn_counter.detect_arrows
=========================

Detects red and purple arrows burned into exported confocal images,
crops corresponding neurons, and classifies them as single-labelled
(without perineuronal nets) or double-labelled (with PNNs).

Domain logic:
-------------
- Red arrows  → neurons without PNNs
- Purple arrows → neurons with PNNs
- Laplacian variance selects the sharpest Z-plane per neuron
- Mixed-hue or boundary-like shapes are filtered out to prevent false positives

This module can be run directly or imported for reuse in a CNN preprocessing pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------

CROP_SIZE: int = 160
FOCUS_PATCH: int = 64
OUTPUT_DIR: Path = Path("dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "with_pnn").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "without_pnn").mkdir(parents=True, exist_ok=True)

# HSV hue thresholds for detecting arrows (OpenCV hue range = [0, 179])
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
# Logging configuration
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Focus and sharpness metrics
# -------------------------------------------------------------------------


def focus_score(patch: np.ndarray) -> float:
    """
    Compute Laplacian variance (focus metric).

    Parameters
    ----------
    patch : np.ndarray
        Grayscale image patch.

    Returns
    -------
    float
        Variance of Laplacian filter (higher = sharper focus).
    """
    return float(cv2.Laplacian(patch, cv2.CV_32F).var())


def pick_best_z(folder: Path, scene_idx: int, x: int, y: int) -> Optional[Path]:
    """
    Select the Z-plane with maximum Laplacian focus near (x, y).

    Prevents data leakage by using only one slice per neuron.

    Parameters
    ----------
    folder : Path
        Folder containing all scene images.
    scene_idx : int
        Scene index (s0, s1, ...).
    x, y : int
        Pixel coordinates in scene space.

    Returns
    -------
    Optional[Path]
        Path to sharpest Z-image, or None if unavailable.
    """
    z_imgs = sorted(folder.glob(f"*s{scene_idx}z*.jpg"))
    best_z, best_score = None, -1.0

    for zf in z_imgs:
        img = cv2.imread(str(zf), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        half = FOCUS_PATCH // 2
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(w, x1 + 2 * half), min(h, y1 + 2 * half)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        score = focus_score(patch)
        if score > best_score:
            best_score = score
            best_z = zf

    return best_z


# -------------------------------------------------------------------------
# Arrow detection and filtering
# -------------------------------------------------------------------------


def is_mixed_color(mask: np.ndarray, img_hsv: np.ndarray) -> bool:
    """
    Determine if a region contains multiple hues (e.g. red + purple overlap).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of the contour region.
    img_hsv : np.ndarray
        HSV representation of the full image.

    Returns
    -------
    bool
        True if hue variance > threshold (mixed color).
    """
    hues = img_hsv[mask > 0, 0]
    if hues.size == 0:
        return False
    return bool(np.var(hues) > 400)  # empirically tuned threshold


def is_boundary_shape(contour: np.ndarray) -> bool:
    """
    Identify large, elongated red boundaries (non-arrow artifacts).

    Parameters
    ----------
    contour : np.ndarray
        Contour array.

    Returns
    -------
    bool
        True if contour likely represents a drawn boundary line.
    """
    _, _, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    return area > 3000 or aspect_ratio > 10


def find_arrows_clean(img_rgb: np.ndarray) -> List[Tuple[int, int, str]]:
    """
    Detect clean arrow tips in an RGB image.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image.

    Returns
    -------
    List[Tuple[int, int, str]]
        List of (x_tip, y_tip, label) for valid arrows.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    detections: List[Tuple[int, int, str]] = []

    for label, bounds in RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], np.uint8)

        for (low, high) in bounds:
            mask_total = np.bitwise_or(mask_total, cv2.inRange(hsv, np.array(low), np.array(high)))

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
            detections.append((tip_x, tip_y, label))

    return detections


# -------------------------------------------------------------------------
# Cropping and I/O
# -------------------------------------------------------------------------


def crop_patch(img: np.ndarray, cx: int, cy: int, label: str, name: str, idx: int) -> None:
    """
    Crop a fixed-size patch centered on (cx, cy) and save to dataset.

    Parameters
    ----------
    img : np.ndarray
        RGB source image.
    cx, cy : int
        Crop center coordinates.
    label : str
        Output label ("with_pnn" or "without_pnn").
    name : str
        Original filename stem.
    idx : int
        Unique crop index.
    """
    h, w = img.shape[:2]
    half = CROP_SIZE // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)

    crop = img[y1:y2, x1:x2]
    out = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    out[:crop.shape[0], :crop.shape[1]] = crop

    out_path = OUTPUT_DIR / label / f"{Path(name).stem}_{idx:04d}.png"
    cv2.imwrite(str(out_path), out)
    logger.info("Saved crop: %s", out_path.name)


def group_by_scene(folder: Path) -> dict[int, list[Path]]:
    """
    Return mapping {scene_idx: [all Z image paths]}.

    Example filename:
        2025_05_08__3005_id78-Image Export-06_s2z3c0-2x67949-18607y40392-11264.jpg
        -> scene_idx = 2, z_idx = 3
    """
    scenes = defaultdict(list)
    all_imgs = sorted(folder.glob("*.jp*g"))  # include .jpg/.jpeg/.JPG

    if not all_imgs:
        logger.error("❌ No JPEG images found in folder: %s", folder)
        return scenes

    logger.info("Found %d images total in %s", len(all_imgs), folder.name)

    for img_path in all_imgs:
        # match patterns like _s0z0_, _s3z2_, etc.
        m = re.search(r"[sS](\d+)[zZ](\d+)", img_path.name)
        if not m:
            logger.warning("⚠️ Skipping (no s#z# pattern): %s", img_path.name)
            continue

        s_idx = int(m.group(1))
        z_idx = int(m.group(2))
        scenes[s_idx].append(img_path)
        logger.debug("Matched scene %d z %d from %s", s_idx, z_idx, img_path.name)

    if not scenes:
        logger.error("❌ No valid scene indices found. Check naming pattern: expected _s#z#_")
    else:
        logger.info("[DEBUG] Scene keys found: %s", sorted(scenes.keys()))

    return scenes


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------


def process_folder(folder_path: str) -> None:
    """
    Process all scenes in a given folder, generating clean crops.

    Parameters
    ----------
    folder_path : str
        Path to folder containing images (with burned-in arrows).
    """
    folder = Path(folder_path)
    scenes = group_by_scene(folder)
    logger.info("Found %d scenes.", len(scenes))

    for scene_idx, z_imgs in sorted(scenes.items()):
        base_img = cv2.imread(str(z_imgs[0]))
        if base_img is None:
            logger.warning("Could not read %s", z_imgs[0])
            continue

        detections = find_arrows_clean(base_img)
        logger.info("[Scene %d] %d arrows detected after filtering", scene_idx, len(detections))

        for i, (cx, cy, label) in enumerate(detections):
            best_z_path = pick_best_z(folder, scene_idx, cx, cy)
            if not best_z_path:
                continue
            img_best = cv2.imread(str(best_z_path))
            if img_best is None:
                logger.warning("Could not read %s", best_z_path)
                continue
            crop_patch(img_best, cx, cy, label, best_z_path.name, i)

        # Save overlay for QC
        overlay = base_img.copy()
        for (cx, cy, label) in detections:
            color = (0, 255, 255) if label == "with_pnn" else (0, 0, 255)
            cv2.circle(overlay, (int(cx), int(cy)), 10, color, 2)
        out_overlay = folder / f"scene{scene_idx}_overlay.png"
        cv2.imwrite(str(out_overlay), overlay)
        logger.info("Overlay saved: %s", out_overlay.name)

    logger.info("All scenes processed. Crops saved to %s", OUTPUT_DIR)


def main() -> None:
    """Entry point for command-line execution."""
    process_folder("/Users/carolinalangaro/Desktop/pnn-cell-counter/data/id_78")


if __name__ == "__main__":
    main()