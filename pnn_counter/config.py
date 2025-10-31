from pathlib import Path

CROP_SIZE: int = 160
FOCUS_PATCH: int = 64
OUTPUT_DIR: Path = Path("dataset")

RANGES = {
    "without_pnn": [ #red
        ((0, 100, 80), (10, 255, 255), (160, 80, 80), (179, 255, 255)),
    ],
    "with_pnn": [ #purple ((135, 80, 60), (165, 255, 255)),
    ],
}

(OUTPUT_DIR / "with_pnn").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "without_pnn").mkdir(parents=True, exist_ok=True)
