# pnn-cell-counter

Automated counting of parvalbumin-positive (PV+) cells and perineuronal nets (PNNs) in confocal images of mouse brain sections.

Counting PV+ cells by hand, and judging whether each one is surrounded by a PNN, is slow. This repository trains an object detector on annotations made in ZEN to find PV+ somata and classify each one as `single_pv` (no PNN) or `double` (PV+ with a PNN). It is work in progress: the detector runs end to end, but it is not yet accurate enough to replace manual counts.

## Data

Merged two-channel confocal images (PV and PNN stains) of annotated brain regions (for example AC, PL, ILA and CA1), exported from ZEN with the annotators' arrows and region polygons as XML. Sections of prefrontal cortex and hippocampus were double-labelled with a rabbit anti-parvalbumin antibody (PV+ cells) and *Wisteria floribunda* agglutinin (WFA; perineuronal nets).

Images, annotations and trained weights are not included. They are available on request, pending lab approval.

## Method

The class of each annotated cell is read from the arrow colour in the ZEN XML, and `scripts/build_smoke_dataset.py` crops each annotated region from the sharpest z-plane, placing a fixed 60 px box on each arrow tip. Each crop is cut into 640 × 640 px tiles at native resolution with a stride of 512 px, so neighbouring tiles overlap by 128 px. Tiles without an annotated cell are dropped, and a cell inside an overlap band is labelled in both tiles. A YOLOv8n model (ultralytics, COCO-pretrained) is trained on the tiles, with training and validation split by animal so that no animal contributes to both. At prediction time, non-maximum suppression (NMS) runs within each tile only, so a cell in an overlap band can be detected twice. `pnn-count-tiles` maps every tile's predictions back into its parent crop and counts same-class detections within 20 px of each other as duplicates.

Design decisions and the reasons for them are logged in [DECISIONS.md](DECISIONS.md). An existing PNN counter (Ciampi et al. 2022) was considered and not used, because it segments PNNs rather than classifying PV+ cells by PNN status.

## Current results

Output of `pnn-count-tiles` with the current weights on the 380 validation tiles from one held-out animal:

| Confidence ≥ | Predictions | Same-class clusters (20 px) | Duplicates, % of predictions | In overlap bands, % of duplicates |
|---|---|---|---|---|
| 0.05 | 46,738 | 12,756 | 72.7% | 72.9% |
| 0.25 | 9,950 | 2,756 | 72.3% | 73.6% |
| 0.50 | 3,337 | 1,087 | 67.4% | 77.1% |

The validation set has 1,571 annotated PV+ cells and 235 PNNs across 380 validation tiles. Tiles overlap, so a cell near a tile edge can be counted twice. `yolo val` on the same weights and tiles gives mAP50 = 0.0205 overall (`single_pv` 0.0377, `double` 0.00334).

## Limitations

- The tiled detector over-predicts. Most surplus predictions are duplicates of the same cell, and roughly three quarters of those duplicates lie in tile overlap bands.
- The confidence threshold, and NMS across tile overlaps, are still being evaluated. No merged cell count is reported yet.
- Validation uses a single held-out animal, so the numbers above are noisy.

## How to run

```bash
pip install -e .                                    # Python 3.10+
python scripts/build_smoke_dataset.py <data_root>   # region crops and labels
python scripts/build_phase2_tiles.py                # 640 px tiles
python scripts/build_phase2_split.py                # split by animal
python scripts/train_phase2.py                      # YOLOv8n training
pnn-count-tiles --weights <best.pt> --device cpu    # see --help for all options
```

Only `pnn-count-tiles` has a command-line interface. The training, tiling and render scripts in `scripts/` take their paths from the repository layout, write to `outputs/`, and must be run from the repository root on Apple silicon (`device="mps"`).

`scripts/extract_czi_annotations.py` converts raw CZI files into the same XML format. It is the only part that needs `aicspylibczi`, installed with `pip install -e ".[czi]"`.

## Reference

Ciampi L, Carrara F, Totaro V, Mazziotti R, Lupori L, Santiago C, Amato G, Pizzorusso T, Gennaro C. Learning to count biological structures with raters' uncertainty. Medical Image Analysis 80: 102500 (2022).

## Licence

MIT
