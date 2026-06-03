"""
pnn_counter.parse_metadata

Parse ZEN annotation XML to extract neuron labels and per-scene JPG coordinates.

Each animal folder is expected to contain:
    <base>.jpg_metadata.xml   ZEN annotation overlay (Arrow elements)
    <base>.jpg_info.xml       per-scene Bounds (StartX/StartY/SizeX/SizeY)
    <base>_s#z#c1-2.jpg       merged-RGB scene exports

Stroke handling:
    #FFFF0000  red     -> single_pv  (training label)
    #FF9900CC  violet  -> double     (training label)
    #FF0000FF  blue    -> masked, silent (annotator-flagged "exclude")
    anything else      -> masked + reported as anomaly

Coordinate transform: jpg_x = mosaic_x - scene.StartX, jpg_y = mosaic_y - scene.StartY,
with host scene chosen by bounds containment of the arrow tip (X2, Y2) from info.xml.
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    '/Users/carolinalangaro/Library/CloudStorage/Box-Box/Cell Couting AI Pictures for Carolina'
)

STROKE_LABEL = {
    '#FFFF0000': 'single_pv',
    '#FF9900CC': 'double',
}
STROKE_BLUE_EXCLUDE = '#FF0000FF'

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArrowRecord:
    animal: str
    scene_idx_0based: int   # StartS in info.xml
    scene_idx_1based: int   # the s# in JPG filenames
    label: str              # 'single_pv' or 'double'
    jpg_x: int
    jpg_y: int
    tip_x_mosaic: float
    tip_y_mosaic: float


@dataclass(frozen=True)
class Anomaly:
    animal: str
    arrow_xml_id: str       # not unique within a file; useful only as a hint
    stroke: str | None      # None == <Stroke> element absent
    tip_x_mosaic: float
    tip_y_mosaic: float


def _local_tag(elem: ET.Element) -> str:
    return elem.tag.split('}')[-1]


def _parse_scene_bounds(info_xml: Path) -> dict[int, tuple[int, int, int, int]]:
    scenes: dict[int, tuple[int, int, int, int]] = {}
    for img in ET.parse(info_xml).iter():
        if _local_tag(img) != 'Image':
            continue
        bounds = next((e for e in img.iter() if _local_tag(e) == 'Bounds'), None)
        if bounds is None:
            continue
        s = int(bounds.attrib['StartS'])
        scenes.setdefault(s, (
            int(bounds.attrib['StartX']),
            int(bounds.attrib['StartY']),
            int(bounds.attrib['SizeX']),
            int(bounds.attrib['SizeY']),
        ))
    return scenes


def _host_scene(
    scenes: dict[int, tuple[int, int, int, int]],
    x: float,
    y: float,
) -> int | None:
    for s, (sx, sy, szx, szy) in scenes.items():
        if sx <= x < sx + szx and sy <= y < sy + szy:
            return s
    return None


def parse_animal(animal_dir: Path) -> tuple[list[ArrowRecord], list[Anomaly], int]:
    """Parse one animal. Returns (records, anomalies, blue_excluded_count)."""
    metadata_xml = next(animal_dir.glob('*_metadata.xml'))
    info_xml = next(animal_dir.glob('*_info.xml'))

    scenes = _parse_scene_bounds(info_xml)
    records: list[ArrowRecord] = []
    anomalies: list[Anomaly] = []
    blue_excluded = 0

    for arrow in ET.parse(metadata_xml).iter():
        if _local_tag(arrow) != 'Arrow':
            continue
        arrow_id = arrow.attrib.get('Id', '')
        stroke: str | None = None
        x2 = y2 = None
        for sub in arrow.iter():
            t = _local_tag(sub)
            if t == 'Stroke':
                stroke = (sub.text or '').strip()
            elif t == 'X2':
                x2 = float(sub.text)
            elif t == 'Y2':
                y2 = float(sub.text)

        if x2 is None or y2 is None:
            anomalies.append(Anomaly(animal_dir.name, arrow_id, stroke, 0.0, 0.0))
            continue

        if stroke == STROKE_BLUE_EXCLUDE:
            blue_excluded += 1
            continue

        label = STROKE_LABEL.get(stroke or '')
        if label is None:
            anomalies.append(Anomaly(animal_dir.name, arrow_id, stroke, x2, y2))
            continue

        sidx = _host_scene(scenes, x2, y2)
        if sidx is None:
            anomalies.append(Anomaly(animal_dir.name, arrow_id, stroke, x2, y2))
            continue

        sx, sy, _, _ = scenes[sidx]
        records.append(ArrowRecord(
            animal=animal_dir.name,
            scene_idx_0based=sidx,
            scene_idx_1based=sidx + 1,
            label=label,
            jpg_x=int(round(x2 - sx)),
            jpg_y=int(round(y2 - sy)),
            tip_x_mosaic=x2,
            tip_y_mosaic=y2,
        ))

    return records, anomalies, blue_excluded


def parse_all(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[list[ArrowRecord], list[Anomaly], dict[str, int]]:
    all_records: list[ArrowRecord] = []
    all_anomalies: list[Anomaly] = []
    blue_per_animal: dict[str, int] = {}

    for animal_dir in sorted(data_root.iterdir()):
        if not animal_dir.is_dir():
            continue
        if not list(animal_dir.glob('*_metadata.xml')):
            logger.warning('skipping %s — no *_metadata.xml found', animal_dir.name)
            continue
        records, anomalies, blue = parse_animal(animal_dir)
        all_records.extend(records)
        all_anomalies.extend(anomalies)
        blue_per_animal[animal_dir.name] = blue

    return all_records, all_anomalies, blue_per_animal


def write_records_csv(records: list[ArrowRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['animal', 'scene_idx_0based', 'scene_idx_1based', 'label',
              'jpg_x', 'jpg_y', 'tip_x_mosaic', 'tip_y_mosaic']
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def _print_anomalies(anomalies: list[Anomaly], max_per_kind: int = 10) -> None:
    if not anomalies:
        print('  (none — XMLs are clean)')
        return
    by_kind: dict[tuple[str, str | None], list[Anomaly]] = defaultdict(list)
    for a in anomalies:
        by_kind[(a.animal, a.stroke)].append(a)
    for (animal, stroke), items in sorted(by_kind.items(), key=lambda kv: (kv[0][0], kv[0][1] or '')):
        label = stroke if stroke else '<no Stroke>'
        print(f'  {animal}  stroke={label}  count={len(items)}')
        for a in items[:max_per_kind]:
            print(f'    arrow_xml_id={a.arrow_xml_id}  tip=({a.tip_x_mosaic:.0f}, {a.tip_y_mosaic:.0f})')
        if len(items) > max_per_kind:
            print(f'    ... +{len(items) - max_per_kind} more (same kind)')


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    records, anomalies, blue_per_animal = parse_all()

    by_animal_label: dict[str, dict[str, int]] = defaultdict(lambda: {'single_pv': 0, 'double': 0})
    for r in records:
        by_animal_label[r.animal][r.label] += 1

    print('\n=== labelled arrows per animal ===')
    print(f'{"animal":<42}{"single_pv":>11}{"double":>9}{"blue_excl":>11}')
    for animal in sorted(set(by_animal_label) | set(blue_per_animal)):
        counts = by_animal_label.get(animal, {'single_pv': 0, 'double': 0})
        blue = blue_per_animal.get(animal, 0)
        print(f'{animal:<42}{counts["single_pv"]:>11}{counts["double"]:>9}{blue:>11}')
    total_sp = sum(c['single_pv'] for c in by_animal_label.values())
    total_d = sum(c['double'] for c in by_animal_label.values())
    total_blue = sum(blue_per_animal.values())
    print(f'{"TOTAL":<42}{total_sp:>11}{total_d:>9}{total_blue:>11}')

    print('\n=== anomalies (masked from dataset; non-red/blue/violet) ===')
    _print_anomalies(anomalies)

    out_csv = Path(__file__).resolve().parents[1] / 'outputs' / 'labels.csv'
    write_records_csv(records, out_csv)
    print(f'\nwrote {len(records)} labelled records to {out_csv}')


if __name__ == '__main__':
    main()
