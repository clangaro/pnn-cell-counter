"""
Extract ZEN-compatible annotation XMLs from raw CZI files and triage each
animal for training usefulness.

For each CZI in the input path (file or directory), writes:
    <out>/<stem>/<stem>.jpg_metadata.xml   Arrows + Beziers, from CZI meta block
    <out>/<stem>/<stem>.jpg_info.xml       Per-scene Bounds, from CZI sub-blocks

Both files match the schema pnn_counter/parse_metadata.py already consumes
(the parser uses recursive .iter(), so the CZI's <Attributes>/<Geometry>
nesting is fine). Feed <out>/ to parse_metadata.py as a data_root and it
runs unchanged.

Also writes:
    <out>/triage_report.md
    <out>/triage_report.csv

Verdict per animal is computed under two scenarios (see DECISIONS.md open
question re: no-Stroke arrows):
    strict       — no-Stroke arrows are dropped as anomalies (current parser)
    red_default  — no-Stroke arrows are treated as red / single_pv

Verdict rules (applied to each scenario, then combined):
    useful    — both scenarios workable: >=5 arrows in each class AND >=1
                semantic (non-Area) Bezier AND <=50% arrows outside scenes
    marginal — at least one scenario workable but with caveats: one class
                dominates >=90%, or Beziers all Area/empty, or scene coverage
                marginal
    drop      — neither scenario yields >=10 usable arrows

Usage:
    python scripts/extract_czi_annotations.py <path/to/czi_or_dir>
    python scripts/extract_czi_annotations.py <path> --out czi_extracts

Requires aicspylibczi. If no wheel exists for your Python version, use a
Python 3.12 environment.
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STROKE_RED = '#FFFF0000'
STROKE_VIOLET = '#FF9900CC'
STROKE_BLUE = '#FF0000FF'
LABEL_STROKES = {STROKE_RED, STROKE_VIOLET}

MIN_ARROWS_TO_KEEP = 10
MIN_PER_CLASS_USEFUL = 5
CLASS_DOMINANCE_MARGIN = 0.90
MAX_OFF_SCENE_FRAC = 0.50


@dataclass
class ArrowRow:
    stroke: str | None
    x2: float | None
    y2: float | None


@dataclass
class AnimalReport:
    stem: str
    czi_path: Path
    scenes: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    arrows: list[ArrowRow] = field(default_factory=list)
    beziers_named: int = 0
    beziers_area: int = 0
    bezier_names: list[str] = field(default_factory=list)
    metadata_out: Path | None = None
    info_out: Path | None = None
    error: str | None = None

    def stroke_counts(self) -> Counter:
        c: Counter = Counter()
        for a in self.arrows:
            key = a.stroke if a.stroke else '<none>'
            c[key] += 1
        return c

    def in_scene_fraction(self) -> float:
        if not self.arrows:
            return 0.0
        inside = 0
        eligible = 0
        for a in self.arrows:
            if a.x2 is None or a.y2 is None:
                continue
            eligible += 1
            for sx, sy, w, h in self.scenes.values():
                if sx <= a.x2 < sx + w and sy <= a.y2 < sy + h:
                    inside += 1
                    break
        return inside / eligible if eligible else 0.0

    def scenario(self, *, no_stroke_is_red: bool) -> dict:
        red = 0
        violet = 0
        blue = 0
        anomaly = 0
        for a in self.arrows:
            s = a.stroke
            if s == STROKE_RED:
                red += 1
            elif s == STROKE_VIOLET:
                violet += 1
            elif s == STROKE_BLUE:
                blue += 1
            elif s is None:
                if no_stroke_is_red:
                    red += 1
                else:
                    anomaly += 1
            else:
                anomaly += 1
        labeled = red + violet
        return {
            'single_pv': red,
            'double': violet,
            'blue': blue,
            'anomaly': anomaly,
            'labeled': labeled,
        }


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def _get_reader():
    try:
        import aicspylibczi

        return aicspylibczi
    except ModuleNotFoundError:
        sys.exit(
            'aicspylibczi is not installed for this Python: pip install aicspylibczi\n'
            '(if there is no wheel for your Python version, use a Python 3.12 environment)'
        )


def _load_metadata_element(czi_path: Path):
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))
    meta = czi.meta
    return czi, meta


def _serialize_metadata(meta) -> str:
    if type(meta).__module__.startswith('lxml'):
        from lxml import etree as LET

        return LET.tostring(meta, pretty_print=True, encoding='unicode')
    ET.indent(meta, space='  ')
    return ET.tostring(meta, encoding='unicode')


def _extract_scenes(czi) -> dict[int, tuple[int, int, int, int]]:
    # aicspylibczi returns per-scene bounds in absolute stage units. ZEN's
    # exported _info.xml normalizes them so the mosaic bounding box origin
    # is (0, 0) — the same space Arrow/Bezier coordinates live in.
    # For non-mosaic single-tile CZIs, get_mosaic_bounding_box raises; fall
    # back to SizeX/SizeY from the metadata XML as a single scene at index 0.
    try:
        mb = czi.get_mosaic_bounding_box()
    except RuntimeError as e:
        if 'not a mosaic file' not in str(e):
            raise
        size_x = size_y = None
        for el in czi.meta.iter():
            t = el.tag.rsplit('}', 1)[-1] if '}' in el.tag else el.tag
            if size_x is None and t == 'SizeX' and el.text and el.text.strip().isdigit():
                size_x = int(el.text.strip())
            elif size_y is None and t == 'SizeY' and el.text and el.text.strip().isdigit():
                size_y = int(el.text.strip())
            if size_x is not None and size_y is not None:
                break
        if size_x is None or size_y is None:
            raise RuntimeError('non-mosaic CZI with no SizeX/SizeY in metadata')
        return {0: (0, 0, size_x, size_y)}
    ox, oy = int(mb.x), int(mb.y)
    scenes: dict[int, tuple[int, int, int, int]] = {}
    for s, bb in czi.get_all_scene_bounding_boxes().items():
        scenes[int(s)] = (int(bb.x) - ox, int(bb.y) - oy, int(bb.w), int(bb.h))
    return scenes


def _build_info_xml(scenes: dict[int, tuple[int, int, int, int]]) -> str:
    root = ET.Element('ImageMetadata')
    images = ET.SubElement(root, 'Images')
    for s in sorted(scenes):
        sx, sy, w, h = scenes[s]
        img = ET.SubElement(images, 'Image')
        ET.SubElement(
            img,
            'Bounds',
            {
                'StartS': str(s),
                'StartX': str(sx),
                'StartY': str(sy),
                'SizeX': str(w),
                'SizeY': str(h),
            },
        )
    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode')


def _scan_annotations(meta) -> tuple[list[ArrowRow], int, int, list[str]]:
    arrows: list[ArrowRow] = []
    beziers_named = 0
    beziers_area = 0
    bezier_names: list[str] = []

    for el in meta.iter() if hasattr(meta, 'iter') else []:
        tag = _local(el.tag)
        if tag == 'Arrow':
            stroke = None
            x2 = y2 = None
            for sub in el.iter():
                t = _local(sub.tag)
                if t == 'Stroke':
                    stroke = (sub.text or '').strip() or None
                elif t == 'X2' and sub.text:
                    try:
                        x2 = float(sub.text)
                    except ValueError:
                        pass
                elif t == 'Y2' and sub.text:
                    try:
                        y2 = float(sub.text)
                    except ValueError:
                        pass
            arrows.append(ArrowRow(stroke=stroke, x2=x2, y2=y2))
        elif tag == 'Bezier':
            name = None
            for sub in el.iter():
                if _local(sub.tag) == 'Name' and name is None:
                    name = (sub.text or '').strip()
                    break
            name = (name or '').strip()
            if not name or name.lower() == 'area':
                beziers_area += 1
            else:
                beziers_named += 1
                bezier_names.append(name)
    return arrows, beziers_named, beziers_area, bezier_names


def _verdict(report: AnimalReport) -> tuple[str, str]:
    strict = report.scenario(no_stroke_is_red=False)
    red_default = report.scenario(no_stroke_is_red=True)
    off_scene = 1.0 - report.in_scene_fraction()
    all_area_beziers = report.beziers_named == 0 and report.beziers_area > 0

    def scenario_verdict(s: dict) -> str:
        if s['labeled'] < MIN_ARROWS_TO_KEEP:
            return 'drop'
        r, v = s['single_pv'], s['double']
        total = r + v
        if r >= MIN_PER_CLASS_USEFUL and v >= MIN_PER_CLASS_USEFUL:
            if off_scene > MAX_OFF_SCENE_FRAC:
                return 'marginal'
            if all_area_beziers:
                return 'marginal'
            majority = max(r, v) / total
            if majority >= CLASS_DOMINANCE_MARGIN:
                return 'marginal'
            return 'useful'
        return 'marginal'

    v_strict = scenario_verdict(strict)
    v_reddef = scenario_verdict(red_default)

    if v_strict == 'useful' or v_reddef == 'useful':
        combined = 'useful'
    elif v_strict == 'drop' and v_reddef == 'drop':
        combined = 'drop'
    else:
        combined = 'marginal'

    notes: list[str] = []
    if off_scene > MAX_OFF_SCENE_FRAC:
        notes.append(f'{off_scene:.0%} arrows off-scene')
    if all_area_beziers:
        notes.append('all-Area Beziers (no region info)')
    if v_strict != v_reddef:
        notes.append(f'scenario-sensitive: strict={v_strict}, red_default={v_reddef}')
    return combined, '; '.join(notes) if notes else ''


def _process_one(czi_path: Path, out_root: Path) -> AnimalReport:
    stem = czi_path.stem
    report = AnimalReport(stem=stem, czi_path=czi_path)
    animal_dir = out_root / stem
    # Failures are recorded per animal and reported, so one bad CZI does not stop the batch.
    try:
        czi, meta = _load_metadata_element(czi_path)
    except Exception as e:  # noqa: BLE001
        report.error = f'CZI load failed: {e}'
        return report

    try:
        report.scenes = _extract_scenes(czi)
    except Exception as e:  # noqa: BLE001
        report.error = f'scene bounding boxes failed: {e}'
        return report

    animal_dir.mkdir(parents=True, exist_ok=True)
    metadata_out = animal_dir / f'{stem}.jpg_metadata.xml'
    info_out = animal_dir / f'{stem}.jpg_info.xml'

    metadata_out.write_text(_serialize_metadata(meta), encoding='utf-8')
    info_out.write_text(_build_info_xml(report.scenes), encoding='utf-8')
    report.metadata_out = metadata_out
    report.info_out = info_out

    arrows, named, area, names = _scan_annotations(meta)
    report.arrows = arrows
    report.beziers_named = named
    report.beziers_area = area
    report.bezier_names = names
    return report


def _collect(target: Path) -> list[Path]:
    # Skip macOS AppleDouble resource-fork stubs ('._<name>.czi') — they share
    # the .czi suffix but aren't valid CZI files, and blow up the CZI reader.
    def _is_real_czi(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() == '.czi' and not p.name.startswith('._')

    if target.is_file():
        return [target] if _is_real_czi(target) else []
    if target.is_dir():
        return sorted(p for p in target.rglob('*.czi') if _is_real_czi(p))
    return []


def _write_reports(reports: list[AnimalReport], out_root: Path) -> None:
    csv_path = out_root / 'triage_report.csv'
    md_path = out_root / 'triage_report.md'

    fieldnames = [
        'stem',
        'verdict',
        'notes',
        'arrows_total',
        'strict_single_pv',
        'strict_double',
        'red_default_single_pv',
        'red_default_double',
        'blue',
        'other_anomaly',
        'no_stroke',
        'beziers_named',
        'beziers_area',
        'bezier_names',
        'scenes',
        'off_scene_frac',
        'error',
    ]
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in reports:
            if r.error:
                w.writerow({'stem': r.stem, 'error': r.error, 'verdict': 'error'})
                continue
            strict = r.scenario(no_stroke_is_red=False)
            reddef = r.scenario(no_stroke_is_red=True)
            counts = r.stroke_counts()
            other_anom = sum(
                n
                for s, n in counts.items()
                if s not in {STROKE_RED, STROKE_VIOLET, STROKE_BLUE, '<none>'}
            )
            verdict, notes = _verdict(r)
            w.writerow(
                {
                    'stem': r.stem,
                    'verdict': verdict,
                    'notes': notes,
                    'arrows_total': len(r.arrows),
                    'strict_single_pv': strict['single_pv'],
                    'strict_double': strict['double'],
                    'red_default_single_pv': reddef['single_pv'],
                    'red_default_double': reddef['double'],
                    'blue': strict['blue'],
                    'other_anomaly': other_anom,
                    'no_stroke': counts.get('<none>', 0),
                    'beziers_named': r.beziers_named,
                    'beziers_area': r.beziers_area,
                    'bezier_names': ','.join(sorted(set(r.bezier_names))),
                    'scenes': len(r.scenes),
                    'off_scene_frac': f'{1.0 - r.in_scene_fraction():.2f}',
                    'error': '',
                }
            )

    lines: list[str] = ['# CZI triage report', '']
    lines.append(f'Animals scanned: {len(reports)}')
    by_verdict: Counter = Counter()
    for r in reports:
        if r.error:
            by_verdict['error'] += 1
        else:
            by_verdict[_verdict(r)[0]] += 1
    for v in ('useful', 'marginal', 'drop', 'error'):
        if by_verdict[v]:
            lines.append(f'- **{v}**: {by_verdict[v]}')
    lines.append('')
    lines.append(
        '| stem | verdict | arrows | strict s/d | red_default s/d | blue | other | no-Stroke | beziers (named/Area) | off-scene | notes |'
    )
    lines.append(
        '|------|---------|--------|-----------|-----------------|------|-------|-----------|----------------------|-----------|-------|'
    )
    for r in reports:
        if r.error:
            lines.append(f'| {r.stem} | ERROR |  |  |  |  |  |  |  |  | {r.error} |')
            continue
        strict = r.scenario(no_stroke_is_red=False)
        reddef = r.scenario(no_stroke_is_red=True)
        counts = r.stroke_counts()
        other_anom = sum(
            n
            for s, n in counts.items()
            if s not in {STROKE_RED, STROKE_VIOLET, STROKE_BLUE, '<none>'}
        )
        verdict, notes = _verdict(r)
        lines.append(
            f'| {r.stem} | **{verdict}** | {len(r.arrows)} | '
            f'{strict["single_pv"]}/{strict["double"]} | '
            f'{reddef["single_pv"]}/{reddef["double"]} | '
            f'{strict["blue"]} | {other_anom} | {counts.get("<none>", 0)} | '
            f'{r.beziers_named}/{r.beziers_area} | '
            f'{1.0 - r.in_scene_fraction():.0%} | {notes} |'
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nwrote {csv_path}')
    print(f'wrote {md_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__ or '')
    ap.add_argument('path', help='CZI file or directory')
    ap.add_argument(
        '--out', default='czi_extracts', help='output directory (default: ./czi_extracts/)'
    )
    args = ap.parse_args()

    _get_reader()
    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        sys.exit(f'Path does not exist: {target}')
    czis = _collect(target)
    if not czis:
        sys.exit(f'No .czi files found at {target}')

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'input : {target}   ({len(czis)} CZI(s))')
    print(f'output: {out_root}')

    reports: list[AnimalReport] = []
    for czi in czis:
        print(f'\n== {czi.name} ==')
        r = _process_one(czi, out_root)
        reports.append(r)
        if r.error:
            print(f'   ERROR: {r.error}')
            continue
        strict = r.scenario(no_stroke_is_red=False)
        reddef = r.scenario(no_stroke_is_red=True)
        verdict, notes = _verdict(r)
        counts = r.stroke_counts()
        print(f'   arrows total : {len(r.arrows)}   scenes: {len(r.scenes)}')
        print(
            f'   strict       : single_pv={strict["single_pv"]}  double={strict["double"]}  blue={strict["blue"]}  anomaly={strict["anomaly"]}'
        )
        print(f'   red_default  : single_pv={reddef["single_pv"]}  double={reddef["double"]}')
        print(f'   no-Stroke    : {counts.get("<none>", 0)}')
        print(f'   beziers      : {r.beziers_named} named / {r.beziers_area} Area')
        if r.bezier_names:
            print(f'   region names : {", ".join(sorted(set(r.bezier_names)))}')
        print(f'   off-scene    : {(1.0 - r.in_scene_fraction()):.0%}')
        print(f'   verdict      : {verdict}' + (f'   ({notes})' if notes else ''))
        print(f'   wrote        : {r.metadata_out.relative_to(out_root)}')
        print(f'                  {r.info_out.relative_to(out_root)}')

    _write_reports(reports, out_root)


if __name__ == '__main__':
    main()
