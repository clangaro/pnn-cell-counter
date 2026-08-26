"""
One-off comparison of v1 (original *_idNN_XML) vs v2 (*_idNN_fixed_XML)
ZEN metadata.xml files in the Box "Cell Couting AI Pictures for Carolina"
folder. Read-only.

Outputs:
  1. Headline diff table (one row per animal).
  2. STEP 2 specific re-checks (id43 PIP, id69/id70 names, id80, id41, totals).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

BOX_ROOT = Path(
    '/Users/carolinalangaro/Library/CloudStorage/Box-Box/'
    'Cell Couting AI Pictures for Carolina'
)

# Which animals have a v2 (_fixed_) version. (id, v1_folder, v2_folder)
ANIMALS: list[tuple[str, str, str | None]] = [
    ('id41', '2025_01_08__2394_id41_XML',         None),
    ('id43', '2025_01_08__2395_id43_XML',         None),
    ('id45', '2025_01_08__2397_id45_XML',         None),
    ('id47', '2025_03_11__2683_id47_XML',         None),
    ('id51', '2025_03_11__2684_id51_XML',         '2025_03_11__2684_id51_fixed_XML'),
    ('id52', '2025_03_11__2685_id52_XML',         None),
    ('id57', '2025_03_11__2686_id57_XML',         '2025_03_11__2686_id57_fixed_XML'),
    ('id62', '2025_03_11__2688_id62_XML',         '2025_03_11__2686_id62_fixed_XML'),
    ('id58', '2025_03_11__2687_id58_XML',         None),
    ('id65', '2025_03_11__2689_id65_XML',         None),
    ('id71', '2025_05_08__3004_id71_XML',         None),
    ('id73', '2025_05_08__3009_id73_XML',         None),
    ('id74', '2025_05_08__3006_id74_XML',         '2025_05_08__3006_id74_fixed_XML'),
    ('id77', '2025_05_08__3008_id77_XML',         None),
    ('id78', '2025_05_08__3005_id78_XML',         '2025_05_08__3005_id78_fixed_XML'),
    ('id69', '2025_05_20__3221_id69_XML',         '2025_05_20__3221_id69_fixed_XML'),
    ('id70', '2025_05_20__3222_id70_XML',         '2025_05_20__3222_id70_fixed_XML'),
    ('id80', '2025_07_15__3472_id80_XML',         None),
    ('id83', '2025_07_15__3476_id83',             None),
]

STROKE_NAMES = {
    '#FFFF0000': 'red',
    '#FF9900CC': 'violet',
    '#FF0000FF': 'blue',
    '#FFFF69B4': 'pink',     # provisional
    '#FF00FFFF': 'cyan',     # provisional
    '#FFFFFF00': 'yellow',   # provisional
    '#FFFFA500': 'orange',   # provisional
}


def classify_stroke(value: str | None) -> str:
    """Bucket a <Stroke> hex value into a friendly category."""
    if value is None:
        return 'no_stroke'
    v = value.upper()
    if v == '#FFFF0000':
        return 'red'
    if v == '#FF9900CC':
        return 'violet'
    if v == '#FF0000FF':
        return 'blue'
    # heuristic hue check for the rare oddballs
    m = re.match(r'#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})$', v)
    if not m:
        return f'other({v})'
    _, r, g, b = (int(x, 16) for x in m.groups())
    if g > 200 and b > 200 and r < 100:
        return 'cyan'
    if r > 200 and g < 150 and b > 100:
        return 'pink'
    if r > 200 and g > 200 and b < 100:
        return 'yellow'
    if r > 200 and 80 < g < 200 and b < 100:
        return 'orange'
    return f'other({v})'


def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag


@dataclass
class Scene:
    start_x: int
    start_y: int
    size_x: int
    size_y: int

    def contains(self, x: float, y: float) -> bool:
        return (
            self.start_x <= x < self.start_x + self.size_x
            and self.start_y <= y < self.start_y + self.size_y
        )


@dataclass
class Bezier:
    name: str
    scene_idx: int | None  # 0-based scene this bezier sits in
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Arrow:
    stroke_bucket: str
    tip_x: float
    tip_y: float
    scene_idx: int | None  # 0-based scene this arrow tip sits in


def parse_info_xml(info_path: Path) -> list[Scene]:
    """Read <Bounds StartX="..." .../> entries, ordered by StartS index."""
    tree = ET.parse(info_path)
    scenes: dict[int, Scene] = {}
    for el in tree.iter():
        if _strip_ns(el.tag) != 'Bounds':
            continue
        try:
            ss = int(el.get('StartS', '0'))
            sx = int(el.get('StartX', '0'))
            sy = int(el.get('StartY', '0'))
            wx = int(el.get('SizeX', '0'))
            wy = int(el.get('SizeY', '0'))
        except ValueError:
            continue
        scenes[ss] = Scene(sx, sy, wx, wy)
    return [scenes[k] for k in sorted(scenes)]


def _scene_for(scenes: list[Scene], x: float, y: float) -> int | None:
    for i, s in enumerate(scenes):
        if s.contains(x, y):
            return i
    return None


def parse_metadata_xml(meta_path: Path, scenes: list[Scene]) -> tuple[list[Arrow], list[Bezier]]:
    tree = ET.parse(meta_path)
    arrows: list[Arrow] = []
    beziers: list[Bezier] = []
    for el in tree.iter():
        tag = _strip_ns(el.tag)
        if tag == 'Arrow':
            stroke = None
            x2 = y2 = None
            for child in el.iter():
                ct = _strip_ns(child.tag)
                if ct == 'Stroke' and child.text:
                    stroke = child.text.strip()
                elif ct == 'X2' and child.text:
                    try: x2 = float(child.text)
                    except ValueError: pass
                elif ct == 'Y2' and child.text:
                    try: y2 = float(child.text)
                    except ValueError: pass
            if x2 is None or y2 is None:
                continue
            arrows.append(Arrow(
                stroke_bucket=classify_stroke(stroke),
                tip_x=x2, tip_y=y2,
                scene_idx=_scene_for(scenes, x2, y2),
            ))
        elif tag == 'Bezier':
            name = ''
            pts: list[tuple[float, float]] = []
            # Region name lives at Bezier/Attributes/Name only — NOT inside
            # Features/Feature/Name (those are measurement labels like "Area",
            # "IntensityMean" and would shadow the real region name).
            for child in el:
                ct = _strip_ns(child.tag)
                if ct == 'Attributes':
                    for sub in child:
                        if _strip_ns(sub.tag) == 'Name' and sub.text:
                            name = sub.text.strip()
                            break
                elif ct == 'Geometry':
                    for sub in child:
                        if _strip_ns(sub.tag) == 'Points' and sub.text:
                            for token in sub.text.split():
                                if ',' in token:
                                    try:
                                        px, py = token.split(',', 1)
                                        pts.append((float(px), float(py)))
                                    except ValueError:
                                        pass
            scene_idx = None
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                scene_idx = _scene_for(scenes, cx, cy)
            beziers.append(Bezier(name=name, scene_idx=scene_idx, points=pts))
    return arrows, beziers


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Standard even-odd ray test."""
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    px, py = poly[-1]
    for qx, qy in poly:
        if ((qy > y) != (py > y)) and (
            x < (px - qx) * (y - qy) / (py - qy + 1e-12) + qx
        ):
            inside = not inside
        px, py = qx, qy
    return inside


def load_animal(folder: str) -> tuple[list[Arrow], list[Bezier]] | None:
    d = BOX_ROOT / folder
    if not d.is_dir():
        return None
    info = next(d.glob('*_info.xml'), None)
    meta = next(d.glob('*_metadata.xml'), None)
    if not info or not meta:
        return None
    scenes = parse_info_xml(info)
    return parse_metadata_xml(meta, scenes)


def summarize(arrows: list[Arrow], beziers: list[Bezier]) -> dict:
    stroke_counts = Counter(a.stroke_bucket for a in arrows)
    bezier_names = sorted({b.name for b in beziers})
    return {
        'arrow_total': len(arrows),
        'strokes': dict(stroke_counts),
        'bezier_total': len(beziers),
        'bezier_names': bezier_names,
    }


# ---------- main ----------
def fmt_strokes(d: dict[str, int]) -> str:
    order = ['red', 'violet', 'blue', 'pink', 'cyan', 'yellow', 'orange', 'no_stroke']
    bits = [f"{k}={d.get(k, 0)}" for k in order if d.get(k, 0)]
    extras = [f"{k}={v}" for k, v in d.items() if k not in order]
    return ', '.join(bits + extras) or '-'


def fmt_names(names: list[str], limit: int = 12) -> str:
    if not names:
        return '-'
    if len(names) <= limit:
        return '{' + ','.join(names) + '}'
    return '{' + ','.join(names[:limit]) + f',...+{len(names)-limit}}}'


def main() -> None:
    rows = []
    totals_v1 = Counter()
    totals_v2_only_for_changed = Counter()
    totals_v2_effective = Counter()  # v2 if exists else v1
    diff_rows = []

    for animal, v1_folder, v2_folder in ANIMALS:
        v1 = load_animal(v1_folder)
        v2 = load_animal(v2_folder) if v2_folder else None
        if v1 is None:
            print(f"!! v1 missing for {animal} at {v1_folder}")
            continue
        v1_sum = summarize(*v1)
        v2_sum = summarize(*v2) if v2 else None

        totals_v1.update(v1_sum['strokes'])
        if v2_sum:
            totals_v2_only_for_changed.update(v2_sum['strokes'])
            totals_v2_effective.update(v2_sum['strokes'])
        else:
            totals_v2_effective.update(v1_sum['strokes'])

        changed = bool(
            v2_sum and (
                v1_sum['arrow_total'] != v2_sum['arrow_total']
                or v1_sum['strokes'] != v2_sum['strokes']
                or v1_sum['bezier_total'] != v2_sum['bezier_total']
                or v1_sum['bezier_names'] != v2_sum['bezier_names']
            )
        )
        rows.append((animal, v1_sum, v2_sum, changed))
        if changed:
            diff_rows.append(animal)

    # --- print headline table ---
    print('Headline diff (animals with v2 only have v1 → v2 deltas; others have no v2)')
    print('=' * 100)
    header = (
        f"{'animal':6} {'Δ':2} "
        f"{'v1_arr':>7} {'v2_arr':>7} | {'v1_bez':>6} {'v2_bez':>6} | "
        f"v1_strokes / v2_strokes / v1_names / v2_names"
    )
    print(header)
    print('-' * 100)
    for animal, v1s, v2s, changed in rows:
        marker = '**' if changed else ('  ' if v2s is None else 'ok')
        v2_arr = v2s['arrow_total'] if v2s else '-'
        v2_bez = v2s['bezier_total'] if v2s else '-'
        print(
            f"{animal:6} {marker:2} "
            f"{v1s['arrow_total']:>7} {str(v2_arr):>7} | "
            f"{v1s['bezier_total']:>6} {str(v2_bez):>6}"
        )
        print(f"    v1 strokes: {fmt_strokes(v1s['strokes'])}")
        if v2s:
            print(f"    v2 strokes: {fmt_strokes(v2s['strokes'])}")
        print(f"    v1 names:   {fmt_names(v1s['bezier_names'])}")
        if v2s:
            print(f"    v2 names:   {fmt_names(v2s['bezier_names'])}")
        print()

    print(f"Rows where v2 differs from v1: {diff_rows or 'none'}")
    print()
    print('Totals across all animals:')
    print(f"  v1                : {fmt_strokes(totals_v1)}")
    print(f"  v2 (changed only) : {fmt_strokes(totals_v2_only_for_changed)}")
    print(f"  v2 effective      : {fmt_strokes(totals_v2_effective)}")
    print()

    # --- STEP 2 specifics ---
    print('=' * 100)
    print('STEP 2 specifics')
    print('=' * 100)

    # id43 PIP recompute (v1 vs v2 if exists)
    def pip_pct(folder: str) -> str:
        loaded = load_animal(folder)
        if loaded is None:
            return 'n/a'
        arrows, beziers = loaded
        red_or_violet = [a for a in arrows if a.stroke_bucket in ('red', 'violet')]
        if not red_or_violet:
            return 'no red/violet arrows'
        hit = 0
        for a in red_or_violet:
            for b in beziers:
                if b.scene_idx is not None and a.scene_idx == b.scene_idx and point_in_polygon(a.tip_x, a.tip_y, b.points):
                    hit += 1
                    break
        return f'{hit}/{len(red_or_violet)} = {100*hit/len(red_or_violet):.1f}%'

    # id43
    id43_v1, id43_v2 = '2025_01_08__2395_id43_XML', None
    for a, v1f, v2f in ANIMALS:
        if a == 'id43':
            id43_v2 = v2f
    print(f"id43 PIP v1: {pip_pct(id43_v1)}")
    if id43_v2:
        print(f"id43 PIP v2: {pip_pct(id43_v2)}")
    else:
        print("id43 PIP v2: no v2 produced for id43")

    # id69 / id70 Bezier names
    for tgt in ('id69', 'id70'):
        for a, v1f, v2f in ANIMALS:
            if a != tgt:
                continue
            v1_loaded = load_animal(v1f)
            v2_loaded = load_animal(v2f) if v2f else None
            v1_names = sorted({b.name for b in v1_loaded[1]}) if v1_loaded else []
            v2_names = sorted({b.name for b in v2_loaded[1]}) if v2_loaded else None
            print(f"{tgt} v1 bezier names: {fmt_names(v1_names)}")
            if v2_names is None:
                print(f"{tgt} v2: no v2")
            else:
                print(f"{tgt} v2 bezier names: {fmt_names(v2_names)}")

    # id80
    for a, v1f, v2f in ANIMALS:
        if a != 'id80':
            continue
        v1_loaded = load_animal(v1f)
        if v1_loaded:
            arrows, beziers = v1_loaded
            by_scene_arrow = Counter(a.scene_idx for a in arrows)
            by_scene_bez = Counter(b.scene_idx for b in beziers)
            scene_names = {}
            for b in beziers:
                scene_names.setdefault(b.scene_idx, set()).add(b.name)
            print(f"id80 v1 arrows per scene: {dict(by_scene_arrow)}")
            print(f"id80 v1 beziers per scene: {dict(by_scene_bez)}")
            for s, names in sorted(scene_names.items(), key=lambda x: (x[0] is None, x[0])):
                print(f"  v1 scene {s} bezier names: {sorted(names)}")
        if v2f:
            print("id80 v2: present")
        else:
            print("id80 v2: no v2 produced for id80")

    # id41 s3
    for a, v1f, v2f in ANIMALS:
        if a != 'id41':
            continue
        v1_loaded = load_animal(v1f)
        if v1_loaded:
            arrows, beziers = v1_loaded
            scene3_names = sorted({b.name for b in beziers if b.scene_idx == 2})  # s3 is 0-based idx 2? scene labels are 1-based
            # Some animals start scenes at s1, others at s0 (see id41 file names). Use info.xml's StartS — but here we just want a quick check on the highest-indexed scene.
            # Group beziers by scene_idx:
            by_scene_names = {}
            for b in beziers:
                by_scene_names.setdefault(b.scene_idx, set()).add(b.name)
            print(f"id41 v1 bezier names by scene_idx: " + ', '.join(
                f"s{k}={sorted(v)}" for k, v in sorted(by_scene_names.items(), key=lambda x: (x[0] is None, x[0]))
            ))
        if v2f:
            print("id41 v2: present")
        else:
            print("id41 v2: no v2 produced for id41")

    # No-stroke total
    v1_no_stroke = totals_v1.get('no_stroke', 0)
    v2_eff_no_stroke = totals_v2_effective.get('no_stroke', 0)
    print(f"No-stroke arrows: v1={v1_no_stroke}, v2_effective={v2_eff_no_stroke}")

    # Oddball totals
    oddballs = ('pink', 'cyan', 'yellow', 'orange')
    v1_odd = sum(totals_v1.get(k, 0) for k in oddballs)
    v2_eff_odd = sum(totals_v2_effective.get(k, 0) for k in oddballs)
    print(f"Oddball strokes (pink/cyan/yellow/orange): v1={v1_odd}, v2_effective={v2_eff_odd}")

    # ACA/ACC presence
    aca_in_v1 = []
    aca_in_v2 = []
    for a, v1f, v2f in ANIMALS:
        v1_loaded = load_animal(v1f)
        if v1_loaded and any(b.name in ('ACA', 'ACC') for b in v1_loaded[1]):
            aca_in_v1.append(a)
        if v2f:
            v2_loaded = load_animal(v2f)
            if v2_loaded and any(b.name in ('ACA', 'ACC') for b in v2_loaded[1]):
                aca_in_v2.append(a)
    print(f"Animals with ACA/ACC in v1: {aca_in_v1}")
    print(f"Animals with ACA/ACC in v2 (only checking the 7 fixed): {aca_in_v2}")


if __name__ == '__main__':
    main()
