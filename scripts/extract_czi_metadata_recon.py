"""
Phase A CZI recon — dump raw metadata XML from one or more CZI files.

Purpose:
    Verify how ZEN stores annotation graphics (arrows, beziers) INSIDE the
    raw CZI. We've been auditing ZEN's *exported* `_metadata.xml`; the CZI's
    internal metadata block may organise those elements differently. One CZI
    dump lets us finalize a production extractor that emits ZEN-compatible
    `<name>.jpg_info.xml` + `<name>.jpg_metadata.xml` files, at which point
    the existing audit runs unchanged over any CZI archive.

Usage:
    python scripts/extract_czi_metadata_recon.py <path/to/one.czi>
    python scripts/extract_czi_metadata_recon.py <path/to/directory>
    python scripts/extract_czi_metadata_recon.py <path> --out <out_dir>

Output (per input CZI, written to --out, default ./czi_recon/):
    <stem>.recon.xml         raw CZI metadata block, pretty-printed
    <stem>.recon_summary.txt element counts + likely annotation container tags

Dependencies (either works, script picks whichever installs):
    pip install aicspylibczi        # community standard
    pip install pylibCZIrw           # Zeiss official; wheel install, no compile

If neither is present the script prints install instructions and exits.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


READER_HELP = """
No CZI reader available. Install one of:

    pip install aicspylibczi          # preferred
        or
    pip install pylibCZIrw            # fallback, wheel-only install

then re-run this script.
""".strip()


def _load_reader() -> tuple[str, object]:
    """Return (name, module) of the first CZI reader that imports."""
    try:
        import aicspylibczi  # noqa: F401
        return 'aicspylibczi', aicspylibczi
    except ModuleNotFoundError:
        pass
    try:
        import pylibCZIrw   # noqa: F401
        return 'pylibCZIrw', pylibCZIrw
    except ModuleNotFoundError:
        pass
    sys.exit(READER_HELP)


def _get_metadata_xml(reader_name: str, czi_path: Path) -> str:
    """Return the CZI's full metadata block as a UTF-8 XML string."""
    if reader_name == 'aicspylibczi':
        from aicspylibczi import CziFile
        czi = CziFile(str(czi_path))
        # `czi.meta` is an lxml.etree.Element (whole <ImageDocument> tree).
        try:
            from lxml import etree as LET
            return LET.tostring(czi.meta, pretty_print=True, encoding='unicode')
        except ModuleNotFoundError:
            return ET.tostring(czi.meta, encoding='unicode')
    if reader_name == 'pylibCZIrw':
        from pylibCZIrw import czi as czirw
        with czirw.open_czi(str(czi_path)) as doc:
            # `.raw_metadata` returns the full metadata XML as a string.
            return doc.raw_metadata
    raise RuntimeError(f'unhandled reader {reader_name}')


def _summarize(xml_text: str) -> str:
    """Structural summary: tag counts, likely annotation containers."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return f'XML PARSE FAILED: {e}\n(dump still written; inspect the .xml file manually)'

    def local(el: ET.Element) -> str:
        return el.tag.rsplit('}', 1)[-1] if '}' in el.tag else el.tag

    tag_counter: Counter = Counter()
    parents_of_interest = ('Arrow', 'Bezier', 'Layer', 'Layers', 'Elements', 'Graphics', 'Annotations')
    interest_paths: list[str] = []

    def walk(el: ET.Element, path: str):
        t = local(el)
        tag_counter[t] += 1
        here = f'{path}/{t}'
        if t in parents_of_interest:
            interest_paths.append(here)
        for child in el:
            walk(child, here)

    walk(root, '')

    lines = ['=== tag counts (top 25) ===']
    for tag, n in tag_counter.most_common(25):
        lines.append(f'  {n:>6}  {tag}')

    lines.append('')
    lines.append('=== interesting elements (annotation-shaped tags) ===')
    keys = ('Arrow', 'Bezier', 'Layer', 'Layers', 'Elements', 'Graphics', 'Annotations',
            'Stroke', 'Points', 'X1', 'Y1', 'X2', 'Y2', 'Name', 'Attributes', 'Geometry')
    for k in keys:
        n = tag_counter.get(k, 0)
        lines.append(f'  {k:<14} count={n}')

    lines.append('')
    lines.append('=== first 15 paths containing annotation-shaped containers ===')
    for p in interest_paths[:15]:
        lines.append(f'  {p}')
    if len(interest_paths) > 15:
        lines.append(f'  ... +{len(interest_paths) - 15} more')

    return '\n'.join(lines)


def _collect_czis(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() == '.czi' else []
    if target.is_dir():
        return sorted(p for p in target.rglob('*.czi') if p.is_file())
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__ or '')
    ap.add_argument('path', help='CZI file or directory containing CZIs')
    ap.add_argument('--out', default='czi_recon',
                    help='output directory (default: ./czi_recon/)')
    args = ap.parse_args()

    target = Path(args.path).expanduser().resolve()
    czis = _collect_czis(target)
    if not czis:
        sys.exit(f'No .czi files found at {target}')

    reader_name, _ = _load_reader()
    print(f'reader: {reader_name}')
    print(f'input : {target}   ({len(czis)} CZI(s))')

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print()

    for czi in czis:
        stem = czi.stem
        print(f'== {czi.name} ==')
        try:
            xml_text = _get_metadata_xml(reader_name, czi)
        except Exception as e:
            print(f'   ERROR reading metadata: {e}')
            continue
        xml_path = out_dir / f'{stem}.recon.xml'
        xml_path.write_text(xml_text, encoding='utf-8')
        summary = _summarize(xml_text)
        (out_dir / f'{stem}.recon_summary.txt').write_text(summary + '\n', encoding='utf-8')
        # Peek: show first 40 lines of summary in-terminal
        for line in summary.splitlines()[:40]:
            print(f'   {line}')
        print(f'   → wrote {xml_path.name}  ({len(xml_text):,} chars)')
        print(f'   → wrote {xml_path.with_suffix("").with_suffix(".recon_summary.txt").name}')
        print()

    print('Done. Send the recon.xml + recon_summary.txt files back for review.')


if __name__ == '__main__':
    main()
