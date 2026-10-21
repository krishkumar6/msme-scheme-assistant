"""Give a scanned (image-only) PDF an invisible text layer, so ingestion can read it unchanged.

Pages that already have text are kept as they are. Image-only pages are rendered, OCR'd with the
engine built into Windows (scripts/ocr_pages.ps1), and each recognised line is written back as
invisible text at its position on the page, sized to the line height. Reading order, row sorting
and heading detection in the extractor then work as for a born-digital PDF.

    python scripts/make_searchable_pdf.py scanned.pdf data/raw/out.pdf [--dpi 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

OCR_SCRIPT = Path(__file__).with_name("ocr_pages.ps1")
MIN_TEXT_CHARS = 50  # a page with less real text than this is treated as a scan


def needs_ocr(page: pymupdf.Page) -> bool:
    return len(page.get_text().strip()) < MIN_TEXT_CHARS


def ocr_dir(folder: Path) -> None:
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(OCR_SCRIPT), str(folder)], check=True)


def add_text_layer(page: pymupdf.Page, lines: list[dict], scale: float) -> int:
    """Write OCR lines as invisible text (render_mode=3). `scale` converts pixels to points."""
    written = 0
    for ln in lines:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        x, y, h = ln["x"] * scale, ln["y"] * scale, max(ln["h"] * scale, 4.0)
        size = h * 0.85
        # stretch the line horizontally to the recognised width so word positions roughly match
        natural = pymupdf.get_text_length(text, fontname="helv", fontsize=size) or 1.0
        page.insert_text((x, y + size), text, fontname="helv", fontsize=size, render_mode=3,
                         morph=(pymupdf.Point(x, y + size), pymupdf.Matrix(ln["w"] * scale / natural, 1)))
        written += 1
    return written


def make_searchable(src: Path, dst: Path, dpi: int = 200) -> tuple[int, int]:
    doc = pymupdf.open(src)
    todo = [p.number for p in doc if needs_ocr(p)]
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for n in todo:
                doc[n].get_pixmap(dpi=dpi).save(folder / f"p{n:04d}.png")
            ocr_dir(folder)
            scale = 72.0 / dpi
            for n in todo:
                lines = json.loads((folder / f"p{n:04d}.json").read_text(encoding="utf-8-sig") or "[]")
                add_text_layer(doc[n], lines if isinstance(lines, list) else [lines], scale)
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst, garbage=3, deflate=True)
    return len(todo), len(doc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("OCR uses the Windows OCR engine; on Linux use `ocrmypdf src dst` instead.")
    ocrd, total = make_searchable(a.src, a.dst, a.dpi)
    print(f"{a.dst}: OCR'd {ocrd} of {total} pages")


if __name__ == "__main__":
    main()
