"""PDF -> clean page text, robust to broken font encodings.

Strategy
--------
* PyMuPDF gives good *layout* (blocks, lines, table rows, coordinates) but, for
  fonts whose ToUnicode map is incomplete, wrong *characters*.
* pdfminer exposes raw character codes, which we decode through the font's
  ``/Differences`` glyph names (see ``glyphs.py``) -> correct *characters*.
* We take PyMuPDF's layout and replace each character on affected pages with the
  pdfminer character found at the same position on the page.

Pages are often two-page spreads (landscape), so each spread is split into a
left and right "printed page" and the printed page number is captured for
citations.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from pdfminer.layout import LTChar, LTContainer
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdffont import PDFCIDFont, PDFSimpleFont
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.converter import PDFPageAggregator
from pdfminer.psparser import PSLiteral

from .glyphs import gid_to_char, glyph_name_to_char

log = logging.getLogger(__name__)

EXTRACTOR_VERSION = 3  # bump when output changes, so per-file extraction caches are rebuilt
PRIVATE_USE_BASE = 0xE000  # unknown glyph ids are emitted here, stripped later
_GLYPH_ID_NAME = re.compile(rb"/g\d+\b")
_PAGE_LABEL = re.compile(r"^(\d{1,3}|[ivxlc]{1,7})$", re.I)


# --------------------------------------------------------------------------- #
# pdfminer patch: decode glyph-id names instead of trusting a partial ToUnicode
# --------------------------------------------------------------------------- #
def _install_pdfminer_glyph_fix() -> None:
    if getattr(PDFSimpleFont, "_nabard_glyph_fix", False):
        return
    simple_init = PDFSimpleFont.__init__
    simple_to_unichr = PDFSimpleFont.to_unichr
    cid_to_unichr = PDFCIDFont.to_unichr

    def patched_init(self, descriptor, widths, spec):  # type: ignore[no-untyped-def]
        simple_init(self, descriptor, widths, spec)
        self._gid_fix = {}
        encoding = spec.get("Encoding")
        try:
            from pdfminer.pdftypes import resolve1

            encoding = resolve1(encoding)
            diffs = resolve1(encoding.get("Differences", [])) if isinstance(encoding, dict) else []
        except Exception:  # noqa: BLE001 - malformed encodings are just skipped
            diffs = []
        code = 0
        for item in diffs:
            if isinstance(item, int):
                code = item
                continue
            if isinstance(item, PSLiteral):
                name = item.name if isinstance(item.name, str) else item.name.decode("latin-1")
                if re.fullmatch(r"g\d+", name):
                    char = glyph_name_to_char(name)
                    self._gid_fix[code] = char if char is not None else chr(
                        PRIVATE_USE_BASE + min(int(name[1:]), 0x18FF)
                    )
                code += 1

    def patched_simple(self, cid):  # type: ignore[no-untyped-def]
        fix = getattr(self, "_gid_fix", None)
        if fix and cid in fix:
            return fix[cid]
        return simple_to_unichr(self, cid)

    def patched_cid(self, cid):  # type: ignore[no-untyped-def]
        try:
            return cid_to_unichr(self, cid)
        except Exception:  # noqa: BLE001 - PDFUnicodeNotDefined and friends
            if self.cidcoding in ("Adobe-Identity", "Adobe-UCS") and not self.unicode_map:
                char = gid_to_char(cid)
                return char if char is not None else chr(PRIVATE_USE_BASE + min(cid, 0x18FF))
            raise

    PDFSimpleFont.__init__ = patched_init  # type: ignore[method-assign]
    PDFSimpleFont.to_unichr = patched_simple  # type: ignore[method-assign]
    PDFCIDFont.to_unichr = patched_cid  # type: ignore[method-assign]
    PDFSimpleFont._nabard_glyph_fix = True  # type: ignore[attr-defined]


_install_pdfminer_glyph_fix()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class PageText:
    """Text of one printed page (half of a spread, or a whole portrait page)."""

    source: str  # file name
    pdf_page: int  # 1-based physical PDF page
    side: str  # "full" | "left" | "right"
    printed_page: str | None  # page label printed on the page, if found
    text: str
    repaired: bool = False  # True if glyph repair was applied
    stats: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Font diagnostics
# --------------------------------------------------------------------------- #
def _page_has_broken_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> bool:
    for xref, _ext, ftype, _base, _name, _enc, *_ in page.get_fonts(full=True):
        tu_kind, _ = doc.xref_get_key(xref, "ToUnicode")
        if ftype == "Type0" and tu_kind == "null":
            return True
        enc_kind, enc_val = doc.xref_get_key(xref, "Encoding")
        if enc_kind == "xref":
            obj = doc.xref_object(int(enc_val.split()[0])).encode("latin-1", "ignore")
            if _GLYPH_ID_NAME.search(obj):
                return True
    return False


# --------------------------------------------------------------------------- #
# pdfminer character positions
# --------------------------------------------------------------------------- #
@dataclass
class _CharIndex:
    xs: list[float]
    items: list[tuple[float, float, str]]  # (x_center, y_center, char) sorted by x

    @classmethod
    def build(cls, chars: list[tuple[float, float, str]]) -> _CharIndex:
        chars.sort(key=lambda t: t[0])
        return cls([c[0] for c in chars], chars)

    def lookup(self, x: float, y: float, size: float) -> str | None:
        tol_x, tol_y = max(0.8, size * 0.15), max(2.0, size * 0.7)
        i = bisect_left(self.xs, x - tol_x)
        best, best_d = None, 1e9
        while i < len(self.items) and self.items[i][0] <= x + tol_x:
            cx, cy, ch = self.items[i]
            if abs(cy - y) <= tol_y:
                d = abs(cx - x) + 0.25 * abs(cy - y)
                if d < best_d:
                    best, best_d = ch, d
            i += 1
        return best


def _iter_ltchars(obj) -> Iterator[LTChar]:  # type: ignore[no-untyped-def]
    for child in obj:
        if isinstance(child, LTChar):
            yield child
        elif isinstance(child, LTContainer):
            yield from _iter_ltchars(child)


def _pdfminer_pages(path: Path) -> Iterator[list[tuple[float, float, str]]]:
    """Yield, per page, character centres in PDF user space with decoded text."""
    with path.open("rb") as fh:
        parser = PDFParser(fh)
        document = PDFDocument(parser)
        rsrc = PDFResourceManager(caching=True)
        device = PDFPageAggregator(rsrc, laparams=None)
        interpreter = PDFPageInterpreter(rsrc, device)
        for page in PDFPage.create_pages(document):
            try:
                interpreter.process_page(page)
                layout = device.get_result()
                yield [
                    ((c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2, c.get_text())
                    for c in _iter_ltchars(layout)
                ]
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill ingestion
                log.warning("pdfminer failed on a page of %s: %s", path.name, exc)
                yield []


# --------------------------------------------------------------------------- #
# Layout reconstruction
# --------------------------------------------------------------------------- #
@dataclass
class _Geometry:
    """Maps PyMuPDF (unrotated, top-left origin) coordinates to the visual page
    and to pdfminer's space (rotation applied, bottom-left origin)."""

    rotation: pymupdf.Matrix
    height: float  # visual page height
    # rotation plus a turn that makes sideways-printed text (a landscape table on a
    # portrait page) read left to right; used for layout only, not for pdfminer lookups
    view: pymupdf.Matrix | None = None

    def visual_rect(self, bbox: tuple[float, float, float, float]) -> pymupdf.Rect:
        return pymupdf.Rect(bbox) * (self.view or self.rotation)

    def miner_point(self, x: float, y: float) -> tuple[float, float]:
        p = pymupdf.Point(x, y) * self.rotation
        return p.x, self.height - p.y


def _block_text(block: dict, index: _CharIndex | None, geo: _Geometry) -> str:
    """Text of one block. Lines sharing a baseline (table cells) are joined
    with `` | `` so a table row stays on one line."""
    if "table_text" in block:
        return block["table_text"]
    rows: list[tuple[pymupdf.Rect, list[tuple[float, str]]]] = []
    for line in block.get("lines", []):
        text = _line_text(line, index, geo)
        if not text.strip():
            continue
        r = geo.visual_rect(line["bbox"])
        if rows:
            prev = rows[-1][0]
            overlap = min(prev.y1, r.y1) - max(prev.y0, r.y0)
            if overlap > 0.5 * min(prev.height, r.height) and r.x0 >= prev.x1 - 1:
                rows[-1][1].append((r.x0, text.strip()))
                rows[-1] = (prev | r, rows[-1][1])
                continue
        rows.append((r, [(r.x0, text)]))
    return "\n".join(" | ".join(t for _, t in sorted(cells)) for _, cells in rows)


def _line_text(line: dict, index: _CharIndex | None, geo: _Geometry) -> str:
    """Characters of one line, swapped for the pdfminer-decoded character at
    the same position when the page uses a broken font."""
    parts: list[str] = []
    for span in line.get("spans", []):
        size = span.get("size", 10.0)
        for ch in span.get("chars", []):
            c = ch["c"]
            if index is not None and not c.isspace():
                x0, y0, x1, y1 = ch["bbox"]
                fixed = index.lookup(*geo.miner_point((x0 + x1) / 2, (y0 + y1) / 2), size)
                if fixed is not None:
                    c = fixed
            parts.append(c)
    return "".join(parts).rstrip()


def _row_sorted(blocks: list[dict], tol: float = 2.0) -> list[dict]:
    """Top-to-bottom, then left-to-right, treating blocks whose tops lie within
    ``tol`` points of each other as one row.

    Table cells set in different fonts start a fraction of a point apart. A
    plain (y0, x0) sort then puts a row's numbers before its label, and every
    value shifts up a row (NABARD 2017-18, Table 4.1)."""
    out: list[dict] = []
    row: list[dict] = []
    for b in sorted(blocks, key=lambda b: b["vbbox"].y0):
        if row and b["vbbox"].y0 - row[0]["vbbox"].y0 > tol:
            out.extend(sorted(row, key=lambda b: b["vbbox"].x0))
            row = []
        row.append(b)
    out.extend(sorted(row, key=lambda b: b["vbbox"].x0))
    return out


def _reading_order(blocks: list[dict], x_lo: float, x_hi: float) -> list[dict]:
    """Two-column aware ordering inside one printed page.

    Full-width blocks act as separators; between separators, the left column
    is read top-to-bottom before the right column.
    """
    mid = (x_lo + x_hi) / 2
    slack = (x_hi - x_lo) * 0.04
    ordered: list[dict] = []
    segment: list[dict] = []

    def flush() -> None:
        left = [b for b in segment if b["vbbox"].x1 <= mid + slack]
        right = [b for b in segment if b["vbbox"].x1 > mid + slack]
        ordered.extend(_row_sorted(left))
        ordered.extend(_row_sorted(right))
        segment.clear()

    for b in _row_sorted(blocks):
        x0, x1 = b["vbbox"].x0, b["vbbox"].x1
        spans_both = x0 < mid - slack and x1 > mid + slack
        if spans_both:
            flush()
            ordered.append(b)
        else:
            segment.append(b)
    flush()
    return ordered


def _upright_turn(blocks: list[dict], rotation: pymupdf.Matrix, width: float, height: float) -> pymupdf.Matrix | None:
    """Extra turn (in visual space) that makes the page's dominant text direction left-to-right,
    or None when it already is. Pages whose /Rotate is set are handled by ``rotation``; this
    catches content drawn sideways on an unrotated page (Tamil Nadu MSME Policy, Annexure I)."""
    votes: dict[tuple[int, int], int] = {}
    for b in blocks:
        for line in b.get("lines", []):
            dx, dy = line["dir"]
            v = pymupdf.Point(dx, dy) * rotation - pymupdf.Point(0, 0) * rotation
            key = (round(v.x), round(v.y))
            votes[key] = votes.get(key, 0) + sum(len(s.get("chars", ())) for s in line.get("spans", []))
    if not votes:
        return None
    total, (best, n) = sum(votes.values()), max(votes.items(), key=lambda kv: kv[1])
    if best == (1, 0) or n < 0.6 * total:
        return None
    return {  # (x, y) -> new upright coordinates
        (0, -1): pymupdf.Matrix(0, 1, -1, 0, height, 0),  # reads bottom-to-top: turn clockwise
        (0, 1): pymupdf.Matrix(0, -1, 1, 0, 0, width),  # reads top-to-bottom: turn anticlockwise
        (-1, 0): pymupdf.Matrix(-1, 0, 0, -1, width, height),  # upside down
    }.get(best)


def _ruled_tables(page: pymupdf.Page, blocks: list[dict], index: _CharIndex | None,
                  geo: _Geometry) -> tuple[list[dict], list[dict]]:
    """Bordered tables as pseudo-blocks with one ``cell | cell | ...`` line per row.

    Cells are found from the ruling lines; each cell's text is gathered from the page's own
    lines (so broken-font repair still applies) and rows/columns are read in upright space.
    Returns (table blocks, the remaining blocks outside any table)."""
    try:
        found = page.find_tables().tables
    except Exception as e:  # noqa: BLE001 - table detection is best effort
        log.debug("find_tables failed on page %d: %s", page.number + 1, e)
        return [], blocks
    lines = [(pymupdf.Rect(ln["bbox"]), ln) for b in blocks for ln in b.get("lines", [])]
    out, used = [], set()
    for tab in found:
        cells = [pymupdf.Rect(c) for c in tab.cells if c]
        if len(cells) < 4:
            continue
        texts = []
        for cell in cells:
            inside = [(geo.visual_rect(r), ln) for r, ln in lines
                      if cell.contains(pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))]
            inside.sort(key=lambda t: (round(t[0].y0), t[0].x0))
            texts.append(" ".join(" ".join(_line_text(ln, index, geo).split()) for _, ln in inside))
        vcells = sorted(zip((geo.visual_rect(c) for c in cells), texts, strict=True), key=lambda t: (t[0].y0, t[0].x0))
        rows: list[list[tuple[pymupdf.Rect, str]]] = []
        for vc, txt in vcells:
            if rows and abs(vc.y0 - rows[-1][0][0].y0) <= 3:
                rows[-1].append((vc, txt))
            else:
                rows.append([(vc, txt)])
        text = "\n".join(" | ".join(t for _, t in sorted(r, key=lambda c: c[0].x0)) for r in rows
                         if any(t for _, t in r))
        out.append({"vbbox": geo.visual_rect(tab.bbox), "table_text": text})
        area = pymupdf.Rect(tab.bbox)
        used |= {id(b) for b in blocks if area.contains(pymupdf.Point(
            (b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2))}
    return out, [b for b in blocks if id(b) not in used]


def _boilerplate_key(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


@dataclass
class _Half:
    pdf_page: int
    side: str
    printed_page: str | None
    repaired: bool
    blocks: list[tuple[str, bool]]  # (text, in_margin) in reading order


def extract_pdf(
    path: str | Path,
    margin_top: float = 0.13,
    margin_bottom: float = 0.10,
    boilerplate_min_pages: int = 3,
) -> list[PageText]:
    """Extract one :class:`PageText` per printed page.

    Running headers/footers are removed by frequency: a short block in the top
    or bottom margin whose letters repeat on ``boilerplate_min_pages`` or more
    pages ("NABARD ANNUAL REPORT 2017-18", chapter names) is dropped. Margin
    blocks that are unique to a page are kept, since they may be content.
    """
    path = Path(path)
    halves_out: list[_Half] = []
    for half in _iter_halves(path, margin_top, margin_bottom):
        halves_out.append(half)

    counts: dict[str, set[int]] = {}
    for h in halves_out:
        for text, in_margin in h.blocks:
            key = _boilerplate_key(text)
            if in_margin and key and len(text) <= 90:
                counts.setdefault(key, set()).add(h.pdf_page)
    boilerplate = {k for k, pages in counts.items() if len(pages) >= boilerplate_min_pages}

    results = []
    for h in halves_out:
        kept = [t for t, in_margin in h.blocks if not (in_margin and _boilerplate_key(t) in boilerplate)]
        results.append(
            PageText(
                source=path.name,
                pdf_page=h.pdf_page,
                side=h.side,
                printed_page=h.printed_page,
                text="\n\n".join(t for t in kept if t.strip()),
                repaired=h.repaired,
            )
        )
    log.info("extracted %s: %d printed pages, %d boilerplate patterns removed", path.name, len(results), len(boilerplate))
    return results


def _iter_halves(path: Path, margin_top: float, margin_bottom: float) -> Iterator[_Half]:
    doc = pymupdf.open(path)
    miner = _pdfminer_pages(path)
    for pno in range(doc.page_count):
        page = doc[pno]
        miner_chars = next(miner, [])
        broken = _page_has_broken_fonts(doc, page)
        index = _CharIndex.build(miner_chars) if broken and miner_chars else None
        width, height = page.rect.width, page.rect.height  # visual (rotated) size
        geo = _Geometry(page.rotation_matrix, height)

        raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
        blocks = [b for b in raw["blocks"] if b.get("type") == 0]
        turn = _upright_turn(blocks, page.rotation_matrix, width, height)
        if turn is not None:
            geo.view = page.rotation_matrix * turn
            upright = pymupdf.Rect(0, 0, width, height) * turn
            width, height = upright.width, upright.height
        for b in blocks:
            b["vbbox"] = geo.visual_rect(b["bbox"])
        if turn is not None:
            # sideways pages are almost always landscape tables, which line-based ordering scrambles
            tables, blocks = _ruled_tables(page, blocks, index, geo)
            blocks += tables
        # a sideways table is one landscape page, not a two-page spread
        spread = width > height * 1.2 and turn is None
        halves = [("left", 0.0, width / 2), ("right", width / 2, width)] if spread else [("full", 0.0, width)]

        for side, x_lo, x_hi in halves:
            mine = [b for b in blocks if x_lo <= (b["vbbox"].x0 + b["vbbox"].x1) / 2 < x_hi]
            # Printed page number: the lowest short numeric block near the bottom.
            printed_page: str | None = None
            label_block = None
            for b in sorted(mine, key=lambda b: -b["vbbox"].y1):
                if b["vbbox"].y0 < height * 0.85:
                    break
                label = _block_text(b, index, geo).strip()
                if _PAGE_LABEL.match(label):
                    printed_page, label_block = label, b
                    break
            body = [b for b in mine if b is not label_block]
            yield _Half(
                pdf_page=pno + 1,
                side=side,
                printed_page=printed_page,
                repaired=index is not None,
                blocks=[
                    (
                        _block_text(b, index, geo),
                        b["vbbox"].y1 <= height * margin_top or b["vbbox"].y0 >= height * (1 - margin_bottom),
                    )
                    for b in _reading_order(body, x_lo, x_hi)
                ],
            )
    doc.close()
