"""Structure-aware chunking.

* Numbered headings (``1.2.5 Production of Major Agricultural Crops``) and
  all-caps headings start a new section; chunks never straddle sections, and
  the section path is stored on each chunk (and prepended for embedding).
* Captions (``Table 1.2 ...``, ``Box 3.1 ...``, ``Showcase 2.1 ...``) label the
  content that follows.
* Tables are chunked row-wise; each piece repeats the caption and header row so
  numbers are never separated from what they measure.
* Narrative text is packed paragraph-by-paragraph to a target size, with a
  sentence-aligned overlap between consecutive chunks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..schemas import Chunk
from .cleaning import clean_page_text
from .pdf_extract import PageText

_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})(\.)?\s*\|?\s*([A-Z][^|]{2,110})$")
_PAGE_REF = re.compile(r"^(\d{1,3}|[xivlc]{1,6})$", re.I)
_SECTION_NO = re.compile(r"^\d{1,2}(\.\d{1,2})*\.?$")
_NUMERIC_CELL = re.compile(r"^[\d.,%₹()\s\-–]+$")


def _is_toc_row(row: str) -> bool:
    """'2.4 | FPO: Size Matters | 24' - a title and a page reference and no data.
    Data rows ('2 | Irrigation | 18,792 | 67 | 10,642 | 39') have several numbers."""
    cells = [c.strip() for c in row.split("|") if c.strip()]
    if len(cells) < 2 or not _PAGE_REF.match(cells[-1]):
        return False
    middle = cells[1:-1] if _SECTION_NO.match(cells[0]) else cells[:-1]
    return bool(middle) and not any(_NUMERIC_CELL.match(c) for c in middle) and any(re.search(r"[A-Za-z]{3,}", c) for c in middle)


def _is_toc(rows: list[str]) -> bool:
    """Most rows look like 'title | page' AND the page numbers run upwards -
    the second condition keeps two-column data tables ('Assam | 56')."""
    toc = [r for r in rows if _is_toc_row(r)]
    if len(rows) < 4 or len(toc) < 0.6 * len(rows):
        return False
    pages = [int(p) for r in toc if (p := r.rsplit("|", 1)[-1].strip()).isdigit()]
    if len(pages) < 3:
        return True  # roman-numbered front matter
    rising = sum(b >= a for a, b in zip(pages, pages[1:]))
    return rising >= 0.8 * (len(pages) - 1)


def _heading(par: str) -> tuple[int, str] | None:
    """(depth, "1.2.5 Title") for numbered section headings.

    Depth-1 numbers need a trailing dot ("8. Financial Performance") so that
    footnotes ("6 Budgetary allocation ...") and table serial numbers are not
    mistaken for headings; headings never end in a full stop."""
    if "\n" in par or len(par) >= 130:
        return None
    m = _NUMBERED.match(par)
    if not m:
        return None
    number, dot, title = m.group(1), m.group(2), m.group(3).strip()
    depth = number.count(".") + 1
    if title.endswith(".") or (depth == 1 and not dot) or len(title.split()) > 14:
        return None
    return depth, f"{number} {title}"
_CAPTION = re.compile(r"^(Table|Box|Showcase|Figure|Schedule|Sidebar|Annexure|Chart)\s+[\dA-Z][\d.A-Z]*\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(₹\d])")


def _words(text: str) -> int:
    return len(text.split())


def _is_caps_heading(par: str) -> bool:
    letters = [c for c in par if c.isalpha()]
    return (
        "\n" not in par
        and 6 <= len(par) <= 80
        and len(letters) >= 5
        and sum(c.isupper() for c in letters) / len(letters) > 0.9
        and not par.endswith((".", ","))
    )


def _is_table(par: str) -> bool:
    lines = par.split("\n")
    return sum("|" in ln for ln in lines) >= max(2, len(lines) // 2) or (len(lines) == 1 and par.count("|") >= 2)


@dataclass
class _Para:
    text: str
    pdf_page: int
    printed_page: str | None


@dataclass
class _Builder:
    source: str
    report: str
    year: str | None
    target: int
    max_words: int
    overlap: int
    chunks: list[Chunk] = field(default_factory=list)

    def emit(self, paras: list[_Para], section: str | None, kind: str) -> None:
        text = "\n\n".join(p.text for p in paras).strip()
        if _words(text) < 8:
            return
        pdf_pages = sorted({p.pdf_page for p in paras})
        printed: list[str] = []
        for p in paras:
            if p.printed_page and p.printed_page not in printed:
                printed.append(p.printed_page)
        digest = hashlib.sha1(f"{self.source}|{section}|{text}".encode()).hexdigest()[:20]
        self.chunks.append(
            Chunk(
                id=digest,
                text=text,
                source=self.source,
                report=self.report,
                year=self.year,
                section=section,
                pdf_pages=pdf_pages,
                printed_pages=printed,
                kind=kind,  # type: ignore[arg-type]
            )
        )


def _split_long(par: _Para, max_words: int) -> list[_Para]:
    if _words(par.text) <= max_words:
        return [par]
    out, cur = [], ""
    for sent in _SENTENCE.split(par.text):
        if cur and _words(cur) + _words(sent) > max_words:
            out.append(_Para(cur, par.pdf_page, par.printed_page))
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(_Para(cur, par.pdf_page, par.printed_page))
    return out


def _tail(paras: list[_Para], overlap_words: int) -> list[_Para]:
    """Trailing sentences (≈ overlap_words) of the last paragraph, for overlap."""
    if not paras or overlap_words <= 0:
        return []
    last = paras[-1]
    sents = _SENTENCE.split(last.text)
    kept: list[str] = []
    while sents and _words(" ".join(kept)) < overlap_words:
        kept.insert(0, sents.pop())
    if not sents:  # whole paragraph would repeat; no useful overlap
        return []
    return [_Para(" ".join(kept), last.pdf_page, last.printed_page)]


def _rows_to_paras(table: list[_Para], rows: list[str], piece: list[str], start: int) -> list[_Para]:
    """Wrap a table piece as one paragraph, attributing it to every page its rows came from."""
    owners: list[_Para] = []
    idx = 0
    for p in table:
        n = len(p.text.split("\n"))
        if idx + n > start and idx <= start + len(piece):
            owners.append(p)
        idx += n
    first = owners[0] if owners else table[0]
    paras = [_Para("\n".join(piece), first.pdf_page, first.printed_page)]
    paras += [_Para("", p.pdf_page, p.printed_page) for p in owners[1:]]  # page attribution only
    return paras


def chunk_document(pages: list[PageText], report: str, year: str | None,
                   target_words: int = 220, max_words: int = 360, overlap_words: int = 40) -> list[Chunk]:
    if not pages:
        return []
    b = _Builder(pages[0].source, report, year, target_words, max_words, overlap_words)
    path: dict[int, str] = {}  # depth -> heading, for "1.2 X > 1.2.5 Y"
    caption: str | None = None
    buf: list[_Para] = []

    def section() -> str | None:
        parts = [path[d] for d in sorted(path)]
        if caption:
            parts.append(caption)
        return " > ".join(parts[-3:]) if parts else None

    table: list[_Para] = []  # consecutive table rows (each row is often its own block)

    def flush_table() -> None:
        nonlocal table
        if not table:
            return
        rows = [ln for p in table for ln in p.text.split("\n")]
        if _is_toc(rows):
            table = []  # table of contents / list of tables: pure noise for retrieval
            return
        header = rows[0]
        piece: list[str] = [header]
        start = 0
        for i, row in enumerate(rows[1:], start=1):
            if _words("\n".join(piece + [row])) > max_words and len(piece) > 1:
                b.emit(_rows_to_paras(table, rows, piece, start), section(), "table")
                piece, start = [header], i
            piece.append(row)
        b.emit(_rows_to_paras(table, rows, piece, start), section(), "table")
        table = []

    def flush() -> None:
        nonlocal buf
        flush_table()
        if buf:
            b.emit(buf, section(), "text")
        buf = []

    for page in pages:
        for raw in clean_page_text(page.text):
            par = _Para(raw, page.pdf_page, page.printed_page)
            if _is_table(raw):
                if buf:  # narrative before the table closes as its own chunk
                    b.emit(buf, section(), "text")
                    buf = []
                table.append(par)
                continue
            if (table and len(raw) < 60 and "\n" not in raw and any(ch.isdigit() for ch in raw)
                    and not _CAPTION.match(raw) and not _NUMBERED.match(raw)):
                table.append(par)  # a lone cell line inside a table ("Total 4,004")
                continue
            flush_table()
            first_line = raw.split("\n", 1)[0]
            heading = _heading(raw.replace(" | ", " "))
            if heading:
                flush()
                depth, title = heading
                for d in [d for d in path if d >= depth]:
                    del path[d]
                path[depth] = title
                caption = None
                continue
            if _is_caps_heading(raw):
                flush()
                path = {1: raw.replace(" | ", " ").title()}
                caption = None
                continue
            if _CAPTION.match(first_line) and len(raw) < 220:
                flush()
                caption = raw.replace("\n", " ").replace(" | ", " ")
                continue
            for part in _split_long(par, max_words):
                if buf and sum(_words(p.text) for p in buf) + _words(part.text) > target_words:
                    prev = buf
                    flush()
                    buf = _tail(prev, overlap_words)
                buf.append(part)
    flush()
    return _merge_small(b.chunks, min_words=60, max_words=max_words)


def _merge_small(chunks: list[Chunk], min_words: int, max_words: int) -> list[Chunk]:
    """Fold fragments (a two-line table, a one-paragraph subsection) into the
    following chunk of the same section (or parent section), so the index is
    not flooded with context-free scraps."""

    def family(c: Chunk) -> str:
        return (c.section or "").split(" > ")[0]

    out: list[Chunk] = []
    pending: Chunk | None = None
    for c in chunks:
        if pending is not None:
            same = pending.source == c.source and family(pending) == family(c)
            if same and _words(pending.text) + _words(c.text) <= max_words:
                c = _combine(pending, c)
            else:
                out.append(pending)
            pending = None
        if _words(c.text) < min_words:
            pending = c
        else:
            out.append(c)
    if pending is not None:
        out.append(pending)
    return out


def _combine(a: Chunk, b: Chunk) -> Chunk:
    printed = a.printed_pages + [p for p in b.printed_pages if p not in a.printed_pages]
    section = b.section if (b.section or "").startswith(a.section or "\x00") else (a.section or b.section)
    # b's heading would otherwise vanish: keep it inline so "2.2 Quantum of Capital Investment
    # Subsidy" still introduces its rate table after merging under a's section label
    b_head = (b.section or "").split(" > ")[-1] if b.section and b.section != section else ""
    text = f"{a.text}\n\n{b_head}\n{b.text}" if b_head else f"{a.text}\n\n{b.text}"
    digest = hashlib.sha1(f"{a.source}|{section}|{text}".encode()).hexdigest()[:20]
    return a.model_copy(update={
        "id": digest,
        "text": text,
        "section": section,
        "pdf_pages": sorted(set(a.pdf_pages) | set(b.pdf_pages)),
        "printed_pages": printed,
        "kind": "table" if "table" in (a.kind, b.kind) else "text",
    })
