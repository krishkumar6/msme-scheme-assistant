"""Text normalisation for extracted pages."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_PUA = re.compile(r"[-]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t   ]+")
_YEAR_SPAN = re.compile(r"\b(20\d\d)\s*[–-]\s*(\d\d)\b")
_WORD = re.compile(r"[A-Za-z]{3,}")
_LINE_SEP = re.compile(r"[  \x85]")
# Pre-2010 "Rupee Foradian"-style fonts draw ₹ on the backtick key, so the text
# layer says "` 15,000" (Ministry of MSME annual reports).
_RUPEE_BACKTICK = re.compile(r"`[ \t ]*\n?[ \t ]*(?=\d)")  # the amount may wrap to the next line


def normalise(text: str) -> str:
    """Unicode NFKC (splits ligatures like 'ﬁ'), strip private-use / control
    characters left by undecodable glyphs, restore ₹ typed as a backtick,
    collapse horizontal whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_SEP.sub("\n", text)
    text = _RUPEE_BACKTICK.sub("₹", text)
    text = _PUA.sub("", text)
    text = _CTRL.sub("", text)
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines)


def _is_noise(par: str) -> bool:
    """Chart axis ticks, legend fragments and stray numbers."""
    if "|" in par:
        return False
    words = _WORD.findall(par)
    if not words and len(par) <= 40:
        return True  # "10", "8.2", "2013–14", "(PE)"
    return len(par) <= 12 and not any(len(w) >= 4 for w in words)


def _join(prev: str, line: str) -> str:
    if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and line[:1].islower():
        return prev[:-1] + line  # de-hyphenate "agri-\nculture"
    if prev.endswith(("–", "/", "(")):
        return prev + line  # "2017–\n18"
    return f"{prev} {line}"


def unwrap_paragraph(par: str) -> str:
    """Join wrapped narrative lines; keep table rows (``a | b | c``) on their own lines."""
    out: list[str] = []
    for line in par.split("\n"):
        if not line:
            continue
        if out and "|" not in line and "|" not in out[-1]:
            out[-1] = _join(out[-1], line)
        else:
            out.append(line)
    return "\n".join(out)


def clean_page_text(text: str) -> list[str]:
    """Return the cleaned paragraphs of one page."""
    paragraphs = []
    for par in normalise(text).split("\n\n"):
        par = par.strip()
        if not par or _is_noise(par):
            continue
        paragraphs.append(unwrap_paragraph(par))
    return paragraphs


def detect_report_year(texts: list[str]) -> str | None:
    """Most frequent consecutive financial-year span (e.g. ``2017-18``)."""
    counts: Counter[str] = Counter()
    for t in texts:
        for a, b in _YEAR_SPAN.findall(t):
            if int(b) == (int(a) + 1) % 100:
                counts[f"{a}-{b}"] += 1
    return counts.most_common(1)[0][0] if counts else None
