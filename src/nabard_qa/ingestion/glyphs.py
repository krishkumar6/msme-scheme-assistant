"""Glyph-name based character recovery for PDFs with broken ToUnicode maps.

Many design-tool exports (this NABARD report included) subset fonts and name
glyphs by glyph id (``/g68``) instead of by character (``/a``). The ToUnicode
CMap they ship only covers a fraction of the codes, so text extractors fall back
to guesses and produce a substitution cipher ("rìê aüèîÜü" for "The Indian").

For TrueType-derived fonts, glyph ids 3..97 follow the standard Macintosh glyph
order, where ``gid = ord(char) - 29`` for printable ASCII. Glyph ids above that
are font specific; the common ones (ligatures, quotes, dashes) are listed in
``HIGH_GID_CHARS`` after checking them against the rendered report.
"""

from __future__ import annotations

import re

ASCII_GID_OFFSET = 29
ASCII_GID_RANGE = range(3, 98)  # space (3) .. asciitilde (97)

# Font-specific glyph ids observed in the NABARD reports (Georgia / Segoe Pro
# subsets). Verified by reading decoded words in context; unknown ids decode to
# nothing rather than to a wrong letter.
HIGH_GID_CHARS: dict[int, str] = {
    106: "à",  # a-grave   (vis-à-vis)
    135: "•",  # bullet
    171: "…",  # ellipsis
    180: "’",  # closing quote
    177: "–",  # en dash   (2017–18)
    178: "–",  # en dash   (Segoe subset)
    179: "—",  # em dash
    181: "‘",  # left single quote
    182: "’",  # right single quote / apostrophe
    183: "’",
    191: "fi",
    192: "fl",
    193: "fl",
    611: "ff",
    612: "ffi",
    613: "ffl",
    861: "₹",  # rupee sign
}
# Georgia small-caps alphabets (used for running headers and section labels).
for _base in (399, 643):
    for _i in range(26):
        HIGH_GID_CHARS[_base + _i] = chr(ord("a") + _i)

_GID_NAME = re.compile(r"g(\d+)")
_UNI_NAME = re.compile(r"uni([0-9A-Fa-f]{4})")


def gid_to_char(gid: int) -> str | None:
    if gid in ASCII_GID_RANGE:
        return chr(gid + ASCII_GID_OFFSET)
    return HIGH_GID_CHARS.get(gid)


def glyph_name_to_char(name: str) -> str | None:
    """Resolve a glyph-id style name (``g68``, ``uni20B9``). Returns None for
    names that should be left to the normal Adobe Glyph List lookup."""
    m = _GID_NAME.fullmatch(name)
    if m:
        return gid_to_char(int(m.group(1)))
    m = _UNI_NAME.fullmatch(name)
    if m:
        return chr(int(m.group(1), 16))
    return None
