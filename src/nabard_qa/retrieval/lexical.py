"""BM25 keyword retrieval.

Financial reports are full of exact terms (scheme acronyms like ST-SAO, LTIF,
RIDF; figures like 1,59,822) that dense embeddings blur, so BM25 runs alongside
the vector search and the two are fused.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from ..schemas import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who whom how when where why does do did can could should would about tell me please give "
    "list show under during per".split()
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        if any(ch.isdigit() for ch in tok):
            out.append(tok)
            plain = tok.replace(",", "")
            if plain != tok:
                out.append(plain)  # "1,59,822" also matches "159822"
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural folding: banks -> bank
        out.append(tok)
    return out


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.ids = [c.id for c in chunks]
        self.years = [c.year for c in chunks]
        corpus = [tokenize(c.embed_text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int, year: str | None = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        if year:
            scores = np.where(np.array(self.years) == year, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
