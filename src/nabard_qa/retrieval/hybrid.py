"""Hybrid retrieval pipeline.

query ─┬─ embed ─► query adapter ─► Chroma top-k ─┐
       └─ BM25 top-k ─────────────────────────────┴─► reciprocal-rank fusion
           ─► + passages rated relevant on similar past questions (feedback recall)
           ─► cross-encoder rerank ─► + feedback boost (similar past questions)
           ─► + soft preference for a report year named in the question ─► top-k
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from ..config import Settings
from ..feedback.store import FeedbackStore
from ..schemas import Chunk, ScoredChunk
from .adapter import QueryAdapter
from .embeddings import Embedder
from .lexical import BM25Index
from .reranker import CrossEncoderReranker
from .vector_store import ChromaStore

log = logging.getLogger(__name__)
_YEAR_IN_QUERY = re.compile(r"\b(20\d\d)\s*[-–/]\s*(20)?(\d\d)\b")


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    query_embedding: np.ndarray  # raw (un-adapted) embedding, used for feedback similarity


def years_in_query(query: str) -> set[str]:
    return {f"{a}-{c}" for a, _, c in _YEAR_IN_QUERY.findall(query)}


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: ChromaStore,
        lexical: BM25Index,
        reranker: CrossEncoderReranker | None,
        adapter: QueryAdapter,
        feedback: FeedbackStore | None,
    ) -> None:
        self.s = settings
        self.embedder = embedder
        self.store = store
        self.lexical = lexical
        self.reranker = reranker
        self.adapter = adapter
        self.feedback = feedback
        self.known_years = {y for y in lexical.years if y}

    def retrieve(self, query: str, year: str | None = None, k: int | None = None,
                 candidates: int | None = None) -> RetrievalResult:
        k = k or self.s.final_k
        n_candidates = max(candidates or self.s.rerank_candidates, k)
        q = self.embedder.embed_query(query)
        q_search = self.adapter.apply(q) if self.s.adapter_enabled else q

        dense = self.store.query(q_search, max(self.s.dense_k, n_candidates), year=year)
        lexical = self.lexical.search(query, max(self.s.lexical_k, n_candidates), year=year)

        # reciprocal-rank fusion
        fused: dict[str, float] = {}
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense, start=1)}
        lex_rank = {cid: r for r, (cid, _) in enumerate(lexical, start=1)}
        for cid, r in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + self.s.dense_weight / (self.s.rrf_k + r)
        for cid, r in lex_rank.items():
            fused[cid] = fused.get(cid, 0.0) + self.s.lexical_weight / (self.s.rrf_k + r)
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:n_candidates]  # type: ignore[arg-type]

        # Feedback recall: passages users marked relevant for *similar* past questions join
        # the candidates even when this phrasing didn't retrieve them. The cross-encoder
        # still scores them, so the relevance gate decides - feedback can't force junk in.
        boosts = self.feedback.boosts(q, self.s.feedback_similarity) if self.feedback else {}
        recalled = [cid for cid, v in sorted(boosts.items(), key=lambda kv: -kv[1])
                    if v > 0 and cid not in fused][: self.s.feedback_recall_k]
        chunks = self.store.get(candidate_ids + recalled)
        if year:
            recalled = [cid for cid in recalled if cid in chunks and chunks[cid].year == year]
        candidates: list[Chunk] = [chunks[c] for c in candidate_ids + recalled if c in chunks]
        if not candidates:
            return RetrievalResult([], q)

        if self.reranker is not None:
            base = self.reranker.score(query, [c.embed_text for c in candidates])
        else:
            # scale RRF into a comparable range (top candidate ~ 0)
            top = max(fused.values())
            # recalled-from-feedback passages have no fusion score: rank them as the weakest
            floor = min(fused.values())
            base = [10.0 * (fused.get(c.id, floor) / top - 1.0) for c in candidates]

        wanted_years = years_in_query(query) & self.known_years if not year else set()

        scored: list[ScoredChunk] = []
        for c, b in zip(candidates, base, strict=True):
            fb = self.s.feedback_boost * boosts.get(c.id, 0.0)
            yr = 1.0 if wanted_years and c.year in wanted_years else 0.0
            scored.append(
                ScoredChunk(
                    chunk=c,
                    score=b + fb + yr,
                    dense_rank=dense_rank.get(c.id),
                    lexical_rank=lex_rank.get(c.id),
                    rerank_score=b if self.reranker is not None else None,
                    feedback_boost=fb,
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return RetrievalResult(_dedupe(scored)[:k], q)


def _dedupe(scored: list[ScoredChunk]) -> list[ScoredChunk]:
    """Drop near-identical passages (the overlap window can create them)."""
    seen: set[str] = set()
    out = []
    for s in scored:
        key = re.sub(r"\W+", "", s.chunk.text.lower())[:300]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
