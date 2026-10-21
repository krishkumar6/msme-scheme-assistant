"""Turn stored feedback / synthetic pairs into a trained query adapter."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..config import Settings
from ..retrieval.adapter import AdapterReport, QueryAdapter, TrainingExample
from ..retrieval.embeddings import Embedder
from ..retrieval.vector_store import ChromaStore
from ..schemas import Chunk
from .store import FeedbackStore

log = logging.getLogger(__name__)
_NUMBER_PREFIX = re.compile(r"^(?:\d+(?:\.\d+)*\.?|(?:Table|Box|Showcase|Figure|Schedule|Sidebar)\s+[\d.A-Z]+:?)\s*", re.I)


def heading_pairs(chunks: list[Chunk]) -> list[tuple[str, str, list[str]]]:
    """Weak supervision without any LLM: the most specific heading of a chunk
    ("Status of WIF Loan Sanctions and Disbursements") is a pseudo-query whose
    positive is that chunk. Used to bootstrap the adapter before real feedback."""
    rows = []
    for c in chunks:
        if not c.section:
            continue
        title = _NUMBER_PREFIX.sub("", c.section.split(" > ")[-1]).strip(" :|-")
        title = re.sub(r"\(Amount in .*?\)", "", title).strip()
        if 3 <= len(title.split()) <= 20:
            rows.append((f"{title} ({c.year})" if c.year else title, c.id, []))
    return rows


def train_adapter(settings: Settings, embedder: Embedder, store: ChromaStore, feedback: FeedbackStore,
                  origins: tuple[str, ...] = ("feedback", "synthetic", "heading")) -> AdapterReport | None:
    pairs = feedback.training_pairs(origins)
    if len(pairs) < settings.adapter_min_examples:
        log.warning("only %d training pairs (< %d); not training", len(pairs), settings.adapter_min_examples)
        return None
    chunk_ids = sorted({p for _, p, _, _ in pairs} | {n for _, _, negs, _ in pairs for n in negs})
    chunk_vecs = store.get_embeddings(chunk_ids)
    pairs = [p for p in pairs if p[1] in chunk_vecs]
    queries = embedder.embed_queries([q for q, _, _, _ in pairs])
    d = queries.shape[1]
    examples = [
        TrainingExample(
            query=queries[i],
            positive=chunk_vecs[pos],
            negatives=np.array([chunk_vecs[n] for n in negs if n in chunk_vecs], dtype=np.float32).reshape(-1, d),
            positive_id=pos,
        )
        for i, (_, pos, negs, _) in enumerate(pairs)
    ]
    _, corpus = store.all()
    adapter, report = QueryAdapter.fit(examples, corpus)
    if report.accepted:
        adapter.save(settings.adapter_path, report)
        log.info("adapter accepted and saved to %s", settings.adapter_path)
    else:
        log.warning("adapter did not beat the base embeddings on held-out pairs; not saved")
    return report
