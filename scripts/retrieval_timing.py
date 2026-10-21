"""Time each retrieval stage (embedding, dense, BM25, rerank) for one query."""

from __future__ import annotations

import sys
import time

from nabard_qa.config import get_settings
from nabard_qa.service import QAService


def main(q: str) -> None:
    svc = QAService.from_settings(get_settings())
    r = svc.retriever
    r.retrieve(q)  # warm-up (ONNX sessions, caches)
    for n in (30, 15):
        t = time.perf_counter(); qv = r.embedder.embed_query(q); t_emb = time.perf_counter() - t
        t = time.perf_counter(); dense = r.store.query(qv, n); t_dense = time.perf_counter() - t
        t = time.perf_counter(); r.lexical.search(q, n); t_bm25 = time.perf_counter() - t
        chunks = r.store.get([c for c, _ in dense])
        texts = [c.embed_text for c in chunks.values()]
        t = time.perf_counter(); r.reranker.score(q, texts); t_rr = time.perf_counter() - t
        avg_words = sum(len(x.split()) for x in texts) / len(texts)
        print(f"candidates={n}: embed {t_emb:.2f}s | dense {t_dense:.2f}s | bm25 {t_bm25:.2f}s | "
              f"rerank {t_rr:.2f}s ({avg_words:.0f} words/passage)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "What support does NABARD provide to Farmer Producer Organisations?")
