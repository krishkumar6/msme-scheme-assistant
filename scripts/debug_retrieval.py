"""Debug helper: show what retrieval returns for golden-set questions and where
the expected facts actually live in the index.

    python scripts/debug_retrieval.py fpo-01 shg-01
"""

from __future__ import annotations

import sys

from nabard_qa.config import PROJECT_ROOT, get_settings
from nabard_qa.evaluation import _norm, load_golden
from nabard_qa.service import QAService


def main(ids: list[str]) -> None:
    svc = QAService.from_settings(get_settings())
    items = {i.id: i for i in load_golden(PROJECT_ROOT / "eval" / "golden_set.jsonl")}
    chunks, _ = svc.store.all()
    for qid in ids:
        it = items[qid]
        print(f"\n=== {qid}: {it.question}  facts={it.expected_facts}")
        holders = [c for c in chunks if all(_norm(f) in _norm(c.text) for f in it.expected_facts)]
        partial = {f: [c.id for c in chunks if _norm(f) in _norm(c.text)][:5] for f in it.expected_facts}
        print("chunks containing all facts:", [(c.id, c.year, c.section) for c in holders[:4]])
        print("chunks per fact:", partial)
        res = svc.retriever.retrieve(it.question)
        for r, sc in enumerate(res.chunks, 1):
            print(f"  {r}. {sc.chunk.id} {sc.chunk.year} score={sc.score:.2f} dense={sc.dense_rank} bm25={sc.lexical_rank} | {sc.chunk.section}")
        for c in holders[:1]:
            print("  holder text:", c.text[:400].replace("\n", " / "))


if __name__ == "__main__":
    main(sys.argv[1:])
