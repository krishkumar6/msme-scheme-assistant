"""Print the top reranker score per golden question to calibrate
``NABARD_QA_MIN_RERANK_SCORE`` (the "is anything relevant?" gate)."""

from __future__ import annotations

from nabard_qa.config import PROJECT_ROOT, get_settings
from nabard_qa.evaluation import load_golden
from nabard_qa.service import QAService

EXTRA_UNANSWERABLE = [
    "What is the GST rate on cement?",
    "How do I apply for a passport?",
    "What is the repo rate set by RBI in 2024?",
    "Which MSME schemes does the Government of Tamil Nadu offer for textile units in 2023?",
    "What is the PM Mudra Yojana interest rate at Canara Bank today?",
]


def main() -> None:
    svc = QAService.from_settings(get_settings())
    rows = []
    for it in load_golden(PROJECT_ROOT / "eval" / "golden_set.jsonl"):
        res = svc.retriever.retrieve(it.question)
        rows.append((res.chunks[0].rerank_score if res.chunks else None, it.answerable, it.question))
    for q in EXTRA_UNANSWERABLE:
        res = svc.retriever.retrieve(q)
        rows.append((res.chunks[0].rerank_score if res.chunks else None, False, q))
    for score, ok, q in sorted(rows, key=lambda r: r[0] or -99):
        print(f"{score:7.2f}  {'ANS' if ok else 'N/A'}  {q[:90]}")


if __name__ == "__main__":
    main()
