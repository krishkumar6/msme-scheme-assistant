"""Offline evaluation against a golden question set.

Golden JSONL rows::

    {"id": "fg-01", "question": "...", "year": "2017-18" | null,
     "expected_facts": ["279.5"],          # strings that must appear (all of them)
     "answerable": true}

Retrieval: a hit means every expected fact appears in the top-k passages taken
together (from the right report year, when given) - that is the context the
LLM actually receives; the rank is where coverage becomes complete. Answers:
fraction of expected facts present in
the generated answer, whether it cites sources, and - for unanswerable
questions - whether the assistant declined instead of inventing an answer.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import AskRequest
from .service import QAService

_DECLINE = re.compile(
    r"(could not find|couldn't find|not (?:contain|cover|mention|include|available|found)|"
    r"no (?:information|relevant|details)|do(?:es)? not (?:provide|specify|state|say)|unable to)",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"[\s,]+", "", s.lower()).replace("₹", "").replace("rs.", "")


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_facts: list[str] = field(default_factory=list)
    year: str | None = None
    answerable: bool = True


def load_golden(path: Path) -> list[GoldenItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            items.append(GoldenItem(**json.loads(line)))
    return items


def evaluate_retrieval(service: QAService, items: list[GoldenItem], k: int = 6) -> dict:
    ranks: list[int | None] = []
    misses = []
    for it in (i for i in items if i.answerable and i.expected_facts):
        res = service.retriever.retrieve(it.question, k=k)
        rank = None
        missing = {_norm(f) for f in it.expected_facts}
        for r, sc in enumerate(res.chunks, start=1):
            if it.year and sc.chunk.year != it.year:
                continue
            text = _norm(sc.chunk.text)
            missing = {f for f in missing if f not in text}
            if not missing:
                rank = r
                break
        ranks.append(rank)
        if rank is None:
            misses.append(it.id)
    n = len(ranks) or 1
    return {
        "n": len(ranks),
        f"hit@{k}": round(sum(r is not None for r in ranks) / n, 3),
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "misses": misses,
    }


def evaluate_answers(service: QAService, items: list[GoldenItem]) -> dict:
    recalls, cited, declines_ok, rows = [], [], [], []
    for it in items:
        res = service.answer(AskRequest(question=it.question, year=None))
        ans = _norm(res.answer)
        if it.answerable:
            recall = (sum(_norm(f) in ans for f in it.expected_facts) / len(it.expected_facts)) if it.expected_facts else 1.0
            recalls.append(recall)
            cited.append(res.grounded)
            rows.append({"id": it.id, "fact_recall": recall, "grounded": res.grounded, "provider": res.provider})
        else:
            # Only an explicit decline counts. "Not grounded" is not enough: a model can
            # invent an answer (even with a fake [S1]) for passages the relevance gate rejected.
            ok = bool(_DECLINE.search(res.answer))
            declines_ok.append(ok)
            rows.append({"id": it.id, "declined": ok, "provider": res.provider})
    return {
        "provider": service.generator.name,
        "fell_back_to_extractive": sum(r["provider"] == "extractive" for r in rows),
        "answerable_n": len(recalls),
        "fact_recall": round(statistics.mean(recalls), 3) if recalls else None,
        "citation_rate": round(sum(cited) / len(cited), 3) if cited else None,
        "unanswerable_n": len(declines_ok),
        "decline_rate": round(sum(declines_ok) / len(declines_ok), 3) if declines_ok else None,
        "rows": rows,
    }
