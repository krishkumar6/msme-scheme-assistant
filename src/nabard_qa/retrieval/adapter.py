"""Feedback-trained query adapter.

Document embeddings stay frozen (no re-indexing); a linear map ``W`` is learned
on the *query* side so that ``W @ q`` lands closer to passages users confirmed
as relevant and further from passages they rejected.

* Loss: InfoNCE over {positive, explicit negatives, other examples' positives}.
* Regulariser: ``lambda * ||W - I||^2`` keeps the adapter near the identity, so
  a handful of feedback events cannot wreck retrieval.
* Safety gate: the adapter is only saved/activated if recall@k on held-out
  pairs is at least as good as the un-adapted embeddings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .embeddings import l2_normalise

log = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    query: np.ndarray  # (d,)
    positive: np.ndarray  # (d,)
    negatives: np.ndarray  # (n, d), may be empty
    positive_id: str = ""  # used to avoid treating a shared positive as an in-batch negative


@dataclass
class AdapterReport:
    n_train: int
    n_val: int
    recall_base: float
    recall_adapted: float
    mrr_base: float
    mrr_adapted: float
    accepted: bool
    k: int


class QueryAdapter:
    def __init__(self, weight: np.ndarray | None = None) -> None:
        self.weight = weight

    @property
    def active(self) -> bool:
        return self.weight is not None

    def apply(self, q: np.ndarray) -> np.ndarray:
        if self.weight is None:
            return q
        return l2_normalise(q @ self.weight.T)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path) -> QueryAdapter:
        if not path.exists():
            return cls(None)
        data = np.load(path)
        log.info("loaded query adapter %s", path)
        return cls(data["weight"].astype(np.float32))

    def save(self, path: Path, report: AdapterReport) -> None:
        assert self.weight is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self.weight)
        path.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2))

    # ------------------------------------------------------------- training
    @staticmethod
    def fit(
        examples: list[TrainingExample],
        corpus: np.ndarray,
        epochs: int = 60,
        lr: float = 0.05,
        reg: float = 0.5,
        temperature: float = 0.05,
        val_fraction: float = 0.2,
        k: int = 5,
        seed: int = 13,
    ) -> tuple[QueryAdapter, AdapterReport]:
        """Train on ``examples``; ``corpus`` is every chunk embedding (n, d),
        used for held-out evaluation exactly as retrieval would rank."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(examples))
        n_val = max(1, int(len(examples) * val_fraction)) if len(examples) >= 5 else 0
        val = [examples[i] for i in idx[:n_val]]
        train = [examples[i] for i in idx[n_val:]]
        d = corpus.shape[1]
        eye = np.eye(d, dtype=np.float32)
        w = eye.copy()
        velocity = np.zeros_like(w)

        q_all = np.stack([e.query for e in train]).astype(np.float32)
        pos_all = np.stack([e.positive for e in train]).astype(np.float32)
        m = max((len(e.negatives) for e in train), default=0)
        neg_all = np.zeros((len(train), max(m, 1), d), dtype=np.float32)
        mask_all = np.zeros((len(train), max(m, 1)), dtype=bool)
        for i, e in enumerate(train):
            if len(e.negatives):
                neg_all[i, : len(e.negatives)] = e.negatives
                mask_all[i, : len(e.negatives)] = True

        pos_ids = np.array([e.positive_id or str(i) for i, e in enumerate(train)])
        batch = 256
        for _ in range(epochs):
            order = rng.permutation(len(train))
            for start in range(0, len(train), batch):
                sel = order[start : start + batch]
                q, pos, neg, mask = q_all[sel], pos_all[sel], neg_all[sel], mask_all[sel]
                b = len(sel)
                wq = q @ w.T  # (b, d)
                # candidates per row: all in-batch positives (own one on the diagonal) + own negatives
                s_pos = wq @ pos.T / temperature  # (b, b)
                ids = pos_ids[sel]
                same = (ids[:, None] == ids[None, :]) & ~np.eye(b, dtype=bool)
                s_pos = np.where(same, -np.inf, s_pos)  # shared positive is not a negative
                s_neg = np.einsum("bd,bmd->bm", wq, neg) / temperature
                s_neg = np.where(mask, s_neg, -np.inf)
                logits = np.concatenate([s_pos, s_neg], axis=1)
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(axis=1, keepdims=True)
                # dL/d(Wq) = (E_p[c] - c_pos) / T ; dL/dW = sum_i outer(dL/d(Wq)_i, q_i)
                expected = p[:, :b] @ pos + np.einsum("bm,bmd->bd", p[:, b:], neg)
                g_wq = (expected - pos) / temperature
                grad = g_wq.T @ q / b + reg * (w - eye)
                velocity = 0.9 * velocity - lr * grad
                w = w + velocity

        def ranks(weight: np.ndarray, data: list[TrainingExample]) -> list[int]:
            out = []
            for ex in data:
                qv = l2_normalise(ex.query @ weight.T)
                sims = corpus @ qv
                target = float(ex.positive @ qv)
                out.append(int((sims > target + 1e-6).sum()) + 1)
            return out

        def metrics(r: list[int]) -> tuple[float, float]:
            if not r:
                return 0.0, 0.0
            return float(np.mean([x <= k for x in r])), float(np.mean([1.0 / x for x in r]))

        eval_set = val or train
        rb, mb = metrics(ranks(eye, eval_set))
        ra, ma = metrics(ranks(w, eval_set))
        accepted = ra >= rb and (ra + ma) > (rb + mb)
        report = AdapterReport(len(train), len(val), rb, ra, mb, ma, accepted, k)
        log.info("adapter trained: %s", report)
        return QueryAdapter(w.astype(np.float32)), report
