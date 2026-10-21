"""Interaction log + feedback store (SQLite).

Every answer is logged with its query embedding and the passages it used, so a
later thumbs-up/down can be turned into retrieval signals:

* ``signals``: (answer, chunk, weight) rows. At query time, feedback left on
  *similar* past questions boosts / demotes those chunks immediately.
* ``training_pairs``: (query, positive, negatives) rows consumed by the query
  adapter trainer (``nabard-qa train-adapter``) and the SFT dataset builder.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    question TEXT NOT NULL,
    standalone_question TEXT NOT NULL,
    answer TEXT NOT NULL,
    provider TEXT NOT NULL,
    grounded INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    retrieved TEXT NOT NULL,          -- json [{chunk_id, ref, score, cited}]
    query_embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id TEXT NOT NULL REFERENCES interactions(id),
    created_at REAL NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    relevant TEXT NOT NULL,
    irrelevant TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    answer_id TEXT NOT NULL REFERENCES interactions(id),
    chunk_id TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (answer_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS training_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    origin TEXT NOT NULL,             -- feedback | synthetic | heading
    query TEXT NOT NULL,
    positive_id TEXT NOT NULL,
    negative_ids TEXT NOT NULL,
    UNIQUE (origin, query, positive_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_answer ON feedback(answer_id);
"""


class UnknownAnswerError(KeyError):
    pass


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._cache: tuple[np.ndarray, list[dict[str, float]]] | None = None

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # ------------------------------------------------------------ logging
    def log_interaction(
        self,
        question: str,
        standalone_question: str,
        answer: str,
        provider: str,
        grounded: bool,
        latency_ms: int,
        retrieved: list[dict],
        query_embedding: np.ndarray,
    ) -> str:
        answer_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (answer_id, time.time(), question, standalone_question, answer, provider, int(grounded),
                 latency_ms, json.dumps(retrieved), query_embedding.astype(np.float32).tobytes()),
            )
        return answer_id

    # ----------------------------------------------------------- feedback
    def add_feedback(
        self,
        answer_id: str,
        rating: int,
        comment: str | None = None,
        relevant: list[str] | None = None,
        irrelevant: list[str] | None = None,
    ) -> dict[str, float]:
        """Store feedback and derive chunk-level signals. Returns the signals."""
        relevant, irrelevant = list(relevant or []), list(irrelevant or [])
        row = self._conn.execute(
            "SELECT standalone_question, retrieved FROM interactions WHERE id=?", (answer_id,)
        ).fetchone()
        if row is None:
            raise UnknownAnswerError(answer_id)
        query, retrieved_json = row
        retrieved = json.loads(retrieved_json)
        known = {r["chunk_id"] for r in retrieved}
        relevant = [c for c in relevant if c in known]
        irrelevant = [c for c in irrelevant if c in known]
        cited = [r["chunk_id"] for r in retrieved if r.get("cited")]

        signals: dict[str, float] = {}
        if rating > 0:
            for c in relevant or cited:
                signals[c] = 1.0
        else:
            for c in cited:
                signals[c] = -0.5  # weak: the passage may be fine and the answer poor
            for c in relevant:
                signals[c] = 1.0
        for c in irrelevant:
            signals[c] = -1.0

        positives = [c for c, w in signals.items() if w > 0]
        negatives = [c for c, w in signals.items() if w < 0]
        now = time.time()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO feedback (answer_id, created_at, rating, comment, relevant, irrelevant) VALUES (?,?,?,?,?,?)",
                (answer_id, now, rating, comment, json.dumps(relevant), json.dumps(irrelevant)),
            )
            for c, w in signals.items():
                conn.execute(
                    "INSERT INTO signals VALUES (?,?,?) ON CONFLICT(answer_id, chunk_id) DO UPDATE SET weight=excluded.weight",
                    (answer_id, c, w),
                )
            for p in positives:
                conn.execute(
                    "INSERT OR IGNORE INTO training_pairs (created_at, origin, query, positive_id, negative_ids) VALUES (?,?,?,?,?)",
                    (now, "feedback", query, p, json.dumps(negatives)),
                )
        self._cache = None
        return signals

    def add_training_pairs(self, origin: str, rows: list[tuple[str, str, list[str]]]) -> int:
        """Bulk insert (query, positive_id, negative_ids). Returns rows inserted."""
        now = time.time()
        before = self._conn.total_changes
        with self._tx() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO training_pairs (created_at, origin, query, positive_id, negative_ids) VALUES (?,?,?,?,?)",
                [(now, origin, q, p, json.dumps(n)) for q, p, n in rows],
            )
        return self._conn.total_changes - before

    def training_pairs(self, origins: tuple[str, ...] = ("feedback", "synthetic", "heading")) -> list[tuple[str, str, list[str], str]]:
        marks = ",".join("?" * len(origins))
        rows = self._conn.execute(
            f"SELECT query, positive_id, negative_ids, origin FROM training_pairs WHERE origin IN ({marks})", origins
        ).fetchall()
        return [(q, p, json.loads(n), o) for q, p, n, o in rows]

    # ------------------------------------------------- query-time boosting
    def _load_signals(self) -> tuple[np.ndarray, list[dict[str, float]]]:
        if self._cache is None:
            rows = self._conn.execute(
                "SELECT i.id, i.query_embedding, s.chunk_id, s.weight "
                "FROM signals s JOIN interactions i ON i.id = s.answer_id"
            ).fetchall()
            by_answer: dict[str, tuple[bytes, dict[str, float]]] = {}
            for aid, emb, chunk, w in rows:
                by_answer.setdefault(aid, (emb, {}))[1][chunk] = w
            if by_answer:
                embs = np.stack([np.frombuffer(e, dtype=np.float32) for e, _ in by_answer.values()])
                self._cache = (embs, [sig for _, sig in by_answer.values()])
            else:
                self._cache = (np.zeros((0, 0), dtype=np.float32), [])
        return self._cache

    def boosts(self, query_embedding: np.ndarray, min_similarity: float) -> dict[str, float]:
        """Similarity-weighted sum of feedback on chunks, over past questions
        similar to this one. Positive = users found the chunk useful here."""
        embs, signals = self._load_signals()
        if not signals:
            return {}
        sims = embs @ query_embedding
        out: dict[str, float] = defaultdict(float)
        for sim, sig in zip(sims, signals, strict=True):
            if sim >= min_similarity:
                scale = (sim - min_similarity) / (1 - min_similarity + 1e-9)
                for chunk, w in sig.items():
                    out[chunk] += w * (0.5 + 0.5 * scale)
        return {c: float(np.tanh(v)) for c, v in out.items()}  # bounded in (-1, 1)

    # --------------------------------------------------------------- stats
    def stats(self) -> dict[str, int]:
        q = lambda sql: self._conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "interactions": q("SELECT COUNT(*) FROM interactions"),
            "feedback": q("SELECT COUNT(*) FROM feedback"),
            "thumbs_up": q("SELECT COUNT(*) FROM feedback WHERE rating > 0"),
            "thumbs_down": q("SELECT COUNT(*) FROM feedback WHERE rating < 0"),
            "training_pairs": q("SELECT COUNT(*) FROM training_pairs"),
        }

    def export_interactions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT i.id, i.question, i.standalone_question, i.answer, i.retrieved, f.rating, f.comment "
            "FROM interactions i LEFT JOIN feedback f ON f.answer_id = i.id ORDER BY i.created_at"
        ).fetchall()
        return [
            {"answer_id": r[0], "question": r[1], "standalone_question": r[2], "answer": r[3],
             "retrieved": json.loads(r[4]), "rating": r[5], "comment": r[6]}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
