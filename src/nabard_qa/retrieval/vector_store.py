"""Persistent vector store (ChromaDB, cosine space)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ..schemas import Chunk

log = logging.getLogger(__name__)
_BATCH = 500


class ChromaStore:
    def __init__(self, path: Path, collection: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._col.count()

    def ids(self) -> set[str]:
        out: set[str] = set()
        offset = 0
        while True:
            page = self._col.get(include=[], limit=5000, offset=offset)
            if not page["ids"]:
                return out
            out.update(page["ids"])
            offset += len(page["ids"])

    def metadatas(self) -> dict[str, dict]:
        """Stored metadata for every id (to detect title/year changes from the manifest)."""
        out: dict[str, dict] = {}
        offset = 0
        while True:
            page = self._col.get(include=["metadatas"], limit=5000, offset=offset)
            if not page["ids"]:
                return out
            out.update(zip(page["ids"], page["metadatas"], strict=True))
            offset += len(page["ids"])

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        for i in range(0, len(chunks), _BATCH):
            part = chunks[i : i + _BATCH]
            self._col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                metadatas=[c.to_metadata() for c in part],
                embeddings=embeddings[i : i + _BATCH].tolist(),
            )

    def delete(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        for i in range(0, len(ids), _BATCH):
            self._col.delete(ids=ids[i : i + _BATCH])

    def query(self, embedding: np.ndarray, k: int, year: str | None = None) -> list[tuple[str, float]]:
        """Top-k (id, cosine similarity)."""
        n = min(k, self.count())
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            where={"year": year} if year else None,
            include=["distances"],
        )
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]

    def get(self, ids: Sequence[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        res = self._col.get(ids=list(ids), include=["documents", "metadatas"])
        return {
            i: Chunk.from_store(i, doc, meta)
            for i, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        }

    def get_embeddings(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not ids:
            return {}
        res = self._col.get(ids=list(ids), include=["embeddings"])
        return {i: np.asarray(e, dtype=np.float32) for i, e in zip(res["ids"], res["embeddings"], strict=True)}

    def all(self) -> tuple[list[Chunk], np.ndarray]:
        chunks: list[Chunk] = []
        vecs: list[np.ndarray] = []
        offset = 0
        while True:
            page = self._col.get(include=["documents", "metadatas", "embeddings"], limit=2000, offset=offset)
            if not page["ids"]:
                break
            for i, doc, meta, emb in zip(page["ids"], page["documents"], page["metadatas"], page["embeddings"], strict=True):
                chunks.append(Chunk.from_store(i, doc, meta))
                vecs.append(np.asarray(emb, dtype=np.float32))
            offset += len(page["ids"])
        return chunks, (np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32))
