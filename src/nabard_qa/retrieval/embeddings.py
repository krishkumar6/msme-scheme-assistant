"""Local ONNX embeddings (fastembed) - no GPU or API key needed."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path, query_instruction: str = "") -> None:
        from fastembed import TextEmbedding

        log.info("loading embedding model %s", model_name)
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))
        self.dim = int(self.embed_documents(["probe"]).shape[1])

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(list(texts), batch_size=batch_size)), dtype=np.float32)
        return l2_normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents([f"{self.query_instruction}{t}" for t in texts])
