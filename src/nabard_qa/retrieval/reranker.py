"""Cross-encoder reranking (local ONNX)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        log.info("loading reranker %s", model_name)
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name, cache_dir=str(cache_dir))

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, list(passages), batch_size=16)]
