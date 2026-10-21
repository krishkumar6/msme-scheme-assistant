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
        # One pair per batch: identical scores, no padding to the longest passage, and peak memory
        # +45 MB instead of +490 MB for 40 passages (batch 16) - the difference between fitting a
        # 512 MB container or not. It is also faster on CPU (2.8 s vs 6.6 s for 40 passages).
        return [float(s) for s in self._model.rerank(query, list(passages), batch_size=1)]
