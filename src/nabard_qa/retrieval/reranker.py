"""Cross-encoder reranking (local ONNX)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)
INT8_SUFFIX = "-int8"


def _register_int8(model_name: str) -> str:
    """Register the int8-quantized ONNX file that Xenova repos ship next to model.onnx.
    About 3x faster on CPU, which matters on a fractional-CPU host."""
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    alias = model_name + INT8_SUFFIX
    if not any(m["model"] == alias for m in TextCrossEncoder.list_supported_models()):
        TextCrossEncoder.add_custom_model(alias, sources=ModelSource(hf=model_name),
                                          model_file="onnx/model_quantized.onnx", size_in_gb=0.03)
    return alias


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path, quantized: bool = False, max_words: int = 0) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        name = _register_int8(model_name) if quantized else model_name
        log.info("loading reranker %s", name)
        self.model_name = name
        self.max_words = max_words
        self._model = TextCrossEncoder(name, cache_dir=str(cache_dir))

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        if self.max_words:  # the model reads at most 512 tokens anyway; less is proportionally faster
            passages = [" ".join(p.split()[: self.max_words]) for p in passages]
        # One pair per batch: identical scores, no padding to the longest passage, and peak memory
        # +45 MB instead of +490 MB for 40 passages (batch 16) - the difference between fitting a
        # 512 MB container or not. It is also faster on CPU (2.8 s vs 6.6 s for 40 passages).
        return [float(s) for s in self._model.rerank(query, list(passages), batch_size=1)]
