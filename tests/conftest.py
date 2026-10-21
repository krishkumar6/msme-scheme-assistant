"""Shared fixtures: a tiny corpus, a deterministic bag-of-words embedder, a
real (temporary) Chroma index and a QAService wired with fakes - no model
downloads or API keys needed."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

import numpy as np
import pytest

from nabard_qa.config import Settings
from nabard_qa.feedback.store import FeedbackStore
from nabard_qa.generation.llm import GenerationRequest
from nabard_qa.retrieval.adapter import QueryAdapter
from nabard_qa.retrieval.embeddings import l2_normalise
from nabard_qa.retrieval.hybrid import HybridRetriever
from nabard_qa.retrieval.lexical import BM25Index
from nabard_qa.retrieval.vector_store import ChromaStore
from nabard_qa.schemas import Chunk
from nabard_qa.service import QAService

DIM = 128


class HashEmbedder:
    """Bag-of-words hashed into DIM buckets: similar wording -> similar vectors."""

    model_name = "hash"
    dim = DIM

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        return v

    def embed_documents(self, texts, batch_size: int = 32) -> np.ndarray:  # noqa: ANN001
        return l2_normalise(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def embed_queries(self, texts) -> np.ndarray:  # noqa: ANN001
        return self.embed_documents(list(texts))


class EchoGenerator:
    """Cites the first source and quotes its first sentence."""

    name = "echo"

    def stream(self, req: GenerationRequest) -> Iterator[str]:
        if not req.sources:
            yield "The reports do not contain this information."
            return
        first = req.sources[0].chunk.text.split(". ")[0]
        yield f"{first} [S1]"

    def complete(self, req: GenerationRequest) -> str:
        return "".join(self.stream(req))


CORPUS = [
    Chunk(id="ridf", text="The RIDF corpus was raised to ₹25,000 crore in 2017–18 (Tranche XXIII) for rural infrastructure.",
          source="ar.pdf", report="NABARD Annual Report 2017-18", year="2017-18",
          section="6.2 Rural Infrastructure Development Fund", pdf_pages=[62], printed_pages=["100"]),
    Chunk(id="ltif", text="Loans under the Long Term Irrigation Fund carry an interest rate of 6 per cent per annum with a 15 year tenure.",
          source="ar.pdf", report="NABARD Annual Report 2017-18", year="2017-18",
          section="6.3 Long-Term Irrigation Fund", pdf_pages=[65], printed_pages=["107"]),
    Chunk(id="shg", text="SHG savings with banks crossed ₹19,592 crore and NPA levels declined to 6.12 per cent.",
          source="ar.pdf", report="NABARD Annual Report 2017-18", year="2017-18",
          section="3.2.1 Scaling-up Microfinance Initiatives", pdf_pages=[39], printed_pages=["55"]),
    Chunk(id="bs20", text="NABARD's balance sheet size reached ₹5,32,075 crore, growing 9.15 per cent year on year.",
          source="ar20.pdf", report="NABARD Annual Report 2019-20", year="2019-20",
          section="From the Chairman", pdf_pages=[7], printed_pages=["v"]),
]


@pytest.fixture
def settings(tmp_path) -> Settings:  # noqa: ANN001
    s = Settings(
        data_dir=tmp_path, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
        index_dir=tmp_path / "index", model_cache_dir=tmp_path / "models", db_path=tmp_path / "app.db",
        adapter_path=tmp_path / "adapter" / "a.npz", reranker_model=None, llm_provider="extractive",
        final_k=3, dense_k=4, lexical_k=4, rerank_candidates=4,
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def store(settings, embedder) -> ChromaStore:  # noqa: ANN001
    st = ChromaStore(settings.index_dir, "test_chunks")
    st.upsert(CORPUS, embedder.embed_documents([c.embed_text for c in CORPUS]))
    return st


@pytest.fixture
def feedback(settings) -> FeedbackStore:  # noqa: ANN001
    fb = FeedbackStore(settings.db_path)
    yield fb
    fb.close()


@pytest.fixture
def service(settings, embedder, store, feedback) -> QAService:  # noqa: ANN001
    retriever = HybridRetriever(settings, embedder, store, BM25Index(CORPUS), None, QueryAdapter(), feedback)
    return QAService(settings, retriever, EchoGenerator(), feedback, store)
