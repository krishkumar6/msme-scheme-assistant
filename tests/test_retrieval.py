import numpy as np

from nabard_qa.retrieval.adapter import QueryAdapter, TrainingExample
from nabard_qa.retrieval.embeddings import l2_normalise
from nabard_qa.retrieval.hybrid import years_in_query
from nabard_qa.retrieval.lexical import BM25Index, tokenize

from .conftest import CORPUS


def test_tokenize_numbers_acronyms_plurals():
    toks = tokenize("What were the RIDF sanctions of ₹1,59,822 crore for banks?")
    assert "ridf" in toks and "1,59,822" in toks and "159822" in toks
    assert "bank" in toks and "sanction" in toks and "what" not in toks


def test_bm25_exact_terms_and_year_filter():
    idx = BM25Index(CORPUS)
    assert idx.search("Tranche XXIII corpus", 2)[0][0] == "ridf"
    assert idx.search("balance sheet size", 3, year="2017-18") == [] or all(
        cid != "bs20" for cid, _ in idx.search("balance sheet size", 3, year="2017-18"))


def test_years_in_query():
    assert years_in_query("compare 2017-18 with 2019–2020") == {"2017-18", "2019-20"}


def test_hybrid_retrieval_ranks_relevant_first(service):
    res = service.retriever.retrieve("interest rate of Long Term Irrigation Fund loans")
    assert res.chunks[0].chunk.id == "ltif"
    assert res.chunks[0].dense_rank is not None and res.chunks[0].lexical_rank is not None


def test_year_restriction(service):
    res = service.retriever.retrieve("balance sheet size crore", year="2019-20")
    assert {c.chunk.year for c in res.chunks} == {"2019-20"}


def test_adapter_learns_a_rotation_and_passes_gate():
    rng = np.random.default_rng(0)
    d, n_docs = 16, 200
    docs = l2_normalise(rng.normal(size=(n_docs, d)).astype(np.float32))
    # queries live in a rotated space: q = R^T doc + noise -> identity retrieval is poor
    rot, _ = np.linalg.qr(rng.normal(size=(d, d)))
    rot = rot.astype(np.float32)
    examples = []
    for i in range(120):
        doc = docs[i % n_docs]
        q = l2_normalise(doc @ rot + 0.05 * rng.normal(size=d).astype(np.float32))
        examples.append(TrainingExample(query=q, positive=doc, negatives=np.zeros((0, d), np.float32), positive_id=str(i)))
    adapter, report = QueryAdapter.fit(examples, docs, epochs=150, lr=0.2, reg=0.01)
    assert report.recall_adapted > report.recall_base + 0.3
    assert report.accepted and adapter.active


def test_adapter_rejected_when_it_does_not_help():
    rng = np.random.default_rng(1)
    d = 16
    docs = l2_normalise(rng.normal(size=(100, d)).astype(np.float32))
    # queries already equal their documents: identity is optimal, nothing to gain
    examples = [TrainingExample(query=docs[i], positive=docs[i], negatives=np.zeros((0, d), np.float32), positive_id=str(i))
                for i in range(40)]
    _, report = QueryAdapter.fit(examples, docs, epochs=20)
    assert not report.accepted


def test_adapter_round_trip(tmp_path):
    w = np.eye(4, dtype=np.float32) * 2
    from nabard_qa.retrieval.adapter import AdapterReport

    QueryAdapter(w).save(tmp_path / "a.npz", AdapterReport(1, 1, 0, 0, 0, 0, True, 5))
    loaded = QueryAdapter.load(tmp_path / "a.npz")
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    assert np.allclose(loaded.apply(q), q)  # scaling is normalised away
    assert not QueryAdapter.load(tmp_path / "missing.npz").active
