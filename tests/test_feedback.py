import numpy as np
import pytest

from nabard_qa.feedback.learner import heading_pairs
from nabard_qa.feedback.store import UnknownAnswerError
from nabard_qa.schemas import AskRequest

from .conftest import CORPUS


def _log(feedback, emb, cited=("a",), retrieved=("a", "b")):  # noqa: ANN001
    return feedback.log_interaction(
        question="q", standalone_question="q", answer="x [S1]", provider="t", grounded=True, latency_ms=1,
        retrieved=[{"chunk_id": c, "ref": f"S{i}", "score": 1.0, "cited": c in cited} for i, c in enumerate(retrieved, 1)],
        query_embedding=emb,
    )


def test_thumbs_up_creates_positive_signals_and_training_pairs(feedback):
    emb = np.array([1, 0, 0], dtype=np.float32)
    aid = _log(feedback, emb)
    signals = feedback.add_feedback(aid, 1, irrelevant=["b"])
    assert signals == {"a": 1.0, "b": -1.0}
    pairs = feedback.training_pairs()
    assert pairs == [("q", "a", ["b"], "feedback")]
    assert feedback.stats()["thumbs_up"] == 1


def test_thumbs_down_is_weak_negative_on_cited(feedback):
    aid = _log(feedback, np.array([0, 1, 0], dtype=np.float32))
    assert feedback.add_feedback(aid, -1) == {"a": -0.5}


def test_ids_not_in_the_answer_are_ignored(feedback):
    aid = _log(feedback, np.array([0, 1, 0], dtype=np.float32))
    assert feedback.add_feedback(aid, 1, relevant=["not-retrieved"]) == {"a": 1.0}


def test_boosts_apply_only_to_similar_queries(feedback):
    aid = _log(feedback, np.array([1, 0, 0], dtype=np.float32))
    feedback.add_feedback(aid, 1)
    near = feedback.boosts(np.array([0.99, 0.14, 0], dtype=np.float32), 0.8)
    far = feedback.boosts(np.array([0, 1, 0], dtype=np.float32), 0.8)
    assert near["a"] > 0 and far == {}


def test_unknown_answer(feedback):
    with pytest.raises(UnknownAnswerError):
        feedback.add_feedback("nope", 1)


def test_feedback_reorders_retrieval_for_similar_question(service):
    q = "rural infrastructure fund crore"
    before = [c.chunk.id for c in service.retriever.retrieve(q).chunks]
    res = service.answer(AskRequest(question=q))
    target = before[-1]  # promote the passage that ranked last
    service.feedback.add_feedback(res.answer_id, 1, relevant=[target],
                                  irrelevant=[c for c in before if c != target])
    after = [c.chunk.id for c in service.retriever.retrieve(q).chunks]
    assert after[0] == target


def test_positive_feedback_recalls_a_passage_retrieval_missed(service, monkeypatch):
    q = "rural infrastructure fund crore"
    res = service.answer(AskRequest(question=q))
    target = res.citations[-1].chunk_id  # the user found the last-ranked source useful
    service.feedback.add_feedback(res.answer_id, 1, relevant=[target])
    # now retrieval alone surfaces a single candidate, and it isn't the target
    for name in ("final_k", "rerank_candidates", "dense_k", "lexical_k"):
        monkeypatch.setattr(service.s, name, 1)
    monkeypatch.setattr(service.s, "feedback_boost", 50.0)  # a recalled passage wins the single slot
    monkeypatch.setattr(service.s, "feedback_recall_k", 0)
    without_recall = [c.chunk.id for c in service.retriever.retrieve(q, k=1).chunks]
    monkeypatch.setattr(service.s, "feedback_recall_k", 5)
    with_recall = [c.chunk.id for c in service.retriever.retrieve(q, k=1).chunks]
    assert without_recall != [target] and with_recall == [target]


def test_heading_pairs_strip_numbering():
    rows = heading_pairs(CORPUS)
    assert ("Rural Infrastructure Development Fund (2017-18)", "ridf", []) in rows
