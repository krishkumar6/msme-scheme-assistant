import json

from fastapi.testclient import TestClient

from nabard_qa.api.app import create_app
from nabard_qa.generation.llm import GenerationError
from nabard_qa.schemas import AskRequest, Turn
from nabard_qa.service import normalise_citations, personalised_query


def test_citation_styles_are_normalised():
    assert normalise_citations("₹10 lakh【S1】【S2】.") == "₹10 lakh[S1][S2]."
    assert normalise_citations("x [S1, S3] y ［S2］ z [ S4 ]") == "x [S1][S3] y [S2] z [S4]"
    assert normalise_citations("x [S1; 2]") == "x [S1][S2]"
    assert normalise_citations("see [Table 1] and [1]") == "see [Table 1] and [1]"  # not citations
    assert normalise_citations("35% [​S3][​S6]") == "35% [S3][S6]"  # gpt-oss zero-width spaces


def test_answer_cites_and_logs(service):
    res = service.answer(AskRequest(question="What was the RIDF corpus for Tranche XXIII?"))
    assert res.grounded and "[S1]" in res.answer
    assert res.citations[0].chunk_id == "ridf" and res.citations[0].cited
    assert res.citations[0].pages == "p. 100"
    assert service.feedback.stats()["interactions"] == 1


def test_deep_mode_uses_more_passages_and_deep_generator(service):
    class Deep:
        name = "deep-model"

        def stream(self, req):
            self.n_sources = len(req.sources)
            yield "deep answer [S1]"

        def complete(self, req):
            return "".join(self.stream(req))

    deep = Deep()
    service.deep_generator = deep
    service.s.deep_k, service.s.final_k = 4, 1
    fast = service.answer(AskRequest(question="crore fund"))
    res = service.answer(AskRequest(question="crore fund", mode="deep"))
    assert len(fast.citations) == 1 and fast.provider == "echo"
    assert res.provider == "deep-model" and deep.n_sources == 4


def test_deep_failure_retries_fast_chain_with_top_passages(service):
    class Failing:
        name = "gemini"

        def stream(self, req):
            raise GenerationError("404 model gone")
            yield  # pragma: no cover

        def complete(self, req):
            raise GenerationError("404")

    seen = {}

    class Fast:
        name = "groq"

        def stream(self, req):
            seen["n"] = len(req.sources)
            yield "fast answer [S1]"

        def complete(self, req):
            return "".join(self.stream(req))

    service.generator, service.deep_generator = Fast(), Failing()
    service.s.deep_k, service.s.final_k = 4, 2
    res = service.answer(AskRequest(question="crore fund", mode="deep"))
    assert res.provider == "groq" and res.grounded and seen["n"] == 2


def test_stream_event_order(service):
    events = list(service.stream(AskRequest(question="SHG savings with banks")))
    assert [e["type"] for e in events][0] == "sources"
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "delta" for e in events)


def test_follow_up_rewrite_heuristic(service):
    service.generator = __import__("nabard_qa.generation.llm", fromlist=["ExtractiveGenerator"]).ExtractiveGenerator()
    q = service.standalone_question("what about its interest rate?",
                                    [Turn(role="user", content="Tell me about the Long Term Irrigation Fund"),
                                     Turn(role="assistant", content="...")])
    assert "Long Term Irrigation Fund" in q


class _FakeReranker:
    model_name = "fake"

    def score(self, query, passages):  # noqa: ANN001
        return [-5.0 for _ in passages]  # nothing relevant


def test_relevance_gate_drops_irrelevant_sources(service):
    service.retriever.reranker = _FakeReranker()
    res = service.answer(AskRequest(question="Who won the IPL in 2018?"))
    assert res.citations == [] and not res.grounded
    assert "do not contain" in res.answer


def test_no_relevant_sources_never_calls_the_llm(service):
    service.retriever.reranker = _FakeReranker()
    service.generator = _BrokenGenerator()  # would raise if called
    res = service.answer(AskRequest(question="What is the PMEGP subsidy for women?"))
    assert res.provider == "no-sources" and "do not contain" in res.answer and "won't guess" in res.answer


class _LengthSensitiveReranker:
    """Like MS MARCO cross-encoders: long first-person questions score low."""

    model_name = "fake"

    def score(self, query, passages):  # noqa: ANN001
        return [3.0 if len(query.split()) <= 8 else -3.0 for _ in passages]


class _CondensingGenerator:
    """Condenses when asked for a search query; otherwise cites the first source."""

    name = "condensing"

    def stream(self, req):  # noqa: ANN001
        yield f"{req.sources[0].chunk.text.split('. ')[0]} [S1]" if req.sources else "not found"

    def complete(self, req):  # noqa: ANN001
        if "search query" in req.system:
            return "Long Term Irrigation Fund interest rate"
        return "".join(self.stream(req))


def test_long_question_is_condensed_when_the_gate_rejects_it(service):
    service.retriever.reranker = _LengthSensitiveReranker()
    service.generator = _CondensingGenerator()
    res = service.answer(AskRequest(
        question="Hello, I am a farmer from Odisha and I would like to know what interest rate applies under the irrigation fund?"))
    assert res.grounded and res.citations
    assert res.standalone_question == "Long Term Irrigation Fund interest rate"


def test_personal_question_searches_the_applicants_state():
    tn = "Woman; SC; Tamil Nadu; rural; manufacturing; new unit"
    assert personalised_query("What capital subsidy can I get?", tn) == \
        "What capital subsidy can I get? Tamil Nadu state MSME scheme"
    # a central scheme, a state already named, an impersonal question, or no state: unchanged
    for q, p in [("Am I eligible for PMEGP?", tn), ("What subsidy can I get in Gujarat?", tn),
                 ("What is RIDF?", tn), ("What subsidy can I get?", "Woman; SC")]:
        assert personalised_query(q, p) == q


def test_profile_reaches_retrieval_and_prompt(service, monkeypatch):
    seen, real = [], service.retriever.retrieve
    monkeypatch.setattr(service.retriever, "retrieve", lambda q, **kw: seen.append(q) or real(q, **kw))
    res = service.answer(AskRequest(question="Which irrigation fund loans can I get?", profile="Farmer, Odisha"))
    assert seen[0].endswith("Odisha state MSME scheme")
    assert res.standalone_question == "Which irrigation fund loans can I get?"


class _BrokenGenerator:
    name = "broken"

    def stream(self, req):  # noqa: ANN001
        raise GenerationError("down")
        yield  # pragma: no cover

    def complete(self, req):  # noqa: ANN001
        raise GenerationError("down")


def test_llm_failure_falls_back_to_extracts(service):
    service.generator = _BrokenGenerator()
    res = service.answer(AskRequest(question="Long Term Irrigation Fund interest rate"))
    assert res.provider == "extractive"
    assert "unavailable" in res.answer and "6 per cent" in res.answer


def test_api_endpoints(settings, service):
    with TestClient(create_app(settings, service)) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["chunks"] == 4 and h["years"] == ["2017-18", "2019-20"]

        r = client.post("/api/ask", json={"question": "RIDF corpus Tranche XXIII"})
        assert r.status_code == 200 and r.headers["x-request-id"]
        answer_id = r.json()["answer_id"]

        with client.stream("POST", "/api/ask/stream", json={"question": "SHG savings"}) as s:
            events = [json.loads(line[5:]) for line in s.iter_lines() if line.startswith("data:")]
        assert events[0]["type"] == "sources" and events[-1]["type"] == "done"

        ok = client.post("/api/feedback", json={"answer_id": answer_id, "rating": 1})
        assert ok.status_code == 200 and ok.json()["signals"] >= 1
        assert client.post("/api/feedback", json={"answer_id": "missing", "rating": 1}).status_code == 404
        assert client.post("/api/ask", json={"question": "x" * 2000}).status_code == 422
        assert client.post("/api/admin/train-adapter").status_code == 403
        assert client.get("/").status_code == 200
