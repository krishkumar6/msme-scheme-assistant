"""Groq / vLLM path against a stub OpenAI-compatible server (no network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nabard_qa.config import Settings
from nabard_qa.generation.llm import (
    ChainGenerator,
    ExtractiveGenerator,
    GenerationError,
    GenerationRequest,
    OpenAICompatibleGenerator,
    build_deep_generator,
    build_generator,
)


class _Stub(BaseHTTPRequestHandler):
    seen: list[dict] = []
    statuses: list[int] = []  # consumed one per request; empty -> 200
    error_message = "slow down"
    retry_after = "0"

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        status = _Stub.statuses.pop(0) if _Stub.statuses else 200
        if status != 200:
            self.send_response(status)
            self.send_header("retry-after", _Stub.retry_after)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": _Stub.error_message}}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("The corpus ", "is ₹25,000 crore [S1]."):
            chunk = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture
def stub_url():
    _Stub.seen, _Stub.statuses, _Stub.error_message = [], [], "slow down"
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/openai/v1"
    server.shutdown()


def _req():
    return GenerationRequest(system="sys", messages=[{"role": "user", "content": "q"}], question="q")


def test_streams_and_sends_auth(stub_url):
    gen = OpenAICompatibleGenerator(stub_url, "llama-3.3-70b-versatile", "gsk_test", name="groq:test")
    assert gen.complete(_req()) == "The corpus is ₹25,000 crore [S1]."
    call = _Stub.seen[0]
    assert call["path"] == "/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer gsk_test"
    assert not call["headers"]["User-Agent"].startswith("Python-urllib")
    assert call["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert call["body"]["stream"] is True


def test_rate_limit_is_retried_then_succeeds(stub_url):
    _Stub.statuses = [429]
    gen = OpenAICompatibleGenerator(stub_url, "m", "gsk_test", name="groq:test")
    assert gen.complete(_req()).endswith("[S1].")
    assert len(_Stub.seen) == 2


def test_persistent_rate_limit_becomes_generation_error(stub_url):
    _Stub.statuses = [429, 429, 429]
    gen = OpenAICompatibleGenerator(stub_url, "m", "gsk_test", name="groq:test", max_retries=2)
    with pytest.raises(GenerationError, match="429"):
        gen.complete(_req())
    assert len(_Stub.seen) == 3


def test_daily_quota_is_not_retried(stub_url):
    # Groq's daily cap: three 20 s retries only delayed the hand-over to Gemini by a minute
    _Stub.statuses = [429, 429, 429]
    _Stub.error_message = "Rate limit reached for model on tokens per day (TPD): Limit 200000"
    gen = OpenAICompatibleGenerator(stub_url, "m", "gsk_test", name="groq:test")
    with pytest.raises(GenerationError, match="daily quota"):
        gen.complete(_req())
    assert len(_Stub.seen) == 1


def test_long_rate_limit_wait_fails_over_instead_of_sleeping(stub_url, monkeypatch):
    _Stub.statuses = [429]
    monkeypatch.setattr(_Stub, "retry_after", "30")
    gen = OpenAICompatibleGenerator(stub_url, "m", "gsk_test", name="groq:test")
    with pytest.raises(GenerationError, match="rate-limited for 30s"):
        gen.complete(_req())
    assert len(_Stub.seen) == 1


def test_client_errors_are_not_retried(stub_url):
    _Stub.statuses = [404]
    gen = OpenAICompatibleGenerator(stub_url, "m", "gsk_test", name="groq:test")
    with pytest.raises(GenerationError, match="404"):
        gen.complete(_req())
    assert len(_Stub.seen) == 1


class _Fixed:
    def __init__(self, name, text=None, fail=False, fail_after=None):
        self.name, self.text, self.fail, self.fail_after = name, text, fail, fail_after

    def stream(self, req):
        if self.fail:
            raise GenerationError(f"{self.name} 429")
        yield self.text
        if self.fail_after:
            raise GenerationError("cut")

    def complete(self, req):
        return "".join(self.stream(req))


def test_chain_falls_through_to_next_provider():
    chain = ChainGenerator([_Fixed("groq", fail=True), _Fixed("gemini", "answer [S1]")])
    assert list(chain.stream_with_provider(_req())) == [("gemini", "answer [S1]")]


def test_chain_does_not_switch_mid_answer():
    chain = ChainGenerator([_Fixed("groq", "partial", fail_after=True), _Fixed("gemini", "x")])
    with pytest.raises(GenerationError, match="cut"):
        list(chain.stream_with_provider(_req()))


def test_chain_all_failed():
    chain = ChainGenerator([_Fixed("groq", fail=True), _Fixed("gemini", fail=True)])
    with pytest.raises(GenerationError, match="all providers failed"):
        chain.complete(_req())


def _clean_env(monkeypatch):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "NABARD_QA_LLM_CHAIN", "NABARD_QA_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


def test_auto_builds_chain_and_deep_puts_gemini_first(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    monkeypatch.setenv("GEMINI_API_KEY", "gem_x")
    s = Settings(_env_file=None)
    fast = build_generator(s)
    assert isinstance(fast, ChainGenerator)
    assert [m.name.split(":")[0] for m in fast.members] == ["groq", "gemini"]
    gem = fast.members[1]
    assert gem.base_url.endswith("/v1beta/openai") and gem.reasoning_effort == "low"
    deep = build_deep_generator(s, fast)
    assert isinstance(deep, OpenAICompatibleGenerator) and deep.name.startswith("gemini:")


def test_deep_falls_back_to_fast_without_gemini(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    s = Settings(_env_file=None)
    fast = build_generator(s)
    assert build_deep_generator(s, fast) is fast


def test_reasoning_effort_sent_only_when_configured(stub_url):
    OpenAICompatibleGenerator(stub_url, "m", "k", name="gemini:m", reasoning_effort="low").complete(_req())
    OpenAICompatibleGenerator(stub_url, "m", "k", name="groq:m").complete(_req())
    assert _Stub.seen[0]["body"]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in _Stub.seen[1]["body"]


def test_auto_provider_picks_groq_when_only_groq_key(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    gen = build_generator(Settings(_env_file=None))
    assert isinstance(gen, OpenAICompatibleGenerator) and gen.name.startswith("groq:")
    assert gen.api_key == "gsk_from_env"
    monkeypatch.delenv("GROQ_API_KEY")
    assert isinstance(build_generator(Settings(_env_file=None)), ExtractiveGenerator)
