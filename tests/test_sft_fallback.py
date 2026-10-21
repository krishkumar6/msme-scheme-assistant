"""Synthetic Q&A generation switches provider when the main one hits its daily cap."""

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "build_sft_dataset", Path(__file__).resolve().parents[1] / "training" / "build_sft_dataset.py")
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b  # pydantic resolves the models' forward references through it
spec.loader.exec_module(b)

SETTINGS = SimpleNamespace(
    groq_api_key="g", groq_base_url="https://groq.test/v1", groq_model="gpt-oss", groq_reasoning_effort=None,
    gemini_api_key="m", gemini_base_url="https://gemini.test/v1", gemini_model="flash",
    gemini_reasoning_effort=None,
)
PAIRS = {"pairs": [{"question": "What is the PMEGP subsidy for rural women?", "answer": "35% [S1]",
                    "question_type": "eligibility"}]}


def _fake_urlopen(calls):  # noqa: ANN001, ANN202
    def urlopen(req, timeout=None):  # noqa: ANN001, ARG001
        calls.append(req.full_url)
        if "groq" in req.full_url:
            body = b'{"error":{"message":"Rate limit reached ... on tokens per day (TPD): Limit 200000"}}'
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO(body))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(PAIRS)}}]}).encode())
    return urlopen


def test_daily_limit_switches_to_fallback_immediately(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(b.urllib.request, "urlopen", _fake_urlopen(calls))
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    gen = b.PairGenerator("groq", SETTINGS, fallback="gemini")
    pairs = gen("passage")
    assert pairs and pairs[0].answer == "35% [S1]"
    assert gen.provider == "gemini" and gen.model == "flash"
    assert sum("groq" in u for u in calls) == 1  # no pointless waiting on a daily cap


def test_without_fallback_it_keeps_waiting_then_gives_up(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(b.urllib.request, "urlopen", _fake_urlopen(calls))
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    gen = b.PairGenerator("groq", SETTINGS)
    assert gen("passage") is None and gen.provider == "groq"
