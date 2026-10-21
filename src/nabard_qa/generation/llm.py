"""Answer generators.

* ``OpenAICompatibleGenerator`` - Groq, Gemini, or any OpenAI-compatible server
  (llama.cpp, vLLM) hosting the LoRA fine-tuned model from ``training/finetune_lora.py``.
* ``ExtractiveGenerator`` - no LLM: returns the best-matching source sentences
  with citations. Keeps the service usable offline and in tests.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from ..config import Settings
from ..retrieval.lexical import tokenize
from ..schemas import ScoredChunk

log = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """The model call failed; the caller may fall back to extractive answers."""


@dataclass
class GenerationRequest:
    system: str
    messages: list[dict]
    question: str
    sources: list[ScoredChunk] = field(default_factory=list)
    max_tokens: int | None = None
    effort: str | None = None  # reasoning effort override for reasoning models (e.g. "low" for query rewriting)


class Generator(Protocol):
    name: str

    def stream(self, req: GenerationRequest) -> Iterator[str]: ...

    def complete(self, req: GenerationRequest) -> str: ...


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat APIs: Groq, Gemini, or the fine-tuned model on llama.cpp / vLLM
# --------------------------------------------------------------------------- #
_DAILY_LIMIT = re.compile(r"per day|\bTPD\b|\bRPD\b|daily", re.I)


class OpenAICompatibleGenerator:
    def __init__(self, base_url: str, model: str, api_key: str | None, name: str, max_tokens: int = 4096,
                 max_retries: int = 3, max_wait_s: float = 20.0, reasoning_effort: str | None = None,
                 timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s  # socket timeout: max silence before giving up
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.reasoning_effort = reasoning_effort

    def _open(self, http_req: urllib.request.Request):  # type: ignore[no-untyped-def]
        """urlopen with retries on 429 / 5xx. Groq's free tier enforces
        tokens-per-minute limits, so a short wait usually succeeds. Retrying is
        safe because the error arrives before any answer text is streamed."""
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(http_req, timeout=self.timeout_s)
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise
                if e.code == 429:
                    detail = e.read().decode("utf-8", "replace")
                    if _DAILY_LIMIT.search(detail):
                        # waiting can't lift a daily cap: hand over to the next provider now
                        # (three 20 s retries used to delay every answer by a minute)
                        raise GenerationError(f"{self.name} daily quota reached") from e
                try:
                    wait = float(e.headers.get("retry-after", ""))
                except ValueError:
                    wait = 2.0 * 2**attempt
                wait = min(max(wait, 0.5), self.max_wait_s)
                log.warning("%s HTTP %s; retrying in %.1fs (%d/%d)", self.name, e.code, wait,
                            attempt + 1, self.max_retries)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def stream(self, req: GenerationRequest) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}, *req.messages],
            "max_tokens": req.max_tokens or self.max_tokens,
            "temperature": 0.1,
            "stream": True,
        }
        effort = req.effort or self.reasoning_effort
        if effort and self.reasoning_effort is not None:  # only for models configured as reasoning models
            body["reasoning_effort"] = effort
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            # Cloudflare-fronted APIs (Groq) block urllib's default "Python-urllib" agent
            "User-Agent": "nabard-qa/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        try:
            with self._open(http_req) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or [{}]
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            reason = {401: "API key is missing or invalid", 429: "rate limit / quota reached"}.get(e.code, detail)
            raise GenerationError(f"{self.name} returned HTTP {e.code}: {reason}") from e
        except OSError as e:
            raise GenerationError(f"{self.name} unreachable at {self.base_url}: {e}") from e

    def complete(self, req: GenerationRequest) -> str:
        return "".join(self.stream(req))


# --------------------------------------------------------------------------- #
# Extractive fallback
# --------------------------------------------------------------------------- #
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(₹\d])")


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def _pick(self, req: GenerationRequest) -> list[tuple[float, int, str]]:
        q = set(tokenize(req.question))
        picked: list[tuple[float, int, str]] = []
        for i, sc in enumerate(req.sources, start=1):
            units = [ln for ln in sc.chunk.text.split("\n") if "|" in ln] if sc.chunk.kind == "table" else []
            units += [s for s in _SENT.split(sc.chunk.text.replace("\n", " ")) if 6 <= len(s.split()) <= 80]
            for u in units:
                toks = set(tokenize(u))
                overlap = len(q & toks) / (len(q) or 1)
                if overlap > 0:
                    picked.append((overlap + 0.15 / i, i, u.strip()))  # small bonus for higher-ranked sources
        picked.sort(reverse=True)
        out, seen = [], set()
        for score, i, u in picked:
            if u not in seen:
                out.append((score, i, u))
                seen.add(u)
            if len(out) == self.max_sentences:
                break
        return out

    def stream(self, req: GenerationRequest) -> Iterator[str]:
        picked = self._pick(req)
        if not picked:
            yield ("I could not find this in the scheme documents and reports I have indexed. "
                   "Try rephrasing with a scheme name, fund or year.")
            return
        yield "Most relevant extracts from the reports (no LLM is configured, so these are quoted, not summarised):\n\n"
        for _, i, u in picked:
            yield f"- {u} [S{i}]\n"

    def complete(self, req: GenerationRequest) -> str:
        return "".join(self.stream(req))


# --------------------------------------------------------------------------- #
# Provider chain: fastest first, next provider on failure
# --------------------------------------------------------------------------- #
class ChainGenerator:
    """Try generators in order. A provider that fails *before* producing text
    (rate limit, quota, outage, bad model id) hands over to the next one; a
    failure mid-answer is raised so the caller can mark the answer interrupted."""

    def __init__(self, members: list[Generator]) -> None:
        if not members:
            raise ValueError("ChainGenerator needs at least one generator")
        self.members = members
        self.name = " > ".join(m.name for m in members)

    def stream_with_provider(self, req: GenerationRequest) -> Iterator[tuple[str, str]]:
        errors: list[str] = []
        for member in self.members:
            produced = False
            try:
                for delta in member.stream(req):
                    produced = True
                    yield member.name, delta
                return
            except GenerationError as e:
                if produced:
                    raise
                errors.append(str(e))
                log.warning("%s failed, trying next provider: %s", member.name, e)
        raise GenerationError("all providers failed: " + " | ".join(errors))

    def stream(self, req: GenerationRequest) -> Iterator[str]:
        for _, delta in self.stream_with_provider(req):
            yield delta

    def complete(self, req: GenerationRequest) -> str:
        return "".join(self.stream(req))


def _make(provider: str, settings: Settings) -> Generator | None:
    """One provider, or None if its credentials are missing."""
    if provider == "groq":
        if not settings.groq_api_key:
            return None
        return OpenAICompatibleGenerator(settings.groq_base_url, settings.groq_model, settings.groq_api_key,
                                         name=f"groq:{settings.groq_model}", max_tokens=settings.max_tokens,
                                         reasoning_effort=settings.groq_reasoning_effort,
                                         timeout_s=settings.groq_timeout_s)
    if provider == "gemini":
        if not settings.gemini_api_key:
            return None
        return OpenAICompatibleGenerator(settings.gemini_base_url, settings.gemini_model, settings.gemini_api_key,
                                         name=f"gemini:{settings.gemini_model}", max_tokens=settings.gemini_max_tokens,
                                         reasoning_effort=settings.gemini_reasoning_effort,
                                         timeout_s=settings.gemini_timeout_s)
    if provider == "local":
        return OpenAICompatibleGenerator(settings.local_base_url, settings.local_model, settings.local_api_key,
                                         name=f"local:{settings.local_model}", max_tokens=settings.max_tokens)
    if provider == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown provider {provider!r} (choose from groq, gemini, local, extractive)")


def _chain_order(settings: Settings) -> list[str]:
    return [p.strip() for p in settings.llm_chain.split(",") if p.strip()]


def build_generator(settings: Settings) -> Generator:
    """Generator for normal questions."""
    if settings.llm_provider != "auto":
        gen = _make(settings.llm_provider, settings)
        if gen is None:
            raise ValueError(f"NABARD_QA_LLM_PROVIDER={settings.llm_provider} but its API key is not set")
        return gen
    members = [g for p in _chain_order(settings) if (g := _make(p, settings)) is not None]
    if not members:
        log.warning("no LLM credentials found (Groq, Gemini): answering with the extractive fallback")
        return ExtractiveGenerator()
    return members[0] if len(members) == 1 else ChainGenerator(members)


def build_deep_generator(settings: Settings, default: Generator) -> Generator:
    """Generator for Deep mode: only the long-context provider.

    Its large prompt (~40 passages) would overflow small per-minute limits
    (Groq's free tier allows 8k tokens/min), so if it fails the service re-asks
    the normal chain with the top ``final_k`` passages instead of passing the
    same oversized request down the chain."""
    deep = _make(settings.deep_provider, settings) if settings.deep_provider else None
    return deep if deep is not None else default
