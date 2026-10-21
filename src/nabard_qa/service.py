"""Question-answering service: the one place that wires the pipeline together."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .feedback.store import FeedbackStore
from .generation.llm import (
    ExtractiveGenerator,
    GenerationError,
    GenerationRequest,
    Generator,
    build_deep_generator,
    build_generator,
)
from .generation.prompts import NO_SOURCES_ANSWER, REWRITE_SYSTEM, SEARCH_QUERY_SYSTEM, SYSTEM_PROMPT, build_messages
from .retrieval.adapter import QueryAdapter
from .retrieval.embeddings import Embedder
from .retrieval.hybrid import HybridRetriever
from .retrieval.lexical import BM25Index
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.vector_store import ChromaStore
from .schemas import AnswerResult, AskRequest, Citation, ScoredChunk, Turn

log = logging.getLogger(__name__)
_CITE = re.compile(r"\[S(\d{1,2})\]")
# Models differ in citation style: 【S1】 (gpt-oss), ［S1］, [S1, S3], [S1; S2], [ S1 ]
_CITE_GROUP = re.compile(r"[\[【［]\s*(S\d{1,2}(?:\s*[,;]\s*S?\d{1,2})*)\s*[\]】］]")


_ZERO_WIDTH = re.compile(r"[​-‍⁠﻿]")


def normalise_citations(text: str) -> str:
    """Rewrite any bracketed source reference to the canonical ``[S1][S3]`` form."""

    def repl(m: re.Match[str]) -> str:
        refs = re.findall(r"\d{1,2}", m.group(1))
        return "".join(f"[S{r}]" for r in refs)

    # gpt-oss sometimes writes "[​S3]" (zero-width space inside the bracket)
    return _CITE_GROUP.sub(repl, _ZERO_WIDTH.sub("", text))
_FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|their|them|same|above|previous|also|more)\b", re.I)

# Personalised search: "what subsidy can I get?" from a Tamil Nadu applicant should reach the
# Tamil Nadu policy. Only for personal questions that don't already name a central scheme.
STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep",
    "Puducherry",
)
_PERSONAL = re.compile(r"\b(i|i'm|me|my|mine|we|our|eligible|eligibility|qualify|apply)\b", re.I)
_CENTRAL_SCHEME = re.compile(r"\b(pmegp|cgtmse|vishwakarma|mudra|pmmy|stand[- ]?up|mse-?cdp|nabard)\b", re.I)


def profile_state(profile: str | None) -> str | None:
    if not profile:
        return None
    return next((s for s in STATES if re.search(rf"\b{re.escape(s)}\b", profile, re.I)), None)


def personalised_query(question: str, profile: str | None) -> str:
    """Add the applicant's state to the search query for personal, scheme-agnostic questions."""
    state = profile_state(profile)
    if (not state or not _PERSONAL.search(question) or _CENTRAL_SCHEME.search(question)
            or any(s.lower() in question.lower() for s in STATES)):
        return question
    return f"{question} {state} state MSME scheme"


class _NoSourcesGenerator:
    """Fixed reply when no passage is relevant. Calling the LLM with an empty
    context invites figures from memory (e.g. a wrong PMEGP subsidy rate)."""

    name = "no-sources"

    def stream(self, req: GenerationRequest) -> Iterator[str]:  # noqa: ARG002
        yield NO_SOURCES_ANSWER

    def complete(self, req: GenerationRequest) -> str:  # noqa: ARG002
        return NO_SOURCES_ANSWER


@dataclass
class _Prepared:
    request: AskRequest
    standalone: str
    sources: list[ScoredChunk]
    query_embedding: np.ndarray
    gen_request: GenerationRequest
    generator: Generator
    started: float
    # Deep mode only: (normal generator, request trimmed to final_k passages)
    fallback: tuple[Generator, GenerationRequest] | None = None


class QAService:
    def __init__(self, settings: Settings, retriever: HybridRetriever, generator: Generator,
                 feedback: FeedbackStore, store: ChromaStore, deep_generator: Generator | None = None) -> None:
        self.s = settings
        self.retriever = retriever
        self.generator = generator
        self.deep_generator = deep_generator or generator
        self.feedback = feedback
        self.store = store
        self._extractive = ExtractiveGenerator()

    # ------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings: Settings, generator: Generator | None = None) -> QAService:
        embedder = Embedder(settings.embedding_model, settings.model_cache_dir, settings.query_instruction)
        store = ChromaStore(settings.index_dir, settings.collection_name)
        chunks, _ = store.all()
        if not chunks:
            log.warning("vector store is empty - run `nabard-qa ingest` first")
        reranker = None
        if settings.reranker_model:
            try:
                reranker = CrossEncoderReranker(settings.reranker_model, settings.model_cache_dir,
                                                settings.reranker_quantized, settings.rerank_max_words)
            except Exception as e:  # noqa: BLE001 - reranking is an optimisation, not a dependency
                log.warning("reranker unavailable (%s); continuing with fused ranking", e)
        feedback = FeedbackStore(settings.db_path)
        adapter = QueryAdapter.load(settings.adapter_path) if settings.adapter_enabled else QueryAdapter()
        retriever = HybridRetriever(settings, embedder, store, BM25Index(chunks), reranker, adapter, feedback)
        generator = generator or build_generator(settings)
        deep = build_deep_generator(settings, generator)
        log.info("generators: fast=%s deep=%s", generator.name, deep.name)
        return cls(settings, retriever, generator, feedback, store, deep)

    def reload_adapter(self) -> bool:
        self.retriever.adapter = QueryAdapter.load(self.s.adapter_path)
        return self.retriever.adapter.active

    # ---------------------------------------------------------- pipeline
    def standalone_question(self, question: str, history: list[Turn]) -> str:
        """Resolve follow-ups ("what about its interest rate?") into a search query."""
        if not history:
            return question
        last_user = next((t.content for t in reversed(history) if t.role == "user"), "")
        looks_follow_up = len(question.split()) <= 8 or bool(_FOLLOW_UP.search(question))
        if not looks_follow_up or not last_user:
            return question
        if isinstance(self.generator, ExtractiveGenerator):
            return f"{last_user} {question}"
        convo = "\n".join(f"{t.role}: {t.content[:600]}" for t in history[-2 * self.s.max_history_turns :])
        req = GenerationRequest(
            system=REWRITE_SYSTEM,
            messages=[{"role": "user", "content": f"Conversation:\n{convo}\n\nLatest message: {question}"}],
            question=question,
            max_tokens=200,
            effort="low",
        )
        try:
            rewritten = self.generator.complete(req).strip().splitlines()[0].strip()
            return rewritten or question
        except (GenerationError, IndexError) as e:
            log.warning("query rewrite failed (%s); using heuristic", e)
            return f"{last_user} {question}"

    def condensed_query(self, question: str) -> str | None:
        """Short keyword-style search query for a long, conversational question."""
        if isinstance(self.generator, ExtractiveGenerator):
            return None
        req = GenerationRequest(system=SEARCH_QUERY_SYSTEM, messages=[{"role": "user", "content": question}],
                                # reasoning models (gpt-oss) spend tokens thinking before the query
                                question=question, max_tokens=400, effort="low")
        try:
            out = self.generator.complete(req).strip().split("\n")[0].strip().strip('"')
        except GenerationError as e:
            log.warning("query condensing failed (%s)", e)
            return None
        return out if out and out.lower() != question.lower() else None

    def _gate(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Question-level gate on the best passage, then a per-passage floor.
        (rerank_score excludes feedback/year boosts, so feedback can't force junk in.)"""
        if not chunks or chunks[0].rerank_score is None:
            return chunks
        if max(sc.rerank_score or float("-inf") for sc in chunks) < self.s.relevance_gate:
            return []
        return [sc for sc in chunks if (sc.rerank_score or 0) >= self.s.min_rerank_score]

    def _prepare(self, req: AskRequest) -> _Prepared:
        t0 = time.perf_counter()
        question = req.question.strip()[: self.s.max_question_chars]
        standalone = self.standalone_question(question, req.history)
        deep = req.mode == "deep"

        def retrieve(query: str):  # noqa: ANN202
            return self.retriever.retrieve(
                query, year=req.year,
                k=self.s.deep_k if deep else None,
                candidates=self.s.deep_candidates if deep else None,
            )

        search = personalised_query(standalone, req.profile)
        if search != standalone:
            log.info("search personalised to %r", search)
        result = retrieve(search)
        sources = self._gate(result.chunks)
        if not sources and result.chunks and len(search.split()) > 8:
            # A long personal question ("I am a 24 year old woman ... PMEGP?") can retrieve
            # the right passages yet fail the gate; retry once with a condensed query.
            condensed = self.condensed_query(search)
            if condensed:
                retry = retrieve(condensed)
                if retry_sources := self._gate(retry.chunks):
                    log.info("gate passed after condensing the query to %r", condensed)
                    standalone, result, sources = condensed, retry, retry_sources

        def request(srcs: list[ScoredChunk]) -> GenerationRequest:
            return GenerationRequest(
                system=SYSTEM_PROMPT,
                messages=build_messages(question, srcs, req.history, req.profile, self.s.max_history_turns),
                question=standalone,
                sources=srcs,
            )

        gen = request(sources)
        if not sources:
            return _Prepared(req, standalone, sources, result.query_embedding, gen, _NoSourcesGenerator(), t0)
        if deep and self.deep_generator is not self.generator:
            # top passages keep their S1..Sk labels, so citations stay valid on fallback
            fallback = (self.generator, request(sources[: self.s.final_k]))
            return _Prepared(req, standalone, sources, result.query_embedding, gen, self.deep_generator, t0, fallback)
        return _Prepared(req, standalone, sources, result.query_embedding, gen, self.generator, t0)

    def _finish(self, p: _Prepared, answer: str, provider: str) -> AnswerResult:
        answer = normalise_citations(answer)
        cited = {int(n) for n in _CITE.findall(answer)}
        citations = [
            Citation(
                ref=f"S{i}",
                chunk_id=sc.chunk.id,
                report=sc.chunk.report,
                year=sc.chunk.year,
                section=sc.chunk.section,
                pages=sc.chunk.page_label,
                snippet=sc.chunk.text[:600],
                score=round(sc.score, 3),
                cited=i in cited,
            )
            for i, sc in enumerate(p.sources, start=1)
        ]
        grounded = bool(cited & set(range(1, len(p.sources) + 1)))
        latency = int((time.perf_counter() - p.started) * 1000)
        answer_id = self.feedback.log_interaction(
            question=p.request.question,
            standalone_question=p.standalone,
            answer=answer,
            provider=provider,
            grounded=grounded,
            latency_ms=latency,
            retrieved=[{"chunk_id": c.chunk_id, "ref": c.ref, "score": c.score, "cited": c.cited} for c in citations],
            query_embedding=p.query_embedding,
        )
        return AnswerResult(
            answer_id=answer_id,
            question=p.request.question,
            standalone_question=p.standalone,
            answer=answer,
            citations=citations,
            grounded=grounded,
            provider=provider,
            latency_ms=latency,
        )

    @staticmethod
    def _pairs(gen: Generator, req: GenerationRequest) -> Iterator[tuple[str, str]]:
        # chains report which member actually answered
        if hasattr(gen, "stream_with_provider"):
            return gen.stream_with_provider(req)
        return ((gen.name, d) for d in gen.stream(req))

    def _generate(self, p: _Prepared) -> Iterator[tuple[str, str]]:
        """Yield (provider, text delta).

        Order of attempts: the chosen generator; in Deep mode, the normal chain
        with only the top passages; finally extractive quotes - so the user
        always gets something grounded."""
        attempts = [(p.generator, p.gen_request)] + ([p.fallback] if p.fallback else [])
        for i, (gen, req) in enumerate(attempts):
            produced = False
            try:
                for provider, delta in self._pairs(gen, req):
                    produced = True
                    yield provider, delta
                return
            except GenerationError as e:
                if produced:
                    log.error("generation interrupted: %s", e)
                    yield self._extractive.name, "\n\n(The answer was interrupted.)"
                    return
                more = i + 1 < len(attempts)
                log.log(logging.WARNING if more else logging.ERROR, "generation failed (%s): %s",
                        "retrying with fewer passages" if more else "falling back to extracts", e)
        yield self._extractive.name, "The language model is unavailable right now, so here are the most relevant extracts.\n\n"
        for delta in self._extractive.stream(p.gen_request):
            yield self._extractive.name, delta

    # --------------------------------------------------------------- api
    def answer(self, req: AskRequest) -> AnswerResult:
        p = self._prepare(req)
        provider, parts = self.generator.name, []
        for provider, delta in self._generate(p):
            parts.append(delta)
        return self._finish(p, "".join(parts).strip(), provider)

    def stream(self, req: AskRequest) -> Iterator[dict]:
        """Server-sent-event payloads: sources -> delta* -> done."""
        p = self._prepare(req)
        yield {
            "type": "sources",
            "model": p.generator.name,
            "standalone_question": p.standalone,
            "sources": [
                {"ref": f"S{i}", "chunk_id": sc.chunk.id, "report": sc.chunk.report, "pages": sc.chunk.page_label,
                 "section": sc.chunk.section, "snippet": sc.chunk.text[:600], "score": round(sc.score, 3)}
                for i, sc in enumerate(p.sources, start=1)
            ],
        }
        provider, parts = self.generator.name, []
        for provider, delta in self._generate(p):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
        result = self._finish(p, "".join(parts).strip(), provider)
        yield {"type": "done", "result": result.model_dump()}
