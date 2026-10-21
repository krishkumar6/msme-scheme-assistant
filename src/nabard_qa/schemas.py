"""Shared data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str  # passage text as shown to the LLM and the user
    source: str  # file name
    report: str  # human title, e.g. "NABARD Annual Report 2017-18"
    year: str | None = None  # "2017-18"
    section: str | None = None  # nearest heading, e.g. "1.2.5 Production of Major Agricultural Crops"
    pdf_pages: list[int] = Field(default_factory=list)
    printed_pages: list[str] = Field(default_factory=list)
    kind: Literal["text", "table"] = "text"

    @property
    def embed_text(self) -> str:
        """Contextual header + body: headings and report year are prepended so
        a passage about "Table 1.2" is still findable by what the table is about."""
        header = " | ".join(x for x in (self.report, self.section) if x)
        return f"{header}\n{self.text}" if header else self.text

    @property
    def page_label(self) -> str:
        if self.printed_pages:
            pages, prefix = self.printed_pages, ""
        elif self.pdf_pages:
            pages, prefix = [str(p) for p in self.pdf_pages], "PDF "
        else:
            return ""
        return f"{prefix}p. {pages[0]}" if len(pages) == 1 else f"{prefix}pp. {pages[0]}–{pages[-1]}"

    def to_metadata(self) -> dict[str, str | int]:
        """Flat metadata for the vector store (scalar values only)."""
        return {
            "source": self.source,
            "report": self.report,
            "year": self.year or "",
            "section": self.section or "",
            "pdf_pages": ",".join(map(str, self.pdf_pages)),
            "printed_pages": ",".join(self.printed_pages),
            "kind": self.kind,
        }

    @classmethod
    def from_store(cls, id: str, text: str, meta: dict) -> Chunk:
        return cls(
            id=id,
            text=text,
            source=meta.get("source", ""),
            report=meta.get("report", ""),
            year=meta.get("year") or None,
            section=meta.get("section") or None,
            pdf_pages=[int(p) for p in str(meta.get("pdf_pages", "")).split(",") if p],
            printed_pages=[p for p in str(meta.get("printed_pages", "")).split(",") if p],
            kind=meta.get("kind", "text"),
        )


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float  # final ranking score (reranker scale when reranking is on)
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    feedback_boost: float = 0.0


class Citation(BaseModel):
    ref: str  # "S1"
    chunk_id: str
    report: str
    year: str | None
    section: str | None
    pages: str
    snippet: str
    score: float
    cited: bool  # True if the answer actually referenced it


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    history: list[Turn] = Field(default_factory=list)
    year: str | None = Field(default=None, description="Restrict to one report year, e.g. '2017-18'")
    profile: str | None = Field(
        default=None,
        max_length=600,
        description="Optional applicant profile (sector, state, category) used to personalise eligibility answers",
    )
    mode: Literal["fast", "deep"] = Field(
        default="fast",
        description="fast: ~6 passages, fastest provider. deep: ~40 passages, long-context provider - "
        "for comparisons, summaries and 'list all ...' questions",
    )


class AnswerResult(BaseModel):
    answer_id: str
    question: str
    standalone_question: str
    answer: str
    citations: list[Citation]
    grounded: bool  # False when no passage was relevant enough / nothing was cited
    provider: str
    latency_ms: int


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    irrelevant_chunk_ids: list[str] = Field(default_factory=list)
