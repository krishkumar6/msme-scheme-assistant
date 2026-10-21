"""Central configuration. Every value can be overridden with an environment
variable prefixed ``NABARD_QA_`` (or in a ``.env`` file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NABARD_QA_", env_file=".env", extra="ignore")

    # ---- paths --------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    model_cache_dir: Path = PROJECT_ROOT / "data" / "models"
    db_path: Path = PROJECT_ROOT / "data" / "app.db"
    adapter_path: Path = PROJECT_ROOT / "data" / "adapter" / "query_adapter.npz"

    # ---- chunking -------------------------------------------------------------
    chunk_target_words: int = 220
    chunk_max_words: int = 360
    chunk_overlap_words: int = 40

    # ---- retrieval ------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    collection_name: str = "nabard_chunks"
    dense_k: int = 30
    lexical_k: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    reranker_model: str | None = "Xenova/ms-marco-MiniLM-L-6-v2"
    # int8 ONNX + the first 128 words of each passage: 2.4x faster reranking than fp32 / full passages,
    # with golden-set hit@6 unchanged (0.96) and hit@1 0.54 -> 0.60. 96 words drops hit@6 to 0.86.
    reranker_quantized: bool = True
    rerank_max_words: int = 128  # 0 = let the model read its full 512 tokens
    # The cross-encoder is the slowest retrieval stage on CPU (~0.12 s/passage here).
    # 15 vs 30 candidates: identical golden-set retrieval, 1.35 s vs 3.6 s.
    rerank_candidates: int = 15
    final_k: int = 6
    # Calibrated on eval/golden_set.jsonl with scripts/calibrate_gate.py: answerable
    # questions had a top reranker score >= 2.0, unanswerable ones <= -0.74.
    relevance_gate: float = 0.0  # top passage below this -> treat question as out of scope
    min_rerank_score: float = -4.0  # per-passage floor for supporting context

    # ---- feedback loop --------------------------------------------------------
    feedback_similarity: float = 0.82  # min cosine between queries to reuse feedback
    feedback_boost: float = 0.6  # reranker-score units per unit of (similar) feedback
    feedback_recall_k: int = 5  # max extra candidates recalled from positive feedback on similar questions
    adapter_enabled: bool = True
    adapter_min_examples: int = 20

    # ---- generation -----------------------------------------------------------
    # auto = every provider you have a key for, tried in `llm_chain` order, then extracts.
    llm_provider: Literal["auto", "groq", "gemini", "local", "extractive"] = "auto"
    llm_chain: str = "groq,gemini"  # fallback order for "auto"; add ",local" for the fine-tuned model
    # "Deep" mode: broad questions get many more passages, sent to the long-context model first
    deep_k: int = 40
    deep_candidates: int = 40
    deep_provider: str = "gemini"
    max_tokens: int = 4096  # answer length cap for Groq and the local model
    # Groq (OpenAI-compatible). The key is read from GROQ_API_KEY or NABARD_QA_GROQ_API_KEY.
    groq_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "NABARD_QA_GROQ_API_KEY")
    )
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"  # list yours: python scripts/list_models.py groq
    groq_reasoning_effort: str | None = None  # gpt-oss: low | medium | high (None = provider default)
    groq_timeout_s: float = 30.0  # max silence (no bytes) before failing over
    # Gemini via its OpenAI-compatible endpoint. Key from GEMINI_API_KEY / GOOGLE_API_KEY.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "NABARD_QA_GEMINI_API_KEY"),
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    # gemini-2.5-* is listed but closed to new accounts; list yours: python scripts/list_models.py gemini
    # Measured 2026-10-20 (scripts/probe_provider.py, ~8k-token prompt, time to first byte):
    # 3.6-flash 4.5 s, 3.5-flash 8.9 s, 3.8-flash 46 s + 503s/timeouts, flash-latest 76 s.
    gemini_model: str = "gemini-3.6-flash"
    # Gemini models think by default and thinking tokens count against max_tokens
    gemini_reasoning_effort: str | None = "low"
    gemini_max_tokens: int = 8192
    # Gemini sends nothing while it thinks; a healthy 40-passage answer started in ~16 s,
    # one hung for 120 s. Fail over after this much silence.
    gemini_timeout_s: float = 60.0
    # Any OpenAI-compatible server (e.g. vLLM) hosting the LoRA fine-tuned model
    local_base_url: str = "http://localhost:8001/v1"
    local_model: str = "nabard-qa-lora"
    local_api_key: str | None = None
    max_history_turns: int = 4

    # ---- API ------------------------------------------------------------------
    max_question_chars: int = 1000
    admin_token: str | None = Field(default=None, description="Bearer token for /api/admin/*")
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.index_dir, self.model_cache_dir,
                  self.adapter_path.parent, self.db_path.parent):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def pages_path(self) -> Path:
        return self.processed_dir / "pages.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
