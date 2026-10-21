"""Ingestion: PDFs -> pages -> chunks -> embeddings -> vector store.

Idempotent: chunk ids are content hashes, so re-running only embeds new or
changed passages and deletes passages that no longer exist. Page extraction is
cached per file hash and extractor version because the glyph repair pass is the slow step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Settings
from ..retrieval.embeddings import Embedder
from ..retrieval.vector_store import ChromaStore
from ..schemas import Chunk
from .chunking import chunk_document
from .cleaning import detect_report_year
from .pdf_extract import EXTRACTOR_VERSION, PageText, extract_pdf

log = logging.getLogger(__name__)
SUPPORTED = {".pdf"}


@dataclass
class IngestReport:
    documents: int
    pages: int
    chunks: int
    embedded: int
    deleted: int
    seconds: float


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _load_manifest(raw_dir: Path) -> dict[str, dict]:
    """Optional ``data/raw/manifest.json``: {"file.pdf": {"title": ..., "year": ...}}."""
    path = raw_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def extract_cached(path: Path, cache_dir: Path) -> list[PageText]:
    cache = cache_dir / f"pages_{path.stem[:40]}_{_file_hash(path)}_v{EXTRACTOR_VERSION}.jsonl"
    if cache.exists():
        # split on "\n" only: str.splitlines() also breaks on U+2028/U+0085, which PDF
        # text can contain and json.dumps(ensure_ascii=False) leaves unescaped
        return [PageText(**json.loads(line)) for line in cache.read_text(encoding="utf-8").split("\n") if line]
    t = time.time()
    pages = extract_pdf(path)
    cache.write_text("\n".join(json.dumps(asdict(p), ensure_ascii=False) for p in pages), encoding="utf-8")
    log.info("extracted %s in %.1fs", path.name, time.time() - t)
    return pages


def build_chunks(paths: list[Path], settings: Settings) -> tuple[list[Chunk], int]:
    manifest = _load_manifest(settings.raw_dir)
    chunks: list[Chunk] = []
    n_pages = 0
    for path in paths:
        pages = extract_cached(path, settings.processed_dir)
        n_pages += len(pages)
        meta = manifest.get(path.name, {})
        year = meta.get("year") or detect_report_year([p.text for p in pages])
        report = meta.get("title") or (f"NABARD Annual Report {year}" if year else path.stem)
        doc_chunks = chunk_document(
            pages, report, year,
            target_words=settings.chunk_target_words,
            max_words=settings.chunk_max_words,
            overlap_words=settings.chunk_overlap_words,
        )
        log.info("%s -> %s (%s): %d pages, %d chunks", path.name, report, year, len(pages), len(doc_chunks))
        chunks.extend(doc_chunks)
    # identical passages across documents (boilerplate) collapse to one id
    unique = {c.id: c for c in chunks}
    return list(unique.values()), n_pages


def ingest(settings: Settings, embedder: Embedder, store: ChromaStore, paths: list[Path] | None = None) -> IngestReport:
    t0 = time.time()
    paths = paths or sorted(p for p in settings.raw_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not paths:
        raise FileNotFoundError(f"no PDFs found in {settings.raw_dir}")
    chunks, n_pages = build_chunks(paths, settings)

    settings.chunks_path.write_text(
        "\n".join(c.model_dump_json() for c in chunks), encoding="utf-8"
    )

    stored = store.metadatas()
    existing = set(stored)
    wanted = {c.id for c in chunks}
    # ids hash the text only, so a manifest change (title, year) keeps the id: re-embed
    # those too, or year filters and citations keep the old labels
    new = [c for c in chunks if c.id not in existing or stored[c.id] != c.to_metadata()]
    stale = existing - wanted
    if stale:
        store.delete(stale)
    for i in range(0, len(new), 256):
        part = new[i : i + 256]
        store.upsert(part, embedder.embed_documents([c.embed_text for c in part]))
        log.info("embedded %d/%d", min(i + 256, len(new)), len(new))
    report = IngestReport(len(paths), n_pages, len(chunks), len(new), len(stale), round(time.time() - t0, 1))
    log.info("ingest done: %s", report)
    return report


def load_chunks(settings: Settings) -> list[Chunk]:
    if not settings.chunks_path.exists():
        return []
    return [Chunk.model_validate_json(line) for line in settings.chunks_path.read_text(encoding="utf-8").split("\n") if line]
