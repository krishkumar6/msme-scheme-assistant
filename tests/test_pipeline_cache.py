"""The page cache must round-trip text that contains Unicode line separators."""

from pathlib import Path

from nabard_qa.ingestion import pipeline
from nabard_qa.ingestion.pdf_extract import PageText


def test_cache_round_trips_unicode_line_separators(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    pages = [PageText(source="doc.pdf", pdf_page=1, side="full", printed_page="1",
                      text="Eligibility age 18+\x85 and \\u-like text", repaired=False)]
    monkeypatch.setattr(pipeline, "extract_pdf", lambda _p: pages)
    first = pipeline.extract_cached(pdf, tmp_path)   # writes the cache
    second = pipeline.extract_cached(pdf, tmp_path)  # reads it back
    assert first == second == pages
