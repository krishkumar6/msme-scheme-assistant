"""Integration test against the real 2017-18 report (skipped if it's not present).

Page 14 of that PDF is a rotated two-page spread set in a Georgia subset whose
ToUnicode map is broken; plain PyMuPDF/pypdf output is a letter-substitution
cipher there ("rìê aüèîÜü êé†ü†û∞"). This pins the glyph repair."""

from pathlib import Path

import pytest

from nabard_qa.config import PROJECT_ROOT

PDF = PROJECT_ROOT / "data" / "raw" / "NABARD_Annual_Report_2017-18.pdf"


@pytest.fixture(scope="module")
def pages():
    if not PDF.exists():
        pytest.skip("report PDF not available")
    from nabard_qa.ingestion.pdf_extract import extract_pdf

    return extract_pdf(Path(PDF))


def test_glyph_repair_on_rotated_spread(pages):
    left = next(p for p in pages if p.pdf_page == 14 and p.side == "left")
    assert left.repaired
    assert "The Indian economy recorded a growth rate of 6.7" in left.text.replace("\n", " ")


def test_tables_keep_rows_and_rupee_sign(pages):
    p = next(p for p in pages if p.pdf_page == 17 and p.side == "right")
    assert "Kerala | 1,11,351 | 8,668 | 1,744 | 20.1 | 7.8" in p.text
    assert "₹ crore" in p.text


def test_printed_page_labels_detected(pages):
    labelled = sum(1 for p in pages if p.printed_page)
    assert labelled / len(pages) > 0.8
