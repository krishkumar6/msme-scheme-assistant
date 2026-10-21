from nabard_qa.ingestion.chunking import _heading, _is_toc, _is_toc_row, chunk_document
from nabard_qa.ingestion.cleaning import clean_page_text, detect_report_year, normalise, unwrap_paragraph
from nabard_qa.ingestion.glyphs import gid_to_char, glyph_name_to_char
import pymupdf

from nabard_qa.ingestion.pdf_extract import PageText, _reading_order, extract_pdf


def test_table_row_values_follow_their_label_despite_subpoint_offsets():
    # NABARD 2017-18 Table 4.1: each row's value cells start ~0.05 pt above the
    # label (different font). A strict (y0, x0) sort shifted every value up a row.
    def blk(text, x0, y0, x1, y1):  # noqa: ANN001, ANN202
        return {"text": text, "vbbox": pymupdf.Rect(x0, y0, x1, y1)}

    cells = [
        blk("i | Other states", 718.3, 467.60, 826.4, 485.6), blk("40", 844.4, 467.58, 872.7, 496.0),
        blk("20", 885.4, 467.58, 913.8, 496.0),
        blk("ii | Eastern Region", 718.3, 505.25, 836.0, 554.4), blk("45", 844.4, 505.20, 872.7, 533.6),
        blk("25", 885.4, 505.20, 913.8, 533.6),
        blk("iii | North-Eastern Region", 718.3, 563.64, 841.1, 602.4), blk("60", 844.4, 563.60, 872.8, 592.0),
        blk("45", 885.4, 563.60, 913.8, 592.0),
    ]
    order = [b["text"] for b in _reading_order(cells, 612.0, 1224.0)]
    assert order == ["i | Other states", "40", "20", "ii | Eastern Region", "45", "25",
                     "iii | North-Eastern Region", "60", "45"]


def test_sideways_ruled_table_is_read_row_by_row(tmp_path):
    # Tamil Nadu MSME Policy, Annexure I: a landscape table drawn bottom-to-top on a portrait
    # page with multi-line cells. Line-based ordering split each row's limit from its scheme.
    rows = [["No.", "Scheme", "Quantum", "Maximum"],
            ["4.1", "Capital subsidy for micro enterprises", "25% of plant and machinery value", "Rs. 25 lakhs"],
            ["4.2", "Additional capital subsidy", "Additional 10% on plant and machinery", "Rs. 5 lakhs"]]
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    col_w, row_h = 180.0, 120.0  # upright (landscape) cell size; drawn rotated onto the page
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            # upright (x, y) -> page (x = left + y, y = bottom - x): text reads bottom-to-top
            x0, y0 = 60 + r * row_h, 800 - (c + 1) * col_w
            cell = pymupdf.Rect(x0, y0, x0 + row_h, y0 + col_w)
            page.draw_rect(cell, color=(0, 0, 0), width=0.8)
            page.insert_textbox(cell + (4, 4, -4, -4), text, fontsize=9, rotate=90)
    pdf = tmp_path / "sideways.pdf"
    doc.save(pdf)
    text = extract_pdf(pdf)[0].text
    assert "4.1 | Capital subsidy for micro enterprises | 25% of plant and machinery value | Rs. 25 lakhs" in text
    assert "4.2 | Additional capital subsidy | Additional 10% on plant and machinery | Rs. 5 lakhs" in text


def test_merged_fragment_keeps_its_own_heading():
    # Gujarat GR: the short "2.2 Quantum of Capital Investment Subsidy" section (just a rate
    # table) was folded into the "1.13" chunk, and its heading - the only thing saying what the
    # rates are for - disappeared.
    from nabard_qa.ingestion.chunking import _combine
    from nabard_qa.schemas import Chunk

    a = Chunk(id="a", text="Eligibility conditions ...", source="gj.pdf", report="GJ",
              section="1 Definitions > 1.13 Eligible Investment Period", pdf_pages=[7])
    b = Chunk(id="b", text="25% of term loan amount up to INR 35 lakhs.", source="gj.pdf", report="GJ",
              section="1 Definitions > 2.2 Quantum of Capital Investment Subsidy", pdf_pages=[8])
    merged = _combine(a, b)
    assert "2.2 Quantum of Capital Investment Subsidy\n25% of term loan" in merged.text
    child = b.model_copy(update={"section": "1 Definitions > 1.13 Eligible Investment Period > (a)"})
    assert _combine(a, child).text == f"{a.text}\n\n{child.text}"  # a sub-section adds no heading


def test_gid_mapping_standard_ascii_and_known_high_glyphs():
    assert gid_to_char(68) == "a"  # /g68
    assert gid_to_char(3) == " "
    assert glyph_name_to_char("g55") == "T"
    assert gid_to_char(191) == "fi" and gid_to_char(861) == "₹" and gid_to_char(177) == "–"
    assert gid_to_char(643 + 13) == "n"  # small-caps alphabet
    assert gid_to_char(5000) is None
    assert glyph_name_to_char("a") is None  # left to the normal glyph list


def test_normalise_splits_ligatures_and_drops_private_use():
    assert normalise("reﬁnance  x") == "refinance x"


def test_normalise_restores_rupee_backtick_and_unicode_line_breaks():
    assert normalise("toolkit of upto ` 15,000 and loans of `3 lakh") == "toolkit of upto ₹15,000 and loans of ₹3 lakh"
    assert normalise("use `code` here") == "use `code` here"  # not followed by a digit
    assert normalise("Eligibility age 18") == "Eligibility\nage 18"


def test_unwrap_keeps_table_rows_and_dehyphenates():
    par = "credit to agri-\nculture rose in 2017–\n18 sharply\nKerala | 1,11,351 | 8,668"
    assert unwrap_paragraph(par) == "credit to agriculture rose in 2017–18 sharply\nKerala | 1,11,351 | 8,668"


def test_clean_page_drops_chart_noise():
    paras = clean_page_text("10\n\n8.2\n\n2013–14\n\nThe economy grew by 6.7 per cent in 2017–18.")
    assert paras == ["The economy grew by 6.7 per cent in 2017–18."]


def test_detect_report_year_prefers_most_frequent_fy():
    assert detect_report_year(["in 2017–18 and 2017-18, compared with 2016–17", "2019–21"]) == "2017-18"


def test_heading_rules():
    assert _heading("1.2.5 | Production of Major Agricultural Crops") == (3, "1.2.5 Production of Major Agricultural Crops")
    assert _heading("8. Financial Performance") == (1, "8 Financial Performance")
    assert _heading("6 Budgetary allocation for agriculture is presented in Table 1.2.") is None  # footnote
    assert _heading("9 Integrated Watershed Development Programme") is None  # table serial number


def test_toc_rows_vs_data_rows():
    assert _is_toc_row("2.4 | FPO: Size Matters | 24")
    assert _is_toc_row("Abbreviations | 201")
    assert not _is_toc_row("2 | Irrigation | 18,792 | 67 | 10,642 | 39")
    assert not _is_toc_row("Kerala | 1,11,351 | 8,668 | 1,744 | 20.1 | 7.8")
    # a two-column data table looks like TOC rows but its numbers don't run upwards
    assert not _is_toc(["State | PACS", "Assam | 56", "Bihar | 12", "Goa | 40", "Kerala | 9"])
    assert _is_toc(["1.1 | Economy | 4", "1.2 | Agriculture | 6", "1.3 | Rural | 15", "2.1 | FPOs | 20"])


def _page(text: str, n: int = 1) -> PageText:
    return PageText(source="t.pdf", pdf_page=n, side="full", printed_page=str(n), text=text)


def test_chunking_sections_tables_and_toc_filter():
    body = " ".join(["The fund supports rural roads and irrigation projects across states."] * 12)
    pages = [
        _page("Contents\n\n1.1 | State of Economy | 4\n\n1.2 | Agriculture | 6\n\n1.3 | Rural Sector | 15\n\n1.4 | Role | 17"),
        _page(f"6.2 | Rural Infrastructure Development Fund\n\n{body}\n\nTable 6.3\nSector-wise Sanctions, 2017–18\n\n"
              "Sector | Projects | Share\n\n1 | Agriculture | 1,728 | 6\n\n2 | Irrigation | 18,792 | 67", 2),
    ]
    chunks = chunk_document(pages, "NABARD Annual Report 2017-18", "2017-18")
    texts = " ".join(c.text for c in chunks)
    assert "State of Economy" not in texts  # table of contents removed
    table = next(c for c in chunks if "18,792" in c.text)
    assert table.kind == "table"
    assert "Irrigation | 18,792 | 67" in table.text
    assert "Table 6.3" in (table.section or "") or "Table 6.3" in table.text
    assert all(c.section and c.section.startswith("6.2") for c in chunks)
    assert table.printed_pages == ["2"]


def test_chunk_ids_are_content_hashes():
    pages = [_page("6.2 | Funds\n\n" + "Loans were sanctioned to states for irrigation and roads. " * 20)]
    a = chunk_document(pages, "R", "2017-18")
    b = chunk_document(pages, "R", "2017-18")
    assert [c.id for c in a] == [c.id for c in b]
