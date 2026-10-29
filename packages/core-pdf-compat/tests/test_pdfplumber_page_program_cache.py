from pathlib import Path

import core_pdf_compat.pdfplumber
from core_pdf_compat.pdfminer import _capture, _projection

FIXTURE = (
    Path(__file__).parents[3] / "tests/fixtures/pdfplumber/tests/pdfs/issue-33-lorem-ipsum.pdf"
)


def test_chars_and_layout_share_one_pdfminer_text_walk_until_flushed(monkeypatch):
    calls = []
    walk = _capture.pdfminer_page_program

    def counting_walk(page):
        calls.append(page.page_number)
        return walk(page)

    monkeypatch.setattr(_capture, "pdfminer_page_program", counting_walk)
    monkeypatch.setattr(_projection, "pdfminer_page_program", counting_walk)
    with core_pdf_compat.pdfplumber.open(FIXTURE) as pdf:
        page = pdf.pages[0]
        chars = page.chars
        layout = page.layout
        assert chars
        assert layout is not None
        assert page.crop((0, 0, page.width / 2, page.height / 2)).chars
        assert calls == [1]

        page.flush_cache()
        assert page.chars == chars
        assert calls == [1, 1]
