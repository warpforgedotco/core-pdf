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


def test_only_the_most_recently_walked_page_keeps_its_pdfminer_capture(monkeypatch):
    calls = []
    walk = _capture.pdfminer_page_program

    def counting_walk(page):
        calls.append(page.page_number)
        return walk(page)

    monkeypatch.setattr(_capture, "pdfminer_page_program", counting_walk)
    monkeypatch.setattr(_projection, "pdfminer_page_program", counting_walk)
    with core_pdf_compat.pdfplumber.open(FIXTURE) as pdf:
        first, second = pdf.pages
        assert first.chars
        assert second.chars
        assert first._adapter._pdfminer_program is None
        assert second.layout is not None
        assert calls == [1, 2]
        assert first.layout is not None
        assert calls == [1, 2, 1]
        assert second._adapter._pdfminer_program is None


def test_layout_of_a_password_opened_pdf_reads_the_open_document(monkeypatch):
    import core_pdf_compat.pdfminer

    def reopen(*args, **kwargs):
        raise AssertionError("Page.layout reopened the source")

    monkeypatch.setattr(core_pdf_compat.pdfminer, "extract_pages", reopen)
    with core_pdf_compat.pdfplumber.open(
        FIXTURE.with_name("password-example.pdf"), password="test"
    ) as pdf:
        layouts = [page.layout for page in pdf.pages]
    assert [layout.pageid for layout in layouts] == [1, 2, 3, 4]
    assert "Backup4all" in "".join(
        item.get_text() for item in layouts[0] if hasattr(item, "get_text")
    )
