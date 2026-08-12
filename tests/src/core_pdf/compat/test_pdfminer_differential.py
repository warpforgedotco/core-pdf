from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pdfminer.high_level")

from pdfminer.high_level import extract_pages as pdfminer_extract_pages
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams as PdfminerLAParams
from pdfminer.layout import LTTextContainer as PdfminerLTTextContainer

from core_pdf.api.compat.pdfminer import LAParams, LTTextContainer, extract_pages, extract_text

SAMPLES = Path("tests/fixtures/pdfminer.six/samples")
HIGH_LEVEL_TEST_PDFS = (
    "simple1.pdf",
    "simple2.pdf",
    "simple3.pdf",
    "simple4.pdf",
    "simple5.pdf",
    "zen_of_python_corrupted.pdf",
    "contrib/issue_495_pdfobjref.pdf",
    "contrib/issue_566_test_1.pdf",
    "contrib/issue_566_test_2.pdf",
    "contrib/issue-625-identity-cmap.pdf",
    "contrib/issue-791-non-unicode-cmap.pdf",
    "contrib/issue-886-xref-stream-widths.pdf",
)


@pytest.mark.parametrize("relative_path", HIGH_LEVEL_TEST_PDFS)
def test_extract_text_matches_pdfminer_on_its_high_level_test_pdfs(relative_path: str) -> None:
    pdf_path = SAMPLES / relative_path

    assert extract_text(pdf_path) == pdfminer_extract_text(pdf_path)


@pytest.mark.parametrize("line_margin", [0.19, 0.21])
def test_extract_pages_matches_pdfminer_layout_on_its_line_margin_pdf(
    line_margin: float,
) -> None:
    pdf_path = SAMPLES / "simple4.pdf"
    actual_pages = list(extract_pages(pdf_path, laparams=LAParams(line_margin=line_margin)))
    expected_pages = list(
        pdfminer_extract_pages(pdf_path, laparams=PdfminerLAParams(line_margin=line_margin))
    )

    assert _page_text_containers(actual_pages) == _page_text_containers(expected_pages)


def _page_text_containers(pages: list[Any]) -> list[list[str]]:
    container_type = (
        LTTextContainer
        if pages and pages[0].__class__.__module__.startswith("core_pdf")
        else PdfminerLTTextContainer
    )
    return [
        [item.get_text() for item in page if isinstance(item, container_type)] for page in pages
    ]
