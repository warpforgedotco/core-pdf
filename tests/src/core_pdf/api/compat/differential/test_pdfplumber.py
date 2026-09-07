from pathlib import Path
from typing import Any

import pytest

from .support import call_pair, differential_pdfs, pdf_id, words

real_pdfplumber = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("pdfplumber"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    def snapshot(open_pdf: Any) -> tuple[tuple[str, list[dict[str, Any]], float, float], ...]:
        with open_pdf(pdf_path) as pdf:
            return tuple(
                (page.extract_text(), page.extract_words(), page.width, page.height)
                for page in pdf.pages
            )

    pair = call_pair(
        lambda: snapshot(real_pdfplumber.open), lambda: snapshot(compat_pdfplumber.open)
    )
    if pair is not None:
        expected_pages, actual_pages = pair
        assert len(actual_pages) == len(expected_pages)
        for actual, expected in zip(actual_pages, expected_pages, strict=True):
            assert actual[0] == expected[0]
            assert actual[1] == words(expected[1])
            assert actual[2:] == expected[2:]
