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
                (page.extract_text(), words(page.extract_words()), page.width, page.height)
                for page in pdf.pages
            )

    pair = call_pair(
        lambda: snapshot(real_pdfplumber.open), lambda: snapshot(compat_pdfplumber.open)
    )
    if pair is not None:
        assert pair[1] == pair[0]
