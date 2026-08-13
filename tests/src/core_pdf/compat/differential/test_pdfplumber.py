from pathlib import Path
from typing import Any

import pytest

from .support import ALL_PDFS, call_pair, words


@pytest.mark.skip(reason="pdfplumber differential tests disabled")
@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pdfplumber = pytest.importorskip("pdfplumber")
    from core_pdf.api.compat._unsupported import pdfplumber as compat_pdfplumber

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
