from pathlib import Path
from typing import Any

import pytest

from .support import ALL_PDFS, call_pair


@pytest.mark.skip(reason="PyMuPDF differential tests disabled")
@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pymupdf = pytest.importorskip("pymupdf")
    from core_pdf.api.compat._unsupported import pymupdf as compat_pymupdf

    def snapshot(open_document: Any) -> tuple[tuple[str, list[Any], tuple[float, ...]], ...]:
        with open_document(pdf_path) as document:
            return tuple(
                (page.get_text(), page.get_text("words"), tuple(page.rect)) for page in document
            )

    pair = call_pair(lambda: snapshot(real_pymupdf.open), lambda: snapshot(compat_pymupdf.open))
    if pair is not None:
        assert pair[1] == pair[0]
