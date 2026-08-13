from pathlib import Path
from typing import Any

import pytest

from .support import ALL_PDFS, call_pair


@pytest.mark.skip(reason="pypdf differential tests disabled")
@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pypdf = pytest.importorskip("pypdf")
    from core_pdf.api.compat._unsupported import pypdf as compat_pypdf

    def snapshot(reader_type: Any) -> tuple[tuple[str, tuple[float, ...], int], ...]:
        with reader_type(pdf_path, strict=False) as reader:
            return tuple(
                (page.extract_text(), tuple(page.mediabox), page.rotation) for page in reader.pages
            )

    pair = call_pair(
        lambda: snapshot(real_pypdf.PdfReader), lambda: snapshot(compat_pypdf.PdfReader)
    )
    if pair is not None:
        assert pair[1] == pair[0]
