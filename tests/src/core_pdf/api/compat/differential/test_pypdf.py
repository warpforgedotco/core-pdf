from pathlib import Path
from typing import Any

import pytest

from .support import call_pair, differential_pdfs, pdf_id

real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("pypdf"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat import pypdf as compat_pypdf

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
