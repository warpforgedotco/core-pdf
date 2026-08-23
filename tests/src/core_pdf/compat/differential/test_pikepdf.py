from contextlib import ExitStack
from pathlib import Path

import pytest

from .support import differential_pdfs, metadata, open_pair, pdf_id

real_pikepdf = pytest.importorskip("pikepdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("pikepdf"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat import pikepdf as compat_pikepdf

    with ExitStack() as stack:
        pair = open_pair(
            stack,
            lambda: real_pikepdf.Pdf.open(pdf_path),
            lambda: compat_pikepdf.Pdf.open(pdf_path),
        )
        if pair is None:
            return
        expected, actual = pair
        assert len(actual.pages) == len(expected.pages)
        assert [tuple(page.mediabox) for page in actual.pages] == [
            tuple(page.mediabox) for page in expected.pages
        ]
        assert metadata(actual.docinfo) == metadata(expected.docinfo)
