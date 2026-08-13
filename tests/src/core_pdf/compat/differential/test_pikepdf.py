from contextlib import ExitStack
from pathlib import Path

import pytest

from .support import ALL_PDFS, metadata, open_pair


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_pikepdf = pytest.importorskip("pikepdf")
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
