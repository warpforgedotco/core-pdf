from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import call_pair, differential_pdfs, open_pair, pdf_id
from .test_pymupdf_geometry import internal_geometry_expected

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


def internal_page_snapshot(document: Any, index: int) -> dict[str, Any]:
    page = document[index]
    return {
        **{
            name: tuple(getattr(page, name))
            for name in ("mediabox", "cropbox", "bleedbox", "trimbox", "artbox", "rect")
        },
        "rotation": page.rotation,
        **{kind: page.get_text(kind) for kind in ("text", "words", "blocks", "rawdict")},
    }


@pytest.mark.parametrize("pdf_path", differential_pdfs(), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    with ExitStack() as stack:
        pair = open_pair(
            stack,
            lambda: real_pymupdf.open(pdf_path),
            lambda: compat_pymupdf.open(pdf_path),
        )
        if pair is None:
            return
        expected, actual = pair
        assert actual.page_count == expected.page_count
        assert actual.metadata == expected.metadata
        for index in range(expected.page_count):
            page_pair = call_pair(
                lambda index=index: internal_page_snapshot(expected, index),
                lambda index=index: internal_page_snapshot(actual, index),
            )
            if page_pair is not None:
                assert page_pair[1] == internal_geometry_expected(page_pair[0]), (
                    f"page {index + 1} of {pdf_id(pdf_path)}"
                )
