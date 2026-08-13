from pathlib import Path

import pytest

from .support import ALL_PDFS, call_pair


@pytest.mark.skip(reason="unstructured differential tests disabled")
@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_library_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
    from core_pdf.api.compat.unstructured import partition_pdf as compat_partition

    pair = call_pair(
        lambda: real_partition(filename=str(pdf_path), strategy="fast"),
        lambda: compat_partition(pdf_path),
    )
    if pair is None:
        return
    expected, actual = pair
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]
