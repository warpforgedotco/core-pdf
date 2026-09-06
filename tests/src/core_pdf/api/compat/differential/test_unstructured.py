from pathlib import Path

import pytest

from .support import call_pair, differential_pdfs, pdf_id

real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("unstructured"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
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
