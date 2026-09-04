# SPDX-License-Identifier: AGPL-3.0-only
"""Fixture pages shared by the performance benchmarks.

Every document named here interprets without OCR, which the benchmarks depend
on twice over. CodSpeed counts instructions on a simulated CPU, so a Tesseract
call would swamp the measurement with work this project does not own, and the
recognizer's output is not deterministic across versions of the native library.
``plan_page`` returning no OCR passes is therefore part of what
``assert_native_only`` pins, not an incidental property of the corpus.

The pages were chosen from a profile of ``PdfDocument.extract`` over the
SCORE-Bench corpus, one per shape that moves independently:

``TEXT_PDF``
    A single text page, no drawings. The cheapest realistic interpretation.
``VECTOR_PDF`` page ``VECTOR_PAGE_INDEX``
    494 drawings and 411 ruled lines, where path construction rather than
    glyph handling dominates. Later pages of the same document do reach OCR,
    so the index matters.
``DENSE_PDF``
    3596 glyphs, 136 drawings and 363 lines on one page: the heaviest
    native-text page in the corpus, and the one that exercises the table and
    layout stages hardest.
``MIXED_PDF``
    An academic page with a table, at a size that keeps the end-to-end
    benchmark to a few hundred milliseconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument
from tests.helpers.paths import FIXTURES, SCORE_BENCH, require_fixture

PYPDF_RESOURCES = FIXTURES / "pypdf" / "resources"

TEXT_PDF = PYPDF_RESOURCES / "crazyones.pdf"
VECTOR_PDF = PYPDF_RESOURCES / "GeoBase_NHNC1_Data_Model_UML_EN.pdf"
VECTOR_PAGE_INDEX = 2
DENSE_PDF = SCORE_BENCH / "EPD-p004.pdf"
MIXED_PDF = SCORE_BENCH / "USDC-compression-vit-2310.11117-p7-p007.pdf"


@contextmanager
def opened_page(path: Path, index: int = 0) -> Iterator[Any]:
    """Yield one page of ``path`` with its content already interpreted.

    Interpreting here rather than inside the benchmark keeps the stream decode,
    font loading and decoder caches out of the measurement, so a benchmark that
    takes the page measures only its own stage.
    """
    with PdfDocument.open(require_fixture(path)) as document:
        page = document.pages[index]
        page.get_page_program()
        yield page


def reinterpret(page: Any) -> Any:
    """Interpret ``page``'s content streams again."""
    return page.get_page_program()


def assert_native_only(plan: Any) -> None:
    """Fail loudly if a benchmark fixture starts routing through OCR."""
    assert not plan.ocr_passes, (
        "benchmark fixture now schedules OCR passes; pick another page or the "
        "measurement will be dominated by Tesseract"
    )
