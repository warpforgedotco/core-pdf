# SPDX-License-Identifier: AGPL-3.0-only
"""Throughput of the content-stream interpreter.

Interpretation is the largest block of Python this project runs. Profiling
``PdfDocument.extract`` over the SCORE-Bench corpus put ``get_page_program`` at
roughly a quarter of the non-OCR time, and its two hottest leaves --
``record_glyph_observations`` and the ``append_cubic_curve`` path builder --
belong to the two different shapes of page benchmarked here. Text pages and
drawing-heavy pages regress independently, so they are measured separately
rather than averaged into one document.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tests.helpers.benchmark_pages import (
    DENSE_PDF,
    TEXT_PDF,
    VECTOR_PAGE_INDEX,
    VECTOR_PDF,
    opened_page,
    reinterpret,
)


@pytest.fixture(scope="module")
def text_page() -> Iterator[Any]:
    with opened_page(TEXT_PDF) as page:
        yield page


@pytest.fixture(scope="module")
def vector_page() -> Iterator[Any]:
    with opened_page(VECTOR_PDF, VECTOR_PAGE_INDEX) as page:
        yield page


@pytest.fixture(scope="module")
def dense_page() -> Iterator[Any]:
    with opened_page(DENSE_PDF) as page:
        yield page


def test_interpret_text_page_benchmark(benchmark, text_page) -> None:
    """A plain text page: glyph observation recording with no path work."""
    program = benchmark(reinterpret, text_page)

    products = program.products
    assert len(products.glyphs) == 730
    assert not products.drawings


def test_interpret_vector_page_benchmark(benchmark, vector_page) -> None:
    """A drawing-heavy page, where path construction outweighs text."""
    program = benchmark(reinterpret, vector_page)

    products = program.products
    assert len(products.drawings) == 494
    assert len(products.lines) == 411


def test_interpret_dense_page_benchmark(benchmark, dense_page) -> None:
    """The heaviest native-text page in the corpus: text, paths and rules."""
    program = benchmark(reinterpret, dense_page)

    products = program.products
    assert len(products.glyphs) == 3596
    assert len(products.lines) == 363
