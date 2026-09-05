from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from core_pdf.impl._impl.render.model import RenderOptions
from core_pdf.impl._impl.render.page import compose_page
from core_pdf_ocr.impl.extract.contracts import PRIMARY_OCR_PIXELS
from tests.helpers.benchmark_pages import (
    TEXT_PDF,
    VECTOR_PAGE_INDEX,
    VECTOR_PDF,
    opened_page,
)

MAX_PIXELS = 16_000_000


def internal_rasterize(rendered: Any, scale: float) -> Any:
    return rendered.rasterize(
        background=(255, 255, 255, 255),
        scale=scale,
        max_pixels=MAX_PIXELS,
    )


@pytest.fixture(scope="module")
def text_page() -> Iterator[tuple[Any, Any]]:
    with opened_page(TEXT_PDF) as page:
        yield page, compose_page(page, RenderOptions(include_text=True))


@pytest.fixture(scope="module")
def vector_page() -> Iterator[tuple[Any, Any]]:
    with opened_page(VECTOR_PDF, VECTOR_PAGE_INDEX) as page:
        yield page, compose_page(page, RenderOptions(include_text=True))


def internal_ocr_scale(page: Any) -> float:
    """The scale a page raster gets under the primary OCR pixel budget."""
    area = max(1.0, float(page.width) * float(page.height))
    return math.sqrt(PRIMARY_OCR_PIXELS / area) * 0.999


def test_rasterize_text_page_at_ocr_scale_benchmark(benchmark, text_page) -> None:
    """The hot path: painting a text page at the OCR primary budget."""
    page, rendered = text_page

    raster = benchmark(internal_rasterize, rendered, internal_ocr_scale(page))

    assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
    assert raster.channels == 4


def test_rasterize_vector_page_at_ocr_scale_benchmark(benchmark, vector_page) -> None:
    """A drawing-heavy page, where path filling rather than glyphs dominates."""
    page, rendered = vector_page

    raster = benchmark(internal_rasterize, rendered, internal_ocr_scale(page))

    assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
    assert raster.channels == 4
