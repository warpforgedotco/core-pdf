# SPDX-License-Identifier: AGPL-3.0-only
"""Page rasterization throughput.

The OCR route renders every page it recognizes, which makes rendering the
largest CPU cost after interpretation. The two halves move independently and are
measured apart: ``compose_page`` builds the display list once, and
``RenderedPage.rasterize`` paints it, which is where ``fast_fill_path`` and the
span blenders live.

Scale is part of the workload, not a knob. Pure black fills leave the analytic
coverage path for the single-sample scanline path once their bounding box clears
ten pixels, which only happens at OCR scale, so the text page is measured at
both 72 DPI and the OCR budget.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from core_pdf.impl.extract.contracts import PRIMARY_OCR_PIXELS
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from tests.helpers.benchmark_pages import (
    TEXT_PDF,
    VECTOR_PAGE_INDEX,
    VECTOR_PDF,
    opened_page,
)

MAX_PIXELS = 16_000_000


def internal_ocr_scale(page: Any) -> float:
    """The scale a page raster gets under the primary OCR pixel budget."""
    area = max(1.0, float(page.width) * float(page.height))
    return math.sqrt(PRIMARY_OCR_PIXELS / area) * 0.999


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


def test_compose_text_page_benchmark(benchmark, text_page) -> None:
    """Display-list construction, paid once per rendered page."""
    page, _ = text_page

    rendered = benchmark(compose_page, page, RenderOptions(include_text=True))

    assert rendered.width > 0
    assert rendered.height > 0


def test_rasterize_text_page_at_ocr_scale_benchmark(benchmark, text_page) -> None:
    """The hot path: painting a text page at the OCR primary budget."""
    page, rendered = text_page

    raster = benchmark(internal_rasterize, rendered, internal_ocr_scale(page))

    assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
    assert raster.channels == 4


def test_rasterize_text_page_at_screen_scale_benchmark(benchmark, text_page) -> None:
    """The same page at 72 DPI, where every fill takes the analytic path."""
    _, rendered = text_page

    raster = benchmark(internal_rasterize, rendered, 1.0)

    assert raster.channels == 4


def test_rasterize_vector_page_at_ocr_scale_benchmark(benchmark, vector_page) -> None:
    """A drawing-heavy page, where path filling rather than glyphs dominates."""
    page, rendered = vector_page

    raster = benchmark(internal_rasterize, rendered, internal_ocr_scale(page))

    assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
    assert raster.channels == 4
