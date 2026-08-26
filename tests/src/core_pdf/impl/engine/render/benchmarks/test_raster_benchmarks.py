# SPDX-License-Identifier: AGPL-3.0-only
"""Performance bounds for page rasterization.

The OCR path renders every page it recognizes, and rendering is the second largest
absolute CPU cost across the corpus, so the rasterizer needs a throughput floor of its
own. These benchmarks separate the two costs that move independently: building the
display list once (`compose_page`) and painting it (`RenderedPage.rasterize`).

Scale matters as much as the page does. The same document is measured at screen scale
and at the OCR primary budget because the fill dispatch differs between them -- pure
black fills switch from analytic coverage to the single-sample scanline path once their
bounding box clears ten pixels, which only happens at OCR scales.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.parse.model import PRIMARY_OCR_PIXELS
from core_pdf.impl.engine.render.display import RenderOptions
from core_pdf.impl.engine.render.page import compose_page

FIXTURES = Path(__file__).parents[6] / "fixtures"
TEXT_PDF = FIXTURES / "pypdf" / "resources" / "crazyones.pdf"
VECTOR_PDF = FIXTURES / "pypdf" / "resources" / "GeoBase_NHNC1_Data_Model_UML_EN.pdf"

MAX_PIXELS = 16_000_000


def internal_ocr_scale(page) -> float:
    """The scale a page raster gets under the primary OCR pixel budget."""
    area = max(1.0, float(page.width) * float(page.height))
    return math.sqrt(PRIMARY_OCR_PIXELS / area) * 0.999


def internal_rasterize(rendered, scale: float) -> object:
    return rendered.rasterize(
        background=(255, 255, 255, 255),
        scale=scale,
        max_pixels=MAX_PIXELS,
        cache=False,
    )


@pytest.mark.benchmark_high_impact
def test_compose_text_page_benchmark(benchmark) -> None:
    """Display-list construction, which the OCR path pays once per page."""
    with PdfDocument.open(TEXT_PDF) as document:
        page = document.pages[0]
        page.get_page_program()  # warm stream, font, and decoder caches
        rendered = benchmark(compose_page, page, RenderOptions(include_text=True))
        assert rendered.width > 0
        assert rendered.height > 0


@pytest.mark.benchmark_high_impact
def test_rasterize_text_page_at_ocr_scale_benchmark(benchmark) -> None:
    """The hot path: painting a text page at the OCR primary budget."""
    with PdfDocument.open(TEXT_PDF) as document:
        page = document.pages[0]
        rendered = compose_page(page, RenderOptions(include_text=True))
        scale = internal_ocr_scale(page)
        raster = benchmark(internal_rasterize, rendered, scale)
        assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
        assert raster.channels == 4


@pytest.mark.benchmark_high_impact
def test_rasterize_text_page_at_screen_scale_benchmark(benchmark) -> None:
    """The same page at 72 DPI, where every fill takes the analytic coverage path."""
    with PdfDocument.open(TEXT_PDF) as document:
        page = document.pages[0]
        rendered = compose_page(page, RenderOptions(include_text=True))
        raster = benchmark(internal_rasterize, rendered, 1.0)
        assert raster.channels == 4


@pytest.mark.benchmark_high_impact
def test_rasterize_vector_page_at_ocr_scale_benchmark(benchmark) -> None:
    """A drawing-heavy page, where path filling rather than glyph painting dominates."""
    with PdfDocument.open(VECTOR_PDF) as document:
        page = document.pages[0]
        rendered = compose_page(page, RenderOptions(include_text=True))
        scale = internal_ocr_scale(page)
        raster = benchmark(internal_rasterize, rendered, scale)
        assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
