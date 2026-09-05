# SPDX-License-Identifier: AGPL-3.0-only
"""Native page composition, screen rasterization, and fallback-font throughput."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from core_pdf.impl.spec.s_09_fonts.fallback import (
    fallback_glyph_outline,
    internal_RasterFontRepository,
)
from tests.helpers.benchmark_pages import (
    TEXT_PDF,
    VECTOR_PAGE_INDEX,
    VECTOR_PDF,
    opened_page,
)

MAX_PIXELS = 16_000_000


def internal_resolve_fallback_line(repository: internal_RasterFontRepository) -> int:
    resolved = 0
    for text in "The quick brown fox jumps over the lazy dog":
        contours = fallback_glyph_outline(
            "Helvetica",
            text,
            is_cid_font=False,
            is_vertical=False,
            provider=repository,
        )
        resolved += bool(contours)
    return resolved


def test_fallback_font_repository_benchmark(benchmark) -> None:
    """Repeated glyph fallback reuses the document-owned parsed font."""
    repository = internal_RasterFontRepository()

    resolved = benchmark(internal_resolve_fallback_line, repository)

    assert resolved == 35


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


def test_rasterize_text_page_at_screen_scale_benchmark(benchmark, text_page) -> None:
    """The same page at 72 DPI, where every fill takes the analytic path."""
    _, rendered = text_page

    raster = benchmark(internal_rasterize, rendered, 1.0)

    assert raster.channels == 4
