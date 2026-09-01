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

import functools
import math
import zlib
from collections.abc import Iterator

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.parse.model import PRIMARY_OCR_PIXELS
from core_pdf.impl.render.display import DisplayList, ImagePaintItem, RenderOptions
from core_pdf.impl.render.page import RenderedPage, compose_page
from tests.helpers.paths import FIXTURES

TEXT_PDF = FIXTURES / "pypdf" / "resources" / "crazyones.pdf"
VECTOR_PDF = FIXTURES / "pypdf" / "resources" / "GeoBase_NHNC1_Data_Model_UML_EN.pdf"

MAX_PIXELS = 16_000_000
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
SOFT_MASK_WIDTH = IMAGE_WIDTH * 2
SOFT_MASK_HEIGHT = IMAGE_HEIGHT * 2


@functools.cache
def internal_image_raw() -> bytes:
    """Flate-encoded RGB samples, synthesised on first use rather than at import."""
    image_x = numpy.arange(IMAGE_WIDTH, dtype=numpy.uint16)[None, :]
    image_y = numpy.arange(IMAGE_HEIGHT, dtype=numpy.uint16)[:, None]
    samples = numpy.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=numpy.uint8)
    samples[:, :, 0] = (image_x + image_y) % 256
    samples[:, :, 1] = (image_x * 3) % 256
    samples[:, :, 2] = (image_y * 5) % 256
    return zlib.compress(samples.tobytes())


@functools.cache
def internal_soft_mask_raw() -> bytes:
    soft_mask = numpy.tile(
        numpy.arange(SOFT_MASK_WIDTH, dtype=numpy.uint16) % 256,
        (SOFT_MASK_HEIGHT, 1),
    ).astype(numpy.uint8)
    return zlib.compress(soft_mask.tobytes())


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


def internal_image_page(*, soft_mask: bool = False) -> RenderedPage:
    dictionary: dict[str, object] = {
        "Filter": "FlateDecode",
        "Width": IMAGE_WIDTH,
        "Height": IMAGE_HEIGHT,
        "ColorSpace": "DeviceRGB",
        "BitsPerComponent": 8,
    }
    if soft_mask:
        dictionary.update(
            {
                "__soft_mask_raw_data__": internal_soft_mask_raw(),
                "__soft_mask_dictionary__": {
                    "Filter": "FlateDecode",
                    "Width": SOFT_MASK_WIDTH,
                    "Height": SOFT_MASK_HEIGHT,
                    "ColorSpace": "DeviceGray",
                    "BitsPerComponent": 8,
                },
            }
        )
    display_list = DisplayList(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    display_list.append(
        "image",
        1,
        raw_data=internal_image_raw(),
        dictionary=dictionary,
        bbox=(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT),
        items=[
            (
                "quad",
                (
                    (0, 0),
                    (IMAGE_WIDTH, 0),
                    (0, IMAGE_HEIGHT),
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                ),
            )
        ],
    )
    return RenderedPage(
        page_number=1,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        rotate=0,
        display_list=display_list,
    )


def internal_cold_image_rasterize() -> object:
    return internal_rasterize(internal_image_page(), 1.0)


@pytest.fixture(scope="module")
def text_page() -> Iterator[tuple[object, RenderedPage]]:
    """The text page opened and composed once for the three text benchmarks."""
    with PdfDocument.open(TEXT_PDF) as document:
        page = document.pages[0]
        page.get_page_program()  # warm stream, font, and decoder caches
        yield page, compose_page(page, RenderOptions(include_text=True))


@pytest.mark.benchmark_high_impact
def test_compose_text_page_benchmark(benchmark, text_page) -> None:
    """Display-list construction, which the OCR path pays once per page."""
    page, _ = text_page
    rendered = benchmark(compose_page, page, RenderOptions(include_text=True))
    assert rendered.width > 0
    assert rendered.height > 0


@pytest.mark.benchmark_high_impact
def test_rasterize_text_page_at_ocr_scale_benchmark(benchmark, text_page) -> None:
    """The hot path: painting a text page at the OCR primary budget."""
    page, rendered = text_page
    raster = benchmark(internal_rasterize, rendered, internal_ocr_scale(page))
    assert raster.width * raster.height <= PRIMARY_OCR_PIXELS
    assert raster.channels == 4


@pytest.mark.benchmark_high_impact
def test_rasterize_text_page_at_screen_scale_benchmark(benchmark, text_page) -> None:
    """The same page at 72 DPI, where every fill takes the analytic coverage path."""
    _, rendered = text_page
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


@pytest.mark.benchmark_high_impact
def test_rasterize_image_cold_preparation_benchmark(benchmark) -> None:
    """Image display construction, Flate decode, preparation, and first paint."""
    raster = benchmark(internal_cold_image_rasterize)
    assert raster.width == IMAGE_WIDTH
    assert raster.height == IMAGE_HEIGHT
    assert raster.channels == 4


@pytest.mark.benchmark_high_impact
def test_rasterize_image_warm_preparation_benchmark(benchmark) -> None:
    """Repeated image painting after the source preparation cache is warm."""
    rendered = internal_image_page()
    item = rendered.display_list.items[0]
    assert isinstance(item, ImagePaintItem)
    assert item.source is not None
    assert item.source.prepare() is not None

    raster = benchmark(internal_rasterize, rendered, 1.0)

    assert raster.channels == 4


@pytest.mark.benchmark_high_impact
def test_rasterize_native_soft_mask_image_benchmark(benchmark) -> None:
    """Paint a colour image with a higher-resolution prepared soft mask."""
    rendered = internal_image_page(soft_mask=True)
    item = rendered.display_list.items[0]
    assert isinstance(item, ImagePaintItem)
    assert item.source is not None
    prepared = item.source.prepare()
    assert prepared is not None
    assert prepared.soft_mask is not None
    assert prepared.soft_mask.width == SOFT_MASK_WIDTH

    raster = benchmark(internal_rasterize, rendered, 1.0)

    assert raster.channels == 4
