from __future__ import annotations

from pathlib import Path

import numpy

from core_pdf import PdfDocument, PdfRasterFontRequest
from core_pdf.impl.engine.rendering import RasterImage, RenderOptions

SIMPLE1 = Path(__file__).parent / "fixtures" / "pdfminer.six" / "samples" / "simple1.pdf"


def internal_nonwhite_pixels(raster: RasterImage) -> int:
    pixels = numpy.frombuffer(raster.pixels, dtype=numpy.uint8).reshape(
        raster.height, raster.width, raster.channels
    )
    return int(numpy.count_nonzero(numpy.any(pixels[:, :, :3] != 255, axis=2)))


def test_unembedded_base14_font_renders_deterministically() -> None:
    with PdfDocument.open(SIMPLE1) as document:
        raster = document.pages[0].render().rasterize()

    assert internal_nonwhite_pixels(raster) > 0


def test_include_text_false_suppresses_fallback_glyph_paint() -> None:
    with PdfDocument.open(SIMPLE1) as document:
        raster = document.pages[0].render(RenderOptions(include_text=False)).rasterize()

    assert internal_nonwhite_pixels(raster) == 0


def test_custom_raster_font_provider_is_consulted_only_for_rendering() -> None:
    requests: list[PdfRasterFontRequest] = []

    def provider(request: PdfRasterFontRequest) -> None:
        requests.append(request)
        return None

    with PdfDocument.open(SIMPLE1, raster_font_provider=provider) as document:
        text = "".join(run.text for run in document.pages[0].chars)
        assert requests == []
        document.pages[0].render().rasterize()

    assert text
    assert requests
    assert all(request.font_name == "Helvetica" for request in requests)
