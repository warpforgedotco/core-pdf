from __future__ import annotations

import zlib
from pathlib import Path

import numpy

from core_pdf import PdfDocument, PdfRasterFontFace, PdfRasterFontRequest
from core_pdf.impl.engine.render.display import RenderOptions
from core_pdf.impl.engine.render.raster_image import RasterImage
from core_pdf.impl.engine.spec.s_09_fonts.fallback import fallback_glyph_outline

SIMPLE1 = Path(__file__).parent / "fixtures" / "pdfminer.six" / "samples" / "simple1.pdf"
FONT_PROGRAM_FIXTURES = Path(__file__).parent / "fixtures" / "font_programs"


def internal_nonwhite_pixels(raster: RasterImage) -> int:
    pixels = numpy.frombuffer(raster.pixels, dtype=numpy.uint8).reshape(
        raster.height, raster.width, raster.channels
    )
    return int(numpy.count_nonzero(numpy.any(pixels[:, :, :3] != 255, axis=2)))


def test_unembedded_base14_font_renders_deterministically() -> None:
    with PdfDocument.open(SIMPLE1) as document:
        raster = document.pages[0].render().rasterize()

    pixels = numpy.frombuffer(raster.pixels, dtype=numpy.uint8).reshape(
        raster.height, raster.width, raster.channels
    )
    foreground = numpy.any(pixels[:, :, :3] != 255, axis=2)
    rows, columns = numpy.where(foreground)

    assert internal_nonwhite_pixels(raster) == 2997
    assert (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ) == (102, 74, 423, 392)


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

    assert text == "Hello World" * 4
    assert requests
    assert all(request.font_name == "Helvetica" for request in requests)


def test_symbol_and_zapf_fallbacks_cover_representative_glyphs() -> None:
    assert fallback_glyph_outline("Symbol", "Ω", is_cid_font=False, is_vertical=False)
    assert fallback_glyph_outline("ZapfDingbats", "✂", is_cid_font=False, is_vertical=False)


def test_cjk_provider_supplies_deterministic_vertical_outline() -> None:
    encoded = FONT_PROGRAM_FIXTURES / "cjk-provider.ttf.zlib.hex"
    font_data = zlib.decompress(bytes.fromhex(encoded.read_text().strip()))
    requests: list[PdfRasterFontRequest] = []

    def provider(request: PdfRasterFontRequest) -> PdfRasterFontFace:
        requests.append(request)
        return PdfRasterFontFace("test-cjk-provider", font_data)

    contours = fallback_glyph_outline(
        "HeiseiKakuGo-W5",
        "日",
        is_cid_font=True,
        is_vertical=True,
        cid_registry="Adobe",
        cid_ordering="Japan1",
        provider=provider,
    )

    assert contours
    assert requests == [
        PdfRasterFontRequest(
            "HeiseiKakuGo-W5",
            "日",
            True,
            True,
            "Adobe",
            "Japan1",
        )
    ]


def test_missing_cjk_coverage_returns_no_outline_without_a_provider() -> None:
    assert (
        fallback_glyph_outline(
            "HeiseiKakuGo-W5",
            "日",
            is_cid_font=True,
            is_vertical=False,
            cid_registry="Adobe",
            cid_ordering="Japan1",
        )
        == ()
    )
