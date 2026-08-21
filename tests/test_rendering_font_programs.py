"""Differential raster coverage for embedded PDF font programs."""

import hashlib
import zlib
from pathlib import Path

import numpy

from core_pdf import PdfDocument, PdfRasterFontRequest
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.objects import PdfStream

FIXTURES = Path(__file__).parent / "fixtures"


def test_embedded_type3_raster_matches_poppler_foreground_mask() -> None:
    """Keep the real Type 3 CharProc raster within the committed Poppler tolerance."""
    pdf = FIXTURES / "PyMuPDF" / "tests" / "resources" / "type3font.pdf"

    with PdfDocument.open(pdf) as document:
        raster = document.pages[0].render().rasterize(scale=2.0, cache=False)

    pixels = numpy.asarray(raster.pixels).reshape(raster.height, raster.width, 4)
    foreground = numpy.any(pixels[..., :3] < 250, axis=-1)
    rows, columns = numpy.where(foreground)

    # Poppler 26.08.0 at 144 DPI paints 380 foreground pixels in the same
    # half-open bounding box. Permit only the two antialiasing-edge pixels by
    # comparing the stable mask statistics instead of requiring byte identity.
    assert int(foreground.sum()) == 378
    assert (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ) == (61, 61, 83, 83)


def test_embedded_type1_uses_actual_outlines_without_fallback() -> None:
    """Render a subsetted Type 1 program without consulting replacement fonts."""
    pdf = FIXTURES / "pdfminer.six" / "samples" / "simple5.pdf"

    def reject_fallback(request: PdfRasterFontRequest) -> None:
        raise AssertionError(f"embedded Type 1 glyph used fallback: {request}")

    with PdfDocument.open(pdf, raster_font_provider=reject_fallback) as document:
        page = document.pages[0]
        program = page.get_page_program()
        glyphs = program.products.glyphs
        raster = page.render().rasterize(cache=False)

    assert len(glyphs) == 119
    assert all(getattr(glyph.font_decoder, "type1_font", None) is not None for glyph in glyphs)
    heading = glyphs[0]
    assert heading.text == "H"
    assert isinstance(heading.font_decoder, FontDecoder)
    assert heading.font_decoder.type1_font is not None
    assert heading.font_decoder.type1_font.glyph_contours("H")

    pixels = numpy.asarray(raster.pixels).reshape(raster.height, raster.width, 4)
    dark_pixels = numpy.all(pixels[..., :3] < 200, axis=-1)
    assert int(dark_pixels.sum()) == 3666


def test_opentype_cff_program_exposes_actual_outlines() -> None:
    """Extract the CFF table from a real OpenType wrapper and resolve by encoding."""
    otf = FIXTURES / "PyMuPDF" / "tests" / "resources" / "PragmaticaC.otf"
    font = {
        "Subtype": "Type1",
        "BaseFont": "PragmaticaC",
        "Encoding": "WinAnsiEncoding",
        "FontDescriptor": {
            "FontFile3": PdfStream(
                {"Subtype": "OpenType"},
                decoded_data=otf.read_bytes(),
            )
        },
    }

    decoder = FontDecoder(font)
    contours = decoder.glyph_outline(ord("A"), text="A")

    assert decoder.cff_font is not None
    assert decoder.opentype_font is not None
    assert decoder.tt_font is None
    assert len(contours) == 2
    assert all(len(contour) >= 3 for contour in contours)
    glyph_id = decoder.opentype_font.glyph_id_for_name("A")
    assert glyph_id is not None
    assert decoder.opentype_font.normalized_glyph_contours(glyph_id)


def test_opentype_cff2_program_exposes_actual_outlines() -> None:
    """Render a real variable CFF2 glyph through the generic OpenType path."""
    encoded = FIXTURES / "font_programs" / "cff2-a.otf.zlib.hex"
    font_data = zlib.decompress(bytes.fromhex(encoded.read_text().strip()))
    assert hashlib.sha256(font_data).hexdigest() == (
        "9c5c093c83c461f39e01e00d0ad1647d2165b0e5d4754260a225a7ba788c5594"
    )
    font = {
        "Subtype": "Type1",
        "BaseFont": "HintOrderTest",
        "Encoding": {"Differences": [65, "/A"]},
        "FontDescriptor": {"FontFile3": PdfStream({"Subtype": "OpenType"}, decoded_data=font_data)},
    }

    decoder = FontDecoder(font)
    contours = decoder.glyph_outline(ord("A"), text="A")

    assert decoder.cff_font is None
    assert decoder.opentype_font is not None
    assert len(contours) == 5
    assert all(len(contour) >= 3 for contour in contours)
