"""Differential raster coverage for embedded PDF font programs."""

import gzip
import zlib

import numpy

from core_pdf import PdfDocument, PdfRasterFontRequest
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.spec.s_09_fonts.font_program import CFFFont
from core_pdf.impl.spec.s_09_fonts.font_program_opentype import OpenTypeFontProgram
from core_pdf.impl.spec.s_09_fonts.font_program_type1 import Type1FontProgram
from tests.helpers.paths import FIXTURES


def internal_poppler_rgb(name: str) -> numpy.ndarray:
    reference = FIXTURES / "font_programs" / "poppler" / name
    header, dimensions, maximum, pixels = gzip.decompress(reference.read_bytes()).split(b"\n", 3)
    assert header == b"P6"
    assert maximum == b"255"
    width, height = (int(value) for value in dimensions.split())
    return numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(height, width, 3)


def test_embedded_type3_raster_matches_poppler_foreground_mask() -> None:
    """Keep the real Type 3 CharProc raster within the committed Poppler tolerance."""
    pdf = FIXTURES / "PyMuPDF" / "tests" / "resources" / "type3font.pdf"

    with PdfDocument.open(pdf) as document:
        raster = document.pages[0].render().rasterize(scale=2.0, cache=False)

    pixels = numpy.asarray(raster.pixels).reshape(raster.height, raster.width, 4)
    foreground = numpy.any(pixels[..., :3] < 250, axis=-1)
    rows, columns = numpy.where(foreground)

    poppler = internal_poppler_rgb("type3font-144dpi.ppm.gz")
    assert poppler.shape == pixels[..., :3].shape
    color_error = numpy.abs(pixels[..., :3].astype(int) - poppler.astype(int))

    # Poppler 26.07.0 at 144 DPI paints 380 foreground pixels in the same
    # half-open bounding box. Permit only the two antialiasing-edge pixels by
    # comparing the stable mask statistics instead of requiring byte identity.
    assert float(color_error.mean()) < 0.38
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
        glyphs = program.glyphs
        raster = page.render(RenderOptions(include_annotations=False)).rasterize(cache=False)

    assert len(glyphs) == 119
    assert "".join(glyph.text for glyph in glyphs) == (
        "HeadingLinktoheadingthatisworkingwithvim-pandoc."
        "Linktoheading“thatis”notworkingwithvim-pandoc."
        "SubheadingSome“moretext”1"
    )
    assert all(
        isinstance(getattr(glyph.font_decoder, "font_program", None), Type1FontProgram)
        for glyph in glyphs
    )
    heading = glyphs[0]
    assert heading.text == "H"
    assert isinstance(heading.font_decoder, FontDecoder)
    assert isinstance(heading.font_decoder.font_program, Type1FontProgram)
    assert heading.font_decoder.font_program.glyph_contours("H")

    pixels = numpy.asarray(raster.pixels).reshape(raster.height, raster.width, 4)
    foreground = numpy.any(pixels[..., :3] < 250, axis=-1)
    rows, columns = numpy.where(foreground)

    poppler = internal_poppler_rgb("simple5-type1-72dpi.ppm.gz")
    assert poppler.shape == pixels[..., :3].shape
    color_error = numpy.abs(pixels[..., :3].astype(int) - poppler.astype(int))

    # Poppler 26.07.0 at 72 DPI produces 3,761 antialiased foreground pixels
    # in (134, 124, 381, 702). The engine now computes analytic coverage rather
    # than sampling a 4x4 grid, so it does partially cover edge pixels: 3,975,
    # down from the 4,844 of the hard-edged mask this used to assert and within
    # 6% of Poppler instead of 29%.
    assert float(color_error.mean()) < 1.57
    assert int(foreground.sum()) == 3975
    assert (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ) == (134, 124, 381, 703)


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

    assert isinstance(decoder.font_program, CFFFont)
    glyph_id = decoder.font_program.glyph_id_for_name("A")
    assert decoder.font_program.has_glyph_id(glyph_id)
    assert not decoder.font_program.has_glyph_id(-1)
    assert len(contours) == 2
    assert all(len(contour) >= 3 for contour in contours)
    assert decoder.font_program.normalized_glyph_contours(glyph_id)


def test_opentype_cff2_program_exposes_actual_outlines() -> None:
    """Render a real variable CFF2 glyph through the generic OpenType path."""
    encoded = FIXTURES / "font_programs" / "cff2-a.otf.zlib.hex"
    font_data = zlib.decompress(bytes.fromhex(encoded.read_text().strip()))
    font = {
        "Subtype": "Type1",
        "BaseFont": "HintOrderTest",
        "Encoding": {"Differences": [65, "/A"]},
        "FontDescriptor": {"FontFile3": PdfStream({"Subtype": "OpenType"}, decoded_data=font_data)},
    }

    decoder = FontDecoder(font)
    contours = decoder.glyph_outline(ord("A"), text="A")

    assert isinstance(decoder.font_program, OpenTypeFontProgram)
    assert len(contours) == 5
    assert all(len(contour) >= 3 for contour in contours)
    assert decoder.glyph_bbox(ord("A")) == (5.0, 0.0, 904.0, 675.0)
    assert decoder.glyph_bitmap(ord("A"))
