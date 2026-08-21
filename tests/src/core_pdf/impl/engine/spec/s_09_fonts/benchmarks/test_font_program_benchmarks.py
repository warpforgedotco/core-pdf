"""Performance and cache bounds for embedded font programs."""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.rendering import RenderOptions
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.engine.spec.s_09_fonts.font_program_opentype import OpenTypeFontProgram
from core_pdf.impl.engine.spec.s_09_fonts.font_program_type1 import Type1FontProgram
from core_pdf.impl.objects import PdfStream

FIXTURES = Path(__file__).parents[7] / "fixtures"
TYPE1_PDF = FIXTURES / "pdfminer.six" / "samples" / "simple5.pdf"
CFF2_HEX = FIXTURES / "font_programs" / "cff2-a.otf.zlib.hex"


def internal_type1_data() -> tuple[bytes, int | None]:
    with PdfDocument.open(TYPE1_PDF) as document:
        decoder = document.pages[0].get_page_program().products.glyphs[0].font_decoder
        assert isinstance(decoder, FontDecoder)
        descriptor = lookup_dict_key(decoder.font, "FontDescriptor")
        assert isinstance(descriptor, dict)
        stream = lookup_dict_key(descriptor, "FontFile")
        assert isinstance(stream, PdfStream)
        value = lookup_dict_key(stream.dictionary, "Length1")
        length1 = int(value) if isinstance(value, (int, float)) else None
        return stream.data, length1


TYPE1_DATA, TYPE1_LENGTH1 = internal_type1_data()
CFF2_DATA = zlib.decompress(bytes.fromhex(CFF2_HEX.read_text().strip()))


@pytest.mark.benchmark_high_impact
def test_type1_cold_program_parse_benchmark(benchmark) -> None:
    program = benchmark(Type1FontProgram, TYPE1_DATA, length1=TYPE1_LENGTH1)
    assert program.charstrings
    assert len(program.internal_contour_cache) == 0


@pytest.mark.benchmark_high_impact
def test_type1_cached_glyph_reuse_benchmark(benchmark) -> None:
    program = Type1FontProgram(TYPE1_DATA, length1=TYPE1_LENGTH1)
    expected = program.glyph_contours("H")
    result = benchmark(program.glyph_contours, "H")
    assert result is expected
    assert len(program.internal_contour_cache) == 1


def test_cff2_cold_program_parse_benchmark(benchmark) -> None:
    program = benchmark(OpenTypeFontProgram, CFF2_DATA)
    assert program.glyph_id_for_name("A") is not None
    assert len(program.internal_contour_cache) == 0


def test_embedded_type1_page_raster_benchmark(benchmark) -> None:
    def rasterize_page() -> tuple[int, int]:
        with PdfDocument.open(TYPE1_PDF) as document:
            raster = (
                document.pages[0]
                .render(RenderOptions(include_annotations=False))
                .rasterize(cache=False)
            )
            return raster.width, raster.height

    assert benchmark(rasterize_page) == (612, 792)
