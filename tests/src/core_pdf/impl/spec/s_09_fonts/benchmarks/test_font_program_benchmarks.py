"""Performance and cache bounds for embedded font programs."""

from __future__ import annotations

import functools
import zlib

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.spec.s_09_fonts.font_program_opentype import OpenTypeFontProgram
from core_pdf.impl.spec.s_09_fonts.font_program_type1 import Type1FontProgram
from tests.helpers.paths import FIXTURES

TYPE1_PDF = FIXTURES / "pdfminer.six" / "samples" / "simple5.pdf"
CFF2_HEX = FIXTURES / "font_programs" / "cff2-a.otf.zlib.hex"


@functools.cache
def internal_type1_data() -> tuple[bytes, int | None]:
    with PdfDocument.open(TYPE1_PDF) as document:
        decoder = document.pages[0].get_page_program().products.glyphs[0].font_decoder
        assert isinstance(decoder, FontDecoder)
        descriptor = decoder.font.get("FontDescriptor")
        assert isinstance(descriptor, dict)
        stream = descriptor.get("FontFile")
        assert isinstance(stream, PdfStream)
        value = stream.dictionary.get("Length1")
        length1 = int(value) if isinstance(value, (int, float)) else None
        return stream.data, length1


@functools.cache
def internal_cff2_data() -> bytes:
    return zlib.decompress(bytes.fromhex(CFF2_HEX.read_text().strip()))


@pytest.mark.benchmark_high_impact
def test_type1_cold_program_parse_benchmark(benchmark) -> None:
    data, length1 = internal_type1_data()
    program = benchmark(Type1FontProgram, data, length1=length1)
    assert program.charstrings
    assert len(program.internal_contour_cache) == 0


@pytest.mark.benchmark_high_impact
def test_type1_cached_glyph_reuse_benchmark(benchmark) -> None:
    data, length1 = internal_type1_data()
    program = Type1FontProgram(data, length1=length1)
    expected = program.glyph_contours("H")
    result = benchmark(program.glyph_contours, "H")
    assert result is expected
    assert len(program.internal_contour_cache) == 1


def test_cff2_cold_program_parse_benchmark(benchmark) -> None:
    program = benchmark(OpenTypeFontProgram, internal_cff2_data())
    assert program.glyph_id_for_name("A") is not None
    assert len(program.internal_contour_cache) == 0
