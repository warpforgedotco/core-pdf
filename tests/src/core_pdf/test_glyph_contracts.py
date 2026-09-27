from typing import Any

import pytest

from core_pdf.impl.fonts_decoder import FontDecoder
from core_pdf.impl.glyph_contracts import CaptureFont, GlyphGeometrySource
from core_pdf_spec.s_07_syntax.stream import PdfStream


def capture_font(font: dict[str, Any]) -> CaptureFont:
    return FontDecoder(font)


@pytest.mark.parametrize(
    "font",
    [
        {"Subtype": "Type1", "BaseFont": "Helvetica"},
        {"Subtype": "Type3", "FontMatrix": [0.001, 0, 0, 0.001, 0, 0]},
        {
            "Subtype": "Type0",
            "BaseFont": "X",
            "Encoding": "Identity-H",
            "DescendantFonts": [{"Subtype": "CIDFontType2", "BaseFont": "X"}],
            "ToUnicode": PdfStream({}, b""),
        },
    ],
)
def test_every_font_decoder_is_a_capture_font(font: dict[str, Any]) -> None:
    decoded = capture_font(font)
    assert isinstance(decoded, CaptureFont)
    assert isinstance(decoded, GlyphGeometrySource)


def test_an_object_without_glyph_geometry_is_not_a_geometry_source() -> None:
    assert not isinstance(object(), GlyphGeometrySource)
