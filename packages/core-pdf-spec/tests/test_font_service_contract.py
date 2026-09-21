from __future__ import annotations

import dataclasses

from core_pdf_spec.s_09_fonts import service
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph


def test_decoded_font_glyph_is_exported() -> None:
    assert "DecodedFontGlyph" in service.__all__


def test_decoded_font_glyph_field_order_is_pinned() -> None:
    assert tuple(field.name for field in dataclasses.fields(DecodedFontGlyph)) == (
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "unicode",
        "width_code",
    )


def test_decoded_font_glyph_binds_positional_arguments_in_declared_order() -> None:
    glyph = DecodedFontGlyph(b"\x01", 2, 3, 4, "five", 6)
    assert glyph.code_bytes == b"\x01"
    assert glyph.char_code == 2
    assert glyph.cid == 3
    assert glyph.gid == 4
    assert glyph.unicode == "five"
    assert glyph.width_code == 6
