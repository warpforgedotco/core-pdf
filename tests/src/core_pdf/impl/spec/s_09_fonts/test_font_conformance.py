"""Font dictionary and character-code rules independent of recovery services."""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import symbol_character_code
from core_pdf.impl.spec.s_09_fonts.widths import parse_font_widths


def test_invalid_code_consumes_the_chosen_codespace_length() -> None:
    decoder = CMapDecoder(
        b"2 begincodespacerange <0000> <D7FF> <E000> <FFFF> endcodespacerange "
        b"1 begincidrange <0000> <FFFF> 0 endcidrange"
    )
    assert decoder.decode_entries(b"\xd8\x00\x00\x41\x00\x42") == [
        (b"\xd8\x00", 0),
        (b"\x00\x41", 65),
        (b"\x00\x42", 66),
    ]


@pytest.mark.parametrize("missing", [None, 0, 123])
def test_simple_font_missing_width_and_explicit_widths(missing: int | None) -> None:
    descriptor = {} if missing is None else {"MissingWidth": missing}
    metrics = parse_font_widths(
        {"FirstChar": 65, "LastChar": 66, "Widths": [500, 600], "FontDescriptor": descriptor},
        "TrueType",
    )
    assert metrics.default_width == (missing or 0)
    assert metrics.widths.width_for(65, metrics.default_width) == 500
    assert metrics.widths.width_for(66, metrics.default_width) == 600
    assert metrics.widths.width_for(32, metrics.default_width) == (missing or 0)


@pytest.mark.parametrize("default", [None, 500])
def test_cid_default_width_uses_dw_and_ignores_descriptor_missing_width(
    default: int | None,
) -> None:
    descendant: dict[str, object] = {
        "W": [120, [400, 325, 500]],
        "FontDescriptor": {"MissingWidth": 250},
    }
    if default is not None:
        descendant["DW"] = default
    metrics = parse_font_widths({"DescendantFonts": [descendant]}, "Type0")
    assert metrics.default_width == (1000 if default is None else default)
    assert metrics.widths.width_for(120, metrics.default_width) == 400


@pytest.mark.parametrize("code", [0x41, 0xF041, 0xF141, 0xF241])
def test_symbol_character_code_uses_the_prescribed_high_byte_ranges(code: int) -> None:
    assert symbol_character_code(code) == 65


def test_symbol_character_code_does_not_alias_other_ranges() -> None:
    assert symbol_character_code(0xF341) == 0xF341
