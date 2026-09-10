"""Reader font recovery stays available alongside independently strict semantics."""

from __future__ import annotations

import re

import pytest

from core_pdf.impl._impl.fonts.cmap_decoder import CMapDecoder
from core_pdf.impl._impl.fonts.cmap_tokenizer import decode_cmap_token
from core_pdf.impl._impl.fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl._impl.fonts.cmap_widths import parse_cid_widths
from core_pdf.impl._impl.fonts.decoder import internal_font_program_for_pdf_font
from core_pdf.impl._impl.fonts.font_program import CFFFont
from core_pdf.impl._impl.fonts.font_program_type1 import Type1FontProgram
from core_pdf.impl._impl.fonts.helpers import build_simple_encoding_glyph_names
from core_pdf.impl._impl.fonts.widths import get_descendant, parse_font_widths

CODESPACE = b"1 begincodespacerange <00> <ff> endcodespacerange\n"


@pytest.mark.parametrize(
    "mapping",
    [
        b"2 begincidchar <01> 7 <02> bad endcidchar",
        b"2 begincidchar <01> 7 <02> 65536 endcidchar",
        b"2 begincidchar <01> 7 <02> endcidchar",
        b"1 begincidchar <01> 7 endcidchar\n1 begincidrange <02> <03> 65535 endcidrange",
    ],
)
def test_reader_cmap_retains_valid_rows(mapping: bytes) -> None:
    assert CMapDecoder(CODESPACE + mapping).decode_entries(b"\x01\x02") == [
        (b"\x01", 7),
        (b"\x02", 0),
    ]


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_reader_keeps_local_maps_when_parent_is_missing(
    reader: type[CMapDecoder | ToUnicodeCMap],
) -> None:
    data = CODESPACE + b"/Missing usecmap"
    cmap = reader(data)
    assert cmap.code_space_ranges


def test_reader_tounicode_recovers_single_byte_destination_and_truncated_suffix() -> None:
    cmap = ToUnicodeCMap(CODESPACE + b"1 beginbfchar <01> <41> endbfchar\n<broken")
    assert cmap.lookup(b"\x01") == "A"


def test_reader_ignores_incomplete_mapping_block() -> None:
    cmap = CMapDecoder(CODESPACE + b"1 begincidchar <01> 7")
    assert cmap.decode_entries(b"\x01") == [(b"\x01", 0)]


def test_reader_type1_ignores_truncated_entries_and_odd_final_hex_nibble() -> None:
    assert (
        list(Type1FontProgram.binary_entries(b"/A 5 RD abc", re.compile(rb"/(\w+) (\d+) RD ")))
        == []
    )
    assert Type1FontProgram.decode_private(b"0000000000", 1) == b""


def test_reader_cff_recovery_uses_public_parser_extension_points() -> None:
    font = CFFFont(None)
    font.charstrings = [b"\x0e", b"\x0e"]
    assert font.read_charset(100, 2) == {0: 0, 1: 1}
    assert font.read_encoding_codes(100) == {}


@pytest.mark.parametrize("descriptor", [42, {"FontFile": 42}])
def test_reader_skips_malformed_font_resources(descriptor: object) -> None:
    assert (
        internal_font_program_for_pdf_font({"Subtype": "Type1", "FontDescriptor": descriptor})
        is None
    )


@pytest.mark.parametrize(
    ("token", "expected"),
    [(b"(A\n\rB)", b"A\nB"), (b"(A\\\n\rB)", b"AB"), (b"(\r\n\r\n)", b"\n\n")],
)
def test_reader_preserves_legacy_literal_line_endings(token: bytes, expected: bytes) -> None:
    assert decode_cmap_token(token) == expected


def test_reader_cff_keeps_fractional_offset_truncation_and_header_tolerance() -> None:
    font = CFFFont(None)
    font.top_dict = {17: [21.5]}
    font.data = b"\x01\x00\x02\x00"
    assert font.dict_offset(17) == 21
    assert font.read_header() == 2


def test_reader_encoding_skips_invalid_codes_and_substitutes_notdef() -> None:
    names = build_simple_encoding_glyph_names(
        None, {256: "A", 65: ""}, {}, authoritative_builtin=False
    )
    assert names[65] == ".notdef"


def test_reader_descendant_selection_keeps_first_usable_dictionary() -> None:
    assert get_descendant({"DescendantFonts": [{"Subtype": "CIDFontType2"}, {}]}) == {
        "Subtype": "CIDFontType2"
    }
    assert get_descendant({"DescendantFonts": [42]}) is None


def test_reader_cid_widths_keep_valid_portion_of_out_of_range_array() -> None:
    assert dict(parse_cid_widths([-1, [500, 600, 700]])) == {0: 600, 1: 700}


def test_reader_cff_dict_retains_complete_entries_before_trailing_operand() -> None:
    assert CFFFont(None).parse_dict(b"\x8b\x11\x8c") == {17: [0.0]}


def test_reader_infers_missing_font_widths_range() -> None:
    assert dict(parse_font_widths({"Widths": [500]}, "Type1").widths) == {0: 500}
