"""Reader font recovery stays available alongside independently strict semantics."""

from __future__ import annotations

import re

import pytest

from core_pdf.impl._impl.fonts.cmap_decoder import CMapDecoder
from core_pdf.impl._impl.fonts.cmap_tokenizer import cmap_tokens, decode_cmap_token
from core_pdf.impl._impl.fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl._impl.fonts.cmap_widths import parse_cid_widths
from core_pdf.impl._impl.fonts.decoder import internal_font_program_for_pdf_font
from core_pdf.impl._impl.fonts.font_program import CFFFont, internal_type2_glyph_geometry_impl
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


@pytest.mark.parametrize(
    ("include_arrays", "expected"),
    [(False, [b"<01>", b"<02>"]), (True, [b"<01>", b"[<02>]"])],
)
def test_reader_cmap_tokens_retain_prefix_before_incomplete_suffix(
    include_arrays: bool, expected: list[bytes]
) -> None:
    assert cmap_tokens(b"<01> [<02>] <broken", include_arrays=include_arrays) == expected


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
def test_reader_type2_keeps_completed_contours_before_invalid_subroutine(operator: int) -> None:
    # The second moveto flushes the first contour before the failing call.
    prefix = bytes([139, 139, 21, 149, 159, 6, 140, 141, 21])
    assert internal_type2_glyph_geometry_impl(
        prefix + bytes([139, operator, 14]),
        local_subrs=(b"\x0b",),
        global_subrs=(b"\x0b",),
    ) == ([[(0.0, 0.0), (10.0, 0.0), (10.0, 20.0)]], (0.0, 0.0, 10.0, 20.0))
    assert internal_type2_glyph_geometry_impl(
        prefix + bytes([32, operator, 14]),
        local_subrs=(b"\x0b",),
        global_subrs=(b"\x0b",),
    ) == (
        [[(0.0, 0.0), (10.0, 0.0), (10.0, 20.0)], [(11.0, 22.0)]],
        (0.0, 0.0, 11.0, 22.0),
    )


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
def test_reader_type2_subroutine_endchar_keeps_geometry_and_stops_caller(operator: int) -> None:
    assert internal_type2_glyph_geometry_impl(
        bytes([139, 139, 21, 149, 159, 6, 32, operator, 149, 139, 21, 149, 6, 14]),
        local_subrs=(b"\x0e",),
        global_subrs=(b"\x0e",),
    ) == ([[(0.0, 0.0), (10.0, 0.0), (10.0, 20.0)]], (0.0, 0.0, 10.0, 20.0))


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


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_reader_cmap_preserves_five_byte_codes(reader: type[CMapDecoder | ToUnicodeCMap]) -> None:
    codespace = b"1 begincodespacerange <0000000000> <ffffffffff> endcodespacerange "
    mapping = (
        b"1 begincidchar <0000000001> 7 endcidchar"
        if reader is CMapDecoder
        else b"1 beginbfchar <0000000001> <0041> endbfchar"
    )
    cmap = reader(codespace + mapping)
    if isinstance(cmap, CMapDecoder):
        assert cmap.decode_entries(b"\x00\x00\x00\x00\x01") == [(b"\x00\x00\x00\x00\x01", 7)]
    else:
        assert cmap.lookup(b"\x00\x00\x00\x00\x01") == "A"
    assert cmap.decode_lengths == (5,)


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_reader_preserves_disjoint_child_codespace_redeclarations(
    reader: type[CMapDecoder | ToUnicodeCMap],
) -> None:
    parent = b"1 begincodespacerange <00> <7f> endcodespacerange "
    child = b"/Parent usecmap 1 begincodespacerange <80> <ff> endcodespacerange "
    cmap = reader(child, usecmap_resolver=lambda name: parent)
    assert tuple(cmap.code_space_ranges) == ((b"\x00", b"\x7f"), (b"\x80", b"\xff"))


def test_reader_tounicode_preserves_overlapping_inherited_codespaces_and_local_mappings() -> None:
    parent = CODESPACE + b"1 beginbfchar <01> <0041> endbfchar"
    child = b"/Parent usecmap " + CODESPACE + b"1 beginbfchar <01> <0042> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert len(cmap.code_space_ranges) == 2
    assert cmap.lookup(b"\x01") == "B"


def test_reader_tounicode_keeps_mapping_outside_parent_codespace() -> None:
    parent = b"1 begincodespacerange <00> <7f> endcodespacerange"
    child = b"/Parent usecmap 1 beginbfchar <80> <0041> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert cmap.lookup(b"\x80") == "A"


def test_reader_tounicode_keeps_mappings_after_invalid_codespace() -> None:
    cmap = ToUnicodeCMap(
        b"1 begincodespacerange <0083> <020c> endcodespacerange "
        b"1 beginbfchar <0100> <0041> endbfchar"
    )
    assert not cmap.code_space_ranges
    assert cmap.lookup(b"\x01\x00") == "A"


@pytest.mark.parametrize("font", [{}, {"FontDescriptor": None}, {"FontDescriptor": {}}])
def test_reader_keeps_default_width_recovery_for_absent_and_null_descriptors(
    font: dict[str, object],
) -> None:
    metrics = parse_font_widths(font, "Type1")
    assert metrics.default_width == 1000.0
    assert metrics.default_width_explicit is False
    assert internal_font_program_for_pdf_font(font) is None


def test_reader_does_not_substitute_for_explicit_zero_missing_width() -> None:
    metrics = parse_font_widths({"FontDescriptor": {"MissingWidth": 0}}, "Type1")
    assert metrics.default_width == 0.0
    assert metrics.default_width_explicit is True
