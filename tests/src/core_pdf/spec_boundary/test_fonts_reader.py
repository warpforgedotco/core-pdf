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


@pytest.mark.parametrize(
    ("font", "subtype", "expected", "default"),
    [
        ({"FirstChar": "0", "LastChar": "0", "Widths": ["500"]}, "Type1", {0: 500}, 1000),
        ({"FontDescriptor": {"MissingWidth": "250"}}, "Type1", {}, 250),
        ({"DescendantFonts": [{"DW": "750", "W": ["0", ["500"]]}]}, "Type0", {0: 500}, 750),
        ({"FontDescriptor": {"MissingWidth": None}, "Widths": None}, "Type1", {}, 1000),
        (
            {"DescendantFonts": [{"DW": None, "DW2": None, "W": None, "W2": None}]},
            "Type0",
            {},
            1000,
        ),
    ],
)
def test_reader_width_coercion_and_null_default_recovery(
    font: dict[str, object], subtype: str, expected: dict[int, float], default: float
) -> None:
    metrics = parse_font_widths(font, subtype)
    assert dict(metrics.widths) == expected
    assert metrics.default_width == default


def test_reader_vertical_widths_coerce_strings_and_clip_invalid_cids() -> None:
    metrics = parse_font_widths(
        {
            "DescendantFonts": [
                {"DW2": ["900", "-750"], "W2": ["65535", ["-500", "250", "880", 1, 2, 3]]}
            ]
        },
        "Type0",
    )
    assert (metrics.default_vertical_origin_y, metrics.default_vertical_displacement_y) == (
        900,
        -750,
    )
    assert metrics.vertical_metrics == {65535: (-500, 250, 880)}


@pytest.mark.parametrize(
    ("font", "encoding", "name", "wmode", "expected"),
    [
        ({"DescendantFonts": [{}]}, None, None, 0, False),
        ({"DescendantFonts": [{"WMode": "1"}]}, None, None, 0, True),
        ({"DescendantFonts": [{}], "WMode": 1}, None, None, 0, True),
        ({"DescendantFonts": [{"WMode": 0}], "WMode": 1}, None, None, 0, False),
        ({"DescendantFonts": [{"WMode": "broken"}], "WMode": 1}, None, None, 0, False),
        ({"DescendantFonts": [{"WMode": 0}]}, None, None, 1, True),
        ({"DescendantFonts": [{}]}, "V", None, 0, True),
        ({"DescendantFonts": [{}]}, "Identity-V", None, 0, True),
        ({"DescendantFonts": [{}]}, None, "Producer-V", 0, True),
        ({"WMode": 1}, None, None, 0, False),
    ],
)
def test_reader_writing_mode_combines_cmap_and_existing_recovery(
    font: dict[str, object], encoding: str | None, name: str | None, wmode: int, expected: bool
) -> None:
    from core_pdf.impl._impl.fonts.decoder import internal_font_is_vertical

    cmap = CMapDecoder.identity(wmode=wmode)
    assert internal_font_is_vertical(font, "Type0", encoding, name, cmap) is expected


def test_reader_resource_cycle_keeps_local_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_pdf.impl._impl.fonts import cmap_resources

    requests: list[str] = []

    def resource(name: str) -> bytes:
        requests.append(name)
        if name == "Child":
            return b"/Loop usecmap " + CODESPACE + b"1 begincidchar <01> 7 endcidchar"
        return b"/Loop usecmap"

    monkeypatch.setattr(cmap_resources, "resolve_cmap_resource", resource)
    cmap = cmap_resources.resolve_cmap_decoder("Child")
    assert cmap is not None
    assert cmap.decode_entries(b"\x01") == [(b"\x01", 7)]
    assert requests == ["Child", "Loop"]


def test_reader_resource_loading_retains_depth_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_pdf.impl._impl.fonts import cmap_resources

    def resource(name: str) -> bytes:
        return b"/" + str(int(name) + 1).encode() + b" usecmap"

    monkeypatch.setattr(cmap_resources, "resolve_cmap_resource", resource)
    assert cmap_resources.resolve_cmap_decoder("0") is None
    with pytest.raises(ValueError, match="nesting too deep"):
        CMapDecoder(b"/0 usecmap", usecmap_resolver=resource)


@pytest.mark.parametrize("name", ["OneByteIdentityH", "OneByteIdentityV"])
def test_reader_raw_resource_callback_keeps_identity_aliases(name: str) -> None:
    cmap = CMapDecoder(b"/" + name.encode() + b" usecmap", usecmap_resolver=lambda name: None)
    assert cmap.decode_entries(b"A") == [(b"A", 65)]
    assert cmap.wmode == int(name.endswith("V"))


def test_reader_cff_matrix_defaults_and_scaled_geometry() -> None:
    from core_pdf_spec.s_08_graphics.matrix import Matrix

    font = CFFFont(None)
    font.charstrings = [bytes([139, 139, 21, 149, 159, 5, 14])]
    font.fd_select = (0,)
    font.local_subrs = ((),)
    font.top_dict = {(12, 7): [0.001, 0, 0, 0.001, 0, 0]}
    font.font_dicts = ({(12, 7): [2, 0, 0, 3, 4, 5]},)
    assert font.font_matrix(0) == Matrix(0.002, 0, 0, 0.003, 0.004, 0.005)
    assert font.glyph_bbox_for_gid(0) == (4, 5, 24, 65)
    font.top_dict = {(12, 7): [float("nan"), 0, 0, 1, 0, 0]}
    assert font.font_matrix(0) == Matrix(2, 0, 0, 3, 4, 5)


@pytest.mark.parametrize("descendant", [False, True])
@pytest.mark.parametrize("textual", [False, True])
def test_reader_font_program_selection_recovers_text_names_without_reparsing_pdf_names(
    monkeypatch: pytest.MonkeyPatch, descendant: bool, textual: bool
) -> None:
    from core_pdf.impl._impl.fonts import decoder
    from core_pdf_spec.s_09_fonts.dictionaries import FontProgramInputs
    from core_pdf_spec.types import PdfName

    name = "/Type1" if textual else PdfName.of("/Type1")
    font: dict[str, object] = {"Subtype": name}
    if descendant:
        font["DescendantFonts"] = [{"Subtype": name}]
    seen: list[FontProgramInputs] = []
    monkeypatch.setattr(decoder, "internal_cff_font", lambda inputs: seen.append(inputs))
    for function in ("internal_tt_font", "internal_type1_font", "internal_opentype_font"):
        monkeypatch.setattr(decoder, function, lambda inputs: None)
    assert decoder.internal_font_program_for_pdf_font(font) is None
    expected = "Type1" if textual else "/Type1"
    assert [(inputs.subtype, inputs.original_subtype) for inputs in seen] == [(expected, expected)]


def test_reader_embedded_encoding_uses_raw_parent_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_pdf.impl._impl.fonts import decoder
    from core_pdf_spec.s_07_syntax.stream import PdfStream
    from core_pdf_spec.types import PdfName

    resources = {
        "Parent": b"/Grandparent usecmap 1 begincidchar <02> 8 endcidchar",
        "Grandparent": CODESPACE + b"/WMode 1 def 1 begincidchar <01> 7 endcidchar",
    }
    monkeypatch.setattr(decoder, "resolve_cmap_resource", resources.get)
    font = decoder.FontDecoder(
        {
            "Subtype": PdfName.of("Type0"),
            "DescendantFonts": [{"Subtype": PdfName.of("CIDFontType2")}],
            "Encoding": PdfStream(raw_data=b"/Parent usecmap 1 begincidchar <01> 9 endcidchar"),
        }
    )
    assert font.cmap is not None
    assert font.cmap.decode_entries(b"\x01\x02") == [(b"\x01", 9), (b"\x02", 8)]
    assert font.is_vertical


@pytest.mark.parametrize("encoded_name", [b"/Identity-H", b"/#2FIdentity-H"])
def test_reader_named_encoding_preserves_the_parsed_pdf_name(encoded_name: bytes) -> None:
    from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
    from core_pdf.impl._impl.fonts.decoder import FontDecoder

    lexer = PdfLexer(
        b"<< /Subtype /Type0 /DescendantFonts [<< >>] /Encoding " + encoded_name + b" >>"
    )
    try:
        font = FontDecoder(lexer.parse_object())
    finally:
        lexer.close()
    literal_slash = encoded_name == b"/#2FIdentity-H"
    assert font.base_encoding == ("/Identity-H" if literal_slash else "Identity-H")
    assert (font.cmap is None) is literal_slash


def test_reader_textual_encoding_slash_recovery_remains_at_input_boundary() -> None:
    from core_pdf.impl._impl.fonts.decoder import FontDecoder

    font = FontDecoder({"Subtype": "/Type0", "DescendantFonts": [{}], "Encoding": "/Identity-H"})
    assert font.base_encoding == "Identity-H"
    assert font.cmap is not None
    assert font.cmap.decode_entries(b"\x00A") == [(b"\x00A", 65)]


def test_reader_cmap_unicode_lookup_does_not_reinterpret_decoded_slashes() -> None:
    from core_pdf.impl._impl.fonts.cmap_resources import (
        predefined_cmap_unicode,
        resolve_cmap_decoder,
    )

    assert resolve_cmap_decoder("/Identity-H") is None
    assert predefined_cmap_unicode("UniJIS-UTF16-H", b"\x00A") == "A"
    assert predefined_cmap_unicode("/UniJIS-UTF16-H", b"\x00A") is None
