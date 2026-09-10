"""Standalone font semantics reject damage and retain prescribed PDF defaults."""

from __future__ import annotations

import importlib
import importlib.util
import re

import pytest

from core_pdf_spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf_spec.s_09_fonts.cmap_resources import resolve_cmap_decoder
from core_pdf_spec.s_09_fonts.cmap_tokenizer import (
    CMapBlock,
    CMapProgram,
    cmap_tokens,
    decode_cmap_hex_token,
    iter_cmap_tokens,
)
from core_pdf_spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf_spec.s_09_fonts.dictionaries import get_descendant, prepare_font_program_inputs
from core_pdf_spec.s_09_fonts.font_program import (
    CFFFont,
    execute_type2_charstring,
)
from core_pdf_spec.s_09_fonts.font_program_type1 import (
    binary_entries,
    decode_charstring,
    decode_eexec_payload,
)
from core_pdf_spec.s_09_fonts.helpers import (
    BASE_ENCODING_GLYPH_NAMES,
    build_simple_encoding_glyph_names,
)
from core_pdf_spec.s_09_fonts.widths import parse_font_widths

CODESPACE = b"1 begincodespacerange <00> <ff> endcodespacerange\n"


def internal_execute_type2(
    program: bytes,
    events: list[tuple[str, tuple[float, ...]]],
    *,
    local_subrs: tuple[bytes, ...] = (),
    global_subrs: tuple[bytes, ...] = (),
) -> bool:
    current_point = False

    def move(x: float, y: float) -> None:
        nonlocal current_point
        current_point = True
        events.append(("move", (x, y)))

    def flush() -> None:
        nonlocal current_point
        current_point = False
        events.append(("flush", ()))

    return execute_type2_charstring(
        program,
        local_subrs=local_subrs,
        global_subrs=global_subrs,
        move=move,
        line=lambda *args: events.append(("line", args)),
        curve=lambda *args: events.append(("curve", args)),
        flush_contour=flush,
        has_current_point=lambda: current_point,
        seac=lambda *args: events.append(("seac", args)),
        random_value=lambda: 0.5,
    )


@pytest.mark.parametrize(
    "mapping",
    [
        b"2 begincidchar <01> 7 <02> bad endcidchar",
        b"2 begincidchar <01> 7 <02> 65536 endcidchar",
        b"2 begincidchar <01> 7 <02> endcidchar",
        b"1 begincidrange <01> <03> 65535 endcidrange",
        b"1 begincidrange <03> <01> 7 endcidrange",
        b"1 begincidchar {<01>} 7 endcidchar",
        b"1 begincidchar <0100> 7 endcidchar",
        b"1 begincidchar <01> 7",
    ],
)
def test_cid_cmap_rejects_malformed_mappings(mapping: bytes) -> None:
    with pytest.raises(ValueError):
        CMapDecoder(CODESPACE + mapping)


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_cmap_rejects_unresolved_and_cyclic_parent(
    reader: type[CMapDecoder | ToUnicodeCMap],
) -> None:
    data = b"/Missing usecmap"
    with pytest.raises(ValueError, match="unresolved"):
        reader(data)
    with pytest.raises(ValueError, match="cyclic"):
        reader(data, usecmap_resolver=lambda name: data)


@pytest.mark.parametrize(
    "suffix",
    [b"<unterminated", b"(unterminated", b"[<00>", b"{dup", b"begincmap"],
)
def test_cmap_rejects_incomplete_tokens_and_scope(suffix: bytes) -> None:
    with pytest.raises(ValueError, match="unterminated"):
        CMapProgram.parse(CODESPACE + suffix)


@pytest.mark.parametrize(
    ("include_arrays", "include_words", "block_values", "direct_values"),
    [
        (False, False, [b"<01>", b"(A)"], [b"<01>", b"(A)", b"<02>", b"(B)"]),
        (
            False,
            True,
            [b"<01>", b"(A)", b"word", b"<<", b">>"],
            [b"<01>", b"(A)", b"[", b"<02>", b"(B)", b"]", b"word", b"<<", b">>"],
        ),
        (
            True,
            False,
            [b"<01>", b"(A)", b"[<02> (B)]"],
            [b"<01>", b"(A)", b"[<02> (B)]"],
        ),
        (
            True,
            True,
            [b"<01>", b"(A)", b"[<02> (B)]", b"word", b"<<", b">>"],
            [b"<01>", b"(A)", b"[<02> (B)]", b"word", b"<<", b">>"],
        ),
    ],
)
def test_cmap_token_selection_preserves_array_grouping(
    include_arrays: bool,
    include_words: bool,
    block_values: list[bytes],
    direct_values: list[bytes],
) -> None:
    data = b"<01> (A) [<02> (B)] word << >> {<03>} % <04> ignored\n"
    block = CMapBlock(data, tuple(iter_cmap_tokens(data, group_arrays=True)))
    assert (
        block.token_values(include_arrays=include_arrays, include_words=include_words)
        == block_values
    )
    assert (
        cmap_tokens(data, include_arrays=include_arrays, include_words=include_words)
        == direct_values
    )
    with pytest.raises(ValueError, match="unterminated"):
        cmap_tokens(data + b"<broken", include_arrays=include_arrays, include_words=include_words)


def test_cmap_preserves_spec_defined_identity_and_invalid_code_consumption() -> None:
    cmap = CMapDecoder(b"/Identity-H usecmap")
    assert cmap.decode_entries(b"\x00A\x00") == [(b"\x00A", 65), (b"\x00", 0)]
    assert decode_cmap_hex_token(b"<4>") == b"@"


def test_tounicode_uses_utf16be_and_local_mapping_overrides_parent() -> None:
    parent = CODESPACE + b"1 beginbfchar <01> <0041> endbfchar"
    child = b"/Parent usecmap 1 beginbfchar <01> <D83DDE00> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert cmap.lookup(b"\x01") == "\U0001f600"
    with pytest.raises(UnicodeDecodeError):
        ToUnicodeCMap(CODESPACE + b"1 beginbfchar <01> <41> endbfchar")


def test_adobe_resources_load_with_parent_inheritance() -> None:
    cmap = resolve_cmap_decoder("UniJIS-UTF16-V")
    assert cmap is not None
    assert cmap.wmode == 1
    assert cmap.decode_entries(b"\x00A")[0][1] > 0


def test_static_annex_d_glyph_names_preserve_exact_character_slots() -> None:
    standard = BASE_ENCODING_GLYPH_NAMES["StandardEncoding"]
    mac = BASE_ENCODING_GLYPH_NAMES["MacRomanEncoding"]
    win = BASE_ENCODING_GLYPH_NAMES["WinAnsiEncoding"]
    assert len(standard) == len(mac) == len(win) == 256
    assert standard[39] == "quoteright"
    assert (mac[0], mac[0xCA], mac[0xDB], mac[0xF0]) == (
        ".notdef",
        "space",
        "currency",
        ".notdef",
    )
    assert (win[0x7F], win[0xAD], win[0xB2]) == ("bullet", "hyphen", "twosuperior")


def test_cff_parses_without_font_backend_and_keeps_standard_encoding() -> None:
    # Header, one name, one Top DICT (CharStrings offset 21), empty String
    # and Global Subr indexes, then one .notdef charstring containing endchar.
    font = CFFFont(
        b"\x01\x00\x04\x04"
        b"\x00\x01\x01\x01\x02F"
        b"\x00\x01\x01\x01\x03\xa0\x11"
        b"\x00\x00\x00\x00"
        b"\x00\x01\x01\x01\x02\x0e"
    )
    assert font.charstrings == [b"\x0e"]
    assert font.glyph_id_for_name("missing") == 0
    assert font.builtin_encoding()[65] == "A"
    assert font.font_matrix(0) == (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
    with pytest.raises(ValueError, match="charset"):
        font.read_charset(len(font.data), 2)
    with pytest.raises(ValueError, match="encoding"):
        font.read_encoding_codes(len(font.data))


def test_type2_operators_emit_exact_displacements() -> None:
    events: list[tuple[str, tuple[float, ...]]] = []
    assert internal_execute_type2(bytes([149, 159, 21, 169, 139, 5, 14]), events) is False
    assert events == [("move", (10.0, 20.0)), ("line", (30.0, 0.0)), ("flush", ())]


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (6, [(10.0, 0.0), (0.0, 20.0), (30.0, 0.0), (0.0, 40.0)]),
        (7, [(0.0, 10.0), (20.0, 0.0), (0.0, 30.0), (40.0, 0.0)]),
    ],
    ids=["hlineto", "vlineto"],
)
@pytest.mark.parametrize("count", [3, 4], ids=["odd", "even"])
def test_type2_axis_lines_alternate_and_clear_operands(
    operator: int, expected: list[tuple[float, float]], count: int
) -> None:
    # Adobe Type 2 Charstring Format, 4.1: each operand alternates the drawing axis.
    values = [10, 20, 30, 40][:count]
    program = bytes([139, 139, 21, *(value + 139 for value in values), operator, 144, 22, 14])
    events: list[tuple[str, tuple[float, ...]]] = []
    assert internal_execute_type2(program, events) is False
    assert events == [
        ("move", (0.0, 0.0)),
        *(("line", point) for point in expected[:count]),
        ("move", (5.0, 0.0)),
        ("flush", ()),
    ]


@pytest.mark.parametrize("operator", [6, 7])
@pytest.mark.parametrize("prefix", [b"\x95", b"\x8b\x8b\x15"], ids=["no-point", "no-operands"])
def test_type2_axis_lines_reject_missing_point_or_operands(operator: int, prefix: bytes) -> None:
    with pytest.raises(ValueError, match="invalid Type 2"):
        internal_execute_type2(prefix + bytes([operator]), [])


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
@pytest.mark.parametrize(("count", "bias"), [(1, 107), (1240, 1131), (33900, 32768)])
def test_type2_subroutines_share_operands_and_use_their_own_bias(
    operator: int, count: int, bias: int
) -> None:
    # Adobe Type 2 Charstring Format, 4.7: calls consume only their index, sharing the stack.
    subrs = (bytes([5, 153, 154, 11]),) + (b"\x0b",) * (count - 1)
    local, global_ = (subrs, (b"\x00",)) if operator == 10 else ((b"\x00",), subrs)
    program = bytes([149, 159, 21, 169, 179, 28]) + (-bias).to_bytes(2, "big", signed=True)
    events: list[tuple[str, tuple[float, ...]]] = []
    assert (
        internal_execute_type2(
            program + bytes([operator, 5, 14]), events, local_subrs=local, global_subrs=global_
        )
        is False
    )
    assert events == [
        ("move", (10.0, 20.0)),
        ("line", (30.0, 40.0)),
        ("line", (14.0, 15.0)),
        ("flush", ()),
    ]


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
@pytest.mark.parametrize(
    "operand",
    [b"", b"\x8b", b"\xfb\x00", b"\xff\x00\x00\x80\x00"],
    ids=["missing", "above-range", "below-range", "fractional"],
)
def test_type2_subroutines_reject_invalid_indices(operator: int, operand: bytes) -> None:
    with pytest.raises(ValueError, match="invalid Type 2"):
        internal_execute_type2(
            operand + bytes([operator]), [], local_subrs=(b"\x0b",), global_subrs=(b"\x0b",)
        )


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
@pytest.mark.parametrize("depth", [10, 11])
def test_type2_subroutine_depth_limit(operator: int, depth: int) -> None:
    subrs = tuple(bytes([33 + index, operator, 11]) for index in range(depth - 1)) + (b"\x0b",)
    program = bytes([32, operator, 14])
    if depth == 10:
        assert internal_execute_type2(program, [], local_subrs=subrs, global_subrs=subrs) is False
    else:
        with pytest.raises(ValueError, match="invalid Type 2"):
            internal_execute_type2(program, [], local_subrs=subrs, global_subrs=subrs)


@pytest.mark.parametrize("operators", [(10,), (29,), (10, 10), (10, 29), (29, 10), (29, 29)])
def test_type2_subroutine_endchar_completes_all_enclosing_calls(operators: tuple[int, ...]) -> None:
    # Adobe Type 2 Charstring Format, 2.3 and 4.2 note 6: endchar may terminate a subroutine.
    subrs = {10: [b"\x00"] * len(operators), 29: [b"\x00"] * len(operators)}
    for index, operator in enumerate(operators):
        subrs[operator][index] = (
            bytes([159, 6, 14])
            if index == len(operators) - 1
            else bytes([33 + index, operators[index + 1], 149, 6, 11])
        )
    events: list[tuple[str, tuple[float, ...]]] = []
    assert (
        internal_execute_type2(
            bytes([139, 139, 21, 32, operators[0], 149, 139, 21, 149, 6, 14]),
            events,
            local_subrs=tuple(subrs[10]),
            global_subrs=tuple(subrs[29]),
        )
        is False
    )
    assert events == [("move", (0.0, 0.0)), ("line", (20.0, 0.0)), ("flush", ())]


def test_type1_byte_primitives_reject_truncation_and_preserve_unencrypted_charstrings() -> None:
    assert decode_charstring(b"\x8b\x0e", -1) == b"\x8b\x0e"
    with pytest.raises(ValueError, match="prefix"):
        decode_charstring(b"\x00", 4)
    with pytest.raises(ValueError, match="odd"):
        decode_eexec_payload(b"0000000000", 1)
    with pytest.raises(ValueError, match="truncated"):
        list(binary_entries(b"/A 5 RD abc", re.compile(rb"/(\w+) (\d+) RD ")))


def test_malformed_font_descriptor_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="descriptor"):
        prepare_font_program_inputs({"Subtype": "Type1", "FontDescriptor": 42})
    with pytest.raises(ValueError, match="stream"):
        prepare_font_program_inputs({"Subtype": "Type1", "FontDescriptor": {"FontFile": 42}})


def test_missing_width_is_spec_defined_zero() -> None:
    assert parse_font_widths({}, "Type1").default_width == 0.0
    assert parse_font_widths(
        {"FontDescriptor": {"MissingWidth": 0}}, "Type1"
    ).default_width_explicit


def test_fonttools_backend_is_not_exposed_by_spec() -> None:
    assert importlib.util.find_spec("core_pdf_spec.s_09_fonts.font_program_opentype") is None
    module = importlib.import_module("core_pdf_spec.s_09_fonts.font_program_type1")
    assert not hasattr(module, "Type1FontProgram")


@pytest.mark.parametrize("offset", [21.5, -1.0, float("nan")])
def test_cff_rejects_non_integral_or_invalid_offsets(offset: float) -> None:
    font = CFFFont.__new__(CFFFont)
    font.top_dict = {17: [offset]}
    with pytest.raises(ValueError, match="offset"):
        font.dict_offset(17)


def test_cff_rejects_invalid_header_fields() -> None:
    with pytest.raises(ValueError, match="header"):
        CFFFont(b"\x01\x00\x02\x00")


@pytest.mark.parametrize("entry", [{256: "A"}, {65: ""}])
def test_simple_encoding_rejects_malformed_explicit_entries(entry: dict[int, str]) -> None:
    with pytest.raises(ValueError, match="entry"):
        build_simple_encoding_glyph_names(None, entry, {}, authoritative_builtin=False)


@pytest.mark.parametrize("descendants", [[], [{}, {}], [42]])
def test_descendant_font_array_must_contain_one_dictionary(descendants: list[object]) -> None:
    with pytest.raises(ValueError, match="DescendantFonts"):
        get_descendant({"DescendantFonts": descendants})


def test_cid_bounds_are_supported_exports() -> None:
    from core_pdf_spec.s_09_fonts import widths

    assert (widths.MIN_CID, widths.MAX_CID) == (0, 65535)
    assert {"MIN_CID", "MAX_CID"} <= set(widths.__all__)


def test_cff_dict_rejects_unconsumed_operands() -> None:
    font = CFFFont.__new__(CFFFont)
    assert font.read_dict_entries(b"\x8b\x11\x8c") == ({17: [0.0]}, [1.0])
    with pytest.raises(ValueError, match="unterminated"):
        font.parse_dict(b"\x8b\x11\x8c")


def test_explicit_font_widths_require_a_character_range() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_font_widths({"Widths": [500]}, "Type1")


def test_cid_widths_reject_nonfinite_numbers_in_compact_array() -> None:
    from core_pdf_spec.s_09_fonts.widths import parse_cid_widths

    with pytest.raises(ValueError, match="CID width"):
        parse_cid_widths([0, [float("nan")]])


def test_resource_loader_preserves_cycle_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_pdf_spec.s_09_fonts import cmap_resources

    requests: list[str] = []

    def resource(name: str) -> bytes:
        requests.append(name)
        return b"/Loop usecmap"

    monkeypatch.setattr(cmap_resources, "resolve_cmap_resource", resource)
    with pytest.raises(ValueError, match="cyclic"):
        cmap_resources.resolve_cmap_decoder("Loop")
    assert requests == ["Loop", "Loop"]


def test_resource_loader_keeps_inheritance_and_local_writing_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_pdf_spec.s_09_fonts import cmap_resources

    resources = {
        "Grandparent": CODESPACE + b"/WMode 1 def 1 begincidchar <01> 7 endcidchar",
        "Parent": b"/Grandparent usecmap 1 begincidchar <02> 8 endcidchar",
        "Child": b"/Parent usecmap /WMode 0 def 1 begincidchar <01> 9 endcidchar",
    }
    monkeypatch.setattr(cmap_resources, "resolve_cmap_resource", resources.get)
    cmap = cmap_resources.resolve_cmap_decoder("Child")
    assert cmap is not None
    assert cmap.wmode == 0
    assert cmap.decode_entries(b"\x01\x02") == [(b"\x01", 9), (b"\x02", 8)]


def test_cmap_resource_names_are_already_decoded() -> None:
    from core_pdf_spec.s_09_fonts.cmap_resources import resolve_cmap_resource

    assert resolve_cmap_decoder("Identity-H") is not None
    assert resolve_cmap_decoder("/Identity-H") is None
    assert resolve_cmap_resource("UniJIS-UTF16-H") is not None
    assert resolve_cmap_resource("/UniJIS-UTF16-H") is None
