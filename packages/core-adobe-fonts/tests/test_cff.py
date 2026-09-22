# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, cast

import pytest

from core_adobe_fonts.cff.charstrings import execute_type2_charstring
from core_adobe_fonts.cff.font import (
    DEFAULT_CFF_FONT_MATRIX,
    CFFFont,
    CffFontMatrix,
    cff_font_matrix,
)


def execute_type2(
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


def type2_operands(*values: float) -> bytes:
    return b"".join(
        b"\x1c" + int(value).to_bytes(2, "big", signed=True)
        if value == int(value)
        else b"\xff" + int(value * 65536).to_bytes(4, "big", signed=True)
        for value in values
    )


def test_cff_parses_without_font_backend_and_keeps_standard_encoding() -> None:
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
    assert execute_type2(bytes([149, 159, 21, 169, 139, 5, 14]), events) is False
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
    values = [10, 20, 30, 40][:count]
    program = bytes([139, 139, 21, *(value + 139 for value in values), operator, 144, 22, 14])
    events: list[tuple[str, tuple[float, ...]]] = []
    assert execute_type2(program, events) is False
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
        execute_type2(prefix + bytes([operator]), [])


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
@pytest.mark.parametrize(("count", "bias"), [(1, 107), (1240, 1131), (33900, 32768)])
def test_type2_subroutines_share_operands_and_use_their_own_bias(
    operator: int, count: int, bias: int
) -> None:
    subrs = (bytes([5, 153, 154, 11]),) + (b"\x0b",) * (count - 1)
    local, global_ = (subrs, (b"\x00",)) if operator == 10 else ((b"\x00",), subrs)
    program = bytes([149, 159, 21, 169, 179, 28]) + (-bias).to_bytes(2, "big", signed=True)
    events: list[tuple[str, tuple[float, ...]]] = []
    assert (
        execute_type2(
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
        execute_type2(
            operand + bytes([operator]), [], local_subrs=(b"\x0b",), global_subrs=(b"\x0b",)
        )


@pytest.mark.parametrize("operator", [10, 29], ids=["local", "global"])
@pytest.mark.parametrize("depth", [10, 11])
def test_type2_subroutine_depth_limit(operator: int, depth: int) -> None:
    subrs = tuple(bytes([33 + index, operator, 11]) for index in range(depth - 1)) + (b"\x0b",)
    program = bytes([32, operator, 14])
    if depth == 10:
        assert execute_type2(program, [], local_subrs=subrs, global_subrs=subrs) is False
    else:
        with pytest.raises(ValueError, match="invalid Type 2"):
            execute_type2(program, [], local_subrs=subrs, global_subrs=subrs)


@pytest.mark.parametrize("operators", [(10,), (29,), (10, 10), (10, 29), (29, 10), (29, 29)])
def test_type2_subroutine_endchar_completes_all_enclosing_calls(operators: tuple[int, ...]) -> None:
    subrs = {10: [b"\x00"] * len(operators), 29: [b"\x00"] * len(operators)}
    for index, operator in enumerate(operators):
        subrs[operator][index] = (
            bytes([159, 6, 14])
            if index == len(operators) - 1
            else bytes([33 + index, operators[index + 1], 149, 6, 11])
        )
    events: list[tuple[str, tuple[float, ...]]] = []
    assert (
        execute_type2(
            bytes([139, 139, 21, 32, operators[0], 149, 139, 21, 149, 6, 14]),
            events,
            local_subrs=tuple(subrs[10]),
            global_subrs=tuple(subrs[29]),
        )
        is False
    )
    assert events == [("move", (0.0, 0.0)), ("line", (20.0, 0.0)), ("flush", ())]


@pytest.mark.parametrize("offset", [21.5, -1.0, float("nan")])
def test_cff_rejects_non_integral_or_invalid_offsets(offset: float) -> None:
    font = CFFFont.__new__(CFFFont)
    font.top_dict = {17: [offset]}
    with pytest.raises(ValueError, match="offset"):
        font.dict_offset(17)


def test_cff_rejects_invalid_header_fields() -> None:
    with pytest.raises(ValueError, match="header"):
        CFFFont(b"\x01\x00\x02\x00")


def test_cff_dict_rejects_unconsumed_operands() -> None:
    font = CFFFont.__new__(CFFFont)
    assert font.read_dict_entries(b"\x8b\x11\x8c") == ({17: [0.0]}, [1.0])
    with pytest.raises(ValueError, match="unterminated"):
        font.parse_dict(b"\x8b\x11\x8c")


@pytest.mark.parametrize(
    ("operator", "operands", "expected"),
    [
        (3, (2, 3), 1),
        (3, (0, 3), 0),
        (3, (2, 0), 0),
        (4, (2, 0), 1),
        (4, (0, 3), 1),
        (4, (0, 0), 0),
        (5, (0,), 1),
        (5, (-2,), 0),
        (9, (-5,), 5),
        (10, (4, 7), 11),
        (11, (4, 7), -3),
        (12, (3, 2), 1.5),
        (14, (5,), -5),
        (15, (3, 3), 1),
        (15, (3, 2), 0),
        (22, (7, 9, 1, 2), 7),
        (22, (7, 9, 2, 1), 9),
        (22, (7, 9, 2, 2), 7),
        (23, (), 0.5),
        (24, (-3, 2), -6),
        (26, (9,), 3),
    ],
)
def test_type2_arithmetic_and_logic_drive_emitted_geometry(operator, operands, expected):
    program = type2_operands(*operands) + bytes([12, operator, 22, 14])
    events = []
    assert execute_type2(program, events) is False
    assert events == [("move", (expected, 0)), ("flush", ())]


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        (type2_operands(4) + b"\x0c\x1b", (4, 4)),
        (type2_operands(4, 7) + b"\x0c\x1c", (7, 4)),
        (type2_operands(4, 7, 8) + b"\x0c\x12", (4, 7)),
        (type2_operands(4, 7, 0) + b"\x0c\x1d\x0c\x0a", (4, 14)),
        (type2_operands(4, 7, -1) + b"\x0c\x1d\x0c\x0a", (4, 14)),
        (type2_operands(4, 7, 1) + b"\x0c\x1d\x0c\x0a", (4, 11)),
        (type2_operands(4, 7, 2, 1) + b"\x0c\x1e", (7, 4)),
        (type2_operands(4, 7, 2, -1) + b"\x0c\x1e", (7, 4)),
        (type2_operands(4, 7, 2, 2) + b"\x0c\x1e", (4, 7)),
        (type2_operands(4, 7, 0, 1) + b"\x0c\x1e", (4, 7)),
        (type2_operands(99) + b"\x0c\0" + type2_operands(4, 7), (4, 7)),
    ],
)
def test_type2_stack_operations_preserve_operand_order(program, expected):
    events = []
    execute_type2(program + b"\x15\x0e", events)
    assert events == [("move", expected), ("flush", ())]


@pytest.mark.parametrize("index", [0, 31])
def test_type2_transient_storage_is_shared_with_subroutines_but_reset_per_glyph(index):
    put = type2_operands(23, index) + b"\x0c\x14\x0b"
    get = type2_operands(index) + b"\x0c\x15\x16\x0e"
    events = []
    execute_type2(type2_operands(-107) + b"\x0a" + get, events, local_subrs=(put,))
    assert events == [("move", (23, 0)), ("flush", ())]
    events.clear()
    execute_type2(get, events)
    assert events == [("move", (0, 0)), ("flush", ())]


@pytest.mark.parametrize(
    ("operator", "operands"),
    [
        (12, (1, 0)),
        (26, (-1,)),
        (20, (5, -1)),
        (20, (5, 32)),
        (20, (5, 1.5)),
        (21, (-1,)),
        (21, (32,)),
        (21, (1.5,)),
        (29, (5, 1)),
        (29, (5, 0.5)),
        (30, (5, -1, 1)),
        (30, (5, 2, 1)),
        (30, (5, 1, 0.5)),
        (99, ()),
    ],
)
def test_type2_invalid_arithmetic_and_indices_raise_before_emitting_geometry(operator, operands):
    events = []
    with pytest.raises(ValueError, match="invalid Type 2 charstring"):
        execute_type2(type2_operands(*operands) + bytes([12, operator]), events)
    assert events == []


@pytest.mark.parametrize(
    "operator", [3, 4, 5, 9, 10, 11, 12, 14, 15, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30]
)
def test_type2_stack_underflow_has_consistent_error_family(operator):
    with pytest.raises(ValueError, match="invalid Type 2 charstring"):
        execute_type2(bytes([12, operator]), [])


@pytest.mark.parametrize(
    ("operator", "operands", "curves"),
    [
        (34, (1, 2, 3, 4, 5, 6, 7), [(1, 0, 2, 3, 4, 0), (5, 0, 6, -3, 7, 0)]),
        (35, tuple(range(1, 14)), [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)]),
        (36, tuple(range(1, 10)), [(1, 2, 3, 4, 5, 0), (6, 0, 7, 8, 9, -14)]),
        (37, (5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 7), [(5, 1, 5, 1, 5, 1), (5, 1, 5, 1, 7, -5)]),
        (37, (1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 7), [(1, 5, 1, 5, 1, 5), (1, 5, 1, 5, -5, 7)]),
    ],
)
def test_type2_flex_emits_two_curves_and_clears_operands(operator, operands, curves):
    events = []
    program = type2_operands(0, 0) + b"\x15" + type2_operands(*operands) + bytes([12, operator])
    execute_type2(program + type2_operands(2) + b"\x16\x0e", events)
    assert events == [
        ("move", (0, 0)),
        *(("curve", points) for points in curves),
        ("move", (2, 0)),
        ("flush", ()),
    ]


@pytest.mark.parametrize("operator", [34, 35, 36, 37])
@pytest.mark.parametrize("current_point", [False, True])
def test_type2_flex_rejects_missing_point_or_wrong_arity(operator, current_point):
    prefix = type2_operands(0, 0) + b"\x15" if current_point else b""
    with pytest.raises(ValueError, match="invalid Type 2 charstring"):
        execute_type2(prefix + type2_operands(1, 2) + bytes([12, operator]), [])


@pytest.mark.parametrize("operator", [4, 22])
@pytest.mark.parametrize("width", [(), (50,)])
def test_type2_axis_move_consumes_optional_width_only_on_first_move(operator, width):
    events = []
    execute_type2(
        type2_operands(*width, 7) + bytes([operator]) + type2_operands(9) + bytes([operator, 14]),
        events,
    )
    expected = [(0, 7), (0, 9)] if operator == 4 else [(7, 0), (9, 0)]
    assert events == [("move", expected[0]), ("move", expected[1]), ("flush", ())]
    with pytest.raises(ValueError, match="invalid Type 2 charstring"):
        execute_type2(
            type2_operands(7) + bytes([operator]) + type2_operands(50, 9) + bytes([operator]), []
        )


@pytest.mark.parametrize("operator", [26, 27])
@pytest.mark.parametrize("offset", [(), (9,)])
def test_type2_parallel_tangent_curves_apply_optional_offset_to_first_curve(operator, offset):
    values = (*offset, 1, 2, 3, 4, 5, 6, 7, 8)
    extra = offset[0] if offset else 0
    curves = (
        [(extra, 1, 2, 3, 0, 4), (0, 5, 6, 7, 0, 8)]
        if operator == 26
        else [(1, extra, 2, 3, 4, 0), (5, 0, 6, 7, 8, 0)]
    )
    events = []
    execute_type2(
        type2_operands(0, 0) + b"\x15" + type2_operands(*values) + bytes([operator, 14]), events
    )
    assert events == [("move", (0, 0)), *(("curve", points) for points in curves), ("flush", ())]


@pytest.mark.parametrize("operator", [4, 22, 26, 27])
@pytest.mark.parametrize("count", [0, 3, 6])
def test_type2_axis_operators_reject_wrong_operand_counts(operator, count):
    with pytest.raises(ValueError, match="invalid Type 2 charstring"):
        execute_type2(
            type2_operands(0, 0) + b"\x15" + type2_operands(*range(count)) + bytes([operator]), []
        )


@pytest.mark.parametrize("off_size", [1, 2, 3, 4])
@pytest.mark.parametrize("buffer_type", [bytes, memoryview])
def test_cff_index_preserves_empty_entries_and_returns_exact_next_offset(off_size, buffer_type):
    font = CFFFont.__new__(CFFFont)
    offsets = b"".join(value.to_bytes(off_size, "big") for value in (1, 3, 3, 6))
    font.data = buffer_type(b"prefix" + b"\0\3" + bytes([off_size]) + offsets + b"abcdeTAIL")
    entries, end = font.read_index(6)
    assert entries == [b"ab", b"", b"cde"]
    assert all(type(entry) is bytes for entry in entries)
    assert bytes(font.data[end:]) == b"TAIL"


@pytest.mark.parametrize(
    "encoded",
    [
        b"\0",
        b"\0\1",
        b"\0\1\0",
        b"\0\1\5",
        b"\0\1\1\1",
        b"\0\1\1\0\1",
        b"\0\2\1\1\3\2ab",
        b"\0\1\1\1\3a",
    ],
)
def test_cff_index_rejects_truncation_invalid_offsets_and_sizes(encoded):
    font = CFFFont.__new__(CFFFont)
    font.data = encoded
    with pytest.raises(ValueError, match="CFF INDEX"):
        font.read_index(0)


@pytest.mark.parametrize(
    ("encoded", "dict_number", "expected"),
    [
        (b"\x20", False, -107),
        (b"\xf6", False, 107),
        (b"\xf7\0", False, 108),
        (b"\xfa\xff", False, 1131),
        (b"\xfb\0", False, -108),
        (b"\xfe\xff", False, -1131),
        (b"\x1c\x80\0", False, -32768),
        (b"\x1c\x7f\xff", True, 32767),
        (b"\x1d\x80\0\0\0", True, -2147483648),
        (b"\xff\xff\xff\x80\0", False, -0.5),
        (b"\x1e\xe1\xa2\x5f", True, -1.25),
        (b"\x1e\x1b\x3f", True, 1000),
        (b"\x1e\x1c\x3f", True, 0.001),
    ],
)
def test_cff_number_encodings_preserve_sign_precision_and_consumption(
    encoded, dict_number, expected
):
    value, end = CFFFont.parse_number(b"prefix" + encoded + b"TAIL", 6, dict_number=dict_number)
    assert value == expected
    assert end == 6 + len(encoded)


@pytest.mark.parametrize(
    ("encoded", "dict_number"),
    [
        (b"", False),
        (b"\xf7", False),
        (b"\xfb", False),
        (b"\x1c\0", False),
        (b"\x1d\0", True),
        (b"\xff\0", False),
        (b"\x1e\x12", True),
        (b"\x1e\x1d\xff", True),
        (b"\x1d\0\0\0\0", False),
        (b"\xff\0\0\0\0", True),
    ],
)
def test_cff_numbers_reject_incomplete_and_wrong_context_encodings(encoded, dict_number):
    with pytest.raises(ValueError):
        CFFFont.parse_number(encoded, 0, dict_number=dict_number)


@pytest.mark.parametrize("count", [48, 49])
def test_type2_operand_stack_limit_has_positive_control(count):
    program = type2_operands(*range(count)) + b"\x0c\0\x0e"
    if count == 48:
        assert execute_type2(program, []) is False
    else:
        with pytest.raises(ValueError, match="invalid Type 2 charstring"):
            execute_type2(program, [])


@pytest.mark.parametrize("value", [True, "1", b"1", float("inf"), float("nan"), 10**400])
def test_cff_matrix_numbers_are_strict(value: object) -> None:
    with pytest.raises(ValueError, match="CFF FontMatrix"):
        cff_font_matrix(cast(Any, {(12, 7): [value, 0, 0, 1, 0, 0]}))


@pytest.mark.parametrize("values", [[1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 0], (1, 0, 0, 1, 0, 0)])
def test_cff_matrix_requires_a_six_number_list(values: object) -> None:
    with pytest.raises(ValueError, match="CFF FontMatrix"):
        cff_font_matrix(cast(Any, {(12, 7): values}))


@pytest.mark.parametrize(
    ("top", "child", "expected"),
    [
        (None, None, DEFAULT_CFF_FONT_MATRIX),
        ([2, 1, 3, 4, 5, 6], None, CffFontMatrix(2, 1, 3, 4, 5, 6)),
        (None, [7, 8, 9, 10, 11, 12], CffFontMatrix(7, 8, 9, 10, 11, 12)),
        ([7, 8, 9, 10, 11, 12], [2, 1, 3, 4, 5, 6], CffFontMatrix(23, 26, 57, 64, 100, 112)),
    ],
)
def test_cff_top_and_font_dictionary_matrix_defaults_and_composition(
    top: list[float] | None, child: list[float] | None, expected: CffFontMatrix
) -> None:
    font = CFFFont.__new__(CFFFont)
    font.charstrings = [b"\x0e"]
    font.top_dict = {} if top is None else {(12, 7): top}
    font.fd_select = (0,)
    font.font_dicts = ({},) if child is None else ({(12, 7): child},)
    result = font.font_matrix(0)
    assert isinstance(result, CffFontMatrix)
    assert result == expected
