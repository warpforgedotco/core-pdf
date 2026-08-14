# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, cast

import pytest

from core_pdf.impl.engine.spec.s_07_content.operations import (
    OperandWindow,
    content_stream_may_show_text,
    count_content_stream_operators,
    dispatch_operations,
    iter_content_operations,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer


@pytest.mark.parametrize(
    "data",
    [
        b"(Tj TJ Do ' \\\" (nested Tj))",
        b"<546a 544a 446f>",
        b"% Tj TJ Do ' \\\"\nq Q",
        b"/Tj /TJ /Do /' /\\\" gs",
        b"[Tj TJ Do ' \\\"] q Q",
        b"<< /Text Tj /XObject Do /Quote ' >> q Q",
        b"prefixTjsuffix prefixDosuffix",
        b"BI /F /A85 ID Tj TJ Do ' \\\"~>\nEI Q",
    ],
)
def test_content_stream_may_show_text_ignores_operand_and_image_bytes(data: bytes) -> None:
    assert not content_stream_may_show_text(data)


@pytest.mark.parametrize(
    "data",
    [
        b"(hello) Tj",
        b"[(hello)] TJ",
        b"/XObject Do",
        b"() '",
        b'0 0 () "',
    ],
)
def test_content_stream_may_show_text_accepts_delimited_operators(data: bytes) -> None:
    assert content_stream_may_show_text(data)


def test_content_stream_may_show_text_supports_sliced_memoryview() -> None:
    content = b"(embedded Tj) q Q"
    data = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]

    assert not content_stream_may_show_text(data)


def test_content_stream_may_show_text_supports_reversed_memoryview() -> None:
    content = b"(embedded Tj) q Q"

    assert not content_stream_may_show_text(memoryview(content[::-1])[::-1])


def test_content_stream_may_show_text_finds_operator_after_container() -> None:
    assert content_stream_may_show_text(b"[(embedded Tj)] TJ")


def test_content_operations_do_not_treat_vertical_tab_as_whitespace() -> None:
    operations = list(iter_content_operations(PdfLexer(b"Tj\vDo")))

    assert operations == [("Tj\vDo", ())]


def test_content_operations_treat_null_as_pdf_whitespace() -> None:
    operations = list(iter_content_operations(PdfLexer(b"Tj\0Do")))

    assert operations == [("Tj", ()), ("Do", ())]


def test_content_operations_support_reversed_memoryview() -> None:
    content = b"q Q"

    assert list(iter_content_operations(PdfLexer(memoryview(content[::-1])[::-1]))) == [
        ("q", ()),
        ("Q", ()),
    ]


@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"q \t% fake Do\n\r % fake Tj\r\n Q", [("q", ()), ("Q", ())]),
        (b"q % fake Do", [("q", ())]),
    ],
)
def test_content_operations_skip_comments_after_whitespace(
    view_kind: str,
    content: bytes,
    expected: list[tuple[str, tuple[object, ...]]],
) -> None:
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content

    assert list(iter_content_operations(PdfLexer(data))) == expected


def test_content_operator_counts_ignore_operand_and_inline_image_bytes() -> None:
    data = b"(Tj) % Do\nBI /W 1 /H 1 /BPC 8 ID Tj Do S EI q Q"

    counts = count_content_stream_operators(data)

    assert counts.text == 0
    assert counts.image == 1
    assert counts.graphics_state == 2


def test_content_operator_counts_group_text_image_and_vector_operators() -> None:
    counts = count_content_stream_operators(
        b"BT /F1 12 Tf (hello) Tj ET q 0 0 10 10 re f /Im1 Do Q"
    )

    assert counts.text >= 4
    assert counts.image == 1
    assert counts.vector_path == 1
    assert counts.vector_paint == 1


class internal_RecordingOperationTarget:
    """Minimal bound ``OperationTarget`` that records every dispatched call.

    Pins the exact behavior of ``dispatch_operations``'s bound two-byte and
    one-byte operator fast paths (Stage C / Stage E), including the
    guard-fails-falls-through-silently edge cases, ahead of converting those
    chains to ``match``/``case``. Methods taking an ``OperandWindow`` record
    only the call name and depth -- the window's contents are incidental to
    the dispatch logic under test, not part of what the refactor touches.
    Methods with explicit typed args (the "_values" fast paths) record the
    full argument tuple, since that's exactly what the fast paths compute.
    """

    def __init__(
        self,
        *,
        capture_graphics: bool = True,
        capture_glyphs: bool = True,
        capture_clipping: bool = True,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.capture_graphics = capture_graphics
        self.capture_glyphs = capture_glyphs
        self.capture_clipping = capture_clipping
        self.current_decoder: object | None = object()

    def get_decoder(self, *, update_metrics: bool = True) -> object:
        self.calls.append(("get_decoder", None))
        return self.current_decoder

    def append_text(
        self,
        operand: object = None,
        *,
        data: object = None,
        decoder: object = None,
        string_syntax: str | None = None,
        compatibility_data: bytes | None = None,
    ) -> None:
        del string_syntax, compatibility_data
        self.calls.append(("append_text", (operand, data, decoder is not None)))

    def append_tj_array(self, array: object) -> None:
        self.calls.append(("append_tj_array", array))

    def op_BT(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_BT", depth))

    def op_TD(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_TD", depth))

    def op_TD_values(self, tx: float, ty: float) -> None:
        self.calls.append(("op_TD_values", (tx, ty)))

    def op_Tc(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_Tc", depth))

    def op_Tc_values(self, char_space: float) -> None:
        self.calls.append(("op_Tc_values", (char_space,)))

    def op_Tf(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_Tf", depth))

    def op_Tf_values(self, font_operand: object, font_size_operand: object) -> None:
        self.calls.append(("op_Tf_values", (font_operand, font_size_operand)))

    def op_Tm(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_Tm", depth))

    def op_Tm_values(self, a: float, b: float, c: float, d_: float, e: float, f: float) -> None:
        self.calls.append(("op_Tm_values", (a, b, c, d_, e, f)))

    def op_Tw(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_Tw", depth))

    def op_Tw_values(self, word_space: float) -> None:
        self.calls.append(("op_Tw_values", (word_space,)))

    def op_Td(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_Td", depth))

    def op_Td_values(self, tx: float, ty: float) -> None:
        self.calls.append(("op_Td_values", (tx, ty)))

    def op_re(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_re", depth))

    def op_re_values(
        self, x: int | float, y: int | float, width: int | float, height: int | float
    ) -> None:
        self.calls.append(("op_re_values", (x, y, width, height)))

    def op_RG_values(self, red: int | float, green: int | float, blue: int | float) -> None:
        self.calls.append(("op_RG_values", (red, green, blue)))

    def op_rg_values(self, red: int | float, green: int | float, blue: int | float) -> None:
        self.calls.append(("op_rg_values", (red, green, blue)))

    def op_w_value(self, line_width: int | float) -> None:
        self.calls.append(("op_w_value", (line_width,)))

    def op_J_value(self, line_cap: int | float) -> None:
        self.calls.append(("op_J_value", (line_cap,)))

    def op_j_value(self, line_join: int | float) -> None:
        self.calls.append(("op_j_value", (line_join,)))

    def op_M_value(self, miter_limit: int | float) -> None:
        self.calls.append(("op_M_value", (miter_limit,)))

    def op_m_values(self, x: int | float, y: int | float) -> None:
        self.calls.append(("op_m_values", (x, y)))

    def op_l_values(self, x: int | float, y: int | float) -> None:
        self.calls.append(("op_l_values", (x, y)))

    def op_paint_stroke(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_paint_stroke", depth))

    def op_paint_fill(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_paint_fill", depth))

    def op_ET(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_ET", depth))

    def op_cm(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_cm", depth))

    def op_cm_values(self, a: float, b: float, c: float, d_: float, e: float, f: float) -> None:
        self.calls.append(("op_cm_values", (a, b, c, d_, e, f)))

    def op_q(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_q", depth))

    def op_Q(self, operands: OperandWindow, depth: int) -> None:
        self.calls.append(("op_Q", depth))


def internal_run_dispatch(
    data: bytes, target: internal_RecordingOperationTarget | None = None
) -> internal_RecordingOperationTarget:
    """Dispatch ``data`` against a fresh (or given) recording target.

    Every handler table is empty/all-None, so any token that falls through
    Stage C/D/E without a fast path taking it reaches Stage F, finds no
    handler there either, and is silently dropped -- exactly like production
    behavior for an operator with no registered handler.
    """
    if target is None:
        target = internal_RecordingOperationTarget()
    single_op_handlers: list[object] = [None] * 65536
    cast(Any, dispatch_operations)(PdfLexer(data), {}, None, single_op_handlers, {}, target, 0)
    return target


# ---- Stage E: bound one-byte operator chain (w, J, j, M, m, l, S, f, F, q, Q) ----


def test_stage_e_value_fast_paths_fire_with_sufficient_operands() -> None:
    target = internal_run_dispatch(b"5 w 1 J 2 j 10 M 1 2 m 3 4 l S f F q Q")

    assert target.calls == [
        ("op_w_value", (5,)),
        ("op_J_value", (1,)),
        ("op_j_value", (2,)),
        ("op_M_value", (10,)),
        ("op_m_values", (1, 2)),
        ("op_l_values", (3, 4)),
        ("op_paint_stroke", 0),
        ("op_paint_fill", 0),
        ("op_paint_fill", 0),
        ("op_q", 0),
        ("op_Q", 0),
    ]


@pytest.mark.parametrize("token", [b"w", b"J", b"j", b"M", b"m", b"1 m", b"l", b"1 l"])
def test_stage_e_guarded_operators_fall_through_silently_without_operands(
    token: bytes,
) -> None:
    target = internal_run_dispatch(token)

    assert target.calls == []


# ---- Stage C: bound two-byte operator chain (T*, RG, rg, re, ET, cm) ----


def test_stage_c_bt_et_fire_unconditionally() -> None:
    target = internal_run_dispatch(b"BT ET")

    assert target.calls == [("op_BT", 0), ("op_ET", 0)]


def test_stage_c_td_value_fast_path() -> None:
    target = internal_run_dispatch(b"1 2 TD")

    assert target.calls == [("op_TD_values", (1, 2))]


@pytest.mark.parametrize("data", [b"1 TD", b"/Foo 2 TD"])
def test_stage_c_td_falls_back_to_generic_handler(data: bytes) -> None:
    target = internal_run_dispatch(data)

    assert target.calls == [("op_TD", 0)]


def test_stage_c_tc_value_fast_path() -> None:
    target = internal_run_dispatch(b"5 Tc")

    assert target.calls == [("op_Tc_values", (5,))]


def test_stage_c_tc_without_operand_falls_through_silently() -> None:
    target = internal_run_dispatch(b"Tc")

    assert target.calls == []


def test_stage_c_tc_wrong_type_falls_back_to_generic_handler() -> None:
    target = internal_run_dispatch(b"/Foo Tc")

    assert target.calls == [("op_Tc", 0)]


def test_stage_c_tf_value_fast_path_has_no_type_check() -> None:
    target = internal_run_dispatch(b"/F1 12 Tf")

    assert len(target.calls) == 1
    name, args = target.calls[0]
    assert name == "op_Tf_values"
    assert args[1] == 12


def test_stage_c_tf_falls_back_to_generic_handler_with_too_few_operands() -> None:
    target = internal_run_dispatch(b"Tf")

    assert target.calls == [("op_Tf", 0)]


def test_stage_c_tj_calls_append_text() -> None:
    target = internal_run_dispatch(b"(hello) Tj")

    assert target.calls == [("append_text", (None, b"hello", True))]


def test_stage_c_tj_without_operand_falls_through_silently() -> None:
    target = internal_run_dispatch(b"Tj")

    assert target.calls == []


def test_stage_c_tm_value_fast_path() -> None:
    target = internal_run_dispatch(b"1 2 3 4 5 6 Tm")

    assert target.calls == [("op_Tm_values", (1, 2, 3, 4, 5, 6))]


@pytest.mark.parametrize("data", [b"1 2 3 4 5 Tm", b"/Foo 2 3 4 5 6 Tm"])
def test_stage_c_tm_falls_back_to_generic_handler(data: bytes) -> None:
    target = internal_run_dispatch(data)

    assert target.calls == [("op_Tm", 0)]


def test_stage_c_tw_value_fast_path() -> None:
    target = internal_run_dispatch(b"2 Tw")

    assert target.calls == [("op_Tw_values", (2,))]


def test_stage_c_tw_without_operand_falls_through_silently() -> None:
    target = internal_run_dispatch(b"Tw")

    assert target.calls == []


def test_stage_c_tj_array_calls_append_tj_array() -> None:
    target = internal_run_dispatch(b"[(hi)] TJ")

    assert len(target.calls) == 1
    assert target.calls[0][0] == "append_tj_array"


def test_stage_c_tj_array_without_operand_falls_through_silently() -> None:
    target = internal_run_dispatch(b"TJ")

    assert target.calls == []


def test_stage_c_td_lowercase_value_fast_path() -> None:
    target = internal_run_dispatch(b"1 2 Td")

    assert target.calls == [("op_Td_values", (1, 2))]


@pytest.mark.parametrize("data", [b"1 Td", b"/Foo 2 Td"])
def test_stage_c_td_lowercase_falls_back_to_generic_handler(data: bytes) -> None:
    target = internal_run_dispatch(data)

    assert target.calls == [("op_Td", 0)]


def test_stage_c_rg_uppercase_value_fast_path() -> None:
    target = internal_run_dispatch(b"1 0 0 RG")

    assert target.calls == [("op_RG_values", (1, 0, 0))]


def test_stage_c_rg_uppercase_falls_through_silently_with_too_few_operands() -> None:
    # No generic `op_RG` exists on OperationTarget -- an unmet guard must
    # fall through to Stage D/E/F and call nothing, not raise.
    target = internal_run_dispatch(b"1 0 RG")

    assert target.calls == []


def test_stage_c_rg_lowercase_value_fast_path() -> None:
    target = internal_run_dispatch(b"1 0 0 rg")

    assert target.calls == [("op_rg_values", (1, 0, 0))]


def test_stage_c_rg_lowercase_falls_through_silently_with_too_few_operands() -> None:
    target = internal_run_dispatch(b"1 0 rg")

    assert target.calls == []


def test_stage_c_re_value_fast_path_when_capturing() -> None:
    target = internal_run_dispatch(b"0 0 10 10 re")

    assert target.calls == [("op_re_values", (0, 0, 10, 10))]


def test_stage_c_re_falls_back_to_generic_handler_with_too_few_operands() -> None:
    target = internal_run_dispatch(b"0 0 10 re")

    assert target.calls == [("op_re", 0)]


def test_stage_c_re_calls_nothing_when_not_capturing() -> None:
    target = internal_run_dispatch(
        b"0 0 10 10 re",
        internal_RecordingOperationTarget(
            capture_graphics=False, capture_glyphs=False, capture_clipping=False
        ),
    )

    assert target.calls == []


def test_stage_c_cm_value_fast_path() -> None:
    target = internal_run_dispatch(b"1 0 0 1 0 0 cm")

    assert target.calls == [("op_cm_values", (1, 0, 0, 1, 0, 0))]


def test_stage_c_cm_wrong_type_falls_back_to_generic_handler() -> None:
    target = internal_run_dispatch(b"/Foo 0 0 1 0 0 cm")

    assert target.calls == [("op_cm", 0)]


def test_stage_c_cm_falls_through_silently_with_too_few_operands() -> None:
    # The `op_count >= 6` guard sits in the outer `elif` condition itself, so
    # an unmet guard skips the whole branch -- no generic `op_cm` fallback is
    # reachable here, unlike the too-few-operands cases above.
    target = internal_run_dispatch(b"1 0 0 1 0 cm")

    assert target.calls == []
