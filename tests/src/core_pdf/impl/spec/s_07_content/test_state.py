import threading
from types import SimpleNamespace
from typing import Any, cast

from core_pdf.impl.spec.s_07_content.operations import OperandWindow
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from tests.helpers.resolvers import IdentityResolver


def test_distinct_stream_slices_with_equal_lengths_have_distinct_execution_keys() -> None:
    source = memoryview(b"first second")
    first = PdfStream(raw_data=source[:5])
    second = PdfStream(raw_data=source[6:11])

    assert TextState.stream_execution_key(first) != TextState.stream_execution_key(second)


def internal_consume(content: bytes) -> TextState:
    document = cast(
        Any,
        SimpleNamespace(resolver=IdentityResolver(), internal_cache_lock=threading.RLock()),
    )
    state = TextState(document, {})
    state.consume_stream(PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0)
    return state


def internal_capture_drawing_kinds(content: bytes) -> list[str]:
    state = internal_consume(content)
    assert not state.stack
    assert not state.clip_scope_stack
    return [drawing.kind for drawing in state.drawings]


def internal_capture_path_points(content: bytes) -> list[tuple[float, float]]:
    state = internal_consume(content)
    path = state.drawings[0].path
    assert path is not None
    return list(path.subpaths[0].points)


def test_graphics_state_markers_are_emitted_only_for_clip_scopes() -> None:
    assert internal_capture_drawing_kinds(b"q 0 0 m 1 1 l S Q") == ["stroke"]
    assert internal_capture_drawing_kinds(b"q 0 0 10 10 re W n 0 0 m 1 1 l S Q") == [
        "state-push",
        "clip",
        "stroke",
        "state-pop",
    ]


def test_graphics_state_restore_recomputes_derived_text_scales() -> None:
    state = internal_consume(b"")
    state.font_size = 9.0
    state.horizontal_scale = 80.0
    state.update_text_scales()
    state.op_q(OperandWindow(()), 0)
    state.font_size = 20.0
    state.horizontal_scale = 50.0
    state.update_text_scales()

    state.op_Q(OperandWindow(()), 0)

    assert state.text_advance_scale == 0.0072


def test_graphics_state_save_restore_covers_every_snapshot_field() -> None:
    state = internal_consume(b"")
    fields = (
        "ca",
        "cb",
        "cc",
        "cd",
        "ce",
        "cf",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "fill_color_space",
        "stroke_color_space",
        "compatibility_depth",
        "blend_mode",
        "group_alpha",
        "flatness",
        "render_intent",
        "clip_bbox",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "font_operand",
        "font_size_operand",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
    )
    before = tuple(getattr(state, name) for name in fields)

    state.op_q(OperandWindow(()), 0)
    sentinel = object()
    for name in fields:
        setattr(state, name, sentinel)
    state.op_Q(OperandWindow(()), 0)

    assert tuple(getattr(state, name) for name in fields) == before


def test_pdfminer_double_quote_policy_omits_next_line_move() -> None:
    def state(legacy: bool) -> TextState:
        result = internal_consume(b"")
        cast(Any, result.document).legacy_pdfminer_text_operators = legacy
        result.leading = 12.0
        result.lm_e = result.tm_e = 10.0
        result.lm_f = result.tm_f = 20.0
        return result

    native = state(False)
    legacy = state(True)
    native.op_double_quote(OperandWindow((2, 3, b""), 3), 0)
    legacy.op_double_quote(OperandWindow((2, 3, b""), 3), 0)

    assert (native.tm_e, native.tm_f) == (10.0, 8.0)
    assert (legacy.tm_e, legacy.tm_f) == (10.0, 20.0)


def test_text_showing_consumes_the_top_operand_from_a_malformed_stack() -> None:
    state = internal_consume(b"")

    state.op_Tj(OperandWindow((b"stale", b"visible"), 2), 0)

    assert state.pending_run is not None
    assert state.pending_run.text == "visible"


def test_curve_y_doubles_the_endpoint_as_its_second_control_point() -> None:
    # `x1 y1 x3 y3 y` is the cubic with control points (x1, y1) and (x3, y3).
    curve_y = internal_capture_path_points(b"0 0 m 10 80 90 10 y S")
    equivalent_c = internal_capture_path_points(b"0 0 m 10 80 90 10 90 10 c S")

    assert curve_y == equivalent_c


def test_curve_v_uses_the_current_point_as_its_first_control_point() -> None:
    # `x2 y2 x3 y3 v` is the cubic with control points (x0, y0) and (x2, y2).
    curve_v = internal_capture_path_points(b"0 0 m 10 80 90 10 v S")
    equivalent_c = internal_capture_path_points(b"0 0 m 0 0 10 80 90 10 c S")

    assert curve_v == equivalent_c


def test_curve_y_is_not_interpreted_as_curve_v() -> None:
    curve_y = internal_capture_path_points(b"0 0 m 10 80 90 10 y S")
    curve_v = internal_capture_path_points(b"0 0 m 10 80 90 10 v S")

    assert curve_y[0] == curve_v[0] == (0.0, 0.0)
    assert curve_y[-1] == curve_v[-1] == (90.0, 10.0)
    # Shared endpoints, but `y` bulges toward its first control point.
    midpoint_y = curve_y[len(curve_y) // 2]
    midpoint_v = curve_v[len(curve_v) // 2]
    assert abs(midpoint_y[0] - midpoint_v[0]) > 10.0


def internal_first_drawing(content: bytes) -> Any:
    state = internal_consume(content)
    assert state.drawings
    return state.drawings[0]


def test_starred_paint_operators_record_evenodd_fill_rule() -> None:
    assert internal_first_drawing(b"0 0 10 10 re f").fill_rule == "nonzero"
    assert internal_first_drawing(b"0 0 10 10 re f*").fill_rule == "evenodd"
    assert internal_first_drawing(b"0 0 10 10 re B").fill_rule == "nonzero"
    assert internal_first_drawing(b"0 0 10 10 re B*").fill_rule == "evenodd"
    assert internal_first_drawing(b"0 0 10 10 re b*").fill_rule == "evenodd"


def test_starred_clip_records_evenodd_fill_rule() -> None:
    state = internal_consume(b"q 0 0 10 10 re W* n Q")
    clips = [drawing for drawing in state.drawings if drawing.kind == "clip"]
    assert [clip.fill_rule for clip in clips] == ["evenodd"]

    state = internal_consume(b"q 0 0 10 10 re W n Q")
    clips = [drawing for drawing in state.drawings if drawing.kind == "clip"]
    assert [clip.fill_rule for clip in clips] == ["nonzero"]


def internal_first_subpath_closed(content: bytes) -> bool:
    path = internal_first_drawing(content).path
    assert path is not None
    return bool(path.subpaths[0].closed)


def test_close_stroke_operators_close_the_subpath() -> None:
    assert not internal_first_subpath_closed(b"0 0 m 10 0 l 10 10 l S")
    assert internal_first_subpath_closed(b"0 0 m 10 0 l 10 10 l s")
    assert internal_first_subpath_closed(b"0 0 m 10 0 l 10 10 l b")
