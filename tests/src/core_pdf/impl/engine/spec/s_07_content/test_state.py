from types import SimpleNamespace
from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_content.state import TextState
from core_pdf.impl.engine.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.objects import PdfStream


def test_distinct_stream_slices_with_equal_lengths_have_distinct_execution_keys() -> None:
    source = memoryview(b"first second")
    first = PdfStream(raw_data=source[:5])
    second = PdfStream(raw_data=source[6:11])

    assert TextState.stream_execution_key(first) != TextState.stream_execution_key(second)
    assert TextState.stream_execution_key(first) == TextState.stream_execution_key(first)


def internal_consume(content: bytes) -> TextState:
    resolver = SimpleNamespace(
        kw_cache={},
        resolve=lambda value: value,
        resolve_dict=lambda value: value if isinstance(value, dict) else None,
        resolve_name=lambda internal_value: None,
        resolve_str=lambda internal_value: None,
    )
    document = cast(Any, SimpleNamespace(resolver=resolver))
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
