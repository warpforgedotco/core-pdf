# SPDX-License-Identifier: AGPL-3.0-only
"""Path completion and nested-stream event ordering defined by PDF."""

from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import ContentSink, PdfPath
from core_pdf_spec.s_07_content.operations import iter_content_operations
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName


class EventSink:
    def __init__(self) -> None:
        self.paths: list[tuple[str, str, PdfPath]] = []
        self.streams: list[tuple[str, int]] = []
        self.saved = 0

    def __getattr__(self, name: str) -> Any:
        return lambda *args: None

    def paint_path(
        self, state: ContentInterpreter, path: PdfPath, kind: str, fill_rule: str
    ) -> None:
        self.paths.append((kind, fill_rule, path))

    def clip_path(self, state: ContentInterpreter, path: PdfPath, fill_rule: str) -> None:
        self.paths.append(("clip", fill_rule, path))

    def save_graphics(self, state: ContentInterpreter) -> None:
        self.saved += 1

    def restore_graphics(self, state: ContentInterpreter) -> None:
        self.saved -= 1

    def enter_stream(self, state: ContentInterpreter, frame: ContentStreamFrame) -> None:
        self.streams.append(("enter", frame.depth))

    def exit_stream(self, state: ContentInterpreter, frame: ContentStreamFrame) -> None:
        self.streams.append(("exit", frame.depth))

    def text_boundary(self, state: ContentInterpreter, reason: str) -> None:
        if reason == "stream-end":
            self.streams.append(("end", state.xobject_depth))


def new_state() -> tuple[ContentInterpreter, EventSink]:
    sink = EventSink()
    state = ContentInterpreter(
        ObjectResolver(b"", {}),
        cast(ContentSink, sink),
        cast(Any, lambda *args: None),
    )
    return state, sink


@pytest.mark.parametrize(("clip", "clip_rule"), [(b"W", "nonzero"), (b"W*", "evenodd")])
@pytest.mark.parametrize(
    ("operator", "kind", "fill_rule", "closed"),
    [
        (b"S", "stroke", "nonzero", False),
        (b"s", "stroke", "nonzero", True),
        (b"f", "fill", "nonzero", False),
        (b"F", "fill", "nonzero", False),
        (b"f*", "fill", "evenodd", False),
        (b"B", "fillstroke", "nonzero", False),
        (b"B*", "fillstroke", "evenodd", False),
        (b"b", "fillstroke", "nonzero", True),
        (b"b*", "fillstroke", "evenodd", True),
        (b"n", None, None, False),
    ],
)
def test_path_paints_before_installing_clip_and_retains_geometry(
    clip: bytes,
    clip_rule: str,
    operator: bytes,
    kind: str | None,
    fill_rule: str | None,
    closed: bool,
) -> None:
    # ISO 32000-1 8.5.4: W/W* modify clipping after the terminating paint or n.
    state, sink = new_state()
    for name, operands in iter_content_operations(PdfLexer(b"0 0 m 10 0 l 0 10 l " + clip)):
        state.execute_operation(name, operands, 0)
    path = state.current_path
    assert sink.paths == []
    for name, operands in iter_content_operations(PdfLexer(operator)):
        state.execute_operation(name, operands, 0)
    expected = [] if kind is None else [(kind, fill_rule, path)]
    assert sink.paths == [*expected, ("clip", clip_rule, path)]
    assert [command.operator for command in path.commands] == ["m", "l", "l"] + (
        ["h"] if closed else []
    )
    assert state.current_path is not path
    assert not state.current_path.commands
    assert state.current_point is state.subpath_start is None
    for name, operands in iter_content_operations(PdfLexer(b"0 0 1 1 re f")):
        state.execute_operation(name, operands, 0)
    assert [event[0] for event in sink.paths].count("clip") == 1


def test_empty_completion_and_stroke_clear_the_current_point() -> None:
    state, sink = new_state()
    state.op_paint_clear((), 0)
    assert sink.paths == []
    for name, operands in iter_content_operations(PdfLexer(b"1 2 m 3 4 l W")):
        state.execute_operation(name, operands, 0)
    state.execute_operation("S", (), 0)
    assert [event[0] for event in sink.paths] == ["stroke", "clip"]
    with pytest.raises(PdfParseError, match="current point"):
        state.execute_operation("l", (5, 6), 0)


def test_pending_clip_is_path_state_and_not_a_graphics_save() -> None:
    # ISO 32000-1 8.5.2.1 excludes the current path from q/Q graphics state.
    state, sink = new_state()
    state.op_q((), 0)
    for name, operands in iter_content_operations(PdfLexer(b"0 0 2 2 re W*")):
        state.execute_operation(name, operands, 0)
    state.op_Q((), 0)
    state.op_paint_clear((), 0)
    assert [(kind, rule) for kind, rule, _ in sink.paths] == [("clip", "evenodd")]


@pytest.mark.parametrize("flatness", [0.0, 0.5, 1.75, 100.0])
def test_flatness_defaults_fractions_and_graphics_restore(flatness: float) -> None:
    # ISO 32000-1 Tables 54 and 57: initial 1.0; i accepts a number in [0, 100].
    state, _ = new_state()
    assert state.graphics.flatness == 1.0
    state.execute_operation("q", (), 0)
    state.execute_operation("i", (flatness,), 0)
    for name, operands in iter_content_operations(PdfLexer(b"0 0 m 0 10 10 10 10 0 c")):
        state.execute_operation(name, operands, 0)
    assert state.current_path.commands[-1].flatness == flatness
    state.execute_operation("Q", (), 0)
    assert state.graphics.flatness == 1.0


def test_nested_streams_resume_once_and_emit_boundaries_before_exit() -> None:
    state, sink = new_state()
    observed = []
    state.operator_overrides["w"] = lambda operands, depth: observed.append((operands, depth))
    child = PdfStream(
        raw_data=b"2 w", dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]}
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"1 w /Child Do 3 w"),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )
    assert observed == [((1,), 0), ((2,), 1), ((3,), 0)]
    assert sink.streams == [
        ("enter", 0),
        ("enter", 1),
        ("end", 1),
        ("exit", 1),
        ("end", 0),
        ("exit", 0),
    ]
    assert sink.saved == 0
    assert not state.stream_executor.active_streams


@pytest.mark.parametrize("error_type", [PdfParseError, RuntimeError])
@pytest.mark.parametrize("stage", ["decode", "enter", "dispatch", "boundary"])
def test_nested_failure_unwinds_every_entered_frame(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: type[Exception],
) -> None:
    state, sink = new_state()
    original = state.capture_stream_state()

    class FailingStream(PdfStream):
        @property
        def data(self) -> bytes:
            raise error_type("child failed")

    child_type = FailingStream if stage == "decode" else PdfStream
    child = child_type(
        raw_data=b"2 w", dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]}
    )
    if stage == "enter":
        enter = sink.enter_stream

        def fail_enter(current: ContentInterpreter, frame: ContentStreamFrame) -> None:
            enter(current, frame)
            if frame.depth == 1:
                raise error_type("child failed")

        monkeypatch.setattr(sink, "enter_stream", fail_enter)
    elif stage == "dispatch":

        def fail_dispatch(operands: Any, depth: int) -> None:
            raise error_type("child failed")

        state.operator_overrides["w"] = fail_dispatch
    elif stage == "boundary":
        boundary = sink.text_boundary

        def fail_boundary(current: ContentInterpreter, reason: str) -> None:
            boundary(current, reason)
            if current.xobject_depth == 1 and reason == "stream-end":
                raise error_type("child failed")

        monkeypatch.setattr(sink, "text_boundary", fail_boundary)
    with pytest.raises(error_type, match="child failed"):
        state.stream_executor.consume(
            PdfStream(raw_data=b"/Child Do"),
            {"XObject": {"Child": child}},
            IDENTITY_MATRIX,
            0,
        )
    assert state.capture_stream_state() == original
    assert sink.saved == 0
    assert not state.stream_executor.active_streams
    assert [depth for event, depth in sink.streams if event == "exit"] == (
        [0] if stage == "decode" else [1, 0]
    )


def test_graphics_save_restores_ctm_and_dash_without_restoring_text_matrices() -> None:
    state, _ = new_state()
    dash = [2, 3]
    state.execute_operation("d", (dash, 1), 0)
    state.execute_operation("q", (), 0)
    dash[0] = 99
    state.execute_operation("cm", (2, 0, 0, 3, 4, 5), 0)
    state.execute_operation("d", ([7, 8], 0), 0)
    state.execute_operation("Tm", (3, 0, 0, 2, 5, 4), 0)
    text_matrix = state.text_matrix
    state.execute_operation("Q", (), 0)
    assert state.graphics.ctm == IDENTITY_MATRIX
    assert state.graphics.dash_pattern == ((2, 3), 1)
    assert state.text_matrix == state.line_matrix == text_matrix


def test_stream_snapshot_is_reusable_after_restoration_and_graphics_changes() -> None:
    state, _ = new_state()
    state.execute_operation("w", (3,), 0)
    snapshot = state.capture_stream_state()
    state.execute_operation("w", (8,), 0)
    state.restore_stream_state(snapshot)
    state.execute_operation("w", (12,), 0)
    assert snapshot.graphics_state.line_width == 3
    state.restore_stream_state(snapshot)
    assert state.graphics.line_width == 3
