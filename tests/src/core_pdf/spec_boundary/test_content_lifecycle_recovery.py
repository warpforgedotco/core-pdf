"""Core capture preserves path ordering and nested-stream recovery boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NoReturn

import pytest

from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.recovery import iter_content_operations
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_07_content.operations import ContentOperands
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, Rectangle


def internal_state() -> TextState:
    resolver = ObjectResolver(b"", {})
    return TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))


def internal_execute(state: TextState, content: bytes) -> None:
    for name, operands in iter_content_operations(PdfLexer(content)):
        assert state.execute_operation(name, operands, 0) is None


def internal_assert_clean(state: TextState) -> None:
    assert not state.stack
    assert not state.capture_graphics_stack
    assert not state.marked_content_stack
    assert not state.capture_marked_entries
    assert not state.capture_frames
    assert not state.stream_executor.active_streams
    assert state.graphics_stack_floor == 0
    assert state.layout_form_id is None
    assert state.group_alpha is None
    assert state.internal_pending_clip_rule is None
    assert state.graphics.ctm == IDENTITY_MATRIX
    scope_depth = 0
    for drawing in state.drawings:
        if drawing.kind == "scope-begin":
            scope_depth += 1
        elif drawing.kind == "scope-end":
            scope_depth -= 1
            assert scope_depth >= 0
    assert scope_depth == 0


@pytest.mark.parametrize(("clip_operator", "clip_rule"), [(b"W", "nonzero"), (b"W*", "evenodd")])
@pytest.mark.parametrize(
    ("terminator", "paint_kind"), [(b"S", "stroke"), (b"f", "fill"), (b"n", None)]
)
def test_reader_defers_clip_capture_until_after_path_paint(
    monkeypatch: pytest.MonkeyPatch,
    clip_operator: bytes,
    clip_rule: str,
    terminator: bytes,
    paint_kind: str | None,
) -> None:
    state = internal_state()
    events: list[tuple[str, str]] = []
    paths: list[PdfPath] = []
    paint = state.paint_path
    clip = state.clip_path

    def record_paint(current: object, path: PdfPath, kind: str, rule: str) -> None:
        events.append((kind, rule))
        paths.append(path)
        assert state.clip_bbox is None
        paint(current, path, kind, rule)

    def record_clip(current: object, path: PdfPath, rule: str) -> None:
        events.append(("clip", rule))
        paths.append(path)
        clip(current, path, rule)

    monkeypatch.setattr(state, "paint_path", record_paint)
    monkeypatch.setattr(state, "clip_path", record_clip)
    internal_execute(state, b"0 0 10 10 re " + clip_operator)
    assert not events
    assert not state.drawings
    assert state.clip_bbox is None
    internal_execute(state, terminator)
    expected = ([] if paint_kind is None else [(paint_kind, "nonzero")]) + [("clip", clip_rule)]
    assert events == expected
    assert [drawing.kind for drawing in state.drawings] == [kind for kind, _ in expected]
    assert [drawing.seqno for drawing in state.drawings] == list(range(len(expected)))
    assert state.clip_bbox == (0.0, 0.0, 10.0, 10.0)
    assert state.current_point is state.subpath_start is None
    assert not state.current_path.commands
    assert state.internal_pending_clip_rule is None
    assert all(path is paths[0] for path in paths)
    internal_execute(state, b"20 20 m")
    assert paths[0].commands[0].operator == "re"
    assert paths[0].commands[0].operands == (0.0, 0.0, 10.0, 10.0)
    for drawing in state.drawings:
        assert drawing.path is not None
        assert drawing.path.bbox() == (0.0, 0.0, 10.0, 10.0)


@pytest.mark.parametrize("scope", ["q", "form"])
def test_reader_restores_deferred_path_clip_before_painting_outside_scope(
    monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    state = internal_state()
    original_clip = (-100.0, -100.0, 100.0, 100.0)
    state.clip_bbox = original_clip
    paint_clips: list[Rectangle | None] = []
    paint = state.paint_path

    def record_paint(current: object, path: PdfPath, kind: str, rule: str) -> None:
        paint_clips.append(state.clip_bbox)
        paint(current, path, kind, rule)

    monkeypatch.setattr(state, "paint_path", record_paint)
    scoped = b"0 0 10 10 re W n 1 1 2 2 re f"
    resources: PdfDict
    if scope == "q":
        content = b"q " + scoped + b" Q 30 30 5 5 re f"
        resources = {}
    else:
        child = PdfStream(
            raw_data=scoped,
            dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 50, 50]},
        )
        content = b"/Child Do 30 30 5 5 re f"
        resources = {"XObject": {"Child": child}}
    state.stream_executor.consume(PdfStream(raw_data=content), resources, IDENTITY_MATRIX, 0)
    assert paint_clips == [(0.0, 0.0, 10.0, 10.0), original_clip]
    scoped_kinds = [
        "state-push",
        "clip",
        "fill",
        "state-pop",
    ]
    if scope == "form":
        scoped_kinds = ["scope-begin", *scoped_kinds, "scope-end"]
    assert [drawing.kind for drawing in state.drawings] == [*scoped_kinds, "fill"]
    assert state.clip_bbox == original_clip
    internal_assert_clean(state)


@pytest.mark.parametrize("flatness", ["0.75", "1.75", 0.75])
def test_reader_preserves_default_and_fractional_flatness_with_numeric_recovery(
    flatness: str | float,
) -> None:
    state = internal_state()
    assert state.graphics.flatness == 1.0
    state.op_q((), 0)
    state.op_i((flatness,), 0)
    assert state.graphics.flatness == float(flatness)
    internal_execute(state, b"0 0 m 1 2 3 4 5 6 c")
    curve = state.current_path.commands[-1]
    assert curve.flatness == float(flatness)
    state.op_Q((), 0)
    assert state.graphics.flatness == 1.0
    assert curve.flatness == float(flatness)
    internal_assert_clean(state)


@pytest.mark.parametrize("stage", ["decode", "enter", "handler", "stream-end"])
@pytest.mark.parametrize("is_form", [False, True], ids=["root", "form"])
@pytest.mark.parametrize("error_type", [PdfParseError, RuntimeError])
def test_reader_recovers_only_form_parse_errors_and_unwinds_all_frames(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    is_form: bool,
    error_type: type[Exception],
) -> None:
    state = internal_state()
    error = error_type(f"failure during {stage}")
    triggered: list[str] = []
    content = b"q /Span BMC 0.75 g" + (b" FAIL" if stage == "handler" else b"")
    failing = PdfStream(
        raw_data=content,
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )

    def fail() -> NoReturn:
        triggered.append(stage)
        raise error

    if stage == "decode":

        def decode(*args: object, **kwargs: object) -> bytes:
            fail()

        failing.decoder = decode
    elif stage == "enter":
        enter = state.enter_stream

        def enter_stream(current: object, frame: ContentStreamFrame) -> None:
            enter(current, frame)
            if frame.stream is failing:
                fail()

        monkeypatch.setattr(state, "enter_stream", enter_stream)
    elif stage == "handler":

        def handler(operands: ContentOperands, depth: int) -> None:
            fail()

        state.operator_overrides["FAIL"] = handler
    else:
        boundary = state.text_boundary

        def text_boundary(current: object, kind: str) -> None:
            boundary(current, kind)
            if kind == "stream-end" and state.xobject_depth == int(is_form):
                fail()

        monkeypatch.setattr(state, "text_boundary", text_boundary)

    source = PdfStream(raw_data=b"0.25 g /Child Do 30 40 5 6 re f") if is_form else failing
    scope_kinds = ["scope-begin", "scope-end"] if is_form and stage != "decode" else []
    if is_form and error_type is PdfParseError:
        state.stream_executor.consume(source, {"XObject": {"Child": failing}}, IDENTITY_MATRIX, 0)
        assert [drawing.kind for drawing in state.drawings] == [*scope_kinds, "fill"]
        drawing = state.drawings[-1]
        assert drawing.kind == "fill"
        assert drawing.fill == (0.25,)
        assert drawing.path is not None
        assert drawing.path.bbox() == (30.0, 40.0, 35.0, 46.0)
    else:
        with pytest.raises(error_type) as raised:
            state.stream_executor.consume(
                source, {"XObject": {"Child": failing}}, IDENTITY_MATRIX, 0
            )
        assert raised.value is error
        assert [drawing.kind for drawing in state.drawings] == scope_kinds
    assert triggered == [stage]
    assert state.clip_bbox is None
    internal_assert_clean(state)


@pytest.mark.parametrize("failure", [False, True], ids=["eof", "handler-failure"])
def test_reader_discards_unfinished_child_clip_before_parent_path(failure: bool) -> None:
    state = internal_state()

    def fail(operands: ContentOperands, depth: int) -> None:
        raise PdfParseError("unfinished child path")

    state.operator_overrides["FAIL"] = fail
    child = PdfStream(
        raw_data=b"0 0 10 10 re W" + (b" FAIL" if failure else b""),
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 50, 50]},
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/Child Do 20 20 5 5 re f"),
        {"XObject": {"Child": child}},
        IDENTITY_MATRIX,
        0,
    )
    assert [drawing.kind for drawing in state.drawings] == ["scope-begin", "scope-end", "fill"]
    assert state.clip_bbox is None
    internal_assert_clean(state)
