"""The executor refuses a stream that reenters itself through reject_reentry."""

from __future__ import annotations

from typing import ClassVar

import pytest
from _content_support import NullSink, make_interpreter

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.streams import ContentStreamExecutor, ContentStreamFrame
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName


def form(content: bytes) -> PdfStream:
    return PdfStream(
        raw_data=content,
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 10, 10]},
    )


def run(state: ContentInterpreter, content: bytes, resources: PdfDict) -> None:
    state.stream_executor.consume(PdfStream(raw_data=content), resources, IDENTITY_MATRIX, 0)


class SkippingExecutor(ContentStreamExecutor):
    max_depth: ClassVar[int | None] = 3
    refused: ClassVar[list[int]] = []

    def reject_reentry(self, frame: ContentStreamFrame) -> bool:
        self.refused.append(frame.depth)
        return True


class SkippingInterpreter(ContentInterpreter):
    stream_executor_type = SkippingExecutor


def chain(length: int) -> PdfDict:
    """Forms F0 .. F{length-1}, each drawing the next."""
    forms: PdfDict = {}
    resources: PdfDict = {"XObject": forms}
    for index in range(length):
        forms[f"F{index}"] = form(f"/F{index + 1} Do".encode() if index + 1 < length else b"")
    return resources


def test_the_default_executor_refuses_a_recursive_form() -> None:
    state = make_interpreter(NullSink())
    looping = form(b"/Self Do")
    with pytest.raises(PdfParseError, match="recursive content stream"):
        run(state, b"/Self Do", {"XObject": {"Self": looping}})
    assert not state.stream_executor.active_streams


def test_the_default_executor_has_no_depth_limit() -> None:
    state = make_interpreter(NullSink())
    run(state, b"/F0 Do", chain(30))
    assert not state.stream_executor.active_streams


def test_a_skipping_executor_drops_a_recursive_form_and_carries_on() -> None:
    state = make_interpreter(NullSink(), interpreter_class=SkippingInterpreter)
    SkippingExecutor.refused.clear()
    looping = form(b"/Self Do 2 w")
    run(state, b"/Self Do", {"XObject": {"Self": looping}})
    assert SkippingExecutor.refused == [2]
    assert state.graphics.line_width == 1.0
    assert not state.stream_executor.active_streams


def test_a_skipping_executor_drops_forms_past_its_depth_limit() -> None:
    state = make_interpreter(NullSink(), interpreter_class=SkippingInterpreter)
    SkippingExecutor.refused.clear()
    run(state, b"/F0 Do", chain(10))
    assert SkippingExecutor.refused == [4]
    assert not state.stream_executor.active_streams


def test_a_frame_entered_too_deep_is_refused_on_entry() -> None:
    state = make_interpreter(NullSink(), interpreter_class=SkippingInterpreter)
    SkippingExecutor.refused.clear()
    frame = ContentStreamFrame(form(b""), {}, IDENTITY_MATRIX, 4, None)
    assert state.stream_executor.enter(frame) is False
    assert frame.old_state is None
    assert SkippingExecutor.refused == [4]
    strict = make_interpreter(NullSink())
    assert strict.stream_executor.enter(ContentStreamFrame(form(b""), {}, IDENTITY_MATRIX, 4, None))
