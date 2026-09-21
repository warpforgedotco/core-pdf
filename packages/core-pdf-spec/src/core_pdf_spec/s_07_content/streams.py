# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import GraphicsState
from core_pdf_spec.s_07_content.operations import iter_content_operations
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.types import Rectangle

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
    from core_pdf_spec.s_07_syntax.lexer import PdfLexer

StreamKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class StreamState:
    graphics_state: GraphicsState
    resources: PdfDict
    text_matrix: Matrix
    line_matrix: Matrix
    graphics_stack_floor: int
    graphics_stack_len: int
    marked_content_stack_len: int
    xobject_depth: int
    compatibility_depth: int = field(default=0, kw_only=True)
    pending_clip_rule: str | None = field(default=None, kw_only=True)
    initial_alpha_is_shape: bool = field(default=False, kw_only=True)
    initial_text_knockout: bool = field(default=True, kw_only=True)
    in_text_object: bool = field(default=False, kw_only=True)


@dataclass(slots=True)
class ContentStreamFrame:
    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: Rectangle | None
    group_alpha: float | None = None
    group_isolated: bool = field(default=True, kw_only=True)
    group_knockout: bool = field(default=False, kw_only=True)
    form_bbox_operand: object = field(default=None, kw_only=True)
    form_bbox: Rectangle | None = field(default=None, kw_only=True)
    is_form: bool = field(default=False, kw_only=True)
    source_key: StreamKey | None = field(default=None, kw_only=True)
    stream_key: StreamKey | None = field(default=None, kw_only=True)
    lexer: PdfLexer | None = field(default=None, init=False)
    old_state: StreamState | None = field(default=None, init=False)


class ContentStreamExecutor:
    __slots__ = ("state", "active_streams")

    def __init__(self, state: ContentInterpreter) -> None:
        self.state = state
        self.active_streams: set[StreamKey] = set()

    @staticmethod
    def execution_key(stream: PdfStream) -> StreamKey:
        return ("stream", id(stream), len(stream.raw_data))

    def queue(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: Rectangle | None = None,
        form_bbox_operand: object = None,
        group_alpha: float | None = None,
        stream_key: StreamKey | None = None,
    ) -> ContentStreamFrame | None:
        execution_key = stream_key or self.execution_key(stream)
        if execution_key in self.active_streams:
            raise PdfParseError("recursive content stream")
        return ContentStreamFrame(
            stream,
            resources,
            ctm,
            depth,
            clip_bbox,
            group_alpha,
            form_bbox_operand=form_bbox_operand,
            is_form=True,
            source_key=stream_key,
            stream_key=execution_key,
        )

    def enter(self, frame: ContentStreamFrame) -> bool:
        state = self.state
        stream_key = frame.stream_key or self.execution_key(frame.stream)
        if stream_key in self.active_streams:
            raise PdfParseError("recursive content stream")
        frame.lexer = state.create_lexer(frame.stream.data)
        frame.old_state = state.capture_stream_state()
        state.initial_alpha_is_shape = state.graphics.alpha_is_shape
        state.initial_text_knockout = state.graphics.text_knockout
        state.op_q((), frame.depth)
        state.graphics_stack_floor = len(state.stack)
        state.compatibility_depth = 0
        state.internal_pending_clip_rule = None
        state.in_text_object = False
        self.active_streams.add(stream_key)
        frame.stream_key = stream_key
        state.sink.enter_stream(state, frame)
        if frame.group_alpha is not None:
            state.graphics.fill_opacity = 1.0
            state.graphics.stroke_opacity = 1.0
            state.graphics.blend_mode = None
            state.graphics.soft_mask = None
        state.resources = frame.resources
        state.graphics.ctm = frame.ctm
        state.xobject_depth = frame.depth
        return True

    def exit(self, frame: ContentStreamFrame) -> None:
        old_state = frame.old_state
        if old_state is None:
            return
        state = self.state
        try:
            state.restore_stream_state(old_state)
        finally:
            if frame.stream_key is not None:
                self.active_streams.discard(frame.stream_key)
            frame.old_state = None
        state.sink.exit_stream(state, frame)

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        for name, operands in iter_content_operations(frame.lexer):
            child = state.execute_operation(name, operands, frame.depth)
            if child is not None:
                return child
        if state.compatibility_depth:
            raise PdfParseError("unterminated compatibility section")
        if len(state.stack) != state.graphics_stack_floor:
            raise PdfParseError("content stream ends with unbalanced graphics saves")
        assert frame.old_state is not None
        if len(state.marked_content_stack) != frame.old_state.marked_content_stack_len:
            raise PdfParseError("content stream ends with unbalanced marked content")
        return None

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        raise error

    def consume(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: Rectangle | None = None,
    ) -> None:
        self.consume_frame(ContentStreamFrame(stream, resources, ctm, depth, clip_bbox))

    def consume_frame(self, frame: ContentStreamFrame) -> None:
        if frame.old_state is not None:
            raise PdfParseError("content stream frame is already entered")
        state = self.state
        stream_stack = [frame]
        try:
            while stream_stack:
                frame = stream_stack[-1]
                try:
                    if frame.old_state is None and not self.enter(frame):
                        stream_stack.pop()
                        continue
                    child = self.dispatch_frame(frame)
                    if child is not None:
                        stream_stack.append(child)
                        continue
                    state.sink.text_boundary(state, "stream-end")
                except PdfParseError as error:
                    self.handle_parse_error(frame, error)
                self.exit(stream_stack.pop())
        finally:
            while stream_stack:
                self.exit(stream_stack.pop())


__all__ = ("StreamKey", "StreamState", "ContentStreamFrame", "ContentStreamExecutor")
