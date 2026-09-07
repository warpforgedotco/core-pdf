# SPDX-License-Identifier: AGPL-3.0-only
"""Nested content-stream execution owned separately from interpreter state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.operations import dispatch_operations
from core_pdf.impl.spec.s_07_content.stream_state import (
    ContentStreamFrame,
    StreamKey,
)
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.types import Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.state import TextState


class NestedStreamRequest(Exception):
    """Suspend the current stream until this child frame has been consumed."""

    def __init__(self, frame: ContentStreamFrame) -> None:
        super().__init__()
        self.frame = frame


class ContentStreamExecutor:
    """Drive nested streams while the interpreter owns PDF graphics/text state."""

    __slots__ = ("state", "active_streams")

    def __init__(self, state: TextState) -> None:
        self.state = state
        # Shared across reentrant Type 3 execution; each consume call owns only
        # its own frames, so unwinding a glyph cannot discard its caller.
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
        swallow_parse_errors: bool = False,
    ) -> None:
        limit = self.state.recovery.max_stream_depth if self.state.recovery is not None else None
        if limit is not None and depth > limit:
            return
        execution_key = stream_key or self.execution_key(stream)
        if execution_key in self.active_streams:
            return
        raise NestedStreamRequest(
            ContentStreamFrame(
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
                swallow_parse_errors=swallow_parse_errors,
            )
        )

    def enter(self, frame: ContentStreamFrame) -> bool:
        state = self.state
        limit = state.recovery.max_stream_depth if state.recovery is not None else None
        if limit is not None and frame.depth > limit:
            return False
        stream_key = frame.stream_key or self.execution_key(frame.stream)
        if stream_key in self.active_streams:
            return False
        # Decode before changing interpreter state or emitting group markers.
        # A failed stream entry must leave its parent exactly as it was.
        frame.lexer = state.lexer_factory(frame.stream.data)
        frame.old_state = state.capture_stream_state()
        # The implicit Form save also owns clips made without an explicit q.
        # Its floor prevents malformed child Q operators from consuming any
        # caller saves, while exit can discard unfinished child scopes safely.
        state.op_q((), frame.depth)
        state.graphics_stack_floor = len(state.stack)
        self.active_streams.add(stream_key)
        frame.stream_key = stream_key
        state.sink.enter_stream(state, frame)
        if frame.group_alpha is not None:
            # The parent's alpha/blend composite the completed group once.
            # Children start with default transparency until their own gs.
            state.fill_opacity = 1.0
            state.stroke_opacity = 1.0
            state.blend_mode = None
        state.resources = frame.resources
        state.resources_id = id(frame.resources)
        state.ctm = frame.ctm
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

    def consume(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: Rectangle | None = None,
    ) -> None:
        state = self.state
        stream_stack = [ContentStreamFrame(stream, resources, ctm, depth, clip_bbox)]
        try:
            while stream_stack:
                frame = stream_stack[-1]
                try:
                    if frame.old_state is None and not self.enter(frame):
                        stream_stack.pop()
                        continue
                    assert frame.lexer is not None
                    dispatch_operations(
                        frame.lexer, state.op_handlers.get, frame.depth, recovery=state.recovery
                    )
                    state.sink.text_boundary(state, "stream-end")
                except NestedStreamRequest as request:
                    stream_stack.append(request.frame)
                    continue
                except PdfParseError:
                    if not frame.swallow_parse_errors:
                        raise
                self.exit(stream_stack.pop())
        finally:
            # A child failure must unwind suspended parents as well. Keeping
            # the current frame on the stack also covers failures during entry.
            while stream_stack:
                self.exit(stream_stack.pop())


__all__ = ("ContentStreamExecutor",)
