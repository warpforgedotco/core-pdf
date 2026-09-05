# SPDX-License-Identifier: AGPL-3.0-only
"""Nested content-stream execution owned separately from interpreter state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.model.geometry import Rectangle, intersect_bbox
from core_pdf.impl.spec.s_07_content.capture import marker_drawing
from core_pdf.impl.spec.s_07_content.operations import dispatch_operations
from core_pdf.impl.spec.s_07_content.stream_state import (
    ContentStreamFrame,
    LayoutFormId,
    StreamKey,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix

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
        layout_form_bbox: Rectangle | None = None,
        layout_form_id: LayoutFormId = None,
        group_alpha: float | None = None,
        stream_key: StreamKey | None = None,
        swallow_parse_errors: bool = False,
    ) -> None:
        if depth > 10:
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
                layout_form_bbox=layout_form_bbox,
                layout_form_id=layout_form_id,
                stream_key=execution_key,
                swallow_parse_errors=swallow_parse_errors,
            )
        )

    def enter(self, frame: ContentStreamFrame) -> bool:
        state = self.state
        if frame.depth > 10:
            return False
        stream_key = frame.stream_key or self.execution_key(frame.stream)
        if stream_key in self.active_streams:
            return False
        # Decode before changing interpreter state or emitting group markers.
        # A failed stream entry must leave its parent exactly as it was.
        frame.lexer = PdfLexer(frame.stream.data)
        frame.old_state = state.capture_stream_state()
        self.active_streams.add(stream_key)
        frame.stream_key = stream_key
        if frame.group_alpha is not None:
            state.drawings.append(
                marker_drawing(
                    "group-begin",
                    state.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=state.blend_mode,
                )
            )
            state.group_alpha = None
        state.resources = frame.resources
        state.resources_id = id(frame.resources)
        state.ctm = frame.ctm
        state.xobject_depth = frame.depth
        state.layout_form_bbox = frame.layout_form_bbox
        state.layout_form_id = frame.layout_form_id
        if frame.clip_bbox is not None:
            state.clip_bbox = intersect_bbox(state.clip_bbox, frame.clip_bbox)
        state.pending_line_break = False
        state.stream_order += 1
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
        if frame.group_alpha is not None:
            state.drawings.append(
                marker_drawing(
                    "group-end",
                    state.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=state.blend_mode,
                )
            )

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
                    dispatch_operations(frame.lexer, state.op_handlers.get, frame.depth)
                    state.run_accumulator.flush()
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
