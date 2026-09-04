# SPDX-License-Identifier: AGPL-3.0-only
"""Nested content-stream execution owned separately from interpreter state."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.model.geometry import Rectangle
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.spec.s_07_content.operations import NestedStreamRequest, dispatch_operations
from core_pdf.impl.spec.s_07_content.stream_state import (
    ContentStreamFrame,
    LayoutFormId,
    StreamKey,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix


class ContentStreamExecutor:
    """Drive nested streams while the interpreter owns PDF graphics/text state."""

    __slots__ = ("state",)

    def __init__(self, state: Any) -> None:
        self.state = state

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
        state = self.state
        if depth > 10:
            return
        execution_key = stream_key or self.execution_key(stream)
        if execution_key in state.active_streams:
            return
        state.queued_stream = ContentStreamFrame(
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
        raise NestedStreamRequest

    def enter(self, frame: ContentStreamFrame, *, initialize_lexer: bool = True) -> bool:
        state = self.state
        if frame.depth > 10:
            return False
        stream_key = frame.stream_key or self.execution_key(frame.stream)
        if stream_key in state.active_streams:
            return False
        state.active_streams.add(stream_key)
        frame.stream_key = stream_key
        if frame.group_alpha is not None:
            frame.outer_group_alpha = state.group_alpha
            state.drawings.append(
                CapturedDrawing(
                    seqno=state.sequence,
                    fill=None,
                    fill_opacity=frame.group_alpha,
                    blend_mode=state.blend_mode,
                    kind="group-begin",
                )
            )
            state.group_alpha = None
        frame.old_state = state.capture_stream_state()
        state.resources = frame.resources
        state.resources_id = id(frame.resources)
        state.ctm = frame.ctm
        state.xobject_depth = frame.depth
        state.layout_form_bbox = frame.layout_form_bbox
        state.layout_form_id = frame.layout_form_id
        if frame.clip_bbox is not None:
            state.clip_bbox = (
                frame.clip_bbox
                if state.clip_bbox is None
                else (
                    max(state.clip_bbox[0], frame.clip_bbox[0]),
                    max(state.clip_bbox[1], frame.clip_bbox[1]),
                    min(state.clip_bbox[2], frame.clip_bbox[2]),
                    min(state.clip_bbox[3], frame.clip_bbox[3]),
                )
            )
        state.pending_line_break = False
        state.stream_order += 1
        if initialize_lexer:
            frame.lexer = PdfLexer(frame.stream.data)
        frame.entered = True
        return True

    def exit(self, frame: ContentStreamFrame) -> None:
        state = self.state
        if frame.old_state is not None:
            state.restore_stream_state(frame.old_state)
        if frame.stream_key is not None:
            state.active_streams.remove(frame.stream_key)
        if frame.group_alpha is not None:
            state.group_alpha = frame.outer_group_alpha
            state.drawings.append(
                CapturedDrawing(
                    seqno=state.sequence,
                    fill=None,
                    fill_opacity=frame.group_alpha,
                    blend_mode=state.blend_mode,
                    kind="group-end",
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
        while stream_stack:
            frame = stream_stack.pop()
            if not frame.entered and not self.enter(frame):
                continue
            if frame.lexer is None:
                self.exit(frame)
                continue
            try:
                dispatch_operations(frame.lexer, state.op_handlers.get, state, frame.depth)
                state.flush_run()
            except NestedStreamRequest:
                queued_stream = state.queued_stream
                state.queued_stream = None
                stream_stack.append(frame)
                if queued_stream is not None:
                    stream_stack.append(queued_stream)
            except PdfParseError:
                self.exit(frame)
                if not frame.swallow_parse_errors:
                    raise
            except Exception:
                self.exit(frame)
                raise
            else:
                self.exit(frame)


__all__ = ("ContentStreamExecutor",)
