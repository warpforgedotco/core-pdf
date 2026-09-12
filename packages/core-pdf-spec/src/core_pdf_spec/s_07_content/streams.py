# SPDX-License-Identifier: AGPL-3.0-only
"""Nested content frames, snapshots, and their execution lifecycle."""

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
    """Graphics snapshot and the additional state isolated by a nested stream."""

    graphics_state: GraphicsState
    resources: PdfDict
    text_matrix: Matrix
    line_matrix: Matrix
    graphics_stack_floor: int
    graphics_stack_len: int
    marked_content_stack_len: int
    xobject_depth: int
    compatibility_depth: int = field(default=0, kw_only=True)
    # Path construction is outside q/Q, but pending child clips must not leak.
    pending_clip_rule: str | None = field(default=None, kw_only=True)


@dataclass(slots=True)
class ContentStreamFrame:
    """One pending nested content stream, plus the state captured on entry."""

    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: Rectangle | None
    group_alpha: float | None = None
    # Legacy explicitly queued groups are isolated; Form dictionaries supply /I.
    group_isolated: bool = field(default=True, kw_only=True)
    form_bbox_operand: object = field(default=None, kw_only=True)
    # Resolved local bounds; clip_bbox is only their transformed enclosing box.
    form_bbox: Rectangle | None = field(default=None, kw_only=True)
    is_form: bool = field(default=False, kw_only=True)
    source_key: StreamKey | None = field(default=None, kw_only=True)
    stream_key: StreamKey | None = field(default=None, kw_only=True)
    lexer: PdfLexer | None = field(default=None, init=False)
    # Present only while this frame is entered, including suspension for a child.
    old_state: StreamState | None = field(default=None, init=False)


class ContentStreamExecutor:
    """Drive nested streams while the interpreter owns PDF graphics/text state."""

    __slots__ = ("state", "active_streams")

    def __init__(self, state: ContentInterpreter) -> None:
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
        # Decode before changing interpreter state or emitting group markers.
        # A failed stream entry must leave its parent exactly as it was.
        frame.lexer = state.create_lexer(frame.stream.data)
        frame.old_state = state.capture_stream_state()
        # The implicit Form save also owns clips made without an explicit q.
        # Its floor prevents malformed child Q operators from consuming any
        # caller saves, while exit can discard unfinished child scopes safely.
        state.op_q((), frame.depth)
        state.graphics_stack_floor = len(state.stack)
        # ISO 32000-1 7.8.2: BX/EX sections are not graphics state. A child
        # starts a fresh scope; its snapshot retains the parent's depth.
        state.compatibility_depth = 0
        state.internal_pending_clip_rule = None
        self.active_streams.add(stream_key)
        frame.stream_key = stream_key
        state.sink.enter_stream(state, frame)
        if frame.group_alpha is not None:
            # The parent's alpha/blend composite the completed group once.
            # Children start with default transparency until their own gs.
            state.graphics.fill_opacity = 1.0
            state.graphics.stroke_opacity = 1.0
            state.graphics.blend_mode = None
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
        """Execute until a child is requested or the stream's scopes are complete.

        Readers may override this method to supply their parsing and EOF policy.
        The driver owns suspension, stream-boundary events, and frame cleanup.
        """
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
        """Propagate entry, dispatch, or boundary errors before frame cleanup.

        A reader override may return to discard this frame and resume its parent.
        Errors raised while exiting a frame are never handled by this hook.
        """
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
        state = self.state
        stream_stack = [ContentStreamFrame(stream, resources, ctm, depth, clip_bbox)]
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
            # A child failure must unwind suspended parents as well. Keeping
            # the current frame on the stack also covers failures during entry.
            while stream_stack:
                self.exit(stream_stack.pop())


__all__ = ("StreamKey", "StreamState", "ContentStreamFrame", "ContentStreamExecutor")
