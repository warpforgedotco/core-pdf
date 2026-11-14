# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.8.2: running content streams and the form XObjects they invoke."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from core_pdf.impl.spec.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.model import GraphicsState
from core_pdf.impl.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.spec.types import Rectangle
from core_records import FrozenFields, PickleFields, RecordType, ReplaceFields, ReprFields, field

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.interpreter import ContentInterpreter
    from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer


StreamKey = tuple[str, int, int]


class StreamState(FrozenFields, PickleFields, ReprFields, ReplaceFields, metaclass=RecordType):
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

    __match_args__ = (
        "graphics_state",
        "resources",
        "text_matrix",
        "line_matrix",
        "graphics_stack_floor",
        "graphics_stack_len",
        "marked_content_stack_len",
        "xobject_depth",
    )


class ContentStreamFrame(ReprFields, metaclass=RecordType, frozen=False):
    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: Rectangle | None
    group_alpha: float | None
    group_isolated: bool
    group_knockout: bool
    form_bbox_operand: object
    form_bbox: Rectangle | None
    is_form: bool
    source_key: StreamKey | None
    stream_key: StreamKey | None
    lexer: PdfLexer | None
    old_state: StreamState | None

    __match_args__ = ("stream", "resources", "ctm", "depth", "clip_bbox", "group_alpha")

    def __init__(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        clip_bbox: Rectangle | None,
        group_alpha: float | None = None,
        *,
        group_isolated: bool = True,
        group_knockout: bool = False,
        form_bbox_operand: object = None,
        form_bbox: Rectangle | None = None,
        is_form: bool = False,
        source_key: StreamKey | None = None,
        stream_key: StreamKey | None = None,
    ) -> None:
        self.stream = stream
        self.resources = resources
        self.ctm = ctm
        self.depth = depth
        self.clip_bbox = clip_bbox
        self.group_alpha = group_alpha
        self.group_isolated = group_isolated
        self.group_knockout = group_knockout
        self.form_bbox_operand = form_bbox_operand
        self.form_bbox = form_bbox
        self.is_form = is_form
        self.source_key = source_key
        self.stream_key = stream_key
        self.lexer = None
        self.old_state = None

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        stream = changes.pop("stream", self.stream)
        resources = changes.pop("resources", self.resources)
        ctm = changes.pop("ctm", self.ctm)
        depth = changes.pop("depth", self.depth)
        clip_bbox = changes.pop("clip_bbox", self.clip_bbox)
        group_alpha = changes.pop("group_alpha", self.group_alpha)
        group_isolated = changes.pop("group_isolated", self.group_isolated)
        group_knockout = changes.pop("group_knockout", self.group_knockout)
        form_bbox_operand = changes.pop("form_bbox_operand", self.form_bbox_operand)
        form_bbox = changes.pop("form_bbox", self.form_bbox)
        is_form = changes.pop("is_form", self.is_form)
        source_key = changes.pop("source_key", self.source_key)
        stream_key = changes.pop("stream_key", self.stream_key)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            stream,
            resources,
            ctm,
            depth,
            clip_bbox,
            group_alpha,
            group_isolated=group_isolated,
            group_knockout=group_knockout,
            form_bbox_operand=form_bbox_operand,
            form_bbox=form_bbox,
            is_form=is_form,
            source_key=source_key,
            stream_key=stream_key,
        )


class ContentStreamExecutor:
    __slots__ = ("state", "active_streams", "decoded_streams")
    max_depth: ClassVar[int | None] = None

    def __init__(self, state: ContentInterpreter) -> None:
        self.state = state
        self.active_streams: set[StreamKey] = set()
        self.decoded_streams: dict[int, tuple[PdfStream, bytes]] = {}

    def stream_data(self, frame: ContentStreamFrame) -> bytes:
        stream = frame.stream
        if frame.depth <= 0:
            return stream.data
        cached = self.decoded_streams.get(id(stream))
        if cached is not None and cached[0] is stream:
            return cached[1]
        data = stream.data
        self.decoded_streams[id(stream)] = (stream, data)
        return data

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
        frame = ContentStreamFrame(
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
        if self.is_reentry(execution_key, depth) and self.reject_reentry(frame):
            return None
        return frame

    def is_reentry(self, stream_key: StreamKey, depth: int) -> bool:
        limit = self.max_depth
        return (limit is not None and depth > limit) or stream_key in self.active_streams

    def reject_reentry(self, frame: ContentStreamFrame) -> bool:
        limit = self.max_depth
        if limit is not None and frame.depth > limit:
            raise PdfParseError("content streams nest too deeply")
        raise PdfParseError("recursive content stream")

    def enter(self, frame: ContentStreamFrame) -> bool:
        state = self.state
        stream_key = frame.stream_key or self.execution_key(frame.stream)
        if self.is_reentry(stream_key, frame.depth) and self.reject_reentry(frame):
            return False
        frame.lexer = state.create_lexer(self.stream_data(frame))
        frame.old_state = state.capture_stream_state()
        state.initial_alpha_is_shape = state.graphics.alpha_is_shape
        state.initial_text_knockout = state.graphics.text_knockout
        state.op_q((), frame.depth)
        state.graphics_stack_floor = len(state.stack)
        state.compatibility_depth = 0
        state.pending_clip_rule_value = None
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
