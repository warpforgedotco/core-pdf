# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import GraphicsState
from core_pdf_spec.s_07_content.operations import iter_content_operations
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.types import Rectangle
from core_records import FrozenFields, PickleFields, ReprFields, frozen_setattr

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
    from core_pdf_spec.s_07_syntax.lexer import PdfLexer


StreamKey = tuple[str, int, int]


class StreamState(FrozenFields, PickleFields, ReprFields):
    __slots__ = (
        "graphics_state",
        "resources",
        "text_matrix",
        "line_matrix",
        "graphics_stack_floor",
        "graphics_stack_len",
        "marked_content_stack_len",
        "xobject_depth",
        "compatibility_depth",
        "pending_clip_rule",
        "initial_alpha_is_shape",
        "initial_text_knockout",
        "in_text_object",
    )

    graphics_state: GraphicsState
    resources: PdfDict
    text_matrix: Matrix
    line_matrix: Matrix
    graphics_stack_floor: int
    graphics_stack_len: int
    marked_content_stack_len: int
    xobject_depth: int
    compatibility_depth: int
    pending_clip_rule: str | None
    initial_alpha_is_shape: bool
    initial_text_knockout: bool
    in_text_object: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "graphics_state",
        "resources",
        "text_matrix",
        "line_matrix",
        "graphics_stack_floor",
        "graphics_stack_len",
        "marked_content_stack_len",
        "xobject_depth",
        "compatibility_depth",
        "pending_clip_rule",
        "initial_alpha_is_shape",
        "initial_text_knockout",
        "in_text_object",
    )
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

    def __init__(
        self,
        graphics_state: GraphicsState,
        resources: PdfDict,
        text_matrix: Matrix,
        line_matrix: Matrix,
        graphics_stack_floor: int,
        graphics_stack_len: int,
        marked_content_stack_len: int,
        xobject_depth: int,
        *,
        compatibility_depth: int = 0,
        pending_clip_rule: str | None = None,
        initial_alpha_is_shape: bool = False,
        initial_text_knockout: bool = True,
        in_text_object: bool = False,
    ) -> None:
        frozen_setattr(self, "graphics_state", graphics_state)
        frozen_setattr(self, "resources", resources)
        frozen_setattr(self, "text_matrix", text_matrix)
        frozen_setattr(self, "line_matrix", line_matrix)
        frozen_setattr(self, "graphics_stack_floor", graphics_stack_floor)
        frozen_setattr(self, "graphics_stack_len", graphics_stack_len)
        frozen_setattr(self, "marked_content_stack_len", marked_content_stack_len)
        frozen_setattr(self, "xobject_depth", xobject_depth)
        frozen_setattr(self, "compatibility_depth", compatibility_depth)
        frozen_setattr(self, "pending_clip_rule", pending_clip_rule)
        frozen_setattr(self, "initial_alpha_is_shape", initial_alpha_is_shape)
        frozen_setattr(self, "initial_text_knockout", initial_text_knockout)
        frozen_setattr(self, "in_text_object", in_text_object)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.graphics_state == other.graphics_state
            and self.resources == other.resources
            and self.text_matrix == other.text_matrix
            and self.line_matrix == other.line_matrix
            and self.graphics_stack_floor == other.graphics_stack_floor
            and self.graphics_stack_len == other.graphics_stack_len
            and self.marked_content_stack_len == other.marked_content_stack_len
            and self.xobject_depth == other.xobject_depth
            and self.compatibility_depth == other.compatibility_depth
            and self.pending_clip_rule == other.pending_clip_rule
            and self.initial_alpha_is_shape == other.initial_alpha_is_shape
            and self.initial_text_knockout == other.initial_text_knockout
            and self.in_text_object == other.in_text_object
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.graphics_state,
                self.resources,
                self.text_matrix,
                self.line_matrix,
                self.graphics_stack_floor,
                self.graphics_stack_len,
                self.marked_content_stack_len,
                self.xobject_depth,
                self.compatibility_depth,
                self.pending_clip_rule,
                self.initial_alpha_is_shape,
                self.initial_text_knockout,
                self.in_text_object,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        graphics_state = changes.pop("graphics_state", self.graphics_state)
        resources = changes.pop("resources", self.resources)
        text_matrix = changes.pop("text_matrix", self.text_matrix)
        line_matrix = changes.pop("line_matrix", self.line_matrix)
        graphics_stack_floor = changes.pop("graphics_stack_floor", self.graphics_stack_floor)
        graphics_stack_len = changes.pop("graphics_stack_len", self.graphics_stack_len)
        marked_content_stack_len = changes.pop(
            "marked_content_stack_len", self.marked_content_stack_len
        )
        xobject_depth = changes.pop("xobject_depth", self.xobject_depth)
        compatibility_depth = changes.pop("compatibility_depth", self.compatibility_depth)
        pending_clip_rule = changes.pop("pending_clip_rule", self.pending_clip_rule)
        initial_alpha_is_shape = changes.pop("initial_alpha_is_shape", self.initial_alpha_is_shape)
        initial_text_knockout = changes.pop("initial_text_knockout", self.initial_text_knockout)
        in_text_object = changes.pop("in_text_object", self.in_text_object)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            graphics_state,
            resources,
            text_matrix,
            line_matrix,
            graphics_stack_floor,
            graphics_stack_len,
            marked_content_stack_len,
            xobject_depth,
            compatibility_depth=compatibility_depth,
            pending_clip_rule=pending_clip_rule,
            initial_alpha_is_shape=initial_alpha_is_shape,
            initial_text_knockout=initial_text_knockout,
            in_text_object=in_text_object,
        )


class ContentStreamFrame(ReprFields):
    __slots__ = (
        "stream",
        "resources",
        "ctm",
        "depth",
        "clip_bbox",
        "group_alpha",
        "group_isolated",
        "group_knockout",
        "form_bbox_operand",
        "form_bbox",
        "is_form",
        "source_key",
        "stream_key",
        "lexer",
        "old_state",
    )

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

    __fields__: ClassVar[tuple[str, ...]] = (
        "stream",
        "resources",
        "ctm",
        "depth",
        "clip_bbox",
        "group_alpha",
        "group_isolated",
        "group_knockout",
        "form_bbox_operand",
        "form_bbox",
        "is_form",
        "source_key",
        "stream_key",
        "lexer",
        "old_state",
    )
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

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.stream == other.stream
            and self.resources == other.resources
            and self.ctm == other.ctm
            and self.depth == other.depth
            and self.clip_bbox == other.clip_bbox
            and self.group_alpha == other.group_alpha
            and self.group_isolated == other.group_isolated
            and self.group_knockout == other.group_knockout
            and self.form_bbox_operand == other.form_bbox_operand
            and self.form_bbox == other.form_bbox
            and self.is_form == other.is_form
            and self.source_key == other.source_key
            and self.stream_key == other.stream_key
            and self.lexer == other.lexer
            and self.old_state == other.old_state
        )

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

    def __init__(self, state: ContentInterpreter) -> None:
        self.state = state
        self.active_streams: set[StreamKey] = set()
        # Nested streams are entered again and again: a Type 3 glyph's CharProc
        # once per glyph shown, a form once per Do. Decoding is a pure function
        # of the stream, so each is decoded once. The stream is kept beside its
        # bytes so its id cannot be reused while the entry exists.
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
