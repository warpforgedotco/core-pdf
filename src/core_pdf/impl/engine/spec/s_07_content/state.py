# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import typing
from math import ceil, hypot
from typing import TypeAlias

from core_layout.impl.layout.glyphs import GlyphCluster, GlyphObservation
from core_layout.impl.layout.models import TextRun

from core_pdf.impl.engine.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedLine,
    CapturedPath,
    ContentCaptureMixin,
)
from core_pdf.impl.engine.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.engine.spec.s_07_content.operations import (
    ContentOperand,
    ContentOperands,
    NestedStreamRequest,
    OperandWindow,
    OperationTarget,
    StateOperationHandler,
    content_stream_may_show_text,
    dispatch_operations,
)
from core_pdf.impl.engine.spec.s_07_content.operators import (
    OperatorMixin,
    detect_rotation_from_linear,
)
from core_pdf.impl.engine.spec.s_07_content.text_helpers import (
    cached_encode_latin1,
    detect_ligature_overrides,
    is_garbage_text,
)
from core_pdf.impl.engine.spec.s_07_content.xobjects import XObjectMixin
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.engine.spec.s_09_fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import (
    MISSING,
    PdfReference,
    PdfStream,
    PdfString,
)
from core_pdf.impl.types import PdfDict

OperationHandler: TypeAlias = StateOperationHandler
ResourceCategoryCache: TypeAlias = dict[str, object]
ResourceCache: TypeAlias = dict[tuple[int, str], ResourceCategoryCache]
ResolvedResourceCache: TypeAlias = dict[tuple[int, str], object]
ObjectCache: TypeAlias = dict[object, object]
InlineImageRecord: TypeAlias = dict[str, object]
DecodedGlyphs: TypeAlias = tuple[DecodedGlyph, ...] | None


class TextResolver(typing.Protocol):
    kw_cache: dict[bytes, object]

    def resolve(self, ref: object) -> object: ...

    def resolve_dict(self, value: object) -> PdfDict | None: ...

    def resolve_font_dict(self, value: PdfDict) -> PdfDict: ...

    def resolve_name(self, value: object) -> str | None: ...

    def resolve_name_like_value(self, value: object) -> str | None: ...

    def resolve_str(self, value: object) -> str | None: ...


class TextDocument(typing.Protocol):
    @property
    def resolver(self) -> TextResolver: ...

    decoder_cache: dict[tuple[int, int] | int, FontDecoder]

    def resolve(self, value: object) -> object: ...


class StreamState:
    __slots__ = (
        "resources",
        "resources_id",
        "ctm",
        "text_matrix",
        "line_matrix",
        "font_size",
        "font_operand",
        "font_size_operand",
        "current_font",
        "current_decoder",
        "current_decoder_resources_id",
        "graphics_stack_len",
        "marked_content_stack_len",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "compatibility_depth",
        "blend_mode",
        "group_alpha",
        "fill_color_space",
        "stroke_color_space",
        "flatness",
        "render_intent",
        "clip_bbox",
        "pending_line_break",
        "xobject_depth",
        "inline_images",
        "resource_cache",
        "resolved_resource_categories",
    )

    resources: PdfDict
    resources_id: int
    ctm: Matrix
    text_matrix: Matrix
    line_matrix: Matrix
    font_size: float
    font_operand: object
    font_size_operand: object
    current_font: str | None
    current_decoder: FontDecoder | None
    current_decoder_resources_id: int | None
    graphics_stack_len: int
    marked_content_stack_len: int
    fill_color: tuple[float, ...] | None
    fill_pattern: PdfDict | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PdfDict | None
    compatibility_depth: int
    blend_mode: str | None
    group_alpha: float | None
    fill_color_space: str
    stroke_color_space: str
    flatness: int
    render_intent: str | None
    pending_line_break: bool
    xobject_depth: int

    def __init__(
        self,
        resources: PdfDict,
        resources_id: int,
        ctm: Matrix,
        text_matrix: Matrix,
        line_matrix: Matrix,
        font_size: float,
        font_operand: object,
        font_size_operand: object,
        current_font: str | None,
        current_decoder: FontDecoder | None,
        current_decoder_resources_id: int | None,
        graphics_stack_len: int,
        marked_content_stack_len: int,
        fill_color: tuple[float, ...] | None,
        fill_pattern: PdfDict | None,
        fill_opacity: float | None,
        stroke_color: tuple[float, ...] | None,
        stroke_pattern: PdfDict | None,
        fill_color_space: str,
        stroke_color_space: str,
        compatibility_depth: int,
        blend_mode: str | None,
        group_alpha: float | None,
        flatness: int,
        render_intent: str | None,
        clip_bbox: tuple[float, float, float, float] | None,
        pending_line_break: bool,
        xobject_depth: int,
        resource_cache: ResourceCache,
        resolved_resource_categories: ResolvedResourceCache,
    ) -> None:
        self.resources = resources
        self.resources_id = resources_id
        self.ctm = ctm
        self.text_matrix = text_matrix
        self.line_matrix = line_matrix
        self.font_size = font_size
        self.font_operand = font_operand
        self.font_size_operand = font_size_operand
        self.current_font = current_font
        self.current_decoder = current_decoder
        self.current_decoder_resources_id = current_decoder_resources_id
        self.graphics_stack_len = graphics_stack_len
        self.marked_content_stack_len = marked_content_stack_len
        self.fill_color = fill_color
        self.fill_pattern = fill_pattern
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_pattern = stroke_pattern
        self.compatibility_depth = compatibility_depth
        self.fill_color_space = fill_color_space
        self.stroke_color_space = stroke_color_space
        self.blend_mode = blend_mode
        self.group_alpha = group_alpha
        self.flatness = flatness
        self.render_intent = render_intent
        self.clip_bbox = clip_bbox
        self.pending_line_break = pending_line_break
        self.xobject_depth = xobject_depth
        self.resource_cache = resource_cache
        self.resolved_resource_categories = resolved_resource_categories


StreamKey: TypeAlias = tuple[str, int, int]


class ContentStreamFrame:
    __slots__ = (
        "stream",
        "resources",
        "ctm",
        "depth",
        "clip_bbox",
        "group_alpha",
        "swallow_parse_errors",
        "lexer",
        "old_state",
        "stream_key",
        "outer_group_alpha",
        "entered",
    )

    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: tuple[float, float, float, float] | None
    group_alpha: float | None
    swallow_parse_errors: bool
    lexer: PdfLexer | None
    old_state: StreamState | None
    stream_key: StreamKey | None
    outer_group_alpha: float | None
    entered: bool

    def __init__(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        clip_bbox: tuple[float, float, float, float] | None,
        group_alpha: float | None = None,
        *,
        stream_key: StreamKey | None = None,
        swallow_parse_errors: bool = False,
    ) -> None:
        self.stream = stream
        self.resources = resources
        self.ctm = ctm
        self.depth = depth
        self.clip_bbox = clip_bbox
        self.group_alpha = group_alpha
        self.swallow_parse_errors = swallow_parse_errors
        self.lexer = None
        self.old_state = None
        self.stream_key = stream_key
        self.outer_group_alpha = None
        self.entered = False


class TextState(XObjectMixin, ContentCaptureMixin, OperatorMixin):
    document: TextDocument
    page: PdfDict
    capture_runs: bool
    capture_glyphs: bool
    capture_glyph_bitmaps: bool
    capture_graphics: bool
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_clusters: list[GlyphCluster]
    lines: list[CapturedLine]
    drawings: list[CapturedDrawing]
    current_path: CapturedPath
    current_point: tuple[float, float] | None
    subpath_start: tuple[float, float] | None
    stack: list[StreamState]
    fill_color: tuple[float, ...] | None
    fill_pattern: PdfDict | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PdfDict | None
    stroke_opacity: float | None
    blend_mode: str | None
    group_alpha: float | None
    flatness: int
    render_intent: str | None
    fill_color_space: str
    stroke_color_space: str
    dash_pattern: tuple[list[float], float]
    font_operand: object
    font_size_operand: object
    font_widths: tuple[float, ...] | None
    current_font: str | None
    current_decoder: FontDecoder | None
    current_decoder_resources_id: int | None
    marked_content_stack: list[MarkedContentEntry]
    active_streams: set[StreamKey]
    queued_stream: ContentStreamFrame | None
    resources: PdfDict
    resolved_resource_categories: ResolvedResourceCache
    resource_cache: ResourceCache
    color_space_cache: ObjectCache
    color_normalization_cache: ObjectCache
    extgstate_cache: ObjectCache
    font_setting_cache: ObjectCache
    operand_decode_cache: ObjectCache
    decoder_cache: dict[tuple[int, int] | int, FontDecoder]
    kw_cache: dict[bytes, object]
    pending_run: TextRun | None
    op_handlers: dict[str, OperationHandler]
    op_handlers_bytes: dict[bytes, OperationHandler]
    single_op_handlers: list[OperationHandler | None]
    double_op_handlers: dict[int, OperationHandler]

    __slots__ = (
        "document",
        "page",
        "capture_runs",
        "capture_glyphs",
        "capture_glyph_bitmaps",
        "capture_graphics",
        "capture_clipping",
        "runs",
        "glyphs",
        "glyph_clusters",
        "lines",
        "drawings",
        "current_path",
        "current_point",
        "subpath_start",
        "stack",
        "tm_a",
        "tm_b",
        "tm_c",
        "tm_d",
        "tm_e",
        "tm_f",
        "ca",
        "cb",
        "cc",
        "cd",
        "ce",
        "cf",
        "lm_a",
        "lm_b",
        "lm_c",
        "lm_d",
        "lm_e",
        "lm_f",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "blend_mode",
        "group_alpha",
        "flatness",
        "render_intent",
        "clip_bbox",
        "fill_color_space",
        "stroke_color_space",
        "compatibility_depth",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "font_operand",
        "font_size_operand",
        "font_scale",
        "font_ascent",
        "font_descent",
        "font_space_width",
        "font_widths",
        "text_advance_scale",
        "char_space_scale",
        "word_space_scale",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
        "current_decoder_resources_id",
        "sequence",
        "stream_order",
        "xobject_depth",
        "marked_content_stack",
        "active_streams",
        "queued_stream",
        "resources",
        "resources_id",
        "hidden_layers",
        "resolved_resource_categories",
        "resource_cache",
        "color_space_cache",
        "color_normalization_cache",
        "extgstate_cache",
        "font_setting_cache",
        "operand_decode_cache",
        "decoder_cache",
        "kw_cache",
        "pending_line_break",
        "pending_run",
        "invisible_text_layer",
        "op_handlers",
        "op_handlers_bytes",
        "single_op_handlers",
        "double_op_handlers",
        "resolve",
        "resolve_dict",
        "resolve_name",
        "resolve_name_like_value",
        "resolve_str",
        "combined_A",
        "combined_B",
        "combined_C",
        "combined_D",
        "cached_rotation",
        "is_garbage",
        "operands",
        "run_pool",
        "run_pool_idx",
        "inline_images",
    )

    def __init__(
        self,
        document: TextDocument,
        page: PdfDict,
        hidden_layers: frozenset[str] = frozenset(),
        capture_runs: bool = True,
        capture_glyphs: bool = False,
        capture_glyph_bitmaps: bool = True,
        capture_graphics: bool = False,
        capture_clipping: bool = True,
        decoder_cache: dict[tuple[int, int] | int, "FontDecoder"] | None = None,
    ):
        self.document = document
        self.page = page

        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.fill_color = (0.0, 0.0, 0.0)
        self.fill_pattern = None
        self.fill_opacity = 1.0
        self.stroke_color = (0.0, 0.0, 0.0)
        self.stroke_pattern = None
        self.stroke_opacity = 1.0
        self.blend_mode = None
        self.group_alpha = None
        self.flatness = 0
        self.render_intent = None
        self.clip_bbox = None
        self.fill_color_space = "DeviceGray"
        self.stroke_color_space = "DeviceGray"
        self.line_width = 1.0
        self.line_cap = 0
        self.line_join = 0
        self.miter_limit = 10.0
        self.dash_pattern = ([], 0.0)
        self.stack = []
        self.capture_runs = capture_runs
        self.capture_glyphs = capture_glyphs
        self.capture_glyph_bitmaps = capture_glyph_bitmaps
        self.capture_graphics = capture_graphics
        self.capture_clipping = capture_clipping
        self.runs = []
        self.glyphs = []
        self.glyph_clusters = []
        self.lines = []
        self.drawings = []
        self.current_path = CapturedPath()
        self.current_point = None
        self.subpath_start = None
        self.font_size = 12.0
        self.font_operand = None
        self.font_size_operand = None
        self.horizontal_scale = 100.0
        self.char_space = 0.0
        self.word_space = 0.0
        self.font_scale = self.font_size / 1000.0
        self.font_ascent = 0.0
        self.font_descent = 0.0
        self.font_space_width = 0.0
        self.font_widths = None
        self.text_advance_scale = self.font_size * self.horizontal_scale / 100000.0
        self.char_space_scale = 0.0
        self.word_space_scale = 0.0
        self.rise = 0.0
        self.leading = 0.0
        self.render_mode = 0
        self.current_font = None
        self.current_decoder = None
        self.current_decoder_resources_id = None
        self.sequence = 0
        self.stream_order = -1
        self.xobject_depth = 0
        self.compatibility_depth = 0
        self.marked_content_stack = []
        self.active_streams = set()
        self.queued_stream = None
        self.resources = {}
        self.resources_id = 0
        self.hidden_layers = hidden_layers
        self.resolved_resource_categories = {}
        self.resource_cache = {}
        self.color_space_cache = {}
        self.color_normalization_cache = {}
        self.extgstate_cache = {}
        self.font_setting_cache = {}
        self.operand_decode_cache = {}
        self.decoder_cache = decoder_cache if decoder_cache is not None else {}
        self.kw_cache = getattr(self.document.resolver, "kw_cache", {})
        self.pending_line_break = False
        self.pending_run = None
        self.invisible_text_layer = False
        cls = type(self)
        shared_attr = (
            "shared_operator_tables_graphics"
            if self.capture_graphics or self.capture_clipping
            else "shared_operator_tables_text"
        )
        shared = getattr(cls, shared_attr, None)
        if shared is None:
            op_map = (
                self.TEXT_OP
                if self.capture_graphics or self.capture_clipping
                else self.TEXT_ONLY_OP
            )
            op_handlers = {op: getattr(cls, method) for op, method in op_map.items()}
            op_handlers_bytes = {
                op.encode("latin-1"): handler for op, handler in op_handlers.items()
            }
            single_op_handlers: list[OperationHandler | None] = [None] * 256
            double_op_handlers = {}
            for op, handler in op_handlers.items():
                if len(op) == 1:
                    single_op_handlers[ord(op[0])] = handler
                elif len(op) == 2:
                    double_op_handlers[(ord(op[0]) << 8) | ord(op[1])] = handler
            shared = (
                op_handlers,
                op_handlers_bytes,
                single_op_handlers,
                double_op_handlers,
            )
            setattr(cls, shared_attr, shared)
        (
            self.op_handlers,
            self.op_handlers_bytes,
            self.single_op_handlers,
            self.double_op_handlers,
        ) = shared

        self.combined_A = 1.0
        self.combined_B = 0.0
        self.combined_C = 0.0
        self.combined_D = 1.0
        self.cached_rotation = 0

        self.resolve = self.document.resolver.resolve
        self.resolve_dict = self.document.resolver.resolve_dict
        self.resolve_name = self.document.resolver.resolve_name
        self.resolve_name_like_value = getattr(
            self.document.resolver,
            "resolve_name_like_value",
            self.resolve_name,
        )
        self.resolve_str = self.document.resolver.resolve_str

        self.is_garbage = is_garbage_text
        self.operands: list[ContentOperand] = [None] * 16
        self.run_pool: list[TextRun] = []
        self.run_pool_idx: int = 0
        self.inline_images: list[InlineImageRecord] = []

    @property
    def ctm(self) -> Matrix:
        return Matrix(self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)

    @ctm.setter
    def ctm(self, val: Matrix) -> None:
        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = val
        self.update_combined()

    @property
    def text_matrix(self) -> Matrix:
        return Matrix(self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f)

    @text_matrix.setter
    def text_matrix(self, val: Matrix) -> None:
        self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f = val
        self.update_combined()

    @property
    def line_matrix(self) -> Matrix:
        return Matrix(self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f)

    @line_matrix.setter
    def line_matrix(self, val: Matrix) -> None:
        self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f = val

    def update_combined(self) -> None:
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        ca, cb, cc, cd = self.ca, self.cb, self.cc, self.cd

        if ta == 1.0 and tb == 0.0 and tc == 0.0 and td == 1.0:
            A, B, C, D = ca, cb, cc, cd
        else:
            A = ta * ca + tb * cc
            B = ta * cb + tb * cd
            C = tc * ca + td * cc
            D = tc * cb + td * cd

        self.combined_A = A
        self.combined_B = B
        self.combined_C = C
        self.combined_D = D

        if A == 1.0 and B == 0.0 and C == 0.0 and D == 1.0:
            self.cached_rotation = 0
        else:
            self.cached_rotation = detect_rotation_from_linear(A, B, C, D)

    def append_cubic_curve(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        segments: int | None = None,
    ) -> None:
        if self.current_point is None:
            self.current_point = (x3, y3)
            return
        x0, y0 = self.current_point
        if segments is None:
            x_scale = hypot(self.ca, self.cb)
            y_scale = hypot(self.cc, self.cd)
            scale = max(x_scale, y_scale, 1.0)
            control_len = (
                hypot(x1 - x0, y1 - y0) + hypot(x2 - x1, y2 - y1) + hypot(x3 - x2, y3 - y2)
            )
            flatness = max(0.1, float(self.flatness) if self.flatness else 0.25)
            segments = max(4, min(128, ceil(control_len * scale / (flatness * 8.0))))
        prev_x, prev_y = x0, y0
        draw_path = (
            self.capture_clipping
            or (self.capture_graphics or self.capture_glyphs)
            and self.is_graphics_visible()
        )
        path = self.current_path if draw_path else None
        segment_step = 1.0 / segments
        for i in range(1, segments + 1):
            t = i * segment_step
            mt = 1.0 - t
            mt2 = mt * mt
            t2 = t * t
            b0 = mt2 * mt
            b1 = 3.0 * mt2 * t
            b2 = 3.0 * mt * t2
            b3 = t2 * t
            px = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
            py = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
            if path is not None:
                if not path.subpaths:
                    path.move_to(prev_x, prev_y)
                path.line_to(px, py)
            prev_x, prev_y = px, py
        self.current_point = (x3, y3)

    def update_text_scales(self) -> None:
        fs = self.font_size
        self.font_scale = fs / 1000.0
        self.text_advance_scale = fs * self.horizontal_scale / 100000.0
        if fs:
            self.char_space_scale = self.char_space * 1000.0 / fs
            self.word_space_scale = self.word_space * 1000.0 / fs
        else:
            self.char_space_scale = 0.0
            self.word_space_scale = 0.0

    def update_char_space_scale(self) -> None:
        fs = self.font_size
        self.char_space_scale = self.char_space * 1000.0 / fs if fs else 0.0

    def update_word_space_scale(self) -> None:
        fs = self.font_size
        self.word_space_scale = self.word_space * 1000.0 / fs if fs else 0.0

    def update_horizontal_text_scale(self) -> None:
        self.text_advance_scale = self.font_size * self.horizontal_scale / 100000.0

    def update_font_metrics(self) -> None:
        decoder = self.current_decoder
        if decoder is None:
            self.font_ascent = 0.0
            self.font_descent = 0.0
            self.font_space_width = 0.0
            self.font_widths = None
            return
        self.font_ascent = decoder.ascent * self.font_scale
        self.font_descent = decoder.descent * self.font_scale
        self.font_space_width = decoder.glyph_width(32) * self.font_size * 0.001
        self.font_widths = decoder.fast_widths

    def shift_line(self, tx: float = 0.0, ty: float = 0.0, *, set_leading: bool = False) -> None:
        if set_leading:
            self.leading = -ty
        self.tm_e += tx
        self.tm_f += ty
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def apply_operation(self, operation: tuple[str, ContentOperands], depth: int) -> None:
        operator, operands = operation
        handler = self.op_handlers.get(operator)
        if handler is not None:
            handler(
                typing.cast(OperationTarget, self),
                OperandWindow(list(operands), len(operands)),
                depth,
            )

    def capture_stream_state(self) -> StreamState:
        return StreamState(
            resources=self.resources,
            resources_id=self.resources_id,
            ctm=self.ctm,
            text_matrix=self.text_matrix,
            line_matrix=self.line_matrix,
            font_size=self.font_size,
            font_operand=self.font_operand,
            font_size_operand=self.font_size_operand,
            current_font=self.current_font,
            current_decoder=self.current_decoder,
            current_decoder_resources_id=self.current_decoder_resources_id,
            graphics_stack_len=len(self.stack),
            marked_content_stack_len=len(self.marked_content_stack),
            fill_color=self.fill_color,
            fill_pattern=self.fill_pattern,
            fill_opacity=self.fill_opacity,
            stroke_color=self.stroke_color,
            stroke_pattern=self.stroke_pattern,
            compatibility_depth=self.compatibility_depth,
            fill_color_space=self.fill_color_space,
            stroke_color_space=self.stroke_color_space,
            blend_mode=self.blend_mode,
            group_alpha=self.group_alpha,
            flatness=self.flatness,
            render_intent=self.render_intent,
            clip_bbox=self.clip_bbox,
            pending_line_break=self.pending_line_break,
            xobject_depth=self.xobject_depth,
            resource_cache=self.resource_cache,
            resolved_resource_categories=self.resolved_resource_categories,
        )

    def restore_stream_state(self, state: StreamState) -> None:
        self.resources = state.resources
        self.resources_id = state.resources_id
        self.ctm = state.ctm
        self.text_matrix = state.text_matrix
        self.line_matrix = state.line_matrix
        self.font_size = state.font_size
        self.font_operand = state.font_operand
        self.font_size_operand = state.font_size_operand
        self.current_font = state.current_font
        self.current_decoder = state.current_decoder
        self.current_decoder_resources_id = state.current_decoder_resources_id
        del self.stack[state.graphics_stack_len :]
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.fill_color = state.fill_color
        self.fill_pattern = state.fill_pattern
        self.fill_opacity = state.fill_opacity
        self.stroke_color = state.stroke_color
        self.stroke_pattern = state.stroke_pattern
        self.fill_color_space = state.fill_color_space
        self.stroke_color_space = state.stroke_color_space
        self.compatibility_depth = state.compatibility_depth
        self.blend_mode = getattr(state, "blend_mode", None)
        self.group_alpha = getattr(state, "group_alpha", None)
        self.flatness = getattr(state, "flatness", 0)
        self.render_intent = getattr(state, "render_intent", None)
        self.clip_bbox = state.clip_bbox
        self.pending_line_break = state.pending_line_break
        self.xobject_depth = state.xobject_depth
        self.resource_cache = state.resource_cache
        self.resolved_resource_categories = state.resolved_resource_categories
        self.update_text_scales()
        self.update_font_metrics()

    @staticmethod
    def stream_execution_key(stream: PdfStream) -> StreamKey:
        return ("stream", id(stream), len(stream.raw_data))

    def queue_stream(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: tuple[float, float, float, float] | None = None,
        group_alpha: float | None = None,
        stream_key: StreamKey | None = None,
        swallow_parse_errors: bool = False,
    ) -> None:
        if depth > 10:
            return
        execution_key = stream_key or self.stream_execution_key(stream)
        if execution_key in self.active_streams:
            return
        self.queued_stream = ContentStreamFrame(
            stream,
            resources,
            ctm,
            depth,
            clip_bbox,
            group_alpha,
            stream_key=execution_key,
            swallow_parse_errors=swallow_parse_errors,
        )
        raise NestedStreamRequest

    def enter_stream_frame(self, frame: ContentStreamFrame) -> bool:
        if frame.depth > 10:
            return False
        stream_key = frame.stream_key or self.stream_execution_key(frame.stream)
        if stream_key in self.active_streams:
            return False

        self.active_streams.add(stream_key)
        frame.stream_key = stream_key
        if frame.group_alpha is not None:
            frame.outer_group_alpha = self.group_alpha
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.blend_mode,
                    kind="group-begin",
                )
            )
            self.group_alpha = frame.group_alpha

        frame.old_state = self.capture_stream_state()
        self.resources = frame.resources
        self.resources_id = id(frame.resources)
        self.ctm = frame.ctm
        self.xobject_depth = frame.depth
        if frame.clip_bbox is not None:
            if self.clip_bbox is None:
                self.clip_bbox = frame.clip_bbox
            else:
                self.clip_bbox = (
                    max(self.clip_bbox[0], frame.clip_bbox[0]),
                    max(self.clip_bbox[1], frame.clip_bbox[1]),
                    min(self.clip_bbox[2], frame.clip_bbox[2]),
                    min(self.clip_bbox[3], frame.clip_bbox[3]),
                )

        self.pending_line_break = False
        self.stream_order += 1
        frame.lexer = PdfLexer(frame.stream.data_view, kw_cache=self.kw_cache)
        frame.entered = True
        return True

    def exit_stream_frame(self, frame: ContentStreamFrame) -> None:
        old_state = frame.old_state
        stream_key = frame.stream_key
        if old_state is not None:
            self.restore_stream_state(old_state)
        if stream_key is not None:
            self.active_streams.remove(stream_key)
        if frame.group_alpha is not None:
            self.group_alpha = frame.outer_group_alpha
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.blend_mode,
                    kind="group-end",
                )
            )

    def consume_stream(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        stream_stack = [
            ContentStreamFrame(stream, resources, ctm, depth, clip_bbox),
        ]
        while stream_stack:
            frame = stream_stack.pop()
            if not frame.entered and not self.enter_stream_frame(frame):
                continue
            lexer = frame.lexer
            if lexer is None:
                self.exit_stream_frame(frame)
                continue
            try:
                if (
                    not self.capture_graphics
                    and not self.capture_glyphs
                    and not content_stream_may_show_text(frame.stream.data_view)
                ):
                    self.flush_run()
                    self.exit_stream_frame(frame)
                    continue

                typing.cast(typing.Any, dispatch_operations)(
                    lexer,
                    self.op_handlers,
                    self.op_handlers_bytes,
                    self.single_op_handlers,
                    self.double_op_handlers,
                    typing.cast(OperationTarget, self),
                    frame.depth,
                    operands=self.operands,
                )

                self.flush_run()
            except NestedStreamRequest:
                queued_stream = self.queued_stream
                self.queued_stream = None
                stream_stack.append(frame)
                if queued_stream is not None:
                    stream_stack.append(queued_stream)
                continue
            except PdfParseError:
                self.exit_stream_frame(frame)
                if not frame.swallow_parse_errors:
                    raise
                continue
            except Exception:
                self.exit_stream_frame(frame)
                raise
            else:
                self.exit_stream_frame(frame)

    def lookup_page_resource(
        self, category: str, name: str, parent_category: str | None = None
    ) -> object:

        cache_key = (self.resources_id, category)
        cat_cache = self.resource_cache.get(cache_key)
        if cat_cache is None:
            self.resource_cache[cache_key] = cat_cache = {}
        else:
            res = cat_cache.get(name, MISSING)
            if res is not MISSING:
                return res

        category_res = self.resolved_resource_categories.get(cache_key, MISSING)
        if category_res is MISSING:
            raw_category = lookup_dict_key(self.resources, category)
            category_res = self.resolve_dict(raw_category) if raw_category is not None else None
            self.resolved_resource_categories[cache_key] = category_res

        if isinstance(category_res, dict):
            res = lookup_dict_key(category_res, name)
            if res is not None:
                resolved = self.resolve(res)
                cat_cache[name] = resolved
                return resolved

        if parent_category:
            parent_key = (self.resources_id, parent_category)
            parent_res = self.resolved_resource_categories.get(parent_key, MISSING)
            if parent_res is MISSING:
                raw_parent = lookup_dict_key(self.resources, parent_category)
                parent_res = self.resolve_dict(raw_parent) if raw_parent is not None else None
                self.resolved_resource_categories[parent_key] = parent_res

            if isinstance(parent_res, dict):
                sub_res_dict = None
                for p_val in parent_res.values():
                    if isinstance(p_val, dict):
                        sub_res_dict = lookup_dict_key(p_val, "Resources")
                    if isinstance(sub_res_dict, dict):
                        sub_cat = lookup_dict_key(sub_res_dict, category)
                        if isinstance(sub_cat, dict):
                            found = lookup_dict_key(sub_cat, name)
                            if found is not None:
                                resolved = self.resolve(found)
                                cat_cache[name] = resolved
                                return resolved

        cat_cache[name] = None
        return None

    def chunk_advance(
        self, code: int, decoder: "FontDecoder", *, char_code: int | None = None
    ) -> float:
        if decoder.is_vertical:
            metric = decoder.vertical_metrics.get(
                code, (decoder.default_vertical_width, decoder.glyph_width(code) / 2.0, 0.0)
            )
            word = char_code == 32 if char_code is not None else code == 32
            extra = self.char_space_scale + (self.word_space_scale if word else 0.0)
            return (metric[0] + extra) * self.font_size / 100000.0
        scale = self.text_advance_scale
        base = decoder.glyph_width(code)
        char_extra = self.char_space_scale
        word_extra = self.word_space_scale if code == 32 else 0.0
        return (base + char_extra + word_extra) * scale

    def decode_operand(
        self, operand: object, decoder: FontDecoder
    ) -> tuple[str, bytes, DecodedGlyphs]:
        if type(operand) is PdfString:
            data = operand.data
            needs_decode = True
            text = None
        elif type(operand) is bytes:
            data = operand
            needs_decode = True
            text = None
        elif type(operand) is str:
            data = cached_encode_latin1(operand)
            needs_decode = False
            text = operand
        else:
            text = self.resolve_str(operand)
            if text is None:
                return "", b"", None
            data = cached_encode_latin1(text)
            needs_decode = False

        glyphs = None
        if needs_decode:
            glyphs = decoder.decode_glyphs(data)
            text = "".join(glyph.unicode for glyph in glyphs)
        if self.capture_glyphs:
            if glyphs is None:
                glyphs = decoder.decode_glyphs(data)
            if text is None:
                text = "".join(glyph.unicode for glyph in glyphs)
        if text is None:
            text = ""
        return text, data, glyphs

    def is_text_visible(self, text: str) -> bool:
        if not text:
            return False
        first_code = ord(text[0])
        if (first_code < 32 or 0xE000 <= first_code <= 0xF8FF) and self.is_garbage(text):
            return False
        if not self.marked_content_stack and self.render_mode != 3 and self.font_size >= 0.1:
            return True

        if self.render_mode == 3 or self.font_size < 0.1:
            if not any(r.visible for r in self.runs):
                self.invisible_text_layer = True
            if not self.invisible_text_layer:
                return False

        for entry in self.marked_content_stack:
            layer = getattr(entry, "layer", entry)
            if layer and layer in self.hidden_layers:
                return False
        return True

    def is_graphics_visible(self) -> bool:
        if self.marked_content_stack:
            for entry in self.marked_content_stack:
                layer = getattr(entry, "layer", entry)
                if layer and layer in self.hidden_layers:
                    return False
        return True

    def get_decoder(self, *, update_metrics: bool = True) -> "FontDecoder":
        if self.current_decoder is not None:
            return self.current_decoder

        try:
            font_obj_ref = (
                self.lookup_page_resource("Font", self.current_font) if self.current_font else None
            )
        except PdfParseError:
            font_obj_ref = None
        if font_obj_ref is None:
            return FontDecoder({})

        try:
            font_obj = self.document.resolver.resolve(font_obj_ref)
        except PdfParseError:
            font_obj = None
        if isinstance(font_obj, PdfStream):
            font_obj = font_obj.dictionary
        if not isinstance(font_obj, dict):
            decoder = FontDecoder({})
            self.current_decoder = decoder
            self.current_decoder_resources_id = self.resources_id
            if update_metrics:
                self.update_font_metrics()
            return decoder

        cache_key = (
            (font_obj_ref.object_number, font_obj_ref.generation_number)
            if isinstance(font_obj_ref, PdfReference)
            else id(font_obj_ref)
        )
        doc_cache = self.document.decoder_cache
        cached_decoder = doc_cache.get(cache_key, MISSING)
        if cached_decoder is not MISSING:
            decoder = typing.cast("FontDecoder", cached_decoder)
            self.current_decoder = decoder
            self.current_decoder_resources_id = self.resources_id
            return decoder

        font_dict = typing.cast(PdfDict, font_obj)
        resolved_font = self.document.resolver.resolve_font_dict(font_dict)
        self.current_decoder = FontDecoder(
            typing.cast(dict[str, object], resolved_font),
            ligature_overrides=detect_ligature_overrides(
                self.document, self.resources, resolved_font
            ),
        )
        decoder = self.current_decoder
        self.current_decoder_resources_id = self.resources_id
        doc_cache[cache_key] = decoder
        if update_metrics:
            self.update_font_metrics()
        return decoder
