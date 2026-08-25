# SPDX-License-Identifier: AGPL-3.0-only
"""Content-stream interpreter state.

Holds the graphics and text state, the operator handlers, and glyph emission.
"""

from __future__ import annotations

import contextlib
import typing
from math import ceil, hypot
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_content.inline_images import InlineImage

from core_pdf.impl.engine.image_cache import ImageCache
from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.layout.glyph_table import GlyphSegment, GlyphTableBuilder
from core_pdf.impl.engine.layout.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
    CapturedPath,
    can_merge_cross_font_word,
    gap_separator,
    glyph_bitmap_dimensions,
    glyph_cluster_from_observations,
    glyph_ink_rect,
    glyph_text_space_boxes,
    glyph_unicode_confidence,
    normalize_extracted_text,
    should_capture_glyph_bitmap,
    should_capture_suspicious_multi_glyph_bitmap,
    transformed_text_line,
    transformed_text_rect,
    type3_font_matrix,
    type3_glyph_names,
)
from core_pdf.impl.engine.spec.s_07_content.geometry import transform_bbox
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
    iter_content_operations,
)
from core_pdf.impl.engine.spec.s_07_content.operator_tables import build_operator_tables
from core_pdf.impl.engine.spec.s_07_content.operators import detect_rotation_from_linear
from core_pdf.impl.engine.spec.s_07_content.stream_state import (
    ContentStreamFrame,
    LayoutFormId,
    ResolvedResourceCache,
    ResourceCache,
    StreamKey,
    StreamState,
)
from core_pdf.impl.engine.spec.s_07_content.text_helpers import (
    NO_SPACE_AFTER,
    NO_SPACE_BEFORE,
    cached_encode_latin1,
    detect_ligature_overrides,
    is_garbage_text,
)
from core_pdf.impl.engine.spec.s_07_document.document_lock import document_cache_lock
from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
    parse_float,
    parse_int,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_objects.resolver_values import PdfValueResolver
from core_pdf.impl.engine.spec.s_07_syntax.content_operators import TYPE3_REPLAY_OPERATORS
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.engine.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.engine.spec.s_09_fonts.decoder import (
    DecodedGlyph,
    FontDecoder,
    Type3CharProcProgram,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import (
    MISSING,
    PdfName,
    PdfReference,
    PdfString,
)
from core_pdf.impl.types import PdfDict, Rectangle

OperationHandler: TypeAlias = StateOperationHandler
ObjectCache: TypeAlias = dict[object, object]
InlineImageRecord = CapturedInlineImage
DecodedGlyphs: TypeAlias = tuple[DecodedGlyph, ...] | None
internal_GraphicsState: TypeAlias = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    tuple[float, ...] | None,
    PdfDict | None,
    float | None,
    tuple[float, ...] | None,
    PdfDict | None,
    float | None,
    str,
    str,
    int,
    str | None,
    float | None,
    int,
    str | None,
    Rectangle | None,
    float,
    int,
    int,
    float,
    tuple[list[float], float],
    float,
    object,
    object,
    float,
    float,
    float,
    float,
    float,
    int,
    str | None,
    FontDecoder | None,
]

TYPE3_REPLAY_OPERAND_TYPES = (int, float, PdfName, PdfString)


class TextResolver(PdfValueResolver, typing.Protocol):
    kw_cache: dict[bytes, object]


class TextDocument(typing.Protocol):
    @property
    def resolver(self) -> TextResolver: ...

    decoder_cache: dict[tuple[int, int] | int, FontDecoder]
    image_cache: ImageCache
    internal_cache_lock: Any
    raster_font_provider: Any

    def resolve(self, value: object) -> object: ...


class TextState:
    document: TextDocument
    page: PdfDict
    capture_runs: bool
    capture_glyphs: bool
    capture_glyph_bitmaps: bool
    capture_images: bool
    capture_graphics: bool
    compat_tj_decoder: FontDecoder | None
    runs: list[TextRun]
    glyphs: GlyphTableBuilder
    glyph_clusters: list[GlyphCluster]
    lines: list[CapturedLine]
    drawings: list[CapturedDrawing]
    current_path: CapturedPath
    current_point: tuple[float, float] | None
    subpath_start: tuple[float, float] | None
    stack: list[internal_GraphicsState]
    clip_scope_stack: list[bool]
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
    clip_bbox: Rectangle | None
    layout_form_bbox: Rectangle | None
    layout_form_id: LayoutFormId
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
    type3_uncolored: bool
    resources: PdfDict
    resolved_resource_categories: ResolvedResourceCache
    resource_cache: ResourceCache
    color_space_cache: ObjectCache
    color_normalization_cache: ObjectCache
    extgstate_cache: ObjectCache
    font_setting_cache: ObjectCache
    operand_decode_cache: ObjectCache
    decoder_cache: dict[tuple[int, int] | int, FontDecoder]
    decoder_memo: dict[tuple[int, str | None], FontDecoder]
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
        "capture_images",
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
        "clip_scope_stack",
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
        "layout_form_bbox",
        "layout_form_id",
        "page_clip",
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
        "text_object_id",
        "stream_order",
        "xobject_depth",
        "capture_source",
        "marked_content_stack",
        "active_streams",
        "queued_stream",
        "type3_uncolored",
        "resources",
        "resources_id",
        "hidden_layers",
        "resolved_resource_categories",
        "resource_cache",
        "image_cache",
        "color_space_cache",
        "color_normalization_cache",
        "extgstate_cache",
        "font_setting_cache",
        "operand_decode_cache",
        "decoder_cache",
        "decoder_memo",
        "kw_cache",
        "pending_line_break",
        "pending_run",
        "compat_tj_active",
        "compat_tj_cursor_x",
        "compat_tj_cursor_y",
        "compat_tj_origin_e",
        "compat_tj_origin_f",
        "compat_tj_decoder",
        "compat_tj_need_charspace",
        "invisible_text_layer",
        "op_handlers",
        "op_handlers_bytes",
        "single_op_handlers",
        "double_op_handlers",
        "combined_A",
        "combined_B",
        "combined_C",
        "combined_D",
        "cached_rotation",
        "is_garbage",
        "operands",
        "run_pool",
        "inline_images",
    )

    def __init__(
        self,
        document: TextDocument,
        page: PdfDict,
        hidden_layers: frozenset[str] = frozenset(),
        decoder_cache: dict[tuple[int, int] | int, "FontDecoder"] | None = None,
        page_clip: Rectangle | None = None,
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
        self.layout_form_bbox = None
        self.layout_form_id = None
        # The page box bounds what can be displayed, but it is not a clip the
        # content stream established, so it is kept out of the graphics state:
        # recording it as one would give every unclipped mark the same clip
        # identity as a mark genuinely clipped to the full page, and the
        # provenance those identities feed is what layout groups runs by.
        self.page_clip = page_clip
        self.fill_color_space = "DeviceGray"
        self.stroke_color_space = "DeviceGray"
        self.line_width = 1.0
        self.line_cap = 0
        self.line_join = 0
        self.miter_limit = 10.0
        self.dash_pattern = ([], 0.0)
        self.stack = []
        self.clip_scope_stack = []
        self.capture_runs = True
        self.capture_glyphs = True
        self.capture_glyph_bitmaps = True
        self.capture_images = True
        self.capture_graphics = True
        self.capture_clipping = True
        self.runs = []
        self.glyphs = GlyphTableBuilder()
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
        self.text_object_id = 0
        self.stream_order = -1
        self.xobject_depth = 0
        self.capture_source = "native_text"
        self.compatibility_depth = 0
        self.marked_content_stack = []
        self.active_streams = set()
        self.queued_stream = None
        self.type3_uncolored = False
        self.resources = {}
        self.resources_id = 0
        self.hidden_layers = hidden_layers
        self.resolved_resource_categories = {}
        # Image XObjects can be referenced by multiple pages. Keep their lazy
        # decoders document-scoped so expensive JPEG2000/JBIG2 work is shared.
        image_cache = getattr(document, "image_cache", None)
        if image_cache is None:
            image_cache = ImageCache()
            with contextlib.suppress(AttributeError):
                document.image_cache = image_cache
        self.image_cache = image_cache
        self.resource_cache = {}
        self.color_space_cache = {}
        self.color_normalization_cache = {}
        self.extgstate_cache = {}
        self.font_setting_cache = {}
        self.operand_decode_cache = {}
        self.decoder_cache = decoder_cache if decoder_cache is not None else {}
        self.decoder_memo = {}
        self.kw_cache = getattr(self.document.resolver, "kw_cache", {})
        self.pending_line_break = False
        self.pending_run = None
        self.compat_tj_active = False
        self.compat_tj_cursor_x = 0.0
        self.compat_tj_cursor_y = 0.0
        self.compat_tj_origin_e = 0.0
        self.compat_tj_origin_f = 0.0
        self.compat_tj_decoder = None
        self.compat_tj_need_charspace = False
        self.invisible_text_layer = False
        cls = type(self)
        shared_attr = "shared_operator_tables_graphics"
        shared = getattr(cls, shared_attr, None)
        if shared is None:
            shared = build_operator_tables(
                cls,
                capture_graphics=True,
                capture_clipping=True,
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

        self.is_garbage = is_garbage_text
        self.operands: list[ContentOperand] = [None] * 16
        self.run_pool: list[TextRun] = []
        self.inline_images: list[InlineImageRecord] = []

    def detect_rotation(self, a: float, b: float, c: float, d: float) -> int:
        return detect_rotation_from_linear(a, b, c, d)

    def append_text(
        self,
        operand: Any = None,
        *,
        data: bytes | memoryview | None = None,
        decoder: FontDecoder | None = None,
        string_syntax: str | None = None,
        compatibility_data: bytes | None = None,
    ) -> None:
        effective_decoder = decoder if decoder is not None else self.get_decoder()
        previous_compat_active = self.compat_tj_active
        previous_compat_origin = (self.compat_tj_origin_e, self.compat_tj_origin_f)
        previous_compat_decoder = self.compat_tj_decoder
        previous_compat_need_charspace = self.compat_tj_need_charspace
        self.compat_tj_active = self.capture_glyphs
        self.compat_tj_origin_e = self.lm_e
        self.compat_tj_origin_f = self.lm_f
        self.compat_tj_decoder = effective_decoder
        self.compat_tj_need_charspace = False
        self._append_text_impl(
            operand,
            data=data,
            decoder=effective_decoder,
            string_syntax=string_syntax,
            compatibility_data=compatibility_data,
        )
        self.compat_tj_active = previous_compat_active
        self.compat_tj_origin_e, self.compat_tj_origin_f = previous_compat_origin
        self.compat_tj_decoder = previous_compat_decoder
        self.compat_tj_need_charspace = previous_compat_need_charspace

    def append_tj_array(self, array: Any) -> None:
        self._append_tj_array_impl(array)

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

    def compile_type3_char_proc(self, stream: PdfStream) -> Type3CharProcProgram:
        compiled: list[tuple[OperationHandler, ContentOperands]] = []
        graphics_depth = 0
        try:
            for operator, operands in iter_content_operations(
                PdfLexer(stream.data_view, kw_cache=self.kw_cache)
            ):
                if operator not in TYPE3_REPLAY_OPERATORS:
                    return Type3CharProcProgram(stream, None)
                if any(type(operand) not in TYPE3_REPLAY_OPERAND_TYPES for operand in operands):
                    return Type3CharProcProgram(stream, None)
                if operator == "q":
                    graphics_depth += 1
                elif operator == "Q":
                    if graphics_depth == 0:
                        return Type3CharProcProgram(stream, None)
                    graphics_depth -= 1
                handler = self.op_handlers.get(operator)
                if handler is None:
                    return Type3CharProcProgram(stream, None)
                compiled.append((handler, operands))
        except (PdfParseError, TypeError, ValueError):
            return Type3CharProcProgram(stream, None)
        if graphics_depth:
            return Type3CharProcProgram(stream, None)
        return Type3CharProcProgram(stream, tuple(compiled))

    def type3_char_proc_program(
        self,
        code: int,
        decoder: FontDecoder,
        glyph_names: dict[int, str],
        char_procs: PdfDict,
    ) -> Type3CharProcProgram:
        cache = decoder.type3_charproc_cache
        cached = cache[code]
        if cached is not None:
            decoder.type3_charproc_cache_hits += 1
            return cached

        decoder.type3_charproc_cache_misses += 1
        glyph_name = glyph_names.get(code)
        char_proc = lookup_dict_key(char_procs, glyph_name) if glyph_name else None
        char_proc = self.document.resolver.resolve(char_proc)
        candidate = (
            self.compile_type3_char_proc(char_proc)
            if isinstance(char_proc, PdfStream)
            else Type3CharProcProgram(None, None)
        )
        with document_cache_lock(self.document):
            cached = cache[code]
            if cached is not None:
                return cached
            cache[code] = candidate
            operations = candidate.operations
            if operations is not None:
                decoder.type3_charproc_compiled_programs += 1
                decoder.type3_charproc_compiled_operations += len(operations)
        return candidate

    def consume_compiled_type3_char_proc(
        self,
        program: Type3CharProcProgram,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
    ) -> None:
        stream = program.stream
        operations = program.operations
        if stream is None or operations is None:
            return
        frame = ContentStreamFrame(stream, resources, ctm, depth, None)
        if not self.enter_stream_frame(frame, initialize_lexer=False):
            return
        try:
            if (
                not self.capture_graphics
                and not self.capture_glyphs
                and not content_stream_may_show_text(stream.data_view)
            ):
                self.flush_run()
            else:
                operand_window = OperandWindow(())
                target = typing.cast(OperationTarget, self)
                for handler, operands in operations:
                    operand_window.operands = operands
                    operand_window.count = len(operands)
                    handler(target, operand_window, depth)
                self.flush_run()
        except Exception:
            self.exit_stream_frame(frame)
            raise
        else:
            self.exit_stream_frame(frame)

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
            horizontal_scale=self.horizontal_scale,
            char_space=self.char_space,
            word_space=self.word_space,
            rise=self.rise,
            leading=self.leading,
            render_mode=self.render_mode,
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
            stroke_opacity=self.stroke_opacity,
            line_width=self.line_width,
            line_cap=self.line_cap,
            line_join=self.line_join,
            miter_limit=self.miter_limit,
            dash_pattern=self.dash_pattern,
            compatibility_depth=self.compatibility_depth,
            fill_color_space=self.fill_color_space,
            stroke_color_space=self.stroke_color_space,
            blend_mode=self.blend_mode,
            group_alpha=self.group_alpha,
            flatness=self.flatness,
            render_intent=self.render_intent,
            clip_bbox=self.clip_bbox,
            layout_form_bbox=self.layout_form_bbox,
            layout_form_id=self.layout_form_id,
            pending_line_break=self.pending_line_break,
            compat_tj_cursor_x=self.compat_tj_cursor_x,
            compat_tj_cursor_y=self.compat_tj_cursor_y,
            invisible_text_layer=self.invisible_text_layer,
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
        self.horizontal_scale = state.horizontal_scale
        self.char_space = state.char_space
        self.word_space = state.word_space
        self.rise = state.rise
        self.leading = state.leading
        self.render_mode = state.render_mode
        self.current_font = state.current_font
        self.current_decoder = state.current_decoder
        self.current_decoder_resources_id = state.current_decoder_resources_id
        del self.stack[state.graphics_stack_len :]
        del self.clip_scope_stack[state.graphics_stack_len :]
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.fill_color = state.fill_color
        self.fill_pattern = state.fill_pattern
        self.fill_opacity = state.fill_opacity
        self.stroke_color = state.stroke_color
        self.stroke_pattern = state.stroke_pattern
        self.stroke_opacity = state.stroke_opacity
        self.line_width = state.line_width
        self.line_cap = state.line_cap
        self.line_join = state.line_join
        self.miter_limit = state.miter_limit
        self.dash_pattern = state.dash_pattern
        self.fill_color_space = state.fill_color_space
        self.stroke_color_space = state.stroke_color_space
        self.compatibility_depth = state.compatibility_depth
        self.blend_mode = getattr(state, "blend_mode", None)
        self.group_alpha = getattr(state, "group_alpha", None)
        self.flatness = getattr(state, "flatness", 0)
        self.render_intent = getattr(state, "render_intent", None)
        self.clip_bbox = state.clip_bbox
        self.layout_form_bbox = state.layout_form_bbox
        self.layout_form_id = state.layout_form_id
        self.pending_line_break = state.pending_line_break
        self.compat_tj_cursor_x = state.compat_tj_cursor_x
        self.compat_tj_cursor_y = state.compat_tj_cursor_y
        self.invisible_text_layer = state.invisible_text_layer
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
        clip_bbox: Rectangle | None = None,
        layout_form_bbox: Rectangle | None = None,
        layout_form_id: LayoutFormId = None,
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
            layout_form_bbox=layout_form_bbox,
            layout_form_id=layout_form_id,
            stream_key=execution_key,
            swallow_parse_errors=swallow_parse_errors,
        )
        raise NestedStreamRequest

    def enter_stream_frame(
        self,
        frame: ContentStreamFrame,
        *,
        initialize_lexer: bool = True,
    ) -> bool:
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
        self.layout_form_bbox = frame.layout_form_bbox
        self.layout_form_id = frame.layout_form_id
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
        if initialize_lexer:
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
        clip_bbox: Rectangle | None = None,
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
            category_res = (
                self.document.resolver.resolve_dict(raw_category)
                if raw_category is not None
                else None
            )
            self.resolved_resource_categories[cache_key] = category_res

        if isinstance(category_res, dict):
            res = lookup_dict_key(category_res, name)
            if res is not None:
                resolved = self.document.resolver.resolve(res)
                cat_cache[name] = resolved
                return resolved

        if parent_category:
            parent_key = (self.resources_id, parent_category)
            parent_res = self.resolved_resource_categories.get(parent_key, MISSING)
            if parent_res is MISSING:
                raw_parent = lookup_dict_key(self.resources, parent_category)
                parent_res = (
                    self.document.resolver.resolve_dict(raw_parent)
                    if raw_parent is not None
                    else None
                )
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
                                resolved = self.document.resolver.resolve(found)
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
            return (metric[0] + extra) * self.font_size / 1000.0
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
            text = self.document.resolver.resolve_str(operand)
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

        # Pages routinely alternate between a few fonts; remember resolved
        # decoders per (resources, font name) so switching back skips the
        # resource walk, reference resolution, and document cache lock.
        memo_key = (self.resources_id, self.current_font)
        memoized = self.decoder_memo.get(memo_key)
        if memoized is not None:
            # Mirrors the document-cache hit below, including its metrics
            # behavior (callers refresh metrics themselves after a hit).
            self.current_decoder = memoized
            self.current_decoder_resources_id = self.resources_id
            return memoized

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
        with document_cache_lock(self.document):
            cached_decoder = doc_cache.get(cache_key, MISSING)
            if cached_decoder is not MISSING:
                decoder = typing.cast("FontDecoder", cached_decoder)
                self.current_decoder = decoder
                self.current_decoder_resources_id = self.resources_id
                self.decoder_memo[memo_key] = decoder
                return decoder

            font_dict = typing.cast(PdfDict, font_obj)
            resolved_font = self.document.resolver.resolve_font_dict(font_dict)
            self.current_decoder = FontDecoder(
                typing.cast(dict[str, object], resolved_font),
                ligature_overrides=detect_ligature_overrides(
                    self.document, self.resources, resolved_font
                ),
                raster_font_provider=self.document.raster_font_provider,
            )
            decoder = self.current_decoder
            self.current_decoder_resources_id = self.resources_id
            doc_cache[cache_key] = decoder
            self.decoder_memo[memo_key] = decoder
        if update_metrics:
            self.update_font_metrics()
        return decoder

    def op_Do(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        self.append_xobject(operands[0], depth)

    def append_xobject(self: Any, name_obj: Any, depth: int) -> None:
        name = self.document.resolver.resolve_name(name_obj)
        if not name:
            return
        xobjects = lookup_dict_key(self.resources, "XObject")
        raw_xobj = lookup_dict_key(xobjects, name) if isinstance(xobjects, dict) else None
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.document.resolver.resolve(raw_xobj) if raw_xobj is not None else None
        if xobj is None:
            xobj = self.lookup_page_resource("XObject", name)
        if not isinstance(xobj, PdfStream):
            return
        xobj_dict = xobj.dictionary
        subtype = self.document.resolver.resolve_name(lookup_dict_key(xobj_dict, "Subtype"))
        if self.document.resolver.resolve_name(lookup_dict_key(xobj_dict, "Type")) == "ObjStm":
            return
        if subtype == "Image":
            if self.capture_images and self.is_graphics_visible():
                width = self.document.resolver.resolve_int(lookup_dict_key(xobj_dict, "Width")) or 0
                height = (
                    self.document.resolver.resolve_int(lookup_dict_key(xobj_dict, "Height")) or 0
                )
                bbox = None
                quad = None
                if width > 0 and height > 0:
                    points = (
                        self.transform_point(0.0, 0.0),
                        self.transform_point(1.0, 0.0),
                        self.transform_point(0.0, 1.0),
                        self.transform_point(1.0, 1.0),
                    )
                    quad = points
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    bbox = RectBox(
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                    )
                smask_alpha = None
                soft_mask_raw_data = None
                soft_mask_dictionary = None
                smask = lookup_dict_key(xobj_dict, "SMask")
                if smask is not None:
                    smask_stream = self.document.resolver.resolve(smask)
                    if isinstance(smask_stream, PdfStream):
                        smask_dict = (
                            self.document.resolver.resolve_dict(smask_stream.dictionary) or {}
                        )
                        smask_data = getattr(smask_stream, "raw_data", b"")
                        soft_mask_raw_data = smask_data
                        soft_mask_dictionary = dict(smask_dict)
                        width = (
                            self.document.resolver.resolve_int(lookup_dict_key(smask_dict, "Width"))
                            or 0
                        )
                        height = (
                            self.document.resolver.resolve_int(
                                lookup_dict_key(smask_dict, "Height")
                            )
                            or 0
                        )
                        if width > 0 and height > 0 and smask_data:
                            total = min(len(smask_data), width * height)
                            if total > 0:
                                smask_alpha = sum(smask_data[:total]) / (255.0 * total)
                source_dictionary = dict(xobj_dict)
                if soft_mask_raw_data is not None:
                    source_dictionary["__soft_mask_raw_data__"] = soft_mask_raw_data
                    source_dictionary["__soft_mask_dictionary__"] = soft_mask_dictionary or {}
                self.drawings.append(
                    CapturedDrawing(
                        seqno=self.sequence,
                        fill=None,
                        fill_opacity=None,
                        blend_mode=self.blend_mode,
                        dash_pattern=self.transformed_dash_pattern(),
                        soft_mask_alpha=smask_alpha,
                        kind="image",
                        image_source=self.internal_image_source(
                            stream_key or ("object", id(xobj)),
                            getattr(xobj, "raw_data", b""),
                            source_dictionary,
                        ),
                        image_clip=self.clip_bbox,
                        items=[("quad", quad)] if quad is not None else [],
                        bbox=bbox,
                        stream_order=self.stream_order,
                        xobject_depth=self.xobject_depth,
                    )
                )
                self.drawings[-1].raw_data = getattr(xobj, "raw_data", b"")
                self.drawings[-1].dictionary = dict(xobj_dict)
                if soft_mask_raw_data is not None:
                    self.drawings[-1].dictionary["__soft_mask_raw_data__"] = soft_mask_raw_data
                    self.drawings[-1].dictionary["__soft_mask_dictionary__"] = (
                        soft_mask_dictionary or {}
                    )
            return
        if subtype != "Form":
            return
        group_alpha = None
        group = lookup_dict_key(xobj_dict, "Group")
        if group is not None:
            group_dict = self.document.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.document.resolver.resolve_name(lookup_dict_key(group_dict, "S"))
                == "Transparency"
            ):
                group_alpha_val = self.document.resolver.resolve_float(
                    lookup_dict_key(group_dict, "ca"), default=None
                )
                if group_alpha_val is not None:
                    group_alpha = max(0.0, min(1.0, group_alpha_val))
        raw_resources = lookup_dict_key(xobj_dict, "Resources")
        resources = (
            raw_resources
            if isinstance(raw_resources, dict)
            else self.document.resolver.resolve_dict(raw_resources)
        ) or self.resources
        xobj_matrix = lookup_dict_key(xobj_dict, "Matrix")
        if isinstance(xobj_matrix, (list, tuple)) and len(xobj_matrix) > 6:
            xobj_matrix = xobj_matrix[:6]
        nested_ctm = (
            Matrix.from_operand(xobj_matrix) if xobj_matrix is not None else IDENTITY_MATRIX
        ).multiply(self.ctm)
        raw_form_bbox = lookup_dict_key(xobj_dict, "BBox")
        form_bbox = self.document.resolver.resolve_box(raw_form_bbox)
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        layout_form_bbox = None
        if isinstance(raw_form_bbox, (list, tuple)) and len(raw_form_bbox) >= 4:
            raw_values = tuple(
                self.document.resolver.resolve_float(value, default=None)
                for value in raw_form_bbox[:4]
            )
            if all(value is not None for value in raw_values):
                raw_x, raw_y, raw_width, raw_height = typing.cast(Rectangle, raw_values)
                # PDFMiner's LTFigure constructor historically interprets the
                # four /BBox values as x, y, width, height. Preserve that raw
                # layout geometry separately from the spec-correct clipping
                # rectangle so compatibility projections can reproduce it.
                layout_form_bbox = transform_bbox(
                    (
                        raw_x,
                        raw_y,
                        raw_x + raw_width,
                        raw_y + raw_height,
                    ),
                    nested_ctm,
                )
        self.queue_stream(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            layout_form_bbox=layout_form_bbox,
            # Retain the invocation tree, not merely the outermost form. PDFMiner
            # emits one nested LTFigure per Form XObject even when a child's
            # historical figure bbox extends beyond its parent.
            layout_form_id=(
                *(self.layout_form_id if isinstance(self.layout_form_id, tuple) else ()),
                (stream_key, layout_form_bbox),
            ),
            group_alpha=group_alpha,
            stream_key=stream_key,
            swallow_parse_errors=True,
        )

    def flush_run(self: Any) -> None:
        if not self.capture_runs:
            self.pending_run = None
            return
        if self.pending_run:
            self.pending_run.freeze_glyph_clusters()
            self.runs.append(self.pending_run)
            self.pending_run = None

    def transform_point(self: Any, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.ca + y * self.cc + self.ce,
            x * self.cb + y * self.cd + self.cf,
        )

    def graphics_scale(self: Any) -> float:
        x_scale = hypot(self.ca, self.cb)
        y_scale = hypot(self.cc, self.cd)
        if x_scale == 0 and y_scale == 0:
            return 1.0
        if x_scale == 0:
            return y_scale
        if y_scale == 0:
            return x_scale
        return (x_scale + y_scale) * 0.5

    def transformed_line_width(self: Any) -> float:
        line_width = max(0.0, self.line_width)
        if line_width == 0:
            return 0.0
        return line_width * self.graphics_scale()

    def transformed_dash_pattern(self: Any) -> tuple[list[float], float] | None:
        dash_pattern = self.dash_pattern
        if not dash_pattern:
            return None
        dash_array, phase = dash_pattern
        scale = self.graphics_scale()
        return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale

    def flush_drawing(self: Any, kind: str, fill_rule: str = "nonzero") -> None:
        if not self.capture_graphics or not self.is_graphics_visible():
            self.current_path.clear()
            return

        if (
            self.ca == 1.0
            and self.cb == 0.0
            and self.cc == 0.0
            and self.cd == 1.0
            and self.ce == 0.0
            and self.cf == 0.0
        ):
            path = self.current_path
            self.current_path = CapturedPath()
        else:
            path = self.current_path.transformed(
                Matrix(self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)
            )
            self.current_path.clear()
        if path.has_segments():
            line_width = self.transformed_line_width()
            if len(path.subpaths) == 1 and len(path.subpaths[0].points) == 2:
                (x0, y0), (x1, y1) = path.subpaths[0].points
                if abs(x1 - x0) > 0.01 or abs(y1 - y0) > 0.01:
                    self.lines.append(CapturedLine(x0, y0, x1, y1, line_width))
            else:
                self.lines.extend(path.derived_lines(line_width))
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=self.fill_color,
                    fill_pattern=self.fill_pattern,
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.stroke_color,
                    stroke_pattern=self.stroke_pattern,
                    stroke_opacity=self.stroke_opacity,
                    line_width=line_width,
                    line_cap=self.line_cap,
                    line_join=self.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    kind=kind,
                    path=path,
                    stream_order=self.stream_order,
                    xobject_depth=self.xobject_depth,
                )
            )
            # A painted path must consume a sequence number like text does.
            # Sharing one with the text that follows lets a seqno-ordered
            # replay paint a cell background over the run's first glyphs.
            self.sequence += 1

    def alloc_run(
        self: Any,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None,
        is_vertical: bool,
        rotation_angle: int,
        visible: bool,
        line_break_before: bool,
        seqno: int,
        fill_color: tuple[float, ...] | None,
        advance_bbox: Rectangle | None = None,
        ink_bbox: Rectangle | None = None,
        baseline: tuple[float, float, float, float] | None = None,
        provenance: tuple[tuple[str, object], ...] = (),
        confidence: float | None = None,
        glyph_clusters: tuple[GlyphCluster, ...] = (),
    ) -> TextRun:
        text = normalize_extracted_text(text)
        return self.alloc_prepared_run(
            text,
            x0,
            y0,
            x1,
            y1,
            tx,
            ty,
            font_size,
            space_width,
            order,
            stream_order,
            xobject_depth,
            font_name,
            is_vertical,
            rotation_angle,
            visible,
            line_break_before,
            seqno,
            fill_color,
            advance_bbox,
            ink_bbox,
            baseline,
            provenance,
            confidence,
            glyph_clusters,
        )

    def alloc_prepared_run(
        self: Any,
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        tx: float,
        ty: float,
        font_size: float,
        space_width: float,
        order: int,
        stream_order: int,
        xobject_depth: int,
        font_name: str | None,
        is_vertical: bool,
        rotation_angle: int,
        visible: bool,
        line_break_before: bool,
        seqno: int,
        fill_color: tuple[float, ...] | None,
        advance_bbox: Rectangle | None = None,
        ink_bbox: Rectangle | None = None,
        baseline: tuple[float, float, float, float] | None = None,
        provenance: tuple[tuple[str, object], ...] = (),
        confidence: float | None = None,
        glyph_clusters: tuple[GlyphCluster, ...] = (),
    ) -> TextRun:
        existing = self.run_pool.pop() if self.capture_glyphs and self.run_pool else None
        r = TextRun.reinit(
            existing,
            text,
            x0,
            y0,
            x1,
            y1,
            tx,
            ty,
            font_size,
            space_width,
            order,
            stream_order,
            xobject_depth,
            font_name,
            is_vertical,
            rotation_angle,
            visible,
            line_break_before,
            seqno,
            fill_color,
            advance_bbox,
            ink_bbox,
            baseline,
            provenance,
            confidence,
            glyph_clusters,
        )
        return r

    def internal_is_clipped_away(self: Any, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Report whether a box falls entirely outside the active clip.

        Text only survives if it overlaps the clip, so a form XObject's /BBox,
        a `W n` clip path and the page box all suppress the marks they exclude.
        Partially clipped text is kept whole: the glyph is on the page, and
        reporting half of one would be worse than reporting it.
        """
        for clip in (self.clip_bbox, self.page_clip):
            if clip is None:
                continue
            if x1 <= clip[0] or x0 >= clip[2] or y1 <= clip[1] or y0 >= clip[3]:
                return True
        return False

    def update_pending_run(self: Any, new_run: TextRun) -> None:
        nc = new_run.coords
        if self.internal_is_clipped_away(
            nc[TextRun.X0], nc[TextRun.Y0], nc[TextRun.X1], nc[TextRun.Y1]
        ):
            if self.capture_glyphs:
                self.run_pool.append(new_run)
            return

        if not self.pending_run:
            self.pending_run = new_run
            return

        p = self.pending_run
        pc = p.coords
        nc = new_run.coords
        p_text = p.text
        new_text = new_run.text
        p_font_size = pc[TextRun.FONT_SIZE]
        p_space_width = pc[TextRun.SPACE_WIDTH]
        p_rotation = p.rotation_angle
        merge_threshold = max(p_space_width * 0.45, 2.0)

        is_same_style = (
            p_rotation == new_run.rotation_angle
            and p.visible == new_run.visible
            and not new_run.line_break_before
            and p_font_size == nc[TextRun.FONT_SIZE]
            and (
                p.font_name == new_run.font_name
                or can_merge_cross_font_word(p_text, new_text)
                or can_merge_cross_font_word(new_text, p_text)
            )
            and p.fill_color == new_run.fill_color
        )

        if is_same_style and p_rotation == 90:
            y_gap = nc[TextRun.Y0] - pc[TextRun.Y1]
            max_y_gap = max(p_space_width * 0.5, p_font_size * 0.8, 2.0)
            if abs(y_gap) > max_y_gap:
                is_same_style = False
        elif is_same_style and p_rotation == 0:
            if abs(pc[TextRun.Y0] - nc[TextRun.Y0]) > p_font_size * 0.5:
                is_same_style = False

        merged = False
        if is_same_style:
            if p_rotation in (0, 90):
                if p_rotation == 0:
                    gap = nc[TextRun.X0] - pc[TextRun.X1]
                    if -2.0 <= gap < merge_threshold:
                        separator = gap_separator(p_text, new_text, gap, p)
                        p.set_text(p_text + separator + new_text)
                        p.union_ink_bbox(new_run.ink_bbox)
                        if nc[TextRun.X1] > pc[TextRun.X1]:
                            p.x1 = nc[TextRun.X1]
                        merged = True
                    else:
                        gap_rtl = pc[TextRun.X0] - nc[TextRun.X1]
                        if -2.0 <= gap_rtl < merge_threshold:
                            separator = gap_separator(new_text, p_text, gap_rtl, p)
                            p.set_text(new_text + separator + p_text)
                            p.union_ink_bbox(new_run.ink_bbox)
                            if nc[TextRun.X0] < pc[TextRun.X0]:
                                p.x0 = nc[TextRun.X0]
                            merged = True
                else:
                    gap = nc[TextRun.Y0] - pc[TextRun.Y1]
                    if -2.0 <= gap < merge_threshold:
                        separator = gap_separator(p_text, new_text, gap, p)
                        p.set_text(p_text + separator + new_text)
                        p.union_ink_bbox(new_run.ink_bbox)
                        if nc[TextRun.Y1] > pc[TextRun.Y1]:
                            p.y1 = nc[TextRun.Y1]
                        merged = True
            else:
                h_gap_inv = pc[TextRun.X0] - nc[TextRun.X1]
                if -2.0 <= h_gap_inv < merge_threshold:
                    separator = gap_separator(p_text, new_text, h_gap_inv, p)
                    p.set_text(p_text + separator + new_text)
                    p.union_ink_bbox(new_run.ink_bbox)
                    if nc[TextRun.X0] < pc[TextRun.X0]:
                        p.x0 = nc[TextRun.X0]
                    merged = True
                else:
                    h_gap_inv_rtl = nc[TextRun.X0] - pc[TextRun.X1]
                    if -2.0 <= h_gap_inv_rtl < merge_threshold:
                        separator = gap_separator(new_text, p_text, h_gap_inv_rtl, p)
                        p.set_text(new_text + separator + p_text)
                        p.union_ink_bbox(new_run.ink_bbox)
                        if nc[TextRun.X1] > pc[TextRun.X1]:
                            p.x1 = nc[TextRun.X1]
                        merged = True

        if not merged:
            p.freeze_glyph_clusters()
            self.runs.append(p)
            self.pending_run = new_run
        else:
            p.extend_glyph_clusters(new_run.glyph_clusters)
            if self.capture_glyphs:
                self.run_pool.append(new_run)

    def merge_pending_horizontal_run(
        self,
        text: str,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        font_size: float,
        font_name: str | None,
        visible: bool,
        fill_color: tuple[float, ...] | None,
    ) -> bool:
        p = self.pending_run
        if p is None or self.pending_line_break:
            return False
        # Refuse the merge for clipped text so it takes the allocating path,
        # where it is dropped rather than absorbed into a visible neighbour.
        if self.internal_is_clipped_away(x0, y0, x1, y1):
            return False

        pc = p.coords
        p_text = p.text
        if (
            p.rotation_angle != 0
            or p.visible != visible
            or pc[TextRun.FONT_SIZE] != font_size
            or p.fill_color != fill_color
            or (
                p.font_name != font_name
                and not can_merge_cross_font_word(p_text, text)
                and not can_merge_cross_font_word(text, p_text)
            )
            or abs(pc[TextRun.Y0] - y0) > font_size * 0.5
        ):
            return False

        gap = x0 - pc[TextRun.X1]
        merge_threshold = max(pc[TextRun.SPACE_WIDTH] * 0.45, 2.0)
        if not (-2.0 <= gap < merge_threshold):
            return False

        p.set_text(p_text + gap_separator(p_text, text, gap, p) + text)
        if x1 > pc[TextRun.X1]:
            p.x1 = x1
            p.ink_bbox = p.advance_bbox
        return True

    def record_glyph_observations(
        self: Any,
        text: str,
        data: bytes | bytearray | memoryview,
        decoder: FontDecoder,
        rotation_angle: int,
        visible: bool,
        glyphs: tuple[DecodedGlyph, ...] | None = None,
        string_syntax: str | None = None,
        compatibility_data: bytes | None = None,
    ) -> (
        tuple[tuple[float, float, float, float], tuple[float, float, float, float], float | None]
        | None
    ):
        """Record observations and return their (advance union, ink union, min
        confidence) aggregate, or ``None`` when nothing was recorded.

        The aggregate replicates ``apply_glyph_geometry_to_run`` exactly —
        same comparison operators over the same values in append order — so
        the caller can skip re-scanning the appended slice.
        """
        if not self.capture_glyphs:
            return None
        if glyphs is None:
            glyphs = decoder.decode_glyphs(data)
        if not glyphs:
            return None

        offset = 0.0
        seqno = self.sequence
        fill = self.fill_color
        font_name = self.current_font
        font_size = self.font_size
        combined_a = self.combined_A
        combined_b = self.combined_B
        combined_c = self.combined_C
        combined_d = self.combined_D
        effective_font_size = font_size * (
            hypot(combined_c, combined_d) if decoder.is_vertical else hypot(combined_a, combined_b)
        )
        effective_font_height = font_size * (
            hypot(combined_a, combined_b) if decoder.is_vertical else hypot(combined_c, combined_d)
        )
        glyph_provenance = (
            ("source", self.capture_source),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            ("text_matrix", (combined_a, combined_b, combined_c, combined_d)),
            ("line_matrix_origin", (self.lm_e, self.lm_f)),
            ("horizontal_scale", self.horizontal_scale),
            ("char_space", self.char_space),
            ("text_rise", self.rise),
            ("string_syntax", string_syntax),
            ("compatibility_data", compatibility_data),
        )
        compat_tj_active = self.compat_tj_active and self.compat_tj_decoder is decoder
        if compat_tj_active:
            compat_cursor_x = self.compat_tj_cursor_x
            compat_cursor_y = self.compat_tj_cursor_y
            compat_need_charspace = self.compat_tj_need_charspace
            compat_spacing_scale = self.horizontal_scale * 0.01
            compat_charspace = self.char_space * compat_spacing_scale
            compat_wordspace = (
                0.0 if decoder.is_cid_font else self.word_space * compat_spacing_scale
            )
            compat_origin_x = (
                self.compat_tj_origin_e * self.ca + self.compat_tj_origin_f * self.cc + self.ce
            )
            compat_origin_y = (
                self.compat_tj_origin_e * self.cb + self.compat_tj_origin_f * self.cd + self.cf
            )
        text_basis = (
            self.tm_e * self.ca + self.tm_f * self.cc + self.ce,
            self.tm_e * self.cb + self.tm_f * self.cd + self.cf,
            combined_a,
            combined_b,
            combined_c,
            combined_d,
        )
        append_glyph = self.glyphs.append_row
        glyph_rows = self.glyphs.rows
        append_glyph_row = glyph_rows.append
        chunk_advance = self.chunk_advance
        glyph_bbox_for_code = decoder.glyph_bbox
        glyph_bbox_cache = decoder.glyph_bbox_cache
        vertical_position = decoder.vertical_glyph_position
        clusters = getattr(self, "glyph_clusters", None)
        if clusters is None:
            clusters = []
            self.glyph_clusters = clusters
        cursor = 0
        is_vertical = decoder.is_vertical
        glyph_width = decoder.glyph_width
        effective_font_name = decoder.font_name or font_name
        capture_glyph_bitmaps = self.capture_glyph_bitmaps
        advance_scale = self.text_advance_scale
        char_space_scale = self.char_space_scale
        word_space_scale = self.word_space_scale
        axis_aligned_horizontal = not is_vertical and combined_b == 0.0 and combined_c == 0.0
        rise = self.rise
        font_scale = self.font_scale
        font_ascent = self.font_ascent
        font_descent = self.font_descent
        glyph_line_width = self.transformed_line_width()
        glyph_dash_pattern = self.transformed_dash_pattern()
        # Per-glyph invariants hoisted out of the loop: clip rectangles for the
        # inlined visibility test, graphics-state fields copied into every
        # observation, and the constant part of the outline transform.
        clip_primary = self.clip_bbox
        clip_page = self.page_clip
        render_mode = self.render_mode
        fill_opacity = self.fill_opacity
        stroke_color = self.stroke_color
        stroke_opacity = self.stroke_opacity
        line_cap = self.line_cap
        line_join = self.line_join
        blend_mode = self.blend_mode
        group_alpha = self.group_alpha
        text_object_id = self.text_object_id
        state_line_width = self.line_width
        transform_a = advance_scale * combined_a
        transform_b = advance_scale * combined_b
        transform_c = font_scale * combined_c
        transform_d = font_scale * combined_d
        rise_offset_x = rise * combined_c
        rise_offset_y = rise * combined_d

        def glyph_transform(
            glyph_offset: float,
            vertical_offset: tuple[float, float] | None = None,
        ) -> tuple[float, float, float, float, float, float]:
            if vertical_offset is None:
                origin_x = glyph_offset
                origin_y = rise
            else:
                position_x, position_y = vertical_offset
                origin_x = position_x
                origin_y = rise + position_y - glyph_offset
            glyph_a = advance_scale * combined_a
            glyph_b = advance_scale * combined_b
            glyph_c = font_scale * combined_c
            glyph_d = font_scale * combined_d
            return (
                glyph_a,
                glyph_b,
                glyph_c,
                glyph_d,
                text_basis[0] + origin_x * combined_a + origin_y * combined_c,
                text_basis[1] + origin_x * combined_b + origin_y * combined_d,
            )

        if axis_aligned_horizontal:
            axis_advance_y0 = text_basis[1] + (font_descent + rise) * combined_d
            axis_advance_y1 = text_basis[1] + (font_ascent + rise) * combined_d
            if axis_advance_y0 > axis_advance_y1:
                axis_advance_y0, axis_advance_y1 = axis_advance_y1, axis_advance_y0
            axis_baseline_y = text_basis[1] + rise * combined_d
        # One segment carries every op-constant observation field for this
        # text-showing operation; fast-path glyphs append compact row tuples
        # that reference it instead of full observations.
        segment = GlyphSegment(
            seqno,
            effective_font_name,
            font_size,
            effective_font_size,
            effective_font_height,
            fill,
            rotation_angle,
            decoder,
            render_mode,
            fill_opacity,
            stroke_color,
            stroke_opacity,
            glyph_line_width,
            line_cap,
            line_join,
            glyph_dash_pattern,
            blend_mode,
            group_alpha,
            text_object_id,
        )
        # Running union of the appended observations' geometry, mirroring
        # apply_glyph_geometry_to_run so the caller need not rescan the slice.
        run_geometry_started = False
        run_advance_x0 = run_advance_y0 = run_advance_x1 = run_advance_y1 = 0.0
        run_ink_x0 = run_ink_y0 = run_ink_x1 = run_ink_y1 = 0.0
        run_confidence: float | None = None
        for glyph in glyphs:
            advance = (
                chunk_advance(
                    glyph.width_code,
                    decoder,
                    char_code=glyph.char_code,
                )
                if is_vertical
                else (
                    glyph_width(glyph.width_code)
                    + char_space_scale
                    + (word_space_scale if glyph.width_code == 32 else 0.0)
                )
                * advance_scale
            )
            chunk_text = glyph.unicode
            if not chunk_text:
                chunk_text = text[cursor : cursor + 1]
            chunk_length = len(chunk_text)
            cursor += max(1, chunk_length)
            if not chunk_text:
                offset += advance
                continue

            if compat_tj_active:
                if compat_need_charspace:
                    if is_vertical:
                        compat_cursor_y += compat_charspace
                    else:
                        compat_cursor_x += compat_charspace
                pdfminer_origin = (
                    compat_cursor_x * combined_a + compat_cursor_y * combined_c + compat_origin_x,
                    compat_cursor_x * combined_b + compat_cursor_y * combined_d + compat_origin_y,
                )
                observation_provenance: tuple[tuple[str, Any], ...] = (
                    *glyph_provenance,
                    ("pdfminer_origin", pdfminer_origin),
                    ("pdfminer_matrix_origin", (compat_origin_x, compat_origin_y)),
                    ("pdfminer_cursor", (compat_cursor_x, compat_cursor_y)),
                    ("pdfminer_need_charspace", compat_need_charspace),
                )
                if is_vertical:
                    metric = decoder.vertical_metrics.get(
                        glyph.width_code,
                        (
                            decoder.default_vertical_width,
                            glyph_width(glyph.width_code) / 2.0,
                            0.0,
                        ),
                    )
                    compat_cursor_y += -float(metric[0]) * 0.001 * font_size * compat_spacing_scale
                    if glyph.char_code == 32:
                        compat_cursor_y += compat_wordspace
                else:
                    compat_cursor_x += (
                        glyph_width(glyph.width_code) * 0.001 * font_size * compat_spacing_scale
                    )
                    if glyph.char_code == 32:
                        compat_cursor_x += compat_wordspace
                compat_need_charspace = True
            else:
                observation_provenance = glyph_provenance

            cluster_id = len(clusters)
            cluster_provenance_id = (seqno, cluster_id)
            if axis_aligned_horizontal:
                advance_x0 = text_basis[0] + offset * combined_a
                advance_x1 = text_basis[0] + (offset + advance) * combined_a
                advance_bbox = (
                    advance_x0 if advance_x0 < advance_x1 else advance_x1,
                    axis_advance_y0,
                    advance_x1 if advance_x1 > advance_x0 else advance_x0,
                    axis_advance_y1,
                )
                baseline = (
                    advance_x0,
                    axis_baseline_y,
                    advance_x1,
                    axis_baseline_y,
                )
            else:
                text_box, baseline_text = glyph_text_space_boxes(
                    self,
                    offset,
                    advance,
                    decoder,
                    vertical_position(glyph.cid, font_size=font_size)
                    if is_vertical
                    else (0.0, 0.0),
                )
                transformed = transformed_text_rect(self, *text_box, text_basis)
                advance_bbox = (
                    transformed.x0,
                    transformed.y0,
                    transformed.x1,
                    transformed.y1,
                )
                baseline = transformed_text_line(*baseline_text, text_basis)
            if is_vertical:
                outline_transform = glyph_transform(
                    offset,
                    vertical_position(glyph.cid, font_size=font_size),
                )
            else:
                # Inlined glyph_transform for the horizontal case; the sums
                # keep the closure's left-to-right evaluation order so the
                # floats are bit-identical.
                outline_transform = (
                    transform_a,
                    transform_b,
                    transform_c,
                    transform_d,
                    text_basis[0] + offset * combined_a + rise_offset_x,
                    text_basis[1] + offset * combined_b + rise_offset_y,
                )
            observation_visible = visible
            if observation_visible:
                box_x0, box_y0, box_x1, box_y1 = advance_bbox
                if (
                    clip_primary is not None
                    and (
                        box_x1 <= clip_primary[0]
                        or box_x0 >= clip_primary[2]
                        or box_y1 <= clip_primary[1]
                        or box_y0 >= clip_primary[3]
                    )
                ) or (
                    clip_page is not None
                    and (
                        box_x1 <= clip_page[0]
                        or box_x0 >= clip_page[2]
                        or box_y1 <= clip_page[1]
                        or box_y0 >= clip_page[3]
                    )
                ):
                    observation_visible = False
            if is_vertical:
                glyph_bbox = None
            else:
                glyph_code = glyph.bitmap_code
                glyph_bbox = glyph_bbox_cache.get(glyph_code)
                if glyph_bbox is None and glyph_code not in glyph_bbox_cache:
                    glyph_bbox = glyph_bbox_for_code(glyph_code)
            if (
                axis_aligned_horizontal
                and glyph_bbox is not None
                and glyph_bbox[0] == 0.0
                and glyph_bbox[1] * font_scale == font_descent
                and glyph_bbox[2] * advance_scale == advance
                and glyph_bbox[3] * font_scale == font_ascent
            ):
                rect = advance_bbox
            else:
                rect = glyph_ink_rect(
                    glyph_bbox,
                    offset,
                    advance_bbox,
                    text_basis,
                    advance_scale,
                    rise,
                    font_scale,
                )
            observation_confidence = glyph_unicode_confidence(
                chunk_text,
                glyph.unicode_source,
                glyph.alternates,
            )

            single_character = chunk_length == 1
            suspicious_multi = (
                False
                if single_character
                else should_capture_suspicious_multi_glyph_bitmap(chunk_text)
            )
            if single_character or suspicious_multi:
                bitmap_width = 0
                bitmap_height = 0
                bitmap_code: int | None = None
                if capture_glyph_bitmaps and (
                    should_capture_glyph_bitmap(chunk_text)
                    if single_character
                    else suspicious_multi
                ):
                    bitmap_width, bitmap_height = glyph_bitmap_dimensions(
                        glyph_bbox,
                        font_size,
                    )
                    bitmap_code = glyph.bitmap_code
                # Single-glyph fast path: append one compact row and a cluster
                # referencing it; the full GlyphObservation materializes only
                # if a row-level consumer asks for it.
                append_glyph_row(
                    (
                        segment,
                        chunk_text,
                        rect,
                        advance_bbox,
                        baseline,
                        glyph.code_bytes,
                        glyph.char_code,
                        glyph.cid,
                        glyph.gid,
                        observation_visible,
                        observation_confidence,
                        glyph.unicode_source,
                        glyph.alternates,
                        bitmap_width,
                        bitmap_height,
                        bitmap_code,
                        outline_transform,
                        observation_provenance,
                        cluster_id,
                    )
                )
                clusters.append(
                    GlyphCluster.from_row(
                        glyph_rows,
                        len(glyph_rows) - 1,
                        cluster_id,
                        chunk_text,
                        advance_bbox,
                        rect,
                        baseline,
                        observation_confidence,
                    )
                )
                if run_geometry_started:
                    box_x0, box_y0, box_x1, box_y1 = advance_bbox
                    if box_x0 < run_advance_x0:
                        run_advance_x0 = box_x0
                    if box_y0 < run_advance_y0:
                        run_advance_y0 = box_y0
                    if box_x1 > run_advance_x1:
                        run_advance_x1 = box_x1
                    if box_y1 > run_advance_y1:
                        run_advance_y1 = box_y1
                    box_x0, box_y0, box_x1, box_y1 = rect
                    if box_x0 < run_ink_x0:
                        run_ink_x0 = box_x0
                    if box_y0 < run_ink_y0:
                        run_ink_y0 = box_y0
                    if box_x1 > run_ink_x1:
                        run_ink_x1 = box_x1
                    if box_y1 > run_ink_y1:
                        run_ink_y1 = box_y1
                    if observation_confidence is not None and (
                        run_confidence is None or observation_confidence < run_confidence
                    ):
                        run_confidence = observation_confidence
                else:
                    run_geometry_started = True
                    run_advance_x0, run_advance_y0, run_advance_x1, run_advance_y1 = advance_bbox
                    run_ink_x0, run_ink_y0, run_ink_x1, run_ink_y1 = rect
                    run_confidence = observation_confidence
                offset += advance
                continue

            cluster_observations: list[GlyphObservation] = []
            if glyph.split_unicode:
                per_char_advance = advance / len(chunk_text)
                char_offset = offset
                for char_index, ch in enumerate(chunk_text):
                    char_confidence = glyph_unicode_confidence(
                        ch,
                        glyph.unicode_source,
                        glyph.alternates,
                    )
                    char_box, char_baseline_text = glyph_text_space_boxes(
                        self, char_offset, per_char_advance, decoder
                    )
                    char_advance_rect = transformed_text_rect(self, *char_box, text_basis)
                    char_baseline = transformed_text_line(*char_baseline_text, text_basis)
                    cluster_observations.append(
                        GlyphObservation(
                            ch,
                            (
                                char_advance_rect.x0,
                                char_advance_rect.y0,
                                char_advance_rect.x1,
                                char_advance_rect.y1,
                            ),
                            (
                                char_advance_rect.x0,
                                char_advance_rect.y0,
                                char_advance_rect.x1,
                                char_advance_rect.y1,
                            ),
                            seqno,
                            glyph.code_bytes,
                            glyph.char_code,
                            glyph.cid,
                            glyph.gid,
                            effective_font_name,
                            font_size,
                            char_baseline,
                            rotation_angle,
                            fill,
                            observation_visible,
                            char_confidence,
                            glyph.unicode_source,
                            glyph.alternates,
                            (),
                            0,
                            0,
                            None,
                            decoder,
                            effective_font_size,
                            effective_font_height,
                            observation_provenance,
                            glyph_transform=outline_transform,
                            text_render_mode=render_mode,
                            fill_opacity=fill_opacity,
                            stroke_color=stroke_color,
                            stroke_opacity=stroke_opacity,
                            line_width=state_line_width,
                            blend_mode=blend_mode,
                            soft_mask_alpha=group_alpha,
                            paint_glyph=char_index == 0,
                            text_object_id=text_object_id,
                            cluster_key=cluster_provenance_id,
                        )
                    )
                    char_offset += per_char_advance
            else:
                cluster_observations.append(
                    GlyphObservation(
                        chunk_text,
                        rect,
                        advance_bbox,
                        seqno,
                        glyph.code_bytes,
                        glyph.char_code,
                        glyph.cid,
                        glyph.gid,
                        effective_font_name,
                        font_size,
                        baseline,
                        rotation_angle,
                        fill,
                        observation_visible,
                        observation_confidence,
                        glyph.unicode_source,
                        glyph.alternates,
                        (),
                        0,
                        0,
                        None,
                        decoder,
                        effective_font_size,
                        effective_font_height,
                        observation_provenance,
                        glyph_transform=outline_transform,
                        text_render_mode=render_mode,
                        fill_opacity=fill_opacity,
                        stroke_color=stroke_color,
                        stroke_opacity=stroke_opacity,
                        line_width=state_line_width,
                        blend_mode=blend_mode,
                        soft_mask_alpha=group_alpha,
                        text_object_id=text_object_id,
                        cluster_key=cluster_provenance_id,
                    )
                )
            for observation in cluster_observations:
                append_glyph(observation)
                if run_geometry_started:
                    box_x0, box_y0, box_x1, box_y1 = observation.advance_bbox
                    if box_x0 < run_advance_x0:
                        run_advance_x0 = box_x0
                    if box_y0 < run_advance_y0:
                        run_advance_y0 = box_y0
                    if box_x1 > run_advance_x1:
                        run_advance_x1 = box_x1
                    if box_y1 > run_advance_y1:
                        run_advance_y1 = box_y1
                    box_x0, box_y0, box_x1, box_y1 = observation.ink_bbox
                    if box_x0 < run_ink_x0:
                        run_ink_x0 = box_x0
                    if box_y0 < run_ink_y0:
                        run_ink_y0 = box_y0
                    if box_x1 > run_ink_x1:
                        run_ink_x1 = box_x1
                    if box_y1 > run_ink_y1:
                        run_ink_y1 = box_y1
                    glyph_confidence = observation.confidence
                    if glyph_confidence is not None and (
                        run_confidence is None or glyph_confidence < run_confidence
                    ):
                        run_confidence = glyph_confidence
                else:
                    run_geometry_started = True
                    run_advance_x0, run_advance_y0, run_advance_x1, run_advance_y1 = (
                        observation.advance_bbox
                    )
                    run_ink_x0, run_ink_y0, run_ink_x1, run_ink_y1 = observation.ink_bbox
                    run_confidence = observation.confidence
            cluster = glyph_cluster_from_observations(
                cluster_id,
                chunk_text,
                tuple(cluster_observations),
            )
            if cluster is not None:
                clusters.append(cluster)
            offset += advance

        if compat_tj_active:
            self.compat_tj_cursor_x = compat_cursor_x
            self.compat_tj_cursor_y = compat_cursor_y
            self.compat_tj_need_charspace = compat_need_charspace
        if not run_geometry_started:
            return None
        return (
            (run_advance_x0, run_advance_y0, run_advance_x1, run_advance_y1),
            (run_ink_x0, run_ink_y0, run_ink_x1, run_ink_y1),
            run_confidence,
        )

    def _append_text_impl(
        self: Any,
        operand: Any = None,
        *,
        data: bytes | memoryview | None = None,
        decoder: FontDecoder | None = None,
        string_syntax: str | None = None,
        compatibility_data: bytes | None = None,
    ) -> None:
        decoder = decoder if decoder is not None else self.get_decoder()

        if data is not None:
            if not self.capture_glyphs:
                data = bytes(data)
            glyphs = None
            if self.capture_glyphs:
                glyphs = decoder.decode_glyphs(data)
                # Keep undecodable painted glyphs in the page program. Native
                # consumers can retain the replacement marker, while legacy
                # facades project the source code as their exact ``(cid:N)``
                # spelling. Dropping them here also lost their cursor advance.
                text = "".join(glyph.unicode for glyph in glyphs)
            else:
                text = decoder.decode(bytes(data))
        else:
            text, data, glyphs = self.decode_operand(operand, decoder)
        data_len = len(data)
        rendered_type3_glyphs = False
        if decoder.is_type3 and self.capture_graphics and data:
            text_matrix = self.text_matrix
            line_matrix = self.line_matrix
            self._render_type3_glyphs_impl(data, decoder)
            rendered_type3_glyphs = True
            self.text_matrix = text_matrix
            self.line_matrix = line_matrix
        if not text:
            if data and rendered_type3_glyphs:
                adv_x, adv_y = decoder.text_advance_vector(
                    data,
                    font_size=self.font_size,
                    char_space=self.char_space,
                    word_space=self.word_space,
                    horizontal_scale=self.horizontal_scale,
                    glyphs=glyphs,
                )
                te, tf = self.tm_e, self.tm_f
                ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
                self.tm_e = te + adv_x * ta + adv_y * tc
                self.tm_f = tf + adv_x * tb + adv_y * td
                self.pending_line_break = False
            return

        simple_horizontal_run = (
            self.capture_runs
            and not self.capture_glyphs
            and self.cached_rotation == 0
            and not decoder.is_vertical
        )
        if (
            simple_horizontal_run
            and not self.marked_content_stack
            and self.render_mode != 3
            and self.font_size >= 0.1
        ):
            visible = True
        else:
            visible = self.is_text_visible(text)

        fs = self.font_size
        rise = self.rise

        if (
            glyphs is None
            and not decoder.is_cid_font
            and decoder.to_unicode is None
            and decoder.cmap is None
        ):
            widths = self.font_widths or decoder.fast_widths
            cs = self.char_space_scale
            ws = self.word_space_scale
            scale = self.text_advance_scale
            if data_len == 1:
                byte = data[0]
                total = widths[byte] + cs
                if byte == 32:
                    total += ws
            else:
                total = 0.0
                space_count = 0
                for b in data:
                    total += widths[b]
                    if b == 32:
                        space_count += 1
                total += data_len * cs + space_count * ws
            if decoder.is_vertical:
                adv_x, adv_y = 0.0, -total * scale
            else:
                adv_x, adv_y = total * scale, 0.0
        else:
            adv_x, adv_y = decoder.text_advance_vector(
                data,
                font_size=fs,
                char_space=self.char_space,
                word_space=self.word_space,
                horizontal_scale=self.horizontal_scale,
                glyphs=glyphs,
            )

        ascent = self.font_ascent
        descent = self.font_descent

        A = self.combined_A
        B = self.combined_B
        C = self.combined_C
        D = self.combined_D

        ca = self.ca
        cb = self.cb
        cc = self.cc
        cd = self.cd
        ce = self.ce
        cf = self.cf
        te, tf = self.tm_e, self.tm_f
        E = te * ca + tf * cc + ce
        F = te * cb + tf * cd + cf

        if decoder.is_vertical:
            c0_x = descent * A + rise * C + E
            c0_y = descent * B + rise * D + F
            c1_x = ascent * A + rise * C + E
            c1_y = ascent * B + rise * D + F
            adv_C = adv_y * C
            adv_D = adv_y * D
            c2_x = adv_C + c0_x
            c2_y = adv_D + c0_y
            c3_x = adv_C + c1_x
            c3_y = adv_D + c1_y
        else:
            ar = ascent + rise
            dr = descent + rise
            c0_x = dr * C + E
            c0_y = dr * D + F
            c1_x = ar * C + E
            c1_y = ar * D + F
            adv_A = adv_x * A
            adv_B = adv_x * B
            c2_x = adv_A + c0_x
            c2_y = adv_B + c0_y
            c3_x = adv_A + c1_x
            c3_y = adv_B + c1_y

        x0 = c0_x if c0_x < c1_x else c1_x
        if c2_x < x0:
            x0 = c2_x
        if c3_x < x0:
            x0 = c3_x

        y0 = c0_y if c0_y < c1_y else c1_y
        if c2_y < y0:
            y0 = c2_y
        if c3_y < y0:
            y0 = c3_y

        x1 = c0_x if c0_x > c1_x else c1_x
        if c2_x > x1:
            x1 = c2_x
        if c3_x > x1:
            x1 = c3_x

        y1 = c0_y if c0_y > c1_y else c1_y
        if c2_y > y1:
            y1 = c2_y
        if c3_y > y1:
            y1 = c3_y

        rot = self.cached_rotation
        seqno = self.sequence
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        scale_factor = hypot(C, D) if decoder.is_vertical else hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_font_height = fs * (hypot(A, B) if decoder.is_vertical else hypot(C, D))
        effective_space_width = self.font_space_width * scale_factor
        baseline = (
            E,
            F,
            E + adv_x * A + adv_y * C,
            F + adv_x * B + adv_y * D,
        )
        provenance = (
            ("source", self.capture_source),
            ("seqno", seqno),
            ("font_name", self.current_font),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("text_render_mode", self.render_mode),
            ("font_size", fs),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            *(
                (("mcid", mcid),)
                if (mcid := self.current_marked_content_mcid()) is not None
                else ()
            ),
        )
        advance_bbox = (x0, y0, x1, y1)

        actual_text_span = self.current_actual_text_span()
        if actual_text_span is not None:
            compatibility_glyphs = glyphs if glyphs is not None else decoder.decode_glyphs(data)
            compatibility_parts: list[str] = []
            for glyph in compatibility_glyphs:
                mapped = (
                    decoder.to_unicode.decode(glyph.code_bytes)
                    if decoder.to_unicode is not None and glyph.code_bytes
                    else glyph.unicode
                )
                # pdfminer represents an explicitly empty ToUnicode target as
                # U+0000 rather than dropping the source character.
                compatibility_parts.append(mapped if mapped else "\x00")
            compatibility_text = "".join(compatibility_parts)
            if self.capture_glyphs:
                glyph_start = len(self.glyphs)
                cluster_start = len(self.glyph_clusters)
                self.record_glyph_observations(
                    text,
                    data,
                    decoder,
                    rot,
                    visible,
                    glyphs=glyphs,
                    string_syntax=string_syntax,
                    compatibility_data=compatibility_data,
                )
                actual_text_span.compatibility_glyphs.extend(self.glyphs.extract_rows(glyph_start))
                self.glyphs.truncate(glyph_start)
                del self.glyph_clusters[cluster_start:]
            actual_text_span.add_extents(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                nbytes=data_len,
                tx=te,
                ty=tf,
                font_size=effective_font_size,
                space_width=effective_space_width,
                order=seqno,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
                font_name=self.current_font,
                is_vertical=decoder.is_vertical,
                rotation_angle=rot,
                visible=visible,
                line_break_before=self.pending_line_break,
                seqno=seqno,
                fill_color=self.fill_color,
                advance_bbox=advance_bbox,
                ink_bbox=advance_bbox,
                baseline=baseline,
                provenance=provenance,
                confidence=1.0,
                font_decoder=decoder,
                effective_font_height=effective_font_height,
                compatibility_text=compatibility_text,
            )
            self.sequence = seqno + 1
            self.tm_e = te + adv_x * ta + adv_y * tc
            self.tm_f = tf + adv_x * tb + adv_y * td
            self.pending_line_break = False
            return

        if not self.capture_runs:
            if self.capture_glyphs:
                self.record_glyph_observations(
                    text,
                    data,
                    decoder,
                    rot,
                    visible,
                    glyphs=glyphs,
                    string_syntax=string_syntax,
                    compatibility_data=compatibility_data,
                )
            self.sequence = seqno + 1
            self.pending_line_break = False
            self.tm_e = te + adv_x * ta + adv_y * tc
            self.tm_f = tf + adv_x * tb + adv_y * td
            return

        prepared_text = text
        prepared_visible = visible
        if simple_horizontal_run:
            normalized_text = normalize_extracted_text(text)
            prepared_text = normalized_text
            if normalized_text and self.merge_pending_horizontal_run(
                normalized_text,
                x0,
                x1,
                y0,
                y1,
                effective_font_size,
                self.current_font,
                prepared_visible,
                self.fill_color,
            ):
                self.sequence = seqno + 1
                self.tm_e = te + adv_x * ta + adv_y * tc
                self.tm_f = tf + adv_x * tb + adv_y * td
                self.pending_line_break = False
                return

        if simple_horizontal_run:
            new_run = self.alloc_prepared_run(
                text=prepared_text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                tx=te,
                ty=tf,
                font_size=effective_font_size,
                font_name=self.current_font,
                space_width=effective_space_width,
                order=seqno,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
                is_vertical=False,
                rotation_angle=0,
                visible=prepared_visible,
                line_break_before=self.pending_line_break,
                seqno=seqno,
                fill_color=self.fill_color,
                advance_bbox=advance_bbox,
                ink_bbox=advance_bbox,
                baseline=baseline,
                provenance=provenance,
                confidence=None,
            )
        else:
            new_run = self.alloc_run(
                text=text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                tx=te,
                ty=tf,
                font_size=effective_font_size,
                font_name=self.current_font,
                space_width=effective_space_width,
                order=seqno,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
                is_vertical=decoder.is_vertical,
                rotation_angle=rot,
                visible=visible,
                line_break_before=self.pending_line_break,
                seqno=seqno,
                fill_color=self.fill_color,
                advance_bbox=advance_bbox,
                ink_bbox=advance_bbox,
                baseline=baseline,
                provenance=provenance,
                confidence=None,
            )
        if self.capture_glyphs:
            cluster_start = len(self.glyph_clusters)
            run_geometry = self.record_glyph_observations(
                text,
                data,
                decoder,
                rot,
                visible,
                glyphs=glyphs,
                string_syntax=string_syntax,
                compatibility_data=compatibility_data,
            )
            recorded_clusters = tuple(self.glyph_clusters[cluster_start:])
            if recorded_clusters:
                new_run.glyph_clusters = recorded_clusters
            if run_geometry is not None:
                run_advance_bbox, run_ink_bbox, run_confidence = run_geometry
                new_run.advance_bbox = run_advance_bbox
                new_run.ink_bbox = run_ink_bbox
                if run_confidence is not None:
                    new_run.confidence = run_confidence

        self.update_pending_run(new_run)

        self.sequence = seqno + 1

        self.tm_e = te + adv_x * ta + adv_y * tc
        self.tm_f = tf + adv_x * tb + adv_y * td
        self.pending_line_break = False

    def _render_type3_glyphs_impl(self: Any, data: bytes, decoder: FontDecoder) -> None:
        font = decoder.font
        char_procs = lookup_dict_key(font, "CharProcs")
        if not isinstance(char_procs, dict):
            return
        glyph_names = decoder.type3_glyph_names
        if glyph_names is None:
            glyph_names = type3_glyph_names(font, decoder)
            decoder.type3_glyph_names = glyph_names

        resources = lookup_dict_key(font, "Resources")
        if not isinstance(resources, dict):
            resources = self.resources
        font_matrix = type3_font_matrix(font)
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale
        scale = self.text_advance_scale

        for code in data:
            program = self.type3_char_proc_program(code, decoder, glyph_names, char_procs)
            char_proc = program.stream
            if char_proc is not None:
                glyph_ctm = Matrix(
                    self.combined_A,
                    self.combined_B,
                    self.combined_C,
                    self.combined_D,
                    self.tm_e * self.ca + self.tm_f * self.cc + self.ce,
                    self.tm_e * self.cb + self.tm_f * self.cd + self.cf,
                ).multiply(font_matrix)
                previous_type3_uncolored = self.type3_uncolored
                self.type3_uncolored = False
                try:
                    if program.operations is None:
                        decoder.type3_charproc_unsafe_fallbacks += 1
                        self.consume_stream(char_proc, resources, glyph_ctm, self.xobject_depth + 1)
                    else:
                        self.consume_compiled_type3_char_proc(
                            program,
                            resources,
                            glyph_ctm,
                            self.xobject_depth + 1,
                        )
                finally:
                    self.type3_uncolored = previous_type3_uncolored

            total = widths[code] + cs
            if code == 32:
                total += ws
            advance = total * scale
            if decoder.is_vertical:
                self.tm_e += -advance * self.tm_c
                self.tm_f += -advance * self.tm_d
            else:
                self.tm_e += advance * self.tm_a
                self.tm_f += advance * self.tm_b

    def append_tj_array_simple(
        self: Any, array: list[Any] | tuple[Any, ...], decoder: FontDecoder
    ) -> None:
        table = decoder.byte_decode_table
        assert table is not None

        if self.cached_rotation == 0 and not decoder.is_vertical:
            self.append_tj_array_simple_horizontal_batched(array, decoder)
            return

        pending_data: bytes | bytearray | None = None
        text_scale = self.text_advance_scale
        adjustment_scale = text_scale
        is_vert = decoder.is_vertical
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale

        fs = self.font_size
        rise = self.rise
        ascent = self.font_ascent
        descent = self.font_descent
        A, B, C, D = self.combined_A, self.combined_B, self.combined_C, self.combined_D
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        te, tf = self.tm_e, self.tm_f
        rot = self.cached_rotation
        scale_factor = hypot(C, D) if is_vert else hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_space_width = self.font_space_width * scale_factor
        stream_order = self.stream_order
        xobject_depth = self.xobject_depth
        font_name = self.current_font
        fill_color = self.fill_color
        seqno = self.sequence
        pending_line_break = self.pending_line_break

        ar = ascent + rise
        dr = descent + rise
        dr_C = dr * C
        dr_D = dr * D
        ar_C = ar * C
        ar_D = ar * D

        for item in array:
            t = type(item)
            if t is bytes:
                item_data = item
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue
            if t is PdfString:
                item_data = item.data
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue

            if t is int or t is float:
                if pending_data:
                    n_data = len(pending_data)
                    if n_data == 1:
                        byte = pending_data[0]
                        text = table[byte]
                        total = widths[byte] + cs
                        if byte == 32:
                            total += ws
                    elif n_data == 2:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        text = table[b0] + table[b1]
                        total = widths[b0] + widths[b1] + (2 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                    elif n_data == 3:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        b2 = pending_data[2]
                        text = table[b0] + table[b1] + table[b2]
                        total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                        if b2 == 32:
                            total += ws
                    else:
                        text = "".join(map(table.__getitem__, pending_data))
                        total = 0.0
                        space_count = 0
                        for byte in pending_data:
                            total += widths[byte]
                            if byte == 32:
                                space_count += 1
                        total += n_data * cs + space_count * ws

                    if text:
                        visible = self.is_text_visible(text)
                        if is_vert:
                            adv_x, adv_y = 0.0, -total * text_scale
                        else:
                            adv_x, adv_y = total * text_scale, 0.0

                        E = te * ca + tf * cc + ce
                        F = te * cb + tf * cd + cf
                        c0_x = dr_C + E
                        c0_y = dr_D + F
                        c1_x = ar_C + E
                        c1_y = ar_D + F
                        adv_A = adv_x * A
                        adv_B = adv_x * B
                        c2_x = adv_A + c0_x
                        c2_y = adv_B + c0_y
                        c3_x = adv_A + c1_x
                        c3_y = adv_B + c1_y

                        x0 = c0_x if c0_x < c1_x else c1_x
                        if c2_x < x0:
                            x0 = c2_x
                        if c3_x < x0:
                            x0 = c3_x
                        y0 = c0_y if c0_y < c1_y else c1_y
                        if c2_y < y0:
                            y0 = c2_y
                        if c3_y < y0:
                            y0 = c3_y
                        x1 = c0_x if c0_x > c1_x else c1_x
                        if c2_x > x1:
                            x1 = c2_x
                        if c3_x > x1:
                            x1 = c3_x
                        y1 = c0_y if c0_y > c1_y else c1_y
                        if c2_y > y1:
                            y1 = c2_y
                        if c3_y > y1:
                            y1 = c3_y

                        new_run = self.alloc_run(
                            text=text,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            tx=te,
                            ty=tf,
                            font_size=effective_font_size,
                            font_name=decoder.font_name or font_name,
                            space_width=effective_space_width,
                            order=seqno,
                            stream_order=stream_order,
                            xobject_depth=xobject_depth,
                            is_vertical=is_vert,
                            rotation_angle=rot,
                            visible=visible,
                            line_break_before=pending_line_break,
                            seqno=seqno,
                            fill_color=fill_color,
                        )
                        self.update_pending_run(new_run)
                        seqno += 1
                        pending_line_break = False
                        te = te + adv_x * ta + adv_y * tc
                        tf = tf + adv_x * tb + adv_y * td

                    pending_data = None

                adjustment = item * adjustment_scale
                if is_vert:
                    te -= adjustment * tc
                    tf -= adjustment * td
                else:
                    te -= adjustment * ta
                    tf -= adjustment * tb
                continue

            if t is str:
                item_data = item.encode("latin-1")
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged

        if pending_data:
            n_data = len(pending_data)
            if n_data == 1:
                byte = pending_data[0]
                text = table[byte]
                total = widths[byte] + cs
                if byte == 32:
                    total += ws
            elif n_data == 2:
                b0 = pending_data[0]
                b1 = pending_data[1]
                text = table[b0] + table[b1]
                total = widths[b0] + widths[b1] + (2 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
            elif n_data == 3:
                b0 = pending_data[0]
                b1 = pending_data[1]
                b2 = pending_data[2]
                text = table[b0] + table[b1] + table[b2]
                total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
                if b2 == 32:
                    total += ws
            else:
                text = "".join(map(table.__getitem__, pending_data))
                total = 0.0
                space_count = 0
                for byte in pending_data:
                    total += widths[byte]
                    if byte == 32:
                        space_count += 1
                total += n_data * cs + space_count * ws
            if text:
                visible = self.is_text_visible(text)
                if is_vert:
                    adv_x, adv_y = 0.0, -total * text_scale
                else:
                    adv_x, adv_y = total * text_scale, 0.0
                E = te * ca + tf * cc + ce
                F = te * cb + tf * cd + cf
                c0_x = dr_C + E
                c0_y = dr_D + F
                c1_x = ar_C + E
                c1_y = ar_D + F
                adv_A = adv_x * A
                adv_B = adv_x * B
                c2_x = adv_A + c0_x
                c2_y = adv_B + c0_y
                c3_x = adv_A + c1_x
                c3_y = adv_B + c1_y

                x0 = c0_x if c0_x < c1_x else c1_x
                if c2_x < x0:
                    x0 = c2_x
                if c3_x < x0:
                    x0 = c3_x
                y0 = c0_y if c0_y < c1_y else c1_y
                if c2_y < y0:
                    y0 = c2_y
                if c3_y < y0:
                    y0 = c3_y
                x1 = c0_x if c0_x > c1_x else c1_x
                if c2_x > x1:
                    x1 = c2_x
                if c3_x > x1:
                    x1 = c3_x
                y1 = c0_y if c0_y > c1_y else c1_y
                if c2_y > y1:
                    y1 = c2_y
                if c3_y > y1:
                    y1 = c3_y

                new_run = self.alloc_run(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    tx=te,
                    ty=tf,
                    font_size=effective_font_size,
                    font_name=decoder.font_name or font_name,
                    space_width=effective_space_width,
                    order=seqno,
                    stream_order=stream_order,
                    xobject_depth=xobject_depth,
                    is_vertical=is_vert,
                    rotation_angle=rot,
                    visible=visible,
                    line_break_before=pending_line_break,
                    seqno=seqno,
                    fill_color=fill_color,
                )
                self.update_pending_run(new_run)
                seqno += 1
                pending_line_break = False
                te = te + adv_x * ta + adv_y * tc
                tf = tf + adv_x * tb + adv_y * td

        self.tm_e, self.tm_f = te, tf
        self.sequence = seqno
        self.pending_line_break = pending_line_break

    def append_tj_array_simple_horizontal_batched(
        self: Any, array: list[Any] | tuple[Any, ...], decoder: FontDecoder
    ) -> None:
        table = decoder.byte_decode_table
        assert table is not None

        pending_data: bytes | bytearray | None = None
        widths = self.font_widths or decoder.fast_widths
        text_scale = self.text_advance_scale
        cs = self.char_space_scale
        ws = self.word_space_scale

        fs = self.font_size
        rise = self.rise
        ascent = self.font_ascent
        descent = self.font_descent
        A, B, C, D = self.combined_A, self.combined_B, self.combined_C, self.combined_D
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        ta, tb = self.tm_a, self.tm_b
        te, tf = self.tm_e, self.tm_f
        scale_factor = hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_space_width = self.font_space_width * scale_factor
        merge_threshold = max(effective_space_width * 0.45, 2.0)
        stream_order = self.stream_order
        xobject_depth = self.xobject_depth
        font_name = self.current_font
        fill_color = self.fill_color
        seqno = self.sequence
        pending_line_break = self.pending_line_break

        ar = ascent + rise
        dr = descent + rise
        dr_C = dr * C
        dr_D = dr * D
        ar_C = ar * C
        ar_D = ar * D

        batch_parts: list[str] | None = None
        batch_last_char = ""
        batch_x0 = batch_y0 = batch_x1 = batch_y1 = 0.0
        batch_tx = batch_ty = 0.0
        batch_order = batch_seqno = 0
        batch_visible = True
        batch_line_break_before = False
        alloc_run = self.alloc_run
        update_pending_run = self.update_pending_run
        is_text_visible = self.is_text_visible
        no_space_before = NO_SPACE_BEFORE
        no_space_after = NO_SPACE_AFTER

        def flush_batch() -> None:
            nonlocal batch_parts, batch_last_char
            if batch_parts is None:
                return
            new_run = alloc_run(
                text="".join(batch_parts),
                x0=batch_x0,
                y0=batch_y0,
                x1=batch_x1,
                y1=batch_y1,
                tx=batch_tx,
                ty=batch_ty,
                font_size=effective_font_size,
                font_name=decoder.font_name or font_name,
                space_width=effective_space_width,
                order=batch_order,
                stream_order=stream_order,
                xobject_depth=xobject_depth,
                is_vertical=False,
                rotation_angle=0,
                visible=batch_visible,
                line_break_before=batch_line_break_before,
                seqno=batch_seqno,
                fill_color=fill_color,
            )
            update_pending_run(new_run)
            batch_parts = None
            batch_last_char = ""

        for item in array:
            t = type(item)
            if t is bytes:
                item_data = item
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue
            if t is PdfString:
                item_data = item.data
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue
            if t is str:
                item_data = item.encode("latin-1")
                if pending_data is None:
                    pending_data = item_data
                elif type(pending_data) is bytearray:
                    pending_data.extend(item_data)
                else:
                    merged = bytearray(pending_data)
                    merged.extend(item_data)
                    pending_data = merged
                continue

            if t is int or t is float:
                if pending_data:
                    n_data = len(pending_data)
                    if n_data == 1:
                        byte = pending_data[0]
                        text = table[byte]
                        total = widths[byte] + cs
                        if byte == 32:
                            total += ws
                    elif n_data == 2:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        text = table[b0] + table[b1]
                        total = widths[b0] + widths[b1] + (2 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                    elif n_data == 3:
                        b0 = pending_data[0]
                        b1 = pending_data[1]
                        b2 = pending_data[2]
                        text = table[b0] + table[b1] + table[b2]
                        total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                        if b0 == 32:
                            total += ws
                        if b1 == 32:
                            total += ws
                        if b2 == 32:
                            total += ws
                    else:
                        text = "".join(map(table.__getitem__, pending_data))
                        total = 0.0
                        space_count = 0
                        for byte in pending_data:
                            total += widths[byte]
                            if byte == 32:
                                space_count += 1
                        total += n_data * cs + space_count * ws

                    if text:
                        visible = self.is_text_visible(text)
                        adv_x = total * text_scale

                        E = te * ca + tf * cc + ce
                        F = te * cb + tf * cd + cf
                        c0_x = dr_C + E
                        c0_y = dr_D + F
                        c1_x = ar_C + E
                        c1_y = ar_D + F
                        adv_A = adv_x * A
                        adv_B = adv_x * B
                        c2_x = adv_A + c0_x
                        c2_y = adv_B + c0_y
                        c3_x = adv_A + c1_x
                        c3_y = adv_B + c1_y

                        x0 = c0_x if c0_x < c1_x else c1_x
                        if c2_x < x0:
                            x0 = c2_x
                        if c3_x < x0:
                            x0 = c3_x
                        y0 = c0_y if c0_y < c1_y else c1_y
                        if c2_y < y0:
                            y0 = c2_y
                        if c3_y < y0:
                            y0 = c3_y
                        x1 = c0_x if c0_x > c1_x else c1_x
                        if c2_x > x1:
                            x1 = c2_x
                        if c3_x > x1:
                            x1 = c3_x
                        y1 = c0_y if c0_y > c1_y else c1_y
                        if c2_y > y1:
                            y1 = c2_y
                        if c3_y > y1:
                            y1 = c3_y

                        if batch_parts is None:
                            batch_parts = [text]
                            batch_last_char = text[-1]
                            batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                            batch_tx, batch_ty = te, tf
                            batch_order = seqno
                            batch_seqno = seqno
                            batch_visible = visible
                            batch_line_break_before = pending_line_break
                        else:
                            gap = x0 - batch_x1
                            if (
                                visible == batch_visible
                                and not pending_line_break
                                and abs(batch_y0 - y0) <= effective_font_size * 0.5
                                and -2.0 <= gap < merge_threshold
                            ):
                                threshold = effective_space_width * 0.12
                                font_threshold = effective_font_size * 0.10
                                if font_threshold > threshold:
                                    threshold = font_threshold
                                if threshold < 1.0:
                                    threshold = 1.0
                                if (
                                    gap <= threshold
                                    or batch_last_char.isspace()
                                    or text[0].isspace()
                                    or text[0] in no_space_before
                                    or batch_last_char in no_space_after
                                ):
                                    separator = ""
                                else:
                                    separator = " "
                                if separator:
                                    batch_parts.append(separator)
                                batch_parts.append(text)
                                batch_last_char = text[-1]
                                if x1 > batch_x1:
                                    batch_x1 = x1
                            else:
                                flush_batch()
                                batch_parts = [text]
                                batch_last_char = text[-1]
                                batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                                batch_tx, batch_ty = te, tf
                                batch_order = seqno
                                batch_seqno = seqno
                                batch_visible = visible
                                batch_line_break_before = pending_line_break

                        seqno += 1
                        pending_line_break = False
                        te = te + adv_x * ta
                        tf = tf + adv_x * tb

                    pending_data = None

                adjustment = item * text_scale
                te -= adjustment * ta
                tf -= adjustment * tb

        if pending_data:
            n_data = len(pending_data)
            if n_data == 1:
                byte = pending_data[0]
                text = table[byte]
                total = widths[byte] + cs
                if byte == 32:
                    total += ws
            elif n_data == 2:
                b0 = pending_data[0]
                b1 = pending_data[1]
                text = table[b0] + table[b1]
                total = widths[b0] + widths[b1] + (2 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
            elif n_data == 3:
                b0 = pending_data[0]
                b1 = pending_data[1]
                b2 = pending_data[2]
                text = table[b0] + table[b1] + table[b2]
                total = widths[b0] + widths[b1] + widths[b2] + (3 * cs)
                if b0 == 32:
                    total += ws
                if b1 == 32:
                    total += ws
                if b2 == 32:
                    total += ws
            else:
                text = "".join(map(table.__getitem__, pending_data))
                total = 0.0
                space_count = 0
                for byte in pending_data:
                    total += widths[byte]
                    if byte == 32:
                        space_count += 1
                total += n_data * cs + space_count * ws

            if text:
                visible = is_text_visible(text)
                adv_x = total * text_scale
                E = te * ca + tf * cc + ce
                F = te * cb + tf * cd + cf
                c0_x = dr_C + E
                c0_y = dr_D + F
                c1_x = ar_C + E
                c1_y = ar_D + F
                adv_A = adv_x * A
                adv_B = adv_x * B
                c2_x = adv_A + c0_x
                c2_y = adv_B + c0_y
                c3_x = adv_A + c1_x
                c3_y = adv_B + c1_y

                x0 = c0_x if c0_x < c1_x else c1_x
                if c2_x < x0:
                    x0 = c2_x
                if c3_x < x0:
                    x0 = c3_x
                y0 = c0_y if c0_y < c1_y else c1_y
                if c2_y < y0:
                    y0 = c2_y
                if c3_y < y0:
                    y0 = c3_y
                x1 = c0_x if c0_x > c1_x else c1_x
                if c2_x > x1:
                    x1 = c2_x
                if c3_x > x1:
                    x1 = c3_x
                y1 = c0_y if c0_y > c1_y else c1_y
                if c2_y > y1:
                    y1 = c2_y
                if c3_y > y1:
                    y1 = c3_y

                if batch_parts is None:
                    batch_parts = [text]
                    batch_last_char = text[-1]
                    batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                    batch_tx, batch_ty = te, tf
                    batch_order = seqno
                    batch_seqno = seqno
                    batch_visible = visible
                    batch_line_break_before = pending_line_break
                else:
                    gap = x0 - batch_x1
                    if (
                        visible == batch_visible
                        and not pending_line_break
                        and abs(batch_y0 - y0) <= effective_font_size * 0.5
                        and -2.0 <= gap < merge_threshold
                    ):
                        threshold = effective_space_width * 0.12
                        font_threshold = effective_font_size * 0.10
                        if font_threshold > threshold:
                            threshold = font_threshold
                        if threshold < 1.0:
                            threshold = 1.0
                        if (
                            gap <= threshold
                            or batch_last_char.isspace()
                            or text[0].isspace()
                            or text[0] in no_space_before
                            or batch_last_char in no_space_after
                        ):
                            separator = ""
                        else:
                            separator = " "
                        if separator:
                            batch_parts.append(separator)
                        batch_parts.append(text)
                        batch_last_char = text[-1]
                        if x1 > batch_x1:
                            batch_x1 = x1
                    else:
                        flush_batch()
                        batch_parts = [text]
                        batch_last_char = text[-1]
                        batch_x0, batch_y0, batch_x1, batch_y1 = x0, y0, x1, y1
                        batch_tx, batch_ty = te, tf
                        batch_order = seqno
                        batch_seqno = seqno
                        batch_visible = visible
                        batch_line_break_before = pending_line_break

                seqno += 1
                pending_line_break = False
                te = te + adv_x * ta
                tf = tf + adv_x * tb

        flush_batch()
        self.tm_e, self.tm_f = te, tf
        self.sequence = seqno
        self.pending_line_break = pending_line_break

    def _append_tj_array_impl(self: Any, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            return
        if not array:
            return
        pending_bytes = bytearray()
        scale = self.text_advance_scale

        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        if (
            self.capture_runs
            and not self.capture_glyphs
            and self.current_actual_text_span() is None
            and decoder.byte_decode_table is not None
            and not decoder.is_cid_font
            and not decoder.is_vertical
            and not (decoder.is_type3 and self.capture_graphics)
            and decoder.to_unicode is None
            and decoder.cmap is None
        ):
            self.append_tj_array_simple(array, decoder)
            return
        is_vert = decoder.is_vertical
        zero_copy_flush = (
            not decoder.is_cid_font and decoder.to_unicode is None and decoder.cmap is None
        )

        te, tf = self.tm_e, self.tm_f
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        previous_compat_tj = (
            self.compat_tj_active,
            self.compat_tj_cursor_x,
            self.compat_tj_cursor_y,
            self.compat_tj_origin_e,
            self.compat_tj_origin_f,
            self.compat_tj_decoder,
            self.compat_tj_need_charspace,
        )
        self.compat_tj_active = self.capture_glyphs
        self.compat_tj_origin_e = self.lm_e
        self.compat_tj_origin_f = self.lm_f
        self.compat_tj_decoder = decoder
        self.compat_tj_need_charspace = False

        for item in array:
            t = type(item)
            if t is PdfString:
                pending_bytes.extend(item.data)
            elif t is bytes:
                pending_bytes.extend(item)
            elif t is int or t is float:
                if pending_bytes:
                    self.tm_e, self.tm_f = te, tf
                    if zero_copy_flush:
                        self._append_text_impl(data=memoryview(pending_bytes), decoder=decoder)
                    else:
                        self._append_text_impl(data=bytes(pending_bytes), decoder=decoder)
                    te, tf = self.tm_e, self.tm_f
                    pending_bytes.clear()
                adjustment = item * scale
                compat_adjustment = item * (0.001 * self.font_size * (self.horizontal_scale * 0.01))
                if is_vert:
                    te -= adjustment * tc
                    tf -= adjustment * td
                    self.compat_tj_cursor_y -= compat_adjustment
                else:
                    te -= adjustment * ta
                    tf -= adjustment * tb
                    self.compat_tj_cursor_x -= compat_adjustment
                self.compat_tj_need_charspace = True
            elif t is str:
                pending_bytes.extend(item.encode("latin-1"))

        if pending_bytes:
            self.tm_e, self.tm_f = te, tf
            if zero_copy_flush:
                self._append_text_impl(data=memoryview(pending_bytes), decoder=decoder)
            else:
                self._append_text_impl(data=bytes(pending_bytes), decoder=decoder)
            te, tf = self.tm_e, self.tm_f

        self.tm_e, self.tm_f = te, tf
        compat_cursor_x = self.compat_tj_cursor_x
        compat_cursor_y = self.compat_tj_cursor_y
        (
            self.compat_tj_active,
            self.compat_tj_cursor_x,
            self.compat_tj_cursor_y,
            self.compat_tj_origin_e,
            self.compat_tj_origin_f,
            self.compat_tj_decoder,
            self.compat_tj_need_charspace,
        ) = previous_compat_tj
        self.compat_tj_cursor_x = compat_cursor_x
        self.compat_tj_cursor_y = compat_cursor_y

    def current_actual_text_span(self: Any) -> Any | None:
        for entry in reversed(self.marked_content_stack):
            if getattr(entry, "actual_text", None) is not None:
                return entry
        return None

    def current_marked_content_mcid(self: Any) -> int | None:
        for entry in reversed(self.marked_content_stack):
            mcid = getattr(entry, "mcid", None)
            if type(mcid) is int:
                return mcid
        return None

    def emit_actual_text_span(self: Any, entry: Any) -> None:
        actual_text = getattr(entry, "actual_text", None)
        if actual_text is None or not getattr(entry, "has_text_extents", False):
            return
        if self.capture_runs:
            new_run = self.alloc_run(
                text=actual_text,
                x0=entry.x0,
                y0=entry.y0,
                x1=entry.x1,
                y1=entry.y1,
                tx=entry.tx,
                ty=entry.ty,
                font_size=entry.font_size,
                font_name=entry.font_name,
                space_width=entry.space_width,
                order=entry.order,
                stream_order=entry.stream_order,
                xobject_depth=entry.xobject_depth,
                is_vertical=entry.is_vertical,
                rotation_angle=entry.rotation_angle,
                visible=entry.visible,
                line_break_before=entry.line_break_before,
                seqno=entry.seqno,
                fill_color=entry.fill_color,
                advance_bbox=entry.advance_bbox,
                ink_bbox=entry.ink_bbox,
                baseline=entry.baseline,
                provenance=(*entry.provenance, ("unicode_source", "actual_text")),
                confidence=entry.confidence,
            )
            self.update_pending_run(new_run)
        if self.capture_glyphs and entry.advance_bbox is not None:
            self.glyphs.append_row(
                GlyphObservation(
                    text=actual_text,
                    ink_bbox=entry.ink_bbox or entry.advance_bbox,
                    advance_bbox=entry.advance_bbox,
                    seqno=entry.seqno,
                    font_name=entry.font_name,
                    font_size=entry.font_size,
                    baseline=entry.baseline,
                    rotation_angle=entry.rotation_angle,
                    fill=entry.fill_color,
                    visible=entry.visible,
                    confidence=entry.confidence,
                    unicode_source="actual_text",
                    # Preserve the text obtained from the font program as an
                    # alternate.  Consumers such as pdfminer do not interpret
                    # marked-content /ActualText and therefore need the font
                    # projection rather than the accessible replacement text.
                    alternates=(entry.compatibility_text,) if entry.compatibility_text else (),
                    font_decoder=entry.font_decoder,
                    effective_font_size=entry.font_size,
                    effective_font_height=entry.effective_font_height,
                    provenance=(
                        *entry.provenance,
                        ("compatibility_glyphs", tuple(entry.compatibility_glyphs)),
                    ),
                )
            )

    def internal_begin_text(self) -> None:
        if self.pending_run:
            self.flush_run()
        self.tm_a = self.lm_a = 1.0
        self.tm_b = self.lm_b = 0.0
        self.tm_c = self.lm_c = 0.0
        self.tm_d = self.lm_d = 1.0
        self.tm_e = self.lm_e = 0.0
        self.tm_f = self.lm_f = 0.0
        self.compat_tj_cursor_x = 0.0
        self.compat_tj_cursor_y = 0.0
        self.combined_A = self.ca
        self.combined_B = self.cb
        self.combined_C = self.cc
        self.combined_D = self.cd
        if self.ca == 1.0 and self.cb == 0.0 and self.cc == 0.0 and self.cd == 1.0:
            self.cached_rotation = 0
        else:
            self.cached_rotation = self.detect_rotation(self.ca, self.cb, self.cc, self.cd)
        self.invisible_text_layer = False

    def internal_end_text(self) -> None:
        pending = self.pending_run
        if pending:
            if self.capture_runs:
                pending.freeze_glyph_clusters()
                self.runs.append(pending)
            self.pending_run = None

    def internal_move_text(self, tx: float, ty: float) -> None:
        if self.pending_run:
            self.pending_run.freeze_glyph_clusters()
            self.runs.append(self.pending_run)
            self.pending_run = None
        # Preserve the specification's affine operation order. Exact layout
        # grouping can hinge on the final ULP at a character-margin boundary.
        self.tm_e = tx * self.lm_a + ty * self.lm_c + self.lm_e
        self.tm_f = tx * self.lm_b + ty * self.lm_d + self.lm_f
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f
        self.compat_tj_cursor_x = 0.0
        self.compat_tj_cursor_y = 0.0

    def internal_show_text(self, operand: ContentOperand) -> None:
        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        if type(operand) is PdfString:
            self.append_text(
                data=operand.data,
                decoder=decoder,
                string_syntax="literal" if operand.is_literal else "hex",
                compatibility_data=operand.compatibility_data,
            )
        else:
            self.append_text(operand, decoder=decoder)

    def op_noop(self, operands: OperandWindow, depth: int) -> None:
        return

    def op_BT(self, operands: OperandWindow, depth: int) -> None:
        self.text_object_id += 1
        self.internal_begin_text()

    def op_ET(self, operands: OperandWindow, depth: int) -> None:
        self.internal_end_text()

    def op_T_star(self, operands: OperandWindow, depth: int) -> None:
        self.internal_move_text(0.0, -self.leading)

    def op_Td(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) < 2:
            return
        try:
            tx = self.as_float(operands[0])
            ty = self.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.op_Td_values(tx, ty)

    def op_Td_values(self: Any, tx: float, ty: float) -> None:
        self.internal_move_text(tx, ty)

    def op_TD(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) < 2:
            return
        try:
            tx = self.as_float(operands[0])
            ty = self.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.op_TD_values(tx, ty)

    def op_TD_values(self: Any, tx: float, ty: float) -> None:
        self.leading = -ty
        self.internal_move_text(tx, ty)

    def op_Tj(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        # Operators consume their operands from the top of the operand stack.
        # A well-formed Tj has exactly one string, but damaged streams sometimes
        # leave older operands before it.  Those older values are not part of
        # the text-showing operation.
        self.internal_show_text(operands[-1])

    def op_TJ(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            self.append_tj_array(operands[0])

    def op_Tm(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) < 6:
            return
        try:
            a = self.as_float(operands[0])
            b = self.as_float(operands[1])
            c = self.as_float(operands[2])
            d_ = self.as_float(operands[3])
            e = self.as_float(operands[4])
            f = self.as_float(operands[5])
        except (TypeError, ValueError):
            return
        self.op_Tm_values(a, b, c, d_, e, f)

    def op_Tm_values(
        self: Any, a: float, b: float, c: float, d_: float, e: float, f: float
    ) -> None:
        self.flush_run()
        self.tm_a = self.lm_a = a
        self.tm_b = self.lm_b = b
        self.tm_c = self.lm_c = c
        self.tm_d = self.lm_d = d_
        self.tm_e = self.lm_e = e
        self.tm_f = self.lm_f = f
        self.compat_tj_cursor_x = 0.0
        self.compat_tj_cursor_y = 0.0
        self.update_combined()

    def op_Tf(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) < 2:
            return
        font_operand = operands[0]
        font_size_operand = operands[1]
        self.op_Tf_values(font_operand, font_size_operand)

    def op_Tf_values(self: Any, font_operand: Any, font_size_operand: Any) -> None:
        decoder_matches_resources = self.current_decoder_resources_id == self.resources_id
        if (
            self.current_decoder is not None
            and decoder_matches_resources
            and font_operand is self.font_operand
        ):
            if font_size_operand is not self.font_size_operand:
                value_type = type(font_size_operand)
                if value_type is float or value_type is int:
                    font_size = font_size_operand
                else:
                    try:
                        font_size = self.as_float(font_size_operand)
                    except (TypeError, ValueError):
                        return
                if self.font_size != font_size:
                    self.font_size = font_size
                    self.update_text_scales()
                    self.update_font_metrics()
                self.font_size_operand = font_size_operand
            return
        cache_key = (self.resources_id, font_operand, font_size_operand)
        cached = self.font_setting_cache.get(cache_key)
        if cached is None:
            font_name = self.document.resolver.resolve_name(font_operand)
            if font_name is None:
                return
            try:
                font_size = self.as_float(font_size_operand)
            except (TypeError, ValueError):
                return
            self.font_setting_cache[cache_key] = (font_name, font_size)
        else:
            font_name, font_size = cached
        if (
            self.current_font == font_name
            and self.current_decoder is not None
            and decoder_matches_resources
        ):
            if self.font_size != font_size:
                self.font_size = font_size
                self.update_text_scales()
                self.update_font_metrics()
            self.font_operand = font_operand
            self.font_size_operand = font_size_operand
            return
        self.current_font = font_name
        self.font_size = font_size
        self.update_text_scales()
        self.font_operand = font_operand
        self.font_size_operand = font_size_operand
        self.current_decoder = None
        self.current_decoder = self.get_decoder(update_metrics=False)
        self.update_font_metrics()

    def op_TL(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        try:
            self.leading = self.as_float(operands[0])
        except (TypeError, ValueError):
            return

    def op_Tc(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        try:
            char_space = self.as_float(operands[0])
        except (TypeError, ValueError):
            return
        self.op_Tc_values(char_space)

    def op_Tc_values(self: Any, char_space: float) -> None:
        self.char_space = char_space
        self.update_char_space_scale()

    def op_Tw(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        try:
            word_space = self.as_float(operands[0])
        except (TypeError, ValueError):
            return
        self.op_Tw_values(word_space)

    def op_Tw_values(self: Any, word_space: float) -> None:
        if self.word_space == word_space:
            return
        self.word_space = word_space
        self.update_word_space_scale()

    def op_Tr(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        try:
            self.render_mode = self.as_int(operands[0])
        except (TypeError, ValueError):
            return

    def op_Tz(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        try:
            self.horizontal_scale = self.as_float(operands[0])
        except (TypeError, ValueError):
            return
        self.update_horizontal_text_scale()

    def op_Ts(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        try:
            self.rise = self.as_float(operands[0])
        except (TypeError, ValueError):
            return

    def op_quote(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        self.internal_move_text(0.0, -self.leading)
        self.pending_line_break = True
        self.internal_show_text(operands[0])

    def op_double_quote(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) < 3:
            return
        try:
            word_space = self.as_float(operands[0])
            char_space = self.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.word_space = word_space
        self.char_space = char_space
        self.update_char_space_scale()
        self.update_word_space_scale()
        if not getattr(self.document, "legacy_pdfminer_text_operators", False):
            self.internal_move_text(0.0, -self.leading)
            self.pending_line_break = True
        self.internal_show_text(operands[2])

    def op_BI(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        operand = operands[0]
        # Duck-typed on purpose: the inline-image parser yields an InlineImage, but any
        # operand exposing ``dictionary`` is accepted here.
        if not hasattr(operand, "dictionary"):
            return
        image = cast("InlineImage", operand)
        if self.capture_images and self.is_graphics_visible():
            dictionary = dict(image.dictionary)
            data = getattr(image, "data", b"")
            source = ImageSource(
                data,
                dictionary,
                cache=getattr(self.document, "image_cache", None),
                cache_key=("inline", self.sequence),
            )
            self.inline_images.append(
                CapturedInlineImage(
                    seqno=self.sequence,
                    dictionary=dictionary,
                    data=data,
                    image_source=source,
                    image_clip=self.clip_bbox,
                    ctm=self.ctm,
                    xobject_depth=self.xobject_depth,
                )
            )
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    kind="inline-image",
                    image_source=source,
                    image_clip=self.clip_bbox,
                    dash_pattern=self.transformed_dash_pattern(),
                    items=[],
                    stream_order=self.stream_order,
                    xobject_depth=self.xobject_depth,
                )
            )

    def internal_image_source(
        self,
        key: object,
        raw: bytes | memoryview,
        dictionary: dict[Any, Any],
    ) -> ImageSource:
        return ImageSource(
            raw,
            dictionary,
            cache=getattr(self.document, "image_cache", None),
            cache_key=("xobject", key),
        )

    def op_BDC(self, operands: OperandWindow, depth: int) -> None:
        tag = self.document.resolver.resolve_name(operands[0]) if operands else None
        layer = (
            self.resolve_marked_content_layer(operands[1])
            if tag == "OC" and len(operands) >= 2
            else None
        )
        actual_text = (
            self.resolve_marked_content_actual_text(operands[1])
            if tag == "Span" and len(operands) >= 2
            else None
        )
        mcid = self.resolve_marked_content_mcid(operands[1]) if len(operands) >= 2 else None
        self.marked_content_stack.append(
            MarkedContentEntry(layer=layer, actual_text=actual_text, mcid=mcid)
        )

    def op_BMC(self, operands: OperandWindow, depth: int) -> None:
        self.marked_content_stack.append(MarkedContentEntry())

    def op_EMC(self, operands: OperandWindow, depth: int) -> None:
        if self.marked_content_stack:
            self.emit_actual_text_span(self.marked_content_stack.pop())

    def op_G(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            self.set_stroke_color(operands[0])

    def op_RG(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 3:
            self.set_stroke_color(operands[0], operands[1], operands[2])

    def op_RG_values(self: Any, red: int | float, green: int | float, blue: int | float) -> None:
        self.set_stroke_color(red, green, blue)

    def op_rg_values(self: Any, red: int | float, green: int | float, blue: int | float) -> None:
        self.set_fill_color(red, green, blue)

    def op_K(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 4:
            self.set_stroke_color(operands[0], operands[1], operands[2], operands[3])

    def op_w(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            try:
                self.line_width = max(0.0, self.as_float(operands[0]))
            except (TypeError, ValueError):
                return

    def op_w_value(self: Any, line_width: int | float) -> None:
        self.line_width = max(0.0, self.as_float(line_width))

    def op_J(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            try:
                self.line_cap = self.as_int(operands[0])
            except (TypeError, ValueError):
                return

    def op_J_value(self: Any, line_cap: int | float) -> None:
        self.line_cap = self.as_int(line_cap)

    def op_j(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            try:
                self.line_join = self.as_int(operands[0])
            except (TypeError, ValueError):
                return

    def op_j_value(self: Any, line_join: int | float) -> None:
        self.line_join = self.as_int(line_join)

    def op_M(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            try:
                self.miter_limit = max(1.0, self.as_float(operands[0]))
            except (TypeError, ValueError):
                return

    def op_M_value(self: Any, miter_limit: int | float) -> None:
        self.miter_limit = max(1.0, self.as_float(miter_limit))

    def op_d(self, operands: OperandWindow, depth: int) -> None:
        if not operands or len(operands) < 2:
            return
        try:
            phase = self.as_float(operands[1])
            array_obj = operands[0]
            dash_array = (
                [self.as_float(value) for value in array_obj]
                if isinstance(array_obj, (list, tuple))
                else []
            )
        except (TypeError, ValueError):
            return
        self.dash_pattern = (dash_array, phase)

    def op_m(self, operands: OperandWindow, depth: int) -> None:
        self._op_m_impl(operands, depth)

    def _op_m_impl(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 2:
            try:
                # op_m_values coerces via as_float; the except below covers non-numerics.
                self.op_m_values(cast("int | float", operands[0]), cast("int | float", operands[1]))
            except (TypeError, ValueError):
                return

    def op_m_values(self: Any, x: int | float, y: int | float) -> None:
        point = (self.as_float(x), self.as_float(y))
        if (
            self.capture_clipping
            or (self.capture_graphics or self.capture_glyphs)
            and self.is_graphics_visible()
        ):
            self.current_path.move_to(*point)
        self.current_point = point
        self.subpath_start = point

    def op_l(self, operands: OperandWindow, depth: int) -> None:
        self._op_l_impl(operands, depth)

    def _op_l_impl(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 2 and self.current_point is not None:
            try:
                # op_l_values coerces via as_float; the except below covers non-numerics.
                self.op_l_values(cast("int | float", operands[0]), cast("int | float", operands[1]))
            except (TypeError, ValueError):
                return

    def op_l_values(self: Any, x: int | float, y: int | float) -> None:
        if self.current_point is None:
            return
        point = (self.as_float(x), self.as_float(y))
        if (
            self.capture_clipping
            or (self.capture_graphics or self.capture_glyphs)
            and self.is_graphics_visible()
        ):
            self.current_path.line_to(*point)
        self.current_point = point

    def op_re(self, operands: OperandWindow, depth: int) -> None:
        self._op_re_impl(operands, depth)

    def _op_re_impl(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 4:
            try:
                x, y = self.as_float(operands[0]), self.as_float(operands[1])
                w, h = self.as_float(operands[2]), self.as_float(operands[3])
            except (TypeError, ValueError):
                return
            self.op_re_values(x, y, w, h)

    def op_re_values(
        self: Any, x: int | float, y: int | float, width: int | float, height: int | float
    ) -> None:
        x_float = float(x)
        y_float = float(y)
        if (
            self.capture_clipping
            or (self.capture_graphics or self.capture_glyphs)
            and self.is_graphics_visible()
        ):
            self.current_path.rect(x_float, y_float, float(width), float(height))
        self.current_point = (x_float, y_float)
        self.subpath_start = (x_float, y_float)

    def op_h(self, operands: OperandWindow, depth: int) -> None:
        self._op_h_impl(operands, depth)

    def _op_h_impl(self, operands: OperandWindow, depth: int) -> None:
        if self.current_point is not None and self.subpath_start is not None:
            if (
                self.capture_clipping
                or (self.capture_graphics or self.capture_glyphs)
                and self.is_graphics_visible()
            ):
                self.current_path.close()
            self.current_point = self.subpath_start

    def op_c(self, operands: OperandWindow, depth: int) -> None:
        self._op_c_impl(operands, depth)

    def _op_c_impl(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 6:
            try:
                x1 = self.as_float(operands[0])
                y1 = self.as_float(operands[1])
                x2 = self.as_float(operands[2])
                y2 = self.as_float(operands[3])
                x3 = self.as_float(operands[4])
                y3 = self.as_float(operands[5])
            except (TypeError, ValueError):
                return
            self.append_cubic_curve(x1, y1, x2, y2, x3, y3)

    def op_v(self, operands: OperandWindow, depth: int) -> None:
        self._op_v_impl(operands, depth)

    def _op_v_impl(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 4 and self.current_point is not None:
            x0, y0 = self.current_point
            try:
                x2 = self.as_float(operands[0])
                y2 = self.as_float(operands[1])
                x3 = self.as_float(operands[2])
                y3 = self.as_float(operands[3])
            except (TypeError, ValueError):
                return
            self.append_cubic_curve(x0, y0, x2, y2, x3, y3)

    def op_y(self, operands: OperandWindow, depth: int) -> None:
        self._op_y_impl(operands, depth)

    def _op_y_impl(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 4 and self.current_point is not None:
            try:
                x1 = self.as_float(operands[0])
                y1 = self.as_float(operands[1])
                x3 = self.as_float(operands[2])
                y3 = self.as_float(operands[3])
            except (TypeError, ValueError):
                return
            # `y` doubles the endpoint as the second control point, unlike `v`,
            # which uses the current point as the first one.
            self.append_cubic_curve(x1, y1, x3, y3, x3, y3)

    def internal_close_current_subpath(self: Any) -> None:
        if (
            self.capture_graphics
            and self.is_graphics_visible()
            and self.current_point is not None
            and self.subpath_start is not None
        ):
            self.current_path.close()

    def op_paint_stroke(self, operands: OperandWindow, depth: int) -> None:
        self.flush_drawing("stroke")
        self.current_point = None
        self.subpath_start = None

    def op_paint_close_stroke(self, operands: OperandWindow, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("stroke")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fill(self, operands: OperandWindow, depth: int) -> None:
        self.flush_drawing("fill", "nonzero")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fill_evenodd(self, operands: OperandWindow, depth: int) -> None:
        self.flush_drawing("fill", "evenodd")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fillstroke(self, operands: OperandWindow, depth: int) -> None:
        self.flush_drawing("fillstroke", "nonzero")
        self.current_point = None
        self.subpath_start = None

    def op_paint_fillstroke_evenodd(self, operands: OperandWindow, depth: int) -> None:
        self.flush_drawing("fillstroke", "evenodd")
        self.current_point = None
        self.subpath_start = None

    def op_paint_close_fillstroke(self, operands: OperandWindow, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("fillstroke", "nonzero")
        self.current_point = None
        self.subpath_start = None

    def op_paint_close_fillstroke_evenodd(self, operands: OperandWindow, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("fillstroke", "evenodd")
        self.current_point = None
        self.subpath_start = None

    def op_paint_clear(self, operands: OperandWindow, depth: int) -> None:
        self.current_path.clear()
        self.current_point = None
        self.subpath_start = None

    def internal_emit_clip_scope_push(self: Any) -> None:
        if not self.clip_scope_stack or self.clip_scope_stack[-1]:
            return
        self.clip_scope_stack[-1] = True
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=None,
                fill_opacity=None,
                kind="state-push",
            )
        )

    def op_W(self, operands: OperandWindow, depth: int) -> None:
        self.internal_record_clip("nonzero")

    def internal_record_clip(self: Any, fill_rule: str) -> None:
        path = self.current_path.transformed(self.ctm)
        if not path.has_segments():
            return
        clip_bbox = path.bbox()
        if clip_bbox is not None:
            if self.clip_bbox is None:
                self.clip_bbox = clip_bbox
            else:
                x0 = max(self.clip_bbox[0], clip_bbox[0])
                y0 = max(self.clip_bbox[1], clip_bbox[1])
                x1 = min(self.clip_bbox[2], clip_bbox[2])
                y1 = min(self.clip_bbox[3], clip_bbox[3])
                self.clip_bbox = (x0, y0, x1, y1)
        if self.capture_graphics and self.is_graphics_visible():
            self.internal_emit_clip_scope_push()
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    line_width=0.0,
                    line_cap=self.line_cap,
                    line_join=self.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    kind="clip",
                    path=path,
                )
            )

    def op_W_star(self, operands: OperandWindow, depth: int) -> None:
        self.internal_record_clip("evenodd")

    def normalize_colors(self: Any, *components: Any) -> tuple[float, ...] | None:
        cache_key = components
        cached = self.color_normalization_cache.get(cache_key, MISSING)
        if cached is not MISSING:
            return cached
        values: list[float] = []
        for component in components:
            try:
                values.append(max(0.0, min(1.0, self.as_float(component))))
            except ValueError:
                self.color_normalization_cache[cache_key] = None
                return None
        if not values:
            self.color_normalization_cache[cache_key] = None
            return None
        normalized = tuple(values)
        self.color_normalization_cache[cache_key] = normalized
        return normalized

    def set_stroke_color(self: Any, *components: Any) -> None:
        if self.type3_uncolored:
            return
        normalized = self.normalize_colors(*components)
        if normalized is not None:
            self.stroke_color = normalized
            self.stroke_pattern = None

    def set_fill_color(self: Any, *components: Any) -> None:
        if self.type3_uncolored:
            return
        normalized = self.normalize_colors(*components)
        if normalized is not None:
            self.fill_color = normalized
            self.fill_pattern = None

    # `o` is an OperandWindow on the hot path and a plain list elsewhere, holding
    # raw PDF operands; `Any` matches the deliberate looseness of `self: Any` here.
    def normalize_color_operands(self: Any, o: Any) -> tuple[float, ...] | None:
        count = len(o)
        if count == 1:
            c0 = o[0]
            t0 = type(c0)
            if t0 is float or t0 is int:
                if c0 < 0.0:
                    return (0.0,)
                if c0 > 1.0:
                    return (1.0,)
                return (float(c0),)
            return self.normalize_colors(c0)
        if count == 3:
            c0 = o[0]
            c1 = o[1]
            c2 = o[2]
            t0 = type(c0)
            t1 = type(c1)
            t2 = type(c2)
            if (
                (t0 is float or t0 is int)
                and (t1 is float or t1 is int)
                and (t2 is float or t2 is int)
            ):
                return (
                    max(0.0, min(1.0, float(c0))),
                    max(0.0, min(1.0, float(c1))),
                    max(0.0, min(1.0, float(c2))),
                )
            return self.normalize_colors(c0, c1, c2)
        if count == 4:
            c0 = o[0]
            c1 = o[1]
            c2 = o[2]
            c3 = o[3]
            t0 = type(c0)
            t1 = type(c1)
            t2 = type(c2)
            t3 = type(c3)
            if (
                (t0 is float or t0 is int)
                and (t1 is float or t1 is int)
                and (t2 is float or t2 is int)
                and (t3 is float or t3 is int)
            ):
                return (
                    max(0.0, min(1.0, float(c0))),
                    max(0.0, min(1.0, float(c1))),
                    max(0.0, min(1.0, float(c2))),
                    max(0.0, min(1.0, float(c3))),
                )
            return self.normalize_colors(c0, c1, c2, c3)
        return self.normalize_colors(*o)

    def resolve_color_space(
        self: Any, name_obj: Any, *, default_fallback: bool = False
    ) -> str | None:
        try:
            cache_key = (self.resources_id, name_obj, default_fallback)
            cached = self.color_space_cache.get(cache_key, MISSING)
            if cached is not MISSING:
                return cached
        except TypeError:
            cache_key = None

        name = self.document.resolver.resolve_name(name_obj)
        if name is None:
            resolved_color_space = "DeviceGray" if default_fallback else None
            if cache_key is not None:
                self.color_space_cache[cache_key] = resolved_color_space
            return resolved_color_space

        color_space: object = self.lookup_page_resource("ColorSpace", name)
        if color_space is None:
            resolved = name if default_fallback else None
            if cache_key is not None:
                self.color_space_cache[cache_key] = resolved
            return resolved

        color_space_name = normalize_pdf_name(color_space)
        if color_space_name is not None:
            if cache_key is not None:
                self.color_space_cache[cache_key] = color_space_name
            return color_space_name
        if isinstance(color_space, (list, tuple)) and color_space:
            base = color_space[0]
            base_name = normalize_pdf_name(base)
            if base_name is not None:
                if cache_key is not None:
                    self.color_space_cache[cache_key] = base_name
                return base_name

        if isinstance(name, str) and not name.startswith("/"):
            if cache_key is not None:
                self.color_space_cache[cache_key] = name
            return name

        resolved = name if default_fallback else None
        if cache_key is not None:
            self.color_space_cache[cache_key] = resolved
        return resolved

    def resolve_pattern_color(
        self: Any, operands: tuple[Any, ...] | OperandWindow
    ) -> PdfDict | None:
        if not operands:
            return None
        pattern_name = self.document.resolver.resolve_name(operands[-1])
        if not pattern_name:
            return None
        pattern = self.lookup_page_resource("Pattern", pattern_name)
        if isinstance(pattern, PdfStream):
            pattern_dict = self.document.resolver.resolve_dict(pattern.dictionary)
        else:
            pattern_dict = (
                self.document.resolver.resolve_dict(pattern) if pattern is not None else None
            )
        if not isinstance(pattern_dict, dict):
            return None
        pattern_type = parse_int(lookup_dict_key(pattern_dict, "PatternType"), None)
        if pattern_type == 2:
            shading = lookup_dict_key(pattern_dict, "Shading")
            shading = self.document.resolver.resolve(shading)
            shading_dict = (
                self.document.resolver.resolve_dict(shading) if shading is not None else None
            )
            if not isinstance(shading_dict, dict):
                return None
            return cast(PdfDict, {"kind": "shading", "dictionary": dict(shading_dict)})
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return None
        paint_type = parse_int(lookup_dict_key(pattern_dict, "PaintType"), 1)
        if paint_type not in {1, 2}:
            return None
        base_color = None
        if paint_type == 2:
            base_color = self.normalize_color_operands(operands[:-1])
            if base_color is None:
                return None
        bbox = self.document.resolver.resolve_box(lookup_dict_key(pattern_dict, "BBox"))
        if bbox is None:
            return None
        x_step = self.document.resolver.resolve_float(
            lookup_dict_key(pattern_dict, "XStep"), default=None
        )
        y_step = self.document.resolver.resolve_float(
            lookup_dict_key(pattern_dict, "YStep"), default=None
        )
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            return None
        try:
            matrix = Matrix.from_operand(lookup_dict_key(pattern_dict, "Matrix"))
        except ValueError:
            matrix = IDENTITY_MATRIX
        resources = (
            self.document.resolver.resolve_dict(lookup_dict_key(pattern_dict, "Resources")) or {}
        )
        nested_state = type(self)(
            self.document,
            self.page,
            hidden_layers=self.hidden_layers,
            decoder_cache=self.decoder_cache,
        )
        try:
            nested_state.consume_stream(pattern, resources, matrix, 0)
        except Exception:
            return None
        return cast(
            PdfDict,
            {
                "kind": "tiling",
                "bbox": tuple(bbox),
                "x_step": float(x_step),
                "y_step": float(y_step),
                "drawings": [
                    {
                        "kind": drawing.kind,
                        "fill": (
                            base_color if drawing.kind in {"fill", "fillstroke"} else drawing.fill
                        ),
                        "fill_pattern": drawing.fill_pattern,
                        "fill_opacity": drawing.fill_opacity,
                        "stroke_color": (
                            base_color
                            if drawing.kind in {"stroke", "fillstroke"}
                            else drawing.stroke_color
                        ),
                        "stroke_pattern": drawing.stroke_pattern,
                        "stroke_opacity": drawing.stroke_opacity,
                        "line_width": drawing.line_width,
                        "line_cap": drawing.line_cap,
                        "line_join": drawing.line_join,
                        "dash_pattern": drawing.dash_pattern,
                        "fill_rule": drawing.fill_rule,
                        "blend_mode": drawing.blend_mode,
                        "soft_mask_alpha": drawing.soft_mask_alpha,
                        "raw_data": drawing.raw_data,
                        "dictionary": drawing.dictionary,
                        "items": list(drawing.items),
                        "path": drawing.path,
                        "rect": drawing.rect,
                    }
                    for drawing in nested_state.drawings
                ],
                "runs": [
                    {
                        "text": run.text,
                        "bbox": (run.x0, run.y0, run.x1, run.y1),
                        "fill_color": run.fill_color,
                        "visible": run.visible,
                    }
                    for run in nested_state.runs
                ],
                "glyphs": [
                    {
                        "text": glyph.text,
                        "bbox": glyph.ink_bbox,
                        "advance_bbox": glyph.advance_bbox,
                        "fill_color": glyph.fill,
                        "visible": glyph.visible,
                        "code": glyph.cid,
                        "gid": glyph.gid,
                        "font_name": glyph.font_name,
                        "unicode_source": glyph.unicode_source,
                        "alternates": glyph.alternates,
                        "bitmap": glyph.resolved_bitmap(),
                        "bitmap_width": glyph.bitmap_width,
                        "bitmap_height": glyph.bitmap_height,
                    }
                    for glyph in nested_state.glyphs
                    if glyph.has_paint
                ],
            },
        )

    def op_CS(self, operands: OperandWindow, depth: int) -> None:
        self._op_CS_impl(operands, depth)

    def internal_set_color_space(self, operands: OperandWindow, *, stroke: bool) -> None:
        if operands:
            name_obj = operands[0]
            try:
                cached = self.color_space_cache.get((self.resources_id, name_obj, True), MISSING)
            except TypeError:
                cached = MISSING
            if cached is not MISSING:
                if cached is not None:
                    if stroke and self.stroke_color_space != cached:
                        self.stroke_color_space = cast("str", cached)
                    elif not stroke and self.fill_color_space != cached:
                        self.fill_color_space = cast("str", cached)
                return
            color_space = self.resolve_color_space(name_obj, default_fallback=True)
            if color_space is not None:
                if stroke:
                    self.stroke_color_space = color_space
                else:
                    self.fill_color_space = color_space

    def _op_CS_impl(self, operands: OperandWindow, depth: int) -> None:
        self.internal_set_color_space(operands, stroke=True)

    def op_cs(self, operands: OperandWindow, depth: int) -> None:
        self._op_cs_impl(operands, depth)

    def _op_cs_impl(self, operands: OperandWindow, depth: int) -> None:
        self.internal_set_color_space(operands, stroke=False)

    def op_SC(self, operands: OperandWindow, depth: int) -> None:
        self._op_SC_impl(operands, depth)

    def _op_SC_impl(self, operands: OperandWindow, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=False)

    def op_SCN(self, operands: OperandWindow, depth: int) -> None:
        self._op_SCN_impl(operands, depth)

    def internal_set_color(
        self, operands: OperandWindow, *, stroke: bool, allow_pattern: bool
    ) -> None:
        if self.type3_uncolored:
            return
        color_space = self.stroke_color_space if stroke else self.fill_color_space
        if allow_pattern and color_space == "Pattern":
            pattern = self.resolve_pattern_color(operands)
            if stroke:
                self.stroke_pattern = pattern
            else:
                self.fill_pattern = pattern
            if len(operands) > 1:
                normalized = self.normalize_color_operands(operands[:-1])
                if normalized is not None:
                    if stroke:
                        self.stroke_color = normalized
                    else:
                        self.fill_color = normalized
            return
        normalized = self.normalize_color_operands(operands)
        if normalized is not None:
            if stroke:
                self.stroke_color = normalized
                self.stroke_pattern = None
            else:
                self.fill_color = normalized
                self.fill_pattern = None

    def _op_SCN_impl(self, operands: OperandWindow, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=True)

    def op_sc(self, operands: OperandWindow, depth: int) -> None:
        self._op_sc_impl(operands, depth)

    def _op_sc_impl(self, operands: OperandWindow, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=False)

    def op_scN(self, operands: OperandWindow, depth: int) -> None:
        self._op_scN_impl(operands, depth)

    def _op_scN_impl(self, operands: OperandWindow, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=True)

    def op_i(self, operands: OperandWindow, depth: int) -> None:
        self._op_i_impl(operands, depth)

    def _op_i_impl(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            try:
                value = self.as_float(operands[0])
            except ValueError:
                return
            self.flatness = max(0, min(100, int(value)))

    def op_ri(self, operands: OperandWindow, depth: int) -> None:
        self._op_ri_impl(operands, depth)

    def _op_ri_impl(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        value = self.document.resolver.resolve_name_like_value(operands[0])
        if isinstance(value, str):
            self.render_intent = value

    def op_MP(self, operands: OperandWindow, depth: int) -> None:
        # A marked-content point is not a scope. Only BMC/BDC push and EMC pops.
        return

    def op_DP(self, operands: OperandWindow, depth: int) -> None:
        # A property-bearing marked-content point likewise has no lasting state.
        return

    def resolve_marked_content_actual_text(self: Any, value: Any) -> str | None:
        props = self.resolve_marked_content_properties(value)
        if not isinstance(props, dict):
            return None
        return self.document.resolver.resolve_str(lookup_dict_key(props, "ActualText"))

    def resolve_marked_content_mcid(self: Any, value: Any) -> int | None:
        props = self.resolve_marked_content_properties(value)
        if not isinstance(props, dict):
            return None
        return self.document.resolver.resolve_int(lookup_dict_key(props, "MCID"))

    def resolve_marked_content_properties(self: Any, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            return resolved
        name = self.document.resolver.resolve_name(value)
        if not name:
            return None
        props = self.lookup_page_resource("Properties", name)
        return props if isinstance(props, dict) else None

    def resolve_marked_content_layer(self: Any, value: Any) -> str | None:
        if value is None:
            return None

        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            oc = lookup_dict_key(resolved, "OC")
            if oc is not None:
                name = self.document.resolver.resolve_name(oc)
                return name or self.document.resolver.resolve_str(oc)

        return self.document.resolver.resolve_name(value) or self.document.resolver.resolve_str(
            value
        )

    def op_BX(self, operands: OperandWindow, depth: int) -> None:
        self.compatibility_depth += 1

    def op_EX(self, operands: OperandWindow, depth: int) -> None:
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def op_d0(self, operands: OperandWindow, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: OperandWindow, depth: int) -> None:
        self.type3_uncolored = True

    def op_sh(self, operands: OperandWindow, depth: int) -> None:
        if not operands or not self.capture_graphics or not self.is_graphics_visible():
            return
        shading_ref = self.document.resolver.resolve_name(operands[0])
        if not shading_ref:
            return
        shading = self.lookup_page_resource("Shading", shading_ref)
        if not isinstance(shading, dict):
            return
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.fill_color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.stroke_color,
                stroke_opacity=self.stroke_opacity,
                line_width=self.line_width,
                line_cap=self.line_cap,
                line_join=self.line_join,
                dash_pattern=self.transformed_dash_pattern(),
                blend_mode=self.blend_mode,
                soft_mask_alpha=self.group_alpha,
                kind="shading",
                items=[],
                dictionary=dict(shading),
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
            )
        )

    @staticmethod
    def as_float(value: Any) -> float:
        value_type = type(value)
        if value_type is float:
            return value
        if value_type is int:
            return float(value)
        if value_type is bool:
            raise ValueError("invalid numeric operand")
        parsed = parse_float(value, None)
        if parsed is None:
            raise ValueError("invalid numeric operand")
        return parsed

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is bool:
            raise ValueError("invalid numeric operand")
        parsed = parse_int(value, None)
        if parsed is None:
            raise ValueError("invalid numeric operand")
        return parsed

    def resolve_extgstate(self: Any, name: str) -> dict[str, Any] | None:
        cache_key = (self.resources_id, name)
        cached = self.extgstate_cache.get(cache_key, MISSING)
        if cached is not MISSING:
            return cached
        resolved = self.lookup_page_resource("ExtGState", name)
        if not isinstance(resolved, dict):
            self.extgstate_cache[cache_key] = None
            return None
        self.extgstate_cache[cache_key] = resolved
        return resolved

    def op_q(self, operands: OperandWindow, depth: int) -> None:
        self.clip_scope_stack.append(False)
        self.stack.append(
            (
                self.ca,
                self.cb,
                self.cc,
                self.cd,
                self.ce,
                self.cf,
                self.fill_color,
                self.fill_pattern,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_pattern,
                self.stroke_opacity,
                self.fill_color_space,
                self.stroke_color_space,
                self.compatibility_depth,
                self.blend_mode,
                self.group_alpha,
                self.flatness,
                self.render_intent,
                self.clip_bbox,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                self.dash_pattern,
                self.font_size,
                self.font_operand,
                self.font_size_operand,
                self.horizontal_scale,
                self.char_space,
                self.word_space,
                self.rise,
                self.leading,
                self.render_mode,
                self.current_font,
                self.current_decoder,
            )
        )

    def op_Q(self, operands: OperandWindow, depth: int) -> None:
        clip_scope_emitted = self.clip_scope_stack.pop() if self.clip_scope_stack else False
        if self.capture_graphics and clip_scope_emitted:
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    kind="state-pop",
                )
            )
        if not self.stack:
            return
        (
            self.ca,
            self.cb,
            self.cc,
            self.cd,
            self.ce,
            self.cf,
            self.fill_color,
            self.fill_pattern,
            self.fill_opacity,
            self.stroke_color,
            self.stroke_pattern,
            self.stroke_opacity,
            self.fill_color_space,
            self.stroke_color_space,
            self.compatibility_depth,
            self.blend_mode,
            self.group_alpha,
            self.flatness,
            self.render_intent,
            self.clip_bbox,
            self.line_width,
            self.line_cap,
            self.line_join,
            self.miter_limit,
            self.dash_pattern,
            self.font_size,
            self.font_operand,
            self.font_size_operand,
            self.horizontal_scale,
            self.char_space,
            self.word_space,
            self.rise,
            self.leading,
            self.render_mode,
            self.current_font,
            self.current_decoder,
        ) = self.stack.pop()
        self.update_combined()
        # Text-state values are included in our graphics-state snapshot for
        # compatibility with malformed producers that change them inside q/Q.
        # Restore their derived scales and metrics together with the raw values.
        self.update_text_scales()
        self.update_font_metrics()

    def op_cm(self, operands: OperandWindow, depth: int) -> None:
        if not operands or len(operands) < 6:
            return
        try:
            m_a, m_b, m_c, m_d, m_e, m_f = (
                self.as_float(operands[0]),
                self.as_float(operands[1]),
                self.as_float(operands[2]),
                self.as_float(operands[3]),
                self.as_float(operands[4]),
                self.as_float(operands[5]),
            )
        except (TypeError, ValueError):
            return

        self.op_cm_values(m_a, m_b, m_c, m_d, m_e, m_f)

    def op_cm_values(
        self: Any,
        m_a: float,
        m_b: float,
        m_c: float,
        m_d: float,
        m_e: float,
        m_f: float,
    ) -> None:
        ca, cb, cc, cd, ce, cf = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        self.ca = m_a * ca + m_b * cc
        self.cb = m_a * cb + m_b * cd
        self.cc = m_c * ca + m_d * cc
        self.cd = m_c * cb + m_d * cd
        self.ce = m_e * ca + m_f * cc + ce
        self.cf = m_e * cb + m_f * cd + cf
        self.update_combined()

    def op_g(self, operands: OperandWindow, depth: int) -> None:
        if operands:
            self.set_fill_color(operands[0])

    def op_rg(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 3:
            self.set_fill_color(operands[0], operands[1], operands[2])

    def op_k(self, operands: OperandWindow, depth: int) -> None:
        if len(operands) >= 4:
            self.set_fill_color(operands[0], operands[1], operands[2], operands[3])

    def op_gs(self, operands: OperandWindow, depth: int) -> None:
        if not operands:
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            return
        try:
            fill_opacity = lookup_dict_key(extgstate, "ca")
            if fill_opacity is not None:
                self.fill_opacity = max(0.0, min(1.0, self.as_float(fill_opacity)))
            stroke_opacity = lookup_dict_key(extgstate, "CA")
            if stroke_opacity is not None:
                self.stroke_opacity = max(0.0, min(1.0, self.as_float(stroke_opacity)))
            blend_mode = lookup_dict_key(extgstate, "BM")
            if blend_mode is not None:
                if isinstance(blend_mode, (list, tuple)):
                    blend_mode = blend_mode[0] if blend_mode else None
                if blend_mode is not None:
                    self.blend_mode = self.document.resolver.resolve_name_like_value(blend_mode)
        except (TypeError, ValueError):
            return
