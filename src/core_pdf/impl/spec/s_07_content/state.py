# SPDX-License-Identifier: AGPL-3.0-only
"""Content-stream interpreter state.

Holds the graphics and text state, the operator handlers, and glyph emission.
"""

from __future__ import annotations

import typing
from math import ceil, hypot
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.inline_images import InlineImage

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.model.geometry import RectBox, transform_bbox
from core_pdf.impl.model.glyphs import (
    GlyphCluster,
    GlyphObservation,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
)
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.primitives import (
    PdfName,
    PdfReference,
    PdfString,
)
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
    CapturedPath,
    PatternPaint,
    PdfminerCursor,
    RunGeometry,
    ShadingPattern,
    TilingPattern,
    glyph_bitmap_dimensions,
    glyph_ink_rect,
    glyph_text_space_boxes,
    should_capture_glyph_bitmap,
    should_capture_suspicious_multi_glyph_bitmap,
    transformed_text_line,
    transformed_text_rect,
    type3_font_matrix,
    type3_glyph_names,
)
from core_pdf.impl.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.spec.s_07_content.operations import (
    ContentOperand,
    ContentOperands,
    NestedStreamRequest,
    OperationHandler,
    dispatch_operations,
)
from core_pdf.impl.spec.s_07_content.operator_tables import build_operator_handlers
from core_pdf.impl.spec.s_07_content.stream_state import (
    STREAM_STATE_MIRRORED,
    ContentStreamFrame,
    LayoutFormId,
    StreamKey,
    StreamState,
)
from core_pdf.impl.spec.s_07_content.text_helpers import (
    can_merge_cross_font_word,
    gap_separator,
    is_garbage_text,
    normalize_extracted_text,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_float_strict,
    parse_int,
    parse_int_strict,
)
from core_pdf.impl.spec.s_08_graphics.color import color_operands_to_srgb
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec, color_spec_from_value
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource, SoftMask
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.spec.s_09_fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.spec.s_09_fonts.ligatures import detect_ligature_overrides
from core_pdf.impl.types import Rectangle

DecodedGlyphs: TypeAlias = tuple[DecodedGlyph, ...] | None


class internal_GraphicsStateSnapshot(typing.NamedTuple):
    """Named, immutable state saved by ``q`` and restored by ``Q``.

    This remains a tuple so restoring it can use the fast tuple-unpack path, but
    construction names every field. Adding or reordering state can therefore no
    longer silently shift a run of same-typed positional values.
    """

    ca: float
    cb: float
    cc: float
    cd: float
    ce: float
    cf: float
    fill_color: tuple[float, ...] | None
    fill_pattern: PatternPaint | None
    fill_opacity: float
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PatternPaint | None
    stroke_opacity: float
    fill_color_space: str
    stroke_color_space: str
    fill_color_spec: ImageColorSpec | None
    stroke_color_spec: ImageColorSpec | None
    compatibility_depth: int
    blend_mode: str | None
    group_alpha: float | None
    flatness: int
    render_intent: str | None
    clip_bbox: Rectangle | None
    line_width: float
    line_cap: int
    line_join: int
    miter_limit: float
    dash_pattern: tuple[list[float], float]
    font_size: float
    font_operand: object
    font_size_operand: object
    horizontal_scale: float
    char_space: float
    word_space: float
    rise: float
    leading: float
    render_mode: int
    current_font: str | None
    current_decoder: FontDecoder | None


MATRIX_TOLERANCE = 0.1


def detect_rotation_from_linear(
    A: float, B: float, C: float, D: float, tolerance: float = MATRIX_TOLERANCE
) -> int:
    scale_x = hypot(A, B)
    scale_y = hypot(C, D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    if (
        abs(na - 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd - 1.0) < tolerance
    ):
        return 0
    if (
        abs(na) < tolerance
        and abs(nb - 1.0) < tolerance
        and abs(nc + 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 90
    if (
        abs(na + 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd + 1.0) < tolerance
    ):
        return 180
    if (
        abs(na) < tolerance
        and abs(nb + 1.0) < tolerance
        and abs(nc - 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 270
    return 0


class TextDocument(typing.Protocol):
    @property
    def resolver(self) -> PdfValueResolver: ...

    raster_font_provider: Any
    legacy_pdfminer_text_operators: bool

    # TextState never calls this itself, but `detect_ligature_overrides` takes a
    # FontResourceDocument, and a document only satisfies that protocol with it.
    def resolve(self, value: object, /) -> object: ...


# ISO 32000-1 Table 106: mode 3 is "neither fill nor stroke text (invisible)"
# and mode 7 is "add text to path for clipping" -- neither adds marks to the
# page. render/page.py already used this pair; extraction checked only mode 3.
internal_NON_PAINTING_RENDER_MODES = frozenset({3, 7})


class TextState:
    document: TextDocument
    page: PdfDict
    capture_glyphs: bool
    capture_graphics: bool
    compat_tj_decoder: FontDecoder | None
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_clusters: list[GlyphCluster]
    lines: list[CapturedLine]
    drawings: list[CapturedDrawing]
    current_path: CapturedPath
    current_point: tuple[float, float] | None
    subpath_start: tuple[float, float] | None
    stack: list[internal_GraphicsStateSnapshot]
    clip_scope_stack: list[bool]
    fill_color: tuple[float, ...] | None
    fill_pattern: PatternPaint | None
    # Always a real number: __init__ seeds 1.0, `gs` clamps into [0, 1], and a
    # restore only ever copies a value that came from here.
    fill_opacity: float
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PatternPaint | None
    stroke_opacity: float
    blend_mode: str | None
    group_alpha: float | None
    flatness: int
    render_intent: str | None
    clip_bbox: Rectangle | None
    layout_form_bbox: Rectangle | None
    layout_form_id: LayoutFormId
    fill_color_space: str
    # The resolved space behind that name. The name alone cannot distinguish two
    # Separation resources with different tint transforms, and 8.6.6.3/8.6.6.4
    # need the palette or the tint transform to turn `sc`/`scn` operands into a
    # colour, so it travels with the name through q/Q.
    fill_color_spec: ImageColorSpec | None
    stroke_color_space: str
    stroke_color_spec: ImageColorSpec | None
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
    pending_run: TextRun | None
    op_handlers: dict[str, OperationHandler]

    __slots__ = (
        "document",
        "page",
        "capture_glyphs",
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
        "fill_color_spec",
        "stroke_color_spec",
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
        "pending_line_break",
        "pending_run",
        "compat_tj_active",
        "compat_tj_enabled",
        "compat_tj_cursor_x",
        "compat_tj_cursor_y",
        "compat_tj_origin_e",
        "compat_tj_origin_f",
        "compat_tj_decoder",
        "compat_tj_need_charspace",
        "op_handlers",
        "combined_A",
        "combined_B",
        "combined_C",
        "combined_D",
        "inline_images",
    )

    def __init__(
        self,
        document: TextDocument,
        page: PdfDict,
        hidden_layers: frozenset[str] = frozenset(),
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
        self.fill_color_spec = None
        self.stroke_color_spec = None
        self.stroke_color_space = "DeviceGray"
        self.line_width = 1.0
        self.line_cap = 0
        self.line_join = 0
        self.miter_limit = 10.0
        self.dash_pattern = ([], 0.0)
        self.stack = []
        self.clip_scope_stack = []
        self.capture_glyphs = True
        self.capture_graphics = True
        # The pdfminer cursor exists only so the pdfminer and pdfplumber
        # facades can reproduce that library's per-glyph advance bookkeeping.
        # Both open the document with legacy_pdfminer_text_operators, and they
        # are the only readers of the provenance it emits, so native extraction
        # does not need to pay for it on every glyph of every page.
        self.compat_tj_enabled = document.legacy_pdfminer_text_operators
        self.capture_clipping = True
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
        self.pending_line_break = False
        self.pending_run = None
        self.compat_tj_active = False
        self.compat_tj_cursor_x = 0.0
        self.compat_tj_cursor_y = 0.0
        self.compat_tj_origin_e = 0.0
        self.compat_tj_origin_f = 0.0
        self.compat_tj_decoder = None
        self.compat_tj_need_charspace = False
        self.op_handlers = build_operator_handlers(self)

        self.combined_A = 1.0
        self.combined_B = 0.0
        self.combined_C = 0.0
        self.combined_D = 1.0
        self.inline_images: list[CapturedInlineImage] = []

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
        self.compat_tj_active = self.capture_glyphs and self.compat_tj_enabled
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
        combined = self.text_matrix.multiply(self.ctm)
        self.combined_A = combined.a
        self.combined_B = combined.b
        self.combined_C = combined.c
        self.combined_D = combined.d

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
        draw_path = self.internal_records_path()
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

    def capture_stream_state(self) -> StreamState:
        return StreamState(
            graphics_stack_len=len(self.stack),
            marked_content_stack_len=len(self.marked_content_stack),
            **{name: getattr(self, name) for name in STREAM_STATE_MIRRORED},
        )

    def restore_stream_state(self, state: StreamState) -> None:
        for name in STREAM_STATE_MIRRORED:
            setattr(self, name, getattr(state, name))
        del self.stack[state.graphics_stack_len :]
        del self.clip_scope_stack[state.graphics_stack_len :]
        del self.marked_content_stack[state.marked_content_stack_len :]
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
            # The group buffer carries the constant alpha: it is applied once,
            # when the finished group composites into its backdrop. Propagating
            # it to each drawing inside as well would square it -- a 0.34 group
            # would paint its contents at 0.12.
            self.group_alpha = None

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
            frame.lexer = PdfLexer(frame.stream.data)
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
                dispatch_operations(
                    lexer,
                    self.op_handlers.get,
                    self,
                    frame.depth,
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
        raw_category = self.resources.get(category)
        category_res = (
            self.document.resolver.resolve_dict(raw_category) if raw_category is not None else None
        )

        if isinstance(category_res, dict):
            res = category_res.get(name)
            if res is not None:
                return self.document.resolver.resolve(res)

        if parent_category:
            raw_parent = self.resources.get(parent_category)
            parent_res = (
                self.document.resolver.resolve_dict(raw_parent) if raw_parent is not None else None
            )

            if isinstance(parent_res, dict):
                for p_val in parent_res.values():
                    sub_res_dict = None
                    if isinstance(p_val, dict):
                        sub_res_dict = p_val.get("Resources")
                    if isinstance(sub_res_dict, dict):
                        sub_cat = sub_res_dict.get(category)
                        if isinstance(sub_cat, dict):
                            found = sub_cat.get(name)
                            if found is not None:
                                return self.document.resolver.resolve(found)

        return None

    def chunk_advance(
        self, code: int, decoder: "FontDecoder", *, word_space: bool = False
    ) -> float:
        advance_x, advance_y = decoder.glyph_advance_vector(
            code,
            font_size=self.font_size,
            char_space=self.char_space,
            word_space=self.word_space,
            horizontal_scale=self.horizontal_scale,
            encoded_space=word_space,
        )
        # Capture's vertical geometry uses a positive distance down the writing
        # line, while the PDF text matrix uses the signed (normally negative)
        # y displacement returned above.
        return -advance_y if decoder.is_vertical else advance_x

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
            data = operand.encode("latin-1", "replace")
            needs_decode = False
            text = operand
        else:
            text = self.document.resolver.resolve_str(operand)
            if text is None:
                return "", b"", None
            data = text.encode("latin-1", "replace")
            needs_decode = False

        glyphs = None
        if needs_decode:
            glyphs = decoder.decode_glyphs(data)
            text = "".join([glyph.unicode for glyph in glyphs])
        if self.capture_glyphs:
            if glyphs is None:
                glyphs = decoder.decode_glyphs(data)
            if text is None:
                text = "".join([glyph.unicode for glyph in glyphs])
        if text is None:
            text = ""
        return text, data, glyphs

    def is_text_visible(self, text: str) -> bool:
        if not text:
            return False
        first_code = ord(text[0])
        if (first_code < 32 or 0xE000 <= first_code <= 0xF8FF) and is_garbage_text(text):
            return False
        if (
            not self.marked_content_stack
            and self.render_mode not in internal_NON_PAINTING_RENDER_MODES
            and self.font_size >= 0.1
        ):
            return True

        # Render mode 3 and sub-0.1pt text paint nothing, so they are not visible
        # here. Whether such a layer is nonetheless the page's real text -- a scan
        # carrying an OCR layer -- is a property of the whole page, not of the runs
        # captured before this operator, so that call belongs to
        # `internal_hidden_text_is_trusted` once parsing has seen every run.
        if self.render_mode in internal_NON_PAINTING_RENDER_MODES or self.font_size < 0.1:
            return False

        for entry in self.marked_content_stack:
            layer = entry.layer
            if layer and layer in self.hidden_layers:
                return False
        return True

    def internal_records_path(self) -> bool:
        """True when path construction operators must record geometry."""
        return self.capture_clipping or (
            (self.capture_graphics or self.capture_glyphs) and self.is_graphics_visible()
        )

    def is_graphics_visible(self) -> bool:
        if self.marked_content_stack:
            for entry in self.marked_content_stack:
                layer = entry.layer
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

        font_dict = typing.cast(PdfDict, font_obj)
        resolved_font = self.document.resolver.resolve_font_dict(font_dict)
        decoder = FontDecoder(
            typing.cast(dict[str, object], resolved_font),
            ligature_overrides=detect_ligature_overrides(
                self.document, self.resources, resolved_font
            ),
            raster_font_provider=self.document.raster_font_provider,
        )
        self.current_decoder = decoder
        self.current_decoder_resources_id = self.resources_id
        if update_metrics:
            self.update_font_metrics()
        return decoder

    def op_Do(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        self.append_xobject(operands[0], depth)

    def append_xobject(self, name_obj: Any, depth: int) -> None:
        name = self.document.resolver.resolve_name(name_obj)
        if not name:
            return
        xobjects = self.resources.get("XObject")
        raw_xobj = xobjects.get(name) if isinstance(xobjects, dict) else None
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
        subtype = self.document.resolver.resolve_name(xobj_dict.get("Subtype"))
        if self.document.resolver.resolve_name(xobj_dict.get("Type")) == "ObjStm":
            return
        if subtype == "Image":
            if self.is_graphics_visible():
                width = self.document.resolver.resolve_int(xobj_dict.get("Width")) or 0
                height = self.document.resolver.resolve_int(xobj_dict.get("Height")) or 0
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
                smask = xobj_dict.get("SMask")
                if smask is not None:
                    smask_stream = self.document.resolver.resolve(smask)
                    if isinstance(smask_stream, PdfStream):
                        smask_dict = (
                            self.document.resolver.resolve_dict(smask_stream.dictionary) or {}
                        )
                        smask_data = getattr(smask_stream, "raw_data", b"")
                        soft_mask_raw_data = smask_data
                        soft_mask_dictionary = dict(smask_dict)
                        width = self.document.resolver.resolve_int(smask_dict.get("Width")) or 0
                        height = self.document.resolver.resolve_int(smask_dict.get("Height")) or 0
                        if width > 0 and height > 0 and smask_data:
                            total = min(len(smask_data), width * height)
                            if total > 0:
                                smask_sum = numpy.frombuffer(
                                    smask_data, numpy.uint8, count=total
                                ).sum(dtype=numpy.uint64)
                                smask_alpha = int(smask_sum) / (255.0 * total)
                drawing_dictionary = dict(xobj_dict)
                soft_mask = (
                    SoftMask(soft_mask_raw_data, soft_mask_dictionary or {})
                    if soft_mask_raw_data is not None
                    else None
                )
                source_dictionary = dict(drawing_dictionary)
                # The colour manager reads the palette and base space straight
                # off this dictionary, so an indirect /ColorSpace (or one whose
                # Indexed lookup table is indirect) left it unable to convert
                # the samples and the whole image was dropped.
                color_space = source_dictionary.get("ColorSpace")
                if color_space is not None:
                    source_dictionary[PdfName.of("ColorSpace")] = (
                        self.document.resolver.deep_resolve(color_space)
                    )
                # A stencil mask carries no colour samples: PDF 8.9.6.2 paints its
                # set bits in the current fill colour. Every other image ignores
                # the fill, so recording it is only meaningful for the mask case,
                # but it costs nothing to carry and the renderer decides.
                image_is_stencil = xobj_dict.get("ImageMask") is True
                self.drawings.append(
                    CapturedDrawing(
                        seqno=self.sequence,
                        fill=self.fill_color if image_is_stencil else None,
                        fill_opacity=self.fill_opacity if image_is_stencil else None,
                        blend_mode=self.blend_mode,
                        dash_pattern=self.transformed_dash_pattern(),
                        soft_mask_alpha=smask_alpha,
                        kind="image",
                        image_source=self.internal_image_source(
                            getattr(xobj, "raw_data", b""),
                            source_dictionary,
                            soft_mask=soft_mask,
                        ),
                        image_clip=self.clip_bbox,
                        items=[("quad", quad)] if quad is not None else [],
                        bbox=bbox,
                        stream_order=self.stream_order,
                        xobject_depth=self.xobject_depth,
                    )
                )
                self.drawings[-1].raw_data = getattr(xobj, "raw_data", b"")
                self.drawings[-1].dictionary = drawing_dictionary
            return
        if subtype != "Form":
            return
        group_alpha = None
        group = xobj_dict.get("Group")
        if group is not None:
            group_dict = self.document.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.document.resolver.resolve_name(group_dict.get("S")) == "Transparency"
            ):
                # PDF 32000-1 Table 147: a transparency group dictionary holds
                # S/CS/I/K and nothing else. The constant alpha and blend mode
                # that composite the finished group into its backdrop come from
                # the graphics state in effect at the `Do` (11.6.6), so reading
                # a /ca off the group dictionary found nothing and dropped the
                # group entirely -- the contents then painted straight onto the
                # page at full opacity in Normal mode, losing the blend.
                #
                # Only isolate the group when compositing would actually differ;
                # at ca == 1 in Normal mode a group buffer is a no-op, and
                # painting directly stays the cheaper path.
                blend = self.blend_mode
                if self.fill_opacity < 1.0 or (blend is not None and blend != "Normal"):
                    group_alpha = max(0.0, min(1.0, self.fill_opacity))
        raw_resources = xobj_dict.get("Resources")
        resources = cast(
            PdfDict,
            (
                raw_resources
                if isinstance(raw_resources, dict)
                else self.document.resolver.resolve_dict(raw_resources)
            )
            or self.resources,
        )
        xobj_matrix = xobj_dict.get("Matrix")
        if isinstance(xobj_matrix, (list, tuple)) and len(xobj_matrix) > 6:
            xobj_matrix = xobj_matrix[:6]
        nested_ctm = (
            Matrix.from_operand(xobj_matrix) if xobj_matrix is not None else IDENTITY_MATRIX
        ).multiply(self.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
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

    def flush_run(self) -> None:
        if self.pending_run:
            self.runs.append(self.pending_run)
            self.pending_run = None

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.ca + y * self.cc + self.ce,
            x * self.cb + y * self.cd + self.cf,
        )

    def graphics_scale(self) -> float:
        x_scale = hypot(self.ca, self.cb)
        y_scale = hypot(self.cc, self.cd)
        if x_scale == 0 and y_scale == 0:
            return 1.0
        if x_scale == 0:
            return y_scale
        if y_scale == 0:
            return x_scale
        return (x_scale + y_scale) * 0.5

    def transformed_line_width(self) -> float:
        line_width = max(0.0, self.line_width)
        if line_width == 0:
            return 0.0
        return line_width * self.graphics_scale()

    def transformed_dash_pattern(self) -> tuple[list[float], float] | None:
        dash_pattern = self.dash_pattern
        if not dash_pattern:
            return None
        dash_array, phase = dash_pattern
        scale = self.graphics_scale()
        return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale

    def flush_drawing(self, kind: str, fill_rule: str = "nonzero") -> None:
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
        self,
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
        return TextRun(
            text=normalize_extracted_text(text),
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            tx=tx,
            ty=ty,
            font_size=font_size,
            space_width=space_width,
            order=order,
            stream_order=stream_order,
            xobject_depth=xobject_depth,
            font_name=font_name,
            is_vertical=is_vertical,
            rotation_angle=rotation_angle,
            visible=visible,
            line_break_before=line_break_before,
            seqno=seqno,
            fill_color=fill_color,
            advance_bbox=advance_bbox,
            ink_bbox=ink_bbox,
            baseline=baseline,
            provenance=provenance,
            confidence=confidence,
            glyph_clusters=glyph_clusters,
        )

    def internal_is_clipped_away(self, x0: float, y0: float, x1: float, y1: float) -> bool:
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

    def update_pending_run(self, new_run: TextRun) -> None:
        if self.internal_is_clipped_away(new_run.x0, new_run.y0, new_run.x1, new_run.y1):
            return

        if not self.pending_run:
            self.pending_run = new_run
            return

        p = self.pending_run
        p_text = p.text
        new_text = new_run.text
        p_font_size = p.font_size
        p_space_width = p.space_width
        p_rotation = p.rotation_angle
        merge_threshold = max(p_space_width * 0.45, 2.0)

        is_same_style = (
            p_rotation == new_run.rotation_angle
            and p.visible == new_run.visible
            and not new_run.line_break_before
            and p_font_size == new_run.font_size
            and (
                p.font_name == new_run.font_name
                or can_merge_cross_font_word(p_text, new_text)
                or can_merge_cross_font_word(new_text, p_text)
            )
            and p.fill_color == new_run.fill_color
        )

        if is_same_style and p_rotation == 90:
            y_gap = new_run.y0 - p.y1
            max_y_gap = max(p_space_width * 0.5, p_font_size * 0.8, 2.0)
            if abs(y_gap) > max_y_gap:
                is_same_style = False
        elif is_same_style and p_rotation == 0:
            if abs(p.y0 - new_run.y0) > p_font_size * 0.5:
                is_same_style = False

        merged = False
        if is_same_style:
            if p_rotation in (0, 90):
                if p_rotation == 0:
                    gap = new_run.x0 - p.x1
                    if -2.0 <= gap < merge_threshold:
                        separator = gap_separator(p_text, new_text, gap, p)
                        p.text = p_text + separator + new_text
                        p.union_ink_bbox(new_run.ink_bbox)
                        if new_run.x1 > p.x1:
                            p.x1 = new_run.x1
                        merged = True
                    else:
                        gap_rtl = p.x0 - new_run.x1
                        if -2.0 <= gap_rtl < merge_threshold:
                            separator = gap_separator(new_text, p_text, gap_rtl, p)
                            p.text = new_text + separator + p_text
                            p.union_ink_bbox(new_run.ink_bbox)
                            if new_run.x0 < p.x0:
                                p.x0 = new_run.x0
                            merged = True
                else:
                    gap = new_run.y0 - p.y1
                    if -2.0 <= gap < merge_threshold:
                        separator = gap_separator(p_text, new_text, gap, p)
                        p.text = p_text + separator + new_text
                        p.union_ink_bbox(new_run.ink_bbox)
                        if new_run.y1 > p.y1:
                            p.y1 = new_run.y1
                        merged = True
            else:
                h_gap_inv = p.x0 - new_run.x1
                if -2.0 <= h_gap_inv < merge_threshold:
                    separator = gap_separator(p_text, new_text, h_gap_inv, p)
                    p.text = p_text + separator + new_text
                    p.union_ink_bbox(new_run.ink_bbox)
                    if new_run.x0 < p.x0:
                        p.x0 = new_run.x0
                    merged = True
                else:
                    h_gap_inv_rtl = new_run.x0 - p.x1
                    if -2.0 <= h_gap_inv_rtl < merge_threshold:
                        separator = gap_separator(new_text, p_text, h_gap_inv_rtl, p)
                        p.text = new_text + separator + p_text
                        p.union_ink_bbox(new_run.ink_bbox)
                        if new_run.x1 > p.x1:
                            p.x1 = new_run.x1
                        merged = True

        if not merged:
            self.runs.append(p)
            self.pending_run = new_run
        else:
            p.advance_bbox = (p.x0, p.y0, p.x1, p.y1)
            p.glyph_clusters += new_run.glyph_clusters

    def record_glyph_observations(
        self,
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

        The union is accumulated as observations are appended, in append
        order, so the caller can skip re-scanning the appended slice.
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
        compat_cursor: PdfminerCursor | None = None
        text_basis = (
            self.tm_e * self.ca + self.tm_f * self.cc + self.ce,
            self.tm_e * self.cb + self.tm_f * self.cd + self.cf,
            combined_a,
            combined_b,
            combined_c,
            combined_d,
        )
        append_glyph = self.glyphs.append
        chunk_advance = self.chunk_advance
        glyph_bbox_for_code = decoder.glyph_bbox
        vertical_position = decoder.vertical_glyph_position
        vertical_metric = decoder.vertical_glyph_metric
        clusters = self.glyph_clusters
        cursor = 0
        is_vertical = decoder.is_vertical
        glyph_width = decoder.glyph_width
        if compat_tj_active:
            compat_spacing_scale = self.horizontal_scale * 0.01
            compat_cursor = PdfminerCursor(
                self.compat_tj_cursor_x,
                self.compat_tj_cursor_y,
                self.compat_tj_need_charspace,
                char_space=self.char_space * compat_spacing_scale,
                word_space=(0.0 if decoder.is_cid_font else self.word_space * compat_spacing_scale),
                spacing_scale=compat_spacing_scale,
                origin_x=(
                    self.compat_tj_origin_e * self.ca + self.compat_tj_origin_f * self.cc + self.ce
                ),
                origin_y=(
                    self.compat_tj_origin_e * self.cb + self.compat_tj_origin_f * self.cd + self.cf
                ),
                combined=(combined_a, combined_b, combined_c, combined_d),
                is_vertical=is_vertical,
                font_size=font_size,
            )
        effective_font_name = decoder.font_name or font_name
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
            return (
                transform_a,
                transform_b,
                transform_c,
                transform_d,
                text_basis[0] + origin_x * combined_a + origin_y * combined_c,
                text_basis[1] + origin_x * combined_b + origin_y * combined_d,
            )

        if axis_aligned_horizontal:
            axis_advance_y0 = text_basis[1] + (font_descent + rise) * combined_d
            axis_advance_y1 = text_basis[1] + (font_ascent + rise) * combined_d
            if axis_advance_y0 > axis_advance_y1:
                axis_advance_y0, axis_advance_y1 = axis_advance_y1, axis_advance_y0
            axis_baseline_y = text_basis[1] + rise * combined_d
        # Accumulated during the append loop so the caller need not rescan
        # the slice it just wrote.
        run_geometry = RunGeometry()
        add_run_geometry = run_geometry.add
        for glyph in glyphs:
            if is_vertical:
                advance = chunk_advance(
                    glyph.width_code,
                    decoder,
                    word_space=glyph.code_bytes == b" ",
                )
                glyph_width_units = 0.0
            else:
                # Reused by the compat cursor below. glyph_width is three frames
                # deep, so calling it twice per glyph is pure overhead.
                glyph_width_units = glyph_width(glyph.width_code)
                advance = (
                    glyph_width_units
                    + char_space_scale
                    + (word_space_scale if glyph.code_bytes == b" " else 0.0)
                ) * advance_scale
            chunk_text = glyph.unicode
            if not chunk_text:
                chunk_text = text[cursor : cursor + 1]
            chunk_length = len(chunk_text)
            cursor += max(1, chunk_length)
            if not chunk_text:
                offset += advance
                continue

            if compat_cursor is not None:
                width_units = (
                    float(vertical_metric(glyph.width_code)[0])
                    if is_vertical
                    else glyph_width_units
                )
                observation_provenance: tuple[tuple[str, Any], ...] = (
                    *glyph_provenance,
                    *compat_cursor.step(width_units, glyph.char_code == 32),
                )
            else:
                observation_provenance = glyph_provenance

            cluster_id = len(clusters)
            cluster_provenance_id = (seqno, cluster_id)
            if is_vertical:
                glyph_vertical_position = vertical_position(
                    glyph.cid,
                    font_size=font_size,
                )
                text_box, baseline_text = glyph_text_space_boxes(
                    self,
                    offset,
                    advance,
                    decoder,
                    glyph_vertical_position,
                )
                transformed = transformed_text_rect(self, *text_box, text_basis)
                advance_bbox = (
                    transformed.x0,
                    transformed.y0,
                    transformed.x1,
                    transformed.y1,
                )
                baseline = transformed_text_line(*baseline_text, text_basis)
                outline_transform = glyph_transform(offset, glyph_vertical_position)
            else:
                # Inlined glyph_transform for horizontal text; the sums keep
                # the closure's evaluation order so floats stay bit-identical.
                outline_transform = (
                    transform_a,
                    transform_b,
                    transform_c,
                    transform_d,
                    text_basis[0] + offset * combined_a + rise_offset_x,
                    text_basis[1] + offset * combined_b + rise_offset_y,
                )
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
                    )
                    transformed = transformed_text_rect(self, *text_box, text_basis)
                    advance_bbox = (
                        transformed.x0,
                        transformed.y0,
                        transformed.x1,
                        transformed.y1,
                    )
                    baseline = transformed_text_line(*baseline_text, text_basis)
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
                glyph_bbox = glyph_bbox_for_code(glyph.bitmap_code)
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
                if (
                    should_capture_glyph_bitmap(chunk_text)
                    if single_character
                    else suspicious_multi
                ):
                    bitmap_width, bitmap_height = glyph_bitmap_dimensions(
                        glyph_bbox,
                        font_size,
                    )
                    bitmap_code = glyph.bitmap_code
                observation = GlyphObservation(
                    text=chunk_text,
                    ink_bbox=rect,
                    advance_bbox=advance_bbox,
                    seqno=seqno,
                    code_bytes=glyph.code_bytes,
                    char_code=glyph.char_code,
                    cid=glyph.cid,
                    gid=glyph.gid,
                    font_name=effective_font_name,
                    font_size=font_size,
                    baseline=baseline,
                    rotation_angle=rotation_angle,
                    fill=fill,
                    visible=observation_visible,
                    confidence=observation_confidence,
                    unicode_source=glyph.unicode_source,
                    alternates=glyph.alternates,
                    bitmap_width=bitmap_width,
                    bitmap_height=bitmap_height,
                    bitmap_code=bitmap_code,
                    font_decoder=decoder,
                    effective_font_size=effective_font_size,
                    effective_font_height=effective_font_height,
                    provenance=observation_provenance,
                    glyph_transform=outline_transform,
                    text_render_mode=render_mode,
                    fill_opacity=fill_opacity,
                    stroke_color=stroke_color,
                    stroke_opacity=stroke_opacity,
                    line_width=glyph_line_width,
                    line_cap=line_cap,
                    line_join=line_join,
                    dash_pattern=glyph_dash_pattern,
                    blend_mode=blend_mode,
                    soft_mask_alpha=group_alpha,
                    text_object_id=text_object_id,
                    cluster_key=cluster_provenance_id,
                )
                append_glyph(observation)
                clusters.append(
                    GlyphCluster(
                        cluster_id=cluster_id,
                        text=chunk_text,
                        glyphs=(observation,),
                        advance_bbox=advance_bbox,
                        ink_bbox=rect,
                        baseline=baseline,
                        confidence=observation_confidence,
                    )
                )
                add_run_geometry(advance_bbox, rect, observation_confidence)
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
                add_run_geometry(
                    observation.advance_bbox, observation.ink_bbox, observation.confidence
                )
            cluster = glyph_cluster_from_observations(
                cluster_id,
                chunk_text,
                tuple(cluster_observations),
            )
            if cluster is not None:
                clusters.append(cluster)
            offset += advance

        if compat_cursor is not None:
            self.compat_tj_cursor_x = compat_cursor.x
            self.compat_tj_cursor_y = compat_cursor.y
            self.compat_tj_need_charspace = compat_cursor.need_charspace
        if not run_geometry.started:
            return None
        return run_geometry.advance, run_geometry.ink, run_geometry.confidence

    def _append_text_impl(
        self,
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
                text = "".join([glyph.unicode for glyph in glyphs])
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

        x0 = min(c0_x, c1_x, c2_x, c3_x)
        y0 = min(c0_y, c1_y, c2_y, c3_y)
        x1 = max(c0_x, c1_x, c2_x, c3_x)
        y1 = max(c0_y, c1_y, c2_y, c3_y)

        rot = detect_rotation_from_linear(A, B, C, D)
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
                actual_text_span.compatibility_glyphs.extend(self.glyphs[glyph_start:])
                del self.glyphs[glyph_start:]
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

    def _render_type3_glyphs_impl(self, data: bytes | memoryview, decoder: FontDecoder) -> None:
        # ISO 32000-1 9.3.6: "Only a value of 3 for text rendering mode shall
        # have any effect on text displayed in a Type 3 font", and Table 106
        # makes mode 3 invisible. Mode 7 deliberately still paints here -- for a
        # Type 3 font the clause says only mode 3 has an effect, unlike the
        # simple-font case where 7 also adds no marks.
        if self.render_mode == 3:
            return
        font = decoder.font
        char_procs = font.get("CharProcs")
        if not isinstance(char_procs, dict):
            return
        glyph_names = decoder.type3_glyph_names
        if glyph_names is None:
            glyph_names = type3_glyph_names(font, decoder)
            decoder.type3_glyph_names = glyph_names

        resources = font.get("Resources")
        if not isinstance(resources, dict):
            resources = self.resources
        font_matrix = type3_font_matrix(font)
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale
        scale = self.text_advance_scale

        for code in data:
            glyph_name = glyph_names.get(code)
            char_proc = self.document.resolver.resolve(
                char_procs.get(glyph_name) if glyph_name else None
            )
            if isinstance(char_proc, PdfStream):
                # ISO 32000-1 9.6.5: when the glyph description begins, the CTM
                # is "the concatenation of the font matrix ... and the text space
                # that was in effect at the time the text-showing operator was
                # invoked". Text space is Trm from 9.4.4 NOTE 2:
                #
                #   Trm = [Tfs x Th, 0, 0; 0, Tfs, 0; 0, Trise, 1] x Tm x CTM
                #
                # `multiply` applies the receiver first, so the font matrix has
                # to lead. It was trailing, and the Tfs/Th/Trise factor was
                # missing entirely, which left every Type 3 glyph painted at
                # FontMatrix scale near the origin and independent of font size.
                text_space = Matrix(
                    self.combined_A,
                    self.combined_B,
                    self.combined_C,
                    self.combined_D,
                    self.tm_e * self.ca + self.tm_f * self.cc + self.ce,
                    self.tm_e * self.cb + self.tm_f * self.cd + self.cf,
                )
                font_size = self.font_size
                glyph_ctm = font_matrix.multiply(
                    Matrix(
                        font_size * self.horizontal_scale / 100.0,
                        0.0,
                        0.0,
                        font_size,
                        0.0,
                        self.rise,
                    ).multiply(text_space)
                )
                previous_type3_uncolored = self.type3_uncolored
                self.type3_uncolored = False
                try:
                    self.consume_stream(char_proc, resources, glyph_ctm, self.xobject_depth + 1)
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

    def append_tj_array(self, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            return
        if not array:
            return
        pending_bytes = bytearray()
        scale = self.text_advance_scale

        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
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
        self.compat_tj_active = self.capture_glyphs and self.compat_tj_enabled
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

    def current_actual_text_span(self) -> Any | None:
        for entry in reversed(self.marked_content_stack):
            if getattr(entry, "actual_text", None) is not None:
                return entry
        return None

    def current_marked_content_mcid(self) -> int | None:
        for entry in reversed(self.marked_content_stack):
            mcid = getattr(entry, "mcid", None)
            if type(mcid) is int:
                return mcid
        return None

    def emit_actual_text_span(self, entry: Any) -> None:
        actual_text = getattr(entry, "actual_text", None)
        if actual_text is None or not getattr(entry, "has_text_extents", False):
            return
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
            order=entry.seqno,
            stream_order=entry.stream_order,
            xobject_depth=entry.xobject_depth,
            is_vertical=entry.is_vertical,
            rotation_angle=entry.rotation_angle,
            visible=entry.visible,
            line_break_before=entry.line_break_before,
            seqno=entry.seqno,
            fill_color=entry.fill_color,
            advance_bbox=entry.advance_bbox,
            ink_bbox=entry.advance_bbox,
            baseline=entry.baseline,
            provenance=(*entry.provenance, ("unicode_source", "actual_text")),
            confidence=entry.confidence,
        )
        self.update_pending_run(new_run)
        if self.capture_glyphs and entry.advance_bbox is not None:
            self.glyphs.append(
                GlyphObservation(
                    text=actual_text,
                    ink_bbox=entry.advance_bbox,
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
        self.flush_run()
        self.tm_a = self.lm_a = 1.0
        self.tm_b = self.lm_b = 0.0
        self.tm_c = self.lm_c = 0.0
        self.tm_d = self.lm_d = 1.0
        self.tm_e = self.lm_e = 0.0
        self.tm_f = self.lm_f = 0.0
        self.compat_tj_cursor_x = 0.0
        self.compat_tj_cursor_y = 0.0
        self.update_combined()

    def op_ET(self, operands: ContentOperands, depth: int) -> None:
        self.flush_run()

    def internal_move_text(self, tx: float, ty: float) -> None:
        self.flush_run()
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

    def op_BT(self, operands: ContentOperands, depth: int) -> None:
        self.text_object_id += 1
        self.internal_begin_text()

    def op_T_star(self, operands: ContentOperands, depth: int) -> None:
        self.internal_move_text(0.0, -self.leading)

    def op_Td(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        try:
            tx = self.as_float(operands[0])
            ty = self.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.internal_move_text(tx, ty)

    def op_TD(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        try:
            tx = self.as_float(operands[0])
            ty = self.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.leading = -ty
        self.internal_move_text(tx, ty)

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        # Operators consume their operands from the top of the operand stack.
        # A well-formed Tj has exactly one string, but damaged streams sometimes
        # leave older operands before it.  Those older values are not part of
        # the text-showing operation.
        self.internal_show_text(operands[-1])

    def op_TJ(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            self.append_tj_array(operands[0])

    def op_Tm(self, operands: ContentOperands, depth: int) -> None:
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

    def op_Tf(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        font_operand = operands[0]
        font_size_operand = operands[1]
        decoder_matches_resources = self.current_decoder_resources_id == self.resources_id
        if (
            self.current_decoder is not None
            and decoder_matches_resources
            and font_operand is self.font_operand
        ):
            if font_size_operand is not self.font_size_operand:
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
        font_name = self.document.resolver.resolve_name(font_operand)
        if font_name is None:
            return
        try:
            font_size = self.as_float(font_size_operand)
        except (TypeError, ValueError):
            return
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

    def op_TL(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            self.leading = self.as_float(operands[0])
        except (TypeError, ValueError):
            return

    def op_Tc(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            char_space = self.as_float(operands[0])
        except (TypeError, ValueError):
            return
        self.char_space = char_space
        self.update_text_scales()

    def op_Tw(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            word_space = self.as_float(operands[0])
        except (TypeError, ValueError):
            return
        if self.word_space == word_space:
            return
        self.word_space = word_space
        self.update_text_scales()

    def op_Tr(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            self.render_mode = self.as_int(operands[0])
        except (TypeError, ValueError):
            return

    def op_Tz(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            self.horizontal_scale = self.as_float(operands[0])
        except (TypeError, ValueError):
            return
        self.update_text_scales()

    def op_Ts(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            self.rise = self.as_float(operands[0])
        except (TypeError, ValueError):
            return

    def op_quote(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        self.internal_move_text(0.0, -self.leading)
        self.pending_line_break = True
        self.internal_show_text(operands[0])

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 3:
            return
        try:
            word_space = self.as_float(operands[0])
            char_space = self.as_float(operands[1])
        except (TypeError, ValueError):
            return
        self.word_space = word_space
        self.char_space = char_space
        self.update_text_scales()
        if not self.document.legacy_pdfminer_text_operators:
            self.internal_move_text(0.0, -self.leading)
            self.pending_line_break = True
        self.internal_show_text(operands[2])

    def op_BI(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        operand = operands[0]
        # Duck-typed on purpose: the inline-image parser yields an InlineImage, but any
        # operand exposing ``dictionary`` is accepted here.
        if not hasattr(operand, "dictionary"):
            return
        image = cast("InlineImage", operand)
        if self.is_graphics_visible():
            dictionary = dict(image.dictionary)
            data = getattr(image, "data", b"")
            source = self.internal_image_source(data, dictionary)
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
        raw: bytes | memoryview,
        dictionary: dict[Any, Any],
        *,
        soft_mask: SoftMask | None = None,
    ) -> ImageSource:
        return ImageSource(
            raw,
            dictionary,
            soft_mask=soft_mask,
        )

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        tag = self.document.resolver.resolve_name(operands[0]) if operands else None
        layer: str | None = None
        actual_text: str | None = None
        mcid: int | None = None
        if len(operands) >= 2:
            properties = operands[1]
            if tag == "OC":
                layer = self.resolve_marked_content_layer(properties)
            # ActualText and MCID both live in this dictionary; resolving it
            # once can mean one fewer page-resource lookup per BDC.
            props = self.resolve_marked_content_properties(properties)
            if props is not None:
                resolver = self.document.resolver
                if tag == "Span":
                    actual_text = resolver.resolve_str(props.get("ActualText"))
                mcid = resolver.resolve_int(props.get("MCID"))
        self.marked_content_stack.append(
            MarkedContentEntry(layer=layer, actual_text=actual_text, mcid=mcid)
        )

    def op_BMC(self, operands: ContentOperands, depth: int) -> None:
        self.marked_content_stack.append(MarkedContentEntry())

    def op_EMC(self, operands: ContentOperands, depth: int) -> None:
        if self.marked_content_stack:
            self.emit_actual_text_span(self.marked_content_stack.pop())

    def op_G(self, operands: ContentOperands, depth: int) -> None:
        if operands and not self.type3_uncolored:
            self.stroke_color_space = "DeviceGray"
            self.set_stroke_color(operands[0])

    def op_RG(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 3 and not self.type3_uncolored:
            self.stroke_color_space = "DeviceRGB"
            self.set_stroke_color(operands[0], operands[1], operands[2])

    def op_K(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 4 and not self.type3_uncolored:
            self.stroke_color_space = "DeviceCMYK"
            self.set_stroke_color(operands[0], operands[1], operands[2], operands[3])

    def op_w(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            try:
                self.line_width = max(0.0, self.as_float(operands[0]))
            except (TypeError, ValueError):
                return

    def op_J(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            try:
                self.line_cap = self.as_int(operands[0])
            except (TypeError, ValueError):
                return

    def op_j(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            try:
                self.line_join = self.as_int(operands[0])
            except (TypeError, ValueError):
                return

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            try:
                self.miter_limit = max(1.0, self.as_float(operands[0]))
            except (TypeError, ValueError):
                return

    def op_d(self, operands: ContentOperands, depth: int) -> None:
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

    def op_m(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 2:
            try:
                point = (self.as_float(operands[0]), self.as_float(operands[1]))
            except (TypeError, ValueError):
                return
            if self.internal_records_path():
                self.current_path.move_to(*point)
            self.current_point = point
            self.subpath_start = point

    def op_l(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 2 and self.current_point is not None:
            try:
                point = (self.as_float(operands[0]), self.as_float(operands[1]))
            except (TypeError, ValueError):
                return
            if self.internal_records_path():
                self.current_path.line_to(*point)
            self.current_point = point

    def op_re(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 4:
            try:
                x, y = self.as_float(operands[0]), self.as_float(operands[1])
                w, h = self.as_float(operands[2]), self.as_float(operands[3])
            except (TypeError, ValueError):
                return
            if self.internal_records_path():
                self.current_path.rect(x, y, w, h)
            self.current_point = (x, y)
            self.subpath_start = (x, y)

    def op_h(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is not None and self.subpath_start is not None:
            if self.internal_records_path():
                self.current_path.close()
            self.current_point = self.subpath_start

    def op_c(self, operands: ContentOperands, depth: int) -> None:
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

    def op_v(self, operands: ContentOperands, depth: int) -> None:
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

    def op_y(self, operands: ContentOperands, depth: int) -> None:
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

    def internal_close_current_subpath(self) -> None:
        if (
            self.capture_graphics
            and self.is_graphics_visible()
            and self.current_point is not None
            and self.subpath_start is not None
        ):
            self.current_path.close()

    def internal_end_path(self) -> None:
        """Discard the current point and subpath origin after a painting operator."""
        self.current_point = None
        self.subpath_start = None

    def op_paint_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("stroke")
        self.internal_end_path()

    def op_paint_close_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("stroke")
        self.internal_end_path()

    def op_paint_fill(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fill", "nonzero")
        self.internal_end_path()

    def op_paint_fill_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fill", "evenodd")
        self.internal_end_path()

    def op_paint_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fillstroke", "nonzero")
        self.internal_end_path()

    def op_paint_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fillstroke", "evenodd")
        self.internal_end_path()

    def op_paint_close_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("fillstroke", "nonzero")
        self.internal_end_path()

    def op_paint_close_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("fillstroke", "evenodd")
        self.internal_end_path()

    def op_paint_clear(self, operands: ContentOperands, depth: int) -> None:
        self.current_path.clear()
        self.internal_end_path()

    def internal_emit_clip_scope_push(self) -> None:
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

    def op_W(self, operands: ContentOperands, depth: int) -> None:
        self.internal_record_clip("nonzero")

    def internal_record_clip(self, fill_rule: str) -> None:
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

    def op_W_star(self, operands: ContentOperands, depth: int) -> None:
        self.internal_record_clip("evenodd")

    def normalize_colors(self, *components: Any) -> tuple[float, ...] | None:
        values: list[float] = []
        for component in components:
            try:
                values.append(max(0.0, min(1.0, self.as_float(component))))
            except ValueError:
                return None
        if not values:
            return None
        return tuple(values)

    def set_stroke_color(self, *components: Any) -> None:
        if self.type3_uncolored:
            return
        normalized = self.normalize_colors(*components)
        if normalized is not None:
            self.stroke_color = normalized
            self.stroke_pattern = None

    def set_fill_color(self, *components: Any) -> None:
        if self.type3_uncolored:
            return
        normalized = self.normalize_colors(*components)
        if normalized is not None:
            self.fill_color = normalized
            self.fill_pattern = None

    # `o` is a tuple or list of raw PDF operands, so `Any` is the honest annotation.
    def normalize_color_operands(self, o: Any) -> tuple[float, ...] | None:
        # Plain numeric operands (the overwhelming majority) clamp directly;
        # anything else -- strings, names, nulls -- goes through the resolver.
        if o and all(type(c) is float or type(c) is int for c in o):
            return tuple(max(0.0, min(1.0, float(c))) for c in o)
        return self.normalize_colors(*o)

    def internal_color_space_value(self, name_obj: Any) -> object:
        """The colour-space resource behind a `cs`/`CS` operand, indirects resolved.

        A colour-space array carries the tint function and the Indexed palette
        as indirect references, so the entries have to be resolved before the
        array can be parsed.
        """
        name = self.document.resolver.resolve_name(name_obj)
        if name is None:
            return None
        resolve = self.document.resolver.resolve
        value = self.lookup_page_resource("ColorSpace", name)
        if value is None:
            # An inline device space (`/DeviceRGB cs`) names no resource.
            return name
        value = resolve(value)
        if isinstance(value, (list, tuple)):
            return [resolve(entry) for entry in value]
        return value

    def internal_resolve_color_spec(self, name_obj: Any) -> ImageColorSpec | None:
        """Resolve a `cs`/`CS` operand to its full colour space, not just a name.

        The parse walks the colour-space array and, for Indexed, pulls the
        palette out of a stream.
        """
        value = self.internal_color_space_value(name_obj)
        try:
            return color_spec_from_value(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def internal_color_from_operands(
        self, operands: Any, spec: ImageColorSpec | None
    ) -> tuple[float, ...] | None:
        """Turn `sc`/`scn` operands into the colour they select.

        Device-space operands are their own components and pass through. An
        Indexed operand is a palette index and a Separation/DeviceN operand is a
        tint (ISO 32000-1 8.6.6.3, 8.6.6.4), so those resolve to sRGB here --
        clamping them to 0..1 and painting them directly rendered a spot colour
        as an inverted grey and ignored the palette entirely.
        """
        if spec is not None and spec.kind in {"Indexed", "Separation", "DeviceN"}:
            values = self.internal_numeric_operands(operands)
            if values is not None:
                converted = color_operands_to_srgb(spec, values)
                if converted is not None:
                    return converted
        return self.normalize_color_operands(operands)

    def internal_numeric_operands(self, operands: Any) -> list[float] | None:
        values: list[float] = []
        for operand in operands:
            if type(operand) is float or type(operand) is int:
                values.append(float(operand))
            else:
                return None
        return values or None

    def resolve_color_space(self, name_obj: Any, *, default_fallback: bool = False) -> str | None:
        name = self.document.resolver.resolve_name(name_obj)
        if name is None:
            return "DeviceGray" if default_fallback else None

        color_space: object = self.lookup_page_resource("ColorSpace", name)
        if color_space is None:
            return name if default_fallback else None

        color_space_name = normalize_pdf_name(color_space)
        if color_space_name is not None:
            return color_space_name
        if isinstance(color_space, (list, tuple)) and color_space:
            base = color_space[0]
            base_name = normalize_pdf_name(base)
            if base_name is not None:
                return base_name

        if isinstance(name, str) and not name.startswith("/"):
            return name

        return name if default_fallback else None

    def resolve_pattern_color(self, operands: tuple[Any, ...]) -> PatternPaint | None:
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
        pattern_type = parse_int(pattern_dict.get("PatternType"), None)
        if pattern_type == 2:
            shading: object = pattern_dict.get("Shading")
            shading = self.document.resolver.resolve(shading)
            shading_dict = (
                self.document.resolver.resolve_dict(shading) if shading is not None else None
            )
            if not isinstance(shading_dict, dict):
                return None
            return ShadingPattern(dict(shading_dict))
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return None
        paint_type = parse_int(pattern_dict.get("PaintType"), 1)
        if paint_type not in {1, 2}:
            return None
        base_color = None
        if paint_type == 2:
            base_color = self.normalize_color_operands(operands[:-1])
            if base_color is None:
                return None
        bbox = self.document.resolver.resolve_box(pattern_dict.get("BBox"))
        if bbox is None:
            return None
        x_step = self.document.resolver.resolve_float(pattern_dict.get("XStep"), default=None)
        y_step = self.document.resolver.resolve_float(pattern_dict.get("YStep"), default=None)
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            return None
        try:
            matrix = Matrix.from_operand(pattern_dict.get("Matrix"))
        except ValueError:
            matrix = IDENTITY_MATRIX
        resources = self.document.resolver.resolve_dict(pattern_dict.get("Resources")) or {}
        nested_state = type(self)(
            self.document,
            self.page,
            hidden_layers=self.hidden_layers,
        )
        try:
            nested_state.consume_stream(pattern, resources, matrix, 0)
        except Exception:
            return None
        # The cell's drawings are owned by this pattern and painted nowhere
        # else, so an uncoloured (PaintType 2) pattern recolours them in place
        # rather than copying every field into a parallel record.
        for drawing in nested_state.drawings:
            if drawing.kind in {"fill", "fillstroke"}:
                drawing.fill = base_color
            if drawing.kind in {"stroke", "fillstroke"}:
                drawing.stroke_color = base_color
        return TilingPattern(
            bbox=bbox,
            x_step=float(x_step),
            y_step=float(y_step),
            drawings=nested_state.drawings,
            glyphs=[glyph for glyph in nested_state.glyphs if glyph.has_paint],
        )

    def internal_set_color_space(self, operands: ContentOperands, *, stroke: bool) -> None:
        if self.type3_uncolored:
            # 9.6.5.2: every colour operator is ignored inside an uncoloured
            # Type 3 glyph, `cs`/`CS` included. The colour setters already
            # refuse to move the colour, so without this the glyph would carry
            # a colour space describing a colour it was not allowed to set.
            return
        if operands:
            name_obj = operands[0]
            color_space = self.resolve_color_space(name_obj, default_fallback=True)
            if color_space is not None:
                spec = self.internal_resolve_color_spec(name_obj)
                if stroke:
                    self.stroke_color_space = color_space
                    self.stroke_color_spec = spec
                else:
                    self.fill_color_space = color_space
                    self.fill_color_spec = spec

    def op_CS(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color_space(operands, stroke=True)

    def op_cs(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color_space(operands, stroke=False)

    def op_SC(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=False)

    def internal_set_color(
        self, operands: ContentOperands, *, stroke: bool, allow_pattern: bool
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
                normalized = self.internal_color_from_operands(
                    operands[:-1], self.stroke_color_spec if stroke else self.fill_color_spec
                )
                if normalized is not None:
                    if stroke:
                        self.stroke_color = normalized
                    else:
                        self.fill_color = normalized
            return
        normalized = self.internal_color_from_operands(
            operands, self.stroke_color_spec if stroke else self.fill_color_spec
        )
        if normalized is not None:
            if stroke:
                self.stroke_color = normalized
                self.stroke_pattern = None
            else:
                self.fill_color = normalized
                self.fill_pattern = None

    def op_SCN(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=True)

    def op_sc(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=False)

    def op_scN(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=True)

    def op_i(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            try:
                value = self.as_float(operands[0])
            except ValueError:
                return
            self.flatness = max(0, min(100, int(value)))

    def op_ri(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        value = self.document.resolver.resolve_name_like_value(operands[0])
        if isinstance(value, str):
            self.render_intent = value

    def op_MP(self, operands: ContentOperands, depth: int) -> None:
        # A marked-content point is not a scope. Only BMC/BDC push and EMC pops.
        return

    def op_DP(self, operands: ContentOperands, depth: int) -> None:
        # A property-bearing marked-content point likewise has no lasting state.
        return

    def resolve_marked_content_properties(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            return cast("dict[str, Any]", resolved)
        name = self.document.resolver.resolve_name(value)
        if not name:
            return None
        props = self.lookup_page_resource("Properties", name)
        return cast("dict[str, Any]", props) if isinstance(props, dict) else None

    def resolve_marked_content_layer(self, value: Any) -> str | None:
        if value is None:
            return None

        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            oc = resolved.get("OC")
            if oc is not None:
                return self.document.resolver.resolve_name_or_text(oc)

        return self.document.resolver.resolve_name_or_text(value)

    def op_BX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth += 1

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
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
        return parse_float_strict(value, "invalid numeric operand")

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is int:
            return value
        return parse_int_strict(value, "invalid numeric operand")

    def resolve_extgstate(self, name: str) -> dict[str, Any] | None:
        resolved = self.lookup_page_resource("ExtGState", name)
        if not isinstance(resolved, dict):
            return None
        return cast("dict[str, Any]", resolved)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.clip_scope_stack.append(False)
        self.stack.append(
            internal_GraphicsStateSnapshot(
                ca=self.ca,
                cb=self.cb,
                cc=self.cc,
                cd=self.cd,
                ce=self.ce,
                cf=self.cf,
                fill_color=self.fill_color,
                fill_pattern=self.fill_pattern,
                fill_opacity=self.fill_opacity,
                stroke_color=self.stroke_color,
                stroke_pattern=self.stroke_pattern,
                stroke_opacity=self.stroke_opacity,
                fill_color_space=self.fill_color_space,
                stroke_color_space=self.stroke_color_space,
                fill_color_spec=self.fill_color_spec,
                stroke_color_spec=self.stroke_color_spec,
                compatibility_depth=self.compatibility_depth,
                blend_mode=self.blend_mode,
                group_alpha=self.group_alpha,
                flatness=self.flatness,
                render_intent=self.render_intent,
                clip_bbox=self.clip_bbox,
                line_width=self.line_width,
                line_cap=self.line_cap,
                line_join=self.line_join,
                miter_limit=self.miter_limit,
                dash_pattern=self.dash_pattern,
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
            )
        )

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
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
            self.fill_color_spec,
            self.stroke_color_spec,
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

    def op_cm(self, operands: ContentOperands, depth: int) -> None:
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

        self.ctm = Matrix(m_a, m_b, m_c, m_d, m_e, m_f).multiply(self.ctm)

    def op_g(self, operands: ContentOperands, depth: int) -> None:
        if operands and not self.type3_uncolored:
            self.fill_color_space = "DeviceGray"
            self.set_fill_color(operands[0])

    def op_rg(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 3 and not self.type3_uncolored:
            self.fill_color_space = "DeviceRGB"
            self.set_fill_color(operands[0], operands[1], operands[2])

    def op_k(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) >= 4 and not self.type3_uncolored:
            self.fill_color_space = "DeviceCMYK"
            self.set_fill_color(operands[0], operands[1], operands[2], operands[3])

    def op_gs(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            return
        try:
            fill_opacity = extgstate.get("ca")
            if fill_opacity is not None:
                self.fill_opacity = max(0.0, min(1.0, self.as_float(fill_opacity)))
            stroke_opacity = extgstate.get("CA")
            if stroke_opacity is not None:
                self.stroke_opacity = max(0.0, min(1.0, self.as_float(stroke_opacity)))
            blend_mode = extgstate.get("BM")
            if blend_mode is not None:
                if isinstance(blend_mode, (list, tuple)):
                    blend_mode = blend_mode[0] if blend_mode else None
                if blend_mode is not None:
                    self.blend_mode = self.document.resolver.resolve_name_like_value(blend_mode)
        except (TypeError, ValueError):
            return
