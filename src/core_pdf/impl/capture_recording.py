# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping
from contextlib import suppress
from copy import copy
from dataclasses import dataclass, replace
from math import hypot, isfinite
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy

from core_pdf.impl.caches import MISSING, BoundedDict, IdentityCache
from core_pdf.impl.capture_glyphs import (
    GlyphCapture,
    GlyphPaint,
    capture_glyphs,
    glyph_style,
)
from core_pdf.impl.capture_program import DEFAULT_CAPTURE, CapturedProgram, CaptureOptions
from core_pdf.impl.capture_records import (
    EMPTY_LINES,
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLines,
    CapturedPath,
    CapturedSoftMask,
    CapturedSubpath,
    CapturedTextBoundary,
    LayoutFormId,
    PaintedDrawingKind,
    PatternPaint,
    ShadingPattern,
    TilingPattern,
    marker_drawing,
)
from core_pdf.impl.capture_recovery import CaptureRecovery, iter_content_operations
from core_pdf.impl.capture_text_runs import (
    RunAccumulator,
    is_garbage_text,
)
from core_pdf.impl.capture_tolerant_state import (
    COLOR_CACHE_LIMIT,
    SOFT_MASK_CACHE_LIMIT,
    RecoveringTextState,
    soft_mask_cache_limit,
)
from core_pdf.impl.fonts_decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.fonts_ligatures import detect_ligature_overrides
from core_pdf.impl.geometry import (
    intersect_bbox,
    transform_bbox,
)
from core_pdf.impl.glyphs import GlyphObservation, GlyphStyle
from core_pdf.impl.graphics_color import color_operands_to_srgb
from core_pdf.impl.graphics_color_spec import raw_color_space_paints
from core_pdf.impl.graphics_images import decode_soft_mask
from core_pdf.impl.graphics_soft_masks import image_overrides_graphics_soft_mask
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.runs import TextRun
from core_pdf.impl.text import normalize_extracted_text
from core_pdf.impl.types import (
    PdfName,
    Rectangle,
)
from core_pdf_cythonized import flatten_path_commands
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import NON_PAINTING_RENDER_MODES, GraphicsState, PdfPath
from core_pdf_spec.s_07_content.model import (
    MarkedContentEntry as SemanticMarkedContentEntry,
)
from core_pdf_spec.s_07_content.model import ShadingPattern as PdfShadingPattern
from core_pdf_spec.s_07_content.model import TilingPattern as PdfTilingPattern
from core_pdf_spec.s_07_content.operations import OperationHandler
from core_pdf_spec.s_07_content.streams import (
    ContentStreamExecutor,
    ContentStreamFrame,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float_strict, parse_int_strict
from core_pdf_spec.s_08_graphics.color import color_space_paints
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    BlackPointCompensation,
    ColorRendering,
    override_color_rendering,
)
from core_pdf_spec.s_08_graphics.geometry import unit_square_placement
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_08_graphics.image_spec import (
    image_source_from_stream as resolve_image_source,
)
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph, FontService
from core_pdf_spec.s_11_transparency.soft_masks import SoftMask as PdfSoftMask

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.inline_images import InlineImage


@dataclass(slots=True)
class CaptureGraphicsSave:
    clip_bbox: Rectangle | None
    group_alpha: float | None
    clip_scope_emitted: bool = False


@dataclass(slots=True)
class MarkedContentEntry:
    layer: str | None = None
    actual_text: str | None = None
    mcid: int | None = None
    run: TextRun | None = None
    font_decoder: object | None = None
    effective_font_height: float = 0.0

    def add_run(
        self,
        run: TextRun,
        *,
        font_decoder: object | None = None,
        effective_font_height: float = 0.0,
    ) -> None:
        captured = self.run
        if captured is None:
            self.run = run
            self.font_decoder = font_decoder
            self.effective_font_height = effective_font_height
            return
        captured.absorb_extent(run)


MATRIX_TOLERANCE = 0.1


def detect_rotation_from_linear(A: float, B: float, C: float, D: float) -> int:
    scale_x = hypot(A, B)
    scale_y = hypot(C, D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    tolerance = MATRIX_TOLERANCE
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


CaptureMarks: TypeAlias = tuple[int, int, int, int, int, int]
NO_MARKS: CaptureMarks = (0, 0, 0, 0, 0, 0)


GLYPH_PAINT_KEEPING_OPERATORS = frozenset(
    ("BT", "Td", "TD", "Tm", "T*", "Tj", "TJ", "'", '"', "Tc", "Tw", "Tz", "TL", "Ts", "Tf")
)


TEXT_LAYOUT_KEEPING_OPERATORS = frozenset(("Tj", "TJ", "TL", "Tw"))
LINE_MOVING_OPERATORS = frozenset(("Td", "TD", "T*", "'"))


class TextLayout:
    __slots__ = (
        "decoder",
        "given_paint",
        "identity_text_matrix",
        "glyph_paint",
        "fill_color",
        "paints_text",
        "pushes_clip_scope",
        "font_size",
        "rise",
        "font_scale",
        "ascent",
        "descent",
        "advance_scale",
        "rotation",
        "scale_factor",
        "effective_font_size",
        "effective_font_height",
        "actual_text_span",
        "style",
        "space_width",
        "run_provenance",
    )

    decoder: FontDecoder
    given_paint: GlyphPaint | None
    identity_text_matrix: bool
    glyph_paint: GlyphPaint | None
    fill_color: tuple[float, ...] | None
    paints_text: bool
    pushes_clip_scope: bool
    font_size: float
    rise: float
    font_scale: float
    ascent: float
    descent: float
    advance_scale: float
    rotation: int
    scale_factor: float
    effective_font_size: float
    effective_font_height: float
    actual_text_span: MarkedContentEntry | None
    style: GlyphStyle | None
    space_width: float
    run_provenance: tuple[tuple[str, object], ...]


NATIVE_PATH_HANDLERS: dict[str, Callable[..., None]] = {
    "m": ContentInterpreter.op_m,
    "l": RecoveringTextState.op_l,
    "c": ContentInterpreter.op_c,
    "v": RecoveringTextState.op_v,
    "y": RecoveringTextState.op_y,
    "re": ContentInterpreter.op_re,
    "h": ContentInterpreter.op_h,
}


def applies_paths_natively(handlers: Mapping[str, OperationHandler], state: object) -> bool:
    kind = type(state)
    if (
        getattr(kind, "append_cubic_curve", None) is not RecoveringTextState.append_cubic_curve
        or getattr(kind, "as_floats", None) is not RecoveringTextState.as_floats
    ):
        return False
    for name, function in NATIVE_PATH_HANDLERS.items():
        if getattr(handlers.get(name), "__func__", None) is not function:
            return False
    return True


class CaptureStreamExecutor(ContentStreamExecutor):
    state: TextState
    _operator_names: frozenset[bytes] | None = None

    max_depth = 10

    def reject_reentry(self, frame: ContentStreamFrame) -> bool:
        return True

    def enter(self, frame: ContentStreamFrame) -> bool:
        if not super().enter(frame):
            return False
        self.state.shared_glyph_paint = None
        self.state.text_layout = None
        return True

    def exit(self, frame: ContentStreamFrame) -> None:
        try:
            super().exit(frame)
        finally:
            self.state.shared_glyph_paint = None
            self.state.text_layout = None

    def operator_names(self, table: Mapping[str, OperationHandler]) -> frozenset[bytes]:
        state = self.state
        if table is not state.default_handlers:
            return frozenset(name.encode("latin-1") for name in table)
        names = self._operator_names
        if names is None:
            names = self._operator_names = frozenset(name.encode("latin-1") for name in table)
        return names

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        handlers = state.operation_table()
        depth = frame.depth
        for name, operands in iter_content_operations(
            frame.lexer,
            recovery=state.recovery,
            is_operator=self.operator_names(handlers).__contains__,
            path_state=state if applies_paths_natively(handlers, state) else None,
        ):
            handler = handlers.get(name)
            if handler is None:
                continue
            if name not in GLYPH_PAINT_KEEPING_OPERATORS:
                state.shared_glyph_paint = None
                state.text_layout = None
            elif name in LINE_MOVING_OPERATORS:
                if (layout := state.text_layout) is not None:
                    layout.style = None
            elif name not in TEXT_LAYOUT_KEEPING_OPERATORS:
                state.text_layout = None
            child = handler(operands, depth)
            if child is not None:
                return child
        return None

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error


class TextState(RecoveringTextState):
    document: Any
    name_resolver: ObjectResolver
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_cluster_count: int
    lines: StrokeLineRows
    drawings: list[CapturedDrawing]
    inline_images: list[CapturedInlineImage]
    hidden_layers: frozenset[str]
    page_clip: Rectangle | None
    options: CaptureOptions
    clip_bbox: Rectangle | None
    layout_form_bbox: Rectangle | None
    layout_form_id: LayoutFormId
    capture_source: str
    stream_order: int
    sequence: int
    text_object_id: int
    text_boundaries: list[CapturedTextBoundary]
    capture_text_open: bool
    capture_text_frames: dict[int, tuple[bool, bool]]
    pending_line_break: bool
    group_alpha: float | None
    run_accumulator: RunAccumulator
    capture_graphics_stack: list[CaptureGraphicsSave]
    capture_marked_entries: dict[int, MarkedContentEntry]
    capture_frames: dict[int, tuple[Rectangle | None, LayoutFormId, bool]]
    capture_patterns: IdentityCache[PatternPaint | None]
    capture_image_sources: IdentityCache[tuple[ImageSource, float | None]]
    capture_shadings: IdentityCache[dict]
    capture_colors: BoundedDict[
        tuple[int, tuple[float, ...], str | None, BlackPointCompensation],
        tuple[object, tuple[float, ...]],
    ]
    capture_soft_masks: IdentityCache[CapturedSoftMask | None]
    capture_mask_resources: IdentityCache[PdfDict]
    capture_active_mask_groups: set[int]
    scale_cache: tuple[Matrix, float] | None
    shared_glyph_paint: GlyphPaint | None
    text_layout: TextLayout | None
    stream_executor: CaptureStreamExecutor
    stream_executor_type = CaptureStreamExecutor

    def __init__(
        self,
        document: Any,
        hidden_layers: frozenset[str] = frozenset(),
        page_clip: Rectangle | None = None,
        *,
        options: CaptureOptions = DEFAULT_CAPTURE,
    ):
        self.document = document
        self.name_resolver = document.resolver
        self.runs = []
        self.glyphs = []
        self.glyph_cluster_count = 0
        self.lines = StrokeLineRows()
        self.drawings = []
        self.inline_images = []
        self.hidden_layers = hidden_layers
        self.page_clip = page_clip
        self.options = options
        self.clip_bbox = None
        self.layout_form_bbox = None
        self.layout_form_id = None
        self.capture_source = "native_text"
        self.stream_order = -1
        self.sequence = 0
        self.text_object_id = 0
        self.text_boundaries = []
        self.capture_text_open = False
        self.capture_text_frames = {}
        self.pending_line_break = False
        self.group_alpha = None
        self.run_accumulator = RunAccumulator(self.runs)
        self.capture_graphics_stack = []
        self.capture_marked_entries = {}
        self.capture_frames = {}
        self.capture_patterns = IdentityCache()
        self.capture_image_sources = IdentityCache()
        self.capture_shadings = IdentityCache()
        self.capture_colors = BoundedDict(COLOR_CACHE_LIMIT)
        self.capture_soft_masks = IdentityCache(SOFT_MASK_CACHE_LIMIT)
        self.capture_mask_resources = IdentityCache()
        self.capture_active_mask_groups = set()
        self.capture_font_decoders = {}
        self.capture_font_companions = IdentityCache()

        def font_provider(font: PdfDict, resources: PdfDict) -> FontDecoder:
            return FontDecoder(
                font,
                ligature_overrides=detect_ligature_overrides(document, resources, font),
                raster_font_provider=getattr(document, "raster_font_provider", None),
                semantic_context=getattr(
                    document,
                    "font_semantic_context",
                    getattr(document.resolver, "semantic_context", None),
                ),
            )

        super().__init__(
            document.resolver,
            sink=self,
            font_provider=font_provider,
            lexer_factory=PdfLexer,
            semantic_context=getattr(document.resolver, "semantic_context", None),
        )

        self.recovery = CaptureRecovery()
        self.normalized_colors = BoundedDict(COLOR_CACHE_LIMIT)
        self.parsed_soft_masks = IdentityCache(soft_mask_cache_limit)
        self.scale_cache = None
        self.shared_glyph_paint = None
        self.text_layout = None

        self.graphics.font_size = 12.0
        self.graphics.fill_color = (0.0, 0.0, 0.0)
        self.graphics.stroke_color = (0.0, 0.0, 0.0)

    @staticmethod
    def as_float(value: Any) -> float:
        parsed = parse_float_strict(value, "invalid numeric operand", python_syntax=True)
        if not isfinite(parsed):
            raise ValueError("invalid numeric operand")
        return parsed

    @staticmethod
    def as_int(value: Any) -> int:
        return parse_int_strict(value, "invalid numeric operand", python_syntax=True)

    def capture_marks(self) -> CaptureMarks:
        return (
            len(self.runs),
            len(self.glyphs),
            len(self.drawings),
            len(self.inline_images),
            self.lines.count,
            len(self.text_boundaries),
        )

    def captured_program(self, since: CaptureMarks = NO_MARKS) -> CapturedProgram:
        runs, glyphs, drawings, inline_images, lines, text_boundaries = since
        return CapturedProgram(
            runs=tuple(self.runs[runs:]),
            glyphs=tuple(self.glyphs[glyphs:]),
            drawings=tuple(self.drawings[drawings:]),
            inline_images=tuple(self.inline_images[inline_images:]),
            lines=self.lines.since(lines),
            text_boundaries=tuple(self.text_boundaries[text_boundaries:]),
            options=self.options,
        )

    def release(self) -> None:
        del self.sink
        del self.stream_executor
        self.default_handlers.clear()

    def resolve_soft_mask(self, value: object) -> PdfSoftMask | None:
        mask = super().resolve_soft_mask(value)
        if mask is not None:
            scopes = self.capture_mask_resources
            if scopes.get(mask, default=MISSING) is MISSING:
                scopes.put(mask, self.resources)
        return mask

    def is_text_visible(self, text: str) -> bool:
        if not text:
            return False
        if self.text_paint_mode(check_colorants=False) in NON_PAINTING_RENDER_MODES:
            return False
        first_code = ord(text[0])
        if (first_code < 32 or 0xE000 <= first_code <= 0xF8FF) and is_garbage_text(text):
            return False
        if self.graphics.font_size < 0.1:
            return False
        return self.is_graphics_visible()

    def is_graphics_visible(self) -> bool:
        for entry in self.marked_content_stack:
            if entry.layer and entry.layer in self.hidden_layers:
                return False
        return True

    def graphics_scale(self) -> float:
        ctm = self.graphics.ctm
        cached = self.scale_cache
        if cached is not None and cached[0] is ctm:
            return cached[1]
        x_scale = hypot(ctm.a, ctm.b)
        y_scale = hypot(ctm.c, ctm.d)
        if x_scale == 0 and y_scale == 0:
            scale = 1.0
        elif x_scale == 0:
            scale = y_scale
        elif y_scale == 0:
            scale = x_scale
        else:
            scale = (x_scale + y_scale) * 0.5
        self.scale_cache = (ctm, scale)
        return scale

    def transformed_line_width(self) -> float:
        line_width = max(0.0, self.graphics.line_width)
        if line_width == 0:
            return 0.0
        return line_width * self.graphics_scale()

    def transformed_dash_pattern(self) -> tuple[list[float], float] | None:
        dash_pattern = self.graphics.dash_pattern
        if not dash_pattern:
            return None
        dash_array, phase = dash_pattern
        scale = self.graphics_scale()
        return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale

    def is_clipped_away(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        for clip in (self.clip_bbox, self.page_clip):
            if clip is None:
                continue
            if x1 <= clip[0] or x0 >= clip[2] or y1 <= clip[1] or y0 >= clip[3]:
                return True
        return False

    def update_pending_run(self, new_run: TextRun) -> None:
        if not self.is_clipped_away(new_run.x0, new_run.y0, new_run.x1, new_run.y1):
            self.run_accumulator.append(new_run)

    def glyph_paint(self, fill_color: tuple[float, ...] | None) -> GlyphPaint:
        return GlyphPaint(
            clip_bbox=self.clip_bbox,
            page_clip=self.page_clip,
            fill=fill_color,
            render_mode=self.text_paint_mode(),
            fill_opacity=self.graphics.fill_opacity,
            stroke_color=self.capture_color(stroke=True),
            stroke_opacity=self.graphics.stroke_opacity,
            line_width=self.transformed_line_width(),
            line_cap=self.graphics.line_cap,
            line_join=self.graphics.line_join,
            dash_pattern=self.transformed_dash_pattern(),
            blend_mode=self.graphics.blend_mode,
            group_alpha=self.group_alpha,
            alpha_is_shape=self.graphics.alpha_is_shape,
            graphics_soft_mask=self.capture_graphics_soft_mask(),
            clip_glyph=4 <= self.graphics.render_mode <= 7 and self.is_graphics_visible(),
        )

    def emit_actual_text_span(self, entry: MarkedContentEntry) -> None:
        actual_text = entry.actual_text
        captured = entry.run
        if actual_text is None or captured is None:
            return
        self.update_pending_run(
            captured.replace(
                text=normalize_extracted_text(actual_text),
                ink_bbox=captured.advance_bbox,
                provenance=(*captured.provenance, ("unicode_source", "actual_text")),
            )
        )
        self.glyphs.append(
            GlyphObservation(
                text=actual_text,
                ink_bbox=captured.advance_bbox,
                advance_bbox=captured.advance_bbox,
                seqno=captured.seqno,
                font_name=captured.font_name,
                font_size=captured.font_size,
                baseline=captured.baseline,
                rotation_angle=captured.rotation_angle,
                fill=captured.fill_color,
                visible=captured.visible,
                confidence=captured.confidence,
                unicode_source="actual_text",
                font_decoder=entry.font_decoder,
                effective_font_size=captured.font_size,
                effective_font_height=entry.effective_font_height,
                provenance=captured.provenance,
            )
        )

    def emit_clip_scope_push(self) -> None:
        if not self.capture_graphics_stack or self.capture_graphics_stack[-1].clip_scope_emitted:
            return
        self.capture_graphics_stack[-1].clip_scope_emitted = True
        self.drawings.append(marker_drawing("state-push", self.sequence))
        self.sequence += 1

    def new_text_layout(
        self,
        font_decoder: FontDecoder,
        glyph_paint: GlyphPaint | None,
        A: float,
        B: float,
        C: float,
        D: float,
    ) -> TextLayout:
        graphics = self.graphics
        layout = TextLayout()
        layout.decoder = font_decoder
        layout.given_paint = glyph_paint
        if glyph_paint is None and not font_decoder.is_type3 and graphics.soft_mask is None:
            glyph_paint = self.shared_glyph_paint
            if glyph_paint is None:
                glyph_paint = self.shared_glyph_paint = self.glyph_paint(
                    self.capture_color(stroke=False)
                )
        layout.glyph_paint = glyph_paint
        graphics_visible = self.is_graphics_visible()
        fs = graphics.font_size
        layout.paints_text = (
            self.text_paint_mode(check_colorants=False) not in NON_PAINTING_RENDER_MODES
            and not fs < 0.1
            and graphics_visible
        )
        layout.pushes_clip_scope = 4 <= graphics.render_mode <= 7 and graphics_visible
        rise = graphics.rise
        font_scale = fs / 1000.0
        metrics_decoder: FontDecoder | None = graphics.current_decoder  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        layout.font_size = fs
        layout.rise = rise
        layout.font_scale = font_scale
        layout.ascent = metrics_decoder.ascent * font_scale if metrics_decoder is not None else 0.0
        layout.descent = (
            metrics_decoder.descent * font_scale if metrics_decoder is not None else 0.0
        )
        layout.advance_scale = fs * graphics.horizontal_scale / 100000.0
        layout.space_width = (
            metrics_decoder.glyph_width(32) * fs * 0.001 if metrics_decoder is not None else 0.0
        )
        layout.rotation = detect_rotation_from_linear(A, B, C, D)
        if font_decoder.is_vertical:
            layout.scale_factor = hypot(C, D)
            layout.effective_font_height = fs * hypot(A, B)
        else:
            layout.scale_factor = hypot(A, B)
            layout.effective_font_height = fs * hypot(C, D)
        layout.effective_font_size = fs * layout.scale_factor
        layout.fill_color = (
            self.capture_color(stroke=False) if glyph_paint is None else glyph_paint.fill
        )
        layout.actual_text_span = self.current_capture_actual_text_span()
        layout.style = None
        mcid = self.current_marked_content_mcid()
        layout.run_provenance = (
            ("font_name", graphics.current_font),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("text_render_mode", graphics.render_mode),
            ("font_size", fs),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            *((("mcid", mcid),) if mcid is not None else ()),
        )
        return layout

    def show_text(
        self,
        state: object,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[DecodedFontGlyph, ...],
        decoder: FontService,
        adv_x: float,
        adv_y: float,
        *,
        glyph_paint: GlyphPaint | None = None,
    ) -> None:
        font_decoder: FontDecoder = decoder  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        decoded_glyphs: tuple[DecodedGlyph, ...] = glyphs  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        text_matrix = self.text_matrix
        ctm = self.graphics.ctm
        combined = text_matrix.multiply(ctm)
        A, B, C, D, E, F = combined
        te, tf = text_matrix.e, text_matrix.f
        identity_text_matrix = text_matrix == IDENTITY_MATRIX
        layout = self.text_layout
        if (
            layout is None
            or layout.given_paint is not glyph_paint
            or layout.decoder is not font_decoder
            or layout.identity_text_matrix is not identity_text_matrix
        ):
            layout = self.new_text_layout(font_decoder, glyph_paint, A, B, C, D)
            layout.identity_text_matrix = identity_text_matrix
            self.text_layout = layout

        visible = False
        if text and layout.paints_text:
            first_code = ord(text[0])
            visible = not (
                (first_code < 32 or 0xE000 <= first_code <= 0xF8FF) and is_garbage_text(text)
            )
        if layout.pushes_clip_scope:
            self.emit_clip_scope_push()

        fs = layout.font_size
        rise = layout.rise
        ascent = layout.ascent
        descent = layout.descent
        rot = layout.rotation
        scale_factor = layout.scale_factor
        effective_font_size = layout.effective_font_size
        effective_font_height = layout.effective_font_height
        fill_color = layout.fill_color
        seqno = self.sequence
        actual_text_span = layout.actual_text_span
        captured: GlyphCapture | None = None
        if actual_text_span is None:
            style = layout.style
            if style is None:
                graphics = self.graphics
                style = layout.style = glyph_style(
                    self.glyph_paint(fill_color)
                    if layout.glyph_paint is None
                    else layout.glyph_paint,
                    font_decoder,
                    fs,
                    rot,
                    effective_font_size,
                    effective_font_height,
                    (
                        ("source", self.capture_source),
                        ("stream_order", self.stream_order),
                        ("xobject_depth", self.xobject_depth),
                        ("clip_bbox", self.clip_bbox),
                        ("layout_form_bbox", self.layout_form_bbox),
                        ("layout_form_id", self.layout_form_id),
                        ("text_matrix", (A, B, C, D)),
                        ("text_render_mode", graphics.render_mode),
                        ("line_matrix_origin", (self.line_matrix.e, self.line_matrix.f)),
                        ("horizontal_scale", graphics.horizontal_scale),
                        ("char_space", graphics.char_space),
                        ("text_rise", rise),
                    ),
                    self.text_object_id,
                )
            graphics = self.graphics
            captured = capture_glyphs(
                text,
                decoded_glyphs,
                font_decoder,
                (E, F, A, B, C, D),
                fs,
                layout.font_scale,
                ascent,
                descent,
                layout.advance_scale,
                graphics.char_space,
                graphics.word_space,
                graphics.horizontal_scale,
                rise,
                style,
                self.clip_bbox,
                self.page_clip,
                visible,
                graphics.current_font,
                seqno,
                self.glyph_cluster_count,
                self.options,
            )
            self.glyphs.extend(captured.glyphs)
            self.glyph_cluster_count += captured.cluster_count
            if not self.options.text_runs:
                self.sequence = seqno + 1
                return

        if font_decoder.is_vertical:
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

        effective_space_width = layout.space_width * scale_factor
        baseline = (
            E,
            F,
            E + adv_x * A + adv_y * C,
            F + adv_x * B + adv_y * D,
        )
        provenance = (("source", self.capture_source), ("seqno", seqno), *layout.run_provenance)
        advance_bbox = (x0, y0, x1, y1)

        new_run = TextRun(
            normalize_extracted_text(text),
            x0,
            y0,
            x1,
            y1,
            te,
            tf,
            effective_font_size,
            effective_space_width,
            seqno,
            self.stream_order,
            self.xobject_depth,
            self.graphics.current_font,
            font_decoder.is_vertical,
            rot,
            visible,
            True,
            self.pending_line_break,
            seqno,
            fill_color,
            advance_bbox,
            advance_bbox,
            baseline,
            provenance,
            None,
        )
        if actual_text_span is not None:
            new_run.confidence = 1.0
            actual_text_span.add_run(
                new_run,
                font_decoder=font_decoder,
                effective_font_height=effective_font_height,
            )
        else:
            assert captured is not None
            new_run.glyph_clusters = tuple(captured.clusters)
            geometry = captured.geometry
            if geometry.started:
                new_run.advance_bbox = geometry.advance
                new_run.ink_bbox = geometry.ink
                new_run.confidence = geometry.confidence
            self.update_pending_run(new_run)

        self.sequence = seqno + 1

    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        if not source.ops:
            return
        if not self.is_graphics_visible():
            return
        graphics = self.graphics
        fill_space = graphics.fill_space
        stroke_space = graphics.stroke_space
        fills = kind != "stroke" and not (
            fill_space.kind == "Pattern" and graphics.fill_pattern is None
        )
        strokes = kind != "fill" and not (
            stroke_space.kind == "Pattern" and graphics.stroke_pattern is None
        )
        if not fills and not strokes:
            return
        painted: PaintedDrawingKind = (
            "fillstroke" if fills and strokes else "fill" if fills else "stroke"
        )
        fill_paints = color_space_paints(fill_space)
        stroke_paints = color_space_paints(stroke_space)

        ctm = graphics.ctm
        line_width = self.transformed_line_width()
        path = flatten_path(source, None if ctm == IDENTITY_MATRIX else ctm, self.lines, line_width)
        if not path.has_segments():
            return
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.capture_color(stroke=False),
                fill_pattern=self.capture_pattern(graphics.fill_pattern)
                if fill_paints and fills
                else None,
                fill_opacity=graphics.fill_opacity,
                stroke_color=self.capture_color(stroke=True),
                stroke_pattern=self.capture_pattern(graphics.stroke_pattern)
                if stroke_paints and strokes
                else None,
                stroke_opacity=graphics.stroke_opacity,
                line_width=line_width,
                line_cap=graphics.line_cap,
                line_join=graphics.line_join,
                dash_pattern=self.transformed_dash_pattern() if graphics.dash_pattern else None,
                fill_rule=fill_rule,
                blend_mode=graphics.blend_mode,
                soft_mask_alpha=self.group_alpha,
                alpha_is_shape=graphics.alpha_is_shape,
                kind=painted,
                graphics_soft_mask=self.capture_graphics_soft_mask()
                if graphics.soft_mask is not None
                else None,
                fill_paints=fill_paints,
                stroke_paints=stroke_paints,
                path=path,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
            )
        )
        self.sequence += 1

    def clip_path(self, state: object, source: PdfPath, fill_rule: str) -> None:
        path = flatten_path(source, self.graphics.ctm)
        if not path.has_segments():
            return
        clip_bbox = path.bbox()
        if clip_bbox is not None:
            self.clip_bbox = intersect_bbox(self.clip_bbox, clip_bbox)
        if self.is_graphics_visible():
            self.emit_clip_scope_push()
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    line_width=0.0,
                    line_cap=self.graphics.line_cap,
                    line_join=self.graphics.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    kind="clip",
                    path=path,
                )
            )
            self.sequence += 1

    def captured_image_source(self, xobj: PdfStream) -> tuple[ImageSource, float | None]:
        rendering = self.graphics.color_rendering
        cache = self.capture_image_sources
        cached = cache.get(xobj, rendering)
        if cached is not None:
            return cached
        return cache.put(
            xobj,
            image_source_from_stream(xobj, self.resolver, color_rendering=rendering),
            rendering,
        )

    def paint_image(self, state: object, xobj: PdfStream) -> None:
        xobj_dict = xobj.dictionary
        if self.is_graphics_visible():
            image_is_stencil = self.resolver.resolve(xobj_dict.get("ImageMask")) is True
            if image_is_stencil and self.initial_pattern(stroke=False):
                return
            width = self.resolver.resolve_int(xobj_dict.get("Width")) or 0
            height = self.resolver.resolve_int(xobj_dict.get("Height")) or 0
            bbox = None
            quad = None
            if width > 0 and height > 0:
                bbox, quad = unit_square_placement(self.graphics.ctm)
            source, smask_alpha = self.captured_image_source(xobj)
            paints = (
                color_space_paints(self.graphics.fill_space)
                if image_is_stencil
                else raw_color_space_paints(source.dictionary.get("ColorSpace"))
            )
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=self.capture_color(stroke=False) if image_is_stencil else None,
                    fill_opacity=self.graphics.fill_opacity,
                    blend_mode=self.graphics.blend_mode,
                    dash_pattern=self.transformed_dash_pattern(),
                    soft_mask_alpha=smask_alpha,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    kind="image",
                    graphics_soft_mask=None
                    if image_overrides_graphics_soft_mask(source)
                    else self.capture_graphics_soft_mask(),
                    paints=paints,
                    image_source=source,
                    raw_data=xobj.raw_data,
                    dictionary=dict(xobj_dict),
                    image_clip=self.clip_bbox,
                    items=[("quad", quad)] if quad is not None else [],
                    bbox=bbox,
                    stream_order=self.stream_order,
                    xobject_depth=self.xobject_depth,
                )
            )
            self.sequence += 1

    def paint_inline_image(self, state: object, image: InlineImage) -> None:
        if self.is_graphics_visible():
            dictionary = dict(image.dictionary)
            if dictionary.get("ImageMask") is True and self.initial_pattern(stroke=False):
                return
            data = getattr(image, "data", b"")
            color_name = recover_pdf_name(dictionary.get("ColorSpace"))
            if color_name is not None:
                color_resource = self.resolver.deep_resolve(
                    self.lookup_page_resource("ColorSpace", color_name)
                )
                if color_resource is not None:
                    dictionary[PdfName.of("ColorSpace")] = color_resource
            source, _ = image_source_from_stream(
                PdfStream(raw_data=data, dictionary=dictionary),
                self.resolver,
                color_rendering=self.graphics.color_rendering,
            )
            paints = (
                color_space_paints(self.graphics.fill_space)
                if dictionary.get("ImageMask") is True
                else raw_color_space_paints(source.dictionary.get("ColorSpace"))
            )
            self.inline_images.append(
                CapturedInlineImage(
                    seqno=self.sequence,
                    dictionary=dictionary,
                    data=data,
                    image_source=source,
                    image_clip=self.clip_bbox,
                    ctm=self.graphics.ctm,
                    graphics_soft_mask=None
                    if image_overrides_graphics_soft_mask(source)
                    else self.capture_graphics_soft_mask(),
                    xobject_depth=self.xobject_depth,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    stream_order=self.stream_order,
                    fill=self.capture_color(stroke=False)
                    if dictionary.get("ImageMask") is True
                    else None,
                    fill_opacity=self.graphics.fill_opacity,
                    paints=paints,
                )
            )
            self.sequence += 1

    def paint_shading(self, state: object, shading: PdfDict) -> None:
        if not self.is_graphics_visible():
            return
        dictionary = self.capture_shading_dictionary(shading)
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.capture_color(stroke=False),
                fill_opacity=self.graphics.fill_opacity,
                stroke_color=self.capture_color(stroke=True),
                stroke_opacity=self.graphics.stroke_opacity,
                line_width=self.graphics.line_width,
                line_cap=self.graphics.line_cap,
                line_join=self.graphics.line_join,
                dash_pattern=self.transformed_dash_pattern(),
                blend_mode=self.graphics.blend_mode,
                soft_mask_alpha=self.group_alpha,
                alpha_is_shape=self.graphics.alpha_is_shape,
                kind="shading",
                graphics_soft_mask=self.capture_graphics_soft_mask(),
                paints=raw_color_space_paints(dictionary.get("ColorSpace")),
                color_rendering=self.graphics.color_rendering,
                items=[],
                dictionary=dictionary,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
            )
        )
        self.sequence += 1

    def text_boundary(self, state: object, kind: str) -> None:
        if kind in {"type3-glyph-begin", "type3-glyph-end"}:
            self.text_boundaries.append(
                CapturedTextBoundary(
                    self.sequence, "glyph-begin" if kind == "type3-glyph-begin" else "glyph-end"
                )
            )
            return
        match kind:
            case "begin":
                if self.capture_text_open:
                    self.text_boundaries.append(CapturedTextBoundary(self.sequence, "end"))
                self.text_boundaries.append(
                    CapturedTextBoundary(self.sequence, "begin", self.graphics.text_knockout)
                )
                self.capture_text_open = True
                self.text_object_id += 1
                self.run_accumulator.flush()
            case "end":
                if self.capture_text_open:
                    self.text_boundaries.append(CapturedTextBoundary(self.sequence, "end"))
                self.capture_text_open = False
                self.run_accumulator.flush()
            case "shown":
                self.pending_line_break = False
            case "quoted":
                self.pending_line_break = True
            case _:
                self.run_accumulator.flush()

    def current_capture_actual_text_span(self) -> MarkedContentEntry | None:
        entry = self.current_actual_text_span()
        if entry is None:
            return None
        key = id(entry)
        captured = self.capture_marked_entries.get(key)
        if captured is None:
            captured = MarkedContentEntry(entry.layer, entry.actual_text, entry.mcid)
            self.capture_marked_entries[key] = captured
        return captured

    def end_marked_content(self, state: object, entry: SemanticMarkedContentEntry) -> None:
        captured = self.capture_marked_entries.pop(id(entry), None)
        if captured is not None:
            self.emit_actual_text_span(captured)

    def save_graphics(self, state: object) -> None:
        self.capture_graphics_stack.append(CaptureGraphicsSave(self.clip_bbox, self.group_alpha))

    def restore_graphics(self, state: object) -> None:
        saved = self.capture_graphics_stack.pop()
        self.clip_bbox = saved.clip_bbox
        self.group_alpha = saved.group_alpha
        if saved.clip_scope_emitted:
            self.drawings.append(marker_drawing("state-pop", self.sequence))
            self.sequence += 1

    def enter_stream(self, state: object, frame: ContentStreamFrame) -> None:
        self.capture_text_frames[id(frame)] = (self.capture_text_open, False)
        self.capture_text_open = False
        self.capture_frames[id(frame)] = (
            self.layout_form_bbox,
            self.layout_form_id,
            self.pending_line_break,
        )
        if frame.form_bbox is not None:
            x0, y0, x1, y1 = frame.form_bbox
            clip_path = CapturedPath()
            if x1 > x0 and y1 > y0:
                clip_path.subpaths.append(
                    CapturedSubpath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)
                )
                clip_path = clip_path.transformed(frame.ctm)
            self.drawings.append(
                CapturedDrawing(
                    kind="scope-begin",
                    seqno=self.sequence,
                    path=clip_path,
                    fill=None,
                    fill_opacity=None,
                    line_width=0.0,
                )
            )
            self.sequence += 1
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-begin",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    group_isolated=frame.group_isolated,
                    group_knockout=frame.group_knockout,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    graphics_soft_mask=self.capture_graphics_soft_mask(),
                )
            )
            self.sequence += 1
            self.group_alpha = None
        self.text_boundaries.append(CapturedTextBoundary(self.sequence, "stream-begin"))
        previous_text_open, _ = self.capture_text_frames[id(frame)]
        self.capture_text_frames[id(frame)] = (previous_text_open, True)
        layout_bbox = None
        raw_bbox = frame.form_bbox_operand
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            x, y, w, h = (
                self.resolver.resolve_float(value, default=None) for value in raw_bbox[:4]
            )
            if x is not None and y is not None and w is not None and h is not None:
                layout_bbox = transform_bbox((x, y, x + w, y + h), frame.ctm)
        self.layout_form_bbox = layout_bbox
        if frame.is_form:
            self.layout_form_id = (*(self.layout_form_id or ()), (frame.source_key, layout_bbox))
        else:
            self.layout_form_id = None
        if frame.clip_bbox is not None:
            self.clip_bbox = intersect_bbox(self.clip_bbox, frame.clip_bbox)
        self.pending_line_break = False
        self.stream_order += 1

    def exit_stream(self, state: object, frame: ContentStreamFrame) -> None:
        if id(frame) in self.capture_text_frames:
            previous_text_open, started = self.capture_text_frames.pop(id(frame))
            if started and self.capture_text_open:
                self.text_boundaries.append(CapturedTextBoundary(self.sequence, "end"))
            if started:
                self.text_boundaries.append(CapturedTextBoundary(self.sequence, "stream-end"))
            self.capture_text_open = previous_text_open
        old = self.capture_frames.pop(id(frame), None)
        if old is not None:
            self.layout_form_bbox, self.layout_form_id, self.pending_line_break = old
        if self.capture_marked_entries:
            active_entries = {id(entry) for entry in self.marked_content_stack}
            self.capture_marked_entries = {
                key: value
                for key, value in self.capture_marked_entries.items()
                if key in active_entries
            }
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-end",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    group_isolated=frame.group_isolated,
                    group_knockout=frame.group_knockout,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                )
            )
            self.sequence += 1
        if old is not None and frame.form_bbox is not None:
            self.drawings.append(marker_drawing("scope-end", self.sequence))
            self.sequence += 1

    def initial_pattern(self, *, stroke: bool) -> bool:
        space = self.graphics.stroke_space if stroke else self.graphics.fill_space
        pattern = self.graphics.stroke_pattern if stroke else self.graphics.fill_pattern
        return space.kind == "Pattern" and pattern is None

    def text_paint_mode(self, *, check_colorants: bool = True) -> int:
        mode = self.graphics.render_mode
        if mode not in range(8):
            return mode
        fills = mode in {0, 2, 4, 6} and not self.initial_pattern(stroke=False)
        strokes = mode in {1, 2, 5, 6} and not self.initial_pattern(stroke=True)
        if check_colorants:
            fills = fills and color_space_paints(self.graphics.fill_space)
            strokes = strokes and color_space_paints(self.graphics.stroke_space)
        paint = 2 if fills and strokes else 0 if fills else 1 if strokes else 3
        return paint + (4 if mode >= 4 else 0)

    def capture_color(self, *, stroke: bool) -> tuple[float, ...] | None:
        graphics = self.graphics
        if stroke:
            color = graphics.stroke_color
            spec = graphics.stroke_space
        else:
            color = graphics.fill_color
            spec = graphics.fill_space
        if color is None or spec is None or not color_space_paints(spec):
            return color
        intent = graphics.render_intent
        black_point = graphics.black_point_compensation
        key = (id(spec), color, intent, black_point)
        previous = self.capture_colors.get(key)
        if previous is not None:
            return previous[1]
        converted = color_operands_to_srgb(spec, list(color), rendering=graphics.color_rendering)
        result = converted if converted is not None else color
        self.capture_colors.put(key, (spec, result))
        return result

    def capture_shading_dictionary(self, dictionary: dict) -> dict:
        cached = self.capture_shadings.get(dictionary)
        if cached is not None:
            return cached
        captured = {
            key: self.resolver.deep_resolve(value)
            if str(key) in {"ColorSpace", "Function", "Coords", "Domain", "Extend", "BBox"}
            else value
            for key, value in dictionary.items()
        }
        return self.capture_shadings.put(dictionary, captured)

    def nested_capture_state(self) -> TextState:
        nested = TextState(
            self.document,
            hidden_layers=self.hidden_layers,
            options=self.options,
        )
        nested.parsed_soft_masks = self.parsed_soft_masks
        nested.capture_soft_masks = self.capture_soft_masks
        nested.capture_mask_resources = self.capture_mask_resources
        nested.capture_active_mask_groups = self.capture_active_mask_groups
        nested.capture_image_sources = self.capture_image_sources
        nested.capture_font_decoders = self.capture_font_decoders
        nested.capture_font_companions = self.capture_font_companions
        nested.capture_colors = self.capture_colors
        nested.capture_shadings = self.capture_shadings
        return nested

    def capture_pattern(self, pattern: object) -> PatternPaint | None:
        if pattern is None:
            return None
        rendering = self.graphics.color_rendering
        initial_alpha_is_shape = (
            pattern.alpha_is_shape if isinstance(pattern, PdfTilingPattern) else False
        )
        initial_text_knockout = (
            pattern.text_knockout if isinstance(pattern, PdfTilingPattern) else True
        )
        key = (rendering, initial_alpha_is_shape, initial_text_knockout)
        cached = self.capture_patterns.get(pattern, *key, default=MISSING)
        if cached is not MISSING:
            return cached
        result: PatternPaint | None = None
        if isinstance(pattern, PdfShadingPattern):
            if pattern.extgstate is not None:
                values = {
                    str(key): self.resolver.resolve(value)
                    for key, value in pattern.extgstate.items()
                    if str(key) in {"RI", "UseBlackPtComp"}
                }
                with suppress(ValueError):
                    rendering = override_color_rendering(values, rendering)
            result = ShadingPattern(
                self.capture_shading_dictionary(pattern.dictionary), color_rendering=rendering
            )
        elif isinstance(pattern, PdfTilingPattern):
            nested = self.nested_capture_state()
            try:
                result = self.capture_tiling_pattern(
                    nested, pattern, rendering, initial_alpha_is_shape, initial_text_knockout
                )
            finally:
                nested.release()
        return self.capture_patterns.put(pattern, result, *key)

    def capture_tiling_pattern(
        self,
        nested: TextState,
        pattern: PdfTilingPattern,
        rendering: ColorRendering,
        initial_alpha_is_shape: bool,
        initial_text_knockout: bool,
    ) -> TilingPattern | None:
        nested.graphics.render_intent = self.graphics.render_intent
        nested.graphics.black_point_compensation = self.graphics.black_point_compensation
        nested.graphics.alpha_is_shape = initial_alpha_is_shape
        nested.graphics.text_knockout = initial_text_knockout
        try:
            nested.stream_executor.consume(pattern.stream, pattern.resources, pattern.matrix, 0)
        except Exception:
            return None
        if pattern.paint_type == 2:
            base_color = pattern.base_color
            if base_color is not None and pattern.base_color_spec is not None:
                converted = color_operands_to_srgb(
                    pattern.base_color_spec, base_color, rendering=rendering
                )
                if converted is not None:
                    base_color = converted
            for drawing in nested.drawings:
                if drawing.kind in {"fill", "fillstroke"}:
                    drawing.fill = base_color
                if drawing.kind in {"stroke", "fillstroke"}:
                    drawing.stroke_color = base_color
            for glyph in nested.glyphs:
                glyph.style = replace(glyph.style, fill=base_color, stroke_color=base_color)
        return TilingPattern(
            pattern.bbox,
            pattern.x_step,
            pattern.y_step,
            CapturedProgram(
                glyphs=tuple(glyph for glyph in nested.glyphs if glyph.has_paint),
                drawings=tuple(nested.drawings),
                inline_images=tuple(nested.inline_images),
                text_boundaries=tuple(nested.text_boundaries),
                options=nested.options,
            ),
        )

    def capture_graphics_soft_mask(self) -> CapturedSoftMask | None:
        mask = self.graphics.soft_mask
        if mask is None or not self.options.render_details:
            return None
        key = tuple(state_key(getattr(self.graphics, name)) for name in MASK_KEYED_FIELDS)
        cached = self.capture_soft_masks.get(mask, key, default=MISSING)
        if cached is not MISSING:
            return cached
        graphics = copy(self.graphics)
        graphics.ctm = mask.ctm
        graphics.soft_mask = None
        graphics.fill_opacity = graphics.stroke_opacity = 1.0
        graphics.blend_mode = None
        self.capture_soft_masks.put(mask, None, key)
        group_key = id(mask.group)
        if mask.subtype != "Alpha" or group_key in self.capture_active_mask_groups:
            return None
        if len(self.capture_active_mask_groups) >= 10:
            return None
        self.capture_active_mask_groups.add(group_key)
        nested: TextState | None = None
        try:
            nested = self.nested_capture_state()
            nested.graphics = copy(graphics)
            scope = self.capture_mask_resources.get(mask, default=MISSING)
            nested.resources = self.resources if scope is MISSING else scope
            frame = nested.append_form_xobject(mask.group, 0)
            if frame is None:
                return None
            nested.stream_executor.consume_frame(frame)
            nested.run_accumulator.flush()
            if not nested.text_boundaries:
                return None
            return self.capture_soft_masks.put(
                mask, CapturedSoftMask(nested.captured_program(), mask.transfer), key
            )
        except PdfParseError, TypeError, ValueError, ArithmeticError:
            return None
        finally:
            self.capture_active_mask_groups.remove(group_key)
            if nested is not None:
                nested.release()

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        resolver = self.name_resolver
        if allow_text:
            return resolver.resolve_name_or_text(value)
        return resolver.resolve_name_like_value(value)


def image_source_from_stream(
    stream: PdfStream,
    resolver: PdfValueResolver,
    *,
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[ImageSource, float | None]:
    source = resolve_image_source(
        stream,
        resolver,
        semantic_context=getattr(resolver, "semantic_context", None),
        color_rendering=color_rendering,
    )
    return source, soft_mask_mean_alpha(source)


def soft_mask_mean_alpha(source: ImageSource) -> float | None:
    mask = source.soft_mask
    if mask is None:
        return None
    try:
        raster = decode_soft_mask(source, mask)
    except PdfParseError, TypeError, ValueError, ArithmeticError:
        return None
    if raster is None or not raster.array.size:
        return None
    samples = raster.array[:, :, 0]
    return int(samples.sum(dtype=numpy.uint64)) / (255.0 * samples.size)


def flatten_path(
    source: PdfPath,
    matrix: Matrix | None,
    lines: StrokeLineRows | None = None,
    line_width: float = 0.0,
) -> CapturedPath:
    xs, ys, spans, bbox, has_segments = flatten_path_commands(
        source.ops,
        source.coords,
        matrix,
        hypot,
        None if lines is None else lines.rows,
        line_width,
    )
    return CapturedPath.deferred_flattened(xs, ys, spans, bbox, has_segments)


class StrokeLineRows:
    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: array[float] = array("d")

    @property
    def count(self) -> int:
        return len(self.rows) // 5

    def since(self, mark: int) -> CapturedLines:
        if mark >= self.count:
            return EMPTY_LINES
        table = numpy.frombuffer(self.rows, dtype=numpy.float64)
        return CapturedLines.from_array(table[mark * 5 :].reshape(-1, 5).copy())


GRAPHICS_STATE_FIELDS = GraphicsState.__fields__

MASK_OVERRIDDEN_FIELDS = frozenset(
    {"ctm", "soft_mask", "fill_opacity", "stroke_opacity", "blend_mode"}
)
MASK_KEYED_FIELDS = tuple(
    name for name in GRAPHICS_STATE_FIELDS if name not in MASK_OVERRIDDEN_FIELDS
)


def state_key(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(state_key(part) for part in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    return ("identity", id(value))
