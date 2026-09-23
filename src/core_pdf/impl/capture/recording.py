# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from contextlib import suppress
from copy import copy
from dataclasses import dataclass
from math import ceil, hypot, isfinite
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy

from core_pdf.impl.capture.glyphs import (
    GlyphCapture,
    GlyphPaint,
    TextBasis,
    TextGeometry,
    capture_glyphs,
)
from core_pdf.impl.capture.program import CapturedProgram
from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
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
from core_pdf.impl.capture.recovery import CaptureRecovery, iter_content_operations
from core_pdf.impl.capture.text_runs import (
    RunAccumulator,
    is_garbage_text,
)
from core_pdf.impl.capture.tolerant_state import COLOR_CACHE_LIMIT, RecoveringTextState
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.fonts.ligatures import detect_ligature_overrides
from core_pdf.impl.graphics.color import color_operands_to_srgb
from core_pdf.impl.graphics.color_spec import raw_color_space_paints
from core_pdf.impl.graphics.soft_masks import image_overrides_graphics_soft_mask
from core_pdf.impl.model.geometry import (
    extend_baseline,
    intersect_bbox,
    transform_bbox,
    union_bbox,
)
from core_pdf.impl.model.glyphs import GlyphObservation, min_optional_confidence
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.model.text import normalize_extracted_text
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.runtime.scalars import parse_float_strict, parse_int_strict
from core_pdf.impl.types import (
    PdfName,
    Rectangle,
)
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import NON_PAINTING_RENDER_MODES, GraphicsState, PdfPath
from core_pdf_spec.s_07_content.model import (
    MarkedContentEntry as SemanticMarkedContentEntry,
)
from core_pdf_spec.s_07_content.model import ShadingPattern as PdfShadingPattern
from core_pdf_spec.s_07_content.model import TilingPattern as PdfTilingPattern
from core_pdf_spec.s_07_content.streams import (
    ContentStreamExecutor,
    ContentStreamFrame,
    StreamKey,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver
from core_pdf_spec.s_08_graphics.color import color_space_paints
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
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
    """A marked-content span's ActualText, and the one run it collapses to."""

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
        captured.x0 = min(captured.x0, run.x0)
        captured.y0 = min(captured.y0, run.y0)
        captured.x1 = max(captured.x1, run.x1)
        captured.y1 = max(captured.y1, run.y1)
        captured.advance_bbox = cast(Rectangle, union_bbox(captured.advance_bbox, run.advance_bbox))
        captured.baseline = extend_baseline(captured.baseline, run.baseline)
        captured.confidence = min_optional_confidence(captured.confidence, run.confidence)


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


# The product lists a CapturedProgram is cut from, in the order its fields
# take them. capture_marks/captured_program are the only places that need to
# know the set, so adding a product means touching this tuple and those two.
CaptureMarks: TypeAlias = tuple[int, int, int, int, int, int]
NO_MARKS: CaptureMarks = (0, 0, 0, 0, 0, 0)


class CaptureStreamExecutor(ContentStreamExecutor):
    state: TextState
    _operator_names: frozenset[bytes] | None = None

    def is_reentrant(self, stream: PdfStream, stream_key: StreamKey | None, depth: int) -> bool:
        return depth > 10 or (stream_key or self.execution_key(stream)) in self.active_streams

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
        if self.is_reentrant(stream, stream_key, depth):
            return None
        return super().queue(
            stream,
            resources,
            ctm,
            depth,
            clip_bbox=clip_bbox,
            form_bbox_operand=form_bbox_operand,
            group_alpha=group_alpha,
            stream_key=stream_key,
        )

    def enter(self, frame: ContentStreamFrame) -> bool:
        if self.is_reentrant(frame.stream, frame.stream_key, frame.depth):
            return False
        return super().enter(frame)

    def operator_names(self) -> frozenset[bytes]:
        # Encoding all 71 handler names costs 6us, and iter_content_operations
        # wants the set once per frame -- a page of form XObjects, tiling
        # patterns and soft masks has thousands. The handler tables are built
        # in the interpreter's __init__ and nothing mutates them afterwards.
        names = self._operator_names
        if names is None:
            state = self.state
            names = self._operator_names = frozenset(
                name.encode("latin-1")
                for name in (*state.default_handlers, *state.operator_overrides)
            )
        return names

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        for name, operands in iter_content_operations(
            frame.lexer,
            recovery=state.recovery,
            is_operator=self.operator_names().__contains__,
        ):
            child = state.execute_operation(name, operands, frame.depth)
            if child is not None:
                return child
        return None

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error


class TextState(RecoveringTextState):
    document: Any
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_cluster_count: int
    lines: list[CapturedLine]
    drawings: list[CapturedDrawing]
    inline_images: list[CapturedInlineImage]
    hidden_layers: frozenset[str]
    page_clip: Rectangle | None
    capture_ink_bounds: bool
    capture_text_runs: bool
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
    capture_patterns: dict[
        tuple[int, ColorRendering, bool, bool], tuple[object, PatternPaint | None]
    ]
    capture_image_sources: dict[
        tuple[int, ColorRendering], tuple[PdfStream, ImageSource, float | None]
    ]
    capture_colors: dict[
        tuple[int, tuple[float, ...], ColorRendering], tuple[object, tuple[float, ...] | None]
    ]
    capture_soft_masks: dict[
        tuple[int, tuple[object, ...]], tuple[PdfSoftMask, GraphicsState, CapturedSoftMask | None]
    ]
    capture_mask_resources: dict[int, tuple[PdfSoftMask, PdfDict]]
    capture_active_mask_groups: set[int]
    scale_cache: tuple[Matrix, float] | None
    stream_executor: CaptureStreamExecutor
    stream_executor_type = CaptureStreamExecutor

    def __init__(
        self,
        document: Any,
        hidden_layers: frozenset[str] = frozenset(),
        page_clip: Rectangle | None = None,
        *,
        capture_ink_bounds: bool = True,
        capture_text_runs: bool = True,
    ):
        self.document = document
        self.runs = []
        self.glyphs = []
        self.glyph_cluster_count = 0
        self.lines = []
        self.drawings = []
        self.inline_images = []
        self.hidden_layers = hidden_layers
        self.page_clip = page_clip
        self.capture_ink_bounds = capture_ink_bounds
        self.capture_text_runs = capture_text_runs
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
        self.capture_patterns = {}
        self.capture_image_sources = {}
        self.capture_colors = {}
        self.capture_soft_masks = {}
        self.capture_mask_resources = {}
        self.capture_active_mask_groups = set()
        self.capture_font_decoders = {}
        self.capture_font_companions = {}

        def font_provider(font: dict[str, Any], resources: dict[str, Any]) -> FontDecoder:
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
        self.normalized_colors = {}
        self.parsed_soft_masks = {}
        self.scale_cache = None

        self.graphics.font_size = 12.0
        self.graphics.fill_color = (0.0, 0.0, 0.0)
        self.graphics.stroke_color = (0.0, 0.0, 0.0)

    @staticmethod
    def as_float(value: Any) -> float:
        parsed = parse_float_strict(value, "invalid numeric operand")
        if not isfinite(parsed):
            raise ValueError("invalid numeric operand")
        return parsed

    @staticmethod
    def as_int(value: Any) -> int:
        return parse_int_strict(value, "invalid numeric operand")

    def capture_marks(self) -> CaptureMarks:
        """Where each product list stands, to cut a later program from."""
        return (
            len(self.runs),
            len(self.glyphs),
            len(self.drawings),
            len(self.inline_images),
            len(self.lines),
            len(self.text_boundaries),
        )

    def captured_program(self, since: CaptureMarks = NO_MARKS) -> CapturedProgram:
        """Everything captured, or everything captured since `since`."""
        runs, glyphs, drawings, inline_images, lines, text_boundaries = since
        return CapturedProgram(
            runs=tuple(self.runs[runs:]),
            glyphs=tuple(self.glyphs[glyphs:]),
            drawings=tuple(self.drawings[drawings:]),
            inline_images=tuple(self.inline_images[inline_images:]),
            lines=tuple(self.lines[lines:]),
            text_boundaries=tuple(self.text_boundaries[text_boundaries:]),
        )

    def resolve_soft_mask(self, value: object) -> PdfSoftMask | None:
        mask = super().resolve_soft_mask(value)
        # The parse cache keys on the resource scope, so a mask only ever comes
        # back under the scope it was parsed with and the entry never changes
        # once written. Rewriting it on every gs was work with no effect.
        if mask is not None and id(mask) not in self.capture_mask_resources:
            self.capture_mask_resources[id(mask)] = (mask, self.resources)
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
        # Matrix is immutable, so identity is enough to know the scale still
        # holds. Drawing-heavy pages emit long runs under one CTM, and every
        # drawing asks for this once for the line width and again for the dash
        # pattern, so the repeat is worth catching.
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

    def record_glyph_observations(
        self,
        text: str,
        decoder: FontDecoder,
        rotation_angle: int,
        visible: bool,
        *,
        fill_color: tuple[float, ...] | None,
        paint: GlyphPaint | None = None,
        glyphs: tuple[DecodedGlyph, ...],
        text_basis: TextBasis,
        effective_font_size: float,
        effective_font_height: float,
        font_scale: float,
        font_ascent: float,
        font_descent: float,
        advance_scale: float,
    ) -> GlyphCapture:
        geometry = TextGeometry(
            basis=text_basis,
            font_size=self.graphics.font_size,
            font_scale=font_scale,
            font_ascent=font_ascent,
            font_descent=font_descent,
            advance_scale=advance_scale,
            char_space=self.graphics.char_space,
            word_space=self.graphics.word_space,
            horizontal_scale=self.graphics.horizontal_scale,
            rise=self.graphics.rise,
            rotation_angle=rotation_angle,
            effective_font_size=effective_font_size,
            effective_font_height=effective_font_height,
        )
        if paint is None:
            paint = self.glyph_paint(fill_color)
        provenance = (
            ("source", self.capture_source),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            ("text_matrix", text_basis[2:]),
            ("text_render_mode", self.graphics.render_mode),
            ("line_matrix_origin", (self.line_matrix.e, self.line_matrix.f)),
            ("horizontal_scale", self.graphics.horizontal_scale),
            ("char_space", self.graphics.char_space),
            ("text_rise", self.graphics.rise),
        )
        return capture_glyphs(
            text,
            glyphs,
            decoder,
            geometry=geometry,
            paint=paint,
            visible=visible,
            font_name=self.graphics.current_font,
            provenance=provenance,
            seqno=self.sequence,
            text_object_id=self.text_object_id,
            cluster_start=self.glyph_cluster_count,
            capture_ink_bounds=self.capture_ink_bounds,
            capture_run_details=self.capture_text_runs,
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
        visible = self.is_text_visible(text)
        if 4 <= self.graphics.render_mode <= 7 and self.is_graphics_visible():
            self.emit_clip_scope_push()

        fs = self.graphics.font_size
        rise = self.graphics.rise

        font_scale = fs / 1000.0
        metrics_decoder: FontDecoder | None = self.graphics.current_decoder  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        ascent = metrics_decoder.ascent * font_scale if metrics_decoder is not None else 0.0
        descent = metrics_decoder.descent * font_scale if metrics_decoder is not None else 0.0
        advance_scale = fs * self.graphics.horizontal_scale / 100000.0

        text_matrix = self.text_matrix
        combined = text_matrix.multiply(self.graphics.ctm)
        # combined already carries the translation: multiply_affine's last two
        # terms are te * ca + tf * cc + ce and te * cb + tf * cd + cf, which is
        # what this used to recompute by hand. Checked bit for bit over 44,001
        # matrix pairs, including both of multiply's identity short circuits.
        A, B, C, D, E, F = combined
        te, tf = text_matrix.e, text_matrix.f

        rot = detect_rotation_from_linear(A, B, C, D)
        seqno = self.sequence
        scale_factor = hypot(C, D) if font_decoder.is_vertical else hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_font_height = fs * (hypot(A, B) if font_decoder.is_vertical else hypot(C, D))
        fill_color = self.capture_color(stroke=False) if glyph_paint is None else glyph_paint.fill
        actual_text_span = self.current_capture_actual_text_span()
        captured: GlyphCapture | None = None
        if actual_text_span is None:
            captured = self.record_glyph_observations(
                text,
                font_decoder,
                rot,
                visible,
                fill_color=fill_color,
                paint=glyph_paint,
                glyphs=decoded_glyphs,
                text_basis=(E, F, A, B, C, D),
                effective_font_size=effective_font_size,
                effective_font_height=effective_font_height,
                font_scale=font_scale,
                font_ascent=ascent,
                font_descent=descent,
                advance_scale=advance_scale,
            )
            self.glyphs.extend(captured.glyphs)
            self.glyph_cluster_count += captured.cluster_count
            if not self.capture_text_runs:
                self.sequence = seqno + 1
                return

        space_width = (
            metrics_decoder.glyph_width(32) * fs * 0.001 if metrics_decoder is not None else 0.0
        )

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

        effective_space_width = space_width * scale_factor
        baseline = (
            E,
            F,
            E + adv_x * A + adv_y * C,
            F + adv_x * B + adv_y * D,
        )
        provenance = (
            ("source", self.capture_source),
            ("seqno", seqno),
            ("font_name", self.graphics.current_font),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("text_render_mode", self.graphics.render_mode),
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

        new_run = TextRun(
            text=normalize_extracted_text(text),
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            tx=te,
            ty=tf,
            font_size=effective_font_size,
            font_name=self.graphics.current_font,
            space_width=effective_space_width,
            order=seqno,
            stream_order=self.stream_order,
            xobject_depth=self.xobject_depth,
            is_vertical=font_decoder.is_vertical,
            rotation_angle=rot,
            visible=visible,
            line_break_before=self.pending_line_break,
            seqno=seqno,
            fill_color=fill_color,
            advance_bbox=advance_bbox,
            ink_bbox=advance_bbox,
            baseline=baseline,
            provenance=provenance,
            confidence=None,
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
        # An empty path paints nothing, and everything below -- pattern and
        # colour-space probing, flattening, the CTM transform -- is setup for a
        # mark that will not exist. Content streams hit this constantly: a
        # paint operator resets current_path, so the common "m l S f" idiom
        # runs f against an empty path. One corpus page does that 18,560 times.
        if not source.commands:
            return
        if not self.is_graphics_visible():
            return
        fills = kind in {"fill", "fillstroke"} and not self.initial_pattern(stroke=False)
        strokes = kind in {"stroke", "fillstroke"} and not self.initial_pattern(stroke=True)
        if not fills and not strokes:
            return
        # The operator asked for one of the three; what actually paints after
        # the pattern and colour-space probes above may be narrower.
        painted: PaintedDrawingKind = (
            "fillstroke" if fills and strokes else "fill" if fills else "stroke"
        )
        fill_paints = color_space_paints(self.graphics.fill_space)
        stroke_paints = color_space_paints(self.graphics.stroke_space)

        captured_path = flatten_path(source)
        if self.graphics.ctm == IDENTITY_MATRIX:
            path = captured_path
        else:
            path = captured_path.transformed(self.graphics.ctm)
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
                    fill=self.capture_color(stroke=False),
                    fill_pattern=self.capture_pattern(self.graphics.fill_pattern)
                    if fill_paints and fills
                    else None,
                    fill_opacity=self.graphics.fill_opacity,
                    stroke_color=self.capture_color(stroke=True),
                    stroke_pattern=self.capture_pattern(self.graphics.stroke_pattern)
                    if stroke_paints and strokes
                    else None,
                    stroke_opacity=self.graphics.stroke_opacity,
                    line_width=line_width,
                    line_cap=self.graphics.line_cap,
                    line_join=self.graphics.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    kind=painted,
                    graphics_soft_mask=self.capture_graphics_soft_mask(),
                    fill_paints=fill_paints,
                    stroke_paints=stroke_paints,
                    path=path,
                    stream_order=self.stream_order,
                    xobject_depth=self.xobject_depth,
                )
            )
            self.sequence += 1

    def clip_path(self, state: object, source: PdfPath, fill_rule: str) -> None:
        path = flatten_path(source).transformed(self.graphics.ctm)
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
        key = (id(xobj), rendering)
        cached = self.capture_image_sources.get(key)
        if cached is not None and cached[0] is xobj:
            return cached[1], cached[2]
        source, smask_alpha = image_source_from_stream(
            xobj, self.resolver, color_rendering=rendering
        )
        self.capture_image_sources[key] = (xobj, source, smask_alpha)
        return source, smask_alpha

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
                    dictionary[PdfName.of("ColorSpace")] = cast(PdfObject, color_resource)
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
            values = tuple(
                self.resolver.resolve_float(value, default=None) for value in raw_bbox[:4]
            )
            if all(value is not None for value in values):
                x, y, w, h = cast(Rectangle, values)
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
        color = self.graphics.stroke_color if stroke else self.graphics.fill_color
        spec = self.graphics.stroke_space if stroke else self.graphics.fill_space
        if color is not None and spec is not None:
            if not color_space_paints(spec):
                return color
            key = (id(spec), color, self.graphics.color_rendering)
            previous = self.capture_colors.get(key)
            if previous is not None:
                return previous[1]
            converted = color_operands_to_srgb(
                spec, list(color), rendering=self.graphics.color_rendering
            )
            result = converted if converted is not None else color
            if len(self.capture_colors) >= COLOR_CACHE_LIMIT:
                self.capture_colors.clear()
            self.capture_colors[key] = (spec, result)
            return result
        return color

    def capture_shading_dictionary(self, dictionary: dict) -> dict:
        return {
            key: self.resolver.deep_resolve(value)
            if str(key) in {"ColorSpace", "Function", "Coords", "Domain", "Extend", "BBox"}
            else value
            for key, value in dictionary.items()
        }

    def nested_capture_state(self) -> TextState:
        nested = TextState(
            self.document,
            hidden_layers=self.hidden_layers,
            capture_ink_bounds=self.capture_ink_bounds,
            capture_text_runs=self.capture_text_runs,
        )
        nested.parsed_soft_masks = self.parsed_soft_masks
        nested.capture_soft_masks = self.capture_soft_masks
        nested.capture_mask_resources = self.capture_mask_resources
        nested.capture_active_mask_groups = self.capture_active_mask_groups
        nested.capture_image_sources = self.capture_image_sources
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
        key = (id(pattern), rendering, initial_alpha_is_shape, initial_text_knockout)
        if key in self.capture_patterns:
            return self.capture_patterns[key][1]
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
            nested.graphics.render_intent = self.graphics.render_intent
            nested.graphics.black_point_compensation = self.graphics.black_point_compensation
            nested.graphics.alpha_is_shape = initial_alpha_is_shape
            nested.graphics.text_knockout = initial_text_knockout
            try:
                nested.stream_executor.consume(pattern.stream, pattern.resources, pattern.matrix, 0)
            except Exception:
                self.capture_patterns[key] = (pattern, None)
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
                    glyph.fill = base_color
                    glyph.stroke_color = base_color
            result = TilingPattern(
                pattern.bbox,
                pattern.x_step,
                pattern.y_step,
                CapturedProgram(
                    glyphs=tuple(glyph for glyph in nested.glyphs if glyph.has_paint),
                    drawings=tuple(nested.drawings),
                    inline_images=tuple(nested.inline_images),
                    text_boundaries=tuple(nested.text_boundaries),
                ),
            )
        self.capture_patterns[key] = (pattern, result)
        return result

    def capture_graphics_soft_mask(self) -> CapturedSoftMask | None:
        mask = self.graphics.soft_mask
        if mask is None:
            return None
        # The five fields the nested capture overrides carry no information: four
        # are the same literals every time, and the fifth is mask.ctm, which id(mask)
        # already pins. Keying on the rest lets the lookup happen before the state is
        # copied, which is the whole cost on a hit -- 1.5us of copy plus five fields
        # of state_key, against a lookup that is measured in nanoseconds.
        key = (
            id(mask),
            tuple(state_key(getattr(self.graphics, name)) for name in MASK_KEYED_FIELDS),
        )
        cached = self.capture_soft_masks.get(key)
        if cached is not None:
            return cached[2]
        graphics = copy(self.graphics)
        graphics.ctm = mask.ctm
        graphics.soft_mask = None
        graphics.fill_opacity = graphics.stroke_opacity = 1.0
        graphics.blend_mode = None
        self.capture_soft_masks[key] = (mask, graphics, None)
        group_key = id(mask.group)
        if mask.subtype != "Alpha" or group_key in self.capture_active_mask_groups:
            return None
        if len(self.capture_active_mask_groups) >= 10:
            return None
        self.capture_active_mask_groups.add(group_key)
        try:
            nested = self.nested_capture_state()
            nested.graphics = copy(graphics)
            scope = self.capture_mask_resources.get(id(mask))
            nested.resources = scope[1] if scope is not None else self.resources
            frame = nested.append_form_xobject(mask.group, 0)
            if frame is None:
                return None
            nested.stream_executor.consume_frame(frame)
            nested.run_accumulator.flush()
            if not nested.text_boundaries:
                return None
            result = CapturedSoftMask(nested.captured_program(), mask.transfer)
            self.capture_soft_masks[key] = (mask, graphics, result)
            return result
        except PdfParseError, TypeError, ValueError, ArithmeticError:
            return None
        finally:
            self.capture_active_mask_groups.remove(group_key)

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        resolver = cast(Any, self.resolver)
        if allow_text:
            return cast(str | None, resolver.resolve_name_or_text(value))
        return cast(str | None, resolver.resolve_name_like_value(value))


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
    mask_alpha = None
    mask = source.soft_mask
    if mask is not None:
        width = resolver.resolve_int(mask.dictionary.get("Width")) or 0
        height = resolver.resolve_int(mask.dictionary.get("Height")) or 0
        data = mask.raw
        if width > 0 and height > 0 and data:
            total = min(len(data), width * height)
            mask_sum = numpy.frombuffer(data, numpy.uint8, count=total).sum(dtype=numpy.uint64)
            mask_alpha = int(mask_sum) / (255.0 * total)
    return source, mask_alpha


def flatten_path(source: PdfPath) -> CapturedPath:
    path = CapturedPath()
    for command in source.commands:
        values = command.operands
        match command.operator:
            case "m":
                path.move_to(*values)
            case "l":
                path.line_to(*values)
            case "h":
                path.close()
            case "re":
                path.rect(*values)
            case "c":
                x0, y0, x1, y1, x2, y2, x3, y3 = values
                matrix = command.ctm
                scale = max(hypot(matrix.a, matrix.b), hypot(matrix.c, matrix.d), 1.0)
                control_len = (
                    hypot(x1 - x0, y1 - y0) + hypot(x2 - x1, y2 - y1) + hypot(x3 - x2, y3 - y2)
                )
                flatness = max(0.1, command.flatness or 0.25)
                segments = max(4, min(128, ceil(control_len * scale / (flatness * 8.0))))
                previous_x, previous_y = x0, y0
                segment_step = 1.0 / segments
                for i in range(1, segments + 1):
                    t = i * segment_step
                    mt = 1.0 - t
                    mt2 = mt * mt
                    t2 = t * t
                    b0, b1, b2, b3 = mt2 * mt, 3.0 * mt2 * t, 3.0 * mt * t2, t2 * t
                    x = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
                    y = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
                    if not path.subpaths:
                        path.move_to(previous_x, previous_y)
                    path.line_to(x, y)
                    previous_x, previous_y = x, y
    return path


GRAPHICS_STATE_FIELDS = GraphicsState.__fields__

# capture_graphics_soft_mask overrides these before capturing, so they are
# constant for a given mask and cannot distinguish two of its cache entries.
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
