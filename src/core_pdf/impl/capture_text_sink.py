# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING

from core_pdf.impl.capture_glyphs import GlyphCapture, GlyphPaint, capture_glyphs, glyph_style
from core_pdf.impl.capture_host import CaptureHost
from core_pdf.impl.capture_records import CapturedTextBoundary
from core_pdf.impl.capture_text_runs import is_garbage_text
from core_pdf.impl.fonts_decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.glyphs import GlyphObservation, GlyphStyle
from core_pdf.impl.runs import TextRun
from core_pdf.impl.text import normalize_extracted_text
from core_pdf_spec.s_07_content.model import NON_PAINTING_RENDER_MODES
from core_pdf_spec.s_07_content.model import MarkedContentEntry as SemanticMarkedContentEntry
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph, FontService

if TYPE_CHECKING:
    pass


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


class GlyphCaptureMixin(CaptureHost):
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


__all__ = (
    "GlyphCaptureMixin",
    "MATRIX_TOLERANCE",
    "MarkedContentEntry",
    "TextLayout",
    "detect_rotation_from_linear",
)
