# SPDX-License-Identifier: AGPL-3.0-only

"""TextState methods: showing text: glyph runs, text layouts and styles, and ActualText spans.

Each function takes the state as self; TextState lists them as its methods.
"""

from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING

from core_pdf.impl.base.glyphs import GlyphObservation, GlyphStyle
from core_pdf.impl.base.runs import TextRun
from core_pdf.impl.base.text import normalize_extracted_text
from core_pdf.impl.capture.glyphs import (
    GlyphCapture,
    GlyphPaint,
    capture_glyph_run,
    capture_glyphs,
    glyph_style,
    glyph_width_source,
)
from core_pdf.impl.capture.recording_support import (
    MarkedContentEntry,
    TextLayout,
    detect_rotation_from_linear,
    same_floats,
)
from core_pdf.impl.capture.records import CapturedTextBoundary
from core_pdf.impl.capture.text_runs import is_visible_text
from core_pdf.impl.fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.spec.s_07_content.model import NON_PAINTING_RENDER_MODES
from core_pdf.impl.spec.s_08_graphics.color import color_space_paints
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.spec.s_09_fonts.service import DecodedFontGlyph, FontService

if TYPE_CHECKING:
    from core_pdf.impl.capture.recording import TextState


def update_pending_run(self: TextState, new_run: TextRun) -> None:
    if not self.is_clipped_away(new_run.x0, new_run.y0, new_run.x1, new_run.y1):
        self.run_accumulator.append(new_run)


def glyph_paint(self: TextState, fill_color: tuple[float, ...] | None) -> GlyphPaint:
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


def emit_actual_text_span(self: TextState, entry: MarkedContentEntry) -> None:
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
    if self.options.glyphs:
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


def shared_paint(self: TextState) -> GlyphPaint:
    """The paint of the current fill color and state, made once until it changes.

    The dispatcher drops it, with the text layout, whenever an operator changes what it
    depends on, so a layout made for it serves every string until then.
    """
    paint = self.shared_glyph_paint
    if paint is None:
        paint = self.shared_glyph_paint = self.glyph_paint(self.capture_color(stroke=False))
    return paint


def new_text_layout(
    self: TextState,
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
        glyph_paint = self.shared_paint()
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
    layout.descent = metrics_decoder.descent * font_scale if metrics_decoder is not None else 0.0
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
    layout.width_source = glyph_width_source(font_decoder.glyph_width)
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


def current_text_layout(
    self: TextState,
    font_decoder: FontDecoder,
    glyph_paint: GlyphPaint | None,
    A: float,
    B: float,
    C: float,
    D: float,
    identity_text_matrix: bool,
) -> TextLayout:
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
    return layout


def new_layout_style(
    self: TextState, layout: TextLayout, A: float, B: float, C: float, D: float
) -> GlyphStyle:
    graphics = self.graphics
    style = layout.style = glyph_style(
        self.glyph_paint(layout.fill_color) if layout.glyph_paint is None else layout.glyph_paint,
        layout.decoder,
        layout.font_size,
        layout.rotation,
        layout.effective_font_size,
        layout.effective_font_height,
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
            ("text_rise", layout.rise),
        ),
        self.text_object_id,
    )
    return style


def show_glyph_run(
    self: TextState,
    glyphs: tuple[DecodedGlyph, ...],
    texts: list[str],
    origins: list[float],
    decoder: FontService,
    *,
    glyph_paint: GlyphPaint | None = None,
) -> tuple[tuple[float, float, float, float], list[float]] | None:
    """Show each horizontal glyph as show_text would show it alone, in one call.

    Glyph i is shown with the text it is given, texts[i] -- its unicode, or one character
    when that is empty, else the run is declined -- and the text matrix moved to
    the origin (origins[2 * i], origins[2 * i + 1]); the text matrix itself is left as
    it was. Returns the linear part of the text matrix times the ctm and each glyph's
    origin under it, or None, having shown nothing, when the capture records runs or
    the glyphs need show_text one at a time: vertical text, an ActualText span, a
    glyph whose text splits, texts other than those.
    """
    font_decoder: FontDecoder = decoder  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    options = self.options
    if not glyphs or font_decoder.is_vertical or options.text_runs or not options.glyphs:
        return None
    text_matrix = self.text_matrix
    ctm = self.graphics.ctm
    ta, tb, tc, td = text_matrix.a, text_matrix.b, text_matrix.c, text_matrix.d
    identity_linear = ta == 1.0 and tb == 0.0 and tc == 0.0 and td == 1.0
    first_identity = identity_linear and origins[0] == 0.0 and origins[1] == 0.0
    # The text matrix times the ctm, as Matrix.multiply computes it for each glyph.
    if ctm == IDENTITY_MATRIX:
        A, B, C, D = ta, tb, tc, td
        combined = origins
    else:
        ca, cb, cc, cd, ce, cf = ctm
        A = ta * ca + tb * cc
        B = ta * cb + tb * cd
        C = tc * ca + td * cc
        D = tc * cb + td * cd
        # multiply returns the ctm itself for a glyph at the identity text matrix, whose
        # linear part can differ from the product's in the sign of a zero.
        ctm_linear_differs = identity_linear and not same_floats((A, B, C, D), (ca, cb, cc, cd))
        combined = [0.0] * len(origins)
        for index in range(0, len(origins), 2):
            e = origins[index]
            f = origins[index + 1]
            if identity_linear and e == 0.0 and f == 0.0:
                if ctm_linear_differs:
                    return None
                combined[index] = ce
                combined[index + 1] = cf
            else:
                combined[index] = e * ca + f * cc + ce
                combined[index + 1] = e * cb + f * cd + cf
    layout = self.current_text_layout(font_decoder, glyph_paint, A, B, C, D, first_identity)
    if layout.actual_text_span is not None:
        return None
    paints_text = layout.paints_text
    visibility = bytearray(len(glyphs))
    for index, (glyph, text) in enumerate(zip(glyphs, texts, strict=True)):
        # The capture takes a glyph's text from the joined texts only when its unicode
        # is empty, so each text must be its unicode, or one character without one.
        if (text != glyph.unicode) if glyph.unicode else (len(text) != 1):
            return None
        if paints_text and is_visible_text(text):
            visibility[index] = 1
    if layout.pushes_clip_scope:
        self.emit_clip_scope_push()
    style = layout.style
    if style is None:
        style = self.new_layout_style(layout, A, B, C, D)
    graphics = self.graphics
    seqno = self.sequence
    count = capture_glyph_run(
        "".join(texts),
        glyphs,
        font_decoder,
        (A, B, C, D),
        combined,
        bytes(visibility),
        layout.font_size,
        layout.font_scale,
        layout.ascent,
        layout.descent,
        layout.advance_scale,
        graphics.char_space,
        graphics.word_space,
        graphics.horizontal_scale,
        layout.rise,
        style,
        self.clip_bbox,
        self.page_clip,
        graphics.current_font,
        seqno,
        self.glyph_cluster_count,
        options,
        layout.width_source,
        self.glyphs,
    )
    if count is None:
        return None
    self.glyph_cluster_count += count
    self.sequence = seqno + len(glyphs)
    return (A, B, C, D), combined


def show_text(
    self: TextState,
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
    layout = self.current_text_layout(
        font_decoder, glyph_paint, A, B, C, D, text_matrix == IDENTITY_MATRIX
    )

    visible = bool(text) and layout.paints_text and is_visible_text(text)
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
        options = self.options
        if not options.glyphs and not options.text_runs:
            # Neither glyphs nor runs are wanted: only the sequence advances.
            self.sequence = seqno + 1
            return
        style = layout.style
        if style is None:
            style = self.new_layout_style(layout, A, B, C, D)
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
            layout.width_source,
            # The observations go straight into the program's list, when it keeps them.
            self.glyphs if options.glyphs else [],
        )
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


def text_boundary(self: TextState, state: object, kind: str) -> None:
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


def current_capture_actual_text_span(self: TextState) -> MarkedContentEntry | None:
    entry = self.current_actual_text_span()
    if entry is None:
        return None
    key = id(entry)
    captured = self.capture_marked_entries.get(key)
    if captured is None:
        captured = MarkedContentEntry(entry.layer, entry.actual_text, entry.mcid)
        self.capture_marked_entries[key] = captured
    return captured


def text_paint_mode(self: TextState, *, check_colorants: bool = True) -> int:
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
