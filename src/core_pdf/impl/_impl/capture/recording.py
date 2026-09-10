# SPDX-License-Identifier: AGPL-3.0-only
"""Content-stream interpreter state.

Holds the graphics and text state, the operator handlers, and glyph emission.
"""

from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl._impl.capture.paths import flatten_path

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.inline_images import InlineImage

from dataclasses import dataclass

from core_pdf.impl._impl.capture.glyphs import (
    GlyphCapture,
    GlyphPaint,
    TextBasis,
    TextGeometry,
    capture_glyphs,
)
from core_pdf.impl._impl.capture.images import image_source_from_stream
from core_pdf.impl._impl.capture.marked_content import MarkedContentEntry
from core_pdf.impl._impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
    LayoutFormId,
    PatternPaint,
    ShadingPattern,
    TilingPattern,
    marker_drawing,
)
from core_pdf.impl._impl.capture.text_runs import (
    RunAccumulator,
    is_garbage_text,
)
from core_pdf.impl._impl.capture.tolerant_state import RecoveringTextState as SemanticTextState
from core_pdf.impl._impl.fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl._impl.graphics.color import color_operands_to_srgb
from core_pdf.impl._impl.graphics.color_spec import color_spec_from_value
from core_pdf.impl._impl.model.geometry import RectBox, intersect_bbox, transform_bbox
from core_pdf.impl._impl.model.glyphs import (
    GlyphCluster,
    GlyphObservation,
)
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl._impl.model.text import normalize_extracted_text
from core_pdf.impl.types import (
    PdfName,
    Rectangle,
)
from core_pdf_spec.s_07_content.image_capture import unit_square_placement
from core_pdf_spec.s_07_content.marked_content import (
    MarkedContentEntry as SemanticMarkedContentEntry,
)
from core_pdf_spec.s_07_content.paths import PdfPath
from core_pdf_spec.s_07_content.patterns import ShadingPattern as PdfShadingPattern
from core_pdf_spec.s_07_content.patterns import TilingPattern as PdfTilingPattern
from core_pdf_spec.s_07_content.stream_state import (
    ContentStreamFrame,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
)
from core_pdf_spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph, FontService


@dataclass(slots=True)
class CaptureGraphicsSave:
    clip_bbox: Rectangle | None
    group_alpha: float | None
    clip_scope_emitted: bool = False


MATRIX_TOLERANCE = 0.1
internal_NON_PAINTING_RENDER_MODES = frozenset({3, 7})


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


class RecordingMethods(SemanticTextState):
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_clusters: list[GlyphCluster]
    lines: list[CapturedLine]
    drawings: list[CapturedDrawing]
    inline_images: list[CapturedInlineImage]
    hidden_layers: frozenset[str]
    page_clip: Rectangle | None
    clip_bbox: Rectangle | None
    layout_form_bbox: Rectangle | None
    layout_form_id: LayoutFormId
    capture_source: str
    stream_order: int
    sequence: int
    text_object_id: int
    pending_line_break: bool
    group_alpha: float | None
    run_accumulator: RunAccumulator
    capture_graphics_stack: list[CaptureGraphicsSave]
    capture_marked_entries: dict[int, MarkedContentEntry]
    capture_frames: dict[int, tuple[Rectangle | None, LayoutFormId, bool]]
    capture_patterns: dict[int, tuple[object, PatternPaint | None]]

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

        return self.is_graphics_visible()

    def is_graphics_visible(self) -> bool:
        for entry in self.marked_content_stack:
            if entry.layer and entry.layer in self.hidden_layers:
                return False
        return True

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
        if not self.internal_is_clipped_away(new_run.x0, new_run.y0, new_run.x1, new_run.y1):
            self.run_accumulator.append(new_run)

    def record_glyph_observations(
        self,
        text: str,
        decoder: FontDecoder,
        rotation_angle: int,
        visible: bool,
        *,
        glyphs: tuple[DecodedGlyph, ...],
        text_basis: TextBasis,
        effective_font_size: float,
        effective_font_height: float,
    ) -> GlyphCapture:
        """Snapshot this text show's inputs for the independent glyph recorder."""
        geometry = TextGeometry(
            basis=text_basis,
            font_size=self.font_size,
            font_scale=self.font_scale,
            font_ascent=self.font_ascent,
            font_descent=self.font_descent,
            advance_scale=self.text_advance_scale,
            char_space_scale=self.char_space_scale,
            word_space_scale=self.word_space_scale,
            char_space=self.char_space,
            word_space=self.word_space,
            horizontal_scale=self.horizontal_scale,
            rise=self.rise,
            rotation_angle=rotation_angle,
            effective_font_size=effective_font_size,
            effective_font_height=effective_font_height,
        )
        paint = GlyphPaint(
            visible=visible,
            clip_bbox=self.clip_bbox,
            page_clip=self.page_clip,
            fill=self.capture_color(stroke=False),
            render_mode=self.render_mode,
            fill_opacity=self.fill_opacity,
            stroke_color=self.capture_color(stroke=True),
            stroke_opacity=self.stroke_opacity,
            line_width=self.transformed_line_width(),
            line_cap=self.line_cap,
            line_join=self.line_join,
            dash_pattern=self.transformed_dash_pattern(),
            blend_mode=self.blend_mode,
            group_alpha=self.group_alpha,
        )
        provenance = (
            ("source", self.capture_source),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            ("text_matrix", text_basis[2:]),
            ("line_matrix_origin", (self.lm_e, self.lm_f)),
            ("horizontal_scale", self.horizontal_scale),
            ("char_space", self.char_space),
            ("text_rise", self.rise),
        )
        return capture_glyphs(
            text,
            glyphs,
            decoder,
            geometry=geometry,
            paint=paint,
            font_name=self.current_font,
            provenance=provenance,
            seqno=self.sequence,
            text_object_id=self.text_object_id,
            cluster_start=len(self.glyph_clusters),
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

    def internal_emit_clip_scope_push(self) -> None:
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
    ) -> None:
        decoder = cast(FontDecoder, decoder)
        glyphs = cast(tuple[DecodedGlyph, ...], glyphs)
        visible = self.is_text_visible(text)

        fs = self.font_size
        rise = self.rise

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

        new_run = TextRun(
            text=normalize_extracted_text(text),
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
            fill_color=self.capture_color(stroke=False),
            advance_bbox=advance_bbox,
            ink_bbox=advance_bbox,
            baseline=baseline,
            provenance=provenance,
            confidence=None,
        )
        actual_text_span = self.current_capture_actual_text_span()
        if actual_text_span is not None:
            new_run.confidence = 1.0
            actual_text_span.add_run(
                new_run,
                font_decoder=decoder,
                effective_font_height=effective_font_height,
            )
        else:
            captured = self.record_glyph_observations(
                text,
                decoder,
                rot,
                visible,
                glyphs=glyphs,
                text_basis=(E, F, A, B, C, D),
                effective_font_size=effective_font_size,
                effective_font_height=effective_font_height,
            )
            self.glyphs.extend(captured.glyphs)
            self.glyph_clusters.extend(captured.clusters)
            new_run.glyph_clusters = tuple(captured.clusters)
            geometry = captured.geometry
            if geometry.started:
                new_run.advance_bbox = geometry.advance
                new_run.ink_bbox = geometry.ink
                new_run.confidence = geometry.confidence
            self.update_pending_run(new_run)

        self.sequence = seqno + 1

    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        if not self.is_graphics_visible():
            return

        captured_path = flatten_path(source)
        if (
            self.ca == 1.0
            and self.cb == 0.0
            and self.cc == 0.0
            and self.cd == 1.0
            and self.ce == 0.0
            and self.cf == 0.0
        ):
            path = captured_path
        else:
            path = captured_path.transformed(
                Matrix(self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)
            )
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
                    fill_pattern=self.capture_pattern(self.fill_pattern),
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.capture_color(stroke=True),
                    stroke_pattern=self.capture_pattern(self.stroke_pattern),
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

    def clip_path(self, state: object, source: PdfPath, fill_rule: str) -> None:
        path = flatten_path(source).transformed(self.ctm)
        if not path.has_segments():
            return
        clip_bbox = path.bbox()
        if clip_bbox is not None:
            self.clip_bbox = intersect_bbox(self.clip_bbox, clip_bbox)
        if self.is_graphics_visible():
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
            self.sequence += 1

    def paint_image(self, state: object, xobj: PdfStream) -> None:
        xobj_dict = xobj.dictionary
        if self.is_graphics_visible():
            width = self.document.resolver.resolve_int(xobj_dict.get("Width")) or 0
            height = self.document.resolver.resolve_int(xobj_dict.get("Height")) or 0
            bbox = None
            quad = None
            if width > 0 and height > 0:
                bounds, quad = unit_square_placement(self.ctm)
                bbox = RectBox(*bounds)
            source, smask_alpha = image_source_from_stream(xobj, self.document.resolver)
            # A stencil mask carries no colour samples: PDF 8.9.6.2 paints its
            # set bits in the current fill colour. Every other image ignores
            # the fill, so recording it is only meaningful for the mask case,
            # but it costs nothing to carry and the renderer decides.
            image_is_stencil = xobj_dict.get("ImageMask") is True
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=self.capture_color(stroke=False) if image_is_stencil else None,
                    fill_opacity=self.fill_opacity,
                    blend_mode=self.blend_mode,
                    dash_pattern=self.transformed_dash_pattern(),
                    soft_mask_alpha=smask_alpha,
                    kind="image",
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

    def paint_inline_image(self, state: object, image: "InlineImage") -> None:
        if self.is_graphics_visible():
            dictionary = dict(image.dictionary)
            data = getattr(image, "data", b"")
            color_name = normalize_pdf_name(dictionary.get("ColorSpace"))
            if color_name is not None:
                color_resource = self.document.resolver.deep_resolve(
                    self.lookup_page_resource("ColorSpace", color_name)
                )
                if color_resource is not None:
                    dictionary[PdfName.of("ColorSpace")] = cast(PdfObject, color_resource)
            source, _ = image_source_from_stream(
                PdfStream(raw_data=data, dictionary=dictionary), self.document.resolver
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
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    stream_order=self.stream_order,
                    fill=self.capture_color(stroke=False)
                    if dictionary.get("ImageMask") is True
                    else None,
                    fill_opacity=self.fill_opacity,
                )
            )
            self.sequence += 1

    def paint_shading(self, state: object, shading: PdfDict) -> None:
        if not self.is_graphics_visible():
            return
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.capture_color(stroke=False),
                fill_opacity=self.fill_opacity,
                stroke_color=self.capture_color(stroke=True),
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
        self.sequence += 1

    def text_boundary(self, state: object, kind: str) -> None:
        if kind == "begin":
            self.text_object_id += 1
            self.run_accumulator.flush()
        elif kind == "shown":
            self.pending_line_break = False
        elif kind == "quoted":
            self.pending_line_break = True
        else:
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
        self.capture_frames[id(frame)] = (
            self.layout_form_bbox,
            self.layout_form_id,
            self.pending_line_break,
        )
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-begin",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.blend_mode,
                )
            )
            self.sequence += 1
            self.group_alpha = None
        layout_bbox = None
        raw_bbox = frame.form_bbox_operand
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            values = tuple(
                self.document.resolver.resolve_float(value, default=None) for value in raw_bbox[:4]
            )
            if all(value is not None for value in values):
                x, y, w, h = cast(Rectangle, values)
                # Preserve the historical PDFMiner LTFigure projection separately.
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
                    blend_mode=self.blend_mode,
                )
            )
            self.sequence += 1

    def capture_color(self, *, stroke: bool) -> tuple[float, ...] | None:
        """Project PDF color components only when creating output records."""
        color = self.stroke_color if stroke else self.fill_color
        spec = self.stroke_color_spec if stroke else self.fill_color_spec
        if (
            color is not None
            and spec is not None
            and spec.kind in {"Indexed", "Separation", "DeviceN"}
        ):
            converted = color_operands_to_srgb(spec, list(color))
            if converted is not None:
                return converted
        return color

    def parse_color_space(self, value: object) -> ImageColorSpec:
        return color_spec_from_value(value)

    def capture_pattern(self, pattern: object) -> PatternPaint | None:
        if pattern is None:
            return None
        key = id(pattern)
        if key in self.capture_patterns:
            return self.capture_patterns[key][1]
        result: PatternPaint | None = None
        if isinstance(pattern, PdfShadingPattern):
            result = ShadingPattern(dict(pattern.dictionary))
        elif isinstance(pattern, PdfTilingPattern):
            from core_pdf.impl._impl.capture.interpreter import TextState

            nested = TextState(self.document, hidden_layers=self.hidden_layers)
            try:
                nested.consume_stream(pattern.stream, pattern.resources, pattern.matrix, 0)
            except Exception:
                self.capture_patterns[key] = (pattern, None)
                return None
            if pattern.paint_type == 2:
                for drawing in nested.drawings:
                    if drawing.kind in {"fill", "fillstroke"}:
                        drawing.fill = pattern.base_color
                    if drawing.kind in {"stroke", "fillstroke"}:
                        drawing.stroke_color = pattern.base_color
                for glyph in nested.glyphs:
                    glyph.fill = pattern.base_color
                    glyph.stroke_color = pattern.base_color
            result = TilingPattern(
                pattern.bbox,
                pattern.x_step,
                pattern.y_step,
                nested.drawings,
                [glyph for glyph in nested.glyphs if glyph.has_paint],
                nested.inline_images,
            )
        self.capture_patterns[key] = (pattern, result)
        return result

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        resolver = cast(Any, self.document.resolver)
        if allow_text:
            return cast(str | None, resolver.resolve_name_or_text(value))
        return cast(str | None, resolver.resolve_name_like_value(value))
