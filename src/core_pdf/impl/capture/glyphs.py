# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from core_pdf.impl.capture.marked_content import min_optional_confidence
from core_pdf.impl.fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.fonts.font_program import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.model.geometry import transform_bbox
from core_pdf.impl.model.glyphs import (
    GlyphCluster,
    GlyphObservation,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
)
from core_pdf.impl.types import Rectangle

TextBasis = tuple[float, float, float, float, float, float]


GLYPH_BITMAP_REPAIR_LABELS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-+/()[]{}<>|_~"
)
SUSPICIOUS_GLYPH_BITMAP_TEXT = {"\ufffd", "\ufffc"}


def should_capture_glyph_bitmap(text: str) -> bool:
    if len(text) != 1:
        return False
    if text in GLYPH_BITMAP_REPAIR_LABELS:
        return True
    if text in SUSPICIOUS_GLYPH_BITMAP_TEXT:
        return True
    code = ord(text)
    return 0xE000 <= code <= 0xF8FF or code < 32


def should_capture_suspicious_multi_glyph_bitmap(text: str) -> bool:
    if len(text) <= 1 or text in LEGITIMATE_MULTI_CHAR_GLYPHS:
        return False
    nonspace = [char for char in text if not char.isspace()]
    if len(nonspace) < 2:
        return False
    punctuation = sum(not char.isalnum() for char in nonspace)
    return punctuation >= 1 and punctuation / len(nonspace) >= 0.25


def glyph_bitmap_dimensions(
    glyph_bbox: Rectangle | None,
    font_size: float,
) -> tuple[int, int]:
    if glyph_bbox is None:
        return (24, 32)
    x0, y0, x1, y1 = glyph_bbox
    width = x1 - x0
    height = y1 - y0
    if width <= 0.0 or height <= 0.0:
        return (24, 32)
    bitmap_h = max(16, min(64, ceil(max(font_size, 1.0) * 2.5)))
    bitmap_w = max(1, min(96, ceil(bitmap_h * width / height)))
    return (bitmap_w, bitmap_h)


def text_basis_rect(x0: float, y0: float, x1: float, y1: float, text_basis: TextBasis) -> Rectangle:
    base_x, base_y, a, b, c, d = text_basis
    return transform_bbox((x0, y0, x1, y1), (a, b, c, d, base_x, base_y))


def glyph_ink_rect(
    glyph_bbox: Rectangle | None,
    advance_start: float,
    fallback_bbox: Rectangle,
    text_basis: TextBasis,
    text_advance_scale: float,
    rise: float,
    font_scale: float,
) -> Rectangle:
    if glyph_bbox is None:
        return fallback_bbox
    gx0, gy0, gx1, gy1 = glyph_bbox
    if gx1 <= gx0 or gy1 <= gy0:
        return fallback_bbox
    text_x0 = advance_start + gx0 * text_advance_scale
    text_x1 = advance_start + gx1 * text_advance_scale
    text_y0 = rise + gy0 * font_scale
    text_y1 = rise + gy1 * font_scale
    rect = text_basis_rect(text_x0, text_y0, text_x1, text_y1, text_basis)
    fallback_height = fallback_bbox[3] - fallback_bbox[1]
    fallback_width = fallback_bbox[2] - fallback_bbox[0]
    rect_x0, rect_y0, rect_x1, rect_y1 = rect
    rect_height = rect_y1 - rect_y0
    rect_width = rect_x1 - rect_x0
    if rect_width <= 0.01 or rect_height <= 0.01:
        return fallback_bbox
    if fallback_width > 0.0 and rect_width > fallback_width * 4.0:
        return fallback_bbox
    if fallback_height > 0.0 and rect_height > fallback_height * 1.5:
        return fallback_bbox
    return rect


def transformed_text_line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text_basis: TextBasis,
) -> tuple[float, float, float, float]:
    base_x, base_y, a, b, c, d = text_basis
    return (
        base_x + x0 * a + y0 * c,
        base_y + x0 * b + y0 * d,
        base_x + x1 * a + y1 * c,
        base_y + x1 * b + y1 * d,
    )


def glyph_text_space_boxes(
    offset: float,
    advance: float,
    *,
    is_vertical: bool,
    rise: float,
    font_ascent: float,
    font_descent: float,
    position: tuple[float, float] = (0.0, 0.0),
) -> tuple[
    Rectangle,
    tuple[float, float, float, float],
]:
    if is_vertical:
        position_x, position_y = position
        start_y = rise + position_y - offset
        end_y = start_y - advance
        ar = font_ascent
        dr = font_descent
        x0 = position_x + (ar if ar < dr else dr)
        x1 = position_x + (ar if ar > dr else dr)
        y0 = start_y if start_y < end_y else end_y
        y1 = end_y if end_y > start_y else start_y
        return (
            (x0, y0, x1, y1),
            (0.0, start_y, 0.0, end_y),
        )
    ar = font_ascent + rise
    dr = font_descent + rise
    return (
        (offset, dr, offset + advance, ar),
        (offset, rise, offset + advance, rise),
    )


@dataclass(slots=True, eq=False)
class RunGeometry:
    started: bool = False
    advance: Rectangle = (0.0, 0.0, 0.0, 0.0)
    ink: Rectangle = (0.0, 0.0, 0.0, 0.0)
    confidence: float | None = None

    def add(
        self,
        advance_bbox: Rectangle,
        ink_bbox: Rectangle,
        confidence: float | None,
    ) -> None:
        if not self.started:
            self.started = True
            self.advance = advance_bbox
            self.ink = ink_bbox
            self.confidence = confidence
            return
        ax0, ay0, ax1, ay1 = self.advance
        bx0, by0, bx1, by1 = advance_bbox
        self.advance = (
            ax0 if ax0 < bx0 else bx0,
            ay0 if ay0 < by0 else by0,
            ax1 if ax1 > bx1 else bx1,
            ay1 if ay1 > by1 else by1,
        )
        ix0, iy0, ix1, iy1 = self.ink
        bx0, by0, bx1, by1 = ink_bbox
        self.ink = (
            ix0 if ix0 < bx0 else bx0,
            iy0 if iy0 < by0 else by0,
            ix1 if ix1 > bx1 else bx1,
            iy1 if iy1 > by1 else by1,
        )
        self.confidence = min_optional_confidence(self.confidence, confidence)


@dataclass(frozen=True, slots=True)
class TextGeometry:
    """The text-state geometry a run of glyphs is laid out under."""

    basis: TextBasis
    font_size: float
    font_scale: float
    font_ascent: float
    font_descent: float
    advance_scale: float
    char_space: float
    word_space: float
    horizontal_scale: float
    rise: float
    rotation_angle: int
    effective_font_size: float
    effective_font_height: float


@dataclass(frozen=True, slots=True)
class GlyphPaint:
    """The painting state a run of glyphs is drawn with."""

    clip_bbox: Rectangle | None
    page_clip: Rectangle | None
    fill: tuple[float, ...] | None
    render_mode: int
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_opacity: float | None
    line_width: float
    line_cap: int
    line_join: int
    dash_pattern: tuple[list[float], float] | None
    blend_mode: str | None
    group_alpha: float | None
    clip_glyph: bool
    alpha_is_shape: bool
    graphics_soft_mask: object | None


@dataclass(slots=True)
class GlyphCapture:
    """What one run of glyphs contributed, accumulated as it is captured."""

    glyphs: list[GlyphObservation] = field(default_factory=list)
    clusters: list[GlyphCluster] = field(default_factory=list)
    cluster_count: int = 0
    geometry: RunGeometry = field(default_factory=RunGeometry)


def capture_glyphs(
    text: str,
    glyphs: tuple[DecodedGlyph, ...],
    decoder: FontDecoder,
    *,
    geometry: TextGeometry,
    paint: GlyphPaint,
    visible: bool,
    font_name: str | None,
    provenance: tuple[tuple[str, object], ...],
    seqno: int,
    text_object_id: int,
    cluster_start: int,
    capture_ink_bounds: bool = True,
    capture_run_details: bool = True,
) -> GlyphCapture:
    result = GlyphCapture()
    if not glyphs:
        return result
    text_basis = geometry.basis
    _, _, combined_a, combined_b, combined_c, combined_d = text_basis
    font_size = geometry.font_size
    font_scale = geometry.font_scale
    font_ascent = geometry.font_ascent
    font_descent = geometry.font_descent
    rise = geometry.rise
    advance_scale = geometry.advance_scale
    effective_font_name = decoder.font_name or font_name
    is_vertical = decoder.is_vertical
    axis_aligned_horizontal = not is_vertical and combined_b == 0.0 and combined_c == 0.0
    glyph_bbox_for_code = decoder.glyph_bbox
    vertical_position = decoder.vertical_glyph_position
    clip_primary = paint.clip_bbox
    clip_page = paint.page_clip
    offset = 0.0
    cursor = 0
    transform_a = advance_scale * combined_a
    transform_b = advance_scale * combined_b
    transform_c = font_scale * combined_c
    transform_d = font_scale * combined_d
    rise_offset_x = rise * combined_c
    rise_offset_y = rise * combined_d

    if axis_aligned_horizontal:
        axis_advance_y0 = text_basis[1] + (font_descent + rise) * combined_d
        axis_advance_y1 = text_basis[1] + (font_ascent + rise) * combined_d
        if axis_advance_y0 > axis_advance_y1:
            axis_advance_y0, axis_advance_y1 = axis_advance_y1, axis_advance_y0
        axis_baseline_y = text_basis[1] + rise * combined_d
    add_run_geometry = result.geometry.add
    char_space = geometry.char_space
    word_space = geometry.word_space
    horizontal_scale = geometry.horizontal_scale
    glyph_width = decoder.glyph_width
    for glyph in glyphs:
        if is_vertical:
            advance_x, advance_y = decoder.glyph_advance_vector(
                glyph.width_code,
                font_size=font_size,
                char_space=char_space,
                word_space=word_space,
                horizontal_scale=horizontal_scale,
                encoded_space=glyph.code_bytes == b" ",
            )
            advance = -advance_y
        else:
            spacing = char_space + (word_space if glyph.code_bytes == b" " else 0.0)
            displacement = glyph_width(glyph.width_code) * font_size / 1000.0 + spacing
            advance = displacement * horizontal_scale / 100.0
        chunk_text = glyph.unicode
        if not chunk_text:
            chunk_text = text[cursor : cursor + 1]
        chunk_length = len(chunk_text)
        cursor += max(1, chunk_length)
        if not chunk_text:
            offset += advance
            continue

        cluster_id = cluster_start + result.cluster_count
        cluster_provenance_id = (seqno, cluster_id)
        if is_vertical:
            glyph_vertical_position = vertical_position(
                glyph.cid,
                font_size=font_size,
            )
            text_box, baseline_text = glyph_text_space_boxes(
                offset,
                advance,
                is_vertical=True,
                rise=rise,
                font_ascent=font_ascent,
                font_descent=font_descent,
                position=glyph_vertical_position,
            )
            advance_bbox = text_basis_rect(*text_box, text_basis)
            baseline = transformed_text_line(*baseline_text, text_basis)
            origin_x, position_y = glyph_vertical_position
            origin_y = rise + position_y - offset
            outline_transform = (
                transform_a,
                transform_b,
                transform_c,
                transform_d,
                text_basis[0] + origin_x * combined_a + origin_y * combined_c,
                text_basis[1] + origin_x * combined_b + origin_y * combined_d,
            )
        else:
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
                    advance_x1 if advance_x1 < advance_x0 else advance_x0,
                    axis_advance_y0,
                    advance_x0 if advance_x0 > advance_x1 else advance_x1,
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
                    offset,
                    advance,
                    is_vertical=False,
                    rise=rise,
                    font_ascent=font_ascent,
                    font_descent=font_descent,
                )
                advance_bbox = text_basis_rect(*text_box, text_basis)
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
        if is_vertical or not capture_ink_bounds:
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
            False if single_character else should_capture_suspicious_multi_glyph_bitmap(chunk_text)
        )
        bitmap_width = bitmap_height = 0
        bitmap_code: int | None = None
        if should_capture_glyph_bitmap(chunk_text) or suspicious_multi:
            bitmap_width, bitmap_height = glyph_bitmap_dimensions(glyph_bbox, font_size)
            bitmap_code = glyph.bitmap_code

        fragments: list[tuple[str, Rectangle, Rectangle, Rectangle, float]] = []
        if glyph.split_unicode and not single_character and not suspicious_multi:
            per_char_advance = advance / len(chunk_text)
            char_offset = offset
            for ch in chunk_text:
                char_confidence = glyph_unicode_confidence(
                    ch, glyph.unicode_source, glyph.alternates
                )
                char_box, char_baseline_text = glyph_text_space_boxes(
                    char_offset,
                    per_char_advance,
                    is_vertical=is_vertical,
                    rise=rise,
                    font_ascent=font_ascent,
                    font_descent=font_descent,
                )
                char_advance_rect = text_basis_rect(*char_box, text_basis)
                char_baseline = transformed_text_line(*char_baseline_text, text_basis)
                fragments.append(
                    (ch, char_advance_rect, char_advance_rect, char_baseline, char_confidence)
                )
                char_offset += per_char_advance
        else:
            fragments.append((chunk_text, rect, advance_bbox, baseline, observation_confidence))

        cluster_observations: list[GlyphObservation] = []
        for index, (fragment_text, ink, advance_rect, fragment_baseline, confidence) in enumerate(
            fragments
        ):
            observation = GlyphObservation(
                fragment_text,
                ink,
                advance_rect,
                seqno,
                glyph.code_bytes,
                glyph.char_code,
                glyph.cid,
                glyph.gid,
                effective_font_name,
                font_size,
                fragment_baseline,
                geometry.rotation_angle,
                paint.fill,
                observation_visible,
                confidence,
                glyph.unicode_source,
                glyph.alternates,
                (),
                bitmap_width,
                bitmap_height,
                bitmap_code,
                decoder,
                geometry.effective_font_size,
                geometry.effective_font_height,
                provenance,
                outline_transform,
                paint.render_mode,
                paint.fill_opacity,
                paint.stroke_color,
                paint.stroke_opacity,
                paint.line_width,
                paint.blend_mode,
                paint.group_alpha,
                index == 0,
                text_object_id,
                paint.line_cap,
                paint.line_join,
                paint.dash_pattern,
                cluster_provenance_id,
                paint.clip_glyph and not decoder.is_type3,
                paint.alpha_is_shape,
                decoder.is_type3,
                paint.graphics_soft_mask,
            )
            result.glyphs.append(observation)
            if capture_run_details:
                cluster_observations.append(observation)
                add_run_geometry(advance_rect, ink, confidence)
        result.cluster_count += 1
        if capture_run_details:
            cluster = glyph_cluster_from_observations(
                cluster_id, chunk_text, tuple(cluster_observations)
            )
            if cluster is not None:
                result.clusters.append(cluster)
        offset += advance
    return result
