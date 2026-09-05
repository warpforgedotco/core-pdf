# SPDX-License-Identifier: AGPL-3.0-only
"""Capture decoded glyphs using the geometry and paint of one text show."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from core_pdf.impl.model.geometry import transform_bbox
from core_pdf.impl.model.glyphs import (
    GlyphCluster,
    GlyphObservation,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
)
from core_pdf.impl.spec.s_07_content.marked_content import min_optional_confidence
from core_pdf.impl.spec.s_09_fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.spec.s_09_fonts.font_program import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.types import Rectangle

# (base_x, base_y, combined_A, combined_B, combined_C, combined_D): invariant across every
# glyph in one text-showing operation, so callers looping over glyphs compute it once and
# pass it in rather than re-deriving it from `state` on every glyph.
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
    """Capture shapes for non-ligature CMap values that look concatenated."""
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


def internal_text_basis_rect(
    x0: float, y0: float, x1: float, y1: float, text_basis: TextBasis
) -> Rectangle:
    """Axis-aligned device bounds of a text-space rect under ``text_basis``."""
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
    rect = internal_text_basis_rect(text_x0, text_y0, text_x1, text_y1, text_basis)
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
        x0 = position_x + (dr if dr < ar else ar)
        x1 = position_x + (ar if ar > dr else dr)
        y0 = end_y if end_y < start_y else start_y
        y1 = start_y if start_y > end_y else end_y
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


class RunGeometry:
    """Running union of glyph advance/ink boxes plus the minimum confidence.

    Accumulated as observations are appended so a caller never has to rescan
    the slice it just wrote. Empty until the first `add`, which is what
    distinguishes "no glyphs recorded" from "a run at the origin".
    """

    __slots__ = ("started", "advance", "ink", "confidence")

    def __init__(self) -> None:
        self.started = False
        self.advance: Rectangle = (0.0, 0.0, 0.0, 0.0)
        self.ink: Rectangle = (0.0, 0.0, 0.0, 0.0)
        self.confidence: float | None = None

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
            bx0 if bx0 < ax0 else ax0,
            by0 if by0 < ay0 else ay0,
            bx1 if bx1 > ax1 else ax1,
            by1 if by1 > ay1 else ay1,
        )
        ix0, iy0, ix1, iy1 = self.ink
        bx0, by0, bx1, by1 = ink_bbox
        self.ink = (
            bx0 if bx0 < ix0 else ix0,
            by0 if by0 < iy0 else iy0,
            bx1 if bx1 > ix1 else ix1,
            by1 if by1 > iy1 else iy1,
        )
        self.confidence = min_optional_confidence(self.confidence, confidence)


@dataclass(frozen=True, slots=True)
class TextGeometry:
    """Font metrics, spacing and transforms fixed throughout a text show."""

    basis: TextBasis
    font_size: float
    font_scale: float
    font_ascent: float
    font_descent: float
    advance_scale: float
    char_space_scale: float
    word_space_scale: float
    char_space: float
    word_space: float
    horizontal_scale: float
    rise: float
    rotation_angle: int
    effective_font_size: float
    effective_font_height: float


@dataclass(frozen=True, slots=True)
class GlyphPaint:
    """Resolved paint and visibility; stroke measurements are in device space."""

    visible: bool
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


@dataclass(slots=True)
class GlyphCapture:
    """Observations, source-glyph clusters and their aggregate geometry."""

    glyphs: list[GlyphObservation] = field(default_factory=list)
    clusters: list[GlyphCluster] = field(default_factory=list)
    geometry: RunGeometry = field(default_factory=RunGeometry)


def capture_glyphs(
    text: str,
    glyphs: tuple[DecodedGlyph, ...],
    decoder: FontDecoder,
    *,
    geometry: TextGeometry,
    paint: GlyphPaint,
    font_name: str | None,
    provenance: tuple[tuple[str, object], ...],
    seqno: int,
    text_object_id: int,
    cluster_start: int,
) -> GlyphCapture:
    """Build observations without changing interpreter state or decoding again.

    A decoded source glyph owns one cluster, even when its Unicode expands
    into multiple observations. Each cluster paints its source outline once.
    """
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
    char_space_scale = geometry.char_space_scale
    word_space_scale = geometry.word_space_scale
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
    # Accumulated during the append loop so the caller need not rescan
    # the slice it just wrote.
    add_run_geometry = result.geometry.add
    for glyph in glyphs:
        if is_vertical:
            _, advance_y = decoder.glyph_advance_vector(
                glyph.width_code,
                font_size=font_size,
                char_space=geometry.char_space,
                word_space=geometry.word_space,
                horizontal_scale=geometry.horizontal_scale,
                encoded_space=glyph.code_bytes == b" ",
            )
            # Capture measures positive distance down the writing line.
            advance = -advance_y
        else:
            advance = (
                decoder.glyph_width(glyph.width_code)
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

        cluster_id = cluster_start + len(result.clusters)
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
            advance_bbox = internal_text_basis_rect(*text_box, text_basis)
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
            # The sums are written out rather than factored so that the
            # evaluation order -- and therefore the floats -- stay identical
            # to the vertical branch above.
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
                    offset,
                    advance,
                    is_vertical=False,
                    rise=rise,
                    font_ascent=font_ascent,
                    font_descent=font_descent,
                )
                advance_bbox = internal_text_basis_rect(*text_box, text_basis)
                baseline = transformed_text_line(*baseline_text, text_basis)
        observation_visible = paint.visible
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
            False if single_character else should_capture_suspicious_multi_glyph_bitmap(chunk_text)
        )
        bitmap_width = bitmap_height = 0
        bitmap_code: int | None = None
        if should_capture_glyph_bitmap(chunk_text) or suspicious_multi:
            bitmap_width, bitmap_height = glyph_bitmap_dimensions(glyph_bbox, font_size)
            bitmap_code = glyph.bitmap_code

        # Only the fragment geometry and text differ between an ordinary
        # glyph, an unsplit mapping and a ligature expanded into characters.
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
                char_advance_rect = internal_text_basis_rect(*char_box, text_basis)
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
                text=fragment_text,
                ink_bbox=ink,
                advance_bbox=advance_rect,
                seqno=seqno,
                code_bytes=glyph.code_bytes,
                char_code=glyph.char_code,
                cid=glyph.cid,
                gid=glyph.gid,
                font_name=effective_font_name,
                font_size=font_size,
                baseline=fragment_baseline,
                rotation_angle=geometry.rotation_angle,
                fill=paint.fill,
                visible=observation_visible,
                confidence=confidence,
                unicode_source=glyph.unicode_source,
                alternates=glyph.alternates,
                bitmap_width=bitmap_width,
                bitmap_height=bitmap_height,
                bitmap_code=bitmap_code,
                font_decoder=decoder,
                effective_font_size=geometry.effective_font_size,
                effective_font_height=geometry.effective_font_height,
                provenance=provenance,
                glyph_transform=outline_transform,
                text_render_mode=paint.render_mode,
                fill_opacity=paint.fill_opacity,
                stroke_color=paint.stroke_color,
                stroke_opacity=paint.stroke_opacity,
                line_width=paint.line_width,
                line_cap=paint.line_cap,
                line_join=paint.line_join,
                dash_pattern=paint.dash_pattern,
                blend_mode=paint.blend_mode,
                soft_mask_alpha=paint.group_alpha,
                paint_glyph=index == 0,
                text_object_id=text_object_id,
                cluster_key=cluster_provenance_id,
            )
            cluster_observations.append(observation)
            result.glyphs.append(observation)
            add_run_geometry(advance_rect, ink, confidence)
        cluster = glyph_cluster_from_observations(
            cluster_id, chunk_text, tuple(cluster_observations)
        )
        if cluster is not None:
            result.clusters.append(cluster)
        offset += advance
    return result
