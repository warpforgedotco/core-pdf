# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field  # noqa: TID251

from core_pdf.impl.capture_glyph_boxes import (
    TextBasis,
    glyph_text_space_boxes,
    text_basis_rect,
    transformed_text_line,
)
from core_pdf.impl.capture_glyph_geometry import NO_BOX, vertical_glyph_geometry
from core_pdf.impl.capture_program import CaptureOptions
from core_pdf.impl.fonts_decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.fonts_helpers import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.glyphs import (
    GlyphClusterLike,
    GlyphObservation,
    GlyphStyle,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
    min_optional_confidence,
)
from core_pdf.impl.types import RecordType, Rectangle, ReplaceFields, ReprFields
from core_pdf_cythonized import horizontal_glyph_geometry

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


class RunGeometry(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False, eq=False):
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


class GlyphPaint(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
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
    glyphs: list[GlyphObservation] = field(default_factory=list)
    clusters: list[GlyphClusterLike] = field(default_factory=list)
    cluster_count: int = 0
    geometry: RunGeometry = field(default_factory=RunGeometry)


def glyph_style(
    paint: GlyphPaint,
    decoder: FontDecoder,
    font_size: float,
    rotation_angle: int,
    effective_font_size: float,
    effective_font_height: float,
    provenance: tuple[tuple[str, object], ...],
    text_object_id: int,
) -> GlyphStyle:
    return GlyphStyle(
        font_size,
        rotation_angle,
        paint.fill,
        decoder,
        effective_font_size,
        effective_font_height,
        provenance,
        paint.render_mode,
        paint.fill_opacity,
        paint.stroke_color,
        paint.stroke_opacity,
        paint.line_width,
        paint.blend_mode,
        paint.group_alpha,
        text_object_id,
        paint.line_cap,
        paint.line_join,
        paint.dash_pattern,
        paint.clip_glyph and not decoder.is_type3,
        paint.alpha_is_shape,
        decoder.is_type3,
        paint.graphics_soft_mask,
    )


def capture_glyphs(
    text: str,
    glyphs: tuple[DecodedGlyph, ...],
    decoder: FontDecoder,
    text_basis: TextBasis,
    font_size: float,
    font_scale: float,
    font_ascent: float,
    font_descent: float,
    advance_scale: float,
    char_space: float,
    word_space: float,
    horizontal_scale: float,
    rise: float,
    style: GlyphStyle,
    clip_bbox: Rectangle | None,
    page_clip: Rectangle | None,
    visible: bool,
    font_name: str | None,
    seqno: int,
    cluster_start: int,
    options: CaptureOptions,
    /,
) -> GlyphCapture:
    result = GlyphCapture()
    if not glyphs:
        return result
    effective_font_name = decoder.font_name or font_name
    is_vertical = decoder.is_vertical
    glyph_width = decoder.glyph_width
    glyph_bbox_for_code = decoder.glyph_bbox
    vertical_position = decoder.vertical_glyph_position
    want_ink = options.ink_bounds and not is_vertical
    want_render = options.render_details
    want_runs = options.text_runs

    kept: list[DecodedGlyph] = []
    chunk_texts: list[str] = []
    offsets: list[float] = []
    advances: list[float] = []
    glyph_boxes: list[float] = []
    want_bitmap: list[int] = []
    split_flags: list[bool] = []
    positions: list[tuple[float, float]] = []
    offset = 0.0
    cursor = 0
    any_split = False
    for glyph in glyphs:
        if is_vertical:
            _, advance_y = decoder.glyph_advance_vector(
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

        kept.append(glyph)
        chunk_texts.append(chunk_text)
        offsets.append(offset)
        advances.append(advance)
        box = glyph_bbox_for_code(glyph.bitmap_code) if want_ink else None
        if box is None:
            glyph_boxes.extend((NO_BOX, NO_BOX, NO_BOX, NO_BOX))
        else:
            glyph_boxes.extend(box)
        suspicious = (
            False if chunk_length == 1 else should_capture_suspicious_multi_glyph_bitmap(chunk_text)
        )
        split = glyph.split_unicode and chunk_length != 1 and not suspicious
        split_flags.append(split)
        any_split = any_split or split
        want_bitmap.append(
            1 if want_render and (should_capture_glyph_bitmap(chunk_text) or suspicious) else 0
        )
        if is_vertical:
            positions.append(vertical_position(glyph.cid, font_size=font_size))
        offset += advance

    if not kept:
        return result

    if is_vertical:
        advance_union = ink_union = None
        advance_f, baseline_f, transform_f, ink_f, visible_f, bitmap_f = vertical_glyph_geometry(
            offsets,
            advances,
            positions,
            basis=text_basis,
            font_ascent=font_ascent,
            font_descent=font_descent,
            rise=rise,
            font_scale=font_scale,
            advance_scale=advance_scale,
            clip_primary=clip_bbox,
            clip_page=page_clip,
            visible=visible,
            want_bitmap=want_bitmap,
            want_transform=want_render,
        )
    else:
        (
            advance_f,
            baseline_f,
            transform_f,
            ink_f,
            visible_f,
            bitmap_f,
            advance_union,
            ink_union,
        ) = horizontal_glyph_geometry(
            offsets,
            advances,
            glyph_boxes,
            text_basis,
            font_ascent,
            font_descent,
            rise,
            font_scale,
            advance_scale,
            font_size,
            clip_bbox,
            page_clip,
            visible,
            want_bitmap,
            want_render,
        )

    fused = want_runs and advance_union is not None and not any_split
    run_confidence: float | None = None
    styled_observation = GlyphObservation.styled
    add_run_geometry = result.geometry.add
    append_glyph = result.glyphs.append
    append_cluster = result.clusters.append
    for index, glyph in enumerate(kept):
        chunk_text = chunk_texts[index]
        chunk_length = len(chunk_text)
        advance_bbox = advance_f[index]
        baseline = baseline_f[index]
        rect = ink_f[index]
        outline_transform = transform_f[index]
        observation_visible = bool(visible_f[index])
        bitmap_width = bitmap_f[2 * index]
        bitmap_height = bitmap_f[2 * index + 1]
        bitmap_code = glyph.bitmap_code if want_bitmap[index] else None

        cluster_id = cluster_start + index
        cluster_provenance_id = (seqno, cluster_id)
        observation_confidence = glyph_unicode_confidence(
            chunk_text,
            glyph.unicode_source,
            glyph.alternates,
        )

        if not split_flags[index]:
            observation = styled_observation(
                style,
                chunk_text,
                rect,
                advance_bbox,
                seqno,
                glyph.code_bytes,
                glyph.char_code,
                glyph.cid,
                glyph.gid,
                effective_font_name,
                baseline,
                observation_visible,
                observation_confidence,
                glyph.unicode_source,
                glyph.alternates,
                (),
                bitmap_width,
                bitmap_height,
                bitmap_code,
                outline_transform,
                True,
                cluster_provenance_id,
            )
            append_glyph(observation)
            if want_runs:
                if not fused:
                    add_run_geometry(advance_bbox, rect, observation_confidence)
                elif run_confidence is None:
                    run_confidence = observation_confidence
                elif observation_confidence is not None:
                    run_confidence = min(run_confidence, observation_confidence)
                append_cluster(observation)
            continue

        per_char_advance = advances[index] / chunk_length
        char_offset = offsets[index]
        cluster_observations: list[GlyphObservation] = []
        for position, ch in enumerate(chunk_text):
            confidence = glyph_unicode_confidence(ch, glyph.unicode_source, glyph.alternates)
            char_box, char_baseline_text = glyph_text_space_boxes(
                char_offset,
                per_char_advance,
                is_vertical=is_vertical,
                rise=rise,
                font_ascent=font_ascent,
                font_descent=font_descent,
            )
            advance_rect = text_basis_rect(*char_box, text_basis)
            observation = styled_observation(
                style,
                ch,
                advance_rect,
                advance_rect,
                seqno,
                glyph.code_bytes,
                glyph.char_code,
                glyph.cid,
                glyph.gid,
                effective_font_name,
                transformed_text_line(*char_baseline_text, text_basis),
                observation_visible,
                confidence,
                glyph.unicode_source,
                glyph.alternates,
                (),
                bitmap_width,
                bitmap_height,
                bitmap_code,
                outline_transform,
                position == 0,
                cluster_provenance_id,
            )
            char_offset += per_char_advance
            append_glyph(observation)
            if want_runs:
                cluster_observations.append(observation)
                add_run_geometry(advance_rect, advance_rect, confidence)
        if want_runs:
            cluster = glyph_cluster_from_observations(
                cluster_id, chunk_text, tuple(cluster_observations)
            )
            if cluster is not None:
                result.clusters.append(cluster)
    if fused and advance_union is not None and ink_union is not None:
        geometry = result.geometry
        geometry.started = True
        geometry.advance = advance_union
        geometry.ink = ink_union
        geometry.confidence = run_confidence
    result.cluster_count = len(kept)
    return result
