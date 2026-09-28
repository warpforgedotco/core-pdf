# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field  # noqa: TID251
from typing import Any

from core_pdf.impl.capture_program import CaptureOptions
from core_pdf.impl.fonts_decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.fonts_helpers import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.fonts_metrics import FontMetricsModel
from core_pdf.impl.geometry import transform_bbox
from core_pdf.impl.glyphs import (
    CONFIDENCE_CACHE,
    GlyphClusterLike,
    GlyphObservation,
    GlyphStyle,
    Matrix6,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
    min_optional_confidence,
)
from core_pdf.impl.types import RecordType, Rectangle, ReplaceFields, ReprFields
from core_pdf_cythonized import (
    DECODED_GLYPH_FIELDS,
    OBSERVATION_FIELDS,
    SlotLayout,
    capture_horizontal_glyphs,
    horizontal_glyph_geometry,
)

TextBasis = tuple[float, float, float, float, float, float]


def text_basis_rect(x0: float, y0: float, x1: float, y1: float, text_basis: TextBasis) -> Rectangle:
    base_x, base_y, a, b, c, d = text_basis
    return transform_bbox((x0, y0, x1, y1), (a, b, c, d, base_x, base_y))


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
        x0 = position_x + (min(dr, ar))
        x1 = position_x + (max(dr, ar))
        y0 = min(end_y, start_y)
        y1 = max(start_y, end_y)
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


NO_BOX = float("nan")


GlyphGeometry = tuple[
    list[Rectangle], list[Rectangle], list[Matrix6 | None], list[Rectangle], list[int], list[int]
]


def vertical_glyph_geometry(
    offsets: list[float],
    advances: list[float],
    positions: list[tuple[float, float]],
    *,
    basis: tuple[float, float, float, float, float, float],
    font_ascent: float,
    font_descent: float,
    rise: float,
    font_scale: float,
    advance_scale: float,
    clip_primary: Rectangle | None,
    clip_page: Rectangle | None,
    visible: bool,
    want_bitmap: list[int],
    want_transform: bool = True,
) -> GlyphGeometry:
    base_x, base_y, a, b, c, d = basis
    transform_a = advance_scale * a
    transform_b = advance_scale * b
    transform_c = font_scale * c
    transform_d = font_scale * d

    n = len(offsets)
    out_advance: list[Rectangle] = []
    out_baseline: list[Rectangle] = []
    out_transform: list[Matrix6 | None] = []
    out_ink: list[Rectangle] = []
    out_visible: list[int] = [0] * n
    out_bitmap: list[int] = [0] * (2 * n)

    for i in range(n):
        offset = offsets[i]
        position = positions[i]
        text_box, baseline_text = glyph_text_space_boxes(
            offset,
            advances[i],
            is_vertical=True,
            rise=rise,
            font_ascent=font_ascent,
            font_descent=font_descent,
            position=position,
        )
        advance_bbox = text_basis_rect(*text_box, basis)
        baseline = transformed_text_line(*baseline_text, basis)
        origin_x, position_y = position
        origin_y = rise + position_y - offset

        out_advance.append(advance_bbox)
        out_baseline.append(baseline)
        out_transform.append(
            (
                transform_a,
                transform_b,
                transform_c,
                transform_d,
                base_x + origin_x * a + origin_y * c,
                base_y + origin_x * b + origin_y * d,
            )
            if want_transform
            else None
        )
        out_ink.append(advance_bbox)

        vis = visible
        if vis:
            x0, y0, x1, y1 = advance_bbox
            for clip in (clip_primary, clip_page):
                if clip is not None and (
                    x1 <= clip[0] or x0 >= clip[2] or y1 <= clip[1] or y0 >= clip[3]
                ):
                    vis = False
                    break
        out_visible[i] = 1 if vis else 0
        if want_bitmap[i]:
            out_bitmap[2 * i] = 24
            out_bitmap[2 * i + 1] = 32

    return out_advance, out_baseline, out_transform, out_ink, out_visible, out_bitmap


GLYPH_BITMAP_REPAIR_LABELS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-+/()[]{}<>|_~"
)


SUSPICIOUS_GLYPH_BITMAP_TEXT = {"\ufffd", "\ufffc"}


GLYPH_BITMAP_LABELS = GLYPH_BITMAP_REPAIR_LABELS | SUSPICIOUS_GLYPH_BITMAP_TEXT


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
    punctuation = len(nonspace) - sum(map(str.isalnum, nonspace))
    return punctuation >= 1 and punctuation / len(nonspace) >= 0.25


DECODED_GLYPH_LAYOUT = SlotLayout(DecodedGlyph, DECODED_GLYPH_FIELDS)
OBSERVATION_LAYOUT = SlotLayout(GlyphObservation, OBSERVATION_FIELDS)


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


def glyph_width_source(
    glyph_width: Callable[[int], float],
) -> Callable[[int], float] | tuple[dict[Any, float], float]:
    """What capture_horizontal_glyphs reads widths from: the (widths, default) pair
    behind a plain FontMetricsModel.glyph_width, looked up without a call per
    glyph, or the callable itself."""
    metrics = getattr(glyph_width, "__self__", None)
    if (
        type(metrics) is FontMetricsModel
        and getattr(glyph_width, "__func__", None) is FontMetricsModel.glyph_width
        and type(metrics.widths) is dict
    ):
        return metrics.widths, metrics.default_width
    return glyph_width


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
    if not is_vertical:
        captured = capture_horizontal_glyphs(
            text,
            glyphs,
            DECODED_GLYPH_LAYOUT,
            OBSERVATION_LAYOUT,
            glyph_width_source(decoder.glyph_width),
            decoder.glyph_bbox if options.ink_bounds else None,
            font_size,
            char_space,
            word_space,
            horizontal_scale,
            options.render_details,
            options.text_runs,
            text_basis,
            font_ascent,
            font_descent,
            rise,
            font_scale,
            advance_scale,
            clip_bbox,
            page_clip,
            visible,
            style,
            seqno,
            effective_font_name,
            cluster_start,
            (CONFIDENCE_CACHE, glyph_unicode_confidence),
            should_capture_suspicious_multi_glyph_bitmap,
            GLYPH_BITMAP_LABELS,
            result.glyphs,
            result.clusters,
        )
        if captured is not None:
            kept_count, kept_advance, kept_ink, kept_confidence = captured
            if options.text_runs and kept_advance is not None and kept_ink is not None:
                geometry = result.geometry
                geometry.started = True
                geometry.advance = kept_advance
                geometry.ink = kept_ink
                geometry.confidence = kept_confidence
            result.cluster_count = kept_count
            return result
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
