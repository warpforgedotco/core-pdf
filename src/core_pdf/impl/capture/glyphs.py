# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from core_pdf.impl.capture.glyph_boxes import (
    glyph_text_space_boxes,
    text_basis_rect,
    transformed_text_line,
)
from core_pdf.impl.capture.glyph_geometry import NO_BOX, vertical_glyph_geometry
from core_pdf.impl.fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.fonts.font_program import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.model.glyphs import (
    GlyphClusterLike,
    GlyphObservation,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
    min_optional_confidence,
)
from core_pdf.impl.types import Rectangle
from core_pdf_cythonized import horizontal_glyph_geometry

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
    clusters: list[GlyphClusterLike] = field(default_factory=list)
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
    capture_render_details: bool = True,
) -> GlyphCapture:
    """Lay out and record one show-text operation's glyphs.

    Three passes. The first walks the glyphs for everything that needs the
    decoder or the source text; the second is pure float arithmetic over flat
    arrays and is the compiled kernel; the third builds the observations. The
    split exists so the middle pass touches no Python object, which is the only
    shape a compiled kernel beats the interpreter at -- a fused loop that
    interleaved the arithmetic with building a forty-field observation would
    not qualify.
    """
    result = GlyphCapture()
    if not glyphs:
        return result
    text_basis = geometry.basis
    font_size = geometry.font_size
    font_scale = geometry.font_scale
    font_ascent = geometry.font_ascent
    font_descent = geometry.font_descent
    rise = geometry.rise
    advance_scale = geometry.advance_scale
    effective_font_name = decoder.font_name or font_name
    is_vertical = decoder.is_vertical
    char_space = geometry.char_space
    word_space = geometry.word_space
    horizontal_scale = geometry.horizontal_scale
    glyph_width = decoder.glyph_width
    glyph_bbox_for_code = decoder.glyph_bbox
    vertical_position = decoder.vertical_glyph_position
    want_ink = capture_ink_bounds and not is_vertical
    # The glyph transform and the bitmap request exist for the rasterizer.
    # A caller that only wants text says so, and neither is computed.
    want_render = capture_render_details

    # ---- pass one: the decoder and the source text ------------------------
    kept: list[DecodedGlyph] = []
    chunk_texts: list[str] = []
    offsets: list[float] = []
    advances: list[float] = []
    glyph_boxes: list[float] = []
    want_bitmap: list[int] = []
    suspicious_flags: list[bool] = []
    positions: list[tuple[float, float]] = []
    offset = 0.0
    cursor = 0
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
        # Computed whichever mode this is: suspicious feeds split_flags below,
        # so gating it on render details would split a multi-character glyph
        # like "A/B" into three observations for a text-only capture and leave
        # it as one otherwise. Only the bitmap request is a render concern.
        suspicious = (
            False if chunk_length == 1 else should_capture_suspicious_multi_glyph_bitmap(chunk_text)
        )
        suspicious_flags.append(suspicious)
        want_bitmap.append(
            1 if want_render and (should_capture_glyph_bitmap(chunk_text) or suspicious) else 0
        )
        if is_vertical:
            positions.append(vertical_position(glyph.cid, font_size=font_size))
        offset += advance

    if not kept:
        return result

    # ---- pass two: the geometry kernel ------------------------------------
    if is_vertical:
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
            clip_primary=paint.clip_bbox,
            clip_page=paint.page_clip,
            visible=visible,
            want_bitmap=want_bitmap,
            want_transform=want_render,
        )
    else:
        advance_f, baseline_f, transform_f, ink_f, visible_f, bitmap_f = horizontal_glyph_geometry(
            offsets,
            advances,
            glyph_boxes,
            basis=text_basis,
            font_ascent=font_ascent,
            font_descent=font_descent,
            rise=rise,
            font_scale=font_scale,
            advance_scale=advance_scale,
            font_size=font_size,
            clip_primary=paint.clip_bbox,
            clip_page=paint.page_clip,
            visible=visible,
            want_bitmap=want_bitmap,
            want_transform=want_render,
        )

    # ---- pass three: the observations -------------------------------------
    add_run_geometry = result.geometry.add
    append_glyph = result.glyphs.append
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

        cluster_id = cluster_start + result.cluster_count
        cluster_provenance_id = (seqno, cluster_id)
        observation_confidence = glyph_unicode_confidence(
            chunk_text,
            glyph.unicode_source,
            glyph.alternates,
        )

        fragments: list[tuple[str, Rectangle, Rectangle, Rectangle, float]] = []
        if glyph.split_unicode and chunk_length != 1 and not suspicious_flags[index]:
            # One code that stands for several characters: re-cut the advance
            # per character. Rare, and the kernel deliberately does not model it.
            per_char_advance = advances[index] / chunk_length
            char_offset = offsets[index]
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
        for position, (
            fragment_text,
            ink,
            advance_rect,
            fragment_baseline,
            confidence,
        ) in enumerate(fragments):
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
                position == 0,
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
            append_glyph(observation)
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
    return result
