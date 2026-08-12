# SPDX-License-Identifier: AGPL-3.0-only
"""Captured text and drawing primitives used by the compiled content state."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Any

from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.layout.glyphs import (
    GlyphCluster,
    GlyphObservation,
    glyph_cluster_from_observations,
    glyph_unicode_confidence,
)
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.text_helpers import (
    NO_SPACE_AFTER,
    NO_SPACE_BEFORE,
    can_merge_cross_font_word,
    gap_separator,
    normalize_extracted_text,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.engine.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.objects import PdfName

if typing.TYPE_CHECKING:
    pass

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
    if len(text) <= 1 or text in {"ff", "fi", "fl", "ffi", "ffl", "st"}:
        return False
    nonspace = [char for char in text if not char.isspace()]
    if len(nonspace) < 2:
        return False
    punctuation = sum(not char.isalnum() for char in nonspace)
    return punctuation >= 1 and punctuation / len(nonspace) >= 0.25


@lru_cache(maxsize=4096)
def glyph_bitmap_dimensions(
    glyph_bbox: tuple[float, float, float, float] | None,
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


def glyph_ink_rect(
    glyph_bbox: tuple[float, float, float, float] | None,
    advance_start: float,
    fallback: RectBox,
    text_basis: TextBasis,
    text_advance_scale: float,
    rise: float,
    font_scale: float,
) -> RectBox:
    if glyph_bbox is None:
        return fallback
    gx0, gy0, gx1, gy1 = glyph_bbox
    if gx1 <= gx0 or gy1 <= gy0:
        return fallback
    text_x0 = advance_start + gx0 * text_advance_scale
    text_x1 = advance_start + gx1 * text_advance_scale
    text_y0 = rise + gy0 * font_scale
    text_y1 = rise + gy1 * font_scale
    base_x, base_y, a, b, c, d = text_basis
    if b == 0.0 and c == 0.0:
        px0 = base_x + text_x0 * a
        px1 = base_x + text_x1 * a
        py0 = base_y + text_y0 * d
        py1 = base_y + text_y1 * d
        rect = RectBox(
            px0 if px0 < px1 else px1,
            py0 if py0 < py1 else py1,
            px1 if px1 > px0 else px0,
            py1 if py1 > py0 else py0,
            seqno=fallback.seqno,
            fill=fallback.fill,
            fill_opacity=fallback.fill_opacity,
        )
    else:
        p00_x = base_x + text_x0 * a + text_y0 * c
        p00_y = base_y + text_x0 * b + text_y0 * d
        p01_x = base_x + text_x0 * a + text_y1 * c
        p01_y = base_y + text_x0 * b + text_y1 * d
        p10_x = base_x + text_x1 * a + text_y0 * c
        p10_y = base_y + text_x1 * b + text_y0 * d
        p11_x = base_x + text_x1 * a + text_y1 * c
        p11_y = base_y + text_x1 * b + text_y1 * d
        rect = RectBox(
            min(p00_x, p01_x, p10_x, p11_x),
            min(p00_y, p01_y, p10_y, p11_y),
            max(p00_x, p01_x, p10_x, p11_x),
            max(p00_y, p01_y, p10_y, p11_y),
            seqno=fallback.seqno,
            fill=fallback.fill,
            fill_opacity=fallback.fill_opacity,
        )
    fallback_height = fallback.y1 - fallback.y0
    fallback_width = fallback.x1 - fallback.x0
    rect_height = rect.y1 - rect.y0
    rect_width = rect.x1 - rect.x0
    if rect_width <= 0.01 or rect_height <= 0.01:
        return fallback
    if fallback_width > 0.0 and rect_width > fallback_width * 4.0:
        return fallback
    if fallback_height > 0.0 and rect_height > fallback_height * 1.5:
        return fallback
    return rect


def transformed_text_rect(
    state: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text_basis: TextBasis,
) -> RectBox:
    base_x, base_y, a, b, c, d = text_basis
    if b == 0.0 and c == 0.0:
        px0 = base_x + x0 * a
        px1 = base_x + x1 * a
        py0 = base_y + y0 * d
        py1 = base_y + y1 * d
        return RectBox(
            px0 if px0 < px1 else px1,
            py0 if py0 < py1 else py1,
            px1 if px1 > px0 else px0,
            py1 if py1 > py0 else py0,
            seqno=state.sequence,
            fill=state.fill_color,
            fill_opacity=state.fill_opacity,
        )
    p00_x = base_x + x0 * a + y0 * c
    p00_y = base_y + x0 * b + y0 * d
    p01_x = base_x + x0 * a + y1 * c
    p01_y = base_y + x0 * b + y1 * d
    p10_x = base_x + x1 * a + y0 * c
    p10_y = base_y + x1 * b + y0 * d
    p11_x = base_x + x1 * a + y1 * c
    p11_y = base_y + x1 * b + y1 * d
    return RectBox(
        min(p00_x, p01_x, p10_x, p11_x),
        min(p00_y, p01_y, p10_y, p11_y),
        max(p00_x, p01_x, p10_x, p11_x),
        max(p00_y, p01_y, p10_y, p11_y),
        seqno=state.sequence,
        fill=state.fill_color,
        fill_opacity=state.fill_opacity,
    )


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
    state: Any,
    offset: float,
    advance: float,
    decoder: Any,
    position: tuple[float, float] = (0.0, 0.0),
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    if decoder.is_vertical:
        position_x, position_y = position
        start_y = state.rise + position_y - offset
        end_y = start_y - advance
        ar = state.font_ascent
        dr = state.font_descent
        x0 = position_x + (dr if dr < ar else ar)
        x1 = position_x + (ar if ar > dr else dr)
        y0 = end_y if end_y < start_y else start_y
        y1 = start_y if start_y > end_y else end_y
        return (
            (x0, y0, x1, y1),
            (0.0, start_y, 0.0, end_y),
        )
    ar = state.font_ascent + state.rise
    dr = state.font_descent + state.rise
    return (
        (offset, dr, offset + advance, ar),
        (offset, state.rise, offset + advance, state.rise),
    )


def type3_glyph_names(font: dict[Any, Any], decoder: Any) -> dict[int, str]:
    encoding = lookup_dict_key(font, "Encoding")
    differences_obj = (
        lookup_dict_key(encoding, "Differences") if isinstance(encoding, dict) else None
    )
    glyph_names = {code: name for code, name in enumerate(StandardEncoding) if name != ".notdef"}
    if decoder.base_encoding == "MacRomanEncoding":
        from core_pdf._vendor.fontTools.encodings.MacRoman import MacRoman

        glyph_names = {code: name for code, name in enumerate(MacRoman) if name != ".notdef"}
    if isinstance(differences_obj, (list, tuple)):
        code = 0
        for item in differences_obj:
            if type(item) is int:
                code = item
                continue
            name = str(item) if isinstance(item, PdfName) else None
            if name is not None and 0 <= code <= 255:
                glyph_names[code] = name
                code += 1
    return glyph_names


def apply_glyph_geometry_to_run(
    run: TextRun,
    glyphs: typing.Iterable[GlyphObservation],
    glyph_clusters: tuple[GlyphCluster, ...] = (),
) -> None:
    if glyph_clusters:
        run.glyph_clusters = glyph_clusters
    advance_bbox: tuple[float, float, float, float] | None = None
    ink_bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    for glyph in glyphs:
        advance = glyph.advance_rect
        advance_values = (advance.x0, advance.y0, advance.x1, advance.y1)
        if advance_bbox is None:
            advance_bbox = advance_values
        else:
            x0, y0, x1, y1 = advance_bbox
            advance_bbox = (
                min(x0, advance.x0),
                min(y0, advance.y0),
                max(x1, advance.x1),
                max(y1, advance.y1),
            )

        ink = glyph.ink_rect
        ink_values = (ink.x0, ink.y0, ink.x1, ink.y1)
        if ink_bbox is None:
            ink_bbox = ink_values
        else:
            x0, y0, x1, y1 = ink_bbox
            ink_bbox = (
                min(x0, ink.x0),
                min(y0, ink.y0),
                max(x1, ink.x1),
                max(y1, ink.y1),
            )

        glyph_confidence = glyph.confidence
        if glyph_confidence is not None and (confidence is None or glyph_confidence < confidence):
            confidence = glyph_confidence

    if advance_bbox is not None:
        run.advance_bbox = advance_bbox
    if ink_bbox is not None:
        run.ink_bbox = ink_bbox
    if confidence is not None:
        run.confidence = confidence


def type3_font_matrix(font: dict[str, Any]) -> Matrix:
    matrix = lookup_dict_key(font, "FontMatrix")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 6:
        return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
    values: list[float] = []
    for value in matrix:
        if type(value) not in (int, float):
            return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
        values.append(float(typing.cast(Any, value)))
    return Matrix(*values)


class CapturedLine:
    __slots__ = ("x0", "y0", "x1", "y1", "line_width")

    def __init__(self, x0: float, y0: float, x1: float, y1: float, line_width: float = 1.0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.line_width = line_width

    def replace(self, **kwargs: Any) -> CapturedLine:
        return CapturedLine(
            x0=kwargs.get("x0", self.x0),
            y0=kwargs.get("y0", self.y0),
            x1=kwargs.get("x1", self.x1),
            y1=kwargs.get("y1", self.y1),
            line_width=kwargs.get("line_width", self.line_width),
        )


@dataclass(frozen=True, slots=True)
class CapturedInlineImage:
    """Typed inline-image product emitted by content interpretation."""

    seqno: int
    dictionary: dict[Any, Any]
    data: bytes
    image_source: ImageSource
    image_clip: tuple[float, float, float, float] | None
    ctm: Matrix
    xobject_depth: int


class CapturedSubpath:
    __slots__ = ("points", "closed")

    def __init__(
        self,
        points: list[tuple[float, float]] | None = None,
        *,
        closed: bool = False,
    ) -> None:
        self.points = points if points is not None else []
        self.closed = closed

    def clone(self) -> CapturedSubpath:
        return CapturedSubpath(list(self.points), closed=self.closed)

    def transformed(self, matrix: Matrix) -> CapturedSubpath:
        a, b, c, d, e, f = matrix
        return CapturedSubpath(
            [(x * a + y * c + e, x * b + y * d + f) for x, y in self.points],
            closed=self.closed,
        )

    def translated(self, tx: float, ty: float) -> CapturedSubpath:
        return CapturedSubpath(
            [(x + tx, y + ty) for x, y in self.points],
            closed=self.closed,
        )

    def line_to(self, x: float, y: float) -> None:
        if self.closed:
            return
        self.points.append((x, y))

    def close(self) -> None:
        if len(self.points) > 1:
            self.closed = True

    def has_segments(self) -> bool:
        return len(self.points) > 1

    def bbox(self) -> tuple[float, float, float, float] | None:
        if not self.points:
            return None
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def edges(self, *, close_open: bool = False) -> list[tuple[float, float, float, float]]:
        if len(self.points) < 2:
            return []
        edges = [(x0, y0, x1, y1) for (x0, y0), (x1, y1) in zip(self.points, self.points[1:])]
        first = self.points[0]
        last = self.points[-1]
        if (self.closed or close_open) and first != last:
            edges.append((last[0], last[1], first[0], first[1]))
        return edges


class CapturedPath:
    __slots__ = ("subpaths",)

    def __init__(self, subpaths: list[CapturedSubpath] | None = None) -> None:
        self.subpaths = subpaths if subpaths is not None else []

    def clone(self) -> CapturedPath:
        return CapturedPath([subpath.clone() for subpath in self.subpaths])

    def transformed(self, matrix: Matrix) -> CapturedPath:
        return CapturedPath([subpath.transformed(matrix) for subpath in self.subpaths])

    def translated(self, tx: float, ty: float) -> CapturedPath:
        return CapturedPath([subpath.translated(tx, ty) for subpath in self.subpaths])

    def clear(self) -> None:
        self.subpaths.clear()

    def move_to(self, x: float, y: float) -> None:
        self.subpaths.append(CapturedSubpath([(x, y)]))

    def line_to(self, x: float, y: float) -> None:
        subpaths = self.subpaths
        if not subpaths:
            self.move_to(x, y)
            return
        subpath = subpaths[-1]
        if not subpath.closed:
            subpath.points.append((x, y))

    def close(self) -> None:
        if self.subpaths:
            self.subpaths[-1].close()

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.subpaths.append(
            CapturedSubpath(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                closed=True,
            )
        )

    def has_segments(self) -> bool:
        return any(subpath.has_segments() for subpath in self.subpaths)

    def bbox(self) -> tuple[float, float, float, float] | None:
        boxes = [box for subpath in self.subpaths if (box := subpath.bbox())]
        if not boxes:
            return None
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def fill_edges(self) -> list[tuple[float, float, float, float]]:
        edges: list[tuple[float, float, float, float]] = []
        for subpath in self.subpaths:
            edges.extend(subpath.edges(close_open=True))
        return edges

    def derived_lines(self, line_width: float) -> list[CapturedLine]:
        lines: list[CapturedLine] = []
        append_line = lines.append
        for subpath in self.subpaths:
            points = subpath.points
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                if abs(x1 - x0) > 0.01 or abs(y1 - y0) > 0.01:
                    append_line(CapturedLine(x0, y0, x1, y1, line_width))
        return lines


DrawingItem = tuple[str, tuple[tuple[float, float], ...]]
internal_EMPTY_DRAWING_ITEMS: tuple[DrawingItem, ...] = ()


class CapturedDrawing:
    __slots__ = (
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "image_source",
        "image_clip",
        "items",
        "path",
        "bbox",
        "internal_rect_cache",
        "kind",
    )

    def __init__(
        self,
        seqno: int,
        fill: tuple[float, ...] | None,
        fill_opacity: float | None,
        fill_pattern: Mapping[object, object] | None = None,
        stroke_color: tuple[float, ...] | None = None,
        stroke_pattern: Mapping[object, object] | None = None,
        stroke_opacity: float | None = None,
        line_width: float = 1.0,
        line_cap: int = 0,
        line_join: int = 0,
        dash_pattern: tuple[list[float], float] | None = None,
        fill_rule: str = "nonzero",
        blend_mode: str | None = None,
        soft_mask_alpha: float | None = None,
        raw_data: bytes | memoryview | None = None,
        dictionary: dict[Any, Any] | None = None,
        image_source: ImageSource | None = None,
        image_clip: tuple[float, float, float, float] | None = None,
        kind: str = "fill",
        items: list[DrawingItem] | None = None,
        path: CapturedPath | None = None,
        bbox: RectBox | None = None,
    ) -> None:
        self.seqno = seqno
        self.fill = fill
        self.fill_pattern = fill_pattern
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_pattern = stroke_pattern
        self.stroke_opacity = stroke_opacity
        self.line_width = line_width
        self.line_cap = line_cap
        self.line_join = line_join
        self.dash_pattern = dash_pattern
        self.fill_rule = fill_rule
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.raw_data = raw_data
        self.dictionary = dictionary
        self.image_source = image_source
        self.image_clip = image_clip
        self.kind = kind
        self.items = items if items else internal_EMPTY_DRAWING_ITEMS
        self.path = path
        self.bbox = bbox
        self.internal_rect_cache: RectBox | None = None

    def replace(self, **kwargs: Any) -> CapturedDrawing:
        return CapturedDrawing(
            seqno=kwargs.get("seqno", self.seqno),
            fill=kwargs.get("fill", self.fill),
            fill_pattern=kwargs.get("fill_pattern", self.fill_pattern),
            fill_opacity=kwargs.get("fill_opacity", self.fill_opacity),
            stroke_color=kwargs.get("stroke_color", self.stroke_color),
            stroke_pattern=kwargs.get("stroke_pattern", self.stroke_pattern),
            stroke_opacity=kwargs.get("stroke_opacity", self.stroke_opacity),
            line_width=kwargs.get("line_width", self.line_width),
            line_cap=kwargs.get("line_cap", self.line_cap),
            line_join=kwargs.get("line_join", self.line_join),
            dash_pattern=kwargs.get("dash_pattern", self.dash_pattern),
            fill_rule=kwargs.get("fill_rule", self.fill_rule),
            blend_mode=kwargs.get("blend_mode", self.blend_mode),
            soft_mask_alpha=kwargs.get("soft_mask_alpha", self.soft_mask_alpha),
            raw_data=kwargs.get("raw_data", self.raw_data),
            dictionary=kwargs.get("dictionary", self.dictionary),
            image_source=kwargs.get("image_source", self.image_source),
            image_clip=kwargs.get("image_clip", self.image_clip),
            kind=kwargs.get("kind", self.kind),
            items=kwargs.get("items", self.items),
            path=kwargs.get("path", self.path),
            bbox=kwargs.get("bbox", self.bbox),
        )

    @property
    def rect(self) -> RectBox | None:
        cached = self.internal_rect_cache
        if cached is not None:
            return cached
        if self.bbox is not None:
            rect = self.bbox.normalize()
            self.internal_rect_cache = rect
            return rect
        if self.path is None:
            return None
        bbox = self.path.bbox()
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        rect = RectBox(
            x0,
            y0,
            x1,
            y1,
            seqno=self.seqno,
            fill=self.fill,
            fill_opacity=self.fill_opacity,
        )
        self.internal_rect_cache = rect
        return rect


__all__ = (
    "CapturedDrawing",
    "CapturedInlineImage",
    "CapturedLine",
    "CapturedPath",
    "CapturedSubpath",
    "NO_SPACE_AFTER",
    "NO_SPACE_BEFORE",
    "can_merge_cross_font_word",
    "gap_separator",
    "glyph_cluster_from_observations",
    "glyph_unicode_confidence",
    "normalize_extracted_text",
)
