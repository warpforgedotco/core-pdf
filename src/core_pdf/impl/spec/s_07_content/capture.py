# SPDX-License-Identifier: AGPL-3.0-only
"""Captured text and drawing primitives used by the compiled content state."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil
from typing import Any

from core_pdf.impl.model.geometry import RectBox, bbox_union
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
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
    if len(text) <= 1 or text in {"ff", "fi", "fl", "ffi", "ffl", "st"}:
        return False
    nonspace = [char for char in text if not char.isspace()]
    if len(nonspace) < 2:
        return False
    punctuation = sum(not char.isalnum() for char in nonspace)
    return punctuation >= 1 and punctuation / len(nonspace) >= 0.25


@lru_cache(maxsize=4096)
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
    base_x, base_y, a, b, c, d = text_basis
    if b == 0.0 and c == 0.0:
        px0 = base_x + text_x0 * a
        px1 = base_x + text_x1 * a
        py0 = base_y + text_y0 * d
        py1 = base_y + text_y1 * d
        rect = (
            px0 if px0 < px1 else px1,
            py0 if py0 < py1 else py1,
            px1 if px1 > px0 else px0,
            py1 if py1 > py0 else py0,
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
        rect = (
            min(p00_x, p01_x, p10_x, p11_x),
            min(p00_y, p01_y, p10_y, p11_y),
            max(p00_x, p01_x, p10_x, p11_x),
            max(p00_y, p01_y, p10_y, p11_y),
        )
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
    Rectangle,
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
    """Project the decoder's normalized simple-font encoding onto byte codes."""
    # Keep ``font`` in the API until Type 3 program-cache ownership moves out of
    # FontDecoder. The decoder was constructed from this dictionary and already
    # normalized its base encoding and Differences.
    del font
    return {
        code: name
        for code in range(256)
        if (name := decoder.internal_simple_glyph_name(code)) not in {"", ".notdef"}
    }


class PdfminerCursor:
    """pdfminer-compatible text cursor tracked alongside spec glyph capture.

    pdfminer advances its own cursor per glyph and reports each glyph's origin
    from it. That bookkeeping is a compatibility concern, not a spec one, so it
    lives here rather than threaded through the capture loop as eight locals.
    """

    __slots__ = (
        "x",
        "y",
        "need_charspace",
        "char_space",
        "word_space",
        "spacing_scale",
        "origin_x",
        "origin_y",
        "combined",
        "is_vertical",
        "font_size",
    )

    def __init__(
        self,
        x: float,
        y: float,
        need_charspace: bool,
        *,
        char_space: float,
        word_space: float,
        spacing_scale: float,
        origin_x: float,
        origin_y: float,
        combined: tuple[float, float, float, float],
        is_vertical: bool,
        font_size: float,
    ) -> None:
        self.x = x
        self.y = y
        self.need_charspace = need_charspace
        self.char_space = char_space
        self.word_space = word_space
        self.spacing_scale = spacing_scale
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.combined = combined
        self.is_vertical = is_vertical
        self.font_size = font_size

    def step(self, width_units: float, is_space: bool) -> tuple[tuple[str, Any], ...]:
        """Apply pending char spacing, report this glyph's provenance, advance."""
        if self.need_charspace:
            if self.is_vertical:
                self.y += self.char_space
            else:
                self.x += self.char_space
        combined_a, combined_b, combined_c, combined_d = self.combined
        provenance: tuple[tuple[str, Any], ...] = (
            (
                "pdfminer_origin",
                (
                    self.x * combined_a + self.y * combined_c + self.origin_x,
                    self.x * combined_b + self.y * combined_d + self.origin_y,
                ),
            ),
            ("pdfminer_matrix_origin", (self.origin_x, self.origin_y)),
            ("pdfminer_cursor", (self.x, self.y)),
            ("pdfminer_need_charspace", self.need_charspace),
        )
        advance = width_units * 0.001 * self.font_size * self.spacing_scale
        if self.is_vertical:
            self.y += advance
            if is_space:
                self.y += self.word_space
        else:
            self.x += advance
            if is_space:
                self.x += self.word_space
        self.need_charspace = True
        return provenance


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
        current = self.confidence
        if confidence is not None and (current is None or confidence < current):
            self.confidence = confidence


def type3_font_matrix(font: dict[str, Any]) -> Matrix:
    try:
        return Matrix.from_operand(font.get("FontMatrix"))
    except ValueError:
        return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)


class CapturedLine:
    __slots__ = ("x0", "y0", "x1", "y1", "line_width")

    def __init__(self, x0: float, y0: float, x1: float, y1: float, line_width: float = 1.0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.line_width = line_width


@dataclass(frozen=True, slots=True)
class CapturedInlineImage:
    """Typed inline-image product emitted by content interpretation."""

    seqno: int
    dictionary: dict[Any, Any]
    data: bytes
    image_source: ImageSource
    image_clip: Rectangle | None
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

    def transformed(self, matrix: Matrix | Sequence[float]) -> CapturedSubpath:
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

    def bbox(self) -> Rectangle | None:
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

    def axis_aligned_rect(self) -> Rectangle | None:
        """The rectangle this path draws when it is exactly one axis-aligned box.

        One segment-bearing subpath (empty ``m``-only subpaths are ignored), four
        corners in either winding, closed explicitly or by repeating the first
        point, and a positive area. Both the router's "simple vector rectangle"
        test and the rasterizer's rect fast path rely on this one definition.
        """
        segment_subpaths = [subpath for subpath in self.subpaths if subpath.has_segments()]
        if len(segment_subpaths) != 1 or self.subpaths[-1] is not segment_subpaths[0]:
            return None
        subpath = segment_subpaths[0]
        points = list(subpath.points)
        if len(points) >= 2 and points[0] == points[-1]:
            points.pop()
        if len(points) != 4:
            return None
        if not subpath.closed and subpath.points[0] != subpath.points[-1]:
            return None
        xs = {point[0] for point in points}
        ys = {point[1] for point in points}
        if len(xs) != 2 or len(ys) != 2:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 <= x0 or y1 <= y0:
            return None
        if set(points) != {(x0, y0), (x0, y1), (x1, y0), (x1, y1)}:
            return None
        for (px0, py0), (px1, py1) in zip(points, points[1:] + points[:1], strict=False):
            if px0 != px1 and py0 != py1:
                return None
        return (x0, y0, x1, y1)

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.subpaths.append(
            CapturedSubpath(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                closed=True,
            )
        )

    def has_segments(self) -> bool:
        return any(subpath.has_segments() for subpath in self.subpaths)

    def bbox(self) -> Rectangle | None:
        return bbox_union(box for subpath in self.subpaths if (box := subpath.bbox()))

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


StrokeStyleKey = tuple[
    tuple[float, ...] | None,
    float,
    float,
    int,
    int,
    tuple[tuple[float, ...], float] | None,
    str | None,
    float | None,
]


@dataclass(slots=True)
class CapturedDrawing:
    seqno: int
    fill: tuple[float, ...] | None
    fill_opacity: float | None
    fill_pattern: Mapping[object, object] | None = None
    stroke_color: tuple[float, ...] | None = None
    stroke_pattern: Mapping[object, object] | None = None
    stroke_opacity: float | None = None
    line_width: float = 1.0
    line_cap: int = 0
    line_join: int = 0
    dash_pattern: tuple[list[float], float] | None = None
    fill_rule: str = "nonzero"
    blend_mode: str | None = None
    soft_mask_alpha: float | None = None
    raw_data: bytes | memoryview | None = None
    dictionary: dict[Any, Any] | None = None
    image_source: ImageSource | None = None
    image_clip: Rectangle | None = None
    kind: str = "fill"
    items: tuple[DrawingItem, ...] | list[DrawingItem] = internal_EMPTY_DRAWING_ITEMS
    path: CapturedPath | None = None
    bbox: RectBox | None = None
    stream_order: int = 0
    xobject_depth: int = 0
    internal_rect_cache: RectBox | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.items:
            self.items = internal_EMPTY_DRAWING_ITEMS

    def replace(self, **kwargs: Any) -> CapturedDrawing:
        return dataclasses.replace(self, **kwargs)

    def stroke_style_key(self) -> StrokeStyleKey | None:
        """Hashable stroke paint style, or None when a pattern paints the stroke.

        Colour, opacity (1.0 when unset), width, cap, join, normalised dash,
        blend mode and soft-mask alpha, in that order. Consumers that group
        strokes by style key off this tuple and layer their own filters on top.
        """
        if self.stroke_pattern is not None:
            return None
        color = self.stroke_color
        dash = self.dash_pattern
        return (
            tuple(float(component) for component in color) if color is not None else None,
            1.0 if self.stroke_opacity is None else float(self.stroke_opacity),
            float(self.line_width),
            int(self.line_cap or 0),
            int(self.line_join or 0),
            (tuple(float(value) for value in dash[0]), float(dash[1])) if dash else None,
            self.blend_mode,
            self.soft_mask_alpha,
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
)
