# SPDX-License-Identifier: AGPL-3.0-only
"""Captured text and drawing primitives used by the compiled content state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from core_pdf.impl.model.geometry import RectBox, bbox_union, points_bbox
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.types import Rectangle


def type3_glyph_names(decoder: FontDecoder) -> dict[int, str]:
    """Project the decoder's normalized simple-font encoding onto byte codes."""
    return {
        code: name
        for code in range(256)
        if (name := decoder.internal_simple_glyph_name(code)) not in {"", ".notdef"}
    }


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
    blend_mode: str | None = None
    soft_mask_alpha: float | None = None
    stream_order: int = 0


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

    def close(self) -> None:
        if len(self.points) > 1:
            self.closed = True

    def has_segments(self) -> bool:
        return len(self.points) > 1

    def bbox(self) -> Rectangle | None:
        return points_bbox(self.points)

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
    fill_pattern: PatternPaint | None = None
    stroke_color: tuple[float, ...] | None = None
    stroke_pattern: PatternPaint | None = None
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

    def __post_init__(self) -> None:
        if not self.items:
            self.items = internal_EMPTY_DRAWING_ITEMS

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
        if self.bbox is not None:
            return self.bbox.normalize()
        if self.path is None:
            return None
        bbox = self.path.bbox()
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        return RectBox(
            x0,
            y0,
            x1,
            y1,
            seqno=self.seqno,
            fill=self.fill,
            fill_opacity=self.fill_opacity,
        )


def marker_drawing(
    kind: str,
    seqno: int,
    *,
    fill_opacity: float | None = None,
    blend_mode: str | None = None,
) -> CapturedDrawing:
    """A zero-geometry drawing that only marks a scope boundary in page order.

    Used for the `state-push`/`state-pop` clip scopes and the
    `group-begin`/`group-end` transparency-group boundaries.
    """
    return CapturedDrawing(
        seqno=seqno,
        fill=None,
        fill_opacity=fill_opacity,
        blend_mode=blend_mode,
        kind=kind,
    )


@dataclass(frozen=True, slots=True)
class ShadingPattern:
    """A PatternType 2 paint: the resolved /Shading dictionary."""

    dictionary: dict[Any, Any]


@dataclass(frozen=True, slots=True)
class TilingPattern:
    """A PatternType 1 paint: one captured cell plus the step to repeat it by.

    Holds the captured records themselves. This used to be a nested dict of
    string keys cast to PdfDict, which meant the renderer re-parsed by key and
    every projected field had to be listed by hand on both sides.
    """

    bbox: Rectangle
    x_step: float
    y_step: float
    drawings: list[CapturedDrawing]
    glyphs: list[GlyphObservation]
    inline_images: list[CapturedInlineImage]


PatternPaint: TypeAlias = ShadingPattern | TilingPattern


__all__ = (
    "CapturedDrawing",
    "CapturedInlineImage",
    "CapturedLine",
    "CapturedPath",
    "CapturedSubpath",
    "PatternPaint",
    "ShadingPattern",
    "TilingPattern",
)
