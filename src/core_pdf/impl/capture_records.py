# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload

import numpy

from core_pdf.impl.geometry import bbox_union, normalize_rect, points_bbox
from core_pdf.impl.types import GeneratedRecord, RecordType, Rectangle, ReplaceFields, ReprFields
from core_pdf_cythonized import fill_edge_rows, path_bounds
from core_pdf_spec.s_07_content.streams import StreamKey
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator

if TYPE_CHECKING:
    from core_pdf.impl.capture_program import CapturedProgram

LayoutFormId: TypeAlias = tuple[tuple[StreamKey | None, Rectangle | None], ...] | None


class CapturedSoftMask(GeneratedRecord, eq=False):
    program: CapturedProgram
    transfer: PdfFunctionEvaluator | None = None
    offset: tuple[float, float] = (0.0, 0.0)


class CapturedTextBoundary(GeneratedRecord):
    seqno: int
    kind: Literal["begin", "end", "glyph-begin", "glyph-end", "stream-begin", "stream-end"]
    knockout: bool = True


class CapturedLine(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False, eq=False):
    x0: float
    y0: float
    x1: float
    y1: float
    line_width: float = 1.0


class CapturedLines(Sequence[CapturedLine]):
    __slots__ = ("array",)

    array: numpy.ndarray[Any, numpy.dtype[numpy.float64]]

    def __init__(self, lines: Iterable[CapturedLine] = ()) -> None:
        rows = [(line.x0, line.y0, line.x1, line.y1, line.line_width) for line in lines]
        self.array = read_only(numpy.array(rows, dtype=numpy.float64).reshape(-1, 5))

    @classmethod
    def from_array(cls, array: numpy.ndarray[Any, Any]) -> CapturedLines:
        lines = cls.__new__(cls)
        lines.array = read_only(array)
        return lines

    @classmethod
    def concatenate(cls, parts: Iterable[CapturedLines]) -> CapturedLines:
        arrays = [part.array for part in parts]
        if not arrays:
            return EMPTY_LINES
        if len(arrays) == 1:
            return cls.from_array(arrays[0])
        return cls.from_array(numpy.concatenate(arrays))

    def columns(
        self,
    ) -> tuple[
        numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        numpy.ndarray[Any, numpy.dtype[numpy.float64]],
        numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    ]:
        array = self.array
        return array[:, 0], array[:, 1], array[:, 2], array[:, 3]

    def __len__(self) -> int:
        return len(self.array)

    @overload
    def __getitem__(self, index: int) -> CapturedLine: ...
    @overload
    def __getitem__(self, index: slice) -> CapturedLines: ...
    def __getitem__(self, index: int | slice) -> CapturedLine | CapturedLines:
        if isinstance(index, slice):
            return CapturedLines.from_array(self.array[index])
        return CapturedLine(*self.array[index].tolist())

    def __iter__(self) -> Iterator[CapturedLine]:
        for x0, y0, x1, y1, line_width in self.array.tolist():
            yield CapturedLine(x0, y0, x1, y1, line_width)

    def __eq__(self, other: object) -> bool:
        if type(other) is not CapturedLines:
            return NotImplemented
        return self.array.shape == other.array.shape and bytes(self.array) == bytes(other.array)

    def __hash__(self) -> int:
        return hash(bytes(self.array))

    def __repr__(self) -> str:
        return f"CapturedLines(<{len(self.array)} lines>)"


def read_only(array: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    array.setflags(write=False)
    return array


EMPTY_LINES = CapturedLines()


class CapturedInlineImage(GeneratedRecord):
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
    fill: tuple[float, ...] | None = None
    fill_opacity: float | None = None
    paints: bool = True
    alpha_is_shape: bool = False
    graphics_soft_mask: CapturedSoftMask | None = None


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


DeferredPoints: TypeAlias = tuple[Any, Any, list[tuple[int, int, bool]], bool]


class CapturedPath:
    __slots__ = ("subpaths", "_deferred", "_parts", "_summary")

    def __init__(self, subpaths: list[CapturedSubpath] | None = None) -> None:
        self.subpaths = subpaths if subpaths is not None else []
        self._deferred: DeferredPoints | None = None
        self._parts: list[DeferredPoints] | None = None
        self._summary: tuple[Rectangle | None, bool] | None = None

    @classmethod
    def deferred_outline(
        cls,
        column_x: Any,
        column_y: Any,
        spans: list[tuple[int, int, bool]],
    ) -> CapturedPath:
        path = cls.__new__(cls)
        path._deferred = (column_x, column_y, spans, True)
        path._parts = None
        path._summary = None
        return path

    @classmethod
    def deferred_flattened(
        cls,
        column_x: Any,
        column_y: Any,
        spans: list[tuple[int, int, bool]],
        bbox: Rectangle | None,
        has_segments: bool,
    ) -> CapturedPath:
        path = cls.__new__(cls)
        path._deferred = (column_x, column_y, spans, False)
        path._parts = None
        path._summary = (bbox, has_segments)
        return path

    def coalesced_with(self, other: CapturedPath) -> CapturedPath | None:
        mine = self.flattened_parts()
        theirs = other.flattened_parts()
        if mine is None or theirs is None:
            return None
        path = CapturedPath.__new__(CapturedPath)
        path._deferred = None
        path._parts = [*mine, *theirs]
        path._summary = None
        return path

    def flattened_parts(self) -> list[DeferredPoints] | None:
        if self._parts is not None:
            return self._parts
        deferred = self._deferred
        if deferred is None or deferred[3]:
            return None
        return [deferred]

    def subpath_count(self) -> int:
        if self._parts is not None:
            return sum(len(part[2]) for part in self._parts)
        deferred = self._deferred
        if deferred is not None:
            return len(deferred[2])
        return len(self.subpaths)

    def __getattr__(self, name: str) -> Any:
        if name == "subpaths":
            deferred = self.deferred_columns()
            if deferred is not None:
                column_x, column_y, spans, outline = deferred
                xs = column_x.tolist()
                ys = column_y.tolist()
                subpaths = [
                    CapturedSubpath(
                        list(zip(xs[start:end], ys[start:end], strict=True)),
                        closed=outline or flag,
                    )
                    for start, end, flag in spans
                ]
                self.subpaths = subpaths
                self._deferred = None
                self._summary = None
                return subpaths
        raise AttributeError(name)

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
        deferred = self.deferred_columns()
        if deferred is not None:
            column_x, column_y, spans, outline = deferred
            if not spans:
                return None
            start, end, flag = spans[-1]
            if outline and (len(spans) != 1 or end - start != 4):
                return None
            if end - start < 2 or any(e - s > 1 for s, e, _ in spans[:-1]):
                return None
            if end - start > 5:
                return None
            points = list(
                zip(column_x[start:end].tolist(), column_y[start:end].tolist(), strict=True)
            )
            return rect_from_points(points, outline or flag)
        segment_subpaths = [subpath for subpath in self.subpaths if subpath.has_segments()]
        if len(segment_subpaths) != 1 or self.subpaths[-1] is not segment_subpaths[0]:
            return None
        subpath = segment_subpaths[0]
        return rect_from_points(subpath.points, subpath.closed)

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.subpaths.append(
            CapturedSubpath(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                closed=True,
            )
        )

    def has_segments(self) -> bool:
        summary = self.deferred_summary()
        if summary is not None:
            return summary[1]
        return any(subpath.has_segments() for subpath in self.subpaths)

    def bbox(self) -> Rectangle | None:
        summary = self.deferred_summary()
        if summary is not None:
            return summary[0]
        return bbox_union(box for subpath in self.subpaths if (box := subpath.bbox()))

    def deferred_summary(self) -> tuple[Rectangle | None, bool] | None:
        if self._summary is not None and (self._deferred is not None or self._parts is not None):
            return self._summary
        deferred = self.deferred_columns()
        if deferred is None:
            return None
        self._summary = path_bounds(deferred[0], deferred[1], deferred[2])
        return self._summary

    def deferred_columns(self) -> DeferredPoints | None:
        parts = self._parts
        if parts is not None:
            spans: list[tuple[int, int, bool]] = []
            offset = 0
            for part_x, _, part_spans, _ in parts:
                spans.extend(
                    (start + offset, end + offset, flag) for start, end, flag in part_spans
                )
                offset += len(part_x)
            self._deferred = (
                numpy.concatenate([part[0] for part in parts]),
                numpy.concatenate([part[1] for part in parts]),
                spans,
                False,
            )
            self._parts = None
        return self._deferred

    def fill_edge_array(self) -> numpy.ndarray[Any, numpy.dtype[numpy.float64]] | None:
        deferred = self.deferred_columns()
        if deferred is None:
            return None
        column_x, column_y, spans, _ = deferred
        return fill_edge_rows(column_x, column_y, spans)

    def fill_edges(self) -> list[tuple[float, float, float, float]]:
        edges: list[tuple[float, float, float, float]] = []
        for subpath in self.subpaths:
            edges.extend(subpath.edges(close_open=True))
        return edges


DrawingItem = tuple[str, tuple[tuple[float, float], ...]]
EMPTY_DRAWING_ITEMS: tuple[DrawingItem, ...] = ()


PaintedDrawingKind: TypeAlias = Literal[
    "fill",
    "stroke",
    "fillstroke",
    "clip",
    "image",
    "shading",
]
MarkerDrawingKind: TypeAlias = Literal[
    "state-push",
    "state-pop",
    "group-begin",
    "group-end",
    "scope-end",
]
DrawingKind: TypeAlias = PaintedDrawingKind | MarkerDrawingKind | Literal["scope-begin"]


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


class CapturedDrawing(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
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
    kind: DrawingKind = "fill"
    items: tuple[DrawingItem, ...] | list[DrawingItem] = EMPTY_DRAWING_ITEMS
    path: CapturedPath | None = None
    bbox: Rectangle | None = None
    stream_order: int = 0
    xobject_depth: int = 0
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING
    paints: bool = True
    fill_paints: bool = True
    stroke_paints: bool = True
    group_isolated: bool = True
    group_knockout: bool = False
    alpha_is_shape: bool = False
    graphics_soft_mask: CapturedSoftMask | None = None

    def __post_init__(self) -> None:
        if not self.items:
            self.items = EMPTY_DRAWING_ITEMS

    def stroke_style_key(self) -> StrokeStyleKey | None:
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
    def rect(self) -> Rectangle | None:
        bbox = self.bbox
        if bbox is None:
            if self.path is None:
                return None
            bbox = self.path.bbox()
            if bbox is None:
                return None
        x0, y0, x1, y1 = bbox
        if x0 <= x1 and y0 <= y1:
            return bbox
        return normalize_rect(bbox)


def marker_drawing(
    kind: MarkerDrawingKind,
    seqno: int,
    *,
    fill_opacity: float | None = None,
    blend_mode: str | None = None,
    soft_mask_alpha: float | None = None,
    group_isolated: bool = True,
    group_knockout: bool = False,
    alpha_is_shape: bool = False,
    graphics_soft_mask: CapturedSoftMask | None = None,
) -> CapturedDrawing:
    return CapturedDrawing(
        seqno=seqno,
        fill=None,
        fill_opacity=fill_opacity,
        blend_mode=blend_mode,
        soft_mask_alpha=soft_mask_alpha,
        group_isolated=group_isolated,
        group_knockout=group_knockout,
        alpha_is_shape=alpha_is_shape,
        graphics_soft_mask=graphics_soft_mask,
        kind=kind,
    )


class ShadingPattern(GeneratedRecord):
    dictionary: dict[Any, Any]
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING


class TilingPattern(GeneratedRecord):
    bbox: Rectangle
    x_step: float
    y_step: float
    program: CapturedProgram


PatternPaint: TypeAlias = ShadingPattern | TilingPattern


__all__ = (
    "CapturedDrawing",
    "DrawingKind",
    "CapturedInlineImage",
    "CapturedLine",
    "CapturedLines",
    "CapturedPath",
    "CapturedSoftMask",
    "CapturedSubpath",
    "CapturedTextBoundary",
    "PatternPaint",
    "ShadingPattern",
    "TilingPattern",
)


def rect_from_points(subpath_points: list[tuple[float, float]], closed: bool) -> Rectangle | None:
    points = list(subpath_points)
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) != 4:
        return None
    if not closed and subpath_points[0] != subpath_points[-1]:
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
