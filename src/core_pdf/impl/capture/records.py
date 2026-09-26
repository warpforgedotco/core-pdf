# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload

import numpy

from core_pdf.impl.geometry import bbox_union, normalize_rect, points_bbox
from core_pdf.impl.types import Rectangle
from core_pdf_spec.s_07_content.streams import StreamKey
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator

if TYPE_CHECKING:
    from core_pdf.impl.capture.program import CapturedProgram

LayoutFormId: TypeAlias = tuple[tuple[StreamKey | None, Rectangle | None], ...] | None


# Compared by identity before this was a dataclass: it defined no __eq__ and
# no Record base supplies one. A generated __eq__ would deep-walk the whole
# CapturedProgram whenever a drawing or inline image compares its soft mask.
@dataclass(frozen=True, slots=True, eq=False)
class CapturedSoftMask:
    program: CapturedProgram
    transfer: PdfFunctionEvaluator | None = None
    offset: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CapturedTextBoundary:
    seqno: int
    kind: Literal["begin", "end", "glyph-begin", "glyph-end", "stream-begin", "stream-end"]
    knockout: bool = True


@dataclass(slots=True, eq=False)
class CapturedLine:
    x0: float
    y0: float
    x1: float
    y1: float
    line_width: float = 1.0


class CapturedLines(Sequence[CapturedLine]):
    """A page's stroke lines as one array: a row of x0, y0, x1, y1, line_width each.

    A path's stroke lines used to be a CapturedLine object per segment -- 733,122
    of them for one corpus page -- built by a Python walk of the flattened path
    and walked again by table detection, which turned them straight back into
    arrays. Capture now writes rows and table detection reads columns.

    It is still a sequence of CapturedLine, built on access, for code that wants
    records rather than columns. Those are new objects on every access, so a line
    has no identity to keep; nothing compares lines by identity.
    """

    __slots__ = ("array",)

    array: numpy.ndarray[Any, numpy.dtype[numpy.float64]]

    def __init__(self, lines: Iterable[CapturedLine] = ()) -> None:
        rows = [(line.x0, line.y0, line.x1, line.y1, line.line_width) for line in lines]
        self.array = read_only(numpy.array(rows, dtype=numpy.float64).reshape(-1, 5))

    @classmethod
    def from_array(cls, array: numpy.ndarray[Any, Any]) -> CapturedLines:
        """Lines over an (n, 5) float64 array, which must not change afterwards."""
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
        """The x0, y0, x1 and y1 columns."""
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


@dataclass(frozen=True, slots=True)
class CapturedInlineImage:
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


# What a deferred path's point lists are rebuilt from: the point columns, a
# (start, end, flag) span per subpath, and whether it is a glyph outline. For an
# outline the flag is the outline kernel's and every subpath is closed; for a
# flattened path the flag is the subpath's own closed state.
DeferredPoints: TypeAlias = tuple[Any, Any, list[tuple[int, int, bool]], bool]


class CapturedPath:
    __slots__ = ("subpaths", "_deferred", "_summary")

    def __init__(self, subpaths: list[CapturedSubpath] | None = None) -> None:
        self.subpaths = subpaths if subpaths is not None else []
        self._deferred: DeferredPoints | None = None
        # bbox() and has_segments() of a deferred flattened path, known without
        # its points. Read only while _deferred is set: once the subpaths exist
        # they can change, and are asked instead.
        self._summary: tuple[Rectangle | None, bool] | None = None

    @classmethod
    def deferred_outline(
        cls,
        column_x: Any,
        column_y: Any,
        spans: list[tuple[int, int, bool]],
    ) -> CapturedPath:
        """A path whose point lists are built only if something asks for them.

        The kernel that builds a glyph's edges already returns everything a
        fill needs -- the edge array and the bounding box -- so the
        CapturedSubpath objects and their point tuples were built for one
        caller, fill_path asking axis_aligned_rect whether the path is a
        rectangle, and then dropped. That is a hundred-odd tuples per glyph,
        several thousand times a page, and it measured at about a fifth of the
        render.

        The subpaths slot is left unset rather than filled, so the first
        attribute access falls through to __getattr__ and builds them there.
        Paths that are not deferred keep a plain slot read, and the type is
        unchanged, which matters because the renderer tests for it by identity
        rather than with isinstance.

        Reading subpaths is the only supported way to fill a deferred path.
        Assigning the slot directly would leave the deferred spans in place and
        axis_aligned_rect would keep answering from them; enforcing that would
        mean a __setattr__ or a property, and both put a Python call on the
        read path this exists to keep free. Nothing outside this class assigns
        subpaths.
        """
        path = cls.__new__(cls)
        path._deferred = (column_x, column_y, spans, True)
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
        """A flattened path, from flatten_path_commands, whose point lists wait.

        Extraction asks a painted path for its bounding box and whether it has a
        segment, both of which the kernel already computed, and for nothing
        else. The renderer and OCR read the subpaths, and build them then, as a
        deferred outline does.
        """
        path = cls.__new__(cls)
        path._deferred = (column_x, column_y, spans, False)
        path._summary = (bbox, has_segments)
        return path

    def __getattr__(self, name: str) -> Any:
        # Only ever reached for an unset slot, which means a deferred outline
        # whose points nobody had needed until now.
        if name == "subpaths":
            deferred = self._deferred
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
        deferred = self._deferred
        if deferred is not None:
            # Settled from the spans and the last one's points, without
            # building a subpath: flattened paths reach here several thousand
            # times a page, and building them all to answer no cost 15us each.
            column_x, column_y, spans, outline = deferred
            if not spans:
                return None
            start, end, flag = spans[-1]
            if outline and (len(spans) != 1 or end - start != 4):
                # A rectangle is one subpath of exactly four points, and the
                # outline kernel has already dropped a duplicated closing one.
                return None
            # The path's only subpath with segments must be its last.
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
        if self._deferred is not None and self._summary is not None:
            return self._summary[1]
        return any(subpath.has_segments() for subpath in self.subpaths)

    def bbox(self) -> Rectangle | None:
        if self._deferred is not None and self._summary is not None:
            return self._summary[0]
        return bbox_union(box for subpath in self.subpaths if (box := subpath.bbox()))

    def fill_edge_array(self) -> numpy.ndarray[Any, numpy.dtype[numpy.float64]] | None:
        """fill_edges as an (n, 4) array, for a deferred path; None for any other.

        The edges come straight from the point columns, in fill_edges' order:
        each subpath's consecutive pairs, then its closing edge when its last
        point differs from its first. No subpath is built.
        """
        deferred = self._deferred
        if deferred is None:
            return None
        column_x, column_y, spans, _ = deferred
        starts: list[int] = []
        ends: list[int] = []
        closing: list[bool] = []
        for start, end, _flag in spans:
            if end - start < 2:
                continue
            starts.extend(range(start, end - 1))
            ends.extend(range(start + 1, end))
            closing.extend([False] * (end - 1 - start))
            starts.append(end - 1)
            ends.append(start)
            closing.append(True)
        if not starts:
            return numpy.empty((0, 4), dtype=numpy.float64)
        first = numpy.asarray(starts, dtype=numpy.intp)
        second = numpy.asarray(ends, dtype=numpy.intp)
        x0 = column_x[first]
        y0 = column_y[first]
        x1 = column_x[second]
        y1 = column_y[second]
        # A closing edge is kept where the points differ, as tuples compare.
        keep = ~numpy.asarray(closing, dtype=numpy.bool_) | (x0 != x1) | (y0 != y1)
        return numpy.column_stack((x0, y0, x1, y1))[keep]

    def fill_edges(self) -> list[tuple[float, float, float, float]]:
        edges: list[tuple[float, float, float, float]] = []
        for subpath in self.subpaths:
            edges.extend(subpath.edges(close_open=True))
        return edges


DrawingItem = tuple[str, tuple[tuple[float, float], ...]]
EMPTY_DRAWING_ITEMS: tuple[DrawingItem, ...] = ()


# CapturedDrawing is one record for twelve things, told apart by `kind`. Six
# paint something; the other six are control-flow signals the renderer replays
# to rebuild the graphics stack, and they leave most of the record unset.
PaintedDrawingKind: TypeAlias = Literal[
    "fill",
    "stroke",
    "fillstroke",
    "clip",
    "image",
    "shading",
]
# What marker_drawing emits. "scope-begin" is not here: it carries the form's
# clip path, so it is built as a full record.
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


@dataclass(frozen=True, slots=True)
class ShadingPattern:
    dictionary: dict[Any, Any]
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING


@dataclass(frozen=True, slots=True)
class TilingPattern:
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
    """The rectangle one subpath's points trace, axis-aligned, or None."""
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
