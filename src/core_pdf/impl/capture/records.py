# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeAlias

from core_pdf.impl.model.geometry import bbox_union, normalize_rect, points_bbox
from core_pdf.impl.records import internal_Record
from core_pdf.impl.types import Rectangle
from core_pdf_spec.s_07_content.streams import StreamKey
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator

if TYPE_CHECKING:
    from core_pdf.impl.capture.program import CapturedProgram

internal_frozen_setattr = object.__setattr__


LayoutFormId: TypeAlias = tuple[tuple[StreamKey | None, Rectangle | None], ...] | None


class CapturedSoftMask(internal_Record):
    __slots__ = ("program", "transfer", "offset")

    program: CapturedProgram
    transfer: PdfFunctionEvaluator | None
    offset: tuple[float, float]

    __fields__: ClassVar[tuple[str, ...]] = ("program", "transfer", "offset")
    __match_args__ = ("program", "transfer", "offset")

    def __init__(
        self,
        program: CapturedProgram,
        transfer: PdfFunctionEvaluator | None = None,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        internal_frozen_setattr(self, "program", program)
        internal_frozen_setattr(self, "transfer", transfer)
        internal_frozen_setattr(self, "offset", offset)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"program={self.program!r}, "
            f"transfer={self.transfer!r}, "
            f"offset={self.offset!r}"
            ")"
        )

    def __replace__(self, /, **changes: Any) -> Self:
        program = changes.pop("program", self.program)
        transfer = changes.pop("transfer", self.transfer)
        offset = changes.pop("offset", self.offset)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(program, transfer, offset)


class CapturedTextBoundary(internal_Record):
    __slots__ = ("seqno", "kind", "knockout")

    seqno: int
    kind: Literal["begin", "end", "glyph-begin", "glyph-end", "stream-begin", "stream-end"]
    knockout: bool

    __fields__: ClassVar[tuple[str, ...]] = ("seqno", "kind", "knockout")
    __match_args__ = ("seqno", "kind", "knockout")

    def __init__(
        self,
        seqno: int,
        kind: Literal["begin", "end", "glyph-begin", "glyph-end", "stream-begin", "stream-end"],
        knockout: bool = True,
    ) -> None:
        internal_frozen_setattr(self, "seqno", seqno)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "knockout", knockout)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"seqno={self.seqno!r}, "
            f"kind={self.kind!r}, "
            f"knockout={self.knockout!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.seqno == other.seqno
            and self.kind == other.kind
            and self.knockout == other.knockout
        )

    def __hash__(self) -> int:
        return hash((self.seqno, self.kind, self.knockout))

    def __replace__(self, /, **changes: Any) -> Self:
        seqno = changes.pop("seqno", self.seqno)
        kind = changes.pop("kind", self.kind)
        knockout = changes.pop("knockout", self.knockout)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(seqno, kind, knockout)


class CapturedLine:
    __slots__ = ("x0", "y0", "x1", "y1", "line_width")

    x0: float
    y0: float
    x1: float
    y1: float
    line_width: float

    __fields__: ClassVar[tuple[str, ...]] = ("x0", "y0", "x1", "y1", "line_width")
    __match_args__ = ("x0", "y0", "x1", "y1", "line_width")

    def __init__(self, x0: float, y0: float, x1: float, y1: float, line_width: float = 1.0) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.line_width = line_width

    def __replace__(self, /, **changes: Any) -> Self:
        x0 = changes.pop("x0", self.x0)
        y0 = changes.pop("y0", self.y0)
        x1 = changes.pop("x1", self.x1)
        y1 = changes.pop("y1", self.y1)
        line_width = changes.pop("line_width", self.line_width)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(x0, y0, x1, y1, line_width)


class CapturedInlineImage(internal_Record):
    __slots__ = (
        "seqno",
        "dictionary",
        "data",
        "image_source",
        "image_clip",
        "ctm",
        "xobject_depth",
        "blend_mode",
        "soft_mask_alpha",
        "stream_order",
        "fill",
        "fill_opacity",
        "paints",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    seqno: int
    dictionary: dict[Any, Any]
    data: bytes
    image_source: ImageSource
    image_clip: Rectangle | None
    ctm: Matrix
    xobject_depth: int
    blend_mode: str | None
    soft_mask_alpha: float | None
    stream_order: int
    fill: tuple[float, ...] | None
    fill_opacity: float | None
    paints: bool
    alpha_is_shape: bool
    graphics_soft_mask: CapturedSoftMask | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "seqno",
        "dictionary",
        "data",
        "image_source",
        "image_clip",
        "ctm",
        "xobject_depth",
        "blend_mode",
        "soft_mask_alpha",
        "stream_order",
        "fill",
        "fill_opacity",
        "paints",
        "alpha_is_shape",
        "graphics_soft_mask",
    )
    __match_args__ = (
        "seqno",
        "dictionary",
        "data",
        "image_source",
        "image_clip",
        "ctm",
        "xobject_depth",
        "blend_mode",
        "soft_mask_alpha",
        "stream_order",
        "fill",
        "fill_opacity",
        "paints",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    def __init__(
        self,
        seqno: int,
        dictionary: dict[Any, Any],
        data: bytes,
        image_source: ImageSource,
        image_clip: Rectangle | None,
        ctm: Matrix,
        xobject_depth: int,
        blend_mode: str | None = None,
        soft_mask_alpha: float | None = None,
        stream_order: int = 0,
        fill: tuple[float, ...] | None = None,
        fill_opacity: float | None = None,
        paints: bool = True,
        alpha_is_shape: bool = False,
        graphics_soft_mask: CapturedSoftMask | None = None,
    ) -> None:
        internal_frozen_setattr(self, "seqno", seqno)
        internal_frozen_setattr(self, "dictionary", dictionary)
        internal_frozen_setattr(self, "data", data)
        internal_frozen_setattr(self, "image_source", image_source)
        internal_frozen_setattr(self, "image_clip", image_clip)
        internal_frozen_setattr(self, "ctm", ctm)
        internal_frozen_setattr(self, "xobject_depth", xobject_depth)
        internal_frozen_setattr(self, "blend_mode", blend_mode)
        internal_frozen_setattr(self, "soft_mask_alpha", soft_mask_alpha)
        internal_frozen_setattr(self, "stream_order", stream_order)
        internal_frozen_setattr(self, "fill", fill)
        internal_frozen_setattr(self, "fill_opacity", fill_opacity)
        internal_frozen_setattr(self, "paints", paints)
        internal_frozen_setattr(self, "alpha_is_shape", alpha_is_shape)
        internal_frozen_setattr(self, "graphics_soft_mask", graphics_soft_mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"seqno={self.seqno!r}, "
            f"dictionary={self.dictionary!r}, "
            f"data={self.data!r}, "
            f"image_source={self.image_source!r}, "
            f"image_clip={self.image_clip!r}, "
            f"ctm={self.ctm!r}, "
            f"xobject_depth={self.xobject_depth!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"stream_order={self.stream_order!r}, "
            f"fill={self.fill!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"paints={self.paints!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"graphics_soft_mask={self.graphics_soft_mask!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.seqno == other.seqno
            and self.dictionary == other.dictionary
            and self.data == other.data
            and self.image_source == other.image_source
            and self.image_clip == other.image_clip
            and self.ctm == other.ctm
            and self.xobject_depth == other.xobject_depth
            and self.blend_mode == other.blend_mode
            and self.soft_mask_alpha == other.soft_mask_alpha
            and self.stream_order == other.stream_order
            and self.fill == other.fill
            and self.fill_opacity == other.fill_opacity
            and self.paints == other.paints
            and self.alpha_is_shape == other.alpha_is_shape
            and self.graphics_soft_mask == other.graphics_soft_mask
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.seqno,
                self.dictionary,
                self.data,
                self.image_source,
                self.image_clip,
                self.ctm,
                self.xobject_depth,
                self.blend_mode,
                self.soft_mask_alpha,
                self.stream_order,
                self.fill,
                self.fill_opacity,
                self.paints,
                self.alpha_is_shape,
                self.graphics_soft_mask,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        seqno = changes.pop("seqno", self.seqno)
        dictionary = changes.pop("dictionary", self.dictionary)
        data = changes.pop("data", self.data)
        image_source = changes.pop("image_source", self.image_source)
        image_clip = changes.pop("image_clip", self.image_clip)
        ctm = changes.pop("ctm", self.ctm)
        xobject_depth = changes.pop("xobject_depth", self.xobject_depth)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        stream_order = changes.pop("stream_order", self.stream_order)
        fill = changes.pop("fill", self.fill)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        paints = changes.pop("paints", self.paints)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        graphics_soft_mask = changes.pop("graphics_soft_mask", self.graphics_soft_mask)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            seqno,
            dictionary,
            data,
            image_source,
            image_clip,
            ctm,
            xobject_depth,
            blend_mode,
            soft_mask_alpha,
            stream_order,
            fill,
            fill_opacity,
            paints,
            alpha_is_shape,
            graphics_soft_mask,
        )


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


class CapturedDrawing:
    __slots__ = (
        "seqno",
        "fill",
        "fill_opacity",
        "fill_pattern",
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
        "kind",
        "items",
        "path",
        "bbox",
        "stream_order",
        "xobject_depth",
        "color_rendering",
        "paints",
        "fill_paints",
        "stroke_paints",
        "group_isolated",
        "group_knockout",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    seqno: int
    fill: tuple[float, ...] | None
    fill_opacity: float | None
    fill_pattern: PatternPaint | None
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PatternPaint | None
    stroke_opacity: float | None
    line_width: float
    line_cap: int
    line_join: int
    dash_pattern: tuple[list[float], float] | None
    fill_rule: str
    blend_mode: str | None
    soft_mask_alpha: float | None
    raw_data: bytes | memoryview | None
    dictionary: dict[Any, Any] | None
    image_source: ImageSource | None
    image_clip: Rectangle | None
    kind: str
    items: tuple[DrawingItem, ...] | list[DrawingItem]
    path: CapturedPath | None
    bbox: Rectangle | None
    stream_order: int
    xobject_depth: int
    color_rendering: ColorRendering
    paints: bool
    fill_paints: bool
    stroke_paints: bool
    group_isolated: bool
    group_knockout: bool
    alpha_is_shape: bool
    graphics_soft_mask: CapturedSoftMask | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "seqno",
        "fill",
        "fill_opacity",
        "fill_pattern",
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
        "kind",
        "items",
        "path",
        "bbox",
        "stream_order",
        "xobject_depth",
        "color_rendering",
        "paints",
        "fill_paints",
        "stroke_paints",
        "group_isolated",
        "group_knockout",
        "alpha_is_shape",
        "graphics_soft_mask",
    )
    __match_args__ = (
        "seqno",
        "fill",
        "fill_opacity",
        "fill_pattern",
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
        "kind",
        "items",
        "path",
        "bbox",
        "stream_order",
        "xobject_depth",
        "color_rendering",
        "paints",
        "fill_paints",
        "stroke_paints",
        "group_isolated",
        "group_knockout",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    def __init__(
        self,
        seqno: int,
        fill: tuple[float, ...] | None,
        fill_opacity: float | None,
        fill_pattern: PatternPaint | None = None,
        stroke_color: tuple[float, ...] | None = None,
        stroke_pattern: PatternPaint | None = None,
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
        image_clip: Rectangle | None = None,
        kind: str = "fill",
        items: tuple[DrawingItem, ...] | list[DrawingItem] = internal_EMPTY_DRAWING_ITEMS,
        path: CapturedPath | None = None,
        bbox: Rectangle | None = None,
        stream_order: int = 0,
        xobject_depth: int = 0,
        color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
        paints: bool = True,
        fill_paints: bool = True,
        stroke_paints: bool = True,
        group_isolated: bool = True,
        group_knockout: bool = False,
        alpha_is_shape: bool = False,
        graphics_soft_mask: CapturedSoftMask | None = None,
    ) -> None:
        self.seqno = seqno
        self.fill = fill
        self.fill_opacity = fill_opacity
        self.fill_pattern = fill_pattern
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
        self.items = items
        self.path = path
        self.bbox = bbox
        self.stream_order = stream_order
        self.xobject_depth = xobject_depth
        self.color_rendering = color_rendering
        self.paints = paints
        self.fill_paints = fill_paints
        self.stroke_paints = stroke_paints
        self.group_isolated = group_isolated
        self.group_knockout = group_knockout
        self.alpha_is_shape = alpha_is_shape
        self.graphics_soft_mask = graphics_soft_mask
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"seqno={self.seqno!r}, "
            f"fill={self.fill!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"fill_pattern={self.fill_pattern!r}, "
            f"stroke_color={self.stroke_color!r}, "
            f"stroke_pattern={self.stroke_pattern!r}, "
            f"stroke_opacity={self.stroke_opacity!r}, "
            f"line_width={self.line_width!r}, "
            f"line_cap={self.line_cap!r}, "
            f"line_join={self.line_join!r}, "
            f"dash_pattern={self.dash_pattern!r}, "
            f"fill_rule={self.fill_rule!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"raw_data={self.raw_data!r}, "
            f"dictionary={self.dictionary!r}, "
            f"image_source={self.image_source!r}, "
            f"image_clip={self.image_clip!r}, "
            f"kind={self.kind!r}, "
            f"items={self.items!r}, "
            f"path={self.path!r}, "
            f"bbox={self.bbox!r}, "
            f"stream_order={self.stream_order!r}, "
            f"xobject_depth={self.xobject_depth!r}, "
            f"color_rendering={self.color_rendering!r}, "
            f"paints={self.paints!r}, "
            f"fill_paints={self.fill_paints!r}, "
            f"stroke_paints={self.stroke_paints!r}, "
            f"group_isolated={self.group_isolated!r}, "
            f"group_knockout={self.group_knockout!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"graphics_soft_mask={self.graphics_soft_mask!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.seqno == other.seqno
            and self.fill == other.fill
            and self.fill_opacity == other.fill_opacity
            and self.fill_pattern == other.fill_pattern
            and self.stroke_color == other.stroke_color
            and self.stroke_pattern == other.stroke_pattern
            and self.stroke_opacity == other.stroke_opacity
            and self.line_width == other.line_width
            and self.line_cap == other.line_cap
            and self.line_join == other.line_join
            and self.dash_pattern == other.dash_pattern
            and self.fill_rule == other.fill_rule
            and self.blend_mode == other.blend_mode
            and self.soft_mask_alpha == other.soft_mask_alpha
            and self.raw_data == other.raw_data
            and self.dictionary == other.dictionary
            and self.image_source == other.image_source
            and self.image_clip == other.image_clip
            and self.kind == other.kind
            and self.items == other.items
            and self.path == other.path
            and self.bbox == other.bbox
            and self.stream_order == other.stream_order
            and self.xobject_depth == other.xobject_depth
            and self.color_rendering == other.color_rendering
            and self.paints == other.paints
            and self.fill_paints == other.fill_paints
            and self.stroke_paints == other.stroke_paints
            and self.group_isolated == other.group_isolated
            and self.group_knockout == other.group_knockout
            and self.alpha_is_shape == other.alpha_is_shape
            and self.graphics_soft_mask == other.graphics_soft_mask
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        seqno = changes.pop("seqno", self.seqno)
        fill = changes.pop("fill", self.fill)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        fill_pattern = changes.pop("fill_pattern", self.fill_pattern)
        stroke_color = changes.pop("stroke_color", self.stroke_color)
        stroke_pattern = changes.pop("stroke_pattern", self.stroke_pattern)
        stroke_opacity = changes.pop("stroke_opacity", self.stroke_opacity)
        line_width = changes.pop("line_width", self.line_width)
        line_cap = changes.pop("line_cap", self.line_cap)
        line_join = changes.pop("line_join", self.line_join)
        dash_pattern = changes.pop("dash_pattern", self.dash_pattern)
        fill_rule = changes.pop("fill_rule", self.fill_rule)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        raw_data = changes.pop("raw_data", self.raw_data)
        dictionary = changes.pop("dictionary", self.dictionary)
        image_source = changes.pop("image_source", self.image_source)
        image_clip = changes.pop("image_clip", self.image_clip)
        kind = changes.pop("kind", self.kind)
        items = changes.pop("items", self.items)
        path = changes.pop("path", self.path)
        bbox = changes.pop("bbox", self.bbox)
        stream_order = changes.pop("stream_order", self.stream_order)
        xobject_depth = changes.pop("xobject_depth", self.xobject_depth)
        color_rendering = changes.pop("color_rendering", self.color_rendering)
        paints = changes.pop("paints", self.paints)
        fill_paints = changes.pop("fill_paints", self.fill_paints)
        stroke_paints = changes.pop("stroke_paints", self.stroke_paints)
        group_isolated = changes.pop("group_isolated", self.group_isolated)
        group_knockout = changes.pop("group_knockout", self.group_knockout)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        graphics_soft_mask = changes.pop("graphics_soft_mask", self.graphics_soft_mask)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            seqno,
            fill,
            fill_opacity,
            fill_pattern,
            stroke_color,
            stroke_pattern,
            stroke_opacity,
            line_width,
            line_cap,
            line_join,
            dash_pattern,
            fill_rule,
            blend_mode,
            soft_mask_alpha,
            raw_data,
            dictionary,
            image_source,
            image_clip,
            kind,
            items,
            path,
            bbox,
            stream_order,
            xobject_depth,
            color_rendering,
            paints,
            fill_paints,
            stroke_paints,
            group_isolated,
            group_knockout,
            alpha_is_shape,
            graphics_soft_mask,
        )

    def _post_init(self) -> None:
        if not self.items:
            self.items = internal_EMPTY_DRAWING_ITEMS

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
    kind: str,
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


class ShadingPattern(internal_Record):
    __slots__ = ("dictionary", "color_rendering")

    dictionary: dict[Any, Any]
    color_rendering: ColorRendering

    __fields__: ClassVar[tuple[str, ...]] = ("dictionary", "color_rendering")
    __match_args__ = ("dictionary", "color_rendering")

    def __init__(
        self,
        dictionary: dict[Any, Any],
        color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    ) -> None:
        internal_frozen_setattr(self, "dictionary", dictionary)
        internal_frozen_setattr(self, "color_rendering", color_rendering)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"dictionary={self.dictionary!r}, "
            f"color_rendering={self.color_rendering!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.dictionary == other.dictionary and self.color_rendering == other.color_rendering

    def __hash__(self) -> int:
        return hash((self.dictionary, self.color_rendering))

    def __replace__(self, /, **changes: Any) -> Self:
        dictionary = changes.pop("dictionary", self.dictionary)
        color_rendering = changes.pop("color_rendering", self.color_rendering)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(dictionary, color_rendering)


class TilingPattern(internal_Record):
    __slots__ = ("bbox", "x_step", "y_step", "program")

    bbox: Rectangle
    x_step: float
    y_step: float
    program: CapturedProgram

    __fields__: ClassVar[tuple[str, ...]] = ("bbox", "x_step", "y_step", "program")
    __match_args__ = ("bbox", "x_step", "y_step", "program")

    def __init__(
        self,
        bbox: Rectangle,
        x_step: float,
        y_step: float,
        program: CapturedProgram,
    ) -> None:
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "x_step", x_step)
        internal_frozen_setattr(self, "y_step", y_step)
        internal_frozen_setattr(self, "program", program)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"bbox={self.bbox!r}, "
            f"x_step={self.x_step!r}, "
            f"y_step={self.y_step!r}, "
            f"program={self.program!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.bbox == other.bbox
            and self.x_step == other.x_step
            and self.y_step == other.y_step
            and self.program == other.program
        )

    def __hash__(self) -> int:
        return hash((self.bbox, self.x_step, self.y_step, self.program))

    def __replace__(self, /, **changes: Any) -> Self:
        bbox = changes.pop("bbox", self.bbox)
        x_step = changes.pop("x_step", self.x_step)
        y_step = changes.pop("y_step", self.y_step)
        program = changes.pop("program", self.program)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(bbox, x_step, y_step, program)


PatternPaint: TypeAlias = ShadingPattern | TilingPattern


__all__ = (
    "CapturedDrawing",
    "CapturedInlineImage",
    "CapturedLine",
    "CapturedPath",
    "CapturedSoftMask",
    "CapturedSubpath",
    "CapturedTextBoundary",
    "PatternPaint",
    "ShadingPattern",
    "TilingPattern",
)
