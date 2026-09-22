# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias

from core_pdf.impl.model.geometry import bbox_union, normalize_rect, points_bbox
from core_pdf.impl.types import Record, Rectangle, ReplaceFields, ReprFields, frozen_setattr
from core_pdf_spec.s_07_content.streams import StreamKey
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator

if TYPE_CHECKING:
    from core_pdf.impl.capture.program import CapturedProgram

LayoutFormId: TypeAlias = tuple[tuple[StreamKey | None, Rectangle | None], ...] | None


class CapturedSoftMask(Record):
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
        frozen_setattr(self, "program", program)
        frozen_setattr(self, "transfer", transfer)
        frozen_setattr(self, "offset", offset)


class CapturedTextBoundary(Record):
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
        frozen_setattr(self, "seqno", seqno)
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "knockout", knockout)

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


class CapturedLine(ReplaceFields):
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


class CapturedInlineImage(Record):
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
        frozen_setattr(self, "seqno", seqno)
        frozen_setattr(self, "dictionary", dictionary)
        frozen_setattr(self, "data", data)
        frozen_setattr(self, "image_source", image_source)
        frozen_setattr(self, "image_clip", image_clip)
        frozen_setattr(self, "ctm", ctm)
        frozen_setattr(self, "xobject_depth", xobject_depth)
        frozen_setattr(self, "blend_mode", blend_mode)
        frozen_setattr(self, "soft_mask_alpha", soft_mask_alpha)
        frozen_setattr(self, "stream_order", stream_order)
        frozen_setattr(self, "fill", fill)
        frozen_setattr(self, "fill_opacity", fill_opacity)
        frozen_setattr(self, "paints", paints)
        frozen_setattr(self, "alpha_is_shape", alpha_is_shape)
        frozen_setattr(self, "graphics_soft_mask", graphics_soft_mask)

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
EMPTY_DRAWING_ITEMS: tuple[DrawingItem, ...] = ()


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


class CapturedDrawing(ReplaceFields, ReprFields):
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
        items: tuple[DrawingItem, ...] | list[DrawingItem] = EMPTY_DRAWING_ITEMS,
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

    def _post_init(self) -> None:
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


class ShadingPattern(Record):
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
        frozen_setattr(self, "dictionary", dictionary)
        frozen_setattr(self, "color_rendering", color_rendering)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.dictionary == other.dictionary and self.color_rendering == other.color_rendering

    def __hash__(self) -> int:
        return hash((self.dictionary, self.color_rendering))


class TilingPattern(Record):
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
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "x_step", x_step)
        frozen_setattr(self, "y_step", y_step)
        frozen_setattr(self, "program", program)

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
