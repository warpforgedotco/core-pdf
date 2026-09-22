# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from enum import IntEnum
from typing import Any, ClassVar, NoReturn, Self

import numpy

from core_pdf.impl._impl.capture.records import CapturedSoftMask, PatternPaint
from core_pdf.impl._impl.render.blend import internal_clamp01
from core_pdf.impl._impl.runtime.array_views import uint8_image_view
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.image_spec import ImageSource

internal_frozen_setattr = object.__setattr__


class RenderOptions:
    __slots__ = (
        "page_number",
        "rotate",
        "crop",
        "include_annotations",
        "include_layers",
        "include_text",
    )

    page_number: int | None
    rotate: int
    crop: tuple[float, float, float, float] | None
    include_annotations: bool
    include_layers: bool
    include_text: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "page_number",
        "rotate",
        "crop",
        "include_annotations",
        "include_layers",
        "include_text",
    )
    __match_args__ = (
        "page_number",
        "rotate",
        "crop",
        "include_annotations",
        "include_layers",
        "include_text",
    )

    def __init__(
        self,
        page_number: int | None = None,
        rotate: int = 0,
        crop: tuple[float, float, float, float] | None = None,
        include_annotations: bool = True,
        include_layers: bool = True,
        include_text: bool = True,
    ) -> None:
        self.page_number = page_number
        self.rotate = rotate
        self.crop = crop
        self.include_annotations = include_annotations
        self.include_layers = include_layers
        self.include_text = include_text
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_number={self.page_number!r}, "
            f"rotate={self.rotate!r}, "
            f"crop={self.crop!r}, "
            f"include_annotations={self.include_annotations!r}, "
            f"include_layers={self.include_layers!r}, "
            f"include_text={self.include_text!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_number == other.page_number
            and self.rotate == other.rotate
            and self.crop == other.crop
            and self.include_annotations == other.include_annotations
            and self.include_layers == other.include_layers
            and self.include_text == other.include_text
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        page_number = changes.pop("page_number", self.page_number)
        rotate = changes.pop("rotate", self.rotate)
        crop = changes.pop("crop", self.crop)
        include_annotations = changes.pop("include_annotations", self.include_annotations)
        include_layers = changes.pop("include_layers", self.include_layers)
        include_text = changes.pop("include_text", self.include_text)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            page_number,
            rotate,
            crop,
            include_annotations,
            include_layers,
            include_text,
        )

    def _post_init(self) -> None:
        if self.rotate % 90:
            raise ValueError("render rotation must be a multiple of 90 degrees")
        self.rotate %= 360


class DisplayListItem:
    __slots__ = ("kind", "seqno", "data")

    kind: str
    seqno: int
    data: dict[str, Any]

    __fields__: ClassVar[tuple[str, ...]] = ("kind", "seqno", "data")
    __match_args__ = ("kind", "seqno", "data")

    def __init__(self, kind: str, seqno: int, data: dict[str, Any] | None = None) -> None:
        self.kind = kind
        self.seqno = seqno
        self.data = {} if data is None else data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"kind={self.kind!r}, "
            f"seqno={self.seqno!r}, "
            f"data={self.data!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.kind == other.kind and self.seqno == other.seqno and self.data == other.data

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        kind = changes.pop("kind", self.kind)
        seqno = changes.pop("seqno", self.seqno)
        data = changes.pop("data", self.data)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(kind, seqno, data)


class PathPaintKind(IntEnum):
    FILL = 0
    STROKE = 1
    FILL_STROKE = 2


class LineCap(IntEnum):
    BUTT = 0
    ROUND = 1
    PROJECTING_SQUARE = 2


class LineJoin(IntEnum):
    MITER = 0
    ROUND = 1
    BEVEL = 2


PATH_PAINT_NAMES = ("fill", "stroke", "fillstroke")


class PathPaintItem:
    __slots__ = (
        "paint_kind",
        "seqno",
        "bbox",
        "path",
        "fill",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "coalesced_path",
        "fill_pattern",
        "stroke_pattern",
        "alpha_is_shape",
        "graphics_soft_mask",
        "edge_array",
    )

    paint_kind: PathPaintKind
    seqno: int
    bbox: Any
    path: Any
    fill: Any
    fill_opacity: float | None
    stroke_color: Any
    stroke_opacity: float | None
    line_width: float
    line_cap: int
    line_join: int
    dash_pattern: tuple[list[float], float] | None
    fill_rule: str
    blend_mode: str | None
    soft_mask_alpha: float | None
    coalesced_path: bool
    fill_pattern: PatternPaint | None
    stroke_pattern: PatternPaint | None
    alpha_is_shape: bool
    graphics_soft_mask: CapturedSoftMask | None
    edge_array: Any

    __fields__: ClassVar[tuple[str, ...]] = (
        "paint_kind",
        "seqno",
        "bbox",
        "path",
        "fill",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "coalesced_path",
        "fill_pattern",
        "stroke_pattern",
        "alpha_is_shape",
        "graphics_soft_mask",
        "edge_array",
    )
    __match_args__ = (
        "paint_kind",
        "seqno",
        "bbox",
        "path",
        "fill",
        "fill_opacity",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "coalesced_path",
        "fill_pattern",
        "stroke_pattern",
        "alpha_is_shape",
        "graphics_soft_mask",
        "edge_array",
    )

    def __init__(
        self,
        paint_kind: PathPaintKind,
        seqno: int,
        bbox: Any,
        path: Any,
        fill: Any,
        fill_opacity: float | None,
        stroke_color: Any,
        stroke_opacity: float | None,
        line_width: float,
        line_cap: int,
        line_join: int,
        dash_pattern: tuple[list[float], float] | None,
        fill_rule: str,
        blend_mode: str | None,
        soft_mask_alpha: float | None,
        coalesced_path: bool = False,
        fill_pattern: PatternPaint | None = None,
        stroke_pattern: PatternPaint | None = None,
        alpha_is_shape: bool = False,
        graphics_soft_mask: CapturedSoftMask | None = None,
        edge_array: Any = None,
    ) -> None:
        self.paint_kind = paint_kind
        self.seqno = seqno
        self.bbox = bbox
        self.path = path
        self.fill = fill
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_opacity = stroke_opacity
        self.line_width = line_width
        self.line_cap = line_cap
        self.line_join = line_join
        self.dash_pattern = dash_pattern
        self.fill_rule = fill_rule
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.coalesced_path = coalesced_path
        self.fill_pattern = fill_pattern
        self.stroke_pattern = stroke_pattern
        self.alpha_is_shape = alpha_is_shape
        self.graphics_soft_mask = graphics_soft_mask
        self.edge_array = edge_array

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"paint_kind={self.paint_kind!r}, "
            f"seqno={self.seqno!r}, "
            f"bbox={self.bbox!r}, "
            f"path={self.path!r}, "
            f"fill={self.fill!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"stroke_color={self.stroke_color!r}, "
            f"stroke_opacity={self.stroke_opacity!r}, "
            f"line_width={self.line_width!r}, "
            f"line_cap={self.line_cap!r}, "
            f"line_join={self.line_join!r}, "
            f"dash_pattern={self.dash_pattern!r}, "
            f"fill_rule={self.fill_rule!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"coalesced_path={self.coalesced_path!r}, "
            f"fill_pattern={self.fill_pattern!r}, "
            f"stroke_pattern={self.stroke_pattern!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"graphics_soft_mask={self.graphics_soft_mask!r}, "
            f"edge_array={self.edge_array!r}"
            ")"
        )

    def __replace__(self, /, **changes: Any) -> Self:
        paint_kind = changes.pop("paint_kind", self.paint_kind)
        seqno = changes.pop("seqno", self.seqno)
        bbox = changes.pop("bbox", self.bbox)
        path = changes.pop("path", self.path)
        fill = changes.pop("fill", self.fill)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        stroke_color = changes.pop("stroke_color", self.stroke_color)
        stroke_opacity = changes.pop("stroke_opacity", self.stroke_opacity)
        line_width = changes.pop("line_width", self.line_width)
        line_cap = changes.pop("line_cap", self.line_cap)
        line_join = changes.pop("line_join", self.line_join)
        dash_pattern = changes.pop("dash_pattern", self.dash_pattern)
        fill_rule = changes.pop("fill_rule", self.fill_rule)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        coalesced_path = changes.pop("coalesced_path", self.coalesced_path)
        fill_pattern = changes.pop("fill_pattern", self.fill_pattern)
        stroke_pattern = changes.pop("stroke_pattern", self.stroke_pattern)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        graphics_soft_mask = changes.pop("graphics_soft_mask", self.graphics_soft_mask)
        edge_array = changes.pop("edge_array", self.edge_array)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            paint_kind,
            seqno,
            bbox,
            path,
            fill,
            fill_opacity,
            stroke_color,
            stroke_opacity,
            line_width,
            line_cap,
            line_join,
            dash_pattern,
            fill_rule,
            blend_mode,
            soft_mask_alpha,
            coalesced_path,
            fill_pattern,
            stroke_pattern,
            alpha_is_shape,
            graphics_soft_mask,
            edge_array,
        )

    @property
    def kind(self) -> str:
        return PATH_PAINT_NAMES[int(self.paint_kind)]

    def to_data(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "path": self.path,
            "fill": self.fill,
            "fill_opacity": self.fill_opacity,
            "stroke_color": self.stroke_color,
            "stroke_opacity": self.stroke_opacity,
            "line_width": self.line_width,
            "line_cap": self.line_cap,
            "line_join": self.line_join,
            "dash_pattern": self.dash_pattern,
            "fill_rule": self.fill_rule,
            "blend_mode": self.blend_mode,
            "soft_mask_alpha": self.soft_mask_alpha,
            "alpha_is_shape": self.alpha_is_shape,
            "graphics_soft_mask": self.graphics_soft_mask,
            "fill_pattern": self.fill_pattern,
            "stroke_pattern": self.stroke_pattern,
        }


class ImagePaintItem:
    __slots__ = (
        "paint_kind",
        "seqno",
        "bbox",
        "source",
        "quad",
        "fill",
        "fill_opacity",
        "blend_mode",
        "soft_mask_alpha",
        "image_clip",
        "source_metadata",
        "ctm",
        "xobject_depth",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    paint_kind: str
    seqno: int
    bbox: Any
    source: ImageSource | None
    quad: tuple[tuple[float, float], ...] | None
    fill: Any
    fill_opacity: float | None
    blend_mode: str | None
    soft_mask_alpha: float | None
    image_clip: Any
    source_metadata: dict[str, Any]
    ctm: Any
    xobject_depth: Any
    alpha_is_shape: bool
    graphics_soft_mask: CapturedSoftMask | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "paint_kind",
        "seqno",
        "bbox",
        "source",
        "quad",
        "fill",
        "fill_opacity",
        "blend_mode",
        "soft_mask_alpha",
        "image_clip",
        "source_metadata",
        "ctm",
        "xobject_depth",
        "alpha_is_shape",
        "graphics_soft_mask",
    )
    __match_args__ = (
        "paint_kind",
        "seqno",
        "bbox",
        "source",
        "quad",
        "fill",
        "fill_opacity",
        "blend_mode",
        "soft_mask_alpha",
        "image_clip",
        "source_metadata",
        "ctm",
        "xobject_depth",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    def __init__(
        self,
        paint_kind: str,
        seqno: int,
        bbox: Any,
        source: ImageSource | None,
        quad: tuple[tuple[float, float], ...] | None,
        fill: Any,
        fill_opacity: float | None,
        blend_mode: str | None,
        soft_mask_alpha: float | None,
        image_clip: Any,
        source_metadata: dict[str, Any],
        ctm: Any = None,
        xobject_depth: Any = None,
        alpha_is_shape: bool = False,
        graphics_soft_mask: CapturedSoftMask | None = None,
    ) -> None:
        self.paint_kind = paint_kind
        self.seqno = seqno
        self.bbox = bbox
        self.source = source
        self.quad = quad
        self.fill = fill
        self.fill_opacity = fill_opacity
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.image_clip = image_clip
        self.source_metadata = source_metadata
        self.ctm = ctm
        self.xobject_depth = xobject_depth
        self.alpha_is_shape = alpha_is_shape
        self.graphics_soft_mask = graphics_soft_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"paint_kind={self.paint_kind!r}, "
            f"seqno={self.seqno!r}, "
            f"bbox={self.bbox!r}, "
            f"source={self.source!r}, "
            f"quad={self.quad!r}, "
            f"fill={self.fill!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"image_clip={self.image_clip!r}, "
            f"source_metadata={self.source_metadata!r}, "
            f"ctm={self.ctm!r}, "
            f"xobject_depth={self.xobject_depth!r}, "
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
            self.paint_kind == other.paint_kind
            and self.seqno == other.seqno
            and self.bbox == other.bbox
            and self.source == other.source
            and self.quad == other.quad
            and self.fill == other.fill
            and self.fill_opacity == other.fill_opacity
            and self.blend_mode == other.blend_mode
            and self.soft_mask_alpha == other.soft_mask_alpha
            and self.image_clip == other.image_clip
            and self.source_metadata == other.source_metadata
            and self.ctm == other.ctm
            and self.xobject_depth == other.xobject_depth
            and self.alpha_is_shape == other.alpha_is_shape
            and self.graphics_soft_mask == other.graphics_soft_mask
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        paint_kind = changes.pop("paint_kind", self.paint_kind)
        seqno = changes.pop("seqno", self.seqno)
        bbox = changes.pop("bbox", self.bbox)
        source = changes.pop("source", self.source)
        quad = changes.pop("quad", self.quad)
        fill = changes.pop("fill", self.fill)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        image_clip = changes.pop("image_clip", self.image_clip)
        source_metadata = changes.pop("source_metadata", self.source_metadata)
        ctm = changes.pop("ctm", self.ctm)
        xobject_depth = changes.pop("xobject_depth", self.xobject_depth)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        graphics_soft_mask = changes.pop("graphics_soft_mask", self.graphics_soft_mask)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            paint_kind,
            seqno,
            bbox,
            source,
            quad,
            fill,
            fill_opacity,
            blend_mode,
            soft_mask_alpha,
            image_clip,
            source_metadata,
            ctm,
            xobject_depth,
            alpha_is_shape,
            graphics_soft_mask,
        )

    @property
    def kind(self) -> str:
        return self.paint_kind

    def to_data(self) -> dict[str, Any]:
        source = self.source
        return {
            "bbox": self.bbox,
            "raw_data": source.raw if source is not None else None,
            "dictionary": source.dictionary if source is not None else None,
            "soft_mask": source.soft_mask if source is not None else None,
            "image_source": source,
            "items": [("quad", self.quad)] if self.quad is not None else [],
            "fill": self.fill,
            "fill_opacity": self.fill_opacity,
            "blend_mode": self.blend_mode,
            "soft_mask_alpha": self.soft_mask_alpha,
            "alpha_is_shape": self.alpha_is_shape,
            "graphics_soft_mask": self.graphics_soft_mask,
            "image_clip": self.image_clip,
            "source_metadata": self.source_metadata,
            "ctm": self.ctm,
            "xobject_depth": self.xobject_depth,
        }


DisplayItem = DisplayListItem | ImagePaintItem | PathPaintItem


class internal_RasterGroup:
    __slots__ = (
        "pixels",
        "composite_alpha",
        "blend_mode",
        "backdrop",
        "source_alpha",
        "source_shape",
        "knockout",
        "alpha_is_shape",
        "mask_alpha",
        "paint_window",
    )

    pixels: bytearray
    composite_alpha: float | None
    blend_mode: str | None
    backdrop: bytearray | None
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
    source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
    knockout: bool
    alpha_is_shape: bool
    mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
    paint_window: list[int]

    __fields__: ClassVar[tuple[str, ...]] = (
        "pixels",
        "composite_alpha",
        "blend_mode",
        "backdrop",
        "source_alpha",
        "source_shape",
        "knockout",
        "alpha_is_shape",
        "mask_alpha",
        "paint_window",
    )
    __match_args__ = ("pixels", "composite_alpha", "blend_mode")

    def __init__(
        self,
        pixels: bytearray,
        composite_alpha: float | None = None,
        blend_mode: str | None = None,
        *,
        backdrop: bytearray | None = None,
        source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
        source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
        knockout: bool = False,
        alpha_is_shape: bool = False,
        mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
        paint_window: list[int] | None = None,
    ) -> None:
        internal_frozen_setattr(self, "pixels", pixels)
        internal_frozen_setattr(self, "composite_alpha", composite_alpha)
        internal_frozen_setattr(self, "blend_mode", blend_mode)
        internal_frozen_setattr(self, "backdrop", backdrop)
        internal_frozen_setattr(self, "source_alpha", source_alpha)
        internal_frozen_setattr(self, "source_shape", source_shape)
        internal_frozen_setattr(self, "knockout", knockout)
        internal_frozen_setattr(self, "alpha_is_shape", alpha_is_shape)
        internal_frozen_setattr(self, "mask_alpha", mask_alpha)
        internal_frozen_setattr(self, "paint_window", [] if paint_window is None else paint_window)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"pixels={self.pixels!r}, "
            f"composite_alpha={self.composite_alpha!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"backdrop={self.backdrop!r}, "
            f"source_alpha={self.source_alpha!r}, "
            f"source_shape={self.source_shape!r}, "
            f"knockout={self.knockout!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"mask_alpha={self.mask_alpha!r}, "
            f"paint_window={self.paint_window!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.pixels == other.pixels
            and self.composite_alpha == other.composite_alpha
            and self.blend_mode == other.blend_mode
            and self.backdrop == other.backdrop
            and self.source_alpha == other.source_alpha
            and self.source_shape == other.source_shape
            and self.knockout == other.knockout
            and self.alpha_is_shape == other.alpha_is_shape
            and self.mask_alpha == other.mask_alpha
            and self.paint_window == other.paint_window
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.pixels,
                self.composite_alpha,
                self.blend_mode,
                self.backdrop,
                self.source_alpha,
                self.source_shape,
                self.knockout,
                self.alpha_is_shape,
                self.mask_alpha,
                self.paint_window,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        pixels = changes.pop("pixels", self.pixels)
        composite_alpha = changes.pop("composite_alpha", self.composite_alpha)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        backdrop = changes.pop("backdrop", self.backdrop)
        source_alpha = changes.pop("source_alpha", self.source_alpha)
        source_shape = changes.pop("source_shape", self.source_shape)
        knockout = changes.pop("knockout", self.knockout)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        mask_alpha = changes.pop("mask_alpha", self.mask_alpha)
        paint_window = changes.pop("paint_window", self.paint_window)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            pixels,
            composite_alpha,
            blend_mode,
            backdrop=backdrop,
            source_alpha=source_alpha,
            source_shape=source_shape,
            knockout=knockout,
            alpha_is_shape=alpha_is_shape,
            mask_alpha=mask_alpha,
            paint_window=paint_window,
        )

    @property
    def source_scale(self) -> float:
        return (
            internal_clamp01(float(self.composite_alpha))
            if is_pdf_number(self.composite_alpha)
            else 1.0
        )

    def extend_paint_window(self, y0: int, y1: int, x0: int, x1: int) -> None:
        window = self.paint_window
        if window:
            window[:] = (
                min(window[0], y0),
                max(window[1], y1),
                min(window[2], x0),
                max(window[3], x1),
            )
        else:
            window[:] = y0, y1, x0, x1


class RasterImage:
    __slots__ = ("pixels", "width", "height", "channels")

    pixels: bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
    width: int
    height: int
    channels: int

    __fields__: ClassVar[tuple[str, ...]] = ("pixels", "width", "height", "channels")
    __match_args__ = ("pixels", "width", "height", "channels")

    def __init__(
        self,
        pixels: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
        width: int,
        height: int,
        channels: int,
    ) -> None:
        internal_frozen_setattr(self, "pixels", pixels)
        internal_frozen_setattr(self, "width", width)
        internal_frozen_setattr(self, "height", height)
        internal_frozen_setattr(self, "channels", channels)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"pixels={self.pixels!r}, "
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"channels={self.channels!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.pixels == other.pixels
            and self.width == other.width
            and self.height == other.height
            and self.channels == other.channels
        )

    def __hash__(self) -> int:
        return hash((self.pixels, self.width, self.height, self.channels))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        pixels = changes.pop("pixels", self.pixels)
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        channels = changes.pop("channels", self.channels)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(pixels, width, height, channels)

    def _post_init(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.channels <= 0:
            raise ValueError("raster dimensions and channel count must be positive")
        try:
            pixels = memoryview(self.pixels)
            if not pixels.c_contiguous or pixels.itemsize != 1:
                raise ValueError
            pixels = pixels.cast("B").toreadonly()
        except (TypeError, ValueError) as exc:
            raise ValueError("raster pixels must be a contiguous byte buffer") from exc
        if pixels.nbytes != self.width * self.height * self.channels:
            raise ValueError("raster byte length does not match its dimensions")
        object.__setattr__(self, "pixels", pixels)

    @property
    def stride(self) -> int:
        return self.width * self.channels

    @property
    def nbytes(self) -> int:
        return memoryview(self.pixels).nbytes

    def array(self) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
        return uint8_image_view(
            self.pixels,
            (self.height, self.width, self.channels),
        )


__all__ = (
    "DisplayItem",
    "DisplayListItem",
    "ImagePaintItem",
    "LineCap",
    "LineJoin",
    "PathPaintItem",
    "PathPaintKind",
    "RasterImage",
    "RenderOptions",
)
