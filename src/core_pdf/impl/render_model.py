# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from copy import replace
from enum import IntEnum
from operator import attrgetter
from typing import Any, ClassVar, Self, TypeIs, final

import numpy

from core_pdf.impl.array_views import UInt8Array, uint8_image_view
from core_pdf.impl.capture_records import CapturedPath, CapturedSoftMask, PatternPaint
from core_pdf.impl.geometry import rect_tuple
from core_pdf.impl.render_blend import color_rgba, declared_blend, scale_rgba_alpha
from core_pdf.impl.render_paths import translate_rect
from core_pdf.impl.scalars import clamp01
from core_pdf.impl.types import Record, ReplaceFields, ReprFields, frozen_setattr
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import ImageSource


class RenderOptions(ReplaceFields, ReprFields):
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

    def _post_init(self) -> None:
        if self.rotate % 90:
            raise ValueError("render rotation must be a multiple of 90 degrees")
        self.rotate %= 360


class DisplayListItem(ReplaceFields, ReprFields):
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

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.kind == other.kind and self.seqno == other.seqno and self.data == other.data

    __hash__ = None  # type: ignore[assignment]

    def page_box(self, scale: float = 1.0) -> tuple[float, float, float, float] | None:  # noqa: ARG002
        data = self.data
        kind = self.kind
        if kind in {"text", "glyph"}:
            value = data.get("bbox")
        elif kind in {"annotation", "widget"}:
            value = data.get("rect")
        elif kind == "shading":
            value = data.get("bbox") or data.get("rect")
        else:
            return None
        return rect_tuple(value)

    def translated(
        self, tx: float, ty: float, parent_blend_mode: str | None = None
    ) -> DisplayListItem:
        return DisplayListItem(
            self.kind, self.seqno, translated_data(self.kind, self.data, tx, ty, parent_blend_mode)
        )

    def to_data(self) -> dict[str, Any]:
        return dict(self.data)


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


UNIT_MASK_TABLE = numpy.arange(256, dtype=numpy.float32) / 255.0
UNIT_MASK_TABLE.setflags(write=False)


@final
class SoftMaskPlane:
    __slots__ = ("alpha", "table")

    def __init__(
        self, alpha: UInt8Array, table: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
    ) -> None:
        self.alpha = alpha
        self.table = table

    def __getitem__(self, index: Any) -> numpy.ndarray[Any, numpy.dtype[numpy.float32]]:
        window = self.alpha[index]
        if self.table is None:
            return window.astype(numpy.float32) / 255.0
        return self.table[window]

    def values(self) -> numpy.ndarray[Any, numpy.dtype[numpy.float32]]:
        return UNIT_MASK_TABLE if self.table is None else self.table

    @property
    def nbytes(self) -> int:
        return self.alpha.nbytes


PATH_PAINT_FIELDS = (
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
    "alpha_is_shape",
    "graphics_soft_mask",
    "fill_pattern",
    "stroke_pattern",
)
path_paint_values = attrgetter(*PATH_PAINT_FIELDS)


def path_paint_fields(source: object) -> dict[str, Any]:
    return dict(zip(PATH_PAINT_FIELDS, path_paint_values(source), strict=True))


class PaintItemBase:
    __slots__ = (
        "seqno",
        "bbox",
        "fill",
        "fill_opacity",
        "blend_mode",
        "soft_mask_alpha",
        "alpha_is_shape",
        "graphics_soft_mask",
    )

    seqno: int
    bbox: Any
    fill: Any
    fill_opacity: float | None
    blend_mode: str | None
    soft_mask_alpha: float | None
    alpha_is_shape: bool
    graphics_soft_mask: CapturedSoftMask | None

    def fill_rgba(self) -> tuple[int, int, int, int]:
        rgba = color_rgba(self.fill, self.fill_opacity)
        if is_pdf_number(self.soft_mask_alpha):
            rgba = scale_rgba_alpha(rgba, self.soft_mask_alpha)
        return rgba

    def page_box(self, scale: float = 1.0) -> tuple[float, float, float, float] | None:  # noqa: ARG002
        return rect_tuple(self.bbox)


@final
class PathPaintItem(PaintItemBase, ReplaceFields, ReprFields):
    __slots__ = (
        "paint_kind",
        "path",
        "stroke_color",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "coalesced_path",
        "fill_pattern",
        "stroke_pattern",
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

    @property
    def kind(self) -> str:
        return PATH_PAINT_NAMES[int(self.paint_kind)]

    def page_box(self, scale: float = 1.0) -> tuple[float, float, float, float] | None:
        value = self.bbox
        if value is None and type(self.path) is CapturedPath:
            value = self.path.bbox()
        box = rect_tuple(value)
        if box is None:
            return None
        if self.paint_kind in {PathPaintKind.STROKE, PathPaintKind.FILL_STROKE}:
            pad = max(0.5 / scale, self.line_width * 0.5)
            box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        return box

    def translated(
        self, tx: float, ty: float, parent_blend_mode: str | None = None
    ) -> PathPaintItem:
        return replace(
            self,
            bbox=translate_rect(self.bbox, tx, ty),
            path=self.path.translated(tx, ty) if isinstance(self.path, CapturedPath) else self.path,
            edge_array=(
                self.edge_array + numpy.array((tx, ty, tx, ty))
                if self.edge_array is not None
                else None
            ),
            blend_mode=self.blend_mode or parent_blend_mode,
            graphics_soft_mask=translated_soft_mask(self.graphics_soft_mask, tx, ty),
        )

    def to_data(self) -> dict[str, Any]:
        return {"bbox": self.bbox, "path": self.path, **path_paint_fields(self)}


def is_plain_fill(item: object) -> TypeIs[PathPaintItem]:
    return (
        type(item) is PathPaintItem
        and item.paint_kind is PathPaintKind.FILL
        and item.edge_array is not None
        and item.bbox is not None
        and item.fill_pattern is None
        and declared_blend(item.blend_mode) is None
    )


@final
class ImagePaintItem(PaintItemBase, ReplaceFields, ReprFields):
    __slots__ = (
        "paint_kind",
        "source",
        "quad",
        "image_clip",
        "source_metadata",
        "ctm",
        "xobject_depth",
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

    @property
    def kind(self) -> str:
        return self.paint_kind

    def translated(
        self, tx: float, ty: float, parent_blend_mode: str | None = None
    ) -> ImagePaintItem:
        return replace(
            self,
            bbox=translate_rect(self.bbox, tx, ty),
            quad=tuple((x + tx, y + ty) for x, y in self.quad) if self.quad else None,
            image_clip=translate_rect(self.image_clip, tx, ty),
            blend_mode=self.blend_mode or parent_blend_mode,
            graphics_soft_mask=translated_soft_mask(self.graphics_soft_mask, tx, ty),
        )

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


def translated_soft_mask(
    mask: CapturedSoftMask | None, tx: float, ty: float
) -> CapturedSoftMask | None:
    if mask is None or (tx == 0 and ty == 0):
        return mask
    return replace(mask, offset=(mask.offset[0] + tx, mask.offset[1] + ty))


def translated_data(
    kind: str, source: dict[str, Any], tx: float, ty: float, parent_blend_mode: str | None
) -> dict[str, Any]:
    data: dict[str, Any] = dict(source)
    if (mask := data.get("graphics_soft_mask")) is not None:
        data["graphics_soft_mask"] = translated_soft_mask(mask, tx, ty)
    for key in ("bbox", "rect"):
        if key in data:
            data[key] = translate_rect(data[key], tx, ty)
    path = data.get("path")
    if isinstance(path, CapturedPath):
        data["path"] = path.translated(tx, ty)
    if kind == "shading" and isinstance(data.get("dictionary"), dict):
        dictionary = dict(data["dictionary"])
        coords = dictionary.get("Coords")
        if isinstance(coords, (list, tuple)):
            coords = list(coords)
            indexes = (0, 2) if dictionary.get("ShadingType") == 2 else (0, 3)
            for index in indexes:
                if len(coords) > index + 1:
                    coords[index] += tx
                    coords[index + 1] += ty
            dictionary["Coords"] = coords
        if "BBox" in dictionary:
            dictionary["BBox"] = translate_rect(dictionary["BBox"], tx, ty)
        data["dictionary"] = dictionary
    data["blend_mode"] = data.get("blend_mode") or parent_blend_mode
    return data


@final
class GlyphBitmapItem(PaintItemBase, ReplaceFields, ReprFields):
    __slots__ = ("visible", "bitmap", "bitmap_width", "bitmap_height", "payload")

    kind: ClassVar[str] = "glyph"
    visible: object
    bitmap: Any
    bitmap_width: Any
    bitmap_height: Any
    payload: dict[str, Any]

    __fields__: ClassVar[tuple[str, ...]] = (
        "seqno",
        "bbox",
        "fill",
        "fill_opacity",
        "blend_mode",
        "soft_mask_alpha",
        "alpha_is_shape",
        "graphics_soft_mask",
        "visible",
        "bitmap",
        "bitmap_width",
        "bitmap_height",
        "payload",
    )
    __match_args__ = ("seqno", "bbox", "bitmap")

    def __init__(
        self,
        seqno: int,
        bbox: Any,
        fill: Any,
        fill_opacity: float | None,
        blend_mode: str | None,
        soft_mask_alpha: float | None,
        alpha_is_shape: bool,
        graphics_soft_mask: CapturedSoftMask | None,
        visible: object,
        bitmap: Any,
        bitmap_width: Any,
        bitmap_height: Any,
        payload: dict[str, Any],
    ) -> None:
        self.seqno = seqno
        self.bbox = bbox
        self.fill = fill
        self.fill_opacity = fill_opacity
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.alpha_is_shape = alpha_is_shape
        self.graphics_soft_mask = graphics_soft_mask
        self.visible = visible
        self.bitmap = bitmap
        self.bitmap_width = bitmap_width
        self.bitmap_height = bitmap_height
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.seqno == other.seqno and self.payload == other.payload

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_data(cls, seqno: int, data: dict[str, Any]) -> Self:
        return cls(
            seqno,
            data.get("bbox"),
            data.get("fill_color"),
            data.get("fill_opacity"),
            data.get("blend_mode"),
            data.get("soft_mask_alpha"),
            data.get("alpha_is_shape") is True,
            data.get("graphics_soft_mask"),
            data.get("visible"),
            data.get("bitmap"),
            data.get("bitmap_width"),
            data.get("bitmap_height"),
            data,
        )

    def translated(self, tx: float, ty: float, parent_blend_mode: str | None = None) -> Self:
        return self.from_data(
            self.seqno, translated_data(self.kind, self.payload, tx, ty, parent_blend_mode)
        )

    def to_data(self) -> dict[str, Any]:
        return dict(self.payload)


@final
class ShadingItem(PaintItemBase, ReplaceFields, ReprFields):
    __slots__ = ("rect", "dictionary", "color_rendering", "payload")

    kind: ClassVar[str] = "shading"
    rect: Any
    dictionary: Any
    color_rendering: ColorRendering
    payload: dict[str, Any]

    __fields__: ClassVar[tuple[str, ...]] = (
        "seqno",
        "bbox",
        "fill",
        "fill_opacity",
        "blend_mode",
        "soft_mask_alpha",
        "alpha_is_shape",
        "graphics_soft_mask",
        "rect",
        "dictionary",
        "color_rendering",
        "payload",
    )
    __match_args__ = ("seqno", "bbox", "dictionary")

    def __init__(
        self,
        seqno: int,
        bbox: Any,
        fill: Any,
        fill_opacity: float | None,
        blend_mode: str | None,
        soft_mask_alpha: float | None,
        alpha_is_shape: bool,
        graphics_soft_mask: CapturedSoftMask | None,
        rect: Any,
        dictionary: Any,
        color_rendering: ColorRendering,
        payload: dict[str, Any],
    ) -> None:
        self.seqno = seqno
        self.bbox = bbox
        self.fill = fill
        self.fill_opacity = fill_opacity
        self.blend_mode = blend_mode
        self.soft_mask_alpha = soft_mask_alpha
        self.alpha_is_shape = alpha_is_shape
        self.graphics_soft_mask = graphics_soft_mask
        self.rect = rect
        self.dictionary = dictionary
        self.color_rendering = color_rendering
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.seqno == other.seqno and self.payload == other.payload

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_data(cls, seqno: int, data: dict[str, Any]) -> Self:
        return cls(
            seqno,
            data.get("bbox"),
            data.get("fill"),
            data.get("fill_opacity"),
            data.get("blend_mode"),
            data.get("soft_mask_alpha"),
            data.get("alpha_is_shape") is True,
            data.get("graphics_soft_mask"),
            data.get("rect"),
            data.get("dictionary"),
            data.get("color_rendering", DEFAULT_COLOR_RENDERING),
            data,
        )

    def page_box(self, scale: float = 1.0) -> tuple[float, float, float, float] | None:  # noqa: ARG002
        return rect_tuple(self.bbox or self.rect)

    def translated(self, tx: float, ty: float, parent_blend_mode: str | None = None) -> Self:
        return self.from_data(
            self.seqno, translated_data(self.kind, self.payload, tx, ty, parent_blend_mode)
        )

    def to_data(self) -> dict[str, Any]:
        return dict(self.payload)


class ControlItemBase(ReplaceFields, ReprFields):
    __slots__ = ()

    kind: str
    seqno: int
    payload: dict[str, Any]

    def page_box(self, scale: float = 1.0) -> tuple[float, float, float, float] | None:  # noqa: ARG002
        return None

    def translated(self, tx: float, ty: float, parent_blend_mode: str | None = None) -> DisplayItem:
        return display_item(
            self.kind,
            self.seqno,
            translated_data(self.kind, self.payload, tx, ty, parent_blend_mode),
        )

    def to_data(self) -> dict[str, Any]:
        return dict(self.payload)


@final
class ControlItem(ControlItemBase):
    __slots__ = ("kind", "seqno", "payload")

    __fields__: ClassVar[tuple[str, ...]] = ("kind", "seqno", "payload")
    __match_args__ = ("kind", "seqno", "payload")

    def __init__(self, kind: str, seqno: int, payload: dict[str, Any]) -> None:
        self.kind = kind
        self.seqno = seqno
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.kind == other.kind and self.seqno == other.seqno and self.payload == other.payload
        )

    __hash__ = None  # type: ignore[assignment]


@final
class ClipItem(ControlItemBase):
    __slots__ = ("seqno", "path", "fill_rule", "payload")

    kind = "clip"
    path: Any
    fill_rule: Any

    __fields__: ClassVar[tuple[str, ...]] = ("seqno", "path", "fill_rule", "payload")
    __match_args__ = ("seqno", "path", "fill_rule")

    def __init__(self, seqno: int, path: Any, fill_rule: Any, payload: dict[str, Any]) -> None:
        self.seqno = seqno
        self.path = path
        self.fill_rule = fill_rule
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.seqno == other.seqno and self.payload == other.payload

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_data(cls, seqno: int, data: dict[str, Any]) -> Self:
        return cls(seqno, data.get("path"), data.get("fill_rule"), data)


@final
class ScopeBeginItem(ControlItemBase):
    __slots__ = ("seqno", "path", "payload")

    kind = "scope-begin"
    path: Any

    __fields__: ClassVar[tuple[str, ...]] = ("seqno", "path", "payload")
    __match_args__ = ("seqno", "path")

    def __init__(self, seqno: int, path: Any, payload: dict[str, Any]) -> None:
        self.seqno = seqno
        self.path = path
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.seqno == other.seqno and self.payload == other.payload

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_data(cls, seqno: int, data: dict[str, Any]) -> Self:
        return cls(seqno, data.get("path"), data)


@final
class GroupBeginItem(ControlItemBase):
    __slots__ = (
        "seqno",
        "fill_opacity",
        "soft_mask_alpha",
        "blend_mode",
        "isolated",
        "knockout",
        "track_shape",
        "alpha_is_shape",
        "graphics_soft_mask",
        "payload",
    )

    kind = "group-begin"
    fill_opacity: Any
    soft_mask_alpha: Any
    blend_mode: str | None
    isolated: Any
    knockout: Any
    track_shape: Any
    alpha_is_shape: Any
    graphics_soft_mask: CapturedSoftMask | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "seqno",
        "fill_opacity",
        "soft_mask_alpha",
        "blend_mode",
        "isolated",
        "knockout",
        "track_shape",
        "alpha_is_shape",
        "graphics_soft_mask",
        "payload",
    )
    __match_args__ = ("seqno", "isolated", "knockout")

    def __init__(
        self,
        seqno: int,
        fill_opacity: Any,
        soft_mask_alpha: Any,
        blend_mode: str | None,
        isolated: Any,
        knockout: Any,
        track_shape: Any,
        alpha_is_shape: Any,
        graphics_soft_mask: CapturedSoftMask | None,
        payload: dict[str, Any],
    ) -> None:
        self.seqno = seqno
        self.fill_opacity = fill_opacity
        self.soft_mask_alpha = soft_mask_alpha
        self.blend_mode = blend_mode
        self.isolated = isolated
        self.knockout = knockout
        self.track_shape = track_shape
        self.alpha_is_shape = alpha_is_shape
        self.graphics_soft_mask = graphics_soft_mask
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.seqno == other.seqno and self.payload == other.payload

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_data(cls, seqno: int, data: dict[str, Any]) -> Self:
        return cls(
            seqno,
            data.get("fill_opacity"),
            data.get("soft_mask_alpha"),
            data.get("blend_mode"),
            data.get("group_isolated", True),
            data.get("group_knockout", False),
            data.get("group_track_shape", False),
            data.get("alpha_is_shape", False),
            data.get("graphics_soft_mask"),
            data,
        )

    def opacity(self) -> Any:
        opacity = self.fill_opacity
        mask = self.soft_mask_alpha
        if is_pdf_number(mask):
            opacity = (float(opacity) if is_pdf_number(opacity) else 1.0) * float(mask)
        return opacity


DisplayItem = (
    DisplayListItem
    | ImagePaintItem
    | PathPaintItem
    | GlyphBitmapItem
    | ShadingItem
    | ClipItem
    | ScopeBeginItem
    | GroupBeginItem
    | ControlItem
)
CONTROL_KINDS = frozenset({"state-push", "state-pop", "group-end", "scope-end"})
DISPLAY_ITEM_PARSERS: dict[str, Callable[[int, dict[str, Any]], DisplayItem]] = {
    "clip": ClipItem.from_data,
    "scope-begin": ScopeBeginItem.from_data,
    "group-begin": GroupBeginItem.from_data,
    "glyph": GlyphBitmapItem.from_data,
    "shading": ShadingItem.from_data,
}


def display_item(kind: str, seqno: int, data: dict[str, Any] | None = None) -> DisplayItem:
    payload = {} if data is None else data
    parse = DISPLAY_ITEM_PARSERS.get(kind)
    if parse is not None:
        return parse(seqno, payload)
    if kind in CONTROL_KINDS:
        return ControlItem(kind, seqno, payload)
    return DisplayListItem(kind, seqno, payload)


class PixelWindow:
    __slots__ = ("empty", "y0", "y1", "x0", "x1")

    def __init__(self) -> None:
        self.empty = True
        self.y0 = 0
        self.y1 = 0
        self.x0 = 0
        self.x1 = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}{self.bounds()!r}"

    def extend(self, y0: int, y1: int, x0: int, x1: int) -> None:
        if self.empty:
            self.empty = False
            self.y0 = y0
            self.y1 = y1
            self.x0 = x0
            self.x1 = x1
            return
        if y0 < self.y0:
            self.y0 = y0
        if y1 > self.y1:
            self.y1 = y1
        if x0 < self.x0:
            self.x0 = x0
        if x1 > self.x1:
            self.x1 = x1

    def extend_box(self, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        self.extend(y0, y1, x0, x1)

    def bounds(self) -> tuple[int, ...]:
        return () if self.empty else (self.y0, self.y1, self.x0, self.x1)

    def slices(self) -> tuple[slice, slice] | None:
        if self.empty:
            return None
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


class RasterGroup(Record):
    __slots__ = (
        "pixels",
        "view",
        "composite_alpha",
        "blend_mode",
        "backdrop",
        "source_alpha",
        "source_shape",
        "knockout",
        "alpha_is_shape",
        "mask_alpha",
        "paint_window",
        "painted_boxes",
    )

    pixels: bytearray
    view: UInt8Array
    composite_alpha: float | None
    blend_mode: str | None
    backdrop: bytearray | None
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
    source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
    knockout: bool
    alpha_is_shape: bool
    mask_alpha: SoftMaskPlane | None
    paint_window: PixelWindow
    painted_boxes: list[tuple[int, int, int, int]] | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "pixels",
        "view",
        "composite_alpha",
        "blend_mode",
        "backdrop",
        "source_alpha",
        "source_shape",
        "knockout",
        "alpha_is_shape",
        "mask_alpha",
        "paint_window",
        "painted_boxes",
    )
    __match_args__ = ("pixels", "composite_alpha", "blend_mode")

    def __init__(
        self,
        pixels: bytearray,
        composite_alpha: float | None = None,
        blend_mode: str | None = None,
        *,
        view: UInt8Array,
        backdrop: bytearray | None = None,
        source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
        source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
        knockout: bool = False,
        alpha_is_shape: bool = False,
        mask_alpha: SoftMaskPlane | None = None,
        paint_window: PixelWindow | None = None,
        painted_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        frozen_setattr(self, "pixels", pixels)
        frozen_setattr(self, "view", view)
        frozen_setattr(self, "composite_alpha", composite_alpha)
        frozen_setattr(self, "blend_mode", blend_mode)
        frozen_setattr(self, "backdrop", backdrop)
        frozen_setattr(self, "source_alpha", source_alpha)
        frozen_setattr(self, "source_shape", source_shape)
        frozen_setattr(self, "knockout", knockout)
        frozen_setattr(self, "alpha_is_shape", alpha_is_shape)
        frozen_setattr(self, "mask_alpha", mask_alpha)
        frozen_setattr(
            self, "paint_window", PixelWindow() if paint_window is None else paint_window
        )
        frozen_setattr(self, "painted_boxes", painted_boxes)

    def __replace__(self, /, **changes: Any) -> Self:
        pixels = changes.pop("pixels", self.pixels)
        view = changes.pop("view", self.view)
        composite_alpha = changes.pop("composite_alpha", self.composite_alpha)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        backdrop = changes.pop("backdrop", self.backdrop)
        source_alpha = changes.pop("source_alpha", self.source_alpha)
        source_shape = changes.pop("source_shape", self.source_shape)
        knockout = changes.pop("knockout", self.knockout)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        mask_alpha = changes.pop("mask_alpha", self.mask_alpha)
        paint_window = changes.pop("paint_window", self.paint_window)
        painted_boxes = changes.pop("painted_boxes", self.painted_boxes)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            pixels,
            composite_alpha,
            blend_mode,
            view=view,
            backdrop=backdrop,
            source_alpha=source_alpha,
            source_shape=source_shape,
            knockout=knockout,
            alpha_is_shape=alpha_is_shape,
            mask_alpha=mask_alpha,
            paint_window=paint_window,
            painted_boxes=painted_boxes,
        )

    @property
    def source_scale(self) -> float:
        return clamp01(float(self.composite_alpha)) if is_pdf_number(self.composite_alpha) else 1.0


class RasterImage(Record):
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
        frozen_setattr(self, "pixels", pixels)
        frozen_setattr(self, "width", width)
        frozen_setattr(self, "height", height)
        frozen_setattr(self, "channels", channels)
        self._post_init()

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
    "ClipItem",
    "ControlItem",
    "DisplayItem",
    "DisplayListItem",
    "GlyphBitmapItem",
    "GroupBeginItem",
    "ImagePaintItem",
    "PaintItemBase",
    "ScopeBeginItem",
    "ShadingItem",
    "display_item",
    "LineCap",
    "LineJoin",
    "PathPaintItem",
    "PathPaintKind",
    "RasterImage",
    "RenderOptions",
    "is_plain_fill",
)
