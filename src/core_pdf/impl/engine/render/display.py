# SPDX-License-Identifier: AGPL-3.0-only
"""Display-list records and crop-aware render planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast

from core_pdf.impl.capture_model.geometry import rect_tuple
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.spec.s_08_graphics.image_metadata import (
    image_display_metadata,
    pdf_number,
)
from core_pdf.impl.spec.s_08_graphics.shading import prepare_shading


@dataclass(slots=True)
class RenderOptions:
    page_number: int | None = None
    rotate: int = 0
    crop: tuple[float, float, float, float] | None = None
    include_annotations: bool = True
    include_layers: bool = True
    include_text: bool = True

    def __post_init__(self) -> None:
        if self.rotate % 90:
            raise ValueError("render rotation must be a multiple of 90 degrees")
        self.rotate %= 360


@dataclass(slots=True)
class DisplayListItem:
    kind: str
    seqno: int
    data: dict[str, Any] = field(default_factory=dict)


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
PATH_PAINT_KINDS = {name: PathPaintKind(index) for index, name in enumerate(PATH_PAINT_NAMES)}
MAX_COALESCED_STROKE_SUBPATHS = 256


@dataclass(slots=True)
class PathPaintItem:
    """Typed, allocation-light record for the common unpatterned path hot path."""

    paint_kind: PathPaintKind
    seqno: int
    bbox: Any
    path: Any
    fill: Any
    fill_opacity: Any
    stroke_color: Any
    stroke_opacity: Any
    line_width: Any
    line_cap: Any
    line_join: Any
    dash_pattern: Any
    fill_rule: Any
    blend_mode: Any
    soft_mask_alpha: Any
    coalesced_path: bool = False

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
        }


@dataclass(slots=True)
class ImagePaintItem:
    """Typed image paint command whose source owns all PDF image preparation."""

    paint_kind: str
    seqno: int
    bbox: Any
    source: ImageSource | None
    quad: tuple[tuple[float, float], ...] | None
    fill: Any
    fill_opacity: Any
    blend_mode: Any
    soft_mask_alpha: Any
    image_clip: Any
    image_metadata: dict[str, Any]
    ctm: Any = None
    xobject_depth: Any = None

    @property
    def kind(self) -> str:
        return self.paint_kind

    def to_data(self) -> dict[str, Any]:
        """Return the legacy diagnostic mapping without duplicating paint ownership."""
        source = self.source
        return {
            "bbox": self.bbox,
            "raw_data": source.raw if source is not None else None,
            "dictionary": source.dictionary if source is not None else None,
            "image_source": source,
            "items": [("quad", self.quad)] if self.quad is not None else [],
            "fill": self.fill,
            "fill_opacity": self.fill_opacity,
            "blend_mode": self.blend_mode,
            "soft_mask_alpha": self.soft_mask_alpha,
            "image_clip": self.image_clip,
            "image_metadata": self.image_metadata,
            "ctm": self.ctm,
            "xobject_depth": self.xobject_depth,
        }


DisplayItem = DisplayListItem | ImagePaintItem | PathPaintItem


def internal_image_quad(data: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    """Normalize an image quad once at display-list construction."""
    quad = data.get("quad")
    if isinstance(quad, (list, tuple)) and len(quad) >= 3:
        try:
            return tuple((float(point[0]), float(point[1])) for point in quad)
        except (TypeError, ValueError, IndexError):
            return None
    items = data.get("items")
    if not isinstance(items, (list, tuple)):
        return None
    for kind, value in items:
        if kind != "quad":
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            return tuple((float(point[0]), float(point[1])) for point in value)
        except (TypeError, ValueError, IndexError):
            return None
    return None


@dataclass(slots=True)
class DisplayList:
    width: float
    height: float
    items: list[DisplayItem] = field(default_factory=list)

    def append(self, kind: str, seqno: int, **data: Any) -> None:
        if kind in {"image", "inline-image"}:
            metadata = image_display_metadata(kind, data)
            if metadata:
                explicit = data.get("image_metadata")
                if isinstance(explicit, dict):
                    metadata.update(explicit)
            source = data.get("image_source")
            if not isinstance(source, ImageSource):
                dictionary = data.get("dictionary")
                raw = data.get("raw_data", data.get("data"))
                if isinstance(dictionary, dict) and isinstance(raw, (bytes, bytearray, memoryview)):
                    source = ImageSource(
                        memoryview(raw).cast("B") if isinstance(raw, bytearray) else raw,
                        dictionary,
                    )
                else:
                    source = None
            self.items.append(
                ImagePaintItem(
                    paint_kind=kind,
                    seqno=seqno,
                    bbox=rect_tuple(data.get("bbox")),
                    source=source,
                    quad=internal_image_quad(data),
                    fill=data.get("fill") or data.get("fill_color"),
                    fill_opacity=data.get("fill_opacity"),
                    blend_mode=data.get("blend_mode"),
                    soft_mask_alpha=data.get("soft_mask_alpha"),
                    image_clip=data.get("image_clip"),
                    image_metadata=metadata,
                    ctm=data.get("ctm"),
                    xobject_depth=data.get("xobject_depth"),
                )
            )
            return
        if kind == "shading" and "prepared_shading" not in data:
            data["prepared_shading"] = prepare_shading(data.get("dictionary"))
        paint_kind = PATH_PAINT_KINDS.get(kind)
        if (
            paint_kind is not None
            and data.get("fill_pattern") is None
            and data.get("stroke_pattern") is None
        ):
            self.items.append(
                PathPaintItem(
                    paint_kind=paint_kind,
                    seqno=seqno,
                    bbox=data.get("bbox"),
                    path=data.get("path"),
                    fill=data.get("fill") or data.get("fill_color"),
                    fill_opacity=data.get("fill_opacity"),
                    stroke_color=data.get("stroke_color"),
                    stroke_opacity=data.get("stroke_opacity"),
                    line_width=data.get("line_width"),
                    line_cap=data.get("line_cap"),
                    line_join=data.get("line_join"),
                    dash_pattern=data.get("dash_pattern"),
                    fill_rule=data.get("fill_rule"),
                    blend_mode=data.get("blend_mode"),
                    soft_mask_alpha=data.get("soft_mask_alpha"),
                )
            )
            return
        self.items.append(DisplayListItem(kind=kind, seqno=seqno, data=data))

    def append_captured_drawing(self, drawing: Any) -> None:
        """Append a captured drawing without rebuilding its keyword-data mapping."""
        paint_kind = PATH_PAINT_KINDS.get(drawing.kind)
        if (
            paint_kind is not None
            and drawing.fill_pattern is None
            and drawing.stroke_pattern is None
        ):
            path = drawing.path
            previous = self.items[-1] if self.items else None
            if (
                paint_kind is PathPaintKind.STROKE
                and type(path) is CapturedPath
                and type(previous) is PathPaintItem
                and previous.paint_kind is PathPaintKind.STROKE
                and type(previous.path) is CapturedPath
                and len(previous.path.subpaths) + len(path.subpaths)
                <= MAX_COALESCED_STROKE_SUBPATHS
                and previous.stroke_color == drawing.stroke_color
                and previous.stroke_opacity == drawing.stroke_opacity
                and previous.line_width == drawing.line_width
                and previous.line_cap == drawing.line_cap
                and previous.line_join == drawing.line_join
                and previous.dash_pattern == drawing.dash_pattern
                and previous.blend_mode == drawing.blend_mode
                and previous.soft_mask_alpha == drawing.soft_mask_alpha
            ):
                previous_box = rect_tuple(previous.bbox)
                drawing_box = rect_tuple(drawing.rect)
                if previous.coalesced_path:
                    previous.path.subpaths.extend(path.subpaths)
                else:
                    previous.path = CapturedPath([*previous.path.subpaths, *path.subpaths])
                    previous.coalesced_path = True
                previous.bbox = (
                    (
                        min(previous_box[0], drawing_box[0]),
                        min(previous_box[1], drawing_box[1]),
                        max(previous_box[2], drawing_box[2]),
                        max(previous_box[3], drawing_box[3]),
                    )
                    if previous_box is not None and drawing_box is not None
                    else None
                )
                return
            self.items.append(
                PathPaintItem(
                    paint_kind=paint_kind,
                    seqno=drawing.seqno,
                    bbox=drawing.rect,
                    path=drawing.path,
                    fill=drawing.fill,
                    fill_opacity=drawing.fill_opacity,
                    stroke_color=drawing.stroke_color,
                    stroke_opacity=drawing.stroke_opacity,
                    line_width=drawing.line_width,
                    line_cap=drawing.line_cap,
                    line_join=drawing.line_join,
                    dash_pattern=drawing.dash_pattern,
                    fill_rule=drawing.fill_rule,
                    blend_mode=drawing.blend_mode,
                    soft_mask_alpha=drawing.soft_mask_alpha,
                )
            )
            return
        self.append(
            drawing.kind,
            drawing.seqno,
            bbox=drawing.rect,
            fill=drawing.fill,
            fill_pattern=drawing.fill_pattern,
            fill_opacity=drawing.fill_opacity,
            stroke_color=drawing.stroke_color,
            stroke_pattern=drawing.stroke_pattern,
            stroke_opacity=drawing.stroke_opacity,
            line_width=drawing.line_width,
            line_cap=drawing.line_cap,
            line_join=drawing.line_join,
            dash_pattern=drawing.dash_pattern,
            fill_rule=drawing.fill_rule,
            blend_mode=drawing.blend_mode,
            soft_mask_alpha=drawing.soft_mask_alpha,
            raw_data=drawing.raw_data,
            dictionary=drawing.dictionary,
            image_source=drawing.image_source,
            image_clip=drawing.image_clip,
            path=drawing.path,
            items=drawing.items,
        )


RASTER_CONTROL_KINDS = frozenset({"state-push", "state-pop", "clip", "group-begin", "group-end"})
RENDER_TILE_SIZE = 128.0
MAX_TILES_PER_ITEM = 256


def internal_display_item_box(item: DisplayItem) -> tuple[float, float, float, float] | None:
    """Compute the conservative paint bounds used by the compiled render plan."""
    if type(item) is ImagePaintItem:
        return rect_tuple(item.bbox)
    if type(item) is PathPaintItem:
        value = item.bbox
        if value is None and type(item.path) is CapturedPath:
            value = item.path.bbox()
        box = rect_tuple(value)
        if box is None:
            return None
        if item.paint_kind in {PathPaintKind.STROKE, PathPaintKind.FILL_STROKE}:
            line_width = item.line_width
            if pdf_number(line_width):
                pad = max(0.0, float(line_width) * 0.5)
                box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        return box
    generic_item = cast(DisplayListItem, item)
    data = generic_item.data
    if generic_item.kind in {"text", "glyph", "image", "inline-image"}:
        value = data.get("bbox")
    elif generic_item.kind in {"annotation", "widget"}:
        value = data.get("rect")
    elif generic_item.kind == "shading":
        value = data.get("bbox") or data.get("rect")
    elif generic_item.kind in {"fill", "fillstroke", "stroke"}:
        value = data.get("bbox")
        if value is None and type(data.get("path")) is CapturedPath:
            value = data["path"].bbox()
    else:
        return None
    box = rect_tuple(value)
    if box is None:
        return None
    if generic_item.kind in {"stroke", "fillstroke"} and pdf_number(data.get("line_width")):
        pad = max(0.0, float(data["line_width"]) * 0.5)
        box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
    return box


@dataclass(frozen=True, slots=True)
class CompiledRenderPlan:
    """Immutable draw sequence with a coarse spatial index for crop rendering."""

    items: tuple[DisplayItem, ...]
    raster_indexes: tuple[int, ...]
    always_indexes: tuple[int, ...]
    boxes: tuple[tuple[float, float, float, float] | None, ...]
    tiles: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def compile(cls, display_list: DisplayList) -> CompiledRenderPlan:
        items = tuple(display_list.items)
        raster_indexes: list[int] = []
        always_indexes: list[int] = []
        boxes: list[tuple[float, float, float, float] | None] = [None] * len(items)
        mutable_tiles: dict[tuple[int, int], list[int]] = {}
        for index, item in enumerate(items):
            if type(item) is DisplayListItem and item.kind == "text":
                continue
            raster_indexes.append(index)
            box = internal_display_item_box(item)
            boxes[index] = box
            if box is None or (type(item) is DisplayListItem and item.kind in RASTER_CONTROL_KINDS):
                always_indexes.append(index)
                continue
            tile_x0 = math.floor(box[0] / RENDER_TILE_SIZE)
            tile_y0 = math.floor(box[1] / RENDER_TILE_SIZE)
            tile_x1 = math.floor(box[2] / RENDER_TILE_SIZE)
            tile_y1 = math.floor(box[3] / RENDER_TILE_SIZE)
            tile_count = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
            if tile_count > MAX_TILES_PER_ITEM:
                always_indexes.append(index)
                continue
            for tile_y in range(tile_y0, tile_y1 + 1):
                for tile_x in range(tile_x0, tile_x1 + 1):
                    mutable_tiles.setdefault((tile_x, tile_y), []).append(index)
        return cls(
            items,
            tuple(raster_indexes),
            tuple(always_indexes),
            tuple(boxes),
            {key: tuple(indexes) for key, indexes in mutable_tiles.items()},
        )

    def items_for_crop(
        self,
        crop: tuple[float, float, float, float] | None,
    ) -> tuple[DisplayItem, ...]:
        if crop is None:
            if len(self.raster_indexes) == len(self.items):
                return self.items
            return tuple(self.items[index] for index in self.raster_indexes)
        tile_x0 = math.floor(crop[0] / RENDER_TILE_SIZE)
        tile_y0 = math.floor(crop[1] / RENDER_TILE_SIZE)
        tile_x1 = math.floor(crop[2] / RENDER_TILE_SIZE)
        tile_y1 = math.floor(crop[3] / RENDER_TILE_SIZE)
        selected = set(self.always_indexes)
        for tile_y in range(tile_y0, tile_y1 + 1):
            for tile_x in range(tile_x0, tile_x1 + 1):
                selected.update(self.tiles.get((tile_x, tile_y), ()))
        candidates = []
        for index in sorted(selected):
            box = self.boxes[index]
            if box is not None and (
                box[2] <= crop[0] or box[0] >= crop[2] or box[3] <= crop[1] or box[1] >= crop[3]
            ):
                continue
            candidates.append(self.items[index])
        return tuple(candidates)


__all__ = (
    "CompiledRenderPlan",
    "DisplayItem",
    "DisplayList",
    "DisplayListItem",
    "ImagePaintItem",
    "LineCap",
    "LineJoin",
    "PathPaintItem",
    "PathPaintKind",
    "RenderOptions",
    "internal_image_quad",
)
