# SPDX-License-Identifier: AGPL-3.0-only
"""Display-list construction and crop-aware render planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from core_pdf.impl._impl.model.geometry import rect_tuple
from core_pdf.impl._impl.render.model import (
    DisplayItem,
    DisplayListItem,
    ImagePaintItem,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedPath
from core_pdf.impl.spec.s_07_filters.registry import declared_filter_names
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    is_pdf_number,
    parse_int,
)
from core_pdf.impl.spec.s_08_graphics.color_spec import describe_color_space
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource, SoftMask

PATH_PAINT_KINDS = {
    name: PathPaintKind(index) for index, name in enumerate(("fill", "stroke", "fillstroke"))
}
MAX_COALESCED_STROKE_SUBPATHS = 256
RASTER_CONTROL_KINDS = frozenset(
    {"state-push", "state-pop", "clip", "group-begin", "group-end", "scope-begin", "scope-end"}
)


def internal_image_display_metadata(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Describe an image while adapting captured data into a display item."""
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, dict):
        return {}

    width = parse_int(dictionary.get("Width"), 0)
    height = parse_int(dictionary.get("Height"), 0)
    width = width if width > 0 else 0
    height = height if height > 0 else 0
    image_mask = dictionary.get("ImageMask") is True
    default_bpc = 1 if image_mask else 0
    bits_per_component = parse_int(dictionary.get("BitsPerComponent"), default_bpc)
    bits_per_component = bits_per_component if bits_per_component > 0 else default_bpc
    image_source = data.get("image_source")
    has_soft_mask = (
        dictionary.get("SMask") is not None
        or isinstance(data.get("soft_mask"), SoftMask)
        or isinstance(image_source, ImageSource)
        and image_source.soft_mask is not None
    )

    metadata: dict[str, Any] = {
        "kind": kind,
        "width": width,
        "height": height,
        "pixels": width * height if width > 0 and height > 0 else 0,
        "bits_per_component": bits_per_component if bits_per_component > 0 else None,
        "color_space": describe_color_space(dictionary.get("ColorSpace")),
        "filters": declared_filter_names(dictionary.get("Filter")),
        "image_mask": image_mask,
        "has_mask": dictionary.get("Mask") is not None,
        "has_soft_mask": has_soft_mask,
    }

    raw_data = data.get("raw_data", data.get("data"))
    if isinstance(raw_data, (bytes, bytearray, memoryview)):
        metadata["raw_bytes"] = len(raw_data)

    bbox = rect_tuple(data.get("bbox"))
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        display_width = abs(x1 - x0)
        display_height = abs(y1 - y0)
        metadata["display_width"] = display_width
        metadata["display_height"] = display_height
        metadata["display_area"] = display_width * display_height

    return metadata


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
            metadata = internal_image_display_metadata(kind, data)
            if metadata:
                explicit = data.get("source_metadata")
                if isinstance(explicit, dict):
                    metadata.update(explicit)
            source = data.get("image_source")
            if not isinstance(source, ImageSource):
                dictionary = data.get("dictionary")
                raw = data.get("raw_data", data.get("data"))
                if isinstance(dictionary, dict) and isinstance(raw, (bytes, bytearray, memoryview)):
                    soft_mask = data.get("soft_mask")
                    source = ImageSource(
                        memoryview(raw).cast("B") if isinstance(raw, bytearray) else raw,
                        dictionary,
                        soft_mask=soft_mask if isinstance(soft_mask, SoftMask) else None,
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
                    source_metadata=metadata,
                    ctm=data.get("ctm"),
                    xobject_depth=data.get("xobject_depth"),
                )
            )
            return
        paint_kind = PATH_PAINT_KINDS.get(kind)
        if paint_kind is not None:
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
                    fill_pattern=data.get("fill_pattern"),
                    stroke_pattern=data.get("stroke_pattern"),
                )
            )
            return
        self.items.append(DisplayListItem(kind=kind, seqno=seqno, data=data))

    def append_captured_drawing(self, drawing: CapturedDrawing) -> None:
        """Append a captured drawing without rebuilding its keyword-data mapping."""
        paint_kind = PATH_PAINT_KINDS.get(drawing.kind)
        if paint_kind is not None:
            path = drawing.path
            previous = self.items[-1] if self.items else None
            if (
                paint_kind is PathPaintKind.STROKE
                and drawing.stroke_pattern is None
                and type(path) is CapturedPath
                and type(previous) is PathPaintItem
                and previous.paint_kind is PathPaintKind.STROKE
                and previous.stroke_pattern is None
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
                    fill_pattern=drawing.fill_pattern,
                    stroke_pattern=drawing.stroke_pattern,
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


def internal_display_item_box(item: DisplayItem) -> tuple[float, float, float, float] | None:
    """Compute conservative bounds for crop-aware rasterization."""
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
            if is_pdf_number(line_width):
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
    if generic_item.kind in {"stroke", "fillstroke"} and is_pdf_number(data.get("line_width")):
        pad = max(0.0, float(data["line_width"]) * 0.5)
        box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
    return box


__all__ = (
    "DisplayList",
    "internal_image_quad",
)
