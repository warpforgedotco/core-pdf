# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Self

from core_pdf.impl.capture_records import CapturedDrawing, CapturedPath, CapturedSoftMask
from core_pdf.impl.geometry import finite_rect, rect_tuple, union_bbox
from core_pdf.impl.glyphs import GlyphStyle
from core_pdf.impl.graphics_color_spec import describe_color_space
from core_pdf.impl.graphics_filter_registry import declared_filter_names
from core_pdf.impl.render_model import (
    DisplayItem,
    DisplayListItem,
    ImagePaintItem,
    PathPaintItem,
    PathPaintKind,
    is_plain_fill,
    path_paint_fields,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number, parse_int
from core_pdf_spec.s_08_graphics.image_spec import ImageSource, SoftMask

PATH_PAINT_KINDS = {
    name: PathPaintKind(index) for index, name in enumerate(("fill", "stroke", "fillstroke"))
}
MAX_COALESCED_STROKE_SUBPATHS = 256
RASTER_CONTROL_KINDS = frozenset(
    {"state-push", "state-pop", "clip", "group-begin", "group-end", "scope-begin", "scope-end"}
)


def image_display_metadata(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, dict):
        return {}

    width = parse_int(dictionary.get("Width"), 0, python_syntax=True)
    height = parse_int(dictionary.get("Height"), 0, python_syntax=True)
    width = max(0, width)
    height = max(0, height)
    image_mask = dictionary.get("ImageMask") is True
    default_bpc = 1 if image_mask else 0
    bits_per_component = parse_int(
        dictionary.get("BitsPerComponent"), default_bpc, python_syntax=True
    )
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


def image_quad(data: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    quad = data.get("quad")
    if isinstance(quad, (list, tuple)) and len(quad) >= 3:
        try:
            return tuple((float(point[0]), float(point[1])) for point in quad)
        except TypeError, ValueError, IndexError:
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
        except TypeError, ValueError, IndexError:
            return None
    return None


def plain_fill_members_box(
    items: list[DisplayItem], start: int
) -> tuple[bool, tuple[float, float, float, float] | None]:
    """Whether items[start:] are all plain fills, and the union of their bboxes.

    A plain fill -- an edge-array fill with no pattern and a Normal blend,
    what RasterTarget.knockout_paint_box accepts -- paints inside its bbox as
    clipped, so a group of nothing else paints inside the union of theirs. A
    text item, which the rasterizer does not paint, may sit among them, and
    a group of nothing but those paints nothing: (True, None). Anything else
    in the group gives (False, None).
    """
    box: tuple[float, float, float, float] | None = None
    for index in range(start, len(items)):
        item = items[index]
        if type(item) is DisplayListItem and item.kind == "text":
            continue
        if not is_plain_fill(item):
            return False, None
        item_box = finite_rect(item.bbox, require_positive=False)
        if item_box is None:
            return False, None
        box = union_bbox(box, item_box)
    return True, box


class DisplayList:
    __slots__ = (
        "width",
        "height",
        "items",
        "preserve_object_boundaries",
        "shape_tracking_groups",
        "group_scope_floors",
        "open_group_indexes",
        "group_member_boxes",
        "glyph_paint_fields",
    )

    width: float
    height: float
    items: list[DisplayItem]
    preserve_object_boundaries: bool
    shape_tracking_groups: list[bool]
    group_scope_floors: list[int]

    __fields__: ClassVar[tuple[str, ...]] = (
        "width",
        "height",
        "items",
        "preserve_object_boundaries",
        "shape_tracking_groups",
        "group_scope_floors",
        "open_group_indexes",
        "group_member_boxes",
        "glyph_paint_fields",
    )
    __match_args__ = ("width", "height", "items")

    def __init__(
        self,
        width: float,
        height: float,
        items: list[DisplayItem] | None = None,
        *,
        preserve_object_boundaries: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.items = [] if items is None else items
        self.preserve_object_boundaries = preserve_object_boundaries
        self.shape_tracking_groups = []
        self.group_scope_floors = []
        self.open_group_indexes: list[int] = []
        # For a group whose members are all plain fills, keyed by the identity
        # of its group-begin item: the page box they paint within, or None if
        # it has none. See plain_fill_members_box.
        self.group_member_boxes: dict[int, tuple[float, float, float, float] | None] = {}
        # A glyph style's paint fields, normalized, keyed by the style's
        # identity and holding the style so that identity stays its own.
        self.glyph_paint_fields: dict[int, tuple[GlyphStyle, tuple[Any, ...]]] = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"items={self.items!r}, "
            f"preserve_object_boundaries={self.preserve_object_boundaries!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.items == other.items
            and self.preserve_object_boundaries == other.preserve_object_boundaries
            and self.shape_tracking_groups == other.shape_tracking_groups
            and self.group_scope_floors == other.group_scope_floors
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        items = changes.pop("items", self.items)
        preserve_object_boundaries = changes.pop(
            "preserve_object_boundaries", self.preserve_object_boundaries
        )
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            width,
            height,
            items,
            preserve_object_boundaries=preserve_object_boundaries,
        )

    def track_group_boundary(self, kind: str, data: dict[str, Any]) -> None:
        match kind:
            case "scope-begin":
                self.group_scope_floors.append(len(self.shape_tracking_groups))
            case "scope-end":
                if self.group_scope_floors:
                    del self.shape_tracking_groups[self.group_scope_floors.pop() :]
            case "group-begin":
                self.shape_tracking_groups.append(
                    data.get("group_knockout") is True or data.get("group_track_shape") is True
                )
                # The group-begin item is appended next, at this index.
                self.open_group_indexes.append(len(self.items))
            case "group-end":
                floor = self.group_scope_floors[-1] if self.group_scope_floors else 0
                if len(self.shape_tracking_groups) > floor:
                    self.shape_tracking_groups.pop()
                if self.open_group_indexes:
                    begin = self.open_group_indexes.pop()
                    plain, box = plain_fill_members_box(self.items, begin + 1)
                    if plain and begin < len(self.items):
                        self.group_member_boxes[id(self.items[begin])] = box

    def append_glyph_paint(
        self,
        paint_kind: PathPaintKind,
        seqno: int,
        bbox: Any,
        path: Any,
        edge_array: Any,
        style: GlyphStyle,
    ) -> None:
        """append() for a glyph's path paint, whose other fields are its style's.

        A page appends one per glyph drawn, and every glyph of a text
        operation shares one style, so its fields are normalized once, as
        append normalizes them, and each item is built positionally from them;
        the keyword calls this replaced were most of the cost. A pattern is
        never given.
        """
        cached = self.glyph_paint_fields.get(id(style))
        if cached is None or cached[0] is not style:
            line_width = style.line_width
            graphics_soft_mask = style.graphics_soft_mask
            fields: tuple[Any, ...] = (
                style.fill or None,
                style.fill_opacity,
                style.stroke_color,
                style.stroke_opacity,
                float(line_width) if is_pdf_number(line_width) else 1.0,
                int(style.line_cap or 0),
                int(style.line_join or 0),
                style.dash_pattern,
                "nonzero",
                style.blend_mode,
                style.soft_mask_alpha,
                False,
                None,
                None,
                style.alpha_is_shape,
                graphics_soft_mask if isinstance(graphics_soft_mask, CapturedSoftMask) else None,
            )
            self.glyph_paint_fields[id(style)] = (style, fields)
        else:
            fields = cached[1]
        # Both checkers miscount a star argument followed by another positional.
        item = PathPaintItem(paint_kind, seqno, bbox, path, *fields, edge_array)  # type: ignore[call-arg]  # ty: ignore[too-many-positional-arguments]
        self.items.append(item)

    def append(self, kind: str, seqno: int, **data: Any) -> None:
        graphics_mask = data.get("graphics_soft_mask")
        if not isinstance(graphics_mask, CapturedSoftMask):
            graphics_mask = None
        if "graphics_soft_mask" in data:
            data["graphics_soft_mask"] = graphics_mask
        if kind in {"image", "inline-image"}:
            metadata = image_display_metadata(kind, data)
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
                    quad=image_quad(data),
                    fill=data.get("fill") or data.get("fill_color"),
                    fill_opacity=data.get("fill_opacity"),
                    blend_mode=data.get("blend_mode"),
                    soft_mask_alpha=data.get("soft_mask_alpha"),
                    alpha_is_shape=data.get("alpha_is_shape", False),
                    graphics_soft_mask=graphics_mask,
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
                    line_width=(
                        float(width) if is_pdf_number(width := data.get("line_width")) else 1.0
                    ),
                    line_cap=int(data.get("line_cap") or 0),
                    line_join=int(data.get("line_join") or 0),
                    dash_pattern=data.get("dash_pattern"),
                    fill_rule=data.get("fill_rule") or "nonzero",
                    blend_mode=data.get("blend_mode"),
                    soft_mask_alpha=data.get("soft_mask_alpha"),
                    alpha_is_shape=data.get("alpha_is_shape", False),
                    graphics_soft_mask=graphics_mask,
                    fill_pattern=data.get("fill_pattern"),
                    stroke_pattern=data.get("stroke_pattern"),
                    edge_array=data.get("edge_array"),
                )
            )
            return
        self.track_group_boundary(kind, data)
        self.items.append(DisplayListItem(kind=kind, seqno=seqno, data=data))

    def append_captured_drawing(self, drawing: CapturedDrawing) -> None:
        if not drawing.paints:
            return
        paint_kind = PATH_PAINT_KINDS.get(drawing.kind)
        if paint_kind is not None:
            fills = paint_kind is not PathPaintKind.STROKE and drawing.fill_paints
            strokes = paint_kind is not PathPaintKind.FILL and drawing.stroke_paints
            if not fills and not strokes:
                return
            paint_kind = (
                PathPaintKind.FILL_STROKE
                if fills and strokes
                else PathPaintKind.FILL
                if fills
                else PathPaintKind.STROKE
            )
            path = drawing.path
            previous = self.items[-1] if self.items else None
            if (
                paint_kind is PathPaintKind.STROKE
                and not self.preserve_object_boundaries
                and not any(self.shape_tracking_groups)
                and drawing.stroke_pattern is None
                and drawing.graphics_soft_mask is None
                and type(path) is CapturedPath
                and type(previous) is PathPaintItem
                and previous.paint_kind is PathPaintKind.STROKE
                and previous.stroke_pattern is None
                and previous.graphics_soft_mask is None
                and type(previous.path) is CapturedPath
                and previous.path.subpath_count() + path.subpath_count()
                <= MAX_COALESCED_STROKE_SUBPATHS
                and previous.stroke_color == drawing.stroke_color
                and previous.stroke_opacity == drawing.stroke_opacity
                and previous.line_width == drawing.line_width
                and previous.line_cap == drawing.line_cap
                and previous.line_join == drawing.line_join
                and previous.dash_pattern == drawing.dash_pattern
                and previous.blend_mode == drawing.blend_mode
                and previous.soft_mask_alpha == drawing.soft_mask_alpha
                and previous.alpha_is_shape == drawing.alpha_is_shape
            ):
                previous_box = rect_tuple(previous.bbox)
                drawing_box = rect_tuple(drawing.rect)
                merged = previous.path.coalesced_with(path)
                if merged is not None:
                    # Both are flattened paths whose points wait: so does
                    # the join.
                    previous.path = merged
                    previous.coalesced_path = True
                elif previous.coalesced_path:
                    previous.path.subpaths.extend(path.subpaths)
                else:
                    previous.path = CapturedPath([*previous.path.subpaths, *path.subpaths])
                    previous.coalesced_path = True
                previous.edge_array = None
                previous.bbox = (
                    union_bbox(previous_box, drawing_box)
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
                    **path_paint_fields(drawing),
                )
            )
            return
        self.append(
            drawing.kind,
            drawing.seqno,
            bbox=drawing.rect,
            **path_paint_fields(drawing),
            group_isolated=drawing.group_isolated,
            group_knockout=drawing.group_knockout,
            raw_data=drawing.raw_data,
            dictionary=drawing.dictionary,
            image_source=drawing.image_source,
            image_clip=drawing.image_clip,
            color_rendering=drawing.color_rendering,
            path=drawing.path,
            items=drawing.items,
        )


def display_item_box(
    item: DisplayItem, *, scale: float = 1.0
) -> tuple[float, float, float, float] | None:
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
            pad = max(0.5 / scale, item.line_width * 0.5)
            box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        return box
    generic_item = item
    data = generic_item.data
    if generic_item.kind in {"text", "glyph"}:
        value = data.get("bbox")
    elif generic_item.kind in {"annotation", "widget"}:
        value = data.get("rect")
    elif generic_item.kind == "shading":
        value = data.get("bbox") or data.get("rect")
    else:
        return None
    return rect_tuple(value)


__all__ = (
    "DisplayList",
    "image_quad",
)
