# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from math import isfinite
from typing import Any, ClassVar, Self

from core_pdf.impl.base.geometry import finite_rect, rect_tuple, union_bbox
from core_pdf.impl.base.glyphs import GlyphStyle
from core_pdf.impl.base.values import lenient_int
from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath, CapturedSoftMask
from core_pdf.impl.graphics.color_spec import describe_color_space
from core_pdf.impl.graphics.filter_registry import declared_filter_names
from core_pdf.impl.render.model import (
    DEFAULT_MITER_LIMIT,
    ControlItem,
    DisplayItem,
    DisplayListItem,
    GroupBeginItem,
    ImagePaintItem,
    PathPaintItem,
    PathPaintKind,
    display_item,
    is_plain_fill,
    path_paint_fields,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource, SoftMask
lazy from core_pdf_cythonized import SlotLayout

PATH_PAINT_KINDS = {
    name: PathPaintKind(index) for index, name in enumerate(("fill", "stroke", "fillstroke"))
}
MAX_COALESCED_STROKE_SUBPATHS = 256


def image_display_metadata(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, dict):
        return {}

    width = lenient_int(dictionary.get("Width"), 0)
    height = lenient_int(dictionary.get("Height"), 0)
    width = max(0, width)
    height = max(0, height)
    image_mask = dictionary.get("ImageMask") is True
    default_bpc = 1 if image_mask else 0
    bits_per_component = lenient_int(dictionary.get("BitsPerComponent"), default_bpc)
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


def quad_points(quad: list[Any] | tuple[Any, ...]) -> tuple[tuple[float, float], ...] | None:
    try:
        return tuple((float(point[0]), float(point[1])) for point in quad)
    except TypeError, ValueError, IndexError:
        return None


def image_quad(data: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    quad = data.get("quad")
    if isinstance(quad, (list, tuple)) and len(quad) >= 3:
        return quad_points(quad)
    items = data.get("items")
    if not isinstance(items, (list, tuple)):
        return None
    for kind, value in items:
        if kind != "quad":
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        return quad_points(value)
    return None


class PlainFillScan:
    """How far plain_fill_members_box has read an open group's members: up to end, all
    of them plain fills or text, their boxes' union x0, y0, x1, y1 when found."""

    __slots__ = ("end", "found", "x0", "y0", "x1", "y1")

    def __init__(self, end: int) -> None:
        self.end = end
        self.found = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0.0

    def add_box(self, bx0: float, by0: float, bx1: float, by1: float) -> None:
        """Take in the next member, a plain fill of this finite box."""
        self.end += 1
        if not self.found:
            self.found = True
            self.x0, self.y0, self.x1, self.y1 = bx0, by0, bx1, by1
            return
        # min() and max() keep the running value on ties, as union_bbox did.
        if bx0 < self.x0:
            self.x0 = bx0
        if by0 < self.y0:
            self.y0 = by0
        if bx1 > self.x1:
            self.x1 = bx1
        if by1 > self.y1:
            self.y1 = by1


def plain_fill_members_box(
    items: list[DisplayItem], scan: PlainFillScan
) -> tuple[bool, tuple[float, float, float, float] | None]:
    """Whether every member of a group, items from the scan's start on, is a plain fill
    or text, and then the union of the fills' boxes; the scan holds what was already
    read of them."""
    for index in range(scan.end, len(items)):
        item = items[index]
        if type(item) is DisplayListItem and item.kind == "text":
            scan.end += 1
            continue
        if not is_plain_fill(item):
            return False, None
        bbox = item.bbox
        if (
            type(bbox) is tuple
            and len(bbox) == 4
            and type(bbox[0]) is float
            and type(bbox[1]) is float
            and type(bbox[2]) is float
            and type(bbox[3]) is float
        ):
            bx0, by0, bx1, by1 = bbox
            if not (isfinite(bx0) and isfinite(by0) and isfinite(bx1) and isfinite(by1)):
                return False, None
        else:
            item_box = finite_rect(bbox, require_positive=False)
            if item_box is None:
                return False, None
            bx0, by0, bx1, by1 = item_box
        scan.add_box(bx0, by0, bx1, by1)
    return True, ((scan.x0, scan.y0, scan.x1, scan.y1) if scan.found else None)


PATH_PAINT_LAYOUT = SlotLayout(PathPaintItem, PathPaintItem.__fields__)


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
        "open_group_scans",
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
        "open_group_scans",
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
        self.group_member_boxes: dict[int, tuple[float, float, float, float] | None] = {}
        self.glyph_paint_fields: dict[int, tuple[GlyphStyle, tuple[Any, ...]]] = {}
        # One for each open group: how far its members are known to be plain fills.
        self.open_group_scans: list[PlainFillScan] = []

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
            width, height, items, preserve_object_boundaries=preserve_object_boundaries
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
                self.open_group_indexes.append(len(self.items))
                self.open_group_scans.append(PlainFillScan(len(self.items) + 1))
            case "group-end":
                floor = self.group_scope_floors[-1] if self.group_scope_floors else 0
                if len(self.shape_tracking_groups) > floor:
                    self.shape_tracking_groups.pop()
                if self.open_group_indexes:
                    begin = self.open_group_indexes.pop()
                    plain, box = plain_fill_members_box(self.items, self.open_group_scans.pop())
                    if plain and begin < len(self.items):
                        self.group_member_boxes[id(self.items[begin])] = box

    def glyph_paint_fields_of(self, style: GlyphStyle) -> tuple[Any, ...]:
        """The fields of a glyph's PathPaintItem that come from its style, from fill to
        stroke_pattern in PathPaintItem's order; ProgramLowering.lower_glyphs builds the
        items."""
        cached = self.glyph_paint_fields.get(id(style))
        if cached is not None and cached[0] is style:
            return cached[1]
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
            style.fill_pattern,
            style.stroke_pattern,
            style.alpha_is_shape,
            graphics_soft_mask if isinstance(graphics_soft_mask, CapturedSoftMask) else None,
        )
        self.glyph_paint_fields[id(style)] = (style, fields)
        return fields

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
                    fill_overprint=data.get("fill_overprint"),
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
                    miter_limit=(
                        float(limit)
                        if is_pdf_number(limit := data.get("miter_limit"))
                        else DEFAULT_MITER_LIMIT
                    ),
                    stroke_matrix=data.get("stroke_matrix"),
                    dash_pattern=data.get("dash_pattern"),
                    fill_rule=data.get("fill_rule") or "nonzero",
                    blend_mode=data.get("blend_mode"),
                    soft_mask_alpha=data.get("soft_mask_alpha"),
                    alpha_is_shape=data.get("alpha_is_shape", False),
                    graphics_soft_mask=graphics_mask,
                    fill_pattern=data.get("fill_pattern"),
                    stroke_pattern=data.get("stroke_pattern"),
                    edge_array=data.get("edge_array"),
                    fill_overprint=data.get("fill_overprint"),
                    stroke_overprint=data.get("stroke_overprint"),
                )
            )
            return
        self.track_group_boundary(kind, data)
        items = self.items
        if kind == "text":
            scans = self.open_group_scans
            if scans and scans[-1].end == len(items):
                # plain_fill_members_box passes over text.
                scans[-1].end += 1
        items.append(display_item(kind, seqno, data))

    def append_text_group(self, seqno: int, *, knockout: bool) -> None:
        """append("group-begin", ...) for a text object's or glyph's group: non-isolated,
        opaque, in the Normal blend mode, tracking shape, and a knockout group when
        knockout."""
        data: dict[str, Any] = {
            "fill_opacity": 1.0,
            "blend_mode": None,
            "group_isolated": False,
            "group_knockout": knockout,
            "group_track_shape": True,
        }
        items = self.items
        # track_group_boundary for a group that tracks shape.
        self.shape_tracking_groups.append(True)
        self.open_group_indexes.append(len(items))
        self.open_group_scans.append(PlainFillScan(len(items) + 1))
        items.append(
            GroupBeginItem(
                seqno, 1.0, None, None, False, knockout, True, False, None, None, None, data
            )
        )

    def append_group_end(self, seqno: int) -> None:
        """append("group-end", seqno)."""
        data: dict[str, Any] = {}
        self.track_group_boundary("group-end", data)
        self.items.append(ControlItem("group-end", seqno, data))

    def append_text(self, seqno: int, **data: Any) -> None:
        """append("text", seqno, **data), for data without a graphics_soft_mask."""
        items = self.items
        scans = self.open_group_scans
        if scans and scans[-1].end == len(items):
            scans[-1].end += 1
        items.append(DisplayListItem("text", seqno, data))

    def append_plain_fill(
        self, item: PathPaintItem, bx0: float, by0: float, bx1: float, by1: float
    ) -> None:
        """Append item, a plain fill (is_plain_fill) of the finite box bx0, by0, bx1, by1,
        reading it into the open group's scan when nothing unread comes before it."""
        items = self.items
        scans = self.open_group_scans
        if scans:
            scan = scans[-1]
            if scan.end == len(items):
                scan.add_box(bx0, by0, bx1, by1)
        items.append(item)

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
                and previous.miter_limit == drawing.miter_limit
                and previous.dash_pattern == drawing.dash_pattern
                and previous.blend_mode == drawing.blend_mode
                and previous.soft_mask_alpha == drawing.soft_mask_alpha
                and previous.alpha_is_shape == drawing.alpha_is_shape
                and previous.stroke_overprint == drawing.stroke_overprint
                # One stroke is one shape painted once, and two strokes are two: the kernel
                # paints each of a coalesced path's in turn, so only strokes it paints, opaque
                # and blending Normal without a stroke matrix, are joined. The previous one
                # matches this one's opacity and blend mode.
                and drawing.stroke_matrix is None
                and previous.stroke_matrix is None
                and (drawing.stroke_opacity is None or drawing.stroke_opacity == 1.0)
                and (drawing.soft_mask_alpha is None or drawing.soft_mask_alpha == 1.0)
                and (drawing.blend_mode is None or drawing.blend_mode == "Normal")
            ):
                previous_box = rect_tuple(previous.bbox)
                drawing_box = rect_tuple(drawing.rect)
                merged = previous.path.coalesced_with(path)
                if merged is not None:
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
            **(
                {"group_color_space": drawing.group_color_space}
                if drawing.group_color_space is not None
                else {}
            ),
            raw_data=drawing.raw_data,
            dictionary=drawing.dictionary,
            image_source=drawing.image_source,
            image_clip=drawing.image_clip,
            color_rendering=drawing.color_rendering,
            path=drawing.path,
            items=drawing.items,
            **({"matrix": drawing.shading_matrix} if drawing.shading_matrix is not None else {}),
        )


__all__ = ("PATH_PAINT_LAYOUT", "DisplayList", "image_quad")
