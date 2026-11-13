# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: painting a display list: each item goes to the handler for its type.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import replace
from typing import TYPE_CHECKING, Any

from core_pdf.impl.capture.records import CapturedPath
from core_pdf.impl.render.blend import (
    color_rgba,
    declared_blend,
    resolve_constant_alpha,
    scale_rgba_alpha,
)
from core_pdf.impl.render.model import (
    ClipItem,
    ControlItem,
    DisplayItem,
    GlyphBitmapItem,
    GroupBeginItem,
    ImagePaintItem,
    PaintItemBase,
    PathPaintItem,
    PathPaintKind,
    ScopeBeginItem,
    ShadingItem,
    is_plain_fill,
)
from core_pdf.impl.render.raster_images import blit_image
from core_pdf.impl.render.raster_support import (
    EMPTY_PIXEL_BOX,
    paint_stroke_once,
    resolve_soft_mask,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget


def paint_items(
    self: RasterTarget,
    items: Iterable[DisplayItem],
    *,
    translation: tuple[float, float] | None = None,
    parent_blend_mode: str | None = None,
    clip_path: CapturedPath | None = None,
) -> None:
    if translation is None:
        for item in items:
            self.paint_item(item)
        return
    self.push_scope(clip_path.translated(*translation) if clip_path is not None else None)
    try:
        for item in items:
            self.paint_item(item.translated(*translation, parent_blend_mode))
    finally:
        self.pop_scope()


def paint_item(self: RasterTarget, item: DisplayItem) -> None:
    if isinstance(item, PaintItemBase):
        previous_shape_state = self.paint_alpha_is_shape, self.shape_alpha
        self.paint_alpha_is_shape = item.alpha_is_shape
        self.shape_alpha = 1.0
        try:
            knockout = self.buffer_stack[-1].knockout
            mask = item.graphics_soft_mask
            mask_alpha = resolve_soft_mask(self, mask) if mask is not None else None
            fillstroke = (
                isinstance(item, PathPaintItem) and item.paint_kind is PathPaintKind.FILL_STROKE
            )
            elementary_group = knockout or mask_alpha is not None
            boxes = self.buffer_stack[-1].painted_boxes
            skip_box = None
            if knockout and boxes is not None and mask_alpha is None and is_plain_fill(item):
                # A plain fill painting pixels no earlier member touched needs no
                # elementary group.
                clipped = self.clip.clipped_pixel_box(item.bbox)
                if clipped is None:
                    skip_box = EMPTY_PIXEL_BOX
                else:
                    skip_box = clipped[1]
                    if skip_box != EMPTY_PIXEL_BOX:
                        x0, y0, x1, y1 = skip_box
                        for bx0, by0, bx1, by1 in boxes:
                            if x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1:
                                skip_box = None
                                break
                if skip_box is not None:
                    elementary_group = False
            if (
                elementary_group
                and knockout
                and mask_alpha is None
                and self.knockout_glyph_fill(item)
            ):
                return
            if elementary_group:
                self.push_scratch_group(
                    None,
                    None,
                    track_shape=mask_alpha is not None,
                    mask_alpha=None if fillstroke else mask_alpha,
                    alpha_is_shape=self.paint_alpha_is_shape,
                )
            try:
                self.paint_display_item(item)
            finally:
                if elementary_group:
                    self.composite_group(self.pop_group())
                elif skip_box is not None:
                    self.record_knockout_paint(skip_box)
        finally:
            self.paint_alpha_is_shape, self.shape_alpha = previous_shape_state
        return
    self.paint_display_item(item)


def paint_display_item(self: RasterTarget, item: DisplayItem) -> None:
    handler = PAINT_HANDLERS.get(type(item))
    if handler is not None:
        handler(self, item)


def paint_scope_begin(self: RasterTarget, item: ScopeBeginItem) -> None:
    path = item.path
    self.push_scope(path if isinstance(path, CapturedPath) else None)


def paint_clip(self: RasterTarget, item: ClipItem) -> None:
    path = item.path
    if isinstance(path, CapturedPath) and path.has_segments():
        self.clip.push(path, item.fill_rule or "nonzero")


def paint_control(self: RasterTarget, item: ControlItem) -> None:
    match item.kind:
        case "scope-end":
            self.pop_scope()
        case "state-push":
            self.clip_stack.append(self.clip.depth)
        case "state-pop":
            if self.clip_stack:
                self.clip.restore(self.clip_stack.pop())
            else:
                self.clip.restore(self.clip_floor)
        case "group-end" if len(self.buffer_stack) > self.group_floor:
            self.composite_group(self.pop_group())


def paint_group_begin(self: RasterTarget, item: GroupBeginItem) -> None:
    opacity = item.opacity()
    isolated = item.isolated
    knockout = item.knockout
    graphics_mask = item.graphics_soft_mask
    group_mask_alpha = resolve_soft_mask(self, graphics_mask) if graphics_mask is not None else None
    region = self.knockout_group_region(item) if not isolated and knockout else None
    if not isolated and (not knockout or region is not None):
        self.push_scratch_group(
            opacity,
            item.blend_mode,
            track_shape=item.track_shape,
            mask_alpha=group_mask_alpha,
            alpha_is_shape=item.alpha_is_shape,
            region=region,
        )
    else:
        self.push_group(
            bytearray(self.width * self.height * 4),
            opacity,
            item.blend_mode,
            isolated=isolated,
            knockout=knockout,
            alpha_is_shape=item.alpha_is_shape,
            track_shape=item.track_shape,
            mask_alpha=group_mask_alpha,
        )


def paint_glyph_bitmap(self: RasterTarget, item: GlyphBitmapItem) -> None:
    if item.visible is False:
        return
    rgba = item.fill_rgba()
    self.set_shape_alpha(rgba[3] / 255.0)
    self.draw_glyph_bitmap(
        item.bbox,
        item.bitmap,
        rgba,
        declared_blend(item.blend_mode),
        item.bitmap_width,
        item.bitmap_height,
    )


def paint_shading_item(self: RasterTarget, item: ShadingItem) -> None:
    self.set_shape_alpha(resolve_constant_alpha(item.fill_opacity, item.soft_mask_alpha))
    self.paint_shading(item, declared_blend(item.blend_mode))


def paint_typed_path(self: RasterTarget, item: PathPaintItem) -> None:
    path = item.path
    if type(path) is not CapturedPath:
        return
    blend_mode = declared_blend(item.blend_mode)
    soft_mask_alpha = item.soft_mask_alpha
    paint_kind = item.paint_kind
    if paint_kind is PathPaintKind.FILL_STROKE and self.group_source_shape is not None:
        self.push_group(bytearray(len(self.pixels)), None, None, isolated=False, knockout=True)
        try:
            self.paint_item(replace(item, paint_kind=PathPaintKind.FILL))
            self.paint_item(replace(item, paint_kind=PathPaintKind.STROKE))
        finally:
            self.composite_group(self.pop_group())
        return
    if paint_kind is not PathPaintKind.STROKE:
        rgba = item.fill_rgba()
        self.set_shape_alpha(rgba[3] / 255.0)
        if item.fill_pattern is None or not self.paint_fill_pattern(item, blend_mode):
            edge_array = item.edge_array
            self.fill_path(
                path,
                rgba,
                blend_mode,
                item.fill_rule,
                bbox=item.bbox if edge_array is not None else None,
                edge_array=edge_array,
            )
    if paint_kind is not PathPaintKind.FILL:
        stroke_rgba = color_rgba(item.stroke_color, item.stroke_opacity)
        if is_pdf_number(soft_mask_alpha):
            stroke_rgba = scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
        self.set_shape_alpha(stroke_rgba[3] / 255.0)
        if self.group_source_shape is not None:
            paint_stroke_once(
                self,
                path,
                item.line_width,
                stroke_rgba,
                item.dash_pattern,
                blend_mode,
                item.line_cap,
                item.line_join,
            )
            return
        self.stroke_path(
            path,
            item.line_width,
            stroke_rgba,
            item.dash_pattern,
            blend_mode,
            item.line_cap,
            item.line_join,
        )


PAINT_HANDLERS: dict[type, Callable[[RasterTarget, Any], None]] = {
    PathPaintItem: paint_typed_path,
    ImagePaintItem: blit_image,
    GlyphBitmapItem: paint_glyph_bitmap,
    ShadingItem: paint_shading_item,
    ScopeBeginItem: paint_scope_begin,
    ClipItem: paint_clip,
    GroupBeginItem: paint_group_begin,
    ControlItem: paint_control,
}
