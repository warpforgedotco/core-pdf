# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import replace
from typing import Any

import numpy

from core_pdf.impl.array_views import UInt8Array, uint8_image_view
from core_pdf.impl.capture_records import CapturedPath, CapturedSoftMask
from core_pdf.impl.render_blend import (
    blend_context,
    color_rgba,
    declared_blend,
    resolve_constant_alpha,
    scale_rgba_alpha,
)
from core_pdf.impl.render_clipping import ClipState
from core_pdf.impl.render_commands import append_captured_program
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_model import (
    ClipItem,
    ControlItem,
    DisplayItem,
    GlyphBitmapItem,
    GroupBeginItem,
    ImagePaintItem,
    PaintItemBase,
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
    ScopeBeginItem,
    ShadingItem,
    SoftMaskPlane,
)
from core_pdf.impl.render_resources import RenderResources
from core_pdf.impl.render_shading import RasterShading
from core_pdf.impl.render_strokes import paint_stroke_once
from core_pdf_cythonized import alpha_channel
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.standards import SemanticContext


def resolve_soft_mask(target: RasterTarget, mask: CapturedSoftMask) -> SoftMaskPlane | None:
    resources = target.resources
    key = (id(mask.program), id(mask.transfer), mask.offset)
    cached = resources.soft_masks.get(key)
    if cached is not None:
        return cached[1]
    if key in resources.active_soft_masks:
        return None
    resources.active_soft_masks.add(key)
    result: SoftMaskPlane | None = None
    try:
        nested, view = target.blank_sibling()
        display = DisplayList(target.width, target.height, preserve_object_boundaries=True)
        append_captured_program(display, mask.program, include_text=True)
        nested.paint_items(display.items, translation=mask.offset)
        alpha, present = alpha_channel(view, mask.transfer is not None)
        alpha.setflags(write=False)
        if mask.transfer is None or present is None:
            result = SoftMaskPlane(alpha, None)
        else:
            samples = numpy.flatnonzero(present)
            values = [mask.transfer(int(sample) / 255.0)[0] for sample in samples]
            table = numpy.zeros(256, dtype=numpy.float32)
            table[samples] = numpy.asarray(values, dtype=numpy.float32)
            table.setflags(write=False)
            result = SoftMaskPlane(alpha, table)
    except Exception:
        result = None
    finally:
        resources.active_soft_masks.remove(key)
    resources.soft_masks.store(key, (mask, result), 0 if result is None else result.nbytes)
    return result


class RasterTarget(RasterShading):
    __slots__ = ()

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: ClipState,
        page_view: UInt8Array,
        semantic_context: SemanticContext | None = None,
        resources: RenderResources | None = None,
    ) -> None:
        self.semantic_context = blend_context(semantic_context)
        self.buffer_stack = [RasterGroup(pixels, view=page_view)]
        self.sync_group_mirrors()
        self.paint_alpha_is_shape = False
        self.shape_alpha = 1.0
        grid = clip.grid
        self.clip = clip
        self.grid = grid
        self.width = grid.width
        self.height = grid.height
        self.scale = grid.scale
        self.crop_x0 = grid.crop_x0
        self.crop_y1 = grid.crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.crop_y0 = grid.crop_y0
        self.clip_stack = []
        self.clip_floor = 0
        self.group_floor = 1
        self.scope_stack = []
        self.resources = RenderResources() if resources is None else resources
        self.elementary_scratch = {}
        self.group_member_boxes = None
        self.stroke_scratch = None
        if group_alpha is not None:
            self.push_group(bytearray(len(pixels)), group_alpha, None)
            self.group_floor = len(self.buffer_stack)

    def blank_sibling(self) -> tuple[RasterTarget, UInt8Array]:
        pixels = bytearray(self.width * self.height * 4)
        view = uint8_image_view(pixels, (self.height, self.width, 4))
        sibling = RasterTarget(
            pixels,
            None,
            clip=ClipState(self.grid),
            page_view=view,
            semantic_context=self.semantic_context,
            resources=self.resources,
        )
        return sibling, view

    def paint_items(
        self,
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

    def paint_item(self, item: DisplayItem) -> None:
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
                if knockout and boxes is not None and mask_alpha is None:
                    skip_box = self.knockout_skip_box(item, boxes)
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

    def paint_display_item(self, item: DisplayItem) -> None:
        handler = PAINT_HANDLERS.get(type(item))
        if handler is not None:
            handler(self, item)

    def paint_scope_begin(self, item: ScopeBeginItem) -> None:
        path = item.path
        self.push_scope(path if isinstance(path, CapturedPath) else None)

    def paint_clip(self, item: ClipItem) -> None:
        path = item.path
        if isinstance(path, CapturedPath) and path.has_segments():
            self.clip.push(path, item.fill_rule or "nonzero")

    def paint_control(self, item: ControlItem) -> None:
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

    def paint_group_begin(self, item: GroupBeginItem) -> None:
        opacity = item.opacity()
        isolated = item.isolated
        knockout = item.knockout
        graphics_mask = item.graphics_soft_mask
        group_mask_alpha = (
            resolve_soft_mask(self, graphics_mask) if graphics_mask is not None else None
        )
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

    def paint_glyph_bitmap(self, item: GlyphBitmapItem) -> None:
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

    def paint_shading_item(self, item: ShadingItem) -> None:
        self.set_shape_alpha(resolve_constant_alpha(item.fill_opacity, item.soft_mask_alpha))
        self.paint_shading(item, declared_blend(item.blend_mode))

    def paint_typed_path(self, item: PathPaintItem) -> None:
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
    PathPaintItem: RasterTarget.paint_typed_path,
    ImagePaintItem: RasterTarget.blit_image,
    GlyphBitmapItem: RasterTarget.paint_glyph_bitmap,
    ShadingItem: RasterTarget.paint_shading_item,
    ScopeBeginItem: RasterTarget.paint_scope_begin,
    ClipItem: RasterTarget.paint_clip,
    GroupBeginItem: RasterTarget.paint_group_begin,
    ControlItem: RasterTarget.paint_control,
}
