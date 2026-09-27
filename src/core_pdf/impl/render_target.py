# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Iterable
from copy import replace
from typing import Any

import numpy

from core_pdf.impl.array_views import (
    UInt8Array,
    uint8_image_view,
)
from core_pdf.impl.caches import MISSING
from core_pdf.impl.capture_records import (
    CapturedPath,
    CapturedSoftMask,
    ShadingPattern,
    TilingPattern,
)
from core_pdf.impl.geometry import normalize_rect, rect_tuple
from core_pdf.impl.graphics_shading import (
    PreparedShading,
    prepare_shading,
)
from core_pdf.impl.render_blend import (
    blend_context,
    color_rgba,
    resolve_constant_alpha,
    scale_rgba_alpha,
)
from core_pdf.impl.render_clipping import ClipState
from core_pdf.impl.render_commands import append_captured_program, translated_command
from core_pdf.impl.render_display import DisplayList
from core_pdf.impl.render_fills import (
    BLEND_MODE_CODES,
)
from core_pdf.impl.render_images import (
    RasterImages,
)
from core_pdf.impl.render_model import (
    DisplayItem,
    ImagePaintItem,
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
    SoftMaskPlane,
)
from core_pdf.impl.render_paths import (
    intersect_box,
)
from core_pdf.impl.render_patterns import (
    cell_paints_nothing,
    shading_color_rgba,
    tiling_cell,
    tiling_pattern_uses_normal_blends,
)
from core_pdf.impl.render_resources import RenderResources
from core_pdf.impl.render_strokes import paint_stroke_once
from core_pdf.impl.scalars import clamp01
from core_pdf_cythonized import (
    alpha_channel,
    shading_blend,
    shading_values,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.standards import SemanticContext


def shading_rgba(
    color_model: str,
    components: list[float] | tuple[float, ...],
    fill_opacity: object,
    rendering: ColorRendering,
    shading_alpha: float | None,
) -> tuple[int, int, int, int]:
    rgba = shading_color_rgba(color_model, components, fill_opacity, rendering)
    return rgba if shading_alpha is None else scale_rgba_alpha(rgba, shading_alpha)


def prepared_shading(
    resources: RenderResources, dictionary: object, rendering: ColorRendering
) -> PreparedShading | None:
    cache = resources.shadings
    cached = cache.get(dictionary, rendering, default=MISSING)
    if cached is not MISSING:
        return cached
    shading = prepare_shading(
        dictionary, rendering=rendering, evaluators=resources.shading_evaluators
    )
    return cache.put(dictionary, shading, rendering)


def graphics_soft_mask(item: DisplayItem) -> CapturedSoftMask | None:
    if isinstance(item, (PathPaintItem, ImagePaintItem)):
        return item.graphics_soft_mask
    mask: CapturedSoftMask | None = item.data.get("graphics_soft_mask")
    return mask


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


class RasterTarget(RasterImages):
    __slots__ = ()

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: ClipState,
        width: int,
        height: int,
        scale: float,
        crop_x0: float,
        crop_y0: float,
        crop_y1: float,
        page_view: UInt8Array,
        semantic_context: SemanticContext | None = None,
        resources: RenderResources | None = None,
    ) -> None:
        self.semantic_context = blend_context(semantic_context)
        self.buffer_stack = [RasterGroup(pixels, view=page_view)]
        self.sync_group_mirrors()
        self.paint_alpha_is_shape = False
        self.shape_alpha = 1.0
        self.clip = clip
        self.width = width
        self.height = height
        self.scale = scale
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.crop_y0 = crop_y0
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
            clip=ClipState(
                crop_x0=self.crop_x0,
                crop_y1=self.crop_y1,
                scale=self.scale,
                width=self.width,
                height=self.height,
            ),
            width=self.width,
            height=self.height,
            scale=self.scale,
            crop_x0=self.crop_x0,
            crop_y0=self.crop_y0,
            crop_y1=self.crop_y1,
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
                self.paint_item(translated_command(item, *translation, parent_blend_mode))
        finally:
            self.pop_scope()

    def paint_item(self, item: DisplayItem) -> None:
        if isinstance(item, (PathPaintItem, ImagePaintItem)) or item.kind in {"glyph", "shading"}:
            previous_shape_state = self.paint_alpha_is_shape, self.shape_alpha
            self.paint_alpha_is_shape = (
                item.alpha_is_shape
                if isinstance(item, (PathPaintItem, ImagePaintItem))
                else item.data.get("alpha_is_shape") is True
            )
            self.shape_alpha = 1.0
            try:
                knockout = self.buffer_stack[-1].knockout
                mask = graphics_soft_mask(item)
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
        if isinstance(item, PathPaintItem):
            self.paint_typed_path(item)
            return
        if isinstance(item, ImagePaintItem):
            self.blit_image(item)
            return
        data = item.data
        blend_mode = data.get("blend_mode")
        if blend_mode == "Normal":
            blend_mode = None
        match item.kind:
            case "scope-begin":
                path = data.get("path")
                self.push_scope(path if isinstance(path, CapturedPath) else None)
            case "scope-end":
                self.pop_scope()
            case "state-push":
                self.clip_stack.append(self.clip.depth)
            case "state-pop":
                if self.clip_stack:
                    self.clip.restore(self.clip_stack.pop())
                else:
                    self.clip.restore(self.clip_floor)
            case "clip":
                path = data.get("path")
                if isinstance(path, CapturedPath) and path.has_segments():
                    self.clip.push(path, data.get("fill_rule") or "nonzero")
            case "group-begin":
                opacity = data.get("fill_opacity")
                mask = data.get("soft_mask_alpha")
                if is_pdf_number(mask):
                    opacity = (float(opacity) if is_pdf_number(opacity) else 1.0) * float(mask)
                isolated = data.get("group_isolated", True)
                knockout = data.get("group_knockout", False)
                group_mask_alpha = (
                    resolve_soft_mask(self, graphics_mask)
                    if (graphics_mask := data.get("graphics_soft_mask")) is not None
                    else None
                )
                region = self.knockout_group_region(item) if not isolated and knockout else None
                if not isolated and (not knockout or region is not None):
                    self.push_scratch_group(
                        opacity,
                        data.get("blend_mode"),
                        track_shape=data.get("group_track_shape", False),
                        mask_alpha=group_mask_alpha,
                        alpha_is_shape=data.get("alpha_is_shape", False),
                        region=region,
                    )
                else:
                    self.push_group(
                        bytearray(self.width * self.height * 4),
                        opacity,
                        data.get("blend_mode"),
                        isolated=isolated,
                        knockout=knockout,
                        alpha_is_shape=data.get("alpha_is_shape", False),
                        track_shape=data.get("group_track_shape", False),
                        mask_alpha=group_mask_alpha,
                    )
            case "group-end" if len(self.buffer_stack) > self.group_floor:
                self.composite_group(self.pop_group())
            case "glyph" if data.get("visible") is not False:
                rgba = color_rgba(data.get("fill_color"), data.get("fill_opacity"))
                if is_pdf_number(mask := data.get("soft_mask_alpha")):
                    rgba = scale_rgba_alpha(rgba, mask)
                self.set_shape_alpha(rgba[3] / 255.0)
                self.draw_glyph_bitmap(
                    data.get("bbox"),
                    data.get("bitmap"),
                    rgba,
                    blend_mode,
                    data.get("bitmap_width"),
                    data.get("bitmap_height"),
                )
            case "shading":
                self.set_shape_alpha(
                    resolve_constant_alpha(data.get("fill_opacity"), data.get("soft_mask_alpha"))
                )
                self.paint_shading(data, blend_mode)

    def paint_typed_path(self, item: PathPaintItem) -> None:
        path = item.path
        if type(path) is not CapturedPath:
            return
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
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
            rgba = color_rgba(item.fill, item.fill_opacity)
            if is_pdf_number(soft_mask_alpha):
                rgba = scale_rgba_alpha(rgba, soft_mask_alpha)
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

    def raster_page_box(self) -> tuple[float, float, float, float]:
        return (self.crop_x0, self.crop_y0, self.crop_x0 + self.width / self.scale, self.crop_y1)

    def shading_box(
        self,
        data: dict[str, Any],
        shading: PreparedShading,
    ) -> tuple[float, float, float, float]:
        box = shading.bbox
        if box is None:
            box = rect_tuple(data.get("bbox"))
        if box is None:
            box = self.raster_page_box()
        return normalize_rect(box)

    def prepared_shading(
        self, dictionary: object, rendering: ColorRendering
    ) -> PreparedShading | None:
        return prepared_shading(self.resources, dictionary, rendering)

    def paint_shading(self, data: dict[str, Any], blend_mode: str | None) -> None:
        shading = self.prepared_shading(
            data.get("dictionary"), data.get("color_rendering", DEFAULT_COLOR_RENDERING)
        )
        if shading is None:
            return
        clipped_box = self.clip.clipped_pixel_box(self.shading_box(data, shading))
        if clipped_box is None:
            return
        ix0, iy0, ix1, iy1 = clipped_box[1]
        soft_mask_alpha = data.get("soft_mask_alpha")
        fill_opacity = data.get("fill_opacity")
        shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
        mode = BLEND_MODE_CODES.get(self.resolved_blend(blend_mode), 0)
        domain = shading.domain
        allowed = numpy.frombuffer(
            self.clip_pixel_mask(ix0, iy0, ix1, iy1), dtype=numpy.uint8
        ).reshape(iy1 - iy0, ix1 - ix0)
        values, painted = shading_values(
            shading.shading_type,
            shading.coords,
            self.crop_x0,
            self.crop_y1,
            self.scale,
            ix0,
            iy0,
            ix1,
            iy1,
            allowed,
            bool(shading.extend_start),
            bool(shading.extend_end),
            domain[0],
            domain[1] - domain[0],
            0.5,
        )
        ordered = values[painted.view(numpy.bool_)]
        if not len(ordered):
            return
        _, first, inverse = numpy.unique(ordered, return_index=True, return_inverse=True)
        by_first = numpy.argsort(first, kind="stable")
        colors = numpy.zeros((len(first), 4), dtype=numpy.int32)
        reached = len(ordered)
        color_error: Exception | None = None
        color_model = shading.color_model
        evaluate = shading.evaluator
        rendering = shading.color_rendering
        for unique_index in by_first.tolist():
            position = int(first[unique_index])
            try:
                colors[unique_index] = shading_rgba(
                    color_model,
                    evaluate(float(ordered[position])),
                    fill_opacity,
                    rendering,
                    shading_alpha,
                )
            except Exception as error:
                color_error = error
                reached = position
                break
        revised, blend_error = self.blend_rules(mode)
        shape_plane = self.group_source_shape
        window, stopped = shading_blend(
            self.pixel_array,
            ix0,
            iy0,
            painted,
            numpy.ascontiguousarray(inverse.reshape(-1)[:reached], dtype=numpy.int64),
            colors,
            mode,
            revised,
            self.group_source_alpha,
            shape_plane,
            255 / 255.0 * self.shape_alpha if shape_plane is not None else 0.0,
            blend_error is not None,
        )
        if window is not None and self.paint_window is not None:
            x0, y0, x1, y1 = window
            self.extend_paint_box(y0, y1, x0, x1)
        if stopped:
            assert blend_error is not None
            raise blend_error
        if color_error is not None:
            raise color_error

    def paint_tiling_pattern(
        self,
        pattern: TilingPattern,
        target_data: PathPaintItem,
        blend_mode: str | None,
    ) -> bool:
        scale = self.scale
        cell_x0, cell_y0, cell_x1, cell_y1 = pattern.bbox
        x_step = abs(pattern.x_step)
        y_step = abs(pattern.y_step)
        if x_step <= 0.0 or y_step <= 0.0:
            return False
        program = pattern.program
        if not program.drawings and not program.glyphs and not program.inline_images:
            return False
        display, cell_clip = tiling_cell(
            self.resources,
            pattern,
            self.width,
            self.height,
            preserve_object_boundaries=self.group_source_shape is not None,
        )
        if cell_paints_nothing(display.items, cell_clip, scale):
            return True
        target_box = target_data.bbox or self.clip.path_bbox(target_data.path)
        if (type(target_box) is list or type(target_box) is tuple) and len(target_box) == 4:
            box = rect_tuple(target_box)
            if box is None:
                return False
        else:
            box = self.raster_page_box()
        x0, y0, x1, y1 = box
        clip_box = self.clip.current_clip()
        if clip_box is not None:
            clipped = intersect_box((x0, y0, x1, y1), clip_box)
            if clipped is None:
                return True
            x0, y0, x1, y1 = clipped
        start_x = cell_x0 + math.floor((x0 - cell_x0) / x_step) * x_step
        start_y = cell_y0 + math.floor((y0 - cell_y0) / y_step) * y_step
        cells = 0
        y = start_y
        opacity = target_data.fill_opacity
        alpha = clamp01(opacity) if is_pdf_number(opacity) else 1.0
        if is_pdf_number(target_data.soft_mask_alpha):
            alpha *= clamp01(target_data.soft_mask_alpha)
        self.push_group(
            bytearray(len(self.pixels)),
            alpha,
            blend_mode,
            isolated=tiling_pattern_uses_normal_blends(pattern),
            alpha_is_shape=target_data.alpha_is_shape,
        )
        try:
            while y < y1 + y_step and cells < 10000:
                x = start_x
                while x < x1 + x_step and cells < 10000:
                    tx = x - cell_x0
                    ty = y - cell_y0
                    if x + (cell_x1 - cell_x0) >= x0 and y + (cell_y1 - cell_y0) >= y0:
                        self.paint_items(
                            display.items,
                            translation=(tx, ty),
                            parent_blend_mode=None,
                            clip_path=cell_clip,
                        )
                    cells += 1
                    x += x_step
                y += y_step
        finally:
            self.composite_group(self.pop_group())
        return True

    def paint_fill_pattern(self, data: PathPaintItem, blend_mode: str | None) -> bool:
        clip_state = self.clip
        pattern = data.fill_pattern
        if not isinstance(pattern, (ShadingPattern, TilingPattern)):
            return False
        path = data.path
        pushed_clip = False
        if type(path) is CapturedPath and path.has_segments():
            clip_state.push(path, data.fill_rule or "nonzero")
            pushed_clip = True
        try:
            if isinstance(pattern, ShadingPattern):
                dictionary = pattern.dictionary
                if not isinstance(dictionary, dict):
                    return False
                shading_data = {
                    "dictionary": dictionary,
                    "bbox": data.bbox or clip_state.path_bbox(path),
                    "fill_opacity": data.fill_opacity,
                    "soft_mask_alpha": data.soft_mask_alpha,
                    "color_rendering": pattern.color_rendering,
                }
                self.paint_shading(shading_data, blend_mode)
                return True
            return self.paint_tiling_pattern(pattern, data, blend_mode)
        finally:
            if pushed_clip:
                clip_state.pop()
