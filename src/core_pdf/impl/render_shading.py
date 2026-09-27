# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math

import numpy

from core_pdf.impl.capture_records import CapturedPath, ShadingPattern, TilingPattern
from core_pdf.impl.geometry import normalize_rect, rect_tuple
from core_pdf.impl.graphics_shading import PreparedShading, prepare_shading
from core_pdf.impl.render_blend import kernel_blend_code, scale_rgba_alpha
from core_pdf.impl.render_images import RasterImages
from core_pdf.impl.render_model import PathPaintItem, ShadingItem
from core_pdf.impl.render_paths import intersect_box
from core_pdf.impl.render_patterns import (
    cell_paints_nothing,
    shading_color_rgba,
    tiling_cell,
    tiling_pattern_uses_normal_blends,
)
from core_pdf.impl.render_resources import RenderResources
from core_pdf.impl.scalars import clamp01
from core_pdf.impl.types import MISSING, MissingObject
from core_pdf_cythonized import shading_blend, shading_values
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.color_rendering import ColorRendering


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
    key = (id(dictionary), rendering)
    cached = cache.get_key(dictionary, key, MISSING)
    if not isinstance(cached, MissingObject):
        return cached
    shading = prepare_shading(
        dictionary, rendering=rendering, evaluators=resources.shading_evaluators
    )
    return cache.put_key(dictionary, key, shading)


class RasterShading(RasterImages):
    __slots__ = ()

    def raster_page_box(self) -> tuple[float, float, float, float]:
        return self.grid.page_box()

    def shading_box(
        self,
        item: ShadingItem,
        shading: PreparedShading,
    ) -> tuple[float, float, float, float]:
        box = shading.bbox
        if box is None:
            box = rect_tuple(item.bbox)
        if box is None:
            box = self.raster_page_box()
        return normalize_rect(box)

    def prepared_shading(
        self, dictionary: object, rendering: ColorRendering
    ) -> PreparedShading | None:
        return prepared_shading(self.resources, dictionary, rendering)

    def paint_shading(self, item: ShadingItem, blend_mode: str | None) -> None:
        shading = self.prepared_shading(item.dictionary, item.color_rendering)
        if shading is None:
            return
        clipped_box = self.clip.clipped_pixel_box(self.shading_box(item, shading))
        if clipped_box is None:
            return
        ix0, iy0, ix1, iy1 = clipped_box[1]
        soft_mask_alpha = item.soft_mask_alpha
        fill_opacity = item.fill_opacity
        shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
        mode = kernel_blend_code(self.resolved_blend(blend_mode))
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
            self.paint_window.extend_box(window)
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
                self.paint_shading(ShadingItem.from_data(data.seqno, shading_data), blend_mode)
                return True
            return self.paint_tiling_pattern(pattern, data, blend_mode)
        finally:
            if pushed_clip:
                clip_state.pop()
