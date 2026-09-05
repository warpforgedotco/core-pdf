# SPDX-License-Identifier: AGPL-3.0-only
"""Gradient-pattern geometry and colour evaluation."""

from __future__ import annotations

import math
from typing import Any, cast

from core_pdf.impl.model.geometry import RectBox, rect_tuple
from core_pdf.impl.render.blend import (
    internal_clamp01,
    internal_color_component,
    internal_scale_rgba_alpha,
)
from core_pdf.impl.render.commands import append_captured_program
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import PathPaintItem
from core_pdf.impl.render.paths import internal_intersect_box
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedPath,
    ShadingPattern,
    TilingPattern,
)
from core_pdf.impl.spec.s_07_content.page_program import CapturedProgram
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf.impl.spec.s_08_graphics.device_profiles import cmyk_floats_to_srgb
from core_pdf.impl.spec.s_08_graphics.shading import PreparedShading, prepare_shading


def axial_shading_t(coords: list[float] | tuple[float, ...], px: float, py: float) -> float | None:
    x0, y0, x1, y1 = coords[:4]
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return None
    return ((px - x0) * dx + (py - y0) * dy) / denom


def radial_shading_t(coords: list[float] | tuple[float, ...], px: float, py: float) -> float | None:
    x0, y0, r0, x1, y1, r1 = coords[:6]
    dx = x1 - x0
    dy = y1 - y0
    dr = r1 - r0
    qx = px - x0
    qy = py - y0
    a = dx * dx + dy * dy - dr * dr
    b = -2.0 * (qx * dx + qy * dy + r0 * dr)
    c = qx * qx + qy * qy - r0 * r0
    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return None
        return -c / b
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = disc**0.5
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    valid = [t for t in (t0, t1) if math.isfinite(t)]
    if not valid:
        return None
    in_range = [t for t in valid if 0.0 <= t <= 1.0]
    return max(in_range) if in_range else min(valid, key=lambda t: abs(t - 0.5))


def internal_shading_color_rgba(
    color_model: str,
    components: list[float] | tuple[float, ...],
    opacity: Any,
) -> tuple[int, int, int, int]:
    alpha = internal_color_component(opacity, 255) if type(opacity) in {int, float} else 255
    name = color_model or "DeviceRGB"
    if name.endswith("DeviceGray") or len(components) == 1:
        gray = internal_color_component(components[0] if components else 0.0)
        return gray, gray, gray, alpha
    if name.endswith("DeviceCMYK") and len(components) >= 4:
        c, m, y, k = (internal_clamp01(v) for v in components[:4])
        red, green, blue = cmyk_floats_to_srgb(c, m, y, k)
        return red, green, blue, alpha
    rgb = [internal_color_component(c) for c in components[:3]]
    while len(rgb) < 3:
        rgb.append(rgb[-1] if rgb else 0)
    return rgb[0], rgb[1], rgb[2], alpha


class internal_PatternTargetMixin:
    """Stateful gradient and tiling-pattern painting operations."""

    __slots__ = ()

    def shading_box(
        self: Any,
        data: dict[str, Any],
        shading: PreparedShading,
    ) -> tuple[float, float, float, float]:
        crop_x0 = self.crop_x0
        crop_y0 = self.crop_y0
        crop_y1 = self.crop_y1
        scale = self.scale
        width = self.width
        box = shading.bbox
        if box is None:
            box = rect_tuple(data.get("bbox"))
        if box is None:
            box = (crop_x0, crop_y0, crop_x0 + width / scale, crop_y1)
        x0, y0, x1, y1 = box
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def paint_shading(self: Any, data: dict[str, Any], blend_mode: str | None) -> None:
        clipped_pixel_box = self.clip.clipped_pixel_box
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        scale = self.scale
        shading_box = self.shading_box
        width = self.width
        shading = prepare_shading(data.get("dictionary"))
        if shading is None:
            return
        shading_type = shading.shading_type
        coords = shading.coords
        domain = shading.domain
        extend0 = shading.extend_start
        extend1 = shading.extend_end
        clipped_box = clipped_pixel_box(shading_box(data, shading))
        if clipped_box is None:
            return
        ix0, iy0, ix1, iy1 = clipped_box[1]
        soft_mask_alpha = data.get("soft_mask_alpha")
        fill_opacity = data.get("fill_opacity")
        normal_fast = can_blend_normal_fast(blend_mode)
        # Fixed for the whole shading; resolving it per pixel re-ran is_pdf_number
        # and float() once per device pixel of the fill.
        shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
        domain_span = domain[1] - domain[0]
        # page_x only depends on the column, so it is identical on every row;
        # computing it once here avoids redoing the same division per pixel.
        page_x_values = [crop_x0 + (px + 0.5) / scale for px in range(ix0, ix1)]
        for py in range(iy0, iy1):
            page_y = crop_y1 - (py + 0.5) / scale
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            # Walking the spans directly answers a per-row question once per
            # span, where the bisect answered it again for every pixel.
            for span_start, span_end in visible_spans:
                for px in range(max(ix0, span_start), min(ix1, span_end)):
                    page_x = page_x_values[px - ix0]
                    unit_t = (
                        axial_shading_t(coords, page_x, page_y)
                        if shading_type == 2
                        else radial_shading_t(coords, page_x, page_y)
                    )
                    if unit_t is None:
                        continue
                    if unit_t < 0.0:
                        if not extend0:
                            continue
                        unit_t = 0.0
                    elif unit_t > 1.0:
                        if not extend1:
                            continue
                        unit_t = 1.0
                    value = domain[0] + unit_t * domain_span
                    rgba = internal_shading_color_rgba(
                        shading.color_model,
                        shading.evaluate(value),
                        fill_opacity,
                    )
                    if shading_alpha is not None:
                        rgba = internal_scale_rgba_alpha(rgba, shading_alpha)
                    if normal_fast:
                        blend_normal_pixel(row + px * 4, *rgba)
                    else:
                        blend_px(row + px * 4, rgba, blend_alpha_scale, blend_resolved_mode)

    def paint_tiling_pattern(
        self: Any,
        pattern: TilingPattern,
        target_data: PathPaintItem,
        blend_mode: str | None,
    ) -> bool:
        crop_x0 = self.crop_x0
        crop_y0 = self.crop_y0
        crop_y1 = self.crop_y1
        scale = self.scale
        width = self.width
        cell_x0, cell_y0, cell_x1, cell_y1 = pattern.bbox
        x_step = abs(pattern.x_step)
        y_step = abs(pattern.y_step)
        if x_step <= 0.0 or y_step <= 0.0:
            return False
        drawings = pattern.drawings
        glyphs = pattern.glyphs
        if not drawings and not glyphs and not pattern.inline_images:
            return False
        display = DisplayList(width, self.height)
        cell_clip = CapturedPath()
        cell_clip.rect(cell_x0, cell_y0, cell_x1 - cell_x0, cell_y1 - cell_y0)
        append_captured_program(
            display,
            CapturedProgram(
                drawings=tuple(drawings),
                glyphs=tuple(glyphs),
                inline_images=tuple(pattern.inline_images),
            ),
            include_text=True,
        )
        target_box = target_data.bbox or self.clip.path_bbox(target_data.path)
        target_box_type = type(target_box)
        if target_box_type is RectBox:
            target_rect = cast(RectBox, target_box)
            x0, y0, x1, y1 = (
                target_rect.x0,
                target_rect.y0,
                target_rect.x1,
                target_rect.y1,
            )
        elif target_box_type is list or target_box_type is tuple:
            target_box = cast(list[Any] | tuple[Any, ...], target_box)
            if len(target_box) == 4:
                try:
                    x0, y0, x1, y1 = (float(value) for value in target_box)
                except (TypeError, ValueError):
                    return False
            else:
                x0, y0, x1, y1 = (
                    crop_x0,
                    crop_y0,
                    crop_x0 + width / scale,
                    crop_y1,
                )
        else:
            x0, y0, x1, y1 = crop_x0, crop_y0, crop_x0 + width / scale, crop_y1
        clip_box = self.clip.current_clip()
        if clip_box is not None:
            clipped = internal_intersect_box((x0, y0, x1, y1), clip_box)
            if clipped is None:
                return True
            x0, y0, x1, y1 = clipped
        start_x = cell_x0 + math.floor((x0 - cell_x0) / x_step) * x_step
        start_y = cell_y0 + math.floor((y0 - cell_y0) / y_step) * y_step
        cells = 0
        y = start_y
        while y < y1 + y_step and cells < 10000:
            x = start_x
            while x < x1 + x_step and cells < 10000:
                tx = x - cell_x0
                ty = y - cell_y0
                if x + (cell_x1 - cell_x0) >= x0 and y + (cell_y1 - cell_y0) >= y0:
                    self.paint_items(
                        display.items,
                        translation=(tx, ty),
                        parent_blend_mode=blend_mode,
                        clip_path=cell_clip,
                    )
                cells += 1
                x += x_step
            y += y_step
        return True

    def paint_fill_pattern(self: Any, data: PathPaintItem, blend_mode: str | None) -> bool:
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
                }
                self.paint_shading(shading_data, blend_mode)
                return True
            return self.paint_tiling_pattern(pattern, data, blend_mode)
        finally:
            if pushed_clip:
                clip_state.pop()
        return False
