# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful path-stroke painting operations for raster targets."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl._impl.render.model import LineCap, LineJoin
from core_pdf.impl._impl.render.paths import (
    RASTER_KERNEL_MIN_PIXEL_AREA,
    RASTER_SAMPLE_OFFSETS,
    internal_intersect_box,
    rasterize_unclipped_line_normal,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedPath


class internal_PathStrokeTargetMixin:
    """Line, join, cap, and path-stroke painting operations."""

    __slots__ = ()

    def fill_line(
        self: Any,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        dash_pattern: tuple[list[float], float] | None = None,
        blend_mode: str | None = None,
        line_cap: int = 0,
    ) -> None:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        buffer_stack = self.buffer_stack
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        fill_circle = self.fill_circle
        fill_line = self.fill_line
        fill_rect = self.fill_rect
        pixel_in_clip = clip.pixel_in_clip
        pixels = self.pixels
        scale = self.scale
        width = self.width
        if dash_pattern and dash_pattern[0]:
            dash_array, phase = dash_pattern
            total = sum((max(0.0, float(v)) for v in dash_array), 0.0)
            if total > 0:
                seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if seg_len > 0:
                    # The prefix-sum walk over `dash_array` is loop-invariant:
                    # normalize once and precompute the cumulative sums, so the
                    # per-step lookup scans plain floats instead of redoing the
                    # max/float conversions on every dash step.
                    dash_cumulative: list[float] = []
                    acc = 0.0
                    for val in dash_array:
                        acc += max(0.0, float(val))
                        dash_cumulative.append(acc)
                    pos = float(phase) % total
                    on = True
                    remaining = seg_len
                    while remaining > 0:
                        dash_idx = 0
                        dash_end = dash_cumulative[-1]
                        for i, acc in enumerate(dash_cumulative):
                            if pos < acc:
                                dash_idx = i
                                dash_end = acc
                                break
                        on = (dash_idx % 2) == 0
                        step = min(
                            remaining,
                            dash_end - pos if dash_end > pos else total - pos,
                        )
                        if on and step > 0:
                            t0 = (seg_len - remaining) / seg_len
                            t1 = (seg_len - remaining + step) / seg_len
                            sx0 = x0 + (x1 - x0) * t0
                            sy0 = y0 + (y1 - y0) * t0
                            sx1 = x0 + (x1 - x0) * t1
                            sy1 = y0 + (y1 - y0) * t1
                            fill_line(
                                sx0,
                                sy0,
                                sx1,
                                sy1,
                                line_width,
                                rgba,
                                None,
                                blend_mode,
                                line_cap,
                            )
                        remaining -= step
                        pos = (pos + step) % total
                        if step <= 0:
                            break
                    return
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
            half = max(0.5 / scale, float(line_width) * 0.5)
            cap_extension = half if line_cap == 2 else 0.0
            if abs(dy) <= 1e-12:
                fill_rect(
                    (
                        min(x0, x1) - cap_extension,
                        y0 - half,
                        max(x0, x1) + cap_extension,
                        y0 + half,
                    ),
                    rgba,
                    blend_mode,
                )
                if line_cap == 1:
                    fill_circle(x0, y0, half, rgba, blend_mode)
                    fill_circle(x1, y1, half, rgba, blend_mode)
            else:
                fill_rect(
                    (
                        x0 - half,
                        min(y0, y1) - cap_extension,
                        x0 + half,
                        max(y0, y1) + cap_extension,
                    ),
                    rgba,
                    blend_mode,
                )
                if line_cap == 1:
                    fill_circle(x0, y0, half, rgba, blend_mode)
                    fill_circle(x1, y1, half, rgba, blend_mode)
            return
        seg_len2 = dx * dx + dy * dy
        half = max(0.5 / scale, float(line_width) * 0.5)
        if seg_len2 <= 1e-12:
            if line_cap == 1:
                fill_circle(x0, y0, half, rgba, blend_mode)
            else:
                fill_rect(
                    (x0 - half, y0 - half, x0 + half, y0 + half),
                    rgba,
                    blend_mode,
                )
            return

        seg_len = seg_len2**0.5
        ux = dx / seg_len
        uy = dy / seg_len
        cap_extension = half if line_cap == 2 else 0.0
        box = (
            min(x0, x1) - half - abs(ux) * cap_extension,
            min(y0, y1) - half - abs(uy) * cap_extension,
            max(x0, x1) + half + abs(ux) * cap_extension,
            max(y0, y1) + half + abs(uy) * cap_extension,
        )
        clipped_box = clipped_pixel_box(box)
        if clipped_box is None:
            return
        box, pixel_box = clipped_box
        ix0, iy0, ix1, iy1 = pixel_box
        samples = 4
        sample_total = samples * samples
        half2 = half * half
        inv_seg_len2 = 1.0 / seg_len2
        extension_t = cap_extension / seg_len
        normal_fast = blend_mode is None and buffer_stack[-1][1] is None
        if (
            (not clip_regions or clip_paths_are_axis_aligned_rects())
            and normal_fast
            and (ix1 - ix0) * (iy1 - iy0) > RASTER_KERNEL_MIN_PIXEL_AREA
        ):
            x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
            y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
            rasterize_unclipped_line_normal(
                pixels,
                width,
                crop_x0,
                crop_y1,
                scale,
                x0,
                y0,
                x1,
                y1,
                line_width,
                rgba,
                line_cap,
                pixel_box,
                target_pixels=self.pixel_view(pixels),
                x_coords=x_coords,
                y_coords=y_coords,
            )
            return
        for py in range(iy0, iy1):
            row = py * width * 4
            page_y_samples = tuple(
                crop_y1 - (py + sample_offset) / scale for sample_offset in RASTER_SAMPLE_OFFSETS
            )
            for px in range(ix0, ix1):
                if clip_regions and not pixel_in_clip(px, py):
                    continue
                page_x_samples = tuple(
                    crop_x0 + (px + sample_offset) / scale
                    for sample_offset in RASTER_SAMPLE_OFFSETS
                )
                covered = 0
                if line_cap == 0:
                    cross_limit = half2 * seg_len2
                    for page_y in page_y_samples:
                        offset_y = page_y - y0
                        for page_x in page_x_samples:
                            offset_x = page_x - x0
                            projection = offset_x * dx + offset_y * dy
                            if projection < 0.0 or projection > seg_len2:
                                continue
                            cross = offset_x * dy - offset_y * dx
                            if cross * cross <= cross_limit:
                                covered += 1
                elif line_cap == 1:
                    cross_limit = half2 * seg_len2
                    for page_y in page_y_samples:
                        offset_y = page_y - y0
                        for page_x in page_x_samples:
                            offset_x = page_x - x0
                            t = (offset_x * dx + offset_y * dy) * inv_seg_len2
                            if 0.0 <= t <= 1.0:
                                cross = offset_x * dy - offset_y * dx
                                if cross * cross <= cross_limit:
                                    covered += 1
                            elif t < 0.0:
                                if offset_x * offset_x + offset_y * offset_y <= half2:
                                    covered += 1
                            else:
                                end_x = page_x - x1
                                end_y = page_y - y1
                                if end_x * end_x + end_y * end_y <= half2:
                                    covered += 1
                else:
                    for page_y in page_y_samples:
                        for page_x in page_x_samples:
                            t = ((page_x - x0) * dx + (page_y - y0) * dy) * inv_seg_len2
                            if line_cap == 2 and (t < -extension_t or t > 1.0 + extension_t):
                                continue
                            closest_t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
                            qx = x0 + dx * closest_t
                            qy = y0 + dy * closest_t
                            dist_x = page_x - qx
                            dist_y = page_y - qy
                            if dist_x * dist_x + dist_y * dist_y <= half2:
                                covered += 1
                if covered:
                    alpha = max(1, min(255, round(rgba[3] * covered / sample_total)))
                    if normal_fast:
                        blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], alpha)
                    else:
                        blend_px(
                            row + px * 4,
                            (rgba[0], rgba[1], rgba[2], alpha),
                            blend_alpha_scale,
                            blend_resolved_mode,
                        )

    def fill_join(
        self: Any,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        line_join: int = 0,
        blend_mode: str | None = None,
    ) -> None:
        fill_circle = self.fill_circle
        fill_rect = self.fill_rect
        scale = self.scale
        radius = max(0.5 / scale, float(line_width) * 0.5)
        match line_join:
            case LineJoin.ROUND:
                fill_circle(px, py, radius, rgba, blend_mode)
            case _:
                fill_rect(
                    (px - radius, py - radius, px + radius, py + radius),
                    rgba,
                    blend_mode,
                )

    def fill_cap(
        self: Any,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        line_cap: int,
        blend_mode: str | None = None,
    ) -> None:
        fill_circle = self.fill_circle
        fill_rect = self.fill_rect
        scale = self.scale
        if line_cap == LineCap.BUTT:
            return
        radius = max(0.5 / scale, float(line_width) * 0.5)
        match line_cap:
            case LineCap.ROUND:
                fill_circle(px, py, radius, rgba, blend_mode)
            case _:
                fill_rect(
                    (px - radius, py - radius, px + radius, py + radius),
                    rgba,
                    blend_mode,
                )

    def stroke_path(
        self: Any,
        path: CapturedPath,
        line_width: float,
        rgba: tuple[int, int, int, int],
        dash_pattern: tuple[list[float], float] | None = None,
        blend_mode: str | None = None,
        line_cap: int = 0,
        line_join: int = 0,
    ) -> None:
        scale = self.scale
        if self.clip.regions:
            clip_box = self.clip.current_clip()
            path_box = self.clip.path_bbox(path)
            if clip_box is not None and path_box is not None:
                stroke_pad = max(0.5 / scale, float(line_width) * 0.5)
                stroke_box = (
                    path_box[0] - stroke_pad,
                    path_box[1] - stroke_pad,
                    path_box[2] + stroke_pad,
                    path_box[3] + stroke_pad,
                )
                if internal_intersect_box(stroke_box, clip_box) is None:
                    return
        for subpath in path.subpaths:
            points = subpath.points
            if len(points) < 2:
                continue
            if (
                len(points) == 2
                and not subpath.closed
                and (not dash_pattern or not dash_pattern[0])
            ):
                (x0, y0), (x1, y1) = points
                self.fill_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    line_width,
                    rgba,
                    None,
                    blend_mode,
                    0,
                )
                if line_cap != 0:
                    self.fill_cap(x0, y0, line_width, rgba, line_cap, blend_mode)
                    self.fill_cap(x1, y1, line_width, rgba, line_cap, blend_mode)
                continue
            dashed = bool(dash_pattern and dash_pattern[0])
            segment_dash = dash_pattern if dashed else None
            segment_cap = line_cap if dashed else 0
            for index in range(len(points) - 1):
                x0, y0 = points[index]
                x1, y1 = points[index + 1]
                self.fill_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    line_width,
                    rgba,
                    segment_dash,
                    blend_mode,
                    segment_cap,
                )
            if subpath.closed and points[0] != points[-1]:
                x0, y0 = points[-1]
                x1, y1 = points[0]
                self.fill_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    line_width,
                    rgba,
                    segment_dash,
                    blend_mode,
                    segment_cap,
                )
            if dashed:
                continue
            for x, y in points[1:-1]:
                self.fill_join(x, y, line_width, rgba, line_join, blend_mode)
            if subpath.closed:
                x, y = points[0]
                self.fill_join(x, y, line_width, rgba, line_join, blend_mode)
            elif line_cap != 0:
                self.fill_cap(
                    points[0][0],
                    points[0][1],
                    line_width,
                    rgba,
                    line_cap,
                    blend_mode,
                )
                self.fill_cap(
                    points[-1][0],
                    points[-1][1],
                    line_width,
                    rgba,
                    line_cap,
                    blend_mode,
                )
