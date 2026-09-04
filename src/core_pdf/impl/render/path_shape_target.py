# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful primitive-shape painting operations for raster targets."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    internal_blend_normal_alpha_array_numpy,
    internal_blend_normal_solid_array_numpy,
    internal_blend_solid_array_numpy,
)
from core_pdf.impl.render.paths import (
    RASTER_CIRCLE_MIN_PIXEL_AREA,
    internal_intersect_box,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number, parse_int


class internal_PathShapeTargetMixin:
    """Rectangle, circle, and glyph-bitmap painting operations."""

    __slots__ = ()

    def fill_rect(
        self: Any,
        box: tuple[float, float, float, float] | None,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        if box is None:
            return
        buffer_stack = self.buffer_stack
        if blend_mode == "Normal" and rgba[3] == 255 and buffer_stack[-1][1] is None:
            blend_mode = None
        clipped_box = self.clipped_pixel_box(box)
        if clipped_box is None:
            return
        (x0, y0, x1, y1), (ix0, iy0, ix1, iy1) = clipped_box
        rectangular_clip = self.clip_paths_are_axis_aligned_rects()
        pixels = self.pixels
        # page_box_to_pixels expands outward (floor left/top, ceil right/bottom),
        # so filling ix0:ix1 solid paints whole pixels the rectangle only partly
        # covers. Every axis-aligned fill went through here unantialiased: on
        # IRS-2023-Form-1095-A the three 1.57px-wide "I" glyphs of "Part III",
        # 1.39px apart, each grew to three whole pixels and merged into one solid
        # white block. A rectangle that lands on pixel boundaries still takes the
        # memset path below; one that does not gets its exact coverage, which is
        # separable -- full in the interior, fractional in the edge row/column.
        scale = self.scale
        left = (x0 - self.crop_x0) * scale
        right = (x1 - self.crop_x0) * scale
        top = (self.crop_y1 - y1) * scale
        bottom = (self.crop_y1 - y0) * scale
        if (
            rectangular_clip
            and blend_mode is None
            and buffer_stack[-1][1] is None
            and not (
                left <= ix0 + 1e-9
                and right >= ix1 - 1e-9
                and top <= iy0 + 1e-9
                and bottom >= iy1 - 1e-9
            )
        ):
            columns = numpy.arange(ix0, ix1, dtype=numpy.float64)
            rows = numpy.arange(iy0, iy1, dtype=numpy.float64)
            x_coverage = numpy.clip(
                numpy.minimum(columns + 1.0, right) - numpy.maximum(columns, left), 0.0, 1.0
            )
            y_coverage = numpy.clip(
                numpy.minimum(rows + 1.0, bottom) - numpy.maximum(rows, top), 0.0, 1.0
            )
            internal_blend_normal_alpha_array_numpy(
                self.pixel_view(pixels)[iy0:iy1, ix0:ix1],
                rgba,
                numpy.rint(numpy.outer(y_coverage, x_coverage) * rgba[3]).astype(numpy.uint8),
            )
            return
        if (
            rgba[3] == 255
            and blend_mode is None
            and buffer_stack[-1][1] is None
            and rectangular_clip
        ):
            span = ix1 - ix0
            if span <= 0:
                return
            if pixels is self.page_buffer:
                self.page_pixels[iy0:iy1, ix0:ix1] = rgba
                return
            target_pixels = self.pixel_view(pixels)
            internal_blend_normal_solid_array_numpy(
                target_pixels[iy0:iy1, ix0:ix1],
                rgba,
            )
            return
        pixel_view = self.pixel_view
        normal_fast = blend_mode is None and buffer_stack[-1][1] is None
        normal_target = pixel_view(pixels) if normal_fast else None
        if rectangular_clip and normal_fast and ix1 > ix0 and iy1 > iy0:
            target_pixels = pixel_view(pixels)
            internal_blend_normal_solid_array_numpy(
                target_pixels[iy0:iy1, ix0:ix1],
                rgba,
            )
            return
        # Only the clipped / blended scanline path below needs these. A
        # transparency-group alpha is invariant for this whole call (it comes
        # from `buffer_stack`, which `fill_rect` never pushes/pops), so it is
        # folded into `blended_rgba` once here instead of on every pixel the
        # way `blend_px` does it -- mirrors the group-alpha scale it would
        # otherwise redo per pixel, and lets wide spans go through one NumPy
        # blend instead of a Python loop.
        # `rectangular_clip and normal_fast` is unreachable below: the
        # `page_box_to_pixels` result already guarantees ix1 > ix0 and
        # iy1 > iy0, so that combination always takes the whole-box return
        # above instead of reaching this scanline loop.
        width = self.width
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        blend_normal_pixel = self.blend_normal_pixel
        clip_row_visible_spans = self.clip_row_visible_spans
        blended_rgba = rgba
        if normal_target is None:
            group_alpha = buffer_stack[-1][1]
            if is_pdf_number(group_alpha):
                sr, sg, sb, sa = rgba
                sa = max(0, min(255, int(round(sa * float(group_alpha)))))
                blended_rgba = (sr, sg, sb, sa)
            if blended_rgba[3] <= 0:
                return
            blend_target = pixel_view(pixels)
            if rectangular_clip:
                # The whole ix0:ix1/iy0:iy1 box is visible with no gaps (same
                # invariant the opaque/Normal fast paths above rely on), so
                # one array-wide blend replaces a numpy call per row.
                internal_blend_solid_array_numpy(
                    blend_target[iy0:iy1, ix0:ix1], blended_rgba, blend_mode
                )
                return
        for y in range(iy0, iy1):
            row = y * width * 4
            visible_spans = clip_row_visible_spans(y)
            if not visible_spans:
                continue
            for start, end in visible_spans:
                start = max(ix0, start)
                end = min(ix1, end)
                if end <= start:
                    continue
                if normal_target is not None:
                    if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
                        internal_blend_normal_solid_array_numpy(normal_target[y, start:end], rgba)
                    else:
                        for x in range(start, end):
                            blend_normal_pixel(row + x * 4, *rgba)
                elif end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
                    internal_blend_solid_array_numpy(
                        blend_target[y, start:end], blended_rgba, blend_mode
                    )
                else:
                    for x in range(start, end):
                        blend_px(row + x * 4, rgba, blend_alpha_scale, blend_resolved_mode)

    def draw_glyph_bitmap(
        self: Any,
        box: tuple[float, float, float, float] | None,
        bitmap: Any,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        bitmap_width: Any = None,
        bitmap_height: Any = None,
    ) -> None:
        clip = self.clip
        buffer_stack = self.buffer_stack
        clip_path_stack = clip.clip_path_stack
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        fill_rect = self.fill_rect
        page_box_to_pixels = self.page_box_to_pixels
        page_buffer = self.page_buffer
        page_pixels = self.page_pixels
        pixel_view = self.pixel_view
        pixels = self.pixels
        scale = self.scale
        bitmap_type = type(bitmap)
        if box is None or (bitmap_type is not list and bitmap_type is not tuple) or not bitmap:
            return
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        rows = [int(row) for row in bitmap if type(row) is int]
        if not rows:
            return
        bitmap_h = parse_int(bitmap_height, 0) or len(rows)
        bitmap_w = parse_int(bitmap_width, 0) or max((row.bit_length() for row in rows), default=0)
        if bitmap_w <= 0 or bitmap_h <= 0:
            return
        cell_w = (x1 - x0) / bitmap_w
        cell_h = (y1 - y0) / bitmap_h
        if cell_w <= 0 or cell_h <= 0:
            return
        opaque_glyph = (
            rgba[3] == 255
            and (blend_mode is None or blend_mode == "Normal")
            and buffer_stack[-1][1] is None
        )
        if opaque_glyph and not clip_path_stack and bitmap_w <= 64:
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            pixel_width = (x1 - x0) * scale
            pixel_height = (y1 - y0) * scale
            origin_x = (x0 - crop_x0) * scale
            origin_y = (crop_y1 - y1) * scale
            cell_pixel_width = pixel_width / bitmap_w
            cell_pixel_height = pixel_height / bitmap_h
            aligned = False
            if pixel_box is not None:
                aligned = (
                    abs(origin_x - round(origin_x)) <= 1e-9
                    and abs(origin_y - round(origin_y)) <= 1e-9
                    and abs(cell_pixel_width - round(cell_pixel_width)) <= 1e-9
                    and abs(cell_pixel_height - round(cell_pixel_height)) <= 1e-9
                    and cell_pixel_width >= 1.0
                    and cell_pixel_height >= 1.0
                    and pixel_box[2] - pixel_box[0] == round(pixel_width)
                    and pixel_box[3] - pixel_box[1] == round(pixel_height)
                )
            if aligned and pixel_box is not None:
                ix0, iy0, ix1, iy1 = pixel_box
                cell_pixel_width = int(round(cell_pixel_width))
                cell_pixel_height = int(round(cell_pixel_height))
                row_values = numpy.asarray(
                    rows[:bitmap_h] + [0] * max(0, bitmap_h - len(rows)),
                    dtype=numpy.uint64,
                )
                columns = numpy.arange(bitmap_w, dtype=numpy.uint64)
                bits = ((row_values[:, None] >> columns[None, :]) & 1).astype(bool)
                expanded = numpy.repeat(
                    numpy.repeat(bits, cell_pixel_height, axis=0),
                    cell_pixel_width,
                    axis=1,
                )
                target_pixels = page_pixels if pixels is page_buffer else pixel_view(pixels)
                target_region = target_pixels[iy0:iy1, ix0:ix1]
                target_region[expanded] = rgba
                return
        for row_index, row in enumerate(rows):
            cell_y1 = y1 - row_index * cell_h
            cell_y0 = y1 - (row_index + 1) * cell_h
            if opaque_glyph:
                remaining = row
                while remaining:
                    run_start = (remaining & -remaining).bit_length() - 1
                    if run_start >= bitmap_w:
                        break
                    shifted = remaining >> run_start
                    run_length = (~shifted & (shifted + 1)).bit_length() - 1
                    run_end = min(bitmap_w, run_start + run_length)
                    fill_rect(
                        (
                            x0 + run_start * cell_w,
                            cell_y0,
                            x0 + run_end * cell_w,
                            cell_y1,
                        ),
                        rgba,
                        blend_mode,
                    )
                    remaining &= ~(((1 << run_length) - 1) << run_start)
                continue
            for col_index in range(bitmap_w):
                if not (row & (1 << col_index)):
                    continue
                cell_x0 = x0 + col_index * cell_w
                cell_x1 = x0 + (col_index + 1) * cell_w
                fill_rect((cell_x0, cell_y0, cell_x1, cell_y1), rgba, blend_mode)

    def fill_circle(
        self: Any,
        cx: float,
        cy: float,
        radius: float,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        clip = self.clip
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_path_stack = clip.clip_path_stack
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = self.current_clip
        page_box_to_pixels = self.page_box_to_pixels
        page_pixels = self.page_pixels
        pixels = self.pixels
        scale = self.scale
        width = self.width
        circle_box = (cx - radius, cy - radius, cx + radius, cy + radius)
        clip_box = current_clip() if clip_path_stack else None
        if clip_box is not None:
            clipped_circle_box = internal_intersect_box(circle_box, clip_box)
            if clipped_circle_box is None:
                return
            circle_box = clipped_circle_box
        pixel_box = page_box_to_pixels(*circle_box)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        radius2 = radius * radius
        normal_fast = can_blend_normal_fast(blend_mode)
        rectangular_clip = not clip_path_stack or clip_paths_are_axis_aligned_rects()
        if normal_fast and rgba[3] >= 255 and rectangular_clip:
            if (ix1 - ix0) * (iy1 - iy0) > RASTER_CIRCLE_MIN_PIXEL_AREA:
                x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
                y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
                circle_page_xs = crop_x0 + (x_coords + 0.5) / scale
                circle_page_ys = crop_y1 - (y_coords + 0.5) / scale
                inside = (circle_page_xs[None, :] - cx) ** 2 + (
                    circle_page_ys[:, None] - cy
                ) ** 2 <= radius2
                page_pixels[iy0:iy1, ix0:ix1][inside] = rgba
                return
            red, green, blue, internal_alpha = rgba
            for py in range(iy0, iy1):
                page_y = crop_y1 - (py + 0.5) / scale
                dy = page_y - cy
                row = py * width * 4
                for px in range(ix0, ix1):
                    page_x = crop_x0 + (px + 0.5) / scale
                    dx = page_x - cx
                    if dx * dx + dy * dy > radius2:
                        continue
                    index = row + px * 4
                    pixels[index] = red
                    pixels[index + 1] = green
                    pixels[index + 2] = blue
                    pixels[index + 3] = 255
            return
        for py in range(iy0, iy1):
            page_y = crop_y1 - (py + 0.5) / scale
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            for clip_start, clip_end in visible_spans:
                start = max(ix0, clip_start)
                end = min(ix1, clip_end)
                if end <= start:
                    continue
                for px in range(start, end):
                    page_x = crop_x0 + (px + 0.5) / scale
                    dx = page_x - cx
                    dy = page_y - cy
                    if dx * dx + dy * dy > radius2:
                        continue
                    if normal_fast:
                        blend_normal_pixel(row + px * 4, *rgba)
                    else:
                        blend_px(row + px * 4, rgba, blend_alpha_scale, blend_resolved_mode)
