# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful path painting operations for raster targets."""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy

from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    internal_blend_normal_alpha_array_numpy,
    internal_blend_normal_solid_array_numpy,
    internal_blend_solid_array_numpy,
    internal_color_rgba,
    internal_scale_rgba_alpha,
)
from core_pdf.impl.render.kernels import RASTER_COORDINATE_CACHE_MAX_ENTRIES
from core_pdf.impl.render.model import (
    LineCap,
    LineJoin,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.render.paths import (
    RASTER_CIRCLE_MIN_PIXEL_AREA,
    RASTER_KERNEL_MIN_PIXEL_AREA,
    RASTER_SAMPLE_OFFSETS,
    internal_fill_path_crossing_spans,
    internal_fill_path_sample_crossings,
    internal_fill_path_sample_crossings_numpy,
    internal_intersect_box,
    internal_signed_area_coverage,
    rasterize_unclipped_line_normal,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_number


class internal_PathTargetMixin:
    """Path, shape, and glyph painting operations for a raster target."""

    __slots__ = ()

    def fill_rect(
        self: Any,
        box: tuple[float, float, float, float] | None,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        # This is the single hottest method in the rasterizer — roughly 1.8M
        # calls over the corpus — so only the fast path's names are hoisted here.
        # The scanline loop below hoists the rest when it is actually reached.
        if box is None:
            return
        clip = self.clip
        buffer_stack = self.buffer_stack
        clip_path_stack = clip.clip_path_stack
        if blend_mode == "Normal" and rgba[3] == 255 and buffer_stack[-1][1] is None:
            blend_mode = None
        x0, y0, x1, y1 = box
        clip_box = self.current_clip() if clip_path_stack else None
        if clip_box is not None:
            clipped = internal_intersect_box((x0, y0, x1, y1), clip_box)
            if clipped is None:
                return
            x0, y0, x1, y1 = clipped
        pixel_box = self.page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
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
            if pdf_number(group_alpha):
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

    def fill_path_scanlines(
        self: Any,
        edge_segments: list[tuple[float, float, float, float, float, float]],
        pixel_box: tuple[int, int, int, int],
        rgba: tuple[int, int, int, int],
        blend_mode: str | None,
        fill_rule: str,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        blend_normal_pixel = self.blend_normal_pixel
        blend_normal_solid_span = self.blend_normal_solid_span
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        buffer_stack = self.buffer_stack
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_buffer = self.page_buffer
        page_pixels = self.page_pixels
        pixel_view = self.pixel_view
        pixels = self.pixels
        scale = self.scale
        width = self.width
        ix0, iy0, ix1, iy1 = pixel_box
        rectangular_clip = clip_paths_are_axis_aligned_rects()
        simple_opaque = (
            rgba[3] == 255
            and blend_mode is None
            and buffer_stack[-1][1] is None
            and rectangular_clip
        )
        normal_fast = can_blend_normal_fast(blend_mode)
        normal_target = pixel_view(pixels) if normal_fast and not simple_opaque else None
        # Same group-alpha hoist as `fill_rect`: invariant for this whole call,
        # so folded into `blended_rgba` once instead of per pixel in `blend_px`.
        blended_rgba = rgba
        blend_target = None
        if not normal_fast:
            group_alpha = buffer_stack[-1][1]
            if pdf_number(group_alpha):
                sr, sg, sb, sa = rgba
                sa = max(0, min(255, int(round(sa * float(group_alpha)))))
                blended_rgba = (sr, sg, sb, sa)
            if blended_rgba[3] > 0:
                blend_target = pixel_view(pixels)

        def span_pixels(start_x: float, end_x: float) -> tuple[int, int] | None:
            if end_x <= start_x:
                return None
            start = math.ceil((start_x - crop_x0) * scale - 0.5)
            end = math.ceil((end_x - crop_x0) * scale - 0.5)
            start = max(ix0, min(ix1, start))
            end = max(ix0, min(ix1, end))
            if end <= start:
                return None
            return start, end

        # Active-edge table: rows are visited with strictly decreasing page_y,
        # so instead of rescanning every edge on every row, each edge is
        # pushed onto a min-heap (by its lower y bound) once page_y drops
        # below its upper bound, and popped once page_y drops below its
        # lower bound. What remains on the heap for a given row is exactly
        # the edges a full per-row scan would have kept -- validated against
        # a brute-force reference over thousands of randomized edge sets,
        # including duplicate-`low` ties and edges shorter than one row step.
        edge_count = len(edge_segments)
        pending_order = sorted(range(edge_count), key=lambda i: -edge_segments[i][5])
        pending_index = 0
        active_heap: list[tuple[float, int]] = []
        for py in range(iy0, iy1):
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            page_y = crop_y1 - (py + 0.5) / scale
            while (
                pending_index < edge_count
                and edge_segments[pending_order[pending_index]][5] > page_y
            ):
                edge_index = pending_order[pending_index]
                heapq.heappush(active_heap, (edge_segments[edge_index][4], edge_index))
                pending_index += 1
            while active_heap and active_heap[0][0] > page_y:
                heapq.heappop(active_heap)
            crossings: list[tuple[float, int]] = []
            for low, edge_index in active_heap:
                ex0, ey0, ex1, ey1, edge_low, edge_high = edge_segments[edge_index]
                if not (edge_low <= page_y < edge_high):
                    continue
                t = (page_y - ey0) / (ey1 - ey0)
                x_intersection = ex0 + t * (ex1 - ex0)
                crossings.append((x_intersection, 1 if ey1 > ey0 else -1))
            if not crossings:
                continue
            row = py * width * 4
            # Same winding sweep as the kernel helper; the helper drops the
            # degenerate evenodd pairs that `span_pixels` would reject anyway.
            scan_spans = internal_fill_path_crossing_spans(crossings, fill_rule)
            for start_x, end_x in scan_spans:
                span = span_pixels(start_x, end_x)
                if span is None:
                    continue
                start, end = span
                for clip_start, clip_end in visible_spans:
                    visible_start = max(start, clip_start)
                    visible_end = min(end, clip_end)
                    if visible_end <= visible_start:
                        continue
                    if simple_opaque:
                        if pixels is page_buffer:
                            page_pixels[py, visible_start:visible_end] = rgba
                        else:
                            pixel_view(pixels)[py, visible_start:visible_end] = rgba
                        continue
                    if rectangular_clip and normal_fast:
                        blend_normal_solid_span(row, visible_start, visible_end, rgba)
                        continue
                    if (
                        normal_target is not None
                        and visible_end - visible_start >= RASTER_NUMPY_SPAN_MIN_PIXELS
                    ):
                        internal_blend_normal_solid_array_numpy(
                            normal_target[py, visible_start:visible_end],
                            rgba,
                        )
                        continue
                    if (
                        blend_target is not None
                        and visible_end - visible_start >= RASTER_NUMPY_SPAN_MIN_PIXELS
                    ):
                        internal_blend_solid_array_numpy(
                            blend_target[py, visible_start:visible_end],
                            blended_rgba,
                            blend_mode,
                        )
                        continue
                    for px in range(visible_start, visible_end):
                        if normal_fast:
                            blend_normal_pixel(row + px * 4, *rgba)
                        else:
                            blend_px(row + px * 4, rgba, blend_alpha_scale, blend_resolved_mode)

    def draw_glyph_bitmap(
        self: Any,
        box: tuple[float, float, float, float] | None,
        bitmap: Any,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        bitmap_width: Any = None,
        bitmap_height: Any = None,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
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
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
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
        raster_x_coordinate_cache = self.raster_x_coordinate_cache
        raster_y_coordinate_cache = self.raster_y_coordinate_cache
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
                x_coords = raster_x_coordinate_cache.get((ix0, ix1))
                if x_coords is None:
                    x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
                    if len(raster_x_coordinate_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                        raster_x_coordinate_cache[(ix0, ix1)] = x_coords
                y_coords = raster_y_coordinate_cache.get((iy0, iy1))
                if y_coords is None:
                    y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
                    if len(raster_y_coordinate_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                        raster_y_coordinate_cache[(iy0, iy1)] = y_coords
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

    def fast_fill_path(
        self: Any,
        edges: list[tuple[float, float, float, float]],
        bbox: tuple[float, float, float, float],
    ) -> bool:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        blend_normal_solid_span = self.blend_normal_solid_span
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_box_to_pixels = self.page_box_to_pixels
        scale = self.scale
        width = self.width
        """Fill opaque black polygons using one winding scan per raster row."""
        x0, y0, x1, y1 = bbox
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return True
        ix0, iy0, ix1, iy1 = pixel_box
        if ix1 - ix0 < 10 or iy1 - iy0 < 10:
            return False
        # Active-edge table, mirroring `fill_path_scanlines`: rows are visited
        # with strictly decreasing scan_y, so instead of rescanning every edge
        # on every row, each edge is pushed onto a min-heap (by its lower y
        # bound) once scan_y drops below its upper bound and popped once
        # scan_y drops below its lower bound. The in-loop bounds recheck keeps
        # the crossing set identical to the full per-row scan.
        edge_bounds = [
            (ex0, ey0, ex1, ey1, ey0 if ey0 < ey1 else ey1, ey1 if ey1 > ey0 else ey0)
            for ex0, ey0, ex1, ey1 in edges
        ]
        edge_count = len(edge_bounds)
        pending_order = sorted(range(edge_count), key=lambda i: -edge_bounds[i][5])
        pending_index = 0
        active_heap: list[tuple[float, int]] = []
        for py in range(iy0, iy1):
            scan_y = crop_y1 - (py + 0.5) / scale
            while (
                pending_index < edge_count and edge_bounds[pending_order[pending_index]][5] > scan_y
            ):
                edge_index = pending_order[pending_index]
                heapq.heappush(active_heap, (edge_bounds[edge_index][4], edge_index))
                pending_index += 1
            while active_heap and active_heap[0][0] > scan_y:
                heapq.heappop(active_heap)
            intersections: list[tuple[float, int]] = []
            for low, edge_index in active_heap:
                ex0, ey0, ex1, ey1, edge_low, edge_high = edge_bounds[edge_index]
                if not (edge_low <= scan_y < edge_high):
                    continue
                intersections.append(
                    (
                        ex0 + (scan_y - ey0) * (ex1 - ex0) / (ey1 - ey0),
                        1 if ey1 > ey0 else -1,
                    )
                )
            intersections.sort()
            winding = 0
            start_x = 0.0
            for end_x, delta in intersections:
                if winding:
                    start = max(ix0, math.ceil((start_x - crop_x0) * scale))
                    end = min(ix1, math.ceil((end_x - crop_x0) * scale))
                    blend_normal_solid_span(py * width * 4, start, end, (0, 0, 0, 255))
                if winding == 0:
                    start_x = end_x
                winding += delta
        return True

    def fill_path(
        self: Any,
        path: CapturedPath,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        fill_rule: str = "nonzero",
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip = self.clip
        axis_aligned_rect_box = clip.axis_aligned_rect_box
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_path_stack = clip.clip_path_stack
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = self.current_clip
        fast_fill_path = self.fast_fill_path
        fill_path_scanlines = self.fill_path_scanlines
        fill_rect = self.fill_rect
        page_box_to_pixels = self.page_box_to_pixels
        path_bbox = clip.path_bbox
        path_edges = clip.path_edges
        pixel_in_clip = self.pixel_in_clip
        pixel_view = self.pixel_view
        pixels = self.pixels
        scale = self.scale
        width = self.width
        rect = axis_aligned_rect_box(path)
        if rect is not None:
            fill_rect(rect, rgba, blend_mode)
            return
        edges = path_edges(path)
        if not edges:
            return
        bbox = path_bbox(path)
        if bbox is None:
            return
        fast_bbox: tuple[float, float, float, float] | None = bbox
        if clip_path_stack:
            if not clip_paths_are_axis_aligned_rects():
                fast_bbox = None
            else:
                clip_box = current_clip()
                if clip_box is not None:
                    fast_bbox = internal_intersect_box(bbox, clip_box)
        if (
            rgba == (0, 0, 0, 255)
            and blend_mode is None
            and fast_bbox is not None
            and fill_rule == "nonzero"
            and fast_fill_path(edges, fast_bbox)
        ):
            return
        x0, y0, x1, y1 = bbox
        clip_box = current_clip()
        if clip_box is not None:
            clipped = internal_intersect_box((x0, y0, x1, y1), clip_box)
            if clipped is None:
                return
            x0, y0, x1, y1 = clipped
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        pixel_area = (ix1 - ix0) * (iy1 - iy0)
        rectangular_clip = clip_paths_are_axis_aligned_rects()
        normal_fast = can_blend_normal_fast(blend_mode)
        if normal_fast and rectangular_clip and fill_rule == "nonzero" and pixel_area < 10_000:
            # Analytic coverage for the whole box in one pass, then one blend.
            # Cost follows the edges' extents rather than rows x edges, and the
            # result is exact rather than quantized to a 4x4 sample grid. This
            # runs first because it needs neither the y-extent columns nor the
            # sample-path array built below, and it takes ~99% of fills.
            source = numpy.asarray(edges, dtype=numpy.float64)
            sloped = source[:, 1] != source[:, 3]
            if not sloped.any():
                return
            source = source[sloped]
            device_edges = numpy.empty(source.shape, dtype=numpy.float64)
            device_edges[:, 0] = (source[:, 0] - crop_x0) * scale - ix0
            device_edges[:, 1] = (crop_y1 - source[:, 1]) * scale - iy0
            device_edges[:, 2] = (source[:, 2] - crop_x0) * scale - ix0
            device_edges[:, 3] = (crop_y1 - source[:, 3]) * scale - iy0
            coverage = internal_signed_area_coverage(device_edges, ix1 - ix0, iy1 - iy0)
            internal_blend_normal_alpha_array_numpy(
                pixel_view(pixels)[iy0:iy1, ix0:ix1],
                rgba,
                numpy.rint(coverage * rgba[3]).astype(numpy.uint8),
            )
            return
        edge_segments = [
            (
                ex0,
                ey0,
                ex1,
                ey1,
                ey0 if ey0 < ey1 else ey1,
                ey1 if ey1 > ey0 else ey0,
            )
            for ex0, ey0, ex1, ey1 in edges
            if ey0 != ey1
        ]
        if not edge_segments:
            return
        edge_segments_array = (
            numpy.asarray(edge_segments, dtype=numpy.float64) if len(edge_segments) >= 8 else None
        )
        if pixel_area >= 10_000:
            fill_path_scanlines(edge_segments, pixel_box, rgba, blend_mode, fill_rule)
            return
        samples = 4
        # Sample every scanline of the box up front. Called per pixel row this
        # handed the kernel four y values at a time, so the numpy work was pure
        # call overhead; one call per fill amortizes it over the whole box.
        # The y values are spelled exactly as the per-row form below to keep
        # each sample bit-identical.
        all_row_crossings = None
        if edge_segments_array is not None:
            row_count = iy1 - iy0
            sample_offsets = (numpy.arange(samples, dtype=numpy.float64) + 0.5) / samples
            page_ys = (
                crop_y1
                - (
                    numpy.repeat(numpy.arange(iy0, iy1, dtype=numpy.float64), samples)
                    + numpy.tile(sample_offsets, row_count)
                )
                / scale
            )
            all_row_crossings = internal_fill_path_sample_crossings_numpy(
                edge_segments_array, page_ys
            )
        for py in range(iy0, iy1):
            row = py * width * 4
            sample_spans = []
            if all_row_crossings is not None:
                base = (py - iy0) * samples
                for sy in range(samples):
                    sample_spans.append(
                        internal_fill_path_crossing_spans(all_row_crossings[base + sy], fill_rule)
                    )
            else:
                for sy in range(samples):
                    page_y = crop_y1 - (py + (sy + 0.5) / samples) / scale
                    crossings = internal_fill_path_sample_crossings(edge_segments, page_y)
                    sample_spans.append(internal_fill_path_crossing_spans(crossings, fill_rule))
            if normal_fast and rectangular_clip:
                # Accumulate into a difference array: each covered span is two
                # integer updates instead of a numpy slice-add, and the row is
                # summed once at the end. Coverage never exceeds samples**2, so
                # it still fits uint8, and integer addition is commutative --
                # reordering the sample loops cannot change the totals.
                deltas = [0] * (ix1 - ix0 + 1)
                covered_any = False
                for spans in sample_spans:
                    for start_x, end_x in spans:
                        span_start = (start_x - crop_x0) * scale
                        span_end = (end_x - crop_x0) * scale
                        for sx in range(samples):
                            sample_offset = (sx + 0.5) / samples
                            start = max(ix0, math.ceil(span_start - sample_offset))
                            end = min(ix1, math.ceil(span_end - sample_offset))
                            if end > start:
                                deltas[start - ix0] += 1
                                deltas[end - ix0] -= 1
                                covered_any = True
                if covered_any:
                    coverage = numpy.cumsum(numpy.asarray(deltas[:-1], dtype=numpy.int16)).astype(
                        numpy.uint8
                    )
                    target = pixel_view(pixels)[py, ix0:ix1]
                    internal_blend_normal_alpha_array_numpy(
                        target,
                        rgba,
                        numpy.rint(
                            coverage.astype(numpy.float32) * rgba[3] / (samples * samples)
                        ).astype(numpy.uint8),
                    )
                continue
            for px in range(ix0, ix1):
                covered = 0
                sample_x0 = crop_x0 + (px + 0.5 / samples) / scale
                sample_step = 1.0 / (samples * scale)
                for spans in sample_spans:
                    if not spans:
                        continue
                    for sx in range(samples):
                        page_x = sample_x0 + sx * sample_step
                        for start_x, end_x in spans:
                            if start_x <= page_x < end_x:
                                covered += 1
                                break
                if covered:
                    if not rectangular_clip and not pixel_in_clip(px, py):
                        continue
                    alpha = max(
                        1,
                        min(255, round(rgba[3] * covered / (samples * samples))),
                    )
                    if normal_fast:
                        blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], alpha)
                    else:
                        blend_px(
                            row + px * 4,
                            (rgba[0], rgba[1], rgba[2], alpha),
                            blend_alpha_scale,
                            blend_resolved_mode,
                        )

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
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip = self.clip
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        buffer_stack = self.buffer_stack
        clip_path_stack = clip.clip_path_stack
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = self.current_clip
        fill_circle = self.fill_circle
        fill_line = self.fill_line
        fill_rect = self.fill_rect
        page_box_to_pixels = self.page_box_to_pixels
        page_pixels = self.page_pixels
        pixel_in_clip = self.pixel_in_clip
        pixels = self.pixels
        raster_x_coordinate_cache = self.raster_x_coordinate_cache
        raster_x_sample_cache = self.raster_x_sample_cache
        raster_y_coordinate_cache = self.raster_y_coordinate_cache
        raster_y_sample_cache = self.raster_y_sample_cache
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
        clip_box = current_clip() if clip_path_stack else None
        if clip_box is not None:
            clipped = internal_intersect_box(box, clip_box)
            if clipped is None:
                return
            box = clipped
        pixel_box = page_box_to_pixels(*box)
        if pixel_box is None:
            return

        ix0, iy0, ix1, iy1 = pixel_box
        samples = 4
        sample_total = samples * samples
        half2 = half * half
        inv_seg_len2 = 1.0 / seg_len2
        extension_t = cap_extension / seg_len
        normal_fast = blend_mode is None and buffer_stack[-1][1] is None
        if (
            (not clip_path_stack or clip_paths_are_axis_aligned_rects())
            and normal_fast
            and (ix1 - ix0) * (iy1 - iy0) > RASTER_KERNEL_MIN_PIXEL_AREA
        ):
            x_coords = raster_x_coordinate_cache.get((ix0, ix1))
            if x_coords is None:
                x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
                if len(raster_x_coordinate_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                    raster_x_coordinate_cache[(ix0, ix1)] = x_coords
            y_coords = raster_y_coordinate_cache.get((iy0, iy1))
            if y_coords is None:
                y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
                if len(raster_y_coordinate_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                    raster_y_coordinate_cache[(iy0, iy1)] = y_coords
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
                target_pixels=page_pixels,
                x_coords=x_coords,
                y_coords=y_coords,
            )
            return
        for py in range(iy0, iy1):
            row = py * width * 4
            page_y_samples = raster_y_sample_cache.get(py)
            if page_y_samples is None:
                page_y_samples = tuple(
                    crop_y1 - (py + sample_offset) / scale
                    for sample_offset in RASTER_SAMPLE_OFFSETS
                )
                raster_y_sample_cache[py] = page_y_samples
            for px in range(ix0, ix1):
                if clip_path_stack and not pixel_in_clip(px, py):
                    continue
                page_x_samples = raster_x_sample_cache.get(px)
                if page_x_samples is None:
                    page_x_samples = tuple(
                        crop_x0 + (px + sample_offset) / scale
                        for sample_offset in RASTER_SAMPLE_OFFSETS
                    )
                    raster_x_sample_cache[px] = page_x_samples
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
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
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
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
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
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip_path_stack = self.clip_path_stack
        current_clip = self.current_clip
        fill_cap = self.fill_cap
        fill_join = self.fill_join
        fill_line = self.fill_line
        path_bbox = self.path_bbox
        scale = self.scale
        if clip_path_stack:
            clip_box = current_clip() if clip_path_stack else None
            path_box = path_bbox(path)
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
                fill_line(
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
                    fill_cap(x0, y0, line_width, rgba, line_cap, blend_mode)
                    fill_cap(x1, y1, line_width, rgba, line_cap, blend_mode)
                continue
            if dash_pattern and dash_pattern[0]:
                for index in range(len(points) - 1):
                    x0, y0 = points[index]
                    x1, y1 = points[index + 1]
                    fill_line(
                        x0,
                        y0,
                        x1,
                        y1,
                        line_width,
                        rgba,
                        dash_pattern,
                        blend_mode,
                        line_cap,
                    )
                if subpath.closed and points[0] != points[-1]:
                    x0, y0 = points[-1]
                    x1, y1 = points[0]
                    fill_line(
                        x0,
                        y0,
                        x1,
                        y1,
                        line_width,
                        rgba,
                        dash_pattern,
                        blend_mode,
                        line_cap,
                    )
                continue
            for index in range(len(points) - 1):
                x0, y0 = points[index]
                x1, y1 = points[index + 1]
                fill_line(
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
            if subpath.closed and points[0] != points[-1]:
                x0, y0 = points[-1]
                x1, y1 = points[0]
                fill_line(
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
            for x, y in points[1:-1]:
                fill_join(x, y, line_width, rgba, line_join, blend_mode)
            if subpath.closed:
                x, y = points[0]
                fill_join(x, y, line_width, rgba, line_join, blend_mode)
            elif line_cap != 0:
                fill_cap(
                    points[0][0],
                    points[0][1],
                    line_width,
                    rgba,
                    line_cap,
                    blend_mode,
                )
                fill_cap(
                    points[-1][0],
                    points[-1][1],
                    line_width,
                    rgba,
                    line_cap,
                    blend_mode,
                )

    def paint_typed_path(self: Any, item: PathPaintItem) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        color_cache = self.color_cache
        fill_path = self.fill_path
        stroke_path = self.stroke_path
        path = item.path
        if type(path) is not CapturedPath:
            return
        item_bbox = item.bbox
        if item_bbox is not None:
            # compose_page already computed ``path.bbox()`` for the item.
            self.clip.path_bbox_cache.setdefault(id(path), item_bbox)
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
        soft_mask_alpha = item.soft_mask_alpha
        paint_kind = item.paint_kind
        if paint_kind is not PathPaintKind.STROKE:
            color_key = (id(item.fill), id(item.fill_opacity))
            rgba = color_cache.get(color_key)
            if rgba is None:
                rgba = internal_color_rgba(item.fill, item.fill_opacity)
                if len(color_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                    color_cache[color_key] = rgba
            if pdf_number(soft_mask_alpha):
                rgba = internal_scale_rgba_alpha(rgba, soft_mask_alpha)
            fill_path(
                path,
                rgba,
                blend_mode,
                item.fill_rule or "nonzero",
            )
        if paint_kind is not PathPaintKind.FILL:
            color_key = (id(item.stroke_color), id(item.stroke_opacity))
            stroke_rgba = color_cache.get(color_key)
            if stroke_rgba is None:
                stroke_rgba = internal_color_rgba(item.stroke_color, item.stroke_opacity)
                if len(color_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                    color_cache[color_key] = stroke_rgba
            if pdf_number(soft_mask_alpha):
                stroke_rgba = internal_scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
            stroke_path(
                path,
                float(item.line_width or 1.0),
                stroke_rgba,
                item.dash_pattern,
                blend_mode,
                int(item.line_cap or 0),
                int(item.line_join or 0),
            )
