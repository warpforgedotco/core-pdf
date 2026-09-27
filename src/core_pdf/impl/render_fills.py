# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy

from core_pdf.impl.array_views import UInt8Array
from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.pdf_names import lenient_int
from core_pdf.impl.render_blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    blend_normal_solid_array_numpy,
    blend_solid_array_numpy,
)
from core_pdf.impl.render_groups import RasterGroups
from core_pdf.impl.render_paths import (
    RASTER_CIRCLE_MIN_PIXEL_AREA,
    fill_path_crossing_spans,
    intersect_box,
)
from core_pdf_cythonized import (
    blend_coverage_counts,
    blend_normal_alpha_array_numpy,
    fill_glyph_coverage,
    fill_rect_coverage,
    supersampled_coverage_plane,
)
from core_pdf_spec.s_11_transparency.blend import BlendMode, blend_component

BLEND_COLOR_DODGE = 3


BLEND_COLOR_BURN = 4


BLEND_MODE_CODES: dict[str | None, int] = {
    "multiply": 1,
    "screen": 2,
    "colordodge": BLEND_COLOR_DODGE,
    "colorburn": BLEND_COLOR_BURN,
}


def edge_tuples(
    edge_array: numpy.ndarray[Any, Any] | None,
) -> list[tuple[float, float, float, float]]:
    if edge_array is None:
        return []
    return [(x0, y0, x1, y1) for x0, y0, x1, y1 in edge_array.tolist()]


class RasterFills(RasterGroups):
    __slots__ = ()

    def resolved_blend(self, blend_mode: str | None) -> str | None:
        return blend_mode.lower() if isinstance(blend_mode, str) else None

    def blend_px(
        self,
        idx: int,
        rgba: tuple[int, int, int, int],
        mode: str | None,
        *,
        shape: int = 255,
    ) -> None:
        pixels = self.pixels
        sr, sg, sb, sa = rgba
        alpha_plane = self.group_source_alpha
        if self.group_source_shape is not None:
            row, column = divmod(idx // 4, self.width)
            self.record_source_shape(row, column, shape)
        elif alpha_plane is None and self.paint_window is not None:
            row, column = divmod(idx // 4, self.width)
            self.extend_paint_window(row, column)
        if sa <= 0:
            return
        if alpha_plane is not None:
            row, column = divmod(idx // 4, self.width)
            self.record_source_alpha(row, column, sa)
        if sa >= 255 and mode is None:
            pixels[idx] = sr
            pixels[idx + 1] = sg
            pixels[idx + 2] = sb
            pixels[idx + 3] = 255
            return
        dr = pixels[idx]
        dg = pixels[idx + 1]
        db = pixels[idx + 2]
        da = pixels[idx + 3]
        src_a = sa / 255.0
        dst_a = da / 255.0
        src_r = sr / 255.0
        src_g = sg / 255.0
        src_b = sb / 255.0
        dst_r = dr / 255.0
        dst_g = dg / 255.0
        dst_b = db / 255.0
        if mode == "multiply":
            src_r = src_r * (1.0 - dst_a) + dst_a * (src_r * dst_r)
            src_g = src_g * (1.0 - dst_a) + dst_a * (src_g * dst_g)
            src_b = src_b * (1.0 - dst_a) + dst_a * (src_b * dst_b)
        elif mode == "screen":
            src_r = src_r * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_r) * (1.0 - dst_r))
            src_g = src_g * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_g) * (1.0 - dst_g))
            src_b = src_b * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_b) * (1.0 - dst_b))
        elif mode in {"colordodge", "colorburn"}:
            component_mode: BlendMode = "ColorDodge" if mode == "colordodge" else "ColorBurn"
            src_r = src_r * (1.0 - dst_a) + dst_a * blend_component(
                dst_r, src_r, component_mode, context=self.semantic_context
            )
            src_g = src_g * (1.0 - dst_a) + dst_a * blend_component(
                dst_g, src_g, component_mode, context=self.semantic_context
            )
            src_b = src_b * (1.0 - dst_a) + dst_a * blend_component(
                dst_b, src_b, component_mode, context=self.semantic_context
            )
        out_a = src_a + dst_a * (1.0 - src_a)
        out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
        out_a_i = int(round(out_a * 255.0))
        pixels[idx] = max(0, min(255, out_r))
        pixels[idx + 1] = max(0, min(255, out_g))
        pixels[idx + 2] = max(0, min(255, out_b))
        pixels[idx + 3] = max(0, min(255, out_a_i))

    def blend_normal_solid_span(
        self, row: int, start: int, end: int, rgba: tuple[int, int, int, int]
    ) -> None:
        sr, sg, sb, sa = rgba
        if end <= start:
            return
        self.record_source_shape(row // (self.width * 4), slice(start, end), 255)
        if sa <= 0:
            return
        self.record_source_alpha(row // (self.width * 4), slice(start, end), sa)
        pixels = self.pixels
        width = self.width
        if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
            target = self.pixel_array
            blend_normal_solid_array_numpy(target[row // (width * 4), start:end], rgba)
            return
        start_offset = row + start * 4
        stop_offset = row + end * 4
        if sa >= 255:
            self.pixel_array[row // (width * 4), start:end] = (sr, sg, sb, 255)
            return
        src_a = sa / 255.0
        one_minus_src_a = 1.0 - src_a
        for idx in range(start_offset, stop_offset, 4):
            dr = pixels[idx]
            dg = pixels[idx + 1]
            db = pixels[idx + 2]
            da = pixels[idx + 3]
            dst_a = da / 255.0
            out_a = src_a + dst_a * one_minus_src_a
            out_r = int(round((sr * src_a + dr * dst_a * one_minus_src_a) / out_a))
            out_g = int(round((sg * src_a + dg * dst_a * one_minus_src_a) / out_a))
            out_b = int(round((sb * src_a + db * dst_a * one_minus_src_a) / out_a))
            out_a_i = int(round(out_a * 255.0))
            pixels[idx] = max(0, min(255, out_r))
            pixels[idx + 1] = max(0, min(255, out_g))
            pixels[idx + 2] = max(0, min(255, out_b))
            pixels[idx + 3] = max(0, min(255, out_a_i))

    def clip_pixel_mask(self, ix0: int, iy0: int, ix1: int, iy1: int) -> bytearray:
        box_width = ix1 - ix0
        allowed = bytearray(box_width * (iy1 - iy0))
        ones = b"\x01" * box_width
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        for py in range(iy0, iy1):
            row_start = (py - iy0) * box_width - ix0
            for span_start, span_end in clip_row_visible_spans(py):
                start = max(ix0, span_start)
                end = min(ix1, span_end)
                if end > start:
                    allowed[row_start + start : row_start + end] = ones[: end - start]
        return allowed

    def fill_rect(
        self,
        box: tuple[float, float, float, float] | None,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        if box is None:
            return
        if blend_mode == "Normal" and rgba[3] == 255:
            blend_mode = None
        clipped_box = self.clip.clipped_pixel_box(box)
        if clipped_box is None:
            return
        (x0, y0, x1, y1), (ix0, iy0, ix1, iy1) = clipped_box
        rectangular_clip = self.clip.clip_paths_are_axis_aligned_rects()
        pixels = self.pixels
        scale = self.scale
        left = (x0 - self.crop_x0) * scale
        right = (x1 - self.crop_x0) * scale
        top = (self.crop_y1 - y1) * scale
        bottom = (self.crop_y1 - y0) * scale
        if (
            rectangular_clip
            and blend_mode is None
            and not (
                left <= ix0 + 1e-9
                and right >= ix1 - 1e-9
                and top <= iy0 + 1e-9
                and bottom >= iy1 - 1e-9
            )
        ):
            rows = slice(iy0, iy1)
            columns = slice(ix0, ix1)
            source_alpha = self.group_source_alpha
            source_shape = self.group_source_shape
            fill_rect_coverage(
                ix0,
                ix1,
                iy0,
                iy1,
                left,
                right,
                top,
                bottom,
                rgba,
                self.pixel_array[rows, columns],
                source_alpha[rows, columns] if source_alpha is not None else None,
                source_shape[rows, columns] if source_shape is not None else None,
                self.shape_alpha,
            )
            self.extend_paint_window(rows, columns)
            return
        if rgba[3] == 255 and blend_mode is None and rectangular_clip:
            span = ix1 - ix0
            if span <= 0:
                return
            if pixels is self.page_buffer:
                self.page_pixels[iy0:iy1, ix0:ix1] = rgba
                self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
                return
            target_pixels = self.pixel_array
            blend_normal_solid_array_numpy(
                target_pixels[iy0:iy1, ix0:ix1],
                rgba,
            )
            self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
            return
        normal_fast = blend_mode is None
        normal_target = self.pixel_array if normal_fast else None
        if rectangular_clip and normal_fast and ix1 > ix0 and iy1 > iy0:
            assert normal_target is not None
            blend_normal_solid_array_numpy(
                normal_target[iy0:iy1, ix0:ix1],
                rgba,
            )
            self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
            return
        width = self.width
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        if normal_target is None:
            if rgba[3] <= 0 and self.group_source_shape is None:
                return
            blend_target = self.pixel_array
            if rectangular_clip:
                blend_solid_array_numpy(
                    blend_target[iy0:iy1, ix0:ix1],
                    rgba,
                    blend_mode,
                    semantic_context=self.semantic_context,
                )
                self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
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
                        blend_normal_solid_array_numpy(normal_target[y, start:end], rgba)
                        self.record_source_coverage(y, slice(start, end), rgba[3])
                    else:
                        for x in range(start, end):
                            blend_px(row + x * 4, rgba, None)
                elif end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
                    blend_solid_array_numpy(
                        blend_target[y, start:end],
                        rgba,
                        blend_mode,
                        semantic_context=self.semantic_context,
                    )
                    self.record_source_coverage(y, slice(start, end), rgba[3])
                else:
                    for x in range(start, end):
                        blend_px(row + x * 4, rgba, blend_resolved_mode)

    def draw_glyph_bitmap(
        self,
        box: tuple[float, float, float, float] | None,
        bitmap: Any,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        bitmap_width: Any = None,
        bitmap_height: Any = None,
    ) -> None:
        clip = self.clip
        clip_regions = clip.regions
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        fill_rect = self.fill_rect
        page_box_to_pixels = clip.page_box_to_pixels
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
        bitmap_h = lenient_int(bitmap_height, 0) or len(rows)
        bitmap_w = lenient_int(bitmap_width, 0) or max(
            (row.bit_length() for row in rows), default=0
        )
        if bitmap_w <= 0 or bitmap_h <= 0:
            return
        cell_w = (x1 - x0) / bitmap_w
        cell_h = (y1 - y0) / bitmap_h
        if cell_w <= 0 or cell_h <= 0:
            return
        opaque_glyph = rgba[3] == 255 and (blend_mode is None or blend_mode == "Normal")
        if opaque_glyph and not clip_regions and bitmap_w <= 64:
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
                target_pixels = self.pixel_array
                target_region = target_pixels[iy0:iy1, ix0:ix1]
                target_region[expanded] = rgba
                self.record_source_coverage(
                    slice(iy0, iy1), slice(ix0, ix1), rgba[3], visible=expanded
                )
                return
        for row_index, row in enumerate(rows[:bitmap_h]):
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
        self,
        cx: float,
        cy: float,
        radius: float,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        clip = self.clip
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        page_box_to_pixels = clip.page_box_to_pixels
        pixels = self.pixels
        scale = self.scale
        width = self.width
        circle_box = (cx - radius, cy - radius, cx + radius, cy + radius)
        clip_box = current_clip() if clip_regions else None
        if clip_box is not None:
            clipped_circle_box = intersect_box(circle_box, clip_box)
            if clipped_circle_box is None:
                return
            circle_box = clipped_circle_box
        pixel_box = page_box_to_pixels(*circle_box)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        radius2 = radius * radius
        normal_fast = blend_mode is None
        rectangular_clip = not clip_regions or clip_paths_are_axis_aligned_rects()
        if normal_fast and rgba[3] >= 255 and rectangular_clip:
            if (ix1 - ix0) * (iy1 - iy0) > RASTER_CIRCLE_MIN_PIXEL_AREA:
                x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
                y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
                circle_page_xs = crop_x0 + (x_coords + 0.5) / scale
                circle_page_ys = crop_y1 - (y_coords + 0.5) / scale
                inside = (circle_page_xs[None, :] - cx) ** 2 + (
                    circle_page_ys[:, None] - cy
                ) ** 2 <= radius2
                self.pixel_array[iy0:iy1, ix0:ix1][inside] = rgba
                self.record_source_coverage(
                    slice(iy0, iy1), slice(ix0, ix1), rgba[3], visible=inside
                )
                return
            red, green, blue, alpha = rgba
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
                    self.record_source_coverage(py, px, 255)
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
                    blend_px(row + px * 4, rgba, blend_resolved_mode)

    def fill_path_scanlines(
        self,
        edge_segments: list[tuple[float, float, float, float, float, float]],
        pixel_box: tuple[int, int, int, int],
        rgba: tuple[int, int, int, int],
        blend_mode: str | None,
        fill_rule: str,
    ) -> None:
        blend_normal_solid_span = self.blend_normal_solid_span
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_paths_are_axis_aligned_rects = self.clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_buffer = self.page_buffer
        page_pixels = self.page_pixels
        pixels = self.pixels
        scale = self.scale
        width = self.width
        ix0, iy0, ix1, iy1 = pixel_box
        rectangular_clip = clip_paths_are_axis_aligned_rects()
        simple_opaque = rgba[3] == 255 and blend_mode is None and rectangular_clip
        normal_fast = blend_mode is None
        normal_target = self.pixel_array if normal_fast and not simple_opaque else None
        blend_target = self.pixel_array if not normal_fast and rgba[3] > 0 else None

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
                heapq.heappush(active_heap, (-edge_segments[edge_index][4], edge_index))
                pending_index += 1
            while active_heap and -active_heap[0][0] > page_y:
                heapq.heappop(active_heap)
            crossings: list[tuple[float, int]] = []
            for _low, edge_index in active_heap:
                ex0, ey0, ex1, ey1, edge_low, edge_high = edge_segments[edge_index]
                if not (edge_low <= page_y < edge_high):
                    continue
                t = (page_y - ey0) / (ey1 - ey0)
                x_intersection = ex0 + t * (ex1 - ex0)
                crossings.append((x_intersection, 1 if ey1 > ey0 else -1))
            if not crossings:
                continue
            row = py * width * 4
            scan_spans = fill_path_crossing_spans(crossings, fill_rule)
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
                            self.pixel_array[py, visible_start:visible_end] = rgba
                        self.record_source_coverage(py, slice(visible_start, visible_end), rgba[3])
                        continue
                    if rectangular_clip and normal_fast:
                        blend_normal_solid_span(row, visible_start, visible_end, rgba)
                        continue
                    if (
                        normal_target is not None
                        and visible_end - visible_start >= RASTER_NUMPY_SPAN_MIN_PIXELS
                    ):
                        blend_normal_solid_array_numpy(
                            normal_target[py, visible_start:visible_end],
                            rgba,
                        )
                        self.record_source_coverage(py, slice(visible_start, visible_end), rgba[3])
                        continue
                    if (
                        blend_target is not None
                        and visible_end - visible_start >= RASTER_NUMPY_SPAN_MIN_PIXELS
                    ):
                        blend_solid_array_numpy(
                            blend_target[py, visible_start:visible_end],
                            rgba,
                            blend_mode,
                            semantic_context=self.semantic_context,
                        )
                        self.record_source_coverage(py, slice(visible_start, visible_end), rgba[3])
                        continue
                    for px in range(visible_start, visible_end):
                        blend_px(row + px * 4, rgba, blend_resolved_mode)

    def fast_fill_path(
        self,
        edges: list[tuple[float, float, float, float]],
        bbox: tuple[float, float, float, float],
    ) -> bool:
        blend_normal_solid_span = self.blend_normal_solid_span
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_box_to_pixels = self.clip.page_box_to_pixels
        scale = self.scale
        width = self.width
        x0, y0, x1, y1 = bbox
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return True
        ix0, iy0, ix1, iy1 = pixel_box
        if ix1 - ix0 < 10 or iy1 - iy0 < 10:
            return False
        edge_bounds = [
            (ex0, ey0, ex1, ey1, min(ey1, ey0), max(ey0, ey1)) for ex0, ey0, ex1, ey1 in edges
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
            for _low, edge_index in active_heap:
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
                    start = max(ix0, math.ceil((start_x - crop_x0) * scale - 0.5))
                    end = min(ix1, math.ceil((end_x - crop_x0) * scale - 0.5))
                    blend_normal_solid_span(py * width * 4, start, end, (0, 0, 0, 255))
                if winding == 0:
                    start_x = end_x
                winding += delta
        return True

    def fill_path(
        self,
        path: CapturedPath,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        fill_rule: str = "nonzero",
        *,
        bbox: tuple[float, float, float, float] | None = None,
        edge_array: numpy.ndarray[Any, Any] | None = None,
    ) -> None:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        fast_fill_path = self.fast_fill_path
        fill_path_scanlines = self.fill_path_scanlines
        fill_rect = self.fill_rect
        scale = self.scale
        rect = path.axis_aligned_rect()
        if rect is not None:
            fill_rect(rect, rgba, blend_mode)
            return
        edges: list[tuple[float, float, float, float]] | None
        if edge_array is None:
            edge_array = path.fill_edge_array()
        if edge_array is None:
            edges = path.fill_edges()
            if not edges:
                return
        else:
            edges = None
            if len(edge_array) == 0:
                return
        if bbox is None:
            bbox = clip.path_bbox(path)
        if bbox is None:
            return
        fast_bbox: tuple[float, float, float, float] | None = bbox
        if clip_regions:
            if not clip_paths_are_axis_aligned_rects():
                fast_bbox = None
            else:
                clip_box = current_clip()
                if clip_box is not None:
                    fast_bbox = intersect_box(bbox, clip_box)
        if (
            rgba == (0, 0, 0, 255)
            and blend_mode is None
            and self.group_source_shape is None
            and fast_bbox is not None
            and fill_rule == "nonzero"
        ):
            if edges is None:
                edges = edge_tuples(edge_array)
            if fast_fill_path(edges, fast_bbox):
                return
        clipped_box = clipped_pixel_box(bbox)
        if clipped_box is None:
            return
        pixel_box = clipped_box[1]
        ix0, iy0, ix1, iy1 = pixel_box
        pixel_area = (ix1 - ix0) * (iy1 - iy0)
        rectangular_clip = clip_paths_are_axis_aligned_rects()
        normal_fast = blend_mode is None
        if pixel_area < 10_000:
            source = (
                edge_array if edge_array is not None else numpy.asarray(edges, dtype=numpy.float64)
            )
            if normal_fast and rectangular_clip and fill_rule == "nonzero":
                rows = slice(iy0, iy1)
                columns = slice(ix0, ix1)
                source_alpha = self.group_source_alpha
                source_shape = self.group_source_shape
                drawn = fill_glyph_coverage(
                    source,
                    crop_x0,
                    crop_y1,
                    scale,
                    ix0,
                    iy0,
                    ix1 - ix0,
                    iy1 - iy0,
                    rgba,
                    self.pixel_array[rows, columns],
                    source_alpha[rows, columns] if source_alpha is not None else None,
                    source_shape[rows, columns] if source_shape is not None else None,
                    self.shape_alpha,
                )
                if drawn is not None:
                    self.extend_paint_window(rows, columns)
                return
            sampled = supersampled_coverage_plane(
                source, crop_x0, crop_y1, scale, ix0, iy0, ix1, iy1, fill_rule == "evenodd"
            )
            if sampled is None:
                return
            counts, first_row = sampled
            if normal_fast and rectangular_clip:
                rows = slice(iy0 + first_row, iy0 + first_row + len(counts))
                columns = slice(ix0, ix1)
                coverage = counts.astype(numpy.float32)
                alpha_plane = numpy.rint(coverage * rgba[3] / 16).astype(numpy.uint8)
                blend_normal_alpha_array_numpy(self.pixel_array[rows, columns], rgba, alpha_plane)
                self.record_source_alpha(rows, columns, alpha_plane)
                if self.group_source_shape is not None:
                    self.record_source_shape(
                        rows, columns, numpy.rint(coverage * 255 / 16).astype(numpy.uint8)
                    )
                return
            top = iy0 + first_row
            self.blend_counts(
                counts,
                ix0,
                top,
                None
                if rectangular_clip
                else self.clip_pixel_mask(ix0, top, ix1, top + len(counts)),
                rgba,
                blend_mode,
            )
            return
        if edges is None:
            edges = edge_tuples(edge_array)
        edge_segments = [
            (
                ex0,
                ey0,
                ex1,
                ey1,
                min(ey1, ey0),
                max(ey0, ey1),
            )
            for ex0, ey0, ex1, ey1 in edges
            if ey0 != ey1
        ]
        if not edge_segments:
            return
        fill_path_scanlines(edge_segments, pixel_box, rgba, blend_mode, fill_rule)

    def blend_rules(self, mode: int) -> tuple[bool, Exception | None]:
        if mode not in (BLEND_COLOR_DODGE, BLEND_COLOR_BURN):
            return True, None
        try:
            revised = blend_component(0.0, 1.0, "ColorDodge", context=self.semantic_context)
        except Exception as error:
            return True, error
        return revised == 0.0, None

    def blend_counts(
        self,
        counts: UInt8Array,
        left: int,
        top: int,
        allowed: bytearray | None,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None,
    ) -> None:
        mode = BLEND_MODE_CODES.get(self.resolved_blend(blend_mode), 0)
        revised, blend_error = self.blend_rules(mode)
        shape_plane = self.group_source_shape
        window, stopped = blend_coverage_counts(
            self.pixel_array,
            counts,
            left,
            top,
            allowed,
            *rgba,
            self.group_source_alpha,
            shape_plane,
            shape_plane is not None,
            self.shape_alpha,
            mode,
            revised,
            blend_error is not None,
        )
        if window is not None:
            self.extend_paint_window(slice(window[1], window[3]), slice(window[0], window[2]))
        if stopped:
            assert blend_error is not None
            raise blend_error
