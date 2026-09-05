# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful path-fill painting operations for raster targets."""

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
    internal_scale_rgba_alpha,
)
from core_pdf.impl.render.paths import (
    internal_fill_path_crossing_spans,
    internal_fill_path_sample_crossings,
    internal_fill_path_sample_crossings_numpy,
    internal_intersect_box,
    internal_signed_area_coverage,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number


class internal_PathFillTargetMixin:
    """Scanline and sampled path-fill painting operations."""

    __slots__ = ()

    def fill_path_scanlines(
        self: Any,
        edge_segments: list[tuple[float, float, float, float, float, float]],
        pixel_box: tuple[int, int, int, int],
        rgba: tuple[int, int, int, int],
        blend_mode: str | None,
        fill_rule: str,
    ) -> None:
        blend_normal_pixel = self.blend_normal_pixel
        blend_normal_solid_span = self.blend_normal_solid_span
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        buffer_stack = self.buffer_stack
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_paths_are_axis_aligned_rects = self.clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip.clip_row_visible_spans
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
            if is_pdf_number(group_alpha):
                blended_rgba = internal_scale_rgba_alpha(rgba, group_alpha)
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

    def fast_fill_path(
        self: Any,
        edges: list[tuple[float, float, float, float]],
        bbox: tuple[float, float, float, float],
    ) -> bool:
        """Fill opaque black polygons using one winding scan per raster row."""
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
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        fast_fill_path = self.fast_fill_path
        fill_path_scanlines = self.fill_path_scanlines
        fill_rect = self.fill_rect
        pixel_in_clip = clip.pixel_in_clip
        pixel_view = self.pixel_view
        pixels = self.pixels
        scale = self.scale
        width = self.width
        rect = path.axis_aligned_rect()
        if rect is not None:
            fill_rect(rect, rgba, blend_mode)
            return
        edges = path.fill_edges()
        if not edges:
            return
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
                    fast_bbox = internal_intersect_box(bbox, clip_box)
        if (
            rgba == (0, 0, 0, 255)
            and blend_mode is None
            and fast_bbox is not None
            and fill_rule == "nonzero"
            and fast_fill_path(edges, fast_bbox)
        ):
            return
        clipped_box = clipped_pixel_box(bbox)
        if clipped_box is None:
            return
        pixel_box = clipped_box[1]
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
