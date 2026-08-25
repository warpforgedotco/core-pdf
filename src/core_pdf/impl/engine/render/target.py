# SPDX-License-Identifier: AGPL-3.0-only
"""Mutable raster target, clipping, and render metrics."""

from __future__ import annotations

import heapq
import math
import time
from bisect import bisect_left
from typing import TYPE_CHECKING, Any, cast

import numpy

from core_pdf.impl.engine.array_views import (
    ByteBuffer,
    nearest_indices,
    uint8_image_view,
    uint8_view,
    unit_sample_positions,
)
from core_pdf.impl.engine.image_cache import ImageCacheKey
from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.render.display import (
    LineCap,
    LineJoin,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.engine.render.kernels import (
    AFFINE_BLIT_SCRATCH_BYTES,
    RASTER_CIRCLE_MIN_PIXEL_AREA,
    RASTER_COORDINATE_CACHE_MAX_ENTRIES,
    RASTER_KERNEL_MIN_PIXEL_AREA,
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    RASTER_SAMPLE_OFFSETS,
    axial_shading_t,
    evaluate_pdf_function,
    internal_blend_normal_alpha_array_numpy,
    internal_blend_normal_masked_array_numpy,
    internal_blend_normal_solid_array_numpy,
    internal_blend_normal_solid_span_numpy,
    internal_blend_solid_array_numpy,
    internal_blit_indexed_channels,
    internal_blit_reshaped_channels,
    internal_cached_raster_coordinates,
    internal_color_rgba,
    internal_composite_normal_group_numpy,
    internal_fill_path_crossing_spans,
    internal_fill_path_sample_crossings,
    internal_fill_path_sample_crossings_numpy,
    internal_image_mask_decode_inverts,
    internal_image_mask_samples,
    internal_image_quad,
    internal_image_raw_bytes,
    internal_image_samples,
    internal_intersect_box,
    internal_make_page_geometry,
    internal_shading_color_rgba,
    internal_signed_area_coverage,
    internal_soft_mask_alpha_at,
    internal_soft_mask_samples,
    internal_translate_rect,
    number_array,
    radial_shading_t,
    rasterize_unclipped_line_normal,
)
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.engine.spec.s_07_filters.models import DecodedImage
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.color import ImageColorManager
from core_pdf.impl.engine.spec.s_08_graphics.image_metadata import (
    pdf_int,
    pdf_number,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.render.page import RenderedPage


class internal_RasterTarget:
    """The RGBA byte buffer being painted, plus the transparency-group stack.

    Lifted out of ``RenderedPage.rasterize``. ``pixels`` is *rebound*, not just
    mutated: a ``group-begin`` pushes a fresh buffer that subsequent painting
    goes to, and ``group-end`` pops it and composites it back down. That is why
    this is an object with explicit push/pop rather than a plain buffer.

    Every method hoists ``self.pixels`` into a local before touching it — these
    run per pixel and per span, where a repeated attribute load is not free.
    """

    __slots__ = (
        "pixels",
        "buffer_stack",
        "pixel_views",
        "clip",
        "width",
        "height",
        "scale",
        "crop_x0",
        "crop_y1",
        "page_pixels",
        "page_buffer",
        "raster_x_coordinate_cache",
        "raster_y_coordinate_cache",
        "raster_x_sample_cache",
        "raster_y_sample_cache",
        # Bound clip methods cached at construction. Reading them back is a plain
        # attribute load; going through `self.clip.<name>` would allocate a fresh
        # bound method on every call, and fill_rect alone makes ~1.8M of them.
        "page_box_to_pixels",
        "current_clip",
        "clip_paths_are_axis_aligned_rects",
        "clip_row_visible_spans",
        "pixel_in_clip",
        "page",
        "raster_metrics",
        "page_x_coordinate_cache",
        "page_y_coordinate_cache",
        "crop_y0",
        "color_cache",
        "mark_clip_metadata_dirty",
        "path_bbox",
        "clip_path_stack",
    )

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: internal_ClipState,
        page: "RenderedPage",
        raster_metrics: internal_RasterMetrics,
        width: int,
        height: int,
        scale: float,
        crop_x0: float,
        crop_y0: float,
        crop_y1: float,
        page_view: numpy.ndarray[Any, Any],
    ) -> None:
        self.pixels = pixels
        self.buffer_stack: list[tuple[bytearray, float | None, str | None]] = [
            (pixels, group_alpha, None)
        ]
        self.pixel_views: dict[int, numpy.ndarray[Any, Any]] = {id(pixels): page_view}
        self.clip = clip
        self.width = width
        self.height = height
        self.scale = scale
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.raster_x_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.raster_y_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.raster_x_sample_cache: dict[int, tuple[float, ...]] = {}
        self.raster_y_sample_cache: dict[int, tuple[float, ...]] = {}
        self.page_box_to_pixels = clip.page_box_to_pixels
        self.current_clip = clip.current_clip
        self.clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        self.clip_row_visible_spans = clip.clip_row_visible_spans
        self.pixel_in_clip = clip.pixel_in_clip
        self.page = page
        self.raster_metrics = raster_metrics
        self.page_x_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.page_y_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.crop_y0 = crop_y0
        self.color_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self.mark_clip_metadata_dirty = clip.mark_clip_metadata_dirty
        self.path_bbox = clip.path_bbox
        self.clip_path_stack = clip.clip_path_stack

    def push_group(
        self, buffer: bytearray, group_alpha: float | None, blend_mode: str | None
    ) -> None:
        self.buffer_stack.append((buffer, group_alpha, blend_mode))
        self.pixels = buffer

    def pop_group(self) -> tuple[bytearray, float | None, str | None]:
        child = self.buffer_stack.pop()
        self.pixels = self.buffer_stack[-1][0]
        return child

    def pixel_view(self, buffer: bytearray | bytes) -> numpy.ndarray[Any, Any]:
        """Return a reusable array view for an active RGBA byte buffer."""
        pixel_views = self.pixel_views
        key = id(buffer)
        view = pixel_views.get(key)
        if view is None:
            view = uint8_image_view(buffer, (self.height, self.width, 4))
            pixel_views[key] = view
        return view

    def blend_px(
        self, idx: int, rgba: tuple[int, int, int, int], blend_mode: str | None = None
    ) -> None:
        pixels = self.pixels
        buffer_stack = self.buffer_stack
        sr, sg, sb, sa = rgba
        if sa <= 0:
            return
        target_alpha = buffer_stack[-1][1] if buffer_stack else None
        if sa >= 255 and target_alpha is None and blend_mode is None:
            pixels[idx] = sr
            pixels[idx + 1] = sg
            pixels[idx + 2] = sb
            pixels[idx + 3] = 255
            return
        if pdf_number(target_alpha):
            sa = max(0, min(255, int(round(sa * float(target_alpha)))))
            if sa <= 0:
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
        mode = blend_mode.lower() if isinstance(blend_mode, str) else None
        if mode == "multiply":
            src_r *= dst_r
            src_g *= dst_g
            src_b *= dst_b
        elif mode == "screen":
            src_r = 1.0 - (1.0 - src_r) * (1.0 - dst_r)
            src_g = 1.0 - (1.0 - src_g) * (1.0 - dst_g)
            src_b = 1.0 - (1.0 - src_b) * (1.0 - dst_b)
        out_a = src_a + dst_a * (1.0 - src_a)
        if out_a <= 0:
            pixels[idx] = 0
            pixels[idx + 1] = 0
            pixels[idx + 2] = 0
            pixels[idx + 3] = 0
            return
        out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
        out_a_i = int(round(out_a * 255.0))
        pixels[idx] = max(0, min(255, out_r))
        pixels[idx + 1] = max(0, min(255, out_g))
        pixels[idx + 2] = max(0, min(255, out_b))
        pixels[idx + 3] = max(0, min(255, out_a_i))

    def can_blend_normal_fast(self, blend_mode: str | None) -> bool:
        return blend_mode is None and self.buffer_stack[-1][1] is None

    def blend_normal_pixel(self, idx: int, sr: int, sg: int, sb: int, sa: int) -> None:
        if sa <= 0:
            return
        pixels = self.pixels
        if sa >= 255:
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
        out_a = src_a + dst_a * (1.0 - src_a)
        if out_a <= 0:
            pixels[idx] = 0
            pixels[idx + 1] = 0
            pixels[idx + 2] = 0
            pixels[idx + 3] = 0
            return
        out_r = int(round((sr * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round((sg * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round((sb * src_a + db * dst_a * (1.0 - src_a)) / out_a))
        out_a_i = int(round(out_a * 255.0))
        pixels[idx] = max(0, min(255, out_r))
        pixels[idx + 1] = max(0, min(255, out_g))
        pixels[idx + 2] = max(0, min(255, out_b))
        pixels[idx + 3] = max(0, min(255, out_a_i))

    def blend_normal_solid_span(
        self, row: int, start: int, end: int, rgba: tuple[int, int, int, int]
    ) -> None:
        sr, sg, sb, sa = rgba
        if sa <= 0 or end <= start:
            return
        pixels = self.pixels
        width = self.width
        if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
            target = self.pixel_view(pixels)
            internal_blend_normal_solid_span_numpy(target, row // (width * 4), start, end, rgba)
            return
        start_offset = row + start * 4
        stop_offset = row + end * 4
        if sa >= 255:
            # Keep the destination as a NumPy view instead of allocating a
            # repeated RGBA byte string for every short span.
            self.pixel_view(pixels)[row // (width * 4), start:end] = (sr, sg, sb, 255)
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
            if out_a <= 0:
                pixels[idx] = 0
                pixels[idx + 1] = 0
                pixels[idx + 2] = 0
                pixels[idx + 3] = 0
                continue
            out_r = int(round((sr * src_a + dr * dst_a * one_minus_src_a) / out_a))
            out_g = int(round((sg * src_a + dg * dst_a * one_minus_src_a) / out_a))
            out_b = int(round((sb * src_a + db * dst_a * one_minus_src_a) / out_a))
            out_a_i = int(round(out_a * 255.0))
            pixels[idx] = max(0, min(255, out_r))
            pixels[idx + 1] = max(0, min(255, out_g))
            pixels[idx + 2] = max(0, min(255, out_b))
            pixels[idx + 3] = max(0, min(255, out_a_i))

    def composite_group(
        self, child: bytearray, group_alpha: float | None, group_blend_mode: str | None
    ) -> None:
        buffer_stack = self.buffer_stack
        normalized_blend_mode = (
            group_blend_mode.casefold() if isinstance(group_blend_mode, str) else None
        )
        parent_alpha = buffer_stack[-1][1] if buffer_stack else None
        if normalized_blend_mode in {None, "normal"} and len(child) >= 4_096:
            source_pixels = self.pixel_view(child)
            target_pixels = self.pixel_view(self.pixels)
            source_scale = float(group_alpha) if pdf_number(group_alpha) else 1.0
            target_scale = float(parent_alpha) if pdf_number(parent_alpha) else 1.0
            internal_composite_normal_group_numpy(
                target_pixels,
                source_pixels,
                source_scale,
                target_scale,
            )
            return
        blend_px = self.blend_px
        for idx in range(0, len(child), 4):
            sa = child[idx + 3]
            if sa <= 0:
                continue
            if pdf_number(group_alpha):
                sa = max(0, min(255, int(round(sa * float(group_alpha)))))
                if sa <= 0:
                    continue
            blend_px(
                idx,
                (child[idx], child[idx + 1], child[idx + 2], sa),
                group_blend_mode,
            )

    def fill_rect(
        self,
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
            cx0, cy0, cx1, cy1 = clip_box
            x0 = max(x0, cx0)
            y0 = max(y0, cy0)
            x1 = min(x1, cx1)
            y1 = min(y1, cy1)
            if x1 <= x0 or y1 <= y0:
                return
        pixel_box = self.page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        rectangular_clip = self.clip_paths_are_axis_aligned_rects()
        pixels = self.pixels
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
                        blend_px(row + x * 4, rgba, blend_mode)

    def fill_path_scanlines(
        self,
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
            if fill_rule == "evenodd":
                xs = sorted(x for x, internal_delta in crossings)
                scan_spans = list(zip(xs[0::2], xs[1::2], strict=False))
            else:
                crossings.sort(key=lambda item: item[0])
                spans_list: list[tuple[float, float]] = []
                winding = 0
                previous_x: float | None = None
                index = 0
                while index < len(crossings):
                    x = crossings[index][0]
                    if previous_x is not None and winding != 0 and x > previous_x:
                        spans_list.append((previous_x, x))
                    delta = 0
                    while index < len(crossings) and crossings[index][0] == x:
                        delta += crossings[index][1]
                        index += 1
                    winding += delta
                    previous_x = x
                scan_spans = spans_list
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
                            blend_px(row + px * 4, rgba, blend_mode)

    def draw_glyph_bitmap(
        self,
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
        bitmap_h = pdf_int(bitmap_height, 0) or len(rows)
        bitmap_w = pdf_int(bitmap_width, 0) or max((row.bit_length() for row in rows), default=0)
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
        self,
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
                        blend_px(row + px * 4, rgba, blend_mode)

    def fast_fill_path(
        self,
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
        for py in range(iy0, iy1):
            scan_y = crop_y1 - (py + 0.5) / scale
            intersections: list[tuple[float, int]] = []
            for ex0, ey0, ex1, ey1 in edges:
                if ey0 <= scan_y < ey1:
                    intersections.append((ex0 + (scan_y - ey0) * (ex1 - ex0) / (ey1 - ey0), 1))
                elif ey1 <= scan_y < ey0:
                    intersections.append((ex0 + (scan_y - ey0) * (ex1 - ex0) / (ey1 - ey0), -1))
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
        self,
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
                            blend_mode,
                        )

    def fill_line(
        self,
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
                    pos = float(phase) % total
                    on = True
                    remaining = seg_len
                    while remaining > 0:
                        dash_idx = 0
                        acc = 0.0
                        for i, val in enumerate(dash_array):
                            acc += max(0.0, float(val))
                            if pos < acc:
                                dash_idx = i
                                break
                        on = (dash_idx % 2) == 0
                        dash_end = acc
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
                            blend_mode,
                        )

    def cached_page_coordinates(
        self,
        cache: dict[tuple[int, int], numpy.ndarray[Any, Any]],
        start: int,
        stop: int,
        origin: float,
        direction: float,
    ) -> numpy.ndarray[Any, Any]:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        scale = self.scale
        key = (start, stop)
        coordinates = cache.get(key)
        if coordinates is None:
            coordinates = origin + (numpy.arange(start, stop) + 0.5) / scale * direction
            if len(cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                cache[key] = coordinates
        return coordinates

    def page_x_coordinates(self, start: int, stop: int) -> numpy.ndarray[Any, Any]:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        cached_page_coordinates = self.cached_page_coordinates
        crop_x0 = self.crop_x0
        page_x_coordinate_cache = self.page_x_coordinate_cache
        return cached_page_coordinates(page_x_coordinate_cache, start, stop, crop_x0, 1.0)

    def page_y_coordinates(self, start: int, stop: int) -> numpy.ndarray[Any, Any]:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        cached_page_coordinates = self.cached_page_coordinates
        crop_y1 = self.crop_y1
        page_y_coordinate_cache = self.page_y_coordinate_cache
        return cached_page_coordinates(page_y_coordinate_cache, start, stop, crop_y1, -1.0)

    def blit_opaque_sampled_tiles(
        self,
        source_pixels: numpy.ndarray[Any, Any],
        target_region: numpy.ndarray[Any, Any],
        source_y: numpy.ndarray[Any, Any],
        source_x: numpy.ndarray[Any, Any],
        valid_rows: numpy.ndarray[Any, Any],
        valid_columns: numpy.ndarray[Any, Any],
        comps: int,
        *,
        transposed: bool = False,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        raster_metrics = self.raster_metrics
        row_count = len(valid_rows)
        column_count = len(valid_columns)
        all_valid = bool(valid_rows.all() and valid_columns.all())
        sampled_channels = 1 if comps == 1 else 3
        scratch_bytes_per_pixel = sampled_channels if all_valid else sampled_channels + 8
        tile_columns = min(
            column_count,
            max(1, AFFINE_BLIT_SCRATCH_BYTES // scratch_bytes_per_pixel),
        )
        tile_rows = min(
            row_count,
            max(
                1,
                AFFINE_BLIT_SCRATCH_BYTES // max(1, tile_columns * scratch_bytes_per_pixel),
            ),
        )
        estimated_scratch = tile_rows * tile_columns * scratch_bytes_per_pixel
        raster_metrics.tiled_affine_blit_count += 1
        raster_metrics.tiled_affine_peak_scratch_bytes = max(
            raster_metrics.tiled_affine_peak_scratch_bytes,
            estimated_scratch,
        )
        for row_start in range(0, row_count, tile_rows):
            row_end = min(row_count, row_start + tile_rows)
            for column_start in range(0, column_count, tile_columns):
                column_end = min(column_count, column_start + tile_columns)
                if transposed:
                    gathered_rows = source_pixels.take(source_y[column_start:column_end], axis=0)
                    gathered = gathered_rows.take(source_x[row_start:row_end], axis=1)
                    sampled = gathered[:, :, :sampled_channels].swapaxes(0, 1)
                else:
                    gathered_rows = source_pixels.take(source_y[row_start:row_end], axis=0)
                    sampled = gathered_rows[:, source_x[column_start:column_end], :sampled_channels]
                target_tile = target_region[
                    row_start:row_end,
                    column_start:column_end,
                ]
                if all_valid:
                    target_tile[:, :, 0:3] = sampled
                    target_tile[:, :, 3] = 255
                    continue
                visible = (
                    valid_rows[row_start:row_end, None]
                    & valid_columns[None, column_start:column_end]
                )
                target_tile[visible, 0:3] = sampled[visible]
                target_tile[visible, 3] = 255

    def blit_affine_image(
        self,
        quad: tuple[tuple[float, float], ...],
        converted: ByteBuffer,
        width_px: int,
        height_px: int,
        comps: int,
        data: dict[str, Any],
        blend_mode: str | None,
    ) -> bool:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip = self.clip
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blit_opaque_sampled_tiles = self.blit_opaque_sampled_tiles
        buffer_stack = self.buffer_stack
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_path_stack = clip.clip_path_stack
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = self.current_clip
        height = self.height
        page_box_to_pixels = self.page_box_to_pixels
        page_pixels = self.page_pixels
        page_x_coordinates = self.page_x_coordinates
        page_y_coordinates = self.page_y_coordinates
        pixel_view = self.pixel_view
        pixels = self.pixels
        raster_x_coordinate_cache = self.raster_x_coordinate_cache
        raster_y_coordinate_cache = self.raster_y_coordinate_cache
        scale = self.scale
        width = self.width
        if len(quad) < 3:
            return False
        converted_len = converted.nbytes if isinstance(converted, numpy.ndarray) else len(converted)
        p00 = quad[0]
        p10 = quad[1]
        p01 = quad[2]
        x0 = min(point[0] for point in quad)
        y0 = min(point[1] for point in quad)
        x1 = max(point[0] for point in quad)
        y1 = max(point[1] for point in quad)
        clip_box = current_clip()
        rectangular_clip = clip_box is not None and clip_paths_are_axis_aligned_rects()
        if clip_box is not None:
            clipped = internal_intersect_box((x0, y0, x1, y1), clip_box)
            if clipped is None:
                return True
            x0, y0, x1, y1 = clipped
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return True
        ix0, iy0, ix1, iy1 = pixel_box
        ux = p10[0] - p00[0]
        uy = p10[1] - p00[1]
        vx = p01[0] - p00[0]
        vy = p01[1] - p00[1]
        det = ux * vy - uy * vx
        if abs(det) < 1e-9:
            return False
        inv_det = 1.0 / det
        soft_mask_alpha = data.get("soft_mask_alpha")
        alpha = 255
        if pdf_number(soft_mask_alpha):
            alpha = max(0, min(255, int(round(alpha * float(soft_mask_alpha)))))
        soft_mask = internal_soft_mask_samples(data)
        if soft_mask is None:
            soft_mask_data = None
            soft_mask_width = 0
            soft_mask_height = 0
            soft_mask_len = 0
        else:
            soft_mask_data, soft_mask_width, soft_mask_height = soft_mask
            soft_mask_len = len(soft_mask_data)
        can_write_opaque = (
            alpha == 255 and blend_mode is None and not buffer_stack[-1][1] and soft_mask is None
        )
        normal_fast = can_blend_normal_fast(blend_mode)
        rect_tolerance = max(abs(ux), abs(vy), 1.0) * 1e-6
        if (
            abs(uy) <= rect_tolerance
            and abs(vx) <= rect_tolerance
            and ux > 0
            and vy > 0
            and alpha == 255
            and blend_mode is None
            and can_write_opaque
            and (not clip_path_stack or rectangular_clip)
        ):
            if converted_len < width_px * height_px * comps:
                return False
            inv_ux = 1.0 / ux
            inv_vy = 1.0 / vy
            page_x = (
                crop_x0
                + (internal_cached_raster_coordinates(raster_x_coordinate_cache, ix0, ix1) + 0.5)
                / scale
            )
            source_u = (page_x - p00[0]) * inv_ux
            source_samples = uint8_view(converted)
            valid_x = (source_u >= 0.0) & (source_u <= 1.0)
            safe_x = numpy.clip(
                (source_u * width_px).astype(numpy.intp),
                0,
                width_px - 1,
            )
            axis_page_y = (
                crop_y1
                - (internal_cached_raster_coordinates(raster_y_coordinate_cache, iy0, iy1) + 0.5)
                / scale
            )
            source_y_array = ((1.0 - (axis_page_y - p00[1]) * inv_vy) * height_px).astype(
                numpy.intp
            )
            valid_y = (axis_page_y - p00[1]) * inv_vy >= 0.0
            valid_y &= (axis_page_y - p00[1]) * inv_vy <= 1.0
            safe_y = numpy.clip(source_y_array, 0, height_px - 1)
            target_region = page_pixels[iy0:iy1, ix0:ix1]
            source_pixels = source_samples[: width_px * height_px * comps].reshape(
                height_px,
                width_px,
                comps,
            )
            blit_opaque_sampled_tiles(
                source_pixels,
                target_region,
                safe_y,
                safe_x,
                valid_y,
                valid_x,
                comps,
            )
            return True
        u_from_x = abs(uy) <= rect_tolerance and abs(ux) > rect_tolerance
        u_from_y = abs(ux) <= rect_tolerance and abs(uy) > rect_tolerance
        v_from_x = abs(vy) <= rect_tolerance and abs(vx) > rect_tolerance
        v_from_y = abs(vx) <= rect_tolerance and abs(vy) > rect_tolerance
        if (
            alpha == 255
            and blend_mode is None
            and can_write_opaque
            and (not clip_path_stack or rectangular_clip)
            and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
            and converted_len >= width_px * height_px * comps
        ):
            target_pixels = pixel_view(pixels)
            source_samples = uint8_view(converted)[: width_px * height_px * comps].reshape(
                height_px, width_px, comps
            )
            if u_from_x:
                inv_ux = 1.0 / ux
                inv_vy = 1.0 / vy
                page_x = page_x_coordinates(ix0, ix1)
                page_y = page_y_coordinates(iy0, iy1)
                source_u = (page_x - p00[0]) * inv_ux
                source_v = (page_y - p00[1]) * inv_vy
                valid_x = (source_u >= 0.0) & (source_u <= 1.0)
                valid_y = (source_v >= 0.0) & (source_v <= 1.0)
                source_x = numpy.clip(
                    (source_u * width_px).astype(numpy.intp),
                    0,
                    width_px - 1,
                )
                source_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp),
                    0,
                    height_px - 1,
                )
            else:
                inv_uy = 1.0 / uy
                inv_vx = 1.0 / vx
                page_x = page_x_coordinates(ix0, ix1)
                page_y = page_y_coordinates(iy0, iy1)
                source_v = (page_x - p00[0]) * inv_vx
                source_u = (page_y - p00[1]) * inv_uy
                valid_x = (source_v >= 0.0) & (source_v <= 1.0)
                valid_y = (source_u >= 0.0) & (source_u <= 1.0)
                source_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp),
                    0,
                    height_px - 1,
                )
                source_x = numpy.clip(
                    (source_u * width_px).astype(numpy.intp),
                    0,
                    width_px - 1,
                )
            target_region = target_pixels[iy0:iy1, ix0:ix1]
            blit_opaque_sampled_tiles(
                source_samples,
                target_region,
                source_y,
                source_x,
                valid_y,
                valid_x,
                comps,
                transposed=not u_from_x,
            )
            return True
        if (
            alpha == 255
            and blend_mode is None
            and can_write_opaque
            and (not clip_path_stack or rectangular_clip)
            and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
        ):
            return self.blit_affine_rows_opaque(
                converted,
                comps,
                width_px,
                height_px,
                converted_len,
                p00,
                ux,
                uy,
                vx,
                vy,
                u_from_x,
                ix0,
                iy0,
                ix1,
                iy1,
            )
        if (
            alpha == 255
            and blend_mode is None
            and can_write_opaque
            and soft_mask_data is None
            and (not clip_path_stack or rectangular_clip)
            and converted_len >= width_px * height_px * comps
        ):
            target_pixels = pixel_view(pixels)
            source_samples = uint8_view(converted)[: width_px * height_px * comps].reshape(
                height_px, width_px, comps
            )
            general_page_x = page_x_coordinates(ix0, ix1)
            general_page_y = page_y_coordinates(iy0, iy1)
            general_rel_x = general_page_x[None, :] - p00[0]
            general_rel_y = general_page_y[:, None] - p00[1]
            general_u = (general_rel_x * vy - general_rel_y * vx) * inv_det
            general_v = (ux * general_rel_y - uy * general_rel_x) * inv_det
            general_valid = (
                (general_u >= 0.0) & (general_u <= 1.0) & (general_v >= 0.0) & (general_v <= 1.0)
            )
            general_source_x = numpy.clip(
                (general_u * width_px).astype(numpy.intp),
                0,
                width_px - 1,
            )
            general_source_y = numpy.clip(
                ((1.0 - general_v) * height_px).astype(numpy.intp),
                0,
                height_px - 1,
            )
            general_sampled = source_samples[general_source_y, general_source_x]
            general_target = target_pixels[iy0:iy1, ix0:ix1]
            if comps == 1:
                if general_valid.all():
                    general_target[:, :, 0:3] = general_sampled[:, :, 0, None]
                    general_target[:, :, 3] = 255
                else:
                    general_target[general_valid, 0:3] = general_sampled[general_valid, 0, None]
                    general_target[general_valid, 3] = 255
            else:
                if general_valid.all():
                    general_target[:, :, 0:3] = general_sampled[:, :, :3]
                    general_target[:, :, 3] = 255
                else:
                    general_target[general_valid, 0:3] = general_sampled[general_valid, :3]
                    general_target[general_valid, 3] = 255
            return True
        if (
            normal_fast
            and blend_mode is None
            and soft_mask_data is not None
            and comps >= 3
            and (not clip_path_stack or rectangular_clip)
            and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
            and converted_len >= width_px * height_px * comps
            and soft_mask_len >= soft_mask_width * soft_mask_height
        ):
            source_pixels = uint8_view(converted)[: width_px * height_px * comps].reshape(
                height_px, width_px, comps
            )
            mask_pixels = uint8_view(soft_mask_data)[: soft_mask_width * soft_mask_height].reshape(
                soft_mask_height, soft_mask_width
            )
            page_x = page_x_coordinates(ix0, ix1)
            page_y = page_y_coordinates(iy0, iy1)
            if u_from_x:
                source_u = (page_x - p00[0]) / ux
                source_v = (page_y - p00[1]) / vy
                valid_x = (source_u >= 0.0) & (source_u <= 1.0)
                valid_y = (source_v >= 0.0) & (source_v <= 1.0)
                source_x = numpy.clip((source_u * width_px).astype(numpy.intp), 0, width_px - 1)
                source_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp), 0, height_px - 1
                )
                mask_x = numpy.clip(
                    (source_u * soft_mask_width).astype(numpy.intp),
                    0,
                    soft_mask_width - 1,
                )
                mask_y = numpy.clip(
                    ((1.0 - source_v) * soft_mask_height).astype(numpy.intp),
                    0,
                    soft_mask_height - 1,
                )
                sampled_rgb = source_pixels[source_y[:, None], source_x[None, :], :3]
                sampled_mask = mask_pixels[mask_y[:, None], mask_x[None, :]]
            else:
                source_u = (page_y - p00[1]) / uy
                source_v = (page_x - p00[0]) / vx
                valid_x = (source_v >= 0.0) & (source_v <= 1.0)
                valid_y = (source_u >= 0.0) & (source_u <= 1.0)
                source_x = numpy.clip((source_u * width_px).astype(numpy.intp), 0, width_px - 1)
                source_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp), 0, height_px - 1
                )
                mask_x = numpy.clip(
                    (source_u * soft_mask_width).astype(numpy.intp),
                    0,
                    soft_mask_width - 1,
                )
                mask_y = numpy.clip(
                    ((1.0 - source_v) * soft_mask_height).astype(numpy.intp),
                    0,
                    soft_mask_height - 1,
                )
                sampled_rgb = source_pixels[source_x[:, None], source_y[None, :], :3]
                sampled_mask = mask_pixels[mask_x[:, None], mask_y[None, :]]
            valid = valid_y[:, None] & valid_x[None, :]
            target_pixels = uint8_image_view(pixels, (height, width, 4))
            target_region = target_pixels[iy0:iy1, ix0:ix1]
            if valid.all():
                internal_blend_normal_masked_array_numpy(
                    target_region,
                    sampled_rgb,
                    sampled_mask,
                    alpha,
                )
            else:
                selected = target_region[valid].copy()
                internal_blend_normal_masked_array_numpy(
                    selected,
                    sampled_rgb[valid],
                    sampled_mask[valid],
                    alpha,
                )
                target_region[valid] = selected
            return True
        if (
            normal_fast
            and blend_mode is None
            and soft_mask_data is not None
            and comps >= 3
            and (not clip_path_stack or rectangular_clip)
            and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
        ):
            return self.blit_affine_rows_soft_masked(
                converted,
                comps,
                width_px,
                height_px,
                alpha,
                soft_mask_data,
                soft_mask_width,
                soft_mask_height,
                p00,
                ux,
                uy,
                vx,
                vy,
                u_from_x,
                u_from_y,
                ix0,
                iy0,
                ix1,
                iy1,
            )
        for py in range(iy0, iy1):
            page_y_value = crop_y1 - (py + 0.5) / scale
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            for px in range(ix0, ix1):
                if not rectangular_clip:
                    index = bisect_left(visible_spans, (px + 1, -1))
                    if index <= 0:
                        continue
                    start, end = visible_spans[index - 1]
                    if not (start <= px < end):
                        continue
                page_x_value = crop_x0 + (px + 0.5) / scale
                rel_x = page_x_value - p00[0]
                rel_y = page_y_value - p00[1]
                scalar_u = (rel_x * vy - rel_y * vx) * inv_det
                scalar_v = (ux * rel_y - uy * rel_x) * inv_det
                if scalar_u < 0.0 or scalar_u > 1.0 or scalar_v < 0.0 or scalar_v > 1.0:
                    continue
                src_x_index = int(scalar_u * width_px)
                if src_x_index < 0:
                    src_x_index = 0
                elif src_x_index >= width_px:
                    src_x_index = width_px - 1
                src_y_index = int((1.0 - scalar_v) * height_px)
                if src_y_index < 0:
                    src_y_index = 0
                elif src_y_index >= height_px:
                    src_y_index = height_px - 1
                src_idx = (src_y_index * width_px + src_x_index) * comps
                if src_idx >= converted_len:
                    continue
                if comps == 1:
                    gray = converted[src_idx]
                    if can_write_opaque:
                        pixels[row + px * 4 : row + px * 4 + 4] = (
                            gray,
                            gray,
                            gray,
                            255,
                        )
                        continue
                    rgba = (gray, gray, gray, 255)
                else:
                    if src_idx + 3 > converted_len:
                        continue
                    if can_write_opaque:
                        pixels[row + px * 4 : row + px * 4 + 4] = (
                            converted[src_idx],
                            converted[src_idx + 1],
                            converted[src_idx + 2],
                            255,
                        )
                        continue
                    rgba = (
                        converted[src_idx],
                        converted[src_idx + 1],
                        converted[src_idx + 2],
                        255,
                    )
                if soft_mask_data is None:
                    pixel_mask_alpha = 255
                else:
                    mask_x_index = int(scalar_u * soft_mask_width)
                    if mask_x_index < 0:
                        mask_x_index = 0
                    elif mask_x_index >= soft_mask_width:
                        mask_x_index = soft_mask_width - 1
                    mask_y_index = int((1.0 - scalar_v) * soft_mask_height)
                    if mask_y_index < 0:
                        mask_y_index = 0
                    elif mask_y_index >= soft_mask_height:
                        mask_y_index = soft_mask_height - 1
                    mask_idx = mask_y_index * soft_mask_width + mask_x_index
                    pixel_mask_alpha = soft_mask_data[mask_idx] if mask_idx < soft_mask_len else 255
                if pixel_mask_alpha <= 0:
                    continue
                if alpha != 255:
                    rgba = (
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        alpha,
                    )
                if pixel_mask_alpha != 255:
                    rgba = (
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        max(0, min(255, int(round(rgba[3] * pixel_mask_alpha / 255)))),
                    )
                if normal_fast:
                    blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], rgba[3])
                else:
                    blend_px(row + px * 4, rgba, blend_mode)
        return True

    def blit_image(
        self,
        box: tuple[float, float, float, float] | None,
        data: dict[str, Any],
        blend_mode: str | None,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip = self.clip
        page = self.page
        blend_px = self.blend_px
        blit_affine_image = self.blit_affine_image
        buffer_stack = self.buffer_stack
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_path_stack = clip.clip_path_stack
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip_row_visible_spans
        current_clip = self.current_clip
        page_box_to_pixels = self.page_box_to_pixels
        raster_metrics = self.raster_metrics
        width = self.width
        if box is None:
            return
        dictionary = data.get("dictionary")
        raw = data.get("raw_data")
        if not isinstance(dictionary, dict) or not isinstance(raw, (bytes, bytearray, memoryview)):
            return
        if lookup_dict_key(dictionary, "ImageMask") is True:
            self.blit_image_mask(data, dictionary, raw, box, blend_mode)
            return
        width_px = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
        height_px = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
        if width_px <= 0 or height_px <= 0:
            return
        shared_source = data.get("image_source")
        source_alpha: numpy.ndarray[Any, Any] | None = None
        decode_started = time.perf_counter()
        try:
            if shared_source is not None and hasattr(shared_source, "decode"):
                shared_raster = shared_source.decode()
                if shared_raster is None:
                    return
                source_channels = 1 if shared_raster.color_model == "gray" else 3
                converted = shared_raster.array[:, :, :source_channels].reshape(-1)
                if shared_raster.has_alpha:
                    source_alpha = shared_raster.array[:, :, source_channels].reshape(-1)
            else:
                raw_bytes = internal_image_raw_bytes(raw)
                page_cache_key = (width_px, height_px, id(raw))
                source_key = getattr(shared_source, "cache_key", None)
                if not isinstance(source_key, tuple):
                    source_key = ("raw", len(raw_bytes), width_px, height_px, id(raw))
                conversion_key = ImageCacheKey(
                    "converted-image",
                    tuple(source_key),
                    (width_px, height_px),
                )
                converted = (
                    page.image_cache.get(conversion_key)
                    if page.image_cache is not None
                    else page.image_conversion_cache.get(page_cache_key)
                )
                source_channels = 0
                if converted is None:
                    converted_cache_key = "__core_pdf_render_converted_image_data__"
                    converted_cache = dictionary.get(converted_cache_key)
                    if (
                        isinstance(converted_cache, tuple)
                        and len(converted_cache) == 4
                        and converted_cache[0] == len(raw_bytes)
                        and converted_cache[1] == width_px
                        and converted_cache[2] == height_px
                        and isinstance(converted_cache[3], (bytes, memoryview, numpy.ndarray))
                    ):
                        converted = converted_cache[3]
                        if page.image_cache is not None:
                            page.image_cache.put(conversion_key, converted)
                        else:
                            page.image_conversion_cache[page_cache_key] = converted
                if converted is None:
                    sample_result = internal_image_samples(raw_bytes, dictionary)
                    samples: bytes | memoryview | DecodedImage
                    sample_dictionary: dict[Any, Any]
                    if sample_result is None:
                        samples = raw_bytes
                        sample_dictionary = dictionary
                    else:
                        samples, sample_dictionary = sample_result
                    if isinstance(samples, DecodedImage):
                        converted = samples.array.reshape(-1)
                    else:
                        converted_data = ImageColorManager.convert_image_data(
                            samples,
                            sample_dictionary,
                        )
                        if converted_data is None:
                            return
                        converted = uint8_view(converted_data)
                    dictionary[converted_cache_key] = (
                        len(raw_bytes),
                        width_px,
                        height_px,
                        converted,
                    )
                    if page.image_cache is not None:
                        page.image_cache.put(conversion_key, converted)
                    else:
                        page.image_conversion_cache[page_cache_key] = converted
        except Exception:
            converted = None
            source_channels = 0
        raster_metrics.image_decode_seconds += time.perf_counter() - decode_started
        if converted is None or len(converted) == 0:
            return
        if source_alpha is not None:
            alpha_view = uint8_view(source_alpha)
            expected_alpha = width_px * height_px
            if len(alpha_view) >= expected_alpha:
                alpha_view = alpha_view[:expected_alpha]
                if not numpy.any(alpha_view) and internal_soft_mask_samples(data) is None:
                    return
                if numpy.all(alpha_view == 255) and internal_soft_mask_samples(data) is None:
                    source_alpha = None
        quad = internal_image_quad(data)
        comps = source_channels or (3 if len(converted) >= width_px * height_px * 3 else 1)
        raster_metrics.image_count += 1
        if quad is not None and source_alpha is None:
            blit_started = time.perf_counter()
            affine_blit = blit_affine_image(
                quad, converted, width_px, height_px, comps, data, blend_mode
            )
            raster_metrics.image_blit_seconds += time.perf_counter() - blit_started
            if affine_blit:
                return
        x0, y0, x1, y1 = box
        clip_box = current_clip()
        if clip_box is not None:
            cx0, cy0, cx1, cy1 = clip_box
            x0 = max(x0, cx0)
            y0 = max(y0, cy0)
            x1 = min(x1, cx1)
            y1 = min(y1, cy1)
            if x1 <= x0 or y1 <= y0:
                return
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        x_span = max(1, ix1 - ix0)
        y_span = max(1, iy1 - iy0)
        src_x_map = nearest_indices(x_span, width_px)
        src_y_map = nearest_indices(y_span, height_px)
        # ImageSource keeps a same-sized alpha plane for general consumers.
        # A PDF soft mask may have substantially higher resolution than its
        # colour image, so use the original mask here instead of the
        # downsampled shared alpha.  This preserves scan text and line art.
        soft_mask = internal_soft_mask_samples(data)
        if soft_mask is not None:
            source_alpha = None
        x_unit_map = (
            unit_sample_positions(x_span)
            if soft_mask is not None
            else numpy.empty(0, dtype=numpy.float64)
        )
        y_unit_map = (
            unit_sample_positions(y_span)
            if soft_mask is not None
            else numpy.empty(0, dtype=numpy.float64)
        )
        soft_mask_alpha = data.get("soft_mask_alpha")
        if pdf_number(soft_mask_alpha):
            has_constant_alpha = True
            constant_alpha = float(soft_mask_alpha)
        else:
            has_constant_alpha = False
            constant_alpha = 1.0
        target_alpha = buffer_stack[-1][1] if buffer_stack else None
        can_write_opaque_rows = (
            (not clip_path_stack or clip_paths_are_axis_aligned_rects())
            and blend_mode is None
            and soft_mask is None
            and source_alpha is None
            and (not has_constant_alpha or constant_alpha >= 1.0)
            and not pdf_number(target_alpha)
        )
        normal_fast = can_blend_normal_fast(blend_mode)
        if can_write_opaque_rows:
            self.blit_image_rows_opaque(
                converted,
                comps,
                src_x_map,
                src_y_map,
                ix0,
                iy0,
                ix1,
                iy1,
                x_span,
                width_px,
                height_px,
            )
            return
        if normal_fast:
            self.blit_image_rows_blended(
                converted,
                comps,
                source_alpha,
                constant_alpha,
                has_constant_alpha,
                soft_mask,
                x_unit_map,
                y_unit_map,
                src_x_map,
                src_y_map,
                ix0,
                iy0,
                ix1,
                iy1,
                x_span,
                y_span,
                width_px,
            )
            return
        for dy, py in enumerate(range(iy0, iy1)):
            src_y = src_y_map[dy]
            mask_v = 1.0 - y_unit_map[dy] if soft_mask is not None else 1.0
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
                    dx = px - ix0
                    src_x = src_x_map[dx]
                    src_idx = (src_y * width_px + src_x) * comps
                    if src_idx >= len(converted):
                        continue
                    if comps == 1:
                        gray = converted[src_idx]
                        rgba = (gray, gray, gray, 255)
                    else:
                        rgba = (
                            converted[src_idx],
                            converted[src_idx + 1],
                            converted[src_idx + 2],
                            255,
                        )
                    if source_alpha is not None:
                        alpha_index = src_y * width_px + src_x
                        if alpha_index >= len(source_alpha):
                            continue
                        rgba = (rgba[0], rgba[1], rgba[2], int(source_alpha[alpha_index]))
                    if soft_mask is not None:
                        pixel_mask_alpha = internal_soft_mask_alpha_at(
                            soft_mask,
                            x_unit_map[dx],
                            mask_v,
                        )
                        if pixel_mask_alpha <= 0:
                            continue
                        if pixel_mask_alpha != 255:
                            rgba = (
                                rgba[0],
                                rgba[1],
                                rgba[2],
                                max(
                                    0,
                                    min(
                                        255,
                                        int(round(rgba[3] * pixel_mask_alpha / 255)),
                                    ),
                                ),
                            )
                    if has_constant_alpha:
                        rgba = (
                            rgba[0],
                            rgba[1],
                            rgba[2],
                            max(0, min(255, int(round(rgba[3] * constant_alpha)))),
                        )
                    blend_px(row + px * 4, rgba, blend_mode)

    def record_image_timings(self) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        page = self.page
        raster_metrics = self.raster_metrics
        page.metadata["__core_pdf_raster_image_timings__"] = raster_metrics.as_metadata()

    def fill_join(
        self,
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
        self,
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

    def shading_box(self, data: dict[str, Any]) -> tuple[float, float, float, float]:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        crop_x0 = self.crop_x0
        crop_y0 = self.crop_y0
        crop_y1 = self.crop_y1
        scale = self.scale
        width = self.width
        dictionary = data.get("dictionary")
        box = None
        if isinstance(dictionary, dict):
            bbox_values = number_array(lookup_dict_key(dictionary, "BBox"))
            if len(bbox_values) >= 4:
                box = tuple(bbox_values[:4])
        if box is None:
            raw_box = data.get("bbox")
            if isinstance(raw_box, RectBox):
                box = (raw_box.x0, raw_box.y0, raw_box.x1, raw_box.y1)
            elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                try:
                    box = tuple(float(value) for value in raw_box[:4])
                except (TypeError, ValueError):
                    box = None
        if box is None:
            box = (crop_x0, crop_y0, crop_x0 + width / scale, crop_y1)
        x0, y0, x1, y1 = box
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def paint_shading(self, data: dict[str, Any], blend_mode: str | None) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_row_visible_spans = self.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = self.current_clip
        page_box_to_pixels = self.page_box_to_pixels
        scale = self.scale
        shading_box = self.shading_box
        width = self.width
        dictionary = data.get("dictionary")
        if not isinstance(dictionary, dict):
            return
        shading_type = pdf_int(lookup_dict_key(dictionary, "ShadingType"), 0)
        if shading_type not in {2, 3}:
            return
        coords = number_array(lookup_dict_key(dictionary, "Coords"))
        if (shading_type == 2 and len(coords) < 4) or (shading_type == 3 and len(coords) < 6):
            return
        domain = number_array(lookup_dict_key(dictionary, "Domain"))
        if len(domain) < 2:
            domain = [0.0, 1.0]
        extend = lookup_dict_key(dictionary, "Extend")
        extend0 = isinstance(extend, (list, tuple)) and len(extend) > 0 and extend[0] is True
        extend1 = isinstance(extend, (list, tuple)) and len(extend) > 1 and extend[1] is True
        function = lookup_dict_key(dictionary, "Function")
        color_space = lookup_dict_key(dictionary, "ColorSpace")
        x0, y0, x1, y1 = shading_box(data)
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
        soft_mask_alpha = data.get("soft_mask_alpha")
        fill_opacity = data.get("fill_opacity")
        normal_fast = can_blend_normal_fast(blend_mode)
        domain_span = domain[1] - domain[0]
        # page_x only depends on the column, so it is identical on every row;
        # computing it once here avoids redoing the same division per pixel.
        page_x_values = [crop_x0 + (px + 0.5) / scale for px in range(ix0, ix1)]
        # A gradient function is evaluated purely from unit_t, and real pages
        # spend most of their pixels in a clamped extend region or an
        # axis-aligned band where unit_t repeats exactly -- cache per call so
        # evaluate_pdf_function (which can run an arbitrary PDF function,
        # including a PostScript calculator) is not repeated for the same t.
        color_cache: dict[float, tuple[int, int, int, int]] = {}
        for py in range(iy0, iy1):
            page_y = crop_y1 - (py + 0.5) / scale
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            for px in range(ix0, ix1):
                index = bisect_left(visible_spans, (px + 1, -1))
                if index <= 0:
                    continue
                start, end = visible_spans[index - 1]
                if not (start <= px < end):
                    continue
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
                rgba = color_cache.get(unit_t)
                if rgba is None:
                    value = domain[0] + unit_t * domain_span
                    rgba = internal_shading_color_rgba(
                        color_space,
                        evaluate_pdf_function(function, value),
                        fill_opacity,
                    )
                    # Bounded like the raster coordinate caches below: a
                    # diagonal or radial gradient can produce a near-unique
                    # unit_t per pixel, so an unbounded cache would grow to
                    # one entry per pixel on a large fill.
                    if len(color_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                        color_cache[unit_t] = rgba
                if pdf_number(soft_mask_alpha):
                    rgba = (
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        max(
                            0,
                            min(255, int(round(rgba[3] * float(soft_mask_alpha)))),
                        ),
                    )
                if normal_fast:
                    blend_normal_pixel(row + px * 4, *rgba)
                else:
                    blend_px(row + px * 4, rgba, blend_mode)

    def stroke_path(
        self,
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

    def paint_tiling_glyphs(
        self,
        glyphs: Any,
        tx: float,
        ty: float,
        blend_mode: str | None,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        draw_glyph_bitmap = self.draw_glyph_bitmap
        if type(glyphs) is not list:
            return
        for glyph in glyphs:
            if type(glyph) is not dict or glyph.get("visible") is False:
                continue
            bbox = internal_translate_rect(glyph.get("bbox"), tx, ty)
            rgba = internal_color_rgba(glyph.get("fill_color"), None)
            draw_glyph_bitmap(
                bbox,
                glyph.get("bitmap"),
                rgba,
                blend_mode,
                glyph.get("bitmap_width"),
                glyph.get("bitmap_height"),
            )

    def paint_tiling_drawing(
        self,
        drawing: dict[str, Any],
        tx: float,
        ty: float,
        parent_blend_mode: str | None,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        fill_path = self.fill_path
        paint_shading = self.paint_shading
        stroke_path = self.stroke_path
        kind = drawing.get("kind")
        blend = drawing.get("blend_mode") or parent_blend_mode
        raw_path = drawing.get("path")
        path = raw_path.translated(tx, ty) if type(raw_path) is CapturedPath else None
        if kind == "shading" and isinstance(drawing.get("dictionary"), dict):
            paint_shading(
                {
                    "dictionary": drawing.get("dictionary"),
                    "bbox": internal_translate_rect(drawing.get("rect"), tx, ty),
                    "fill_opacity": drawing.get("fill_opacity"),
                    "soft_mask_alpha": drawing.get("soft_mask_alpha"),
                },
                blend,
            )
            return
        if kind not in {"fill", "fillstroke", "stroke"}:
            return
        if path is None:
            return
        fill_rgba = internal_color_rgba(drawing.get("fill"), drawing.get("fill_opacity"))
        if kind in {"fill", "fillstroke"}:
            fill_path(
                path,
                fill_rgba,
                blend,
                drawing.get("fill_rule") or "nonzero",
            )
        if kind in {"stroke", "fillstroke"}:
            stroke_rgba = internal_color_rgba(
                drawing.get("stroke_color"), drawing.get("stroke_opacity")
            )
            stroke_path(
                path,
                float(drawing.get("line_width") or 1.0),
                stroke_rgba,
                drawing.get("dash_pattern"),
                blend,
                int(drawing.get("line_cap") or 0),
                int(drawing.get("line_join") or 0),
            )

    def paint_tiling_pattern(
        self,
        pattern: dict[str, Any],
        target_data: dict[str, Any],
        blend_mode: str | None,
    ) -> bool:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        crop_x0 = self.crop_x0
        crop_y0 = self.crop_y0
        crop_y1 = self.crop_y1
        current_clip = self.current_clip
        paint_tiling_drawing = self.paint_tiling_drawing
        paint_tiling_glyphs = self.paint_tiling_glyphs
        path_bbox = self.path_bbox
        scale = self.scale
        width = self.width
        raw_bbox = pattern.get("bbox")
        raw_bbox_type = type(raw_bbox)
        if raw_bbox_type is not list and raw_bbox_type is not tuple:
            return False
        raw_bbox = cast(list[Any] | tuple[Any, ...], raw_bbox)
        if len(raw_bbox) < 4:
            return False
        try:
            cell_x0, cell_y0, cell_x1, cell_y1 = (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
            x_step = abs(float(pattern.get("x_step", 0.0)))
            y_step = abs(float(pattern.get("y_step", 0.0)))
        except (TypeError, ValueError):
            return False
        if x_step <= 0.0 or y_step <= 0.0:
            return False
        drawings = pattern.get("drawings")
        glyphs = pattern.get("glyphs")
        if type(drawings) is not list:
            drawings = []
        if type(glyphs) is not list:
            glyphs = []
        if not drawings and not glyphs:
            return False
        target_box = target_data.get("bbox") or path_bbox(target_data.get("path"))
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
        clip_box = current_clip()
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
                    for drawing in drawings:
                        if type(drawing) is not dict:
                            continue
                        paint_tiling_drawing(drawing, tx, ty, blend_mode)
                    paint_tiling_glyphs(glyphs, tx, ty, blend_mode)
                cells += 1
                x += x_step
            y += y_step
        return True

    def paint_fill_pattern(self, data: dict[str, Any], blend_mode: str | None) -> bool:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip_path_stack = self.clip_path_stack
        mark_clip_metadata_dirty = self.mark_clip_metadata_dirty
        paint_shading = self.paint_shading
        paint_tiling_pattern = self.paint_tiling_pattern
        path_bbox = self.path_bbox
        pattern = data.get("fill_pattern")
        if not isinstance(pattern, dict):
            return False
        path = data.get("path")
        pushed_clip = False
        if type(path) is CapturedPath and path.has_segments():
            clip_path_stack.append((path, data.get("fill_rule") or "nonzero"))
            mark_clip_metadata_dirty()
            pushed_clip = True
        try:
            if pattern.get("kind") == "shading":
                dictionary = pattern.get("dictionary")
                if not isinstance(dictionary, dict):
                    return False
                shading_data = {
                    "dictionary": dictionary,
                    "bbox": data.get("bbox") or path_bbox(path),
                    "fill_opacity": data.get("fill_opacity"),
                    "soft_mask_alpha": data.get("soft_mask_alpha"),
                }
                paint_shading(shading_data, blend_mode)
                return True
            if pattern.get("kind") == "tiling":
                return paint_tiling_pattern(pattern, data, blend_mode)
        finally:
            if pushed_clip:
                clip_path_stack.pop()
                mark_clip_metadata_dirty()
        return False

    def paint_typed_path(self, item: PathPaintItem) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        color_cache = self.color_cache
        fill_path = self.fill_path
        stroke_path = self.stroke_path
        path = item.path
        if type(path) is not CapturedPath:
            return
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
                rgba = (
                    rgba[0],
                    rgba[1],
                    rgba[2],
                    max(0, min(255, int(round(rgba[3] * float(soft_mask_alpha))))),
                )
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
                stroke_rgba = (
                    stroke_rgba[0],
                    stroke_rgba[1],
                    stroke_rgba[2],
                    max(
                        0,
                        min(
                            255,
                            int(round(stroke_rgba[3] * float(soft_mask_alpha))),
                        ),
                    ),
                )
            stroke_path(
                path,
                float(item.line_width or 1.0),
                stroke_rgba,
                item.dash_pattern,
                blend_mode,
                int(item.line_cap or 0),
                int(item.line_join or 0),
            )

    def blit_image_mask(
        self, data: Any, dictionary: Any, raw: Any, box: Any, blend_mode: Any
    ) -> None:
        """Paint a stencil-mask image: 1 bit per sample selecting the fill colour.

        Split out of :meth:`blit_image`, which dispatches to it before any
        decode work — an ImageMask carries no colour samples of its own."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        buffer_stack = self.buffer_stack
        can_blend_normal_fast = self.can_blend_normal_fast
        clip_path_stack = self.clip_path_stack
        clip_paths_are_axis_aligned_rects = self.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip_row_visible_spans
        current_clip = self.current_clip
        page_box_to_pixels = self.page_box_to_pixels
        pixel_view = self.pixel_view
        pixels = self.pixels
        width = self.width
        width_px = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
        height_px = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
        if width_px <= 0 or height_px <= 0:
            return
        shared_mask_source = data.get("image_source")
        shared_mask = (
            shared_mask_source.decode()
            if shared_mask_source is not None and hasattr(shared_mask_source, "decode")
            else None
        )
        mask = (
            shared_mask.array[:, :, 1].reshape(-1)
            if shared_mask is not None and shared_mask.has_alpha
            else internal_image_mask_samples(
                internal_image_raw_bytes(raw), dictionary, width_px, height_px
            )
        )
        if mask is None or len(mask) == 0:
            return
        x0, y0, x1, y1 = box
        clip_box = current_clip()
        if clip_box is not None:
            cx0, cy0, cx1, cy1 = clip_box
            x0 = max(x0, cx0)
            y0 = max(y0, cy0)
            x1 = min(x1, cx1)
            y1 = min(y1, cy1)
            if x1 <= x0 or y1 <= y0:
                return
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        x_span = max(1, ix1 - ix0)
        y_span = max(1, iy1 - iy0)
        src_x_map = nearest_indices(x_span, width_px)
        src_y_map = nearest_indices(y_span, height_px)
        decode = lookup_dict_key(dictionary, "Decode")
        invert = internal_image_mask_decode_inverts(decode)
        target_alpha = buffer_stack[-1][1] if buffer_stack else None
        if (
            (not clip_path_stack or clip_paths_are_axis_aligned_rects())
            and blend_mode is None
            and not pdf_number(target_alpha)
        ):
            if len(mask) >= width_px * height_px:
                source_mask = uint8_image_view(mask, (height_px, width_px), allow_trailing=True)
                source_x = src_x_map
                source_y = src_y_map
                sampled_mask = source_mask[source_y[:, None], source_x[None, :]]
                if invert:
                    sampled_mask = 255 - sampled_mask
                target_pixels = pixel_view(pixels)
                target_region = target_pixels[iy0:iy1, ix0:ix1]
                visible = sampled_mask != 0
                target_region[visible, :3] = 0
                target_region[visible, 3] = sampled_mask[visible]
                return
            for dy, py in enumerate(range(iy0, iy1)):
                src_y = src_y_map[dy]
                row = py * width * 4
                source_row = src_y * width_px
                for dx, px in enumerate(range(ix0, ix1)):
                    src_idx = source_row + src_x_map[dx]
                    if src_idx >= len(mask):
                        continue
                    alpha = 255 - mask[src_idx] if invert else mask[src_idx]
                    if alpha:
                        idx = row + px * 4
                        pixels[idx] = 0
                        pixels[idx + 1] = 0
                        pixels[idx + 2] = 0
                        pixels[idx + 3] = alpha
            return
        normal_fast = can_blend_normal_fast(blend_mode)
        for dy, py in enumerate(range(iy0, iy1)):
            src_y = src_y_map[dy]
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            for dx, px in enumerate(range(ix0, ix1)):
                index = bisect_left(visible_spans, (px + 1, -1))
                if index <= 0:
                    continue
                start, end = visible_spans[index - 1]
                if not (start <= px < end):
                    continue
                src_x = src_x_map[dx]
                src_idx = src_y * width_px + src_x
                if src_idx >= len(mask):
                    continue
                alpha = mask[src_idx]
                if invert:
                    alpha = 255 - alpha
                if normal_fast:
                    blend_normal_pixel(row + px * 4, 0, 0, 0, alpha)
                else:
                    blend_px(row + px * 4, (0, 0, 0, alpha), blend_mode)
        return

    def blit_image_rows_blended(
        self,
        converted: Any,
        comps: Any,
        source_alpha: Any,
        constant_alpha: Any,
        has_constant_alpha: Any,
        soft_mask: Any,
        x_unit_map: Any,
        y_unit_map: Any,
        src_x_map: Any,
        src_y_map: Any,
        ix0: Any,
        iy0: Any,
        ix1: Any,
        iy1: Any,
        x_span: Any,
        y_span: Any,
        width_px: Any,
    ) -> None:
        """Blend an axis-aligned image over the buffer with NumPy, row band by band.

        The path :meth:`blit_image` takes when the image is translucent, carries
        a soft mask, or is clipped — anything that rules out a straight copy but
        still allows vectorised normal blending."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        clip_row_visible_spans = self.clip_row_visible_spans
        pixel_view = self.pixel_view
        pixels = self.pixels
        visible = numpy.zeros((y_span, x_span), dtype=bool)
        for dy, py in enumerate(range(iy0, iy1)):
            for clip_start, clip_end in clip_row_visible_spans(py):
                start = max(ix0, clip_start)
                end = min(ix1, clip_end)
                if end > start:
                    visible[dy, start - ix0 : end - ix0] = True
        source_view = uint8_view(converted)
        source_length = len(source_view)
        base_index = src_y_map[:, None] * width_px + src_x_map[None, :]
        if comps == 1:
            sample_index = base_index
            bounds_ok = sample_index < source_length
        else:
            sample_index = base_index * comps
            bounds_ok = sample_index + 2 < source_length
        alpha_grid = numpy.full((y_span, x_span), 255, dtype=numpy.uint8)
        if source_alpha is not None and len(source_alpha) > 0:
            alpha_view = uint8_view(source_alpha)
            alpha_grid = alpha_view[numpy.minimum(base_index, len(alpha_view) - 1)]
            bounds_ok &= base_index < len(alpha_view)
        if soft_mask is not None:
            samples, mask_width, mask_height = soft_mask
            mask_view = uint8_view(samples)
            mask_x = numpy.clip((x_unit_map * mask_width).astype(numpy.intp), 0, mask_width - 1)
            mask_y = numpy.clip((y_unit_map * mask_height).astype(numpy.intp), 0, mask_height - 1)
            mask_index = mask_y[:, None] * mask_width + mask_x[None, :]
            mask_alpha = numpy.where(
                mask_index < len(mask_view),
                mask_view[numpy.minimum(mask_index, len(mask_view) - 1)],
                255,
            )
            bounds_ok &= mask_alpha > 0
            scaled = numpy.rint(alpha_grid.astype(numpy.float64) * mask_alpha / 255.0)
            alpha_grid = numpy.clip(scaled, 0, 255).astype(numpy.uint8)
        if has_constant_alpha:
            scaled = numpy.rint(alpha_grid.astype(numpy.float64) * constant_alpha)
            alpha_grid = numpy.clip(scaled, 0, 255).astype(numpy.uint8)
        selected = visible & bounds_ok & (alpha_grid > 0)
        if selected.any():
            target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
            safe_index = numpy.minimum(sample_index, source_length - comps)
            source_rgb = numpy.empty((y_span, x_span, 3), dtype=numpy.uint8)
            if comps == 1:
                gray = source_view[safe_index]
                source_rgb[:, :, 0] = gray
                source_rgb[:, :, 1] = gray
                source_rgb[:, :, 2] = gray
            else:
                source_rgb[:, :, 0] = source_view[safe_index]
                source_rgb[:, :, 1] = source_view[safe_index + 1]
                source_rgb[:, :, 2] = source_view[safe_index + 2]
            dest = target_region[selected]
            src_a = alpha_grid[selected] / 255.0
            dst_a = dest[:, 3] / 255.0
            inverse = 1.0 - src_a
            out_a = src_a + dst_a * inverse
            out_rgb = (
                source_rgb[selected].astype(numpy.float64) * src_a[:, None]
                + dest[:, :3].astype(numpy.float64) * dst_a[:, None] * inverse[:, None]
            ) / out_a[:, None]
            blended_output = numpy.rint(numpy.column_stack((out_rgb, out_a[:, None] * 255.0)))
            numpy.clip(blended_output, 0, 255, out=blended_output)
            target_region[selected] = blended_output.astype(numpy.uint8)

    def blit_image_rows_opaque(
        self,
        converted: Any,
        comps: Any,
        src_x_map: Any,
        src_y_map: Any,
        ix0: Any,
        iy0: Any,
        ix1: Any,
        iy1: Any,
        x_span: Any,
        width_px: Any,
        height_px: Any,
    ) -> None:
        """Copy an axis-aligned image straight into the buffer, row by row.

        The fastest path in :meth:`blit_image`: fully opaque, unclipped, no
        blend mode, so each destination row is a slice assignment and
        identical source rows are reused from a per-row cache."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        pixel_view = self.pixel_view
        pixels = self.pixels
        width = self.width
        converted_length = (
            converted.nbytes if isinstance(converted, numpy.ndarray) else len(converted)
        )
        expected_length = width_px * height_px * comps
        if converted_length >= expected_length:
            source_samples = uint8_view(converted)[:expected_length]
            if comps == 1:
                source_samples = source_samples.reshape(height_px, width_px)
            else:
                source_samples = source_samples.reshape(height_px, width_px, comps)
            source_x = src_x_map
            source_y = src_y_map
            sampled = source_samples[source_y[:, None], source_x[None, :]]
            target_pixels = pixel_view(pixels)
            target_region = target_pixels[iy0:iy1, ix0:ix1]
            if comps == 1:
                target_region[:, :, :3] = sampled[:, :, None]
            else:
                target_region[:, :, :3] = sampled[:, :, :3]
            target_region[:, :, 3] = 255
            return
        rect_row_cache: dict[int, bytes] = {}
        for dy, py in enumerate(range(iy0, iy1)):
            src_y = src_y_map[dy]
            row_bytes = rect_row_cache.get(src_y)
            if row_bytes is None:
                row_out = bytearray(x_span * 4)
                out = 0
                if comps == 1:
                    row_base = src_y * width_px
                    for src_x in src_x_map:
                        src_idx = row_base + src_x
                        if src_idx >= len(converted):
                            break
                        gray = converted[src_idx]
                        row_out[out] = gray
                        row_out[out + 1] = gray
                        row_out[out + 2] = gray
                        row_out[out + 3] = 255
                        out += 4
                else:
                    row_base = src_y * width_px * comps
                    for src_x in src_x_map:
                        src_idx = row_base + src_x * comps
                        if src_idx + 2 >= len(converted):
                            break
                        row_out[out] = converted[src_idx]
                        row_out[out + 1] = converted[src_idx + 1]
                        row_out[out + 2] = converted[src_idx + 2]
                        row_out[out + 3] = 255
                        out += 4
                row_bytes = bytes(row_out)
                rect_row_cache[src_y] = row_bytes
            row = py * width * 4 + ix0 * 4
            pixels[row : row + x_span * 4] = row_bytes

    def blit_affine_rows_soft_masked(
        self,
        converted: Any,
        comps: Any,
        width_px: Any,
        height_px: Any,
        alpha: Any,
        soft_mask_data: Any,
        soft_mask_width: Any,
        soft_mask_height: Any,
        p00: Any,
        ux: Any,
        uy: Any,
        vx: Any,
        vy: Any,
        u_from_x: Any,
        u_from_y: Any,
        ix0: Any,
        iy0: Any,
        ix1: Any,
        iy1: Any,
    ) -> bool:
        """Sample a sheared or rotated image that carries a soft mask.

        As :meth:`blit_affine_rows_opaque`, but each sampled pixel is scaled by
        the mask alpha looked up at the same unit-square coordinate, so the
        result is composited rather than written."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        page_x_coordinates = self.page_x_coordinates
        page_y_coordinates = self.page_y_coordinates
        pixel_view = self.pixel_view
        pixels = self.pixels
        target_pixels = pixel_view(pixels)
        span_source_pixels = uint8_view(converted)[: width_px * height_px * comps].reshape(
            height_px,
            width_px,
            comps,
        )
        span_mask_pixels = uint8_view(soft_mask_data)[: soft_mask_width * soft_mask_height].reshape(
            soft_mask_height,
            soft_mask_width,
        )

        def blend_normal_span(
            py: int,
            px0: int,
            src_xs: list[int],
            src_ys: list[int],
            mask_xs: list[int],
            mask_ys: list[int],
        ) -> None:
            if not src_xs:
                return
            src_x = numpy.asarray(src_xs, dtype=numpy.intp)
            src_y = numpy.asarray(src_ys, dtype=numpy.intp)
            mask_x = numpy.asarray(mask_xs, dtype=numpy.intp)
            mask_y = numpy.asarray(mask_ys, dtype=numpy.intp)
            src_rgb = span_source_pixels[src_y, src_x, :3].astype(numpy.float32)
            src_alpha = span_mask_pixels[mask_y, mask_x].astype(numpy.float32)
            if alpha != 255:
                src_alpha = numpy.rint(src_alpha * (alpha / 255.0))
            if not numpy.any(src_alpha > 0):
                return
            dst = target_pixels[py, px0 : px0 + len(src_xs), :].astype(numpy.float32)
            opaque = src_alpha >= 255.0
            if numpy.any(opaque):
                dst[opaque, 0:3] = src_rgb[opaque]
                dst[opaque, 3] = 255.0
            partial = (~opaque) & (src_alpha > 0.0)
            if numpy.any(partial):
                src_a = src_alpha[partial] / 255.0
                dst_a = dst[partial, 3] / 255.0
                out_a = src_a + dst_a * (1.0 - src_a)
                safe_out_a = numpy.where(out_a > 0.0, out_a, 1.0)
                dst_rgb = dst[partial, 0:3]
                out_rgb = (
                    src_rgb[partial] * src_a[:, None]
                    + dst_rgb * dst_a[:, None] * (1.0 - src_a)[:, None]
                ) / safe_out_a[:, None]
                dst[partial, 0:3] = numpy.rint(out_rgb)
                dst[partial, 3] = numpy.rint(out_a * 255.0)
            target_pixels[py, px0 : px0 + len(src_xs), :] = numpy.clip(
                numpy.rint(dst), 0, 255
            ).astype(numpy.uint8)

        if u_from_x:
            inv_ux = 1.0 / ux
            inv_vy = 1.0 / vy
            u = (page_x_coordinates(ix0, ix1) - p00[0]) * inv_ux
            valid_u = (u >= 0.0) & (u <= 1.0)
            source_x_map = numpy.where(
                valid_u,
                numpy.clip((u * width_px).astype(numpy.intp), 0, width_px - 1),
                -1,
            )
            mask_x_map = numpy.where(
                valid_u,
                numpy.clip((u * soft_mask_width).astype(numpy.intp), 0, soft_mask_width - 1),
                -1,
            )
            v = (page_y_coordinates(iy0, iy1) - p00[1]) * inv_vy
            valid_v = (v >= 0.0) & (v <= 1.0)
            source_y_map = numpy.where(
                valid_v,
                numpy.clip(((1.0 - v) * height_px).astype(numpy.intp), 0, height_px - 1),
                -1,
            )
            mask_y_map = numpy.where(
                valid_v,
                numpy.clip(
                    ((1.0 - v) * soft_mask_height).astype(numpy.intp),
                    0,
                    soft_mask_height - 1,
                ),
                -1,
            )
        else:
            inv_uy = 1.0 / uy
            inv_vx = 1.0 / vx
            u = (page_y_coordinates(iy0, iy1) - p00[1]) * inv_uy
            valid_u = (u >= 0.0) & (u <= 1.0)
            source_x_map = numpy.where(
                valid_u,
                numpy.clip((u * width_px).astype(numpy.intp), 0, width_px - 1),
                -1,
            )
            mask_x_map = numpy.where(
                valid_u,
                numpy.clip((u * soft_mask_width).astype(numpy.intp), 0, soft_mask_width - 1),
                -1,
            )
            v = (page_x_coordinates(ix0, ix1) - p00[0]) * inv_vx
            valid_v = (v >= 0.0) & (v <= 1.0)
            source_y_map = numpy.where(
                valid_v,
                numpy.clip(((1.0 - v) * height_px).astype(numpy.intp), 0, height_px - 1),
                -1,
            )
            mask_y_map = numpy.where(
                valid_v,
                numpy.clip(
                    ((1.0 - v) * soft_mask_height).astype(numpy.intp),
                    0,
                    soft_mask_height - 1,
                ),
                -1,
            )
        for dy, py in enumerate(range(iy0, iy1)):
            dy_src_x = source_x_map[dy] if u_from_y else None
            dy_mask_x = mask_x_map[dy] if u_from_y else None
            dy_src_y = source_y_map[dy] if u_from_x else None
            dy_mask_y = mask_y_map[dy] if u_from_x else None
            if u_from_x and (dy_src_y is None or dy_src_y < 0):
                continue
            if u_from_y and (dy_src_x is None or dy_src_x < 0):
                continue
            span_src_x: list[int] = []
            span_src_y: list[int] = []
            span_mask_x: list[int] = []
            span_mask_y: list[int] = []
            span_start: int | None = None
            for dx, px in enumerate(range(ix0, ix1)):
                mapped_src_x = source_x_map[dx] if u_from_x else dy_src_x
                mapped_src_y = source_y_map[dx] if u_from_y else dy_src_y
                mapped_mask_x = mask_x_map[dx] if u_from_x else dy_mask_x
                mapped_mask_y = mask_y_map[dx] if u_from_y else dy_mask_y
                if (
                    mapped_src_x is None
                    or mapped_src_y is None
                    or mapped_src_x < 0
                    or mapped_src_y < 0
                    or mapped_mask_x is None
                    or mapped_mask_y is None
                    or mapped_mask_x < 0
                    or mapped_mask_y < 0
                ):
                    if span_start is not None:
                        blend_normal_span(
                            py,
                            span_start,
                            span_src_x,
                            span_src_y,
                            span_mask_x,
                            span_mask_y,
                        )
                        span_src_x = []
                        span_src_y = []
                        span_mask_x = []
                        span_mask_y = []
                        span_start = None
                    continue
                if span_start is None:
                    span_start = px
                span_src_x.append(mapped_src_x)
                span_src_y.append(mapped_src_y)
                span_mask_x.append(mapped_mask_x)
                span_mask_y.append(mapped_mask_y)
            if span_start is not None:
                blend_normal_span(
                    py,
                    span_start,
                    span_src_x,
                    span_src_y,
                    span_mask_x,
                    span_mask_y,
                )
        return False

    def blit_affine_rows_opaque(
        self,
        converted: Any,
        comps: Any,
        width_px: Any,
        height_px: Any,
        converted_len: Any,
        p00: Any,
        ux: Any,
        uy: Any,
        vx: Any,
        vy: Any,
        u_from_x: Any,
        ix0: Any,
        iy0: Any,
        ix1: Any,
        iy1: Any,
    ) -> bool:
        """Sample a sheared or rotated image into the buffer at full opacity.

        Split out of :meth:`blit_affine_image`. Walks destination rows and maps
        each back through the inverse transform to a source sample, so the
        source is read in whatever order the quad dictates."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_y_coordinates = self.page_y_coordinates
        pixel_view = self.pixel_view
        pixels = self.pixels
        raster_x_coordinate_cache = self.raster_x_coordinate_cache
        raster_y_coordinate_cache = self.raster_y_coordinate_cache
        scale = self.scale
        if u_from_x:
            inv_ux = 1.0 / ux
            inv_vy = 1.0 / vy
            page_x = (
                crop_x0
                + (internal_cached_raster_coordinates(raster_x_coordinate_cache, ix0, ix1) + 0.5)
                / scale
            )
            page_y = (
                crop_y1
                - (internal_cached_raster_coordinates(raster_y_coordinate_cache, iy0, iy1) + 0.5)
                / scale
            )
            source_u = (page_x - p00[0]) * inv_ux
            source_v = (page_y - p00[1]) * inv_vy
            valid_x = (source_u >= 0.0) & (source_u <= 1.0)
            valid_y = (source_v >= 0.0) & (source_v <= 1.0)
            src_x_map = numpy.where(
                valid_x,
                numpy.clip((source_u * width_px).astype(numpy.intp), 0, width_px - 1),
                -1,
            )
            src_y_map = numpy.where(
                valid_y,
                numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp),
                    0,
                    height_px - 1,
                ),
                -1,
            )
            expected_length = width_px * height_px * comps
            if converted_len >= expected_length:
                source_ready = uint8_view(converted)
                if source_ready is not None:
                    source_samples = source_ready[:expected_length].reshape(
                        height_px, width_px, comps
                    )
                    target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
                    target_region[valid_y] = 0
                    sampled = source_samples[
                        src_y_map[valid_y][:, None],
                        numpy.maximum(src_x_map, 0)[None, :],
                    ]
                    valid = valid_y[:, None] & (src_x_map >= 0)[None, :]
                    internal_blit_reshaped_channels(target_region, sampled, valid, comps)
                    return True
            target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
            source_bytes = uint8_view(converted)
            src_y = numpy.maximum(src_y_map, 0)
            src_x = numpy.maximum(src_x_map, 0)
            if comps == 1:
                source_index = src_y[:, None] * width_px + src_x[None, :]
                bounds_ok = source_index < converted_len
            else:
                source_index = (src_y[:, None] * width_px + src_x[None, :]) * comps
                bounds_ok = source_index + 2 < converted_len
            valid = (src_y_map >= 0)[:, None] & (src_x_map >= 0)[None, :] & bounds_ok
            target_region[src_y_map >= 0] = 0
            if converted_len > 0:
                safe_index = numpy.where(valid, source_index, 0)
                internal_blit_indexed_channels(
                    target_region, source_bytes, safe_index, valid, comps
                )
            return True
        inv_uy = 1.0 / uy
        inv_vx = 1.0 / vx
        page_x = (
            crop_x0
            + (internal_cached_raster_coordinates(raster_x_coordinate_cache, ix0, ix1) + 0.5)
            / scale
        )
        source_v = (page_x - p00[0]) * inv_vx
        src_y_map = numpy.where(
            (source_v >= 0.0) & (source_v <= 1.0),
            numpy.clip(
                ((1.0 - source_v) * height_px).astype(numpy.intp),
                0,
                height_px - 1,
            ),
            -1,
        )
        page_y = (
            crop_y1
            - (internal_cached_raster_coordinates(raster_y_coordinate_cache, iy0, iy1) + 0.5)
            / scale
        )
        source_u = (page_y - p00[1]) * inv_uy
        valid_rows = (source_u >= 0.0) & (source_u <= 1.0)
        src_x_rows = numpy.where(
            valid_rows,
            numpy.clip((source_u * width_px).astype(numpy.intp), 0, width_px - 1),
            -1,
        )
        expected_length = width_px * height_px * comps
        if converted_len >= expected_length:
            source_ready = uint8_view(converted)
            if source_ready is not None:
                source_samples = source_ready[:expected_length].reshape(height_px, width_px, comps)
                target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
                target_region[valid_rows] = 0
                sampled = source_samples[
                    numpy.maximum(src_x_rows, 0)[:, None],
                    numpy.maximum(src_y_map, 0)[None, :],
                ]
                valid = valid_rows[:, None] & (src_y_map >= 0)[None, :]
                internal_blit_reshaped_channels(target_region, sampled, valid, comps)
                return True
        target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
        source_bytes = uint8_view(converted)
        row_u = (page_y_coordinates(iy0, iy1) - p00[1]) * inv_uy
        row_valid = (row_u >= 0.0) & (row_u <= 1.0)
        src_x_rows = numpy.where(
            row_valid,
            numpy.clip((row_u * width_px).astype(numpy.intp), 0, width_px - 1),
            0,
        )
        src_y = numpy.maximum(src_y_map, 0)
        if comps == 1:
            source_index = src_y[None, :] * width_px + src_x_rows[:, None]
            bounds_ok = source_index < converted_len
        else:
            source_index = (src_y[None, :] * width_px + src_x_rows[:, None]) * comps
            bounds_ok = source_index + 2 < converted_len
        valid = row_valid[:, None] & (src_y_map >= 0)[None, :] & bounds_ok
        target_region[row_valid] = 0
        if converted_len > 0:
            safe_index = numpy.where(valid, source_index, 0)
            internal_blit_indexed_channels(target_region, source_bytes, safe_index, valid, comps)
        return False


class internal_RasterMetrics:
    """Image-decode and tiled-blit tallies collected while a page rasterizes.

    These are not debug counters: ``tests/benchmarks`` asserts on every field
    (an image is decoded exactly once, tiled affine blitting stays under a 1 MiB
    scratch budget). Keep them wired up through any refactor.
    """

    __slots__ = (
        "image_count",
        "image_decode_seconds",
        "image_blit_seconds",
        "tiled_affine_blit_count",
        "tiled_affine_peak_scratch_bytes",
    )

    def __init__(self) -> None:
        self.image_count = 0
        self.image_decode_seconds = 0.0
        self.image_blit_seconds = 0.0
        self.tiled_affine_blit_count = 0
        self.tiled_affine_peak_scratch_bytes = 0

    def as_metadata(self) -> dict[str, float | int]:
        return {
            "image_count": self.image_count,
            "decode_seconds": self.image_decode_seconds,
            "blit_seconds": self.image_blit_seconds,
            "tiled_affine_blit_count": self.tiled_affine_blit_count,
            "tiled_affine_peak_scratch_bytes": self.tiled_affine_peak_scratch_bytes,
        }


class internal_ClipState:
    """Clip stack, its derived metadata, and the caches that memoize both.

    Lifted out of ``RenderedPage.rasterize``. ``clip_path_stack`` is shared by
    reference with the rasterizer, which pushes and pops it as the content
    stream nests; every mutation must go through ``mark_dirty`` so the cached
    box and the row-span generation counter stay in step.

    Instance attributes are hoisted into locals inside the hot methods, matching
    the convention the content-stream dispatch loop already uses.
    """

    __slots__ = (
        "clip_path_stack",
        "path_bbox_cache",
        "path_rect_cache",
        "path_edge_cache",
        "clip_edge_cache",
        "clip_row_span_cache",
        "clip_visible_row_cache",
        "crop_x0",
        "crop_y1",
        "scale",
        "width",
        "height",
        "metadata_dirty",
        "stack_generation",
        "cached_box",
        "cached_is_rectangular",
        "page_box_to_pixels",
        "page_x_to_pixel_span",
    )

    def __init__(
        self,
        clip_path_stack: list[tuple[CapturedPath, str]],
        *,
        crop_x0: float,
        crop_y1: float,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        self.clip_path_stack = clip_path_stack
        self.path_bbox_cache: dict[int, tuple[float, float, float, float] | None] = {}
        self.path_rect_cache: dict[int, tuple[float, float, float, float] | None] = {}
        self.path_edge_cache: dict[int, list[tuple[float, float, float, float]]] = {}
        self.clip_edge_cache: dict[int, list[tuple[float, float, float, float]]] = {}
        self.clip_row_span_cache: dict[tuple[int, int, str], tuple[tuple[int, int], ...]] = {}
        self.clip_visible_row_cache: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.scale = scale
        self.width = width
        self.height = height
        self.metadata_dirty = True
        self.stack_generation = 0
        self.cached_box: tuple[float, float, float, float] | None = None
        self.page_box_to_pixels, self.page_x_to_pixel_span = internal_make_page_geometry(
            crop_x0, crop_y1, scale, width, height
        )
        self.cached_is_rectangular = True

    def refresh_clip_metadata(self) -> None:
        if not self.metadata_dirty:
            return
        clip: tuple[float, float, float, float] | None = None
        rectangular = True
        axis_aligned_rect_box = self.axis_aligned_rect_box
        path_bbox = self.path_bbox
        for path, internal_rule in self.clip_path_stack:
            rect = axis_aligned_rect_box(path)
            if rect is None:
                rectangular = False
                box = path_bbox(path)
            else:
                box = rect
            if box is None:
                continue
            clip = box if clip is None else internal_intersect_box(clip, box)
            if clip is None:
                break
        self.cached_box = clip
        self.cached_is_rectangular = rectangular
        self.metadata_dirty = False

    def mark_clip_metadata_dirty(self) -> None:
        self.metadata_dirty = True
        self.stack_generation += 1

    def current_clip(self) -> tuple[float, float, float, float] | None:
        self.refresh_clip_metadata()
        return self.cached_box

    def clip_paths_are_axis_aligned_rects(self) -> bool:
        self.refresh_clip_metadata()
        return self.cached_is_rectangular

    def path_bbox(self, path: Any) -> tuple[float, float, float, float] | None:
        if type(path) is not CapturedPath:
            return None
        cache = self.path_bbox_cache
        cache_key = id(path)
        if cache_key in cache:
            return cache[cache_key]
        box = path.bbox()
        cache[cache_key] = box
        return box

    def axis_aligned_rect_box(self, path: CapturedPath) -> tuple[float, float, float, float] | None:
        path_rect_cache = self.path_rect_cache
        cache_key = id(path)
        if cache_key in path_rect_cache:
            return path_rect_cache[cache_key]
        segment_subpaths = [subpath for subpath in path.subpaths if subpath.has_segments()]
        if len(segment_subpaths) != 1 or path.subpaths[-1] is not segment_subpaths[0]:
            path_rect_cache[cache_key] = None
            return None
        subpath = segment_subpaths[0]
        points = list(subpath.points)
        if len(points) >= 2 and points[0] == points[-1]:
            points.pop()
        if len(points) != 4:
            path_rect_cache[cache_key] = None
            return None
        if not subpath.closed and subpath.points[0] != subpath.points[-1]:
            path_rect_cache[cache_key] = None
            return None
        xs = {point[0] for point in points}
        ys = {point[1] for point in points}
        if len(xs) != 2 or len(ys) != 2:
            path_rect_cache[cache_key] = None
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 <= x0 or y1 <= y0:
            path_rect_cache[cache_key] = None
            return None
        corners = {(x0, y0), (x0, y1), (x1, y0), (x1, y1)}
        if set(points) != corners:
            path_rect_cache[cache_key] = None
            return None
        for (px0, py0), (px1, py1) in zip(points, points[1:] + points[:1], strict=False):
            if px0 != px1 and py0 != py1:
                path_rect_cache[cache_key] = None
                return None
        rect = (x0, y0, x1, y1)
        path_rect_cache[cache_key] = rect
        return rect

    def path_edges(self, path: CapturedPath) -> list[tuple[float, float, float, float]]:
        path_edge_cache = self.path_edge_cache
        cache_key = id(path)
        cached = path_edge_cache.get(cache_key)
        if cached is not None:
            return cached
        edges = path.fill_edges()
        path_edge_cache[cache_key] = edges
        return edges

    def clip_path_row_spans(
        self, path: CapturedPath, py: int, fill_rule: str
    ) -> tuple[tuple[int, int], ...]:
        clip_row_span_cache = self.clip_row_span_cache
        cache_key = (id(path), py, fill_rule)
        cached = clip_row_span_cache.get(cache_key)
        if cached is not None:
            return cached
        clip_edge_cache = self.clip_edge_cache
        edges = clip_edge_cache.get(id(path))
        if edges is None:
            edges = self.path_edges(path)
            clip_edge_cache[id(path)] = edges
        if not edges:
            clip_row_span_cache[cache_key] = ()
            return ()
        page_y = self.crop_y1 - (py + 0.5) / self.scale
        crossings: list[tuple[float, int]] = []
        for x0, y0, x1, y1 in edges:
            if y0 == y1:
                continue
            low = y0 if y0 < y1 else y1
            high = y1 if y1 > y0 else y0
            if not (low <= page_y < high):
                continue
            t = (page_y - y0) / (y1 - y0)
            crossings.append((x0 + t * (x1 - x0), 1 if y1 > y0 else -1))
        if not crossings:
            clip_row_span_cache[cache_key] = ()
            return ()
        spans: list[tuple[int, int]] = []
        page_x_to_pixel_span = self.page_x_to_pixel_span
        if fill_rule == "evenodd":
            xs = sorted(x for x, internal_delta in crossings)
            for start_x, end_x in zip(xs[0::2], xs[1::2], strict=False):
                span = page_x_to_pixel_span(start_x, end_x)
                if span is not None:
                    spans.append(span)
        else:
            crossings.sort(key=lambda item: item[0])
            winding = 0
            previous_x: float | None = None
            index = 0
            while index < len(crossings):
                x = crossings[index][0]
                if previous_x is not None and winding != 0 and x > previous_x:
                    span = page_x_to_pixel_span(previous_x, x)
                    if span is not None:
                        spans.append(span)
                delta = 0
                while index < len(crossings) and crossings[index][0] == x:
                    delta += crossings[index][1]
                    index += 1
                winding += delta
                previous_x = x
        cached_spans = tuple(spans)
        clip_row_span_cache[cache_key] = cached_spans
        return cached_spans

    def pixel_in_clip(self, px: int, py: int) -> bool:
        spans = self.clip_row_visible_spans(py)
        if not spans:
            return False
        index = bisect_left(spans, (px + 1, -1))
        if index <= 0:
            return False
        start, end = spans[index - 1]
        return start <= px < end

    def clip_row_visible_spans(self, py: int) -> tuple[tuple[int, int], ...]:
        clip_visible_row_cache = self.clip_visible_row_cache
        cache_key = (self.stack_generation, py)
        cached = clip_visible_row_cache.get(cache_key)
        if cached is not None:
            return cached
        clip_path_stack = self.clip_path_stack
        if not clip_path_stack:
            clip_visible_row_cache[cache_key] = ((0, self.width),)
            return clip_visible_row_cache[cache_key]
        spans: tuple[tuple[int, int], ...] | None = None
        clip_path_row_spans = self.clip_path_row_spans
        for path, fill_rule in clip_path_stack:
            path_spans = clip_path_row_spans(path, py, fill_rule)
            if not path_spans:
                clip_visible_row_cache[cache_key] = ()
                return ()
            if spans is None:
                spans = path_spans
                continue
            left_index = 0
            right_index = 0
            merged: list[tuple[int, int]] = []
            while left_index < len(spans) and right_index < len(path_spans):
                left_start, left_end = spans[left_index]
                right_start, right_end = path_spans[right_index]
                start = max(left_start, right_start)
                end = min(left_end, right_end)
                if end > start:
                    merged.append((start, end))
                if left_end < right_end:
                    left_index += 1
                else:
                    right_index += 1
            spans = tuple(merged)
            if not spans:
                clip_visible_row_cache[cache_key] = ()
                return ()
        result = spans or ()
        clip_visible_row_cache[cache_key] = result
        return result
