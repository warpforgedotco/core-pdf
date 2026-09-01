# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful decoded-image painting operations for raster targets."""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from typing import Any

import numpy

from core_pdf.impl.model.geometry import points_bbox
from core_pdf.impl.render.blend import (
    internal_blend_normal_masked_array_numpy,
    internal_color_rgba,
)
from core_pdf.impl.render.images import (
    AFFINE_BLIT_SCRATCH_BYTES,
    internal_blit_indexed_channels,
    internal_blit_reshaped_channels,
    internal_box_downsample,
    internal_soft_mask_alpha_at,
)
from core_pdf.impl.render.kernels import (
    RASTER_COORDINATE_CACHE_MAX_ENTRIES,
    internal_cached_raster_coordinates,
)
from core_pdf.impl.render.model import ImagePaintItem
from core_pdf.impl.render.paths import internal_intersect_box
from core_pdf.impl.runtime.array_views import (
    ByteBuffer,
    nearest_indices,
    uint8_image_view,
    uint8_view,
    unit_sample_positions,
)
from core_pdf.impl.spec.s_08_graphics.image_decode import PreparedImage
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_number


class internal_ImageTargetMixin:
    """Decoded-image sampling and blitting operations for a raster target."""

    __slots__ = ()

    def cached_page_coordinates(
        self: Any,
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

    def page_x_coordinates(self: Any, start: int, stop: int) -> numpy.ndarray[Any, Any]:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        cached_page_coordinates = self.cached_page_coordinates
        crop_x0 = self.crop_x0
        page_x_coordinate_cache = self.page_x_coordinate_cache
        return cached_page_coordinates(page_x_coordinate_cache, start, stop, crop_x0, 1.0)

    def page_y_coordinates(self: Any, start: int, stop: int) -> numpy.ndarray[Any, Any]:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        cached_page_coordinates = self.cached_page_coordinates
        crop_y1 = self.crop_y1
        page_y_coordinate_cache = self.page_y_coordinate_cache
        return cached_page_coordinates(page_y_coordinate_cache, start, stop, crop_y1, -1.0)

    def blit_opaque_sampled_tiles(
        self: Any,
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
        self: Any,
        quad: tuple[tuple[float, float], ...],
        converted: ByteBuffer,
        width_px: int,
        height_px: int,
        comps: int,
        soft_mask: numpy.ndarray[Any, Any] | None,
        constant_alpha: float | None,
        blend_mode: str | None,
    ) -> bool:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip = self.clip
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
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
        quad_box = points_bbox(quad)
        if quad_box is None:
            return False
        x0, y0, x1, y1 = quad_box
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
        alpha = 255
        if constant_alpha is not None:
            alpha = max(0, min(255, int(round(alpha * constant_alpha))))
        if soft_mask is None:
            soft_mask_data = None
            soft_mask_width = 0
            soft_mask_height = 0
            soft_mask_len = 0
        else:
            soft_mask_height, soft_mask_width = soft_mask.shape
            soft_mask_data = soft_mask.reshape(-1)
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
                    blend_px(row + px * 4, rgba, blend_alpha_scale, blend_resolved_mode)
        return True

    def blit_image(
        self: Any,
        item: ImagePaintItem,
    ) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clip = self.clip
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
        box = item.bbox
        if box is None:
            return
        source = item.source
        if source is None:
            return
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        decode_started = time.perf_counter()
        try:
            prepared = source.prepare()
        except Exception:
            prepared = None
        raster_metrics.image_decode_seconds += time.perf_counter() - decode_started
        if prepared is None:
            return
        if prepared.is_stencil:
            self.blit_image_mask(item, prepared, blend_mode)
            return
        shared_raster = prepared.raster
        width_px = shared_raster.width
        height_px = shared_raster.height
        source_channels = 1 if shared_raster.color_model == "gray" else 3
        converted = shared_raster.array[:, :, :source_channels].reshape(-1)
        source_alpha: numpy.ndarray[Any, Any] | None = None
        if shared_raster.has_alpha:
            source_alpha = shared_raster.array[:, :, source_channels].reshape(-1)
        native_soft_mask = prepared.soft_mask
        soft_mask = native_soft_mask.array[:, :, 0] if native_soft_mask is not None else None
        if source_alpha is not None:
            alpha_view = uint8_view(source_alpha)
            expected_alpha = width_px * height_px
            if len(alpha_view) >= expected_alpha:
                alpha_view = alpha_view[:expected_alpha]
                if not numpy.any(alpha_view) and soft_mask is None:
                    return
                if numpy.all(alpha_view == 255) and soft_mask is None:
                    source_alpha = None
        # Average before sampling when the image is being shrunk. Both blit
        # paths below resample with nearest-neighbour, which at a 4x reduction
        # keeps about one source pixel in sixteen -- enough to drop the thin
        # rules and letter stems of a scanned page. This sits ahead of the
        # affine dispatch so the rotated path gets an averaged source too.
        #
        # Both source axes are held to the *larger* device extent rather than
        # matched to width and height separately: a quad may rotate the image,
        # in which case its height runs along the box's width, and reducing per
        # axis would shrink the wrong one. Taking the maximum can only reduce
        # less than strictly necessary, never more.
        scale = self.scale
        device_extent = max(
            1,
            int(math.ceil((box[2] - box[0]) * scale)),
            int(math.ceil((box[3] - box[1]) * scale)),
        )
        target_width = device_extent
        target_height = device_extent
        if width_px > target_width or height_px > target_height:
            source_samples = (
                converted
                if isinstance(converted, numpy.ndarray)
                else numpy.frombuffer(converted, dtype=numpy.uint8)
            )
            pixel_total = width_px * height_px
            alpha_samples = (
                numpy.asarray(source_alpha, dtype=numpy.uint8) if source_alpha is not None else None
            )
            # Colour and alpha reduce together or not at all. A decoded alpha
            # plane that does not match Width x Height cannot be reduced with
            # them, and reducing only the colour would leave the alpha indexed
            # by the old width -- every sample below reads the wrong offset.
            reducible = alpha_samples is None or alpha_samples.size == pixel_total
            if pixel_total and reducible and source_samples.size % pixel_total == 0:
                reduced, reduced_width, reduced_height = internal_box_downsample(
                    source_samples,
                    width_px,
                    height_px,
                    source_samples.size // pixel_total,
                    target_width,
                    target_height,
                )
                if reduced_width != width_px or reduced_height != height_px:
                    if alpha_samples is not None:
                        source_alpha = internal_box_downsample(
                            alpha_samples,
                            width_px,
                            height_px,
                            1,
                            target_width,
                            target_height,
                        )[0]
                    converted = reduced
                    width_px = reduced_width
                    height_px = reduced_height
        soft_mask_alpha = item.soft_mask_alpha
        constant_alpha_value = float(soft_mask_alpha) if pdf_number(soft_mask_alpha) else None
        quad = item.quad
        comps = source_channels
        raster_metrics.image_count += 1
        if quad is not None and source_alpha is None:
            blit_started = time.perf_counter()
            affine_blit = blit_affine_image(
                quad,
                converted,
                width_px,
                height_px,
                comps,
                soft_mask,
                constant_alpha_value,
                blend_mode,
            )
            raster_metrics.image_blit_seconds += time.perf_counter() - blit_started
            if affine_blit:
                return
        # Preserve the established sampling choice for soft-masked images.
        # The shared raster's same-size alpha plane makes those images bypass
        # affine dispatch; the native-resolution mask then replaces that plane
        # in the axis-aligned path below.  Clearing it before affine dispatch
        # changes which sampler paints the image and therefore its raster.
        if soft_mask is not None:
            source_alpha = None
        x0, y0, x1, y1 = box
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
        x_span = max(1, ix1 - ix0)
        y_span = max(1, iy1 - iy0)
        src_x_map = nearest_indices(x_span, width_px)
        src_y_map = nearest_indices(y_span, height_px)
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
                    blend_px(row + px * 4, rgba, blend_alpha_scale, blend_resolved_mode)

    def record_image_timings(self: Any) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        page = self.page
        raster_metrics = self.raster_metrics
        page.metadata["__core_pdf_raster_image_timings__"] = raster_metrics.as_metadata()

    def blit_image_mask(
        self: Any,
        item: ImagePaintItem,
        prepared: PreparedImage,
        blend_mode: str | None,
    ) -> None:
        """Paint a stencil-mask image: 1 bit per sample selecting the fill colour.

        Split out of :meth:`blit_image`, which dispatches to it before any
        decode work — an ImageMask carries no colour samples of its own.

        PDF 8.9.6.2: the set samples are painted in the fill colour that was
        current when the image was drawn. Capture records that colour on the
        drawing for stencil masks only; when it is absent the PDF default of
        black applies, which is what every mask in the corpus resolves to."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        blend_normal_pixel = self.blend_normal_pixel
        blend_px = self.blend_px
        blend_alpha_scale, blend_resolved_mode = self.internal_resolved_blend(blend_mode)
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
        stencil_red, stencil_green, stencil_blue, stencil_alpha = internal_color_rgba(
            item.fill,
            item.fill_opacity,
        )
        raster = prepared.raster
        width_px = raster.width
        height_px = raster.height
        if not raster.has_alpha:
            return
        mask = raster.array[:, :, raster.channels - 1].reshape(-1)
        box = item.bbox
        if box is None or len(mask) == 0:
            return
        x0, y0, x1, y1 = box
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
        x_span = max(1, ix1 - ix0)
        y_span = max(1, iy1 - iy0)
        src_x_map = nearest_indices(x_span, width_px)
        src_y_map = nearest_indices(y_span, height_px)
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
                target_pixels = pixel_view(pixels)
                target_region = target_pixels[iy0:iy1, ix0:ix1]
                visible = sampled_mask != 0
                target_region[visible, :3] = (stencil_red, stencil_green, stencil_blue)
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
                    alpha = mask[src_idx]
                    if alpha:
                        idx = row + px * 4
                        pixels[idx] = stencil_red
                        pixels[idx + 1] = stencil_green
                        pixels[idx + 2] = stencil_blue
                        pixels[idx + 3] = alpha
            return
        normal_fast = can_blend_normal_fast(blend_mode)
        for dy, py in enumerate(range(iy0, iy1)):
            src_y = src_y_map[dy]
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            # Same per-row span walk as paint_shading: the bisect re-answered a
            # per-row question once per pixel.
            for span_start, span_end in visible_spans:
                for px in range(max(ix0, span_start), min(ix1, span_end)):
                    src_x = src_x_map[px - ix0]
                    src_idx = src_y * width_px + src_x
                    if src_idx >= len(mask):
                        continue
                    alpha = mask[src_idx]
                    if normal_fast:
                        blend_normal_pixel(
                            row + px * 4, stencil_red, stencil_green, stencil_blue, alpha
                        )
                    else:
                        blend_px(
                            row + px * 4,
                            (stencil_red, stencil_green, stencil_blue, alpha),
                            blend_alpha_scale,
                            blend_resolved_mode,
                        )
        return

    def blit_image_rows_blended(
        self: Any,
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
            mask_height, mask_width = soft_mask.shape
            mask_x = numpy.clip((x_unit_map * mask_width).astype(numpy.intp), 0, mask_width - 1)
            mask_y = numpy.clip((y_unit_map * mask_height).astype(numpy.intp), 0, mask_height - 1)
            mask_alpha = soft_mask[mask_y[:, None], mask_x[None, :]]
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
        self: Any,
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
        self: Any,
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

        # The two orientations differ only in which page axis feeds u and v;
        # everything downstream of that selection is shared.
        if u_from_x:
            u_coordinates = page_x_coordinates(ix0, ix1) - p00[0]
            v_coordinates = page_y_coordinates(iy0, iy1) - p00[1]
            inv_u = 1.0 / ux
            inv_v = 1.0 / vy
        else:
            u_coordinates = page_y_coordinates(iy0, iy1) - p00[1]
            v_coordinates = page_x_coordinates(ix0, ix1) - p00[0]
            inv_u = 1.0 / uy
            inv_v = 1.0 / vx
        u = u_coordinates * inv_u
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
        v = v_coordinates * inv_v
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
        self: Any,
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
        src_x_rows = numpy.maximum(src_x_rows, 0)
        src_y = numpy.maximum(src_y_map, 0)
        if comps == 1:
            source_index = src_y[None, :] * width_px + src_x_rows[:, None]
            bounds_ok = source_index < converted_len
        else:
            source_index = (src_y[None, :] * width_px + src_x_rows[:, None]) * comps
            bounds_ok = source_index + 2 < converted_len
        valid = valid_rows[:, None] & (src_y_map >= 0)[None, :] & bounds_ok
        target_region[valid_rows] = 0
        if converted_len > 0:
            safe_index = numpy.where(valid, source_index, 0)
            internal_blit_indexed_channels(target_region, source_bytes, safe_index, valid, comps)
        return False
