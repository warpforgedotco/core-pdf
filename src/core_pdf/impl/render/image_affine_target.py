# SPDX-License-Identifier: AGPL-3.0-only
"""Affine decoded-image sampling and blitting for raster targets."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy

from core_pdf.impl.model.geometry import points_bbox
from core_pdf.impl.render.kernels import (
    AFFINE_BLIT_SCRATCH_BYTES,
    RASTER_COORDINATE_CACHE_MAX_ENTRIES,
    internal_cached_raster_coordinates,
)
from core_pdf.impl.runtime.array_views import (
    ByteBuffer,
    uint8_view,
)


class internal_ImageAffineTargetMixin:
    """Affine decoded-image sampling and blitting for a raster target."""

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
        constant_alpha: float | None,
        blend_mode: str | None,
    ) -> bool:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
        clipped_pixel_box = self.clipped_pixel_box
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
        p00 = quad[0]
        p10 = quad[1]
        p01 = quad[2]
        quad_box = points_bbox(quad)
        if quad_box is None:
            return False
        rectangular_clip = current_clip() is not None and clip_paths_are_axis_aligned_rects()
        clipped_box = clipped_pixel_box(quad_box)
        if clipped_box is None:
            return True
        ix0, iy0, ix1, iy1 = clipped_box[1]
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
        can_write_opaque = alpha == 255 and blend_mode is None and not buffer_stack[-1][1]
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
                if alpha != 255:
                    rgba = (
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        alpha,
                    )
                if normal_fast:
                    blend_normal_pixel(row + px * 4, rgba[0], rgba[1], rgba[2], rgba[3])
                else:
                    blend_px(row + px * 4, rgba, blend_alpha_scale, blend_resolved_mode)
        return True
