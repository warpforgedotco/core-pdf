# SPDX-License-Identifier: AGPL-3.0-only
"""Affine decoded-image sampling and blitting for raster targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl._impl.model.geometry import points_bbox
from core_pdf.impl._impl.render.blend import internal_blend_channels_f64
from core_pdf.impl._impl.render.kernels import (
    AFFINE_BLIT_SCRATCH_BYTES,
    internal_sample_image_plane,
)
from core_pdf.impl._impl.render.paths import internal_intersect_box
from core_pdf.impl._impl.runtime.array_views import (
    ByteBuffer,
    UInt8Array,
    uint8_view,
)

if TYPE_CHECKING:
    from core_pdf.impl._impl.render.target_state import internal_RasterState


class internal_ImageAffineTargetMixin:
    """Affine decoded-image sampling and blitting for a raster target."""

    __slots__ = ()

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
                    # Gather only this tile; taking rows first copies the full
                    # source width, which can exceed the scratch budget.
                    sampled = source_pixels[
                        source_y[None, column_start:column_end],
                        source_x[row_start:row_end, None],
                        :sampled_channels,
                    ]
                else:
                    sampled = source_pixels[
                        source_y[row_start:row_end, None],
                        source_x[None, column_start:column_end],
                        :sampled_channels,
                    ]
                target_tile = target_region[
                    row_start:row_end,
                    column_start:column_end,
                ]
                if all_valid:
                    target_tile[:, :, 0:3] = sampled
                    target_tile[:, :, 3] = 255
                    del sampled
                    continue
                visible = (
                    valid_rows[row_start:row_end, None]
                    & valid_columns[None, column_start:column_end]
                )
                numpy.copyto(target_tile[:, :, 0:3], sampled, where=visible[:, :, None])
                numpy.copyto(target_tile[:, :, 3], 255, where=visible)
                # Release this tile before allocating the next one.
                del sampled, visible

    def blit_affine_image(
        self: internal_RasterState,
        quad: tuple[tuple[float, float], ...],
        converted: ByteBuffer,
        width_px: int,
        height_px: int,
        comps: int,
        constant_alpha: float | None,
        blend_mode: str | None,
        *,
        source_alpha: UInt8Array | None = None,
        soft_mask: UInt8Array | None = None,
        image_clip: tuple[float, float, float, float] | None = None,
    ) -> bool:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_resolved_mode = self.internal_resolved_blend(blend_mode)
        blit_opaque_sampled_tiles = self.blit_opaque_sampled_tiles
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        pixel_view = self.pixel_view
        pixels = self.pixels
        scale = self.scale
        if len(quad) < 3:
            return False
        p00 = quad[0]
        p10 = quad[1]
        p01 = quad[2]
        quad_box = points_bbox(quad)
        if quad_box is None:
            return False
        if image_clip is not None:
            quad_box = internal_intersect_box(quad_box, image_clip)
            if quad_box is None:
                return True
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
        if alpha <= 0:
            return True
        if source_alpha is not None:
            if not numpy.any(source_alpha):
                return True
            if numpy.all(source_alpha == 255):
                source_alpha = None
        if soft_mask is not None:
            if not numpy.any(soft_mask):
                return True
            if numpy.all(soft_mask == 255):
                soft_mask = None
        can_write_opaque = (
            alpha == 255 and blend_mode is None and source_alpha is None and soft_mask is None
        )
        rect_tolerance = max(abs(ux), abs(vy), 1.0) * 1e-6
        if (
            abs(uy) <= rect_tolerance
            and abs(vx) <= rect_tolerance
            and ux > 0
            and vy > 0
            and alpha == 255
            and blend_mode is None
            and can_write_opaque
            and (not clip_regions or rectangular_clip)
        ):
            inv_ux = 1.0 / ux
            inv_vy = 1.0 / vy
            page_x = crop_x0 + (numpy.arange(ix0, ix1) + 0.5) / scale
            source_u = (page_x - p00[0]) * inv_ux
            source_samples = uint8_view(converted)
            valid_x = (source_u >= 0.0) & (source_u <= 1.0)
            safe_x = numpy.clip(
                (source_u * width_px).astype(numpy.intp),
                0,
                width_px - 1,
            )
            axis_page_y = crop_y1 - (numpy.arange(iy0, iy1) + 0.5) / scale
            source_y_array = ((1.0 - (axis_page_y - p00[1]) * inv_vy) * height_px).astype(
                numpy.intp
            )
            valid_y = (axis_page_y - p00[1]) * inv_vy >= 0.0
            valid_y &= (axis_page_y - p00[1]) * inv_vy <= 1.0
            safe_y = numpy.clip(source_y_array, 0, height_px - 1)
            target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
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
            and (not clip_regions or rectangular_clip)
            and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
        ):
            target_pixels = pixel_view(pixels)
            source_samples = uint8_view(converted)[: width_px * height_px * comps].reshape(
                height_px, width_px, comps
            )
            if u_from_x:
                inv_ux = 1.0 / ux
                inv_vy = 1.0 / vy
                page_x = crop_x0 + (numpy.arange(ix0, ix1) + 0.5) / scale
                page_y = crop_y1 - (numpy.arange(iy0, iy1) + 0.5) / scale
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
                page_x = crop_x0 + (numpy.arange(ix0, ix1) + 0.5) / scale
                page_y = crop_y1 - (numpy.arange(iy0, iy1) + 0.5) / scale
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
        source_pixels = uint8_view(converted)[: width_px * height_px * comps].reshape(
            height_px, width_px, comps
        )
        target_pixels = pixel_view(pixels)
        # All general, translucent and nonrectangular-clipped images share this
        # inverse map. Tiling bounds temporary coordinate and sample arrays.
        tile_columns = min(ix1 - ix0, max(1, AFFINE_BLIT_SCRATCH_BYTES // 160))
        tile_rows = max(1, AFFINE_BLIT_SCRATCH_BYTES // (160 * tile_columns))
        for row_start in range(iy0, iy1, tile_rows):
            row_end = min(iy1, row_start + tile_rows)
            page_y = crop_y1 - (numpy.arange(row_start, row_end) + 0.5) / scale
            rel_y = page_y[:, None] - p00[1]
            for column_start in range(ix0, ix1, tile_columns):
                column_end = min(ix1, column_start + tile_columns)
                page_x = crop_x0 + (numpy.arange(column_start, column_end) + 0.5) / scale
                rel_x = page_x[None, :] - p00[0]
                source_u = (rel_x * vy - rel_y * vx) * inv_det
                source_v = (ux * rel_y - uy * rel_x) * inv_det
                visible = (
                    (source_u >= 0.0) & (source_u <= 1.0) & (source_v >= 0.0) & (source_v <= 1.0)
                )
                if clip_regions and not rectangular_clip:
                    allowed = numpy.zeros(visible.shape, dtype=numpy.bool_)
                    for local_y, py in enumerate(range(row_start, row_end)):
                        for start, end in clip_row_visible_spans(py):
                            start, end = max(start, column_start), min(end, column_end)
                            if end > start:
                                allowed[local_y, start - column_start : end - column_start] = True
                    visible &= allowed
                if not numpy.any(visible):
                    continue
                sample_x = numpy.clip((source_u * width_px).astype(numpy.intp), 0, width_px - 1)
                sample_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp), 0, height_px - 1
                )
                sampled = source_pixels[sample_y, sample_x, : 1 if comps == 1 else 3]
                target = target_pixels[row_start:row_end, column_start:column_end]
                if can_write_opaque:
                    numpy.copyto(target[:, :, :3], sampled, where=visible[:, :, None])
                    numpy.copyto(target[:, :, 3], 255, where=visible)
                    continue
                alpha_grid = (
                    internal_sample_image_plane(source_alpha, source_u, source_v)
                    if source_alpha is not None
                    else numpy.full(visible.shape, 255, dtype=numpy.uint8)
                )
                if soft_mask is not None:
                    mask_alpha = internal_sample_image_plane(soft_mask, source_u, source_v)
                    alpha_grid = numpy.rint(
                        alpha_grid.astype(numpy.float64) * mask_alpha / 255.0
                    ).astype(numpy.uint8)
                if constant_alpha is not None:
                    alpha_grid = numpy.clip(
                        numpy.rint(alpha_grid.astype(numpy.float64) * constant_alpha), 0, 255
                    ).astype(numpy.uint8)
                visible &= alpha_grid > 0
                if not numpy.any(visible):
                    continue
                source_colors = numpy.broadcast_to(sampled, (*visible.shape, 3))[visible]
                destination = target[visible].astype(numpy.float64)
                channels = internal_blend_channels_f64(
                    source_colors[:, 0] / 255.0,
                    source_colors[:, 1] / 255.0,
                    source_colors[:, 2] / 255.0,
                    alpha_grid[visible] / 255.0,
                    destination[:, 0],
                    destination[:, 1],
                    destination[:, 2],
                    destination[:, 3],
                    blend_resolved_mode,
                    semantic_context=self.semantic_context,
                )
                target[visible] = numpy.clip(numpy.column_stack(channels), 0, 255).astype(
                    numpy.uint8
                )
        return True
