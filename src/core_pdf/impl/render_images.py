# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any

import numpy

from core_pdf.impl.array_views import ByteBuffer, UInt8Array, uint8_view
from core_pdf.impl.geometry import points_bbox, rect_tuple
from core_pdf.impl.graphics_images import PreparedImage, prepare_image
from core_pdf.impl.graphics_soft_masks import image_color_key_mask_is_shape
from core_pdf.impl.render_blend import (
    blend_visible_pixels,
    color_rgba,
    declared_blend,
    resolve_constant_alpha,
)
from core_pdf.impl.render_model import ImagePaintItem
from core_pdf.impl.render_paths import intersect_box
from core_pdf.impl.render_resources import RenderResources
from core_pdf.impl.render_strokes import RasterStrokes
from core_pdf_cythonized import box_downsample_blocks, sample_opaque_pixels
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.image_spec import ImageSource


def prepared_image_bytes(prepared: PreparedImage | None) -> int:
    if prepared is None:
        return 0
    soft_mask = prepared.soft_mask
    return prepared.raster.array.nbytes + (0 if soft_mask is None else soft_mask.array.nbytes)


def prepared_image(resources: RenderResources, source: ImageSource) -> PreparedImage | None:
    cache = resources.images
    cached = cache.get(id(source))
    if cached is not None and cached[0] is source:
        return cached[1]
    try:
        prepared = prepare_image(source)
    except Exception:
        prepared = None
    cache.store(id(source), (source, prepared), prepared_image_bytes(prepared))
    return prepared


def image_placement(item: ImagePaintItem) -> tuple[tuple[float, float], ...] | None:
    if item.quad is not None:
        return item.quad
    box = rect_tuple(item.bbox)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


AFFINE_BLIT_SCRATCH_BYTES = 1 << 20


def box_downsample(
    samples: numpy.ndarray[Any, Any],
    source_width: int,
    source_height: int,
    channels: int,
    target_width: int,
    target_height: int,
) -> tuple[numpy.ndarray[Any, Any], int, int]:
    if target_width <= 0 or target_height <= 0:
        return samples, source_width, source_height
    if source_width <= target_width and source_height <= target_height:
        return samples, source_width, source_height
    target_width = min(target_width, source_width)
    target_height = min(target_height, source_height)
    grid = samples.reshape(source_height, source_width, channels)
    row_edges = (numpy.arange(target_height + 1, dtype=numpy.int64) * source_height) // (
        target_height
    )
    column_edges = (numpy.arange(target_width + 1, dtype=numpy.int64) * source_width) // (
        target_width
    )
    if grid.dtype == numpy.uint8:
        reduced = box_downsample_blocks(grid, row_edges, column_edges)
        return reduced.reshape(-1), target_width, target_height
    totals = numpy.add.reduceat(grid, row_edges[:-1], axis=0, dtype=numpy.uint32)
    totals = numpy.add.reduceat(totals, column_edges[:-1], axis=1, dtype=numpy.uint32)
    counts = numpy.diff(row_edges)[:, None, None] * numpy.diff(column_edges)[None, :, None]
    reduced = (totals // numpy.maximum(counts, 1)).astype(numpy.uint8)
    return reduced.reshape(-1), target_width, target_height


def sample_image_plane(
    plane: UInt8Array, u: numpy.ndarray[Any, Any], v: numpy.ndarray[Any, Any]
) -> UInt8Array:
    height, width = plane.shape
    source_x = numpy.clip((u * width).astype(numpy.intp), 0, width - 1)
    source_y = numpy.clip(((1.0 - v) * height).astype(numpy.intp), 0, height - 1)
    return plane[source_y, source_x]


class RasterImages(RasterStrokes):
    __slots__ = ()

    def blit_opaque_sampled_tiles(
        self,
        source_pixels: numpy.ndarray[Any, Any],
        target_region: numpy.ndarray[Any, Any],
        source_y: numpy.ndarray[Any, Any],
        source_x: numpy.ndarray[Any, Any],
        valid_rows: numpy.ndarray[Any, Any],
        valid_columns: numpy.ndarray[Any, Any],
        *,
        transposed: bool = False,
        target_origin: tuple[int, int] = (0, 0),
    ) -> None:
        target_x, target_y = target_origin
        rows = slice(target_y, target_y + target_region.shape[0])
        columns = slice(target_x, target_x + target_region.shape[1])
        all_valid = bool(valid_rows.all() and valid_columns.all())
        sample_opaque_pixels(
            target_region,
            numpy.ascontiguousarray(source_pixels),
            numpy.ascontiguousarray(source_y, dtype=numpy.intp),
            numpy.ascontiguousarray(source_x, dtype=numpy.intp),
            numpy.ascontiguousarray(valid_rows, dtype=numpy.bool_).view(numpy.uint8),
            numpy.ascontiguousarray(valid_columns, dtype=numpy.bool_).view(numpy.uint8),
            transposed,
        )
        if all_valid:
            self.record_source_coverage(rows, columns, 255)
        else:
            self.record_source_coverage(
                rows, columns, 255, visible=valid_rows[:, None] & valid_columns[None, :]
            )

    def blit_affine_image(
        self,
        quad: tuple[tuple[float, float], ...],
        converted: ByteBuffer,
        width_px: int,
        height_px: int,
        comps: int,
        constant_alpha: float | None,
        blend_mode: str | None,
        *,
        source_alpha: UInt8Array | None = None,
        source_shape: UInt8Array | None = None,
        soft_mask: UInt8Array | None = None,
        image_clip: tuple[float, float, float, float] | None = None,
    ) -> bool:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_resolved_mode = self.resolved_blend(blend_mode)
        blit_opaque_sampled_tiles = self.blit_opaque_sampled_tiles
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        current_clip = clip.current_clip
        x_centers = self.grid.x_centers
        y_centers = self.grid.y_centers
        if len(quad) < 3:
            return False
        p00 = quad[0]
        p10 = quad[1]
        p01 = quad[2]
        quad_box = points_bbox(quad)
        if quad_box is None:
            return False
        if image_clip is not None:
            quad_box = intersect_box(quad_box, image_clip)
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
        tracking_shape = self.group_source_shape is not None
        if alpha <= 0 and not tracking_shape:
            return True
        if source_alpha is not None:
            if not numpy.any(source_alpha) and not tracking_shape:
                return True
            if numpy.all(source_alpha == 255):
                source_alpha = None
        if soft_mask is not None:
            if not numpy.any(soft_mask) and not tracking_shape:
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
            page_x = x_centers(ix0, ix1)
            source_u = (page_x - p00[0]) * inv_ux
            source_samples = uint8_view(converted)
            valid_x = (source_u >= 0.0) & (source_u <= 1.0)
            safe_x = numpy.clip(
                (source_u * width_px).astype(numpy.intp),
                0,
                width_px - 1,
            )
            axis_page_y = y_centers(iy0, iy1)
            source_y_array = ((1.0 - (axis_page_y - p00[1]) * inv_vy) * height_px).astype(
                numpy.intp
            )
            valid_y = (axis_page_y - p00[1]) * inv_vy >= 0.0
            valid_y &= (axis_page_y - p00[1]) * inv_vy <= 1.0
            safe_y = numpy.clip(source_y_array, 0, height_px - 1)
            target_region = self.pixel_array[iy0:iy1, ix0:ix1]
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
                target_origin=(ix0, iy0),
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
            target_pixels = self.pixel_array
            source_samples = uint8_view(converted)[: width_px * height_px * comps].reshape(
                height_px, width_px, comps
            )
            if u_from_x:
                inv_ux = 1.0 / ux
                inv_vy = 1.0 / vy
                page_x = x_centers(ix0, ix1)
                page_y = y_centers(iy0, iy1)
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
                page_x = x_centers(ix0, ix1)
                page_y = y_centers(iy0, iy1)
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
                transposed=not u_from_x,
                target_origin=(ix0, iy0),
            )
            return True
        source_pixels = uint8_view(converted)[: width_px * height_px * comps].reshape(
            height_px, width_px, comps
        )
        target_pixels = self.pixel_array
        clip_mask = (
            numpy.frombuffer(self.clip_pixel_mask(ix0, iy0, ix1, iy1), dtype=numpy.bool_).reshape(
                iy1 - iy0, ix1 - ix0
            )
            if clip_regions and not rectangular_clip
            else None
        )
        tile_columns = min(ix1 - ix0, max(1, AFFINE_BLIT_SCRATCH_BYTES // 160))
        tile_rows = max(1, AFFINE_BLIT_SCRATCH_BYTES // (160 * tile_columns))
        for row_start in range(iy0, iy1, tile_rows):
            row_end = min(iy1, row_start + tile_rows)
            page_y = y_centers(row_start, row_end)
            rel_y = page_y[:, None] - p00[1]
            for column_start in range(ix0, ix1, tile_columns):
                column_end = min(ix1, column_start + tile_columns)
                page_x = x_centers(column_start, column_end)
                rel_x = page_x[None, :] - p00[0]
                source_u = (rel_x * vy - rel_y * vx) * inv_det
                source_v = (ux * rel_y - uy * rel_x) * inv_det
                visible = (
                    (source_u >= 0.0) & (source_u <= 1.0) & (source_v >= 0.0) & (source_v <= 1.0)
                )
                if clip_mask is not None:
                    visible &= clip_mask[
                        row_start - iy0 : row_end - iy0, column_start - ix0 : column_end - ix0
                    ]
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
                    self.record_source_coverage(
                        slice(row_start, row_end),
                        slice(column_start, column_end),
                        255,
                        visible=visible,
                    )
                    continue
                alpha_grid = (
                    sample_image_plane(source_alpha, source_u, source_v)
                    if source_alpha is not None
                    else numpy.full(visible.shape, 255, dtype=numpy.uint8)
                )
                if soft_mask is not None:
                    mask_alpha = sample_image_plane(soft_mask, source_u, source_v)
                    alpha_grid = numpy.rint(
                        alpha_grid.astype(numpy.float64) * mask_alpha / 255.0
                    ).astype(numpy.uint8)
                if tracking_shape:
                    shape_grid: int | UInt8Array = (
                        alpha_grid
                        if self.paint_alpha_is_shape
                        else sample_image_plane(source_shape, source_u, source_v)
                        if source_shape is not None
                        else 255
                    )
                    self.record_source_shape(
                        slice(row_start, row_end),
                        slice(column_start, column_end),
                        shape_grid,
                        visible=visible,
                    )
                if constant_alpha is not None:
                    alpha_grid = numpy.clip(
                        numpy.rint(alpha_grid.astype(numpy.float64) * constant_alpha), 0, 255
                    ).astype(numpy.uint8)
                visible &= alpha_grid > 0
                if not numpy.any(visible):
                    continue
                source_colors = numpy.broadcast_to(sampled, (*visible.shape, 3))[visible]
                blend_visible_pixels(
                    target,
                    visible,
                    source_colors[:, 0] / 255.0,
                    source_colors[:, 1] / 255.0,
                    source_colors[:, 2] / 255.0,
                    alpha_grid[visible] / 255.0,
                    blend_resolved_mode,
                    semantic_context=self.semantic_context,
                )
                self.record_source_alpha(
                    slice(row_start, row_end),
                    slice(column_start, column_end),
                    alpha_grid,
                    visible=visible,
                )
        return True

    def blit_image(self, item: ImagePaintItem) -> None:
        quad = image_placement(item)
        if quad is None or item.source is None:
            return
        box = points_bbox(quad)
        if box is None:
            return
        blend_mode = declared_blend(item.blend_mode)
        prepared = prepared_image(self.resources, item.source)
        if prepared is None:
            return
        if prepared.is_stencil:
            self.blit_image_mask(item, prepared, blend_mode)
            return
        raster = prepared.raster
        width_px, height_px = raster.width, raster.height
        components = 1 if raster.color_model == "gray" else 3
        converted = raster.array[:, :, :components].reshape(-1)
        native_soft_mask = prepared.soft_mask
        soft_mask = native_soft_mask.array[:, :, 0] if native_soft_mask is not None else None
        source_alpha: numpy.ndarray[Any, Any] | None = None
        if raster.has_alpha and soft_mask is None:
            source_alpha = raster.array[:, :, components].reshape(-1)
        device_extent = max(
            1,
            int(math.ceil((box[2] - box[0]) * self.scale)),
            int(math.ceil((box[3] - box[1]) * self.scale)),
        )
        if width_px > device_extent or height_px > device_extent:
            reduced, reduced_width, reduced_height = box_downsample(
                converted, width_px, height_px, components, device_extent, device_extent
            )
            if source_alpha is not None:
                source_alpha = box_downsample(
                    source_alpha, width_px, height_px, 1, device_extent, device_extent
                )[0]
            converted = reduced
            width_px, height_px = reduced_width, reduced_height
        if source_alpha is not None:
            source_alpha = source_alpha.reshape(height_px, width_px)
        source_shape = (
            source_alpha if image_color_key_mask_is_shape(item.source.dictionary) else None
        )
        scalar_mask = item.soft_mask_alpha if native_soft_mask is None else None
        opacity = item.fill_opacity
        constant_alpha = (
            resolve_constant_alpha(opacity, scalar_mask)
            if is_pdf_number(opacity) or is_pdf_number(scalar_mask)
            else None
        )
        self.set_shape_alpha(1.0 if constant_alpha is None else constant_alpha)
        self.blit_affine_image(
            quad,
            converted,
            width_px,
            height_px,
            components,
            constant_alpha,
            blend_mode,
            source_alpha=source_alpha,
            source_shape=source_shape,
            soft_mask=soft_mask,
            image_clip=rect_tuple(item.image_clip),
        )

    def blit_image_mask(
        self,
        item: ImagePaintItem,
        prepared: PreparedImage,
        blend_mode: str | None,
    ) -> None:
        quad = image_placement(item)
        raster = prepared.raster
        if quad is None or not raster.has_alpha:
            return
        red, green, blue, alpha = color_rgba(item.fill, item.fill_opacity)
        if is_pdf_number(item.soft_mask_alpha) and prepared.soft_mask is None:
            alpha = max(0, min(255, round(alpha * item.soft_mask_alpha)))
        self.set_shape_alpha(alpha / 255.0)
        if alpha <= 0 and self.group_source_shape is None:
            return
        self.blit_affine_image(
            quad,
            bytes((red, green, blue)),
            1,
            1,
            3,
            alpha / 255.0,
            blend_mode,
            source_alpha=raster.array[:, :, raster.channels - 1],
            source_shape=raster.array[:, :, raster.channels - 1],
            image_clip=rect_tuple(item.image_clip),
        )
