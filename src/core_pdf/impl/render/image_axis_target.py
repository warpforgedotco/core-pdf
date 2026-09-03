# SPDX-License-Identifier: AGPL-3.0-only
"""Axis-aligned decoded-image sampling and mask painting for raster targets."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy

from core_pdf.impl.render.blend import internal_color_rgba
from core_pdf.impl.render.kernels import (
    internal_box_downsample,
    internal_soft_mask_alpha_at,
)
from core_pdf.impl.render.model import ImagePaintItem
from core_pdf.impl.render.paths import internal_intersect_box
from core_pdf.impl.runtime.array_views import (
    nearest_indices,
    uint8_image_view,
    uint8_view,
    unit_sample_positions,
)
from core_pdf.impl.spec.s_08_graphics.image_decode import PreparedImage
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_number


class internal_ImageAxisTargetMixin:
    """Decoded-image dispatch and axis-aligned painting for a raster target."""

    __slots__ = ()

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
        base_index = src_y_map[:, None] * width_px + src_x_map[None, :]
        if comps == 1:
            sample_index = base_index
        else:
            sample_index = base_index * comps
        alpha_grid = numpy.full((y_span, x_span), 255, dtype=numpy.uint8)
        if source_alpha is not None:
            alpha_view = uint8_view(source_alpha)
            alpha_grid = alpha_view[base_index]
        if soft_mask is not None:
            mask_height, mask_width = soft_mask.shape
            mask_x = numpy.clip((x_unit_map * mask_width).astype(numpy.intp), 0, mask_width - 1)
            mask_y = numpy.clip((y_unit_map * mask_height).astype(numpy.intp), 0, mask_height - 1)
            mask_alpha = soft_mask[mask_y[:, None], mask_x[None, :]]
            scaled = numpy.rint(alpha_grid.astype(numpy.float64) * mask_alpha / 255.0)
            alpha_grid = numpy.clip(scaled, 0, 255).astype(numpy.uint8)
        if has_constant_alpha:
            scaled = numpy.rint(alpha_grid.astype(numpy.float64) * constant_alpha)
            alpha_grid = numpy.clip(scaled, 0, 255).astype(numpy.uint8)
        selected = visible & (alpha_grid > 0)
        if selected.any():
            target_region = pixel_view(pixels)[iy0:iy1, ix0:ix1]
            source_rgb = numpy.empty((y_span, x_span, 3), dtype=numpy.uint8)
            if comps == 1:
                gray = source_view[sample_index]
                source_rgb[:, :, 0] = gray
                source_rgb[:, :, 1] = gray
                source_rgb[:, :, 2] = gray
            else:
                source_rgb[:, :, 0] = source_view[sample_index]
                source_rgb[:, :, 1] = source_view[sample_index + 1]
                source_rgb[:, :, 2] = source_view[sample_index + 2]
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
        width_px: Any,
        height_px: Any,
    ) -> None:
        """Copy an opaque axis-aligned image into the target region.

        The fastest path in :meth:`blit_image`: fully opaque, unclipped, no
        blend mode, so NumPy can gather the sampled source region and write it
        directly to the destination."""

        # Captured target state hoisted into locals, as elsewhere in this class.
        pixel_view = self.pixel_view
        pixels = self.pixels
        expected_length = width_px * height_px * comps
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
