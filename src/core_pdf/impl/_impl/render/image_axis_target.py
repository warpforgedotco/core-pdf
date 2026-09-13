# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare image color and alpha for the common affine raster sampler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl._impl.graphics.images import PreparedImage, prepare_image
from core_pdf.impl._impl.graphics.soft_masks import image_color_key_mask_is_shape
from core_pdf.impl._impl.model.geometry import points_bbox, rect_tuple
from core_pdf.impl._impl.render.blend import internal_color_rgba, internal_constant_alpha
from core_pdf.impl._impl.render.kernels import internal_box_downsample
from core_pdf.impl._impl.render.model import ImagePaintItem
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number

if TYPE_CHECKING:
    from core_pdf.impl._impl.render.target_state import internal_RasterState


def internal_image_placement(item: ImagePaintItem) -> tuple[tuple[float, float], ...] | None:
    """Keep the captured placement; synthesize a unit-square map only if absent."""
    if item.quad is not None:
        return item.quad
    box = rect_tuple(item.bbox)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


class internal_ImageAxisTargetMixin:
    """Image preparation shared by opaque, masked and stencil painting."""

    __slots__ = ()

    def blit_image(self: internal_RasterState, item: ImagePaintItem) -> None:
        quad = internal_image_placement(item)
        if quad is None or item.source is None:
            return
        box = points_bbox(quad)
        if box is None:
            return
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
        try:
            prepared = prepare_image(item.source)
        except Exception:
            prepared = None
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
        # The prepared raster already contains a resized copy of the native
        # mask. Use the native plane in its own coordinates instead of applying
        # that derived alpha a second time or binding it to the color size.
        source_alpha: numpy.ndarray[Any, Any] | None = None
        if raster.has_alpha and soft_mask is None:
            source_alpha = raster.array[:, :, components].reshape(-1)
        # Retain the existing color/embedded-alpha reduction policy. A native
        # soft mask stays at its own resolution and uses the same image UVs.
        device_extent = max(
            1,
            int(math.ceil((box[2] - box[0]) * self.scale)),
            int(math.ceil((box[3] - box[1]) * self.scale)),
        )
        if width_px > device_extent or height_px > device_extent:
            reduced, reduced_width, reduced_height = internal_box_downsample(
                converted, width_px, height_px, components, device_extent, device_extent
            )
            if source_alpha is not None:
                source_alpha = internal_box_downsample(
                    source_alpha, width_px, height_px, 1, device_extent, device_extent
                )[0]
            converted = reduced
            width_px, height_px = reduced_width, reduced_height
        if source_alpha is not None:
            source_alpha = source_alpha.reshape(height_px, width_px)
        source_shape = (
            source_alpha if image_color_key_mask_is_shape(item.source.dictionary) else None
        )
        # The captured mean of a native mask is diagnostic metadata, not an
        # additional paint opacity. Its actual sample is applied by the sampler.
        scalar_mask = item.soft_mask_alpha if native_soft_mask is None else None
        opacity = item.fill_opacity
        constant_alpha = (
            internal_constant_alpha(opacity, scalar_mask)
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
        self: internal_RasterState,
        item: ImagePaintItem,
        prepared: PreparedImage,
        blend_mode: str | None,
    ) -> None:
        """Sample a stencil's marking plane with its original image transform."""
        quad = internal_image_placement(item)
        raster = prepared.raster
        if quad is None or not raster.has_alpha:
            return
        red, green, blue, alpha = internal_color_rgba(item.fill, item.fill_opacity)
        if is_pdf_number(item.soft_mask_alpha) and prepared.soft_mask is None:
            alpha = max(0, min(255, round(alpha * item.soft_mask_alpha)))
        self.set_shape_alpha(alpha / 255.0)
        if alpha <= 0 and self.group_source_shape is None:
            return
        # A stencil has one constant color; only its alpha plane has the image's
        # dimensions. Both are sampled by the same original unit-square map.
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
