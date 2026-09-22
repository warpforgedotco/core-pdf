# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl.graphics.images import PreparedImage, prepare_image
from core_pdf.impl.graphics.soft_masks import image_color_key_mask_is_shape
from core_pdf.impl.model.geometry import points_bbox, rect_tuple
from core_pdf.impl.render.blend import internal_color_rgba, internal_constant_alpha
from core_pdf.impl.render.kernels import internal_box_downsample
from core_pdf.impl.render.model import ImagePaintItem
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.image_spec import ImageSource

if TYPE_CHECKING:
    from core_pdf.impl.render.target_state import internal_RasterState

PREPARED_IMAGE_CACHE_BYTES = 256 << 20


def internal_prepared_image_bytes(prepared: PreparedImage | None) -> int:
    if prepared is None:
        return 0
    soft_mask = prepared.soft_mask
    return prepared.raster.array.nbytes + (0 if soft_mask is None else soft_mask.array.nbytes)


class PreparedImageCache:
    __slots__ = ("budget", "entries", "size")

    def __init__(self, budget: int = PREPARED_IMAGE_CACHE_BYTES) -> None:
        self.budget = budget
        self.entries: dict[int, tuple[ImageSource, PreparedImage | None, int]] = {}
        self.size = 0

    def store(self, source: ImageSource, prepared: PreparedImage | None) -> None:
        size = internal_prepared_image_bytes(prepared)
        if size > self.budget:
            return
        entries = self.entries
        while entries and self.size + size > self.budget:
            oldest = next(iter(entries))
            self.size -= entries.pop(oldest)[2]
        entries[id(source)] = (source, prepared, size)
        self.size += size


def internal_prepared_image(cache: PreparedImageCache, source: ImageSource) -> PreparedImage | None:
    cached = cache.entries.get(id(source))
    if cached is not None and cached[0] is source:
        return cached[1]
    try:
        prepared = prepare_image(source)
    except Exception:
        prepared = None
    cache.store(source, prepared)
    return prepared


def internal_image_placement(item: ImagePaintItem) -> tuple[tuple[float, float], ...] | None:
    if item.quad is not None:
        return item.quad
    box = rect_tuple(item.bbox)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


class internal_ImageAxisTargetMixin:
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
        prepared = internal_prepared_image(self.prepared_image_cache, item.source)
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
