# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve captured graphics-state Alpha masks in the current device coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl._impl.capture.records import CapturedSoftMask
from core_pdf.impl._impl.graphics.soft_masks import image_overrides_graphics_soft_mask
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.commands import append_captured_program
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import DisplayItem, ImagePaintItem, PathPaintItem
from core_pdf.impl._impl.runtime.array_views import uint8_image_view

if TYPE_CHECKING:
    from core_pdf.impl._impl.render.target_state import internal_RasterState

SoftMaskPlane = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
SoftMaskKey = tuple[int, int, tuple[float, float]]
SoftMaskCache = dict[SoftMaskKey, tuple[CapturedSoftMask, SoftMaskPlane | None]]


def internal_graphics_soft_mask(item: DisplayItem) -> CapturedSoftMask | None:
    """An image's own mask overrides the graphics mask, ISO 32000-2 11.6.4.3."""
    if (
        isinstance(item, ImagePaintItem)
        and item.source is not None
        and image_overrides_graphics_soft_mask(item.source)
    ):
        return None
    mask = (
        item.graphics_soft_mask
        if isinstance(item, (PathPaintItem, ImagePaintItem))
        else item.data.get("graphics_soft_mask")
    )
    return mask if isinstance(mask, CapturedSoftMask) else None


def internal_resolve_soft_mask(
    target: internal_RasterState, mask: CapturedSoftMask
) -> SoftMaskPlane | None:
    """Rasterize group alpha on transparent pixels, then apply TR everywhere.

    The captured program owns the installation CTM and Form BBox clip. It is
    independent of the destination backdrop and the later paint's clipping
    path. A pattern-cell translation moves the complete mask program too.
    """
    key = (id(mask.program), id(mask.transfer), mask.offset)
    cached = target.soft_mask_cache.get(key)
    if cached is not None:
        return cached[1]
    if key in target.active_soft_masks:
        return None
    target.active_soft_masks.add(key)
    result: SoftMaskPlane | None = None
    try:
        # Imported at execution time because the concrete target owns this
        # helper; mask programs use precisely the same paint/group semantics.
        from core_pdf.impl._impl.render.target import internal_RasterTarget

        pixels = bytearray(target.width * target.height * 4)
        view = uint8_image_view(pixels, (target.height, target.width, 4))
        nested = internal_RasterTarget(
            pixels,
            None,
            clip=internal_ClipState(
                crop_x0=target.crop_x0,
                crop_y1=target.crop_y1,
                scale=target.scale,
                width=target.width,
                height=target.height,
            ),
            width=target.width,
            height=target.height,
            scale=target.scale,
            crop_x0=target.crop_x0,
            crop_y0=target.crop_y0,
            crop_y1=target.crop_y1,
            page_view=view,
            semantic_context=target.semantic_context,
        )
        nested.soft_mask_cache = target.soft_mask_cache
        nested.active_soft_masks = target.active_soft_masks
        display = DisplayList(target.width, target.height, preserve_object_boundaries=True)
        append_captured_program(display, mask.program, include_text=True)
        nested.paint_items(display.items, translation=mask.offset)
        alpha = view[..., 3]
        if mask.transfer is None:
            result = alpha.astype(numpy.float32) / 255.0
        else:
            # Group alpha uses the renderer's byte precision; evaluate only
            # samples actually present, including zero outside the group BBox.
            samples, inverse = numpy.unique(alpha, return_inverse=True)
            values: list[float] = []
            for sample in samples:
                output = mask.transfer(int(sample) / 255.0)
                if len(output) != 1 or not numpy.isfinite(output[0]):
                    raise ValueError("invalid soft-mask transfer output")
                values.append(max(0.0, min(1.0, output[0])))
            result = numpy.asarray(values, dtype=numpy.float32)[inverse].reshape(alpha.shape)
        result.setflags(write=False)
    except Exception:
        # Capture may retain usable surrounding page content even when an
        # optional mask program or its transfer function cannot be evaluated.
        result = None
    finally:
        target.active_soft_masks.remove(key)
    target.soft_mask_cache[key] = (mask, result)
    return result
