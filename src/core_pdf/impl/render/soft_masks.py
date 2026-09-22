# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from core_pdf.impl.capture.records import CapturedSoftMask
from core_pdf.impl.render.commands import append_captured_program
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import DisplayItem, ImagePaintItem, PathPaintItem

if TYPE_CHECKING:
    from core_pdf.impl.render.target_state import internal_RasterState

SoftMaskPlane = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
SoftMaskKey = tuple[int, int, tuple[float, float]]
SoftMaskCache = dict[SoftMaskKey, tuple[CapturedSoftMask, SoftMaskPlane | None]]


def internal_graphics_soft_mask(item: DisplayItem) -> CapturedSoftMask | None:
    if isinstance(item, (PathPaintItem, ImagePaintItem)):
        return item.graphics_soft_mask
    mask: CapturedSoftMask | None = item.data.get("graphics_soft_mask")
    return mask


def internal_resolve_soft_mask(
    target: internal_RasterState, mask: CapturedSoftMask
) -> SoftMaskPlane | None:
    key = (id(mask.program), id(mask.transfer), mask.offset)
    cached = target.soft_mask_cache.get(key)
    if cached is not None:
        return cached[1]
    if key in target.active_soft_masks:
        return None
    target.active_soft_masks.add(key)
    result: SoftMaskPlane | None = None
    try:
        nested, view = target.blank_sibling()
        display = DisplayList(target.width, target.height, preserve_object_boundaries=True)
        append_captured_program(display, mask.program, include_text=True)
        nested.paint_items(display.items, translation=mask.offset)
        alpha = view[..., 3]
        if mask.transfer is None:
            result = alpha.astype(numpy.float32) / 255.0
        else:
            samples, inverse = numpy.unique(alpha, return_inverse=True)
            values = [mask.transfer(int(sample) / 255.0)[0] for sample in samples]
            result = numpy.asarray(values, dtype=numpy.float32)[inverse].reshape(alpha.shape)
        result.setflags(write=False)
    except Exception:
        result = None
    finally:
        target.active_soft_masks.remove(key)
    target.soft_mask_cache[key] = (mask, result)
    return result
