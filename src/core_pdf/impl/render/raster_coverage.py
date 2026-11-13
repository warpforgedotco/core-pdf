# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: where paint lands. The paint window, source coverage, alpha and
shape planes, and the knockout bookkeeping of the current group.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array, uint8_image_view
from core_pdf.impl.base.geometry import clamp01
from core_pdf.impl.capture.records import (
    CapturedPath,
)
from core_pdf.impl.render.blend import (
    declared_blend,
)
from core_pdf.impl.render.model import (
    DisplayItem,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.render.raster_support import (
    EMPTY_PIXEL_BOX,
    PixelBox,
)
lazy from core_pdf_cythonized import (
    accumulate_source_plane,
    fill_glyph_knockout_at,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget


def set_shape_alpha(self: RasterTarget, alpha: float) -> None:
    self.shape_alpha = clamp01(alpha) if self.paint_alpha_is_shape else 1.0


def extend_paint_window(self: RasterTarget, rows: int | slice, columns: int | slice) -> None:
    if self.paint_window is None:
        return
    if isinstance(rows, slice):
        y0, y1, _ = rows.indices(self.height)
        y1 = max(y0, y1)
    else:
        y0, y1 = rows, rows + 1
    if isinstance(columns, slice):
        x0, x1, _ = columns.indices(self.width)
        x1 = max(x0, x1)
    else:
        x0, x1 = columns, columns + 1
    self.extend_paint_box(y0, y1, x0, x1)


def extend_paint_box(self: RasterTarget, y0: int, y1: int, x0: int, x1: int) -> None:
    window = self.paint_window
    if window is None:
        return
    if window.empty:
        window.empty = False
        window.y0 = y0
        window.y1 = y1
        window.x0 = x0
        window.x1 = x1
        return
    if y0 < window.y0:
        window.y0 = y0
    if y1 > window.y1:
        window.y1 = y1
    if x0 < window.x0:
        window.x0 = x0
    if x1 > window.x1:
        window.x1 = x1


def record_source_coverage(
    self: RasterTarget,
    rows: int | slice,
    columns: int | slice,
    alpha: int | UInt8Array,
    *,
    shape: int | UInt8Array = 255,
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
) -> None:
    if self.group_source_alpha is None and self.group_source_shape is None:
        self.extend_paint_window(rows, columns)
        return
    self.record_source_alpha(rows, columns, alpha, visible=visible)
    self.record_source_shape(rows, columns, shape, visible=visible)


def record_source_alpha(
    self: RasterTarget,
    rows: int | slice,
    columns: int | slice,
    alpha: int | UInt8Array,
    *,
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
) -> None:
    plane = self.group_source_alpha
    if plane is None:
        self.extend_paint_window(rows, columns)
    elif not self.accumulate_coverage(plane, rows, columns, alpha, 1.0, visible):
        self.record_plane(plane, rows, columns, alpha / 255.0, visible)


def pixel_view(self: RasterTarget, buffer: bytearray | bytes) -> UInt8Array:
    if buffer is self.page_buffer:
        return self.page_pixels
    return uint8_image_view(buffer, (self.height, self.width, 4))


def record_source_shape(
    self: RasterTarget,
    rows: int | slice,
    columns: int | slice,
    shape: int | UInt8Array,
    *,
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
) -> None:
    plane = self.group_source_shape
    if plane is None:
        self.extend_paint_window(rows, columns)
    elif not self.accumulate_coverage(plane, rows, columns, shape, self.shape_alpha, visible):
        self.record_plane(plane, rows, columns, shape / 255.0 * self.shape_alpha, visible)


def accumulate_coverage(
    self: RasterTarget,
    plane: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    rows: int | slice,
    columns: int | slice,
    coverage: int | UInt8Array,
    scale: float,
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None,
) -> bool:
    if (
        visible is not None
        or not isinstance(coverage, numpy.ndarray)
        or type(coverage) is not numpy.ndarray
        or coverage.dtype != numpy.uint8
        or type(rows) is not slice
        or type(columns) is not slice
        or plane.dtype != numpy.float32
    ):
        return False
    window = plane[rows, columns]
    if window.shape != coverage.shape:
        return False
    self.extend_paint_window(rows, columns)
    accumulate_source_plane(window, coverage, float(scale))
    return True


def record_plane(
    self: RasterTarget,
    plane: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    rows: int | slice,
    columns: int | slice,
    source: float | numpy.ndarray[Any, Any],
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None,
) -> None:
    self.extend_paint_window(rows, columns)
    previous = plane[rows, columns]
    updated = previous + (1.0 - previous) * source
    plane[rows, columns] = updated if visible is None else numpy.where(visible, updated, previous)


def knockout_glyph_fill(self: RasterTarget, item: DisplayItem) -> bool:
    if type(item) is not PathPaintItem or item.paint_kind is not PathPaintKind.FILL:
        return False
    edge_array = item.edge_array
    bbox = item.bbox
    if (
        type(item.path) is not CapturedPath
        or edge_array is None
        or bbox is None
        or item.fill_pattern is not None
        or declared_blend(item.blend_mode) is not None
        or item.fill_rule != "nonzero"
    ):
        return False
    parent = self.buffer_stack[-1]
    if parent.backdrop is None or parent.source_alpha is None or parent.source_shape is None:
        return False
    if item.path.axis_aligned_rect() is not None:
        return False
    # Consecutive glyphs of one style share these objects; fill_rgba reads only them.
    fill = item.fill
    fill_opacity = item.fill_opacity
    soft_mask_alpha = item.soft_mask_alpha
    memo = self.fill_memo
    if (
        memo is not None
        and memo[0] is fill
        and memo[1] is fill_opacity
        and memo[2] is soft_mask_alpha
    ):
        rgba = memo[3]
    else:
        rgba = item.fill_rgba()
        self.fill_memo = (fill, fill_opacity, soft_mask_alpha, rgba)
    if len(edge_array) == 0:
        return True
    clipped = self.clip.clipped_pixel_box(bbox)
    if clipped is None:
        return True
    ix0, iy0, ix1, iy1 = clipped[1]
    if not ((ix1 - ix0) * (iy1 - iy0) < 10_000 and self.clip.clip_paths_are_axis_aligned_rects()):
        return False
    self.shape_alpha = clamp01(rgba[3] / 255.0) if self.paint_alpha_is_shape else 1.0
    backdrop = parent.backdrop
    drawn = fill_glyph_knockout_at(
        edge_array,
        self.crop_x0,
        self.crop_y1,
        self.scale,
        ix0,
        iy0,
        ix1 - ix0,
        iy1 - iy0,
        rgba,
        parent.view,
        self.page_pixels if backdrop is self.page_buffer else self.pixel_view(backdrop),
        parent.source_alpha,
        parent.source_shape,
        self.shape_alpha,
    )
    if drawn is None:
        return True
    boxes = parent.painted_boxes
    if boxes is not None:
        # record_knockout_paint, inlined; a drawn window is never empty.
        if len(boxes) >= self.KNOCKOUT_DISJOINT_LIMIT:
            boxes.clear()
            boxes.append((0, 0, self.width, self.height))
        else:
            boxes.append((ix0, iy0, ix1, iy1))
    self.extend_paint_box(iy0, iy1, ix0, ix1)
    return True


def knockout_group_region(self: RasterTarget, item: DisplayItem) -> PixelBox | None:
    boxes = self.group_member_boxes
    if boxes is None:
        return None
    key = id(item)
    if key not in boxes:
        return None
    bbox = boxes[key]
    clipped = self.clip.clipped_pixel_box(bbox) if bbox is not None else None
    return EMPTY_PIXEL_BOX if clipped is None else clipped[1]


def record_knockout_paint(self: RasterTarget, box: PixelBox) -> None:
    boxes = self.buffer_stack[-1].painted_boxes
    if boxes is None or box == EMPTY_PIXEL_BOX:
        return
    if len(boxes) >= self.KNOCKOUT_DISJOINT_LIMIT:
        boxes.clear()
        boxes.append((0, 0, self.width, self.height))
    else:
        boxes.append(box)
