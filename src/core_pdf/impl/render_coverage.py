# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.array_views import UInt8Array, uint8_image_view
from core_pdf.impl.render_raster_state import RasterState
from core_pdf.impl.scalars import clamp01
from core_pdf_cythonized import accumulate_source_plane


class RasterCoverage(RasterState):
    __slots__ = ()

    def set_shape_alpha(self, alpha: float) -> None:
        self.shape_alpha = clamp01(alpha) if self.paint_alpha_is_shape else 1.0

    def extend_paint_window(self, rows: int | slice, columns: int | slice) -> None:
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

    def extend_paint_box(self, y0: int, y1: int, x0: int, x1: int) -> None:
        window = self.paint_window
        if window is None:
            return
        if window:
            if y0 < window[0]:
                window[0] = y0
            if y1 > window[1]:
                window[1] = y1
            if x0 < window[2]:
                window[2] = x0
            if x1 > window[3]:
                window[3] = x1
        else:
            window[:] = (y0, y1, x0, x1)

    def record_source_coverage(
        self,
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
        self,
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

    def pixel_view(self, buffer: bytearray | bytes) -> UInt8Array:
        if buffer is self.page_buffer:
            return self.page_pixels
        return uint8_image_view(buffer, (self.height, self.width, 4))

    def record_source_shape(
        self,
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
        self,
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
        self,
        plane: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
        rows: int | slice,
        columns: int | slice,
        source: float | numpy.ndarray[Any, Any],
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None,
    ) -> None:
        self.extend_paint_window(rows, columns)
        previous = plane[rows, columns]
        updated = previous + (1.0 - previous) * source
        plane[rows, columns] = (
            updated if visible is None else numpy.where(visible, updated, previous)
        )
