# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from math import ceil, floor
from typing import Any

import numpy

from core_pdf.impl.types import GeneratedRecord

type FloatArray = numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]


class DeviceGrid(GeneratedRecord):
    crop_x0: float
    crop_y0: float
    crop_y1: float
    scale: float
    width: int
    height: int

    def page_box_to_pixels(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[int, int, int, int] | None:
        width = self.width
        height = self.height
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        scale = self.scale
        ix0 = floor((x0 - crop_x0) * scale)
        ix0 = width if ix0 > width else max(ix0, 0)
        ix1 = ceil((x1 - crop_x0) * scale)
        ix1 = width if ix1 > width else max(ix1, 0)
        iy0 = floor((crop_y1 - y1) * scale)
        iy0 = height if iy0 > height else max(iy0, 0)
        iy1 = ceil((crop_y1 - y0) * scale)
        iy1 = height if iy1 > height else max(iy1, 0)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return ix0, iy0, ix1, iy1

    def x_span(self, start_x: float, end_x: float) -> tuple[int, int] | None:
        if end_x <= start_x:
            return None
        start = math.ceil((start_x - self.crop_x0) * self.scale - 0.5)
        end = math.ceil((end_x - self.crop_x0) * self.scale - 0.5)
        start = max(0, min(self.width, start))
        end = max(0, min(self.width, end))
        if end <= start:
            return None
        return start, end

    def x_centers(self, start: int, stop: int) -> FloatArray:
        return self.crop_x0 + (numpy.arange(start, stop) + 0.5) / self.scale

    def y_centers(self, start: int, stop: int) -> FloatArray:
        return self.crop_y1 - (numpy.arange(start, stop) + 0.5) / self.scale

    def page_box(self) -> tuple[float, float, float, float]:
        return (self.crop_x0, self.crop_y0, self.crop_x0 + self.width / self.scale, self.crop_y1)
