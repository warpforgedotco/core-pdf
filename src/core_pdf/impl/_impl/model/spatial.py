# SPDX-License-Identifier: AGPL-3.0-only
"""Vectorized spatial queries over immutable rectangle collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy

from core_pdf.impl.types import Rectangle


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """Packed rectangles with canonical intersection and overlap queries."""

    boxes: numpy.ndarray[Any, Any]
    areas: numpy.ndarray[Any, Any]

    @classmethod
    def from_boxes(cls, boxes: Iterable[Rectangle]) -> SpatialFrame:
        packed = numpy.asarray(tuple(boxes), dtype=numpy.float64).reshape((-1, 4))
        widths = numpy.maximum(0.0, packed[:, 2] - packed[:, 0])
        heights = numpy.maximum(0.0, packed[:, 3] - packed[:, 1])
        packed.setflags(write=False)
        areas = widths * heights
        areas.setflags(write=False)
        return cls(packed, areas)

    def intersection_areas(self, box: Rectangle) -> numpy.ndarray[Any, Any]:
        widths = numpy.maximum(
            0.0,
            numpy.minimum(self.boxes[:, 2], box[2]) - numpy.maximum(self.boxes[:, 0], box[0]),
        )
        heights = numpy.maximum(
            0.0,
            numpy.minimum(self.boxes[:, 3], box[3]) - numpy.maximum(self.boxes[:, 1], box[1]),
        )
        return widths * heights

    def overlap_min(self, box: Rectangle) -> numpy.ndarray[Any, Any]:
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        denominator = numpy.minimum(self.areas, box_area)
        return numpy.divide(
            self.intersection_areas(box),
            denominator,
            out=numpy.zeros_like(denominator),
            where=denominator > 0.0,
        )

    def overlap_of_query(self, box: Rectangle) -> numpy.ndarray[Any, Any]:
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if box_area <= 0.0:
            return numpy.zeros(len(self.boxes), dtype=numpy.float64)
        return self.intersection_areas(box) / box_area

    def matching_overlap_min(self, box: Rectangle, minimum: float) -> numpy.ndarray[Any, Any]:
        return numpy.flatnonzero(self.overlap_min(box) >= minimum)


__all__ = ("SpatialFrame",)
