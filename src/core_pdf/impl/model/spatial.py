# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, Self

import numpy

from core_pdf.impl.records import Record
from core_pdf.impl.types import Rectangle

frozen_setattr = object.__setattr__


class SpatialFrame(Record):
    __slots__ = ("boxes", "areas")

    boxes: numpy.ndarray[Any, Any]
    areas: numpy.ndarray[Any, Any]

    __fields__: ClassVar[tuple[str, ...]] = ("boxes", "areas")
    __match_args__ = ("boxes", "areas")

    def __init__(self, boxes: numpy.ndarray[Any, Any], areas: numpy.ndarray[Any, Any]) -> None:
        frozen_setattr(self, "boxes", boxes)
        frozen_setattr(self, "areas", areas)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(boxes={self.boxes!r}, areas={self.areas!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.boxes == other.boxes and self.areas == other.areas

    def __hash__(self) -> int:
        return hash((self.boxes, self.areas))

    def __replace__(self, /, **changes: Any) -> Self:
        boxes = changes.pop("boxes", self.boxes)
        areas = changes.pop("areas", self.areas)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(boxes, areas)

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

    def matching_overlap_min(self, box: Rectangle, minimum: float) -> numpy.ndarray[Any, Any]:
        return numpy.flatnonzero(self.overlap_min(box) >= minimum)


__all__ = ("SpatialFrame",)
