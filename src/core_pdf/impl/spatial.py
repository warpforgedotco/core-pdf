# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from heapq import heappop, heappush
from typing import Any, Literal, TypeAlias

import numpy

from core_pdf.impl.geometry import bbox_area
from core_pdf.impl.types import GeneratedRecord, Rectangle

ClusterLinkage: TypeAlias = Literal["chain", "anchor", "mean"]
BandLinkage: TypeAlias = Literal["anchor", "window"]


class BoxIndex(GeneratedRecord):
    boxes: numpy.ndarray[Any, Any]
    areas: numpy.ndarray[Any, Any]

    @classmethod
    def from_boxes(cls, boxes: Iterable[Rectangle], dtype: Any = numpy.float64) -> BoxIndex:
        packed = numpy.asarray(tuple(boxes), dtype=dtype).reshape((-1, 4))
        packed.setflags(write=False)
        return cls.from_array(packed)

    @classmethod
    def from_array(cls, boxes: numpy.ndarray[Any, Any]) -> BoxIndex:
        widths = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        areas = widths * heights
        areas.setflags(write=False)
        return cls(boxes, areas)

    def __len__(self) -> int:
        return len(self.boxes)

    def intersection_areas(self, box: Sequence[float]) -> numpy.ndarray[Any, Any]:
        boxes = self.boxes
        widths = numpy.maximum(
            0.0,
            numpy.minimum(boxes[:, 2], box[2]) - numpy.maximum(boxes[:, 0], box[0]),
        )
        heights = numpy.maximum(
            0.0,
            numpy.minimum(boxes[:, 3], box[3]) - numpy.maximum(boxes[:, 1], box[1]),
        )
        return widths * heights

    def overlap_min(self, box: Rectangle) -> numpy.ndarray[Any, Any]:
        box_area = bbox_area(box)
        denominator = numpy.minimum(self.areas, box_area)
        return numpy.divide(
            self.intersection_areas(box),
            denominator,
            out=numpy.zeros_like(denominator),
            where=denominator > 0.0,
        )

    def overlap_of(
        self,
        box: Rectangle,
        intersections: numpy.ndarray[Any, Any] | None = None,
    ) -> numpy.ndarray[Any, Any]:
        if intersections is None:
            intersections = self.intersection_areas(box)
        box_area = bbox_area(box)
        if box_area > 0.0:
            return intersections / box_area
        return numpy.zeros(len(intersections), dtype=intersections.dtype)

    def matching_overlap_min(self, box: Rectangle, minimum: float) -> numpy.ndarray[Any, Any]:
        return numpy.flatnonzero(self.overlap_min(box) >= minimum)

    def pairwise_intersection(
        self,
        queries: numpy.ndarray[Any, Any],
        rows_per_chunk: int,
    ) -> Iterator[tuple[int, numpy.ndarray[Any, Any]]]:
        boxes = self.boxes
        x0 = boxes[:, 0][None, :]
        y0 = boxes[:, 1][None, :]
        x1 = boxes[:, 2][None, :]
        y1 = boxes[:, 3][None, :]
        for start in range(0, len(queries), rows_per_chunk):
            chunk = queries[start : start + rows_per_chunk]
            widths = numpy.maximum(
                0.0,
                numpy.minimum(chunk[:, None, 2], x1) - numpy.maximum(chunk[:, None, 0], x0),
            )
            heights = numpy.maximum(
                0.0,
                numpy.minimum(chunk[:, None, 3], y1) - numpy.maximum(chunk[:, None, 1], y0),
            )
            numpy.multiply(widths, heights, out=widths)
            yield start, widths


def cluster_1d(
    sorted_values: Sequence[float] | numpy.ndarray[Any, Any],
    gap: float,
    linkage: ClusterLinkage = "chain",
) -> list[list[int]]:
    clusters: list[list[int]] = []
    reference = 0.0
    for position, value in enumerate(sorted_values):
        if clusters:
            if linkage == "chain":
                joins = value - reference <= gap
            elif linkage == "anchor":
                joins = not value - reference > gap
            else:
                joins = abs(value - reference) <= gap
            if joins:
                cluster = clusters[-1]
                if linkage == "chain":
                    reference = value
                elif linkage == "mean":
                    count = len(cluster)
                    reference = (reference * count + value) / (count + 1)
                cluster.append(position)
                continue
        clusters.append([position])
        reference = value
    return clusters


def band_rows(
    centers: Sequence[float],
    tolerance: float,
    order: Iterable[int],
    linkage: BandLinkage = "anchor",
) -> list[list[int]]:
    bands: list[list[int]] = []
    anchor = 0.0
    for item in order:
        center = centers[item]
        if bands:
            if linkage == "anchor":
                joins = not anchor - center > tolerance
            else:
                joins = abs(anchor - center) <= tolerance
            if joins:
                bands[-1].append(item)
                continue
        bands.append([item])
        anchor = center
    return bands


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def interval_overlap_pairs(
    starts: numpy.ndarray[Any, Any], ends: numpy.ndarray[Any, Any]
) -> set[tuple[int, int]]:
    order = numpy.argsort(starts, kind="stable")
    active: set[int] = set()
    ending: list[tuple[float, int]] = []
    pairs: set[tuple[int, int]] = set()
    for raw_index in order:
        index = int(raw_index)
        start = float(starts[index])
        while ending and ending[0][0] <= start:
            _end, expired = heappop(ending)
            active.discard(expired)
        for other in active:
            pairs.add((other, index) if other < index else (index, other))
        active.add(index)
        heappush(ending, (float(ends[index]), index))
    return pairs


__all__ = (
    "BandLinkage",
    "BoxIndex",
    "ClusterLinkage",
    "DisjointSet",
    "band_rows",
    "cluster_1d",
    "interval_overlap_pairs",
)
