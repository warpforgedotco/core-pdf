# SPDX-License-Identifier: AGPL-3.0-only
"""Lightweight page-space spatial indexing."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy

from core_pdf.impl.engine.model.geometry import (
    bbox_area,
    bbox_intersection_area,
    finite_rect,
)
from core_pdf.impl.types import Rectangle

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SpatialHit(Generic[T]):
    item: T
    bbox: Rectangle


class SpatialIndex(Generic[T]):
    """Uniform-grid broad-phase index for immutable page-space rectangles."""

    __slots__ = (
        "internal_bounds",
        "internal_cell_height",
        "internal_cell_width",
        "internal_cells",
        "internal_column_count",
        "internal_entries",
        "internal_bbox_array",
        "internal_overflow",
        "internal_row_count",
    )

    def __init__(
        self,
        entries: Iterable[tuple[T, Sequence[float]]],
        *,
        bounds: Sequence[float] | None = None,
        target_cell_count: int | None = None,
        max_axis_cells: int = 128,
        max_cells_per_item: int = 64,
    ) -> None:
        materialized: list[SpatialHit[T]] = []
        for item, box in entries:
            bbox = finite_rect(box)
            if bbox is not None:
                materialized.append(SpatialHit(item, bbox))
        self.internal_entries = tuple(materialized)
        if not materialized:
            self.internal_bounds = (0.0, 0.0, 1.0, 1.0)
            self.internal_row_count = 1
            self.internal_column_count = 1
            self.internal_cell_width = 1.0
            self.internal_cell_height = 1.0
            self.internal_cells: dict[tuple[int, int], list[int]] = {}
            self.internal_overflow: tuple[int, ...] = ()
            self.internal_bbox_array = numpy.empty((0, 4), dtype=numpy.float64)
            return

        bbox_array = numpy.asarray([hit.bbox for hit in materialized], dtype=numpy.float64)
        resolved_bounds = finite_rect(bounds) if bounds is not None else None
        if resolved_bounds is None:
            resolved_bounds = (
                float(bbox_array[:, 0].min()),
                float(bbox_array[:, 1].min()),
                float(bbox_array[:, 2].max()),
                float(bbox_array[:, 3].max()),
            )
        self.internal_bounds = resolved_bounds
        width = max(1.0, resolved_bounds[2] - resolved_bounds[0])
        height = max(1.0, resolved_bounds[3] - resolved_bounds[1])
        target = max(1, target_cell_count or int(math.sqrt(len(materialized)) * 4))
        columns = max(1, min(max_axis_cells, int(round(math.sqrt(target * width / height)))))
        rows = max(1, min(max_axis_cells, int(math.ceil(target / columns))))
        self.internal_column_count = columns
        self.internal_row_count = rows
        self.internal_cell_width = width / columns
        self.internal_cell_height = height / rows

        cells: dict[tuple[int, int], list[int]] = {}
        overflow: list[int] = []
        bounds_array = numpy.asarray(self.internal_bounds, dtype=numpy.float64)
        x0_array = numpy.floor(
            (bbox_array[:, 0] - bounds_array[0]) / self.internal_cell_width
        ).astype(
            numpy.intp,
            copy=False,
        )
        y0_array = numpy.floor(
            (bbox_array[:, 1] - bounds_array[1]) / self.internal_cell_height
        ).astype(
            numpy.intp,
            copy=False,
        )
        x1_array = numpy.floor(
            (bbox_array[:, 2] - bounds_array[0]) / self.internal_cell_width
        ).astype(
            numpy.intp,
            copy=False,
        )
        y1_array = numpy.floor(
            (bbox_array[:, 3] - bounds_array[1]) / self.internal_cell_height
        ).astype(
            numpy.intp,
            copy=False,
        )
        x0_array = numpy.clip(x0_array, 0, columns - 1)
        y0_array = numpy.clip(y0_array, 0, rows - 1)
        x1_array = numpy.clip(x1_array, 0, columns - 1)
        y1_array = numpy.clip(y1_array, 0, rows - 1)

        x0_list = x0_array.tolist()
        y0_list = y0_array.tolist()
        x1_list = x1_array.tolist()
        y1_list = y1_array.tolist()

        for index in range(len(materialized)):
            x0 = x0_list[index]
            y0 = y0_list[index]
            x1 = x1_list[index]
            y1 = y1_list[index]
            cell_count = (x1 - x0 + 1) * (y1 - y0 + 1)
            if cell_count > max_cells_per_item:
                overflow.append(index)
                continue
            for row in range(y0, y1 + 1):
                for column in range(x0, x1 + 1):
                    cells.setdefault((row, column), []).append(index)
        self.internal_cells = cells
        self.internal_overflow = tuple(overflow)
        self.internal_bbox_array = bbox_array

    @classmethod
    def from_items(
        cls,
        items: Iterable[T],
        bbox: Callable[[T], Sequence[float] | None],
        *,
        bounds: Sequence[float] | None = None,
        target_cell_count: int | None = None,
        max_axis_cells: int = 128,
        max_cells_per_item: int = 64,
    ) -> SpatialIndex[T]:
        return cls(
            ((item, item_box) for item in items if (item_box := bbox(item)) is not None),
            bounds=bounds,
            target_cell_count=target_cell_count,
            max_axis_cells=max_axis_cells,
            max_cells_per_item=max_cells_per_item,
        )

    @classmethod
    def from_boxes(
        cls,
        boxes: Iterable[Sequence[float]],
        *,
        bounds: Sequence[float] | None = None,
        target_cell_count: int | None = None,
        max_axis_cells: int = 128,
        max_cells_per_item: int = 64,
    ) -> SpatialIndex[int]:
        return SpatialIndex[int](
            enumerate(boxes),
            bounds=bounds,
            target_cell_count=target_cell_count,
            max_axis_cells=max_axis_cells,
            max_cells_per_item=max_cells_per_item,
        )

    def __len__(self) -> int:
        return len(self.internal_entries)

    def internal_cell_range(self, box: Sequence[float]) -> tuple[int, int, int, int]:
        bx0, by0, bx1, by1 = self.internal_bounds
        x0 = int(math.floor((float(box[0]) - bx0) / self.internal_cell_width))
        y0 = int(math.floor((float(box[1]) - by0) / self.internal_cell_height))
        x1 = int(math.floor((float(box[2]) - bx0) / self.internal_cell_width))
        y1 = int(math.floor((float(box[3]) - by0) / self.internal_cell_height))
        return (
            max(0, min(self.internal_column_count - 1, x0)),
            max(0, min(self.internal_row_count - 1, y0)),
            max(0, min(self.internal_column_count - 1, x1)),
            max(0, min(self.internal_row_count - 1, y1)),
        )

    def candidate_hits(self, box: Sequence[float]) -> tuple[SpatialHit[T], ...]:
        indexes = self.internal_candidate_indexes(box)
        return tuple(self.internal_entries[index] for index in indexes)

    def internal_candidate_indexes(
        self,
        box: Sequence[float],
        normalized: Rectangle | None = None,
    ) -> list[int]:
        if not self.internal_entries:
            return []
        if normalized is None:
            normalized = finite_rect(box)
        if normalized is None:
            return []
        x0, y0, x1, y1 = self.internal_cell_range(normalized)
        seen: set[int] = set()
        indexes: list[int] = []
        for index in self.internal_overflow:
            seen.add(index)
            indexes.append(index)
        for row in range(y0, y1 + 1):
            for column in range(x0, x1 + 1):
                for index in self.internal_cells.get((row, column), ()):
                    if index not in seen:
                        seen.add(index)
                        indexes.append(index)
        indexes.sort()
        return indexes

    def candidates(self, box: Sequence[float]) -> tuple[T, ...]:
        return tuple(hit.item for hit in self.candidate_hits(box))

    def intersecting_hits(self, box: Sequence[float]) -> tuple[SpatialHit[T], ...]:
        normalized = finite_rect(box)
        indexes = self.internal_candidate_indexes(box, normalized)
        if normalized is None or not indexes:
            return ()
        if len(indexes) <= 64:
            # Small candidate sets are cheaper with plain float comparisons
            # than with ~7 numpy dispatches.
            query_x0, query_y0, query_x1, query_y1 = normalized
            entries = self.internal_entries
            hits: list[SpatialHit[T]] = []
            for index in indexes:
                hit = entries[index]
                hit_x0, hit_y0, hit_x1, hit_y1 = hit.bbox
                if (
                    hit_x0 < query_x1
                    and hit_x1 > query_x0
                    and hit_y0 < query_y1
                    and hit_y1 > query_y0
                ):
                    hits.append(hit)
            return tuple(hits)
        # Avoid double numpy array conversion by creating intp array once.
        np_indexes = numpy.asarray(indexes, dtype=numpy.intp)
        candidates = self.internal_bbox_array[np_indexes]
        query = numpy.asarray(normalized, dtype=numpy.float64)
        overlaps = (
            (candidates[:, 0] < query[2])
            & (candidates[:, 2] > query[0])
            & (candidates[:, 1] < query[3])
            & (candidates[:, 3] > query[1])
        )
        return tuple(self.internal_entries[index] for index in np_indexes[overlaps].tolist())

    def intersecting(self, box: Sequence[float]) -> tuple[T, ...]:
        return tuple(hit.item for hit in self.intersecting_hits(box))

    def __iter__(self) -> Iterator[SpatialHit[T]]:
        return iter(self.internal_entries)


__all__ = (
    "SpatialHit",
    "SpatialIndex",
    "bbox_area",
    "bbox_intersection_area",
)
