# SPDX-License-Identifier: AGPL-3.0-only
"""Generic page-space geometry primitives."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import numpy

from core_pdf.impl.types import Rectangle

T = TypeVar("T")


def internal_float_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


def rect_tuple(value: object) -> Rectangle | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        rect = cast(Sequence[object], value)
        try:
            return (
                internal_float_value(rect[0]),
                internal_float_value(rect[1]),
                internal_float_value(rect[2]),
                internal_float_value(rect[3]),
            )
        except (TypeError, ValueError):
            return None
    x0 = getattr(value, "x0", None)
    y0 = getattr(value, "y0", None)
    x1 = getattr(value, "x1", None)
    y1 = getattr(value, "y1", None)
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    try:
        return (float(x0), float(y0), float(x1), float(y1))
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: Sequence[float]) -> float:
    if type(bbox) is tuple and len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)
    w = float(bbox[2]) - float(bbox[0])
    h = float(bbox[3]) - float(bbox[1])
    return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)


def bbox_intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    if type(left) is tuple and type(right) is tuple and len(left) == 4 and len(right) == 4:
        w = min(left[2], right[2]) - max(left[0], right[0])
        h = min(left[3], right[3]) - max(left[1], right[1])
        return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)
    w = min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))
    h = min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)


def finite_rect(box: object, *, require_positive: bool = True) -> Rectangle | None:
    """Coerce a 4-item box to finite floats, or None if it cannot represent one."""
    try:
        rect = cast("Sequence[Any]", box)
        x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    if require_positive and (x1 <= x0 or y1 <= y0):
        return None
    return (x0, y0, x1, y1)


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
        bx0, by0, _, _ = self.internal_bounds
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


def union_bbox(left: Rectangle | None, right: Rectangle | None) -> Rectangle | None:
    """Smallest rectangle containing both boxes; a missing side yields the other."""
    if left is None:
        return right
    if right is None:
        return left
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def points_bbox(points: Iterable[tuple[float, float]]) -> Rectangle | None:
    """Axis-aligned bounds of a point sequence, or None for no points."""
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for x, y in points:
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    if x0 > x1:
        return None
    return (x0, y0, x1, y1)


def transform_bbox(bbox: Rectangle, matrix: Sequence[float]) -> Rectangle:
    """Axis-aligned bounds of ``bbox`` after the PDF matrix ``(a, b, c, d, e, f)``."""
    x0, y0, x1, y1 = bbox
    a, b, c, d, e, f = matrix
    xs = (x0 * a + y0 * c + e, x1 * a + y0 * c + e, x0 * a + y1 * c + e, x1 * a + y1 * c + e)
    ys = (x0 * b + y0 * d + f, x1 * b + y0 * d + f, x0 * b + y1 * d + f, x1 * b + y1 * d + f)
    return (min(xs), min(ys), max(xs), max(ys))


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap between the intervals ``[a0, a1]`` and ``[b0, b1]``."""
    overlap = min(a1, b1) - max(a0, b0)
    return overlap if overlap > 0.0 else 0.0


def bbox_union(boxes: Iterable[Sequence[float]]) -> Rectangle | None:
    """Return the smallest rectangle containing every box, or None for no boxes."""
    result: Rectangle | None = None
    for box in boxes:
        x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if result is None:
            result = (x0, y0, x1, y1)
        else:
            result = (
                result[0] if result[0] < x0 else x0,
                result[1] if result[1] < y0 else y0,
                result[2] if result[2] > x1 else x1,
                result[3] if result[3] > y1 else y1,
            )
    return result


def bbox_intersects(left: Sequence[float], right: Sequence[float]) -> bool:
    """True if ``left`` and ``right`` overlap with positive area on both axes."""
    return bbox_intersection_area(left, right) > 0.0


def bbox_contains(container: Sequence[float], subject: Sequence[float]) -> bool:
    """True if ``subject`` lies entirely within ``container``."""
    return (
        subject[0] >= container[0]
        and subject[2] <= container[2]
        and subject[1] >= container[1]
        and subject[3] <= container[3]
    )


def overlap_ratio_min(left: Sequence[float], right: Sequence[float]) -> float:
    """Intersection area relative to the smaller box (denominator floored at 1.0)."""
    intersection = bbox_intersection_area(left, right)
    if intersection <= 0.0:
        return 0.0
    return intersection / max(1.0, min(bbox_area(left), bbox_area(right)))


def overlap_ratio_min_exact(left: Sequence[float], right: Sequence[float]) -> float:
    """Intersection area relative to the smaller positive box, without an area floor."""
    smaller_area = min(bbox_area(left), bbox_area(right))
    if smaller_area <= 0.0:
        return 0.0
    return bbox_intersection_area(left, right) / smaller_area


def horizontal_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    """Horizontal intersection relative to the narrower box (denominator floored at 1.0)."""
    intersection = interval_overlap(left[0], left[2], right[0], right[2])
    return intersection / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def overlap_ratio_of(subject: Sequence[float], container: Sequence[float]) -> float:
    """Fraction of ``subject``'s own area that ``container`` covers."""
    subject_area = bbox_area(subject)
    if subject_area <= 0.0:
        return 0.0
    return bbox_intersection_area(subject, container) / subject_area


def page_rotation_matrix(
    rotate: int, page_width: float, page_height: float
) -> tuple[float, float, float, float, float, float]:
    """PDF matrix mapping page space into the frame displayed at ``rotate`` degrees."""
    match rotate % 360:
        case 90:
            return (0.0, -1.0, 1.0, 0.0, 0.0, page_width)
        case 180:
            return (-1.0, 0.0, 0.0, -1.0, page_width, page_height)
        case 270:
            return (0.0, 1.0, -1.0, 0.0, page_height, 0.0)
        case _:
            return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def flip_rect_vertical(rect: Sequence[float], page_height: float) -> Rectangle:
    """Convert between bottom-left- and top-left-origin page coordinates."""
    return (
        float(rect[0]),
        page_height - float(rect[3]),
        float(rect[2]),
        page_height - float(rect[1]),
    )


class RectBox:
    __slots__ = ("x0", "y0", "x1", "y1", "seqno", "fill", "fill_opacity")

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        seqno: int = -1,
        fill: tuple[float, ...] | None = None,
        fill_opacity: float | None = None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.seqno = seqno
        self.fill = fill
        self.fill_opacity = fill_opacity

    def normalize(self) -> RectBox:
        if self.x0 <= self.x1 and self.y0 <= self.y1:
            return self
        return RectBox(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
            seqno=self.seqno,
            fill=self.fill,
            fill_opacity=self.fill_opacity,
        )

    def __iter__(self) -> Iterator[float]:
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> float:
        return (self.x0, self.y0, self.x1, self.y1)[index]
