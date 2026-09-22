from __future__ import annotations

from collections.abc import Sequence
from math import inf
from typing import Any, TypeAlias

import numpy

FEATURE_GRID_WIDTH = 18
FEATURE_GRID_HEIGHT = 24

FeatureArrays: TypeAlias = tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
]


def internal_cell_distance_map(cells: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    if not cells:
        return ()
    limit = FEATURE_GRID_WIDTH + FEATURE_GRID_HEIGHT
    distances = numpy.full((FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH), limit, dtype=numpy.int64)
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            distances[y, x] = 0
    # The two-pass 4-neighbour chamfer is the exact L1 distance transform, which is
    # separable: one forward and one backward running minimum per axis. `limit` exceeds
    # the largest achievable distance on this grid, so the all-unseeded case still
    # returns `limit` everywhere exactly as the sequential scan did.
    for axis, extent in ((1, FEATURE_GRID_WIDTH), (0, FEATURE_GRID_HEIGHT)):
        offsets = numpy.arange(extent, dtype=numpy.int64)
        if axis == 0:
            offsets = offsets[:, None]
        forward = numpy.minimum.accumulate(distances - offsets, axis=axis) + offsets
        reversed_slice: tuple[Any, ...] = (
            (slice(None, None, -1),)
            if axis == 0
            else (
                slice(None),
                slice(None, None, -1),
            )
        )
        backward = (
            numpy.minimum.accumulate((distances + offsets)[reversed_slice], axis=axis)[
                reversed_slice
            ]
            - offsets
        )
        distances = numpy.minimum(forward, backward)
    return tuple(distances.reshape(-1).tolist())


def internal_average_nearest_distance(
    cells: tuple[tuple[int, int], ...], distance_map: tuple[int, ...]
) -> float:
    total = 0.0
    count = 0
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            total += distance_map[y * FEATURE_GRID_WIDTH + x]
            count += 1
    return total / count if count else inf


def internal_bitmap_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    intersection = 0
    union = 0
    for left_row, right_row in zip(left, right, strict=True):
        intersection += (left_row & right_row).bit_count()
        union += (left_row | right_row).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


def feature_distance(
    left_cells: tuple[tuple[int, int], ...],
    left_bitmap: tuple[int, ...],
    left_aspect: float,
    left_contours: int,
    right_cells: tuple[tuple[int, int], ...],
    right_bitmap: tuple[int, ...],
    right_aspect: float,
    right_contours: int,
) -> float:
    if not left_cells or not right_cells:
        return inf
    left_map = internal_cell_distance_map(left_cells)
    right_map = internal_cell_distance_map(right_cells)
    return (
        internal_average_nearest_distance(left_cells, right_map)
        + internal_average_nearest_distance(right_cells, left_map)
        + internal_bitmap_distance(left_bitmap, right_bitmap) * 0.75
        + abs(left_aspect - right_aspect) * 2.0
        + abs(left_contours - right_contours) * 0.2
    )


def internal_feature_arrays(
    cells: Sequence[tuple[tuple[int, int], ...]],
    bitmaps: Sequence[tuple[int, ...]],
    aspects: Sequence[float],
    contours: Sequence[int],
) -> FeatureArrays:
    count = len(cells)
    masks = numpy.zeros((count, FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH), dtype=numpy.float64)
    distance_maps = numpy.zeros_like(masks)
    valid_counts = numpy.zeros(count, dtype=numpy.float64)
    for index, feature_cells in enumerate(cells):
        valid_cells = tuple(
            (x, y)
            for x, y in feature_cells
            if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT
        )
        if not valid_cells:
            continue
        valid_counts[index] = len(valid_cells)
        for x, y in valid_cells:
            masks[index, y, x] += 1.0
        distance_maps[index] = numpy.asarray(
            internal_cell_distance_map(feature_cells), dtype=numpy.float64
        ).reshape(FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH)

    bitmap_width = max((len(bitmap) for bitmap in bitmaps), default=0)
    bitmap_rows = numpy.zeros((count, bitmap_width), dtype=numpy.uint64)
    for index, bitmap in enumerate(bitmaps):
        if bitmap:
            bitmap_rows[index, : len(bitmap)] = bitmap
    return (
        masks,
        distance_maps,
        valid_counts,
        numpy.asarray(aspects, dtype=numpy.float64),
        numpy.asarray(contours, dtype=numpy.float64),
        bitmap_rows,
    )


def feature_distance_matrix(
    left_cells: Sequence[tuple[tuple[int, int], ...]],
    left_bitmaps: Sequence[tuple[int, ...]],
    left_aspects: Sequence[float],
    left_contours: Sequence[int],
    right_cells: Sequence[tuple[tuple[int, int], ...]],
    right_bitmaps: Sequence[tuple[int, ...]],
    right_aspects: Sequence[float],
    right_contours: Sequence[int],
    *,
    internal_right_arrays: FeatureArrays | None = None,
) -> numpy.ndarray[Any, Any]:
    (
        left_masks,
        left_maps,
        left_counts,
        left_aspects_array,
        left_contours_array,
        left_bitmap_rows,
    ) = internal_feature_arrays(left_cells, left_bitmaps, left_aspects, left_contours)
    if internal_right_arrays is None:
        internal_right_arrays = internal_feature_arrays(
            right_cells, right_bitmaps, right_aspects, right_contours
        )
    (
        right_masks,
        right_maps,
        right_counts,
        right_aspects_array,
        right_contours_array,
        right_bitmap_rows,
    ) = internal_right_arrays

    distance = numpy.full((len(left_cells), len(right_cells)), numpy.inf, dtype=numpy.float64)
    valid = (left_counts[:, None] > 0) & (right_counts[None, :] > 0)
    if not numpy.any(valid):
        return distance

    left_to_right = numpy.einsum("lxy,rxy->lr", left_masks, right_maps)
    left_to_right = numpy.divide(
        left_to_right,
        left_counts[:, None],
        out=numpy.zeros_like(left_to_right),
        where=left_counts[:, None] > 0,
    )
    right_to_left = numpy.einsum("rxy,lxy->rl", right_masks, left_maps).T
    right_to_left = numpy.divide(
        right_to_left,
        right_counts[None, :],
        out=numpy.zeros_like(right_to_left),
        where=right_counts[None, :] > 0,
    )

    if left_bitmap_rows.shape[1] == 0 and right_bitmap_rows.shape[1] == 0:
        bitmap_distance = numpy.zeros_like(distance)
    else:
        bitmap_width = max(left_bitmap_rows.shape[1], right_bitmap_rows.shape[1])
        if left_bitmap_rows.shape[1] != bitmap_width:
            left_bitmap_rows = numpy.pad(
                left_bitmap_rows,
                ((0, 0), (0, bitmap_width - left_bitmap_rows.shape[1])),
            )
        if right_bitmap_rows.shape[1] != bitmap_width:
            right_bitmap_rows = numpy.pad(
                right_bitmap_rows,
                ((0, 0), (0, bitmap_width - right_bitmap_rows.shape[1])),
            )
        intersection = numpy.bitwise_count(
            left_bitmap_rows[:, None, :] & right_bitmap_rows[None, :, :]
        ).sum(axis=2)
        union = numpy.bitwise_count(
            left_bitmap_rows[:, None, :] | right_bitmap_rows[None, :, :]
        ).sum(axis=2)
        same_bitmap_shape = numpy.equal(
            numpy.asarray([len(bitmap) for bitmap in left_bitmaps])[:, None],
            numpy.asarray([len(bitmap) for bitmap in right_bitmaps])[None, :],
        )
        bitmap_ratio = numpy.divide(
            intersection,
            union,
            out=numpy.zeros_like(intersection, dtype=numpy.float64),
            where=union != 0,
        )
        bitmap_distance = numpy.where(
            same_bitmap_shape & (union != 0),
            1.0 - bitmap_ratio,
            0.0,
        )

    combined = (
        left_to_right
        + right_to_left
        + bitmap_distance * 0.75
        + numpy.abs(left_aspects_array[:, None] - right_aspects_array[None, :]) * 2.0
        + numpy.abs(left_contours_array[:, None] - right_contours_array[None, :]) * 0.2
    )
    distance[valid] = combined[valid]
    return distance
