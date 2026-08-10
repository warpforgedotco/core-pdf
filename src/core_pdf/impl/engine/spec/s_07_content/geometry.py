# SPDX-License-Identifier: AGPL-3.0-only
"""Native content geometry and bbox helpers."""

from __future__ import annotations

from core_pdf.impl.engine.spec.s_08_graphics.matrix import Matrix

BBox = tuple[float, float, float, float]


def transform_bbox(bbox: BBox, matrix: Matrix) -> BBox:
    x0, y0, x1, y1 = bbox
    a, b, c, d, e, f = matrix
    points = (
        (x0 * a + y0 * c + e, x0 * b + y0 * d + f),
        (x1 * a + y0 * c + e, x1 * b + y0 * d + f),
        (x0 * a + y1 * c + e, x0 * b + y1 * d + f),
        (x1 * a + y1 * c + e, x1 * b + y1 * d + f),
    )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def union_bbox(left: BBox | None, right: BBox | None) -> BBox | None:
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


def extend_baseline(left: BBox | None, right: BBox | None) -> BBox | None:
    if left is None:
        return right
    if right is None:
        return left
    return (left[0], left[1], right[2], right[3])


def min_optional_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


__all__ = ("extend_baseline", "min_optional_confidence", "transform_bbox", "union_bbox")
