# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from core_pdf_spec.types import Rectangle


def points_bbox(points: Iterable[tuple[float, float]]) -> Rectangle | None:
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
    x0, y0, x1, y1 = bbox
    a, b, c, d, e, f = matrix
    xs = (x0 * a + y0 * c + e, x1 * a + y0 * c + e, x0 * a + y1 * c + e, x1 * a + y1 * c + e)
    ys = (x0 * b + y0 * d + f, x1 * b + y0 * d + f, x0 * b + y1 * d + f, x1 * b + y1 * d + f)
    return (min(xs), min(ys), max(xs), max(ys))


def unit_square_placement(
    matrix: Sequence[float],
) -> tuple[Rectangle, tuple[tuple[float, float], ...]]:
    a, b, c, d, e, f = matrix
    quad = ((e, f), (a + e, b + f), (c + e, d + f), (a + c + e, b + d + f))
    bbox = points_bbox(quad)
    assert bbox is not None
    return bbox, quad


__all__ = (
    "points_bbox",
    "transform_bbox",
    "unit_square_placement",
)
