from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy

Point = tuple[float, float]
Contours = tuple[tuple[Point, ...], ...]


def transform_contours(
    contours: Sequence[Sequence[Point]], matrix: Sequence[float], scale: float = 1000.0
) -> Contours:
    """Apply a font matrix to every point, then scale into 1000-unit glyph space."""
    a, b, c, d, e, f = matrix
    return tuple(
        tuple(((x * a + y * c + e) * scale, (x * b + y * d + f) * scale) for x, y in contour)
        for contour in contours
    )


def scale_contours(contours: Sequence[Sequence[Point]], scale: float) -> Contours:
    """Uniformly scale every point of every contour."""
    return tuple(tuple((x * scale, y * scale) for x, y in contour) for contour in contours)


def rasterize_contours(
    contours: Sequence[Sequence[Point]], *, width: int, height: int
) -> tuple[int, ...]:
    points = [point for contour in contours for point in contour]
    if not points or width <= 0 or height <= 0:
        return ()
    point_array = numpy.asarray(points, dtype=numpy.float64)
    xs = point_array[:, 0]
    ys = point_array[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    glyph_width = max(max_x - min_x, 1.0)
    glyph_height = max(max_y - min_y, 1.0)
    edges: list[tuple[float, float, float, float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        normalized = [
            (
                (px - min_x) / glyph_width * (width - 1),
                (py - min_y) / glyph_height * (height - 1),
            )
            for px, py in contour
        ]
        previous = normalized[-1]
        for point in normalized:
            x0, y0 = previous
            x1, y1 = point
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
            previous = point
    if not edges:
        return ()
    edge_array = numpy.asarray(edges, dtype=numpy.float64)
    rows: list[int] = []
    edge_x0, edge_y0, edge_x1, edge_y1 = edge_array.T
    for y in range(height - 1, -1, -1):
        y_mid = y + 0.5
        crosses = (edge_y0 > y_mid) != (edge_y1 > y_mid)
        if not numpy.any(crosses):
            rows.append(0)
            continue
        x0 = edge_x0[crosses]
        y0 = edge_y0[crosses]
        intersections = numpy.sort(
            x0 + (edge_x1[crosses] - x0) * (y_mid - y0) / (edge_y1[crosses] - y0)
        )
        row = 0
        for index in range(0, len(intersections) - 1, 2):
            start_x = max(0, ceil(float(intersections[index]) - 0.5))
            end_x = min(width - 1, ceil(float(intersections[index + 1]) - 0.5) - 1)
            if end_x >= start_x:
                row |= ((1 << (end_x - start_x + 1)) - 1) << start_x
        rows.append(row)
    return tuple(rows)
