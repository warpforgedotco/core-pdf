# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy
import pytest

from core_pdf.impl.spec.s_09_fonts.feature_distance_kernel import feature_distance_matrix
from core_pdf.impl.spec.s_09_fonts.raster_kernel import rasterize_contours


def glyph_shape_contours(
    seed: int, points_per_contour: int = 24
) -> tuple[tuple[tuple[float, float], ...], ...]:
    rng = numpy.random.default_rng(seed)
    angles = numpy.linspace(0.0, 2 * numpy.pi, points_per_contour, endpoint=False)
    outer_radius = 400.0 + rng.uniform(-20, 20, size=points_per_contour)
    inner_radius = outer_radius * 0.4
    outer = tuple(
        (float(400 + r * numpy.cos(a)), float(400 + r * numpy.sin(a)))
        for a, r in zip(angles, outer_radius, strict=True)
    )
    inner = tuple(
        (float(400 + r * numpy.cos(a)), float(400 + r * numpy.sin(a)))
        for a, r in zip(angles, inner_radius, strict=True)
    )
    return (outer, inner)


GLYPH_CONTOURS = glyph_shape_contours(seed=1)


def build_feature_set(
    seed: int, count: int
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[float, ...],
    tuple[int, ...],
]:
    rng = numpy.random.default_rng(seed)
    cells = []
    bitmaps = []
    aspects = []
    contours = []
    for _ in range(count):
        cell_count = int(rng.integers(10, 60))
        xs = rng.integers(0, 18, size=cell_count)
        ys = rng.integers(0, 24, size=cell_count)
        cells.append(tuple(sorted({(int(x), int(y)) for x, y in zip(xs, ys, strict=True)})))
        bitmaps.append(tuple(int(rng.integers(0, 1 << 18)) for _ in range(24)))
        aspects.append(float(rng.uniform(0.4, 1.6)))
        contours.append(int(rng.integers(1, 4)))
    return tuple(cells), tuple(bitmaps), tuple(aspects), tuple(contours)


LEFT_CELLS, LEFT_BITMAPS, LEFT_ASPECTS, LEFT_CONTOURS = build_feature_set(seed=10, count=30)
RIGHT_CELLS, RIGHT_BITMAPS, RIGHT_ASPECTS, RIGHT_CONTOURS = build_feature_set(seed=11, count=30)


@pytest.mark.benchmark_high_impact
def test_rasterize_glyph_contours_benchmark(benchmark) -> None:
    result = benchmark(rasterize_contours, GLYPH_CONTOURS, width=24, height=32)
    assert any(result)


@pytest.mark.benchmark_high_impact
def test_feature_distance_matrix_benchmark(benchmark) -> None:
    result = benchmark(
        feature_distance_matrix,
        LEFT_CELLS,
        LEFT_BITMAPS,
        LEFT_ASPECTS,
        LEFT_CONTOURS,
        RIGHT_CELLS,
        RIGHT_BITMAPS,
        RIGHT_ASPECTS,
        RIGHT_CONTOURS,
    )
    assert result.shape == (30, 30)
