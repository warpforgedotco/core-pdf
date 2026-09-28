# SPDX-License-Identifier: AGPL-3.0-only

import numpy
import pytest

from core_pdf_cythonized import merge_collinear_rows

TOLERANCE = 1.5


def python_merge(rows, coordinate, start, end):
    merged = []
    for current in rows.tolist():
        if merged:
            previous = merged[-1]
            if (
                abs(current[coordinate] - previous[coordinate]) <= TOLERANCE
                and current[start] <= previous[end] + TOLERANCE * 2.0
            ):
                previous[start] = min(previous[start], current[start])
                previous[end] = max(previous[end], current[end])
                previous[coordinate] = (previous[coordinate] + current[coordinate]) * 0.5
                continue
        merged.append(current)
    return numpy.asarray(merged, dtype=numpy.float32).reshape((-1, 3))


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize(("coordinate", "start", "end"), [(2, 0, 1), (0, 1, 2)])
def test_kernel_matches_python_merge(seed, coordinate, start, end):
    rng = numpy.random.default_rng(seed)
    count = int(rng.integers(0, 400))
    rows = numpy.empty((count, 3), dtype=numpy.float32)
    rows[:, coordinate] = rng.integers(0, 12, count) * 1.2 + rng.uniform(-0.8, 0.8, count)
    rows[:, start] = rng.uniform(0, 300, count)
    rows[:, end] = rows[:, start] + rng.uniform(0, 40, count)
    if count:
        rows[rng.integers(0, count, 3), coordinate] = numpy.nan
    rows = rows[numpy.lexsort((rows[:, start], rows[:, coordinate]))]
    got = merge_collinear_rows(
        numpy.ascontiguousarray(rows, dtype=numpy.float64),
        coordinate,
        start,
        end,
        TOLERANCE,
        TOLERANCE * 2.0,
    )
    assert got.dtype == numpy.float32
    numpy.testing.assert_array_equal(got, python_merge(rows, coordinate, start, end))


def test_rows_must_have_three_columns():
    with pytest.raises(ValueError, match="three columns"):
        merge_collinear_rows(numpy.zeros((2, 4)), 0, 1, 2, 1.5, 3.0)
