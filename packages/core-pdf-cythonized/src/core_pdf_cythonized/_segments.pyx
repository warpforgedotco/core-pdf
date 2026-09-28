# SPDX-License-Identifier: AGPL-3.0-only

import numpy


def merge_collinear_rows(
    const double[:, ::1] rows,
    Py_ssize_t coordinate,
    Py_ssize_t start,
    Py_ssize_t end,
    double tolerance,
    double reach,
):
    """Merge sorted (n, 3) segment rows in one pass: a row joins the last kept row when
    their coordinates are within tolerance and it starts by that row's end plus reach.
    A join widens the kept extent and averages the coordinates, in double precision.
    Returns the kept rows as a float32 (m, 3) array."""
    cdef Py_ssize_t count = rows.shape[0], index, kept = 0
    if rows.shape[1] != 3:
        raise ValueError("segment rows must have three columns")
    if not (0 <= coordinate < 3 and 0 <= start < 3 and 0 <= end < 3):
        raise ValueError("column indexes must be 0, 1 or 2")
    merged = numpy.empty((count, 3), dtype=numpy.float64)
    cdef double[:, ::1] out = merged
    cdef double value, low, high
    with nogil:
        for index in range(count):
            if kept:
                # abs(), min() and max() as Python computes them on floats.
                value = rows[index, coordinate] - out[kept - 1, coordinate]
                if value < 0.0:
                    value = -value
                if value <= tolerance and rows[index, start] <= out[kept - 1, end] + reach:
                    low = rows[index, start]
                    if low < out[kept - 1, start]:
                        out[kept - 1, start] = low
                    high = rows[index, end]
                    if high > out[kept - 1, end]:
                        out[kept - 1, end] = high
                    out[kept - 1, coordinate] = (
                        out[kept - 1, coordinate] + rows[index, coordinate]
                    ) * 0.5
                    continue
            out[kept, 0] = rows[index, 0]
            out[kept, 1] = rows[index, 1]
            out[kept, 2] = rows[index, 2]
            kept += 1
    return merged[:kept].astype(numpy.float32)
