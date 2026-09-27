# SPDX-License-Identifier: AGPL-3.0-only

cimport numpy as cnp
from libc.math cimport isnan

from core_pdf_cythonized._ndarray cimport (
    empty_float64,
    empty_float64_rows,
    float64_data,
    is_float64_vector,
    vector_length,
)

import numpy

cnp.import_array()


def outline_edges(double[::1] xs, double[::1] ys, spans):
    cdef Py_ssize_t count = min(xs.shape[0], ys.shape[0])
    if not _spans_inside(spans, count):
        raise IndexError("outline span outside the columns")
    return _outline_edges(&xs[0], &ys[0], spans)


cdef tuple _outline_edges(const double* xs, const double* ys, spans):
    cdef Py_ssize_t start, end, i, total = 0
    cdef bint dropped = False
    cdef bint closes
    cdef list kept = []

    for span in spans:
        start = span[0]
        end = span[1]
        if end > start and xs[start] == xs[end - 1] and ys[start] == ys[end - 1]:
            end -= 1
        if end - start >= 2:
            closes = xs[start] != xs[end - 1] or ys[start] != ys[end - 1]
            kept.append((start, end, closes))
            total += (end - start - 1) + (1 if closes else 0)
        else:
            dropped = True

    if not kept:
        return None, kept, dropped

    edges = empty_float64_rows(total, 4)
    cdef double* out = float64_data(edges)
    for span in kept:
        start = span[0]
        end = span[1]
        closes = span[2]
        for i in range(start, end - 1):
            out[0] = xs[i]
            out[1] = ys[i]
            out[2] = xs[i + 1]
            out[3] = ys[i + 1]
            out += 4
        if closes:
            out[0] = xs[end - 1]
            out[1] = ys[end - 1]
            out[2] = xs[start]
            out[3] = ys[start]
            out += 4
    return edges, kept, dropped


cdef bint _spans_inside(spans, Py_ssize_t count) except -1:
    cdef Py_ssize_t start, end
    for span in spans:
        start = span[0]
        end = span[1]
        if start < 0 or end > count:
            return False
    return True


cdef enum:
    HAS_NAN = 1
    POSITIVE_ZERO = 2
    NEGATIVE_ZERO = 4


cdef int column_range(
    const double* values, Py_ssize_t count, double* low, double* high
) noexcept nogil:
    cdef Py_ssize_t i
    cdef double value, lo = values[0], hi = values[0]
    cdef int flags = 0
    for i in range(count):
        value = values[i]
        if value != value:
            flags |= HAS_NAN
        elif value == 0.0:
            flags |= POSITIVE_ZERO if 1.0 / value > 0.0 else NEGATIVE_ZERO
        if value < lo:
            lo = value
        if value > hi:
            hi = value
    low[0] = lo
    high[0] = hi
    return flags


cdef object column_extreme(
    const double* values, Py_ssize_t count, object column, bint maximum, double bound, int flags
):
    cdef Py_ssize_t i
    if flags & HAS_NAN:
        for i in range(count):
            if values[i] != values[i]:
                return values[i]
    if (flags & POSITIVE_ZERO) and (flags & NEGATIVE_ZERO):
        return float(column.max() if maximum else column.min())
    return bound


def translated_outline_edges(linear_x, linear_y, double e, double f, spans):
    if not (is_float64_vector(linear_x) and is_float64_vector(linear_y)):
        return translated_outline_edges_view(linear_x, linear_y, e, f, spans)
    cdef Py_ssize_t count = vector_length(linear_x)
    if vector_length(linear_y) != count:
        raise ValueError("columns differ in length")
    return _translated_outline_edges(
        float64_data(linear_x), float64_data(linear_y), count, e, f, spans
    )


def translated_outline_edges_view(
    double[::1] linear_x, double[::1] linear_y, double e, double f, spans
):
    cdef Py_ssize_t count = linear_x.shape[0]
    if linear_y.shape[0] != count:
        raise ValueError("columns differ in length")
    return _translated_outline_edges(&linear_x[0], &linear_y[0], count, e, f, spans)


cdef tuple _translated_outline_edges(
    const double* linear_x, const double* linear_y, Py_ssize_t count, double e, double f, spans
):
    cdef Py_ssize_t i
    column_x = empty_float64(count)
    column_y = empty_float64(count)
    cdef double* xs = float64_data(column_x)
    cdef double* ys = float64_data(column_y)
    for i in range(count):
        xs[i] = linear_x[i] + e
        ys[i] = linear_y[i] + f
    if not _spans_inside(spans, count):
        raise IndexError("outline span outside the columns")
    edges, kept, dropped = _outline_edges(xs, ys, spans)
    if edges is None:
        return column_x, column_y, None, kept, dropped, None
    cdef double low_x, high_x, low_y, high_y
    cdef int flags_x, flags_y
    with nogil:
        flags_x = column_range(xs, count, &low_x, &high_x)
        flags_y = column_range(ys, count, &low_y, &high_y)
    bounds = (
        column_extreme(xs, count, column_x, False, low_x, flags_x),
        column_extreme(ys, count, column_y, False, low_y, flags_y),
        column_extreme(xs, count, column_x, True, high_x, flags_x),
        column_extreme(ys, count, column_y, True, high_y, flags_y),
    )
    return column_x, column_y, edges, kept, dropped, bounds
