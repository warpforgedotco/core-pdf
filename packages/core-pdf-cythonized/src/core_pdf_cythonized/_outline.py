# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports import numpy as cnp
from cython.cimports.core_pdf_cythonized._ndarray import (
    empty_float64,
    empty_float64_rows,
    float64_data,
    is_float64_vector,
    vector_length,
)
from cython.cimports.core_pdf_cythonized._outline import HAS_NAN, NEGATIVE_ZERO, POSITIVE_ZERO

cnp.import_array()


def outline_edges(xs: cython.double[::1], ys: cython.double[::1], spans):
    count: cython.Py_ssize_t = min(xs.shape[0], ys.shape[0])
    if not _spans_inside(spans, count):
        raise IndexError("outline span outside the columns")
    return _outline_edges(cython.address(xs[0]), cython.address(ys[0]), spans)


@cython.cfunc
def _outline_edges(
    xs: cython.p_const_double,
    ys: cython.p_const_double,
    spans,
) -> tuple:
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    i: cython.Py_ssize_t
    total: cython.Py_ssize_t = 0
    dropped: cython.bint = False
    closes: cython.bint
    kept: list = []

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
    out: cython.p_double = float64_data(edges)
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


@cython.cfunc
@cython.exceptval(-1, check=False)
def _spans_inside(spans, count: cython.Py_ssize_t) -> cython.bint:
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    for span in spans:
        start = span[0]
        end = span[1]
        if start < 0 or end > count:
            return False
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def column_range(
    values: cython.p_const_double,
    count: cython.Py_ssize_t,
    low: cython.p_double,
    high: cython.p_double,
) -> cython.int:
    i: cython.Py_ssize_t
    value: cython.double
    lo: cython.double = values[0]
    hi: cython.double = values[0]
    flags: cython.int = 0
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


@cython.cfunc
def column_extreme(
    values: cython.p_const_double,
    count: cython.Py_ssize_t,
    column: object,
    maximum: cython.bint,
    bound: cython.double,
    flags: cython.int,
) -> object:
    i: cython.Py_ssize_t
    if flags & HAS_NAN:
        for i in range(count):
            if values[i] != values[i]:
                return values[i]
    if (flags & POSITIVE_ZERO) and (flags & NEGATIVE_ZERO):
        return float(column.max() if maximum else column.min())
    return bound


def translated_outline_edges(linear_x, linear_y, e: cython.double, f: cython.double, spans):
    if not (is_float64_vector(linear_x) and is_float64_vector(linear_y)):
        return translated_outline_edges_view(linear_x, linear_y, e, f, spans)
    count: cython.Py_ssize_t = vector_length(linear_x)
    if vector_length(linear_y) != count:
        raise ValueError("columns differ in length")
    return _translated_outline_edges(
        float64_data(linear_x), float64_data(linear_y), count, e, f, spans
    )


def translated_outline_edges_view(
    linear_x: cython.double[::1],
    linear_y: cython.double[::1],
    e: cython.double,
    f: cython.double,
    spans,
):
    count: cython.Py_ssize_t = linear_x.shape[0]
    if linear_y.shape[0] != count:
        raise ValueError("columns differ in length")
    return _translated_outline_edges(
        cython.address(linear_x[0]), cython.address(linear_y[0]), count, e, f, spans
    )


@cython.cfunc
def _translated_outline_edges(
    linear_x: cython.p_const_double,
    linear_y: cython.p_const_double,
    count: cython.Py_ssize_t,
    e: cython.double,
    f: cython.double,
    spans,
) -> tuple:
    i: cython.Py_ssize_t
    column_x = empty_float64(count)
    column_y = empty_float64(count)
    xs: cython.p_double = float64_data(column_x)
    ys: cython.p_double = float64_data(column_y)
    for i in range(count):
        xs[i] = linear_x[i] + e
        ys[i] = linear_y[i] + f
    if not _spans_inside(spans, count):
        raise IndexError("outline span outside the columns")
    edges, kept, dropped = _outline_edges(xs, ys, spans)
    if edges is None:
        return column_x, column_y, None, kept, dropped, None
    low_x = cython.declare(cython.double)
    high_x = cython.declare(cython.double)
    low_y = cython.declare(cython.double)
    high_y = cython.declare(cython.double)
    flags_x: cython.int
    flags_y: cython.int
    with cython.nogil:
        flags_x = column_range(xs, count, cython.address(low_x), cython.address(high_x))
        flags_y = column_range(ys, count, cython.address(low_y), cython.address(high_y))
    bounds = (
        column_extreme(xs, count, column_x, False, low_x, flags_x),
        column_extreme(ys, count, column_y, False, low_y, flags_y),
        column_extreme(xs, count, column_x, True, high_x, flags_x),
        column_extreme(ys, count, column_y, True, high_y, flags_y),
    )
    return column_x, column_y, edges, kept, dropped, bounds


def outline_coordinate_arrays(contours):
    """The x and y columns of every contour with at least two points, and each
    such contour's (start, end) span in them; None when no contour qualifies."""
    total: cython.Py_ssize_t = 0
    index: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    xs_data: cython.p_double
    ys_data: cython.p_double
    for contour in contours:
        if len(contour) >= 2:
            total += len(contour)
    if total == 0:
        return None
    xs = empty_float64(total)
    ys = empty_float64(total)
    xs_data = float64_data(xs)
    ys_data = float64_data(ys)
    spans = []
    for contour in contours:
        if len(contour) < 2:
            continue
        start = index
        for point in contour:
            x, y = point
            xs_data[index] = x
            ys_data[index] = y
            index += 1
        spans.append((start, index))
    return xs, ys, tuple(spans)
