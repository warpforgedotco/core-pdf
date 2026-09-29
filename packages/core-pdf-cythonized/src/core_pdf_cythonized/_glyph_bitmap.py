# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._glyph_bitmap import MAX_WIDTH
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.math import ceil, isfinite


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def sort_doubles(values: cython.p_double, count: cython.Py_ssize_t) -> cython.void:
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    value: cython.double
    for i in range(1, count):
        value = values[i]
        j = i - 1
        while j >= 0 and values[j] > value:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = value


def glyph_bitmap_rows(contours, width: cython.int, height: cython.int):
    """Even-odd scanline bitmap of glyph contours scaled into width x height cells.

    Returns one int per row, top row first, bit x set when cell x is inside; or
    None for input it does not own (a non-finite coordinate, or a width past
    MAX_WIDTH), which the caller rasterizes in Python.
    """
    total: cython.Py_ssize_t = 0
    contour_count: cython.Py_ssize_t = 0
    for contour in contours:
        total += len(contour)
        contour_count += 1
    if total == 0 or width <= 0 or height <= 0:
        return ()
    if width > MAX_WIDTH:
        return None

    xs: cython.p_double = cython.cast(
        cython.p_double, PyMem_Malloc(total * cython.sizeof(cython.double))
    )
    ys: cython.p_double = cython.cast(
        cython.p_double, PyMem_Malloc(total * cython.sizeof(cython.double))
    )
    lengths: cython.p_Py_ssize_t = cython.cast(
        cython.p_Py_ssize_t, PyMem_Malloc(contour_count * cython.sizeof(cython.Py_ssize_t))
    )
    edges: cython.p_double = cython.cast(
        cython.p_double, PyMem_Malloc(total * 4 * cython.sizeof(cython.double))
    )
    crossings: cython.p_double = cython.cast(
        cython.p_double, PyMem_Malloc(total * cython.sizeof(cython.double))
    )
    index: cython.Py_ssize_t = 0
    contour_index: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t
    first: cython.Py_ssize_t
    i: cython.Py_ssize_t
    edge_count: cython.Py_ssize_t = 0
    count: cython.Py_ssize_t
    min_x: cython.double
    min_y: cython.double
    max_x: cython.double
    max_y: cython.double
    span_x: cython.double
    span_y: cython.double
    scale_x: cython.double
    scale_y: cython.double
    x0: cython.double
    y0: cython.double
    x1: cython.double
    y1: cython.double
    y_mid: cython.double
    previous_x: cython.double
    previous_y: cython.double
    x: cython.double
    y: cython.double
    start_x: cython.longlong
    end_x: cython.longlong
    row: cython.ulonglong
    line: cython.int
    rows: list
    if (
        xs == cython.NULL
        or ys == cython.NULL
        or lengths == cython.NULL
        or edges == cython.NULL
        or crossings == cython.NULL
    ):
        PyMem_Free(xs)
        PyMem_Free(ys)
        PyMem_Free(lengths)
        PyMem_Free(edges)
        PyMem_Free(crossings)
        raise MemoryError()
    try:
        for contour in contours:
            length = 0
            for point in contour:
                x, y = point
                if not isfinite(x) or not isfinite(y):
                    return None
                xs[index] = x
                ys[index] = y
                index += 1
                length += 1
            lengths[contour_index] = length
            contour_index += 1

        min_x = max_x = xs[0]
        min_y = max_y = ys[0]
        for i in range(1, total):
            if xs[i] < min_x:
                min_x = xs[i]
            if xs[i] > max_x:
                max_x = xs[i]
            if ys[i] < min_y:
                min_y = ys[i]
            if ys[i] > max_y:
                max_y = ys[i]
        span_x = max_x - min_x
        span_y = max_y - min_y
        if 1.0 > span_x:
            span_x = 1.0
        if 1.0 > span_y:
            span_y = 1.0
        scale_x = cython.cast(cython.double, width - 1)
        scale_y = cython.cast(cython.double, height - 1)

        first = 0
        for contour_index in range(contour_count):
            length = lengths[contour_index]
            if length >= 3:
                previous_x = (xs[first + length - 1] - min_x) / span_x * scale_x
                previous_y = (ys[first + length - 1] - min_y) / span_y * scale_y
                for i in range(first, first + length):
                    x = (xs[i] - min_x) / span_x * scale_x
                    y = (ys[i] - min_y) / span_y * scale_y
                    if previous_y != y:
                        edges[4 * edge_count] = previous_x
                        edges[4 * edge_count + 1] = previous_y
                        edges[4 * edge_count + 2] = x
                        edges[4 * edge_count + 3] = y
                        edge_count += 1
                    previous_x = x
                    previous_y = y
            first += length
        if edge_count == 0:
            return ()

        rows = []
        for line in range(height - 1, -1, -1):
            y_mid = line + 0.5
            count = 0
            for i in range(edge_count):
                x0 = edges[4 * i]
                y0 = edges[4 * i + 1]
                x1 = edges[4 * i + 2]
                y1 = edges[4 * i + 3]
                if (y0 > y_mid) != (y1 > y_mid):
                    crossings[count] = x0 + (x1 - x0) * (y_mid - y0) / (y1 - y0)
                    count += 1
            row = 0
            if count:
                sort_doubles(crossings, count)
                for i in range(0, count - 1, 2):
                    start_x = cython.cast(cython.longlong, ceil(crossings[i] - 0.5))
                    if start_x < 0:
                        start_x = 0
                    end_x = cython.cast(cython.longlong, ceil(crossings[i + 1] - 0.5)) - 1
                    if end_x > width - 1:
                        end_x = width - 1
                    if end_x >= start_x:
                        row |= (
                            (cython.cast(cython.ulonglong, 1) << (end_x - start_x + 1)) - 1
                        ) << start_x
            rows.append(row)
        return tuple(rows)
    finally:
        PyMem_Free(xs)
        PyMem_Free(ys)
        PyMem_Free(lengths)
        PyMem_Free(edges)
        PyMem_Free(crossings)
