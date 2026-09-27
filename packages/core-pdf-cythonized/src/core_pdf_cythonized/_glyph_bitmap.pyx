# SPDX-License-Identifier: AGPL-3.0-only

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.math cimport ceil, isfinite


cdef enum:
    MAX_WIDTH = 62


cdef void sort_doubles(double *values, Py_ssize_t count) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double value
    for i in range(1, count):
        value = values[i]
        j = i - 1
        while j >= 0 and values[j] > value:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = value


def glyph_bitmap_rows(contours, int width, int height):
    """Even-odd scanline bitmap of glyph contours scaled into width x height cells.

    Returns one int per row, top row first, bit x set when cell x is inside; or
    None for input it does not own (a non-finite coordinate, or a width past
    MAX_WIDTH), which the caller rasterizes in Python.
    """
    cdef Py_ssize_t total = 0
    cdef Py_ssize_t contour_count = 0
    for contour in contours:
        total += len(contour)
        contour_count += 1
    if total == 0 or width <= 0 or height <= 0:
        return ()
    if width > MAX_WIDTH:
        return None

    cdef double *xs = <double *>PyMem_Malloc(total * sizeof(double))
    cdef double *ys = <double *>PyMem_Malloc(total * sizeof(double))
    cdef Py_ssize_t *lengths = <Py_ssize_t *>PyMem_Malloc(contour_count * sizeof(Py_ssize_t))
    cdef double *edges = <double *>PyMem_Malloc(total * 4 * sizeof(double))
    cdef double *crossings = <double *>PyMem_Malloc(total * sizeof(double))
    cdef Py_ssize_t index = 0, contour_index = 0, length, first, i, edge_count = 0, count
    cdef double min_x, min_y, max_x, max_y, span_x, span_y, scale_x, scale_y
    cdef double x0, y0, x1, y1, y_mid, previous_x, previous_y, x, y
    cdef long long start_x, end_x
    cdef unsigned long long row
    cdef int line
    cdef list rows
    if xs == NULL or ys == NULL or lengths == NULL or edges == NULL or crossings == NULL:
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
        scale_x = <double>(width - 1)
        scale_y = <double>(height - 1)

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
                    start_x = <long long>ceil(crossings[i] - 0.5)
                    if start_x < 0:
                        start_x = 0
                    end_x = <long long>ceil(crossings[i + 1] - 0.5) - 1
                    if end_x > width - 1:
                        end_x = width - 1
                    if end_x >= start_x:
                        row |= ((1ULL << (end_x - start_x + 1)) - 1) << start_x
            rows.append(row)
        return tuple(rows)
    finally:
        PyMem_Free(xs)
        PyMem_Free(ys)
        PyMem_Free(lengths)
        PyMem_Free(edges)
        PyMem_Free(crossings)
