# SPDX-License-Identifier: AGPL-3.0-only
"""Signed-area scanline coverage (core_pdf.impl.render.paths).

The numpy original was not slow because the algorithm is heavy. It was slow
because it ran about thirty array operations to fill a buffer averaging forty
pixels, several thousand times per page: pure per-call overhead.

Bit-exactness rests on one detail. The original accumulated with
numpy.bincount over a concatenation of two index blocks, and bincount is
defined as a sequential out[idx[i]] += w[i]. So every left-hand weight is
added before any right-hand weight, and reproducing that split -- rather than
adding both weights per piece as you go -- is what keeps the float rounding
identical. The two accumulate loops at the bottom are that split.
"""

from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.math cimport ceil, fabs, floor

import numpy


cdef inline double dmin(double a, double b) noexcept nogil:
    return a if a < b else b


cdef inline double dmax(double a, double b) noexcept nogil:
    return a if a > b else b


def signed_area_coverage(edges, int width, int height):
    if height <= 0 or width <= 0 or edges.size == 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)

    cdef double[:, ::1] view = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    cdef Py_ssize_t count = view.shape[0]
    cdef Py_ssize_t stride = width + 2
    result = numpy.zeros((height, width), numpy.float64)
    cdef double[:, ::1] out = result

    cdef double[::1] acc = numpy.zeros(height * stride, numpy.float64)

    cdef Py_ssize_t capacity = 1024, pieces = 0
    cdef Py_ssize_t* idx = <Py_ssize_t*> PyMem_Malloc(capacity * sizeof(Py_ssize_t))
    cdef double* left_w = <double*> PyMem_Malloc(capacity * sizeof(double))
    cdef double* right_w = <double*> PyMem_Malloc(capacity * sizeof(double))
    if idx == NULL or left_w == NULL or right_w == NULL:
        PyMem_Free(idx); PyMem_Free(left_w); PyMem_Free(right_w)
        raise MemoryError

    cdef Py_ssize_t i, r, c, first_row, last_row, column, column_count, column_offset
    cdef double sx, sy, ex, ey, direction, top_x, top_y, bottom_x, bottom_y, slope
    cdef double row_top, row_bottom, row_height, entry_x, exit_x, left_x, right_x
    cdef double walk_left, walk_right, floor_left, column_start
    cdef double fragment_left, fragment_right, piece_span, share, fragment_height
    cdef double midpoint, offset_in_cell, signed_height, running
    cdef bint vertical
    cdef Py_ssize_t* new_idx
    cdef double* new_left
    cdef double* new_right

    try:
        for i in range(count):
            sx = view[i, 0]; sy = view[i, 1]; ex = view[i, 2]; ey = view[i, 3]
            if sy == ey:
                continue
            if ey > sy:
                direction = 1.0
                top_x = sx; top_y = sy; bottom_x = ex; bottom_y = ey
            else:
                direction = -1.0
                top_x = ex; top_y = ey; bottom_x = sx; bottom_y = sy
            slope = (bottom_x - top_x) / (bottom_y - top_y)

            first_row = <Py_ssize_t> dmax(0.0, floor(top_y))
            last_row = <Py_ssize_t> dmin(<double> height, ceil(bottom_y))

            for r in range(first_row, last_row):
                row_top = dmax(top_y, <double> r)
                row_bottom = dmin(bottom_y, r + 1.0)
                row_height = row_bottom - row_top
                if row_height <= 0.0:
                    continue
                entry_x = top_x + (row_top - top_y) * slope
                exit_x = top_x + (row_bottom - top_y) * slope
                left_x = dmin(entry_x, exit_x)
                right_x = dmax(entry_x, exit_x)
                walk_left = dmin(dmax(left_x, -1.0), width + 1.0)
                walk_right = dmin(dmax(right_x, -1.0), width + 1.0)
                floor_left = floor(walk_left)
                column_count = <Py_ssize_t> (floor(walk_right) - floor_left + 1.0)
                piece_span = right_x - left_x
                vertical = piece_span <= 0.0

                for column_offset in range(column_count):
                    column_start = floor_left + column_offset
                    fragment_left = left_x if column_offset == 0 else column_start
                    fragment_right = (
                        right_x if column_offset == column_count - 1 else column_start + 1.0
                    )
                    if not (vertical or fragment_right > fragment_left):
                        continue
                    share = 1.0 if vertical else (fragment_right - fragment_left) / piece_span
                    fragment_height = row_height * share
                    midpoint = (fragment_left + fragment_right) * 0.5
                    column = <Py_ssize_t> dmin(dmax(floor(midpoint), 0.0), <double> width)
                    offset_in_cell = dmin(dmax(midpoint - column, 0.0), 1.0)
                    signed_height = direction * fragment_height

                    if pieces == capacity:
                        capacity *= 2
                        new_idx = <Py_ssize_t*> PyMem_Realloc(
                            idx, capacity * sizeof(Py_ssize_t))
                        new_left = <double*> PyMem_Realloc(left_w, capacity * sizeof(double))
                        new_right = <double*> PyMem_Realloc(right_w, capacity * sizeof(double))
                        if new_idx == NULL or new_left == NULL or new_right == NULL:
                            if new_idx != NULL: idx = new_idx
                            if new_left != NULL: left_w = new_left
                            if new_right != NULL: right_w = new_right
                            raise MemoryError
                        idx = new_idx; left_w = new_left; right_w = new_right
                    idx[pieces] = r * stride + column
                    left_w[pieces] = signed_height * (1.0 - offset_in_cell)
                    right_w[pieces] = signed_height * offset_in_cell
                    pieces += 1

        # bincount order: every left weight, then every right weight.
        with nogil:
            for i in range(pieces):
                acc[idx[i]] += left_w[i]
            for i in range(pieces):
                acc[idx[i] + 1] += right_w[i]

            for r in range(height):
                running = 0.0
                for c in range(width):
                    running += acc[r * stride + c]
                    out[r, c] = dmin(fabs(running), 1.0)
    finally:
        PyMem_Free(idx); PyMem_Free(left_w); PyMem_Free(right_w)

    return result
