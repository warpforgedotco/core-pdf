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
from libc.math cimport ceil, fabs, floor, rint

import numpy


cdef inline double dmin(double a, double b) noexcept nogil:
    return a if a < b else b


cdef inline double dmax(double a, double b) noexcept nogil:
    return a if a > b else b


cdef object _coverage_from_device(const double* e, Py_ssize_t count, int width, int height):
    """Accumulate device-space edges into a coverage plane.

    ``e`` points at ``count`` rows of four doubles: x0, y0, x1, y1, already in
    pixel coordinates relative to the plane's top-left corner.
    """
    cdef Py_ssize_t stride = width + 2
    result = numpy.zeros((height, width), numpy.float64)
    cdef double[:, ::1] out = result
    cdef double* acc = _zeroed_cells(height * stride)
    cdef Py_ssize_t r, c
    cdef double running
    try:
        _accumulate_device(e, count, width, height, acc)
        with nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    running += acc[r * stride + c]
                    out[r, c] = dmin(fabs(running), 1.0)
    finally:
        PyMem_Free(acc)
    return result


cdef double* _zeroed_cells(Py_ssize_t cells) except NULL:
    cdef double* acc = <double*> PyMem_Malloc((cells if cells > 0 else 1) * sizeof(double))
    if acc == NULL:
        raise MemoryError
    cdef Py_ssize_t i
    for i in range(cells):
        acc[i] = 0.0
    return acc


cdef int _accumulate_device(
    const double* e, Py_ssize_t count, int width, int height, double* acc
) except -1:
    """Add each edge's signed area into ``acc``, height rows of width + 2 cells.

    A pixel's coverage is the running sum along its row, clamped to [0, 1];
    the finishers above and below read it that way.
    """
    cdef Py_ssize_t stride = width + 2

    cdef Py_ssize_t capacity = 1024, pieces = 0
    cdef Py_ssize_t* idx = <Py_ssize_t*> PyMem_Malloc(capacity * sizeof(Py_ssize_t))
    cdef double* left_w = <double*> PyMem_Malloc(capacity * sizeof(double))
    cdef double* right_w = <double*> PyMem_Malloc(capacity * sizeof(double))
    if idx == NULL or left_w == NULL or right_w == NULL:
        PyMem_Free(idx); PyMem_Free(left_w); PyMem_Free(right_w)
        raise MemoryError

    cdef Py_ssize_t i, r, first_row, last_row, column, column_count, column_offset
    cdef double sx, sy, ex, ey, direction, top_x, top_y, bottom_x, bottom_y, slope
    cdef double row_top, row_bottom, row_height, entry_x, exit_x, left_x, right_x
    cdef double walk_left, walk_right, floor_left, column_start
    cdef double fragment_left, fragment_right, piece_span, share, fragment_height
    cdef double midpoint, offset_in_cell, signed_height
    cdef bint vertical
    cdef Py_ssize_t* new_idx
    cdef double* new_left
    cdef double* new_right

    try:
        for i in range(count):
            sx = e[i * 4]; sy = e[i * 4 + 1]; ex = e[i * 4 + 2]; ey = e[i * 4 + 3]
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
    finally:
        PyMem_Free(idx); PyMem_Free(left_w); PyMem_Free(right_w)
    return 0


def signed_area_coverage(edges, int width, int height):
    if height <= 0 or width <= 0 or edges.size == 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)

    cdef double[:, ::1] view = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    if view.shape[0] == 0:
        return numpy.zeros((height, width), numpy.float64)
    return _coverage_from_device(&view[0, 0], view.shape[0], width, height)


def glyph_coverage_plane(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    double ix0,
    double iy0,
    int width,
    int height,
):
    """Transform page-space edges to device space and accumulate coverage.

    The numpy original spent about twelve array operations flattening a
    hundred-odd edges into a device-space copy, to fill a plane averaging
    forty pixels. The transform is four independent affine expressions per
    edge, so it fuses into the accumulation loop's own read of each edge and
    the intermediate array disappears.

    Returns ``None`` when no edge has distinct endpoints in y, which is the
    early return the caller used to get from ``sloped.any()``. A plane of that
    shape would be entirely zero.
    """
    cdef double[:, ::1] view = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    cdef Py_ssize_t kept = _sloped_edge_count(view)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)
    cdef double* device = _device_edges(view, kept, crop_x0, crop_y1, scale, ix0, iy0)
    try:
        return _coverage_from_device(device, kept, width, height)
    finally:
        PyMem_Free(device)


cdef Py_ssize_t _sloped_edge_count(const double[:, ::1] view) noexcept:
    cdef Py_ssize_t i, kept = 0
    for i in range(view.shape[0]):
        if view[i, 1] != view[i, 3]:
            kept += 1
    return kept


cdef double* _device_edges(
    const double[:, ::1] view,
    Py_ssize_t kept,
    double crop_x0,
    double crop_y1,
    double scale,
    double ix0,
    double iy0,
) except NULL:
    """The sloped edges moved to the plane's pixel coordinates, to be freed."""
    cdef double* device = <double*> PyMem_Malloc(kept * 4 * sizeof(double))
    if device == NULL:
        raise MemoryError
    cdef Py_ssize_t i, out = 0
    cdef double sy, ey
    # Each expression matches the numpy original term for term, and the
    # build disables float contraction so the compiler cannot fold any of
    # them into an FMA and shift the result by an ULP.
    for i in range(view.shape[0]):
        sy = view[i, 1]
        ey = view[i, 3]
        if sy == ey:
            continue
        device[out] = (view[i, 0] - crop_x0) * scale - ix0
        device[out + 1] = (crop_y1 - sy) * scale - iy0
        device[out + 2] = (view[i, 2] - crop_x0) * scale - ix0
        device[out + 3] = (crop_y1 - ey) * scale - iy0
        out += 4
    return device


def glyph_alpha_planes(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    double ix0,
    double iy0,
    int width,
    int height,
    double alpha,
    bint want_shape,
):
    """glyph_coverage_plane, quantized: the uint8 alpha plane and, if asked, shape.

    fill_path turned the float64 coverage plane into what it blends and
    records with numpy.rint(coverage * alpha).astype(uint8), and a second
    pass at 255 for a group's shape plane: three array operations and two
    temporaries per glyph, around a kernel of a few microseconds. This
    writes both from the running sum directly. The coverage is the same
    double glyph_coverage_plane would store, and rint rounds half to even
    as numpy.rint does, so each byte is the one numpy produced.

    Returns None where glyph_coverage_plane does, else (alpha, shape or None).
    """
    cdef double[:, ::1] view = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    cdef Py_ssize_t kept = _sloped_edge_count(view)
    if kept == 0:
        return None
    alpha_plane = numpy.zeros((max(height, 0), max(width, 0)), numpy.uint8)
    shape_plane = (
        numpy.zeros((max(height, 0), max(width, 0)), numpy.uint8) if want_shape else None
    )
    if height <= 0 or width <= 0:
        return alpha_plane, shape_plane
    cdef unsigned char[:, ::1] alpha_out = alpha_plane
    cdef unsigned char[:, ::1] shape_out
    if want_shape:
        shape_out = shape_plane
    cdef Py_ssize_t stride = width + 2
    cdef double* device = _device_edges(view, kept, crop_x0, crop_y1, scale, ix0, iy0)
    cdef double* acc = NULL
    cdef Py_ssize_t r, c
    cdef double running, coverage
    try:
        acc = _zeroed_cells(height * stride)
        _accumulate_device(device, kept, width, height, acc)
        with nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    running += acc[r * stride + c]
                    coverage = dmin(fabs(running), 1.0)
                    alpha_out[r, c] = <unsigned char> rint(coverage * alpha)
                    if want_shape:
                        shape_out[r, c] = <unsigned char> rint(coverage * 255.0)
    finally:
        PyMem_Free(device)
        PyMem_Free(acc)
    return alpha_plane, shape_plane
