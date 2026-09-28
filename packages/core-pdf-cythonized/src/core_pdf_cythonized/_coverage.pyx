# SPDX-License-Identifier: AGPL-3.0-only

cimport numpy as cnp
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.math cimport ceil, fabs, floor, rint
from libc.string cimport memcpy, memset

from core_pdf_cythonized._alpha_blend cimport accumulate_plane, blend_one, opaque_channel
from core_pdf_cythonized._byte_clamp cimport unit_to_byte
from core_pdf_cythonized._knockout_math cimport knockout_component
from core_pdf_cythonized._pymath cimport py_max, py_min

import numpy

cnp.import_array()


cdef enum:
    STACK_PIECES = 512
    # Glyph fills keep their device edges and accumulation cells on the stack
    # up to these sizes; typical glyphs have about a hundred edges in a 5x7 window.
    STACK_EDGES = 256
    STACK_CELLS = 1024


cdef struct BytePlane:
    unsigned char* data
    Py_ssize_t row
    Py_ssize_t column
    Py_ssize_t channel


cdef struct FloatPlane:
    float* data
    Py_ssize_t row
    Py_ssize_t column


cdef object edge_rows(object edges):
    if not (
        cnp.PyArray_Check(edges)
        and cnp.PyArray_TYPE(<cnp.ndarray> edges) == cnp.NPY_FLOAT64
        and cnp.PyArray_NDIM(<cnp.ndarray> edges) == 2
        and cnp.PyArray_ISCARRAY_RO(<cnp.ndarray> edges)
    ):
        edges = numpy.ascontiguousarray(edges, dtype=numpy.float64)
        if cnp.PyArray_NDIM(<cnp.ndarray> edges) != 2:
            _require_edge_matrix(edges)
    return edges


cdef int _require_edge_matrix(const double[:, ::1] view) except -1:
    return 0


cdef inline const double* edge_data(object rows) noexcept:
    return <const double*> cnp.PyArray_DATA(<cnp.ndarray> rows)


cdef inline Py_ssize_t edge_count(object rows) noexcept:
    return cnp.PyArray_DIM(<cnp.ndarray> rows, 0)


cdef inline Py_ssize_t edge_width(object rows) noexcept:
    return cnp.PyArray_DIM(<cnp.ndarray> rows, 1)


cdef object _coverage_from_device(const double* e, Py_ssize_t count, int width, int height):
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
                    out[r, c] = py_min(1.0, fabs(running))
    finally:
        PyMem_Free(acc)
    return result


cdef double* _zeroed_cells(
    Py_ssize_t cells, double* scratch=NULL, Py_ssize_t scratch_cells=0
) except NULL:
    """cells zeroed doubles: scratch when they fit in it, else a PyMem block."""
    cdef double* acc
    if cells <= scratch_cells:
        acc = scratch
    else:
        acc = <double*> PyMem_Malloc((cells if cells > 0 else 1) * sizeof(double))
        if acc == NULL:
            raise MemoryError
    if cells > 0:
        memset(acc, 0, cells * sizeof(double))
    return acc


cdef int _accumulate_device(
    const double* e, Py_ssize_t count, int width, int height, double* acc
) except -1:
    cdef Py_ssize_t stride = width + 2

    # Each piece adds its left weight to its cell now and its right weight to the
    # next cell after every left weight is in, the order the sums were defined in.
    # The right weights wait in stack buffers, moved to the heap past STACK_PIECES.
    cdef Py_ssize_t idx_stack[STACK_PIECES]
    cdef double right_stack[STACK_PIECES]
    cdef Py_ssize_t capacity = STACK_PIECES, pieces = 0
    cdef Py_ssize_t* idx = idx_stack
    cdef double* right_w = right_stack

    cdef Py_ssize_t i, r, first_row, last_row, column, column_count, column_offset
    cdef double sx, sy, ex, ey, direction, top_x, top_y, bottom_x, bottom_y, slope
    cdef double row_top, row_bottom, row_height, entry_x, exit_x, left_x, right_x
    cdef double walk_left, walk_right, floor_left, column_start
    cdef double fragment_left, fragment_right, piece_span, share, fragment_height
    cdef double midpoint, offset_in_cell, signed_height
    cdef bint vertical
    cdef Py_ssize_t* new_idx
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

            first_row = <Py_ssize_t> py_max(floor(top_y), 0.0)
            last_row = <Py_ssize_t> py_min(ceil(bottom_y), <double> height)

            for r in range(first_row, last_row):
                row_top = py_max(<double> r, top_y)
                row_bottom = py_min(r + 1.0, bottom_y)
                row_height = row_bottom - row_top
                if row_height <= 0.0:
                    continue
                entry_x = top_x + (row_top - top_y) * slope
                exit_x = top_x + (row_bottom - top_y) * slope
                left_x = py_min(exit_x, entry_x)
                right_x = py_max(exit_x, entry_x)
                walk_left = py_min(width + 1.0, py_max(-1.0, left_x))
                walk_right = py_min(width + 1.0, py_max(-1.0, right_x))
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
                    column = <Py_ssize_t> py_min(<double> width, py_max(0.0, floor(midpoint)))
                    offset_in_cell = py_min(1.0, py_max(0.0, midpoint - column))
                    signed_height = direction * fragment_height

                    if pieces == capacity:
                        new_idx = <Py_ssize_t*> PyMem_Malloc(
                            2 * capacity * sizeof(Py_ssize_t))
                        new_right = <double*> PyMem_Malloc(2 * capacity * sizeof(double))
                        if new_idx == NULL or new_right == NULL:
                            PyMem_Free(new_idx); PyMem_Free(new_right)
                            raise MemoryError
                        memcpy(new_idx, idx, capacity * sizeof(Py_ssize_t))
                        memcpy(new_right, right_w, capacity * sizeof(double))
                        if idx != idx_stack:
                            PyMem_Free(idx); PyMem_Free(right_w)
                        idx = new_idx; right_w = new_right
                        capacity *= 2
                    column = r * stride + column
                    acc[column] += signed_height * (1.0 - offset_in_cell)
                    idx[pieces] = column
                    right_w[pieces] = signed_height * offset_in_cell
                    pieces += 1

        with nogil:
            for i in range(pieces):
                acc[idx[i] + 1] += right_w[i]
    finally:
        if idx != idx_stack:
            PyMem_Free(idx); PyMem_Free(right_w)
    return 0


def signed_area_coverage(edges, int width, int height):
    if height <= 0 or width <= 0 or edges.size == 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)

    cdef double[:, ::1] view = numpy.ascontiguousarray(edges, dtype=numpy.float64)
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
    rows = edge_rows(edges)
    cdef Py_ssize_t kept = _sloped_edge_count(rows)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)
    cdef double* device = _device_edges(rows, kept, crop_x0, crop_y1, scale, ix0, iy0)
    try:
        return _coverage_from_device(device, kept, width, height)
    finally:
        PyMem_Free(device)


cdef Py_ssize_t _sloped_edge_count(object rows) noexcept:
    cdef const double* view = edge_data(rows)
    cdef Py_ssize_t count = edge_count(rows), width = edge_width(rows)
    cdef Py_ssize_t i, kept = 0
    for i in range(count):
        if view[i * width + 1] != view[i * width + 3]:
            kept += 1
    return kept


cdef double* _device_edges(
    object rows,
    Py_ssize_t kept,
    double crop_x0,
    double crop_y1,
    double scale,
    double ix0,
    double iy0,
    double* scratch=NULL,
    Py_ssize_t scratch_edges=0,
) except NULL:
    """The kept edges in device space: in scratch when they fit, else a PyMem block."""
    cdef const double* view = edge_data(rows)
    cdef Py_ssize_t count = edge_count(rows), width = edge_width(rows)
    cdef double* device
    if kept <= scratch_edges:
        device = scratch
    else:
        device = <double*> PyMem_Malloc(kept * 4 * sizeof(double))
        if device == NULL:
            raise MemoryError
    cdef Py_ssize_t i, out = 0
    cdef const double* row
    cdef double sy, ey
    for i in range(count):
        row = view + i * width
        sy = row[1]
        ey = row[3]
        if sy == ey:
            continue
        device[out] = (row[0] - crop_x0) * scale - ix0
        device[out + 1] = (crop_y1 - sy) * scale - iy0
        device[out + 2] = (row[2] - crop_x0) * scale - ix0
        device[out + 3] = (crop_y1 - ey) * scale - iy0
        out += 4
    return device


def fill_glyph_coverage(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    int width,
    int height,
    rgba,
    unsigned char[:, :, :] target,
    float[:, :] source_alpha,
    float[:, :] source_shape,
    double shape_scale,
):
    rows = edge_rows(edges)
    cdef Py_ssize_t kept = _sloped_edge_count(rows)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return True
    if target.shape[0] != height or target.shape[1] != width or target.shape[2] != 4:
        raise ValueError("target differs from the plane in shape")
    cdef bint has_alpha = source_alpha is not None
    cdef bint has_shape = source_shape is not None
    if has_alpha and (source_alpha.shape[0] != height or source_alpha.shape[1] != width):
        raise ValueError("source_alpha differs from the plane in shape")
    if has_shape and (source_shape.shape[0] != height or source_shape.shape[1] != width):
        raise ValueError("source_shape differs from the plane in shape")
    cdef BytePlane pixels = BytePlane(&target[0, 0, 0], target.strides[0], target.strides[1], target.strides[2])
    cdef FloatPlane alpha_plane = FloatPlane(NULL, 0, 0)
    cdef FloatPlane shape_plane = FloatPlane(NULL, 0, 0)
    if has_alpha:
        alpha_plane = FloatPlane(&source_alpha[0, 0], source_alpha.strides[0], source_alpha.strides[1])
    if has_shape:
        shape_plane = FloatPlane(&source_shape[0, 0], source_shape.strides[0], source_shape.strides[1])
    _coverage_fill(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, width, height, rgba,
        pixels, alpha_plane, shape_plane, shape_scale,
    )
    return True


def fill_glyph_coverage_at(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    int width,
    int height,
    rgba,
    target,
    source_alpha,
    source_shape,
    double shape_scale,
):
    """fill_glyph_coverage over whole planes, painting the width x height window at (ix0, iy0)."""
    cdef BytePlane pixels
    cdef FloatPlane alpha_plane = FloatPlane(NULL, 0, 0)
    cdef FloatPlane shape_plane = FloatPlane(NULL, 0, 0)
    if height <= 0 or width <= 0 or not (
        _byte_plane(target, True, iy0, ix0, height, width, &pixels)
        and (source_alpha is None or _float_plane(source_alpha, iy0, ix0, height, width, &alpha_plane))
        and (source_shape is None or _float_plane(source_shape, iy0, ix0, height, width, &shape_plane))
    ):
        rows_window = slice(iy0, iy0 + height)
        columns_window = slice(ix0, ix0 + width)
        return fill_glyph_coverage(
            edges, crop_x0, crop_y1, scale, ix0, iy0, width, height, rgba,
            target[rows_window, columns_window],
            None if source_alpha is None else source_alpha[rows_window, columns_window],
            None if source_shape is None else source_shape[rows_window, columns_window],
            shape_scale,
        )
    rows = edge_rows(edges)
    cdef Py_ssize_t kept = _sloped_edge_count(rows)
    if kept == 0:
        return None
    _coverage_fill(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, width, height, rgba,
        pixels, alpha_plane, shape_plane, shape_scale,
    )
    return True


cdef int _coverage_fill(
    object rows,
    Py_ssize_t kept,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    int width,
    int height,
    rgba,
    BytePlane pixels,
    FloatPlane alpha_plane,
    FloatPlane shape_plane,
    double shape_scale,
) except -1:
    cdef bint has_alpha = alpha_plane.data != NULL
    cdef bint has_shape = shape_plane.data != NULL
    cdef float red = <float> <int> rgba[0]
    cdef float green = <float> <int> rgba[1]
    cdef float blue = <float> <int> rgba[2]
    cdef int cap = <int> rgba[3]
    cdef double alpha = <double> rgba[3]
    cdef int opaque_from = 255 if cap >= 255 else 256
    cdef unsigned char opaque_red = opaque_channel(red)
    cdef unsigned char opaque_green = opaque_channel(green)
    cdef unsigned char opaque_blue = opaque_channel(blue)
    cdef Py_ssize_t stride = width + 2
    cdef double device_stack[STACK_EDGES * 4]
    cdef double acc_stack[STACK_CELLS]
    cdef double* device = _device_edges(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, device_stack, STACK_EDGES
    )
    cdef double* acc = NULL
    cdef Py_ssize_t r, c
    cdef Py_ssize_t channel = pixels.channel
    cdef double running, coverage
    cdef unsigned char raw, shape
    cdef unsigned char* pixel
    cdef float* cell
    try:
        acc = _zeroed_cells(height * stride, acc_stack, STACK_CELLS)
        _accumulate_device(device, kept, width, height, acc)
        with nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    running += acc[r * stride + c]
                    coverage = py_min(1.0, fabs(running))
                    raw = <unsigned char> rint(coverage * alpha)
                    if raw != 0:
                        pixel = pixels.data + r * pixels.row + c * pixels.column
                        if raw >= opaque_from:
                            pixel[0] = opaque_red
                            pixel[channel] = opaque_green
                            pixel[2 * channel] = opaque_blue
                            pixel[3 * channel] = 255
                        else:
                            blend_one(
                                &pixel[0], &pixel[channel], &pixel[2 * channel],
                                &pixel[3 * channel], raw, cap, red, green, blue,
                            )
                    if has_alpha:
                        cell = <float*> (<char*> alpha_plane.data + r * alpha_plane.row + c * alpha_plane.column)
                        cell[0] = accumulate_plane(cell[0], raw, 1.0)
                    if has_shape:
                        shape = <unsigned char> rint(coverage * 255.0)
                        cell = <float*> (<char*> shape_plane.data + r * shape_plane.row + c * shape_plane.column)
                        cell[0] = accumulate_plane(cell[0], shape, shape_scale)
    finally:
        if device != device_stack:
            PyMem_Free(device)
        if acc != acc_stack:
            PyMem_Free(acc)
    return 0


def fill_glyph_knockout(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    int width,
    int height,
    rgba,
    unsigned char[:, :, :] destination,
    const unsigned char[:, :, :] backdrop,
    float[:, :] group_alpha,
    float[:, :] parent_shape,
    double shape_scale,
):
    rows = edge_rows(edges)
    cdef Py_ssize_t kept = _sloped_edge_count(rows)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return True
    if destination.shape[0] != height or destination.shape[1] != width or destination.shape[2] != 4:
        raise ValueError("destination differs from the plane in shape")
    if backdrop.shape[0] != height or backdrop.shape[1] != width or backdrop.shape[2] != 4:
        raise ValueError("backdrop differs from the plane in shape")
    if group_alpha.shape[0] != height or group_alpha.shape[1] != width:
        raise ValueError("group_alpha differs from the plane in shape")
    cdef bint has_parent_shape = parent_shape is not None
    if has_parent_shape and (parent_shape.shape[0] != height or parent_shape.shape[1] != width):
        raise ValueError("parent_shape differs from the plane in shape")
    cdef BytePlane target = BytePlane(
        &destination[0, 0, 0], destination.strides[0], destination.strides[1], destination.strides[2]
    )
    cdef BytePlane base = BytePlane(
        <unsigned char*> &backdrop[0, 0, 0], backdrop.strides[0], backdrop.strides[1], backdrop.strides[2]
    )
    cdef FloatPlane alpha_plane = FloatPlane(&group_alpha[0, 0], group_alpha.strides[0], group_alpha.strides[1])
    cdef FloatPlane shape_plane = FloatPlane(NULL, 0, 0)
    if has_parent_shape:
        shape_plane = FloatPlane(&parent_shape[0, 0], parent_shape.strides[0], parent_shape.strides[1])
    _knockout_fill(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, width, height, rgba,
        target, base, alpha_plane, shape_plane, shape_scale,
    )
    return True


cdef bint _byte_plane(object plane, bint writable, Py_ssize_t y0, Py_ssize_t x0, int height, int width, BytePlane* out) noexcept:
    if not cnp.PyArray_Check(plane):
        return False
    cdef cnp.ndarray array = <cnp.ndarray> plane
    if (
        cnp.PyArray_TYPE(array) != cnp.NPY_UINT8
        or cnp.PyArray_NDIM(array) != 3
        or cnp.PyArray_DIM(array, 2) != 4
        or (writable and not cnp.PyArray_ISWRITEABLE(array))
        or y0 < 0 or x0 < 0
        or y0 + height > cnp.PyArray_DIM(array, 0)
        or x0 + width > cnp.PyArray_DIM(array, 1)
    ):
        return False
    out.row = cnp.PyArray_STRIDE(array, 0)
    out.column = cnp.PyArray_STRIDE(array, 1)
    out.channel = cnp.PyArray_STRIDE(array, 2)
    out.data = <unsigned char*> cnp.PyArray_DATA(array) + y0 * out.row + x0 * out.column
    return True


cdef bint _float_plane(object plane, Py_ssize_t y0, Py_ssize_t x0, int height, int width, FloatPlane* out) noexcept:
    if not cnp.PyArray_Check(plane):
        return False
    cdef cnp.ndarray array = <cnp.ndarray> plane
    if (
        cnp.PyArray_TYPE(array) != cnp.NPY_FLOAT32
        or cnp.PyArray_NDIM(array) != 2
        or not cnp.PyArray_ISWRITEABLE(array)
        or not cnp.PyArray_ISALIGNED(array)
        or not cnp.PyArray_ISNOTSWAPPED(array)
        or y0 < 0 or x0 < 0
        or y0 + height > cnp.PyArray_DIM(array, 0)
        or x0 + width > cnp.PyArray_DIM(array, 1)
    ):
        return False
    out.row = cnp.PyArray_STRIDE(array, 0)
    out.column = cnp.PyArray_STRIDE(array, 1)
    out.data = <float*> (<char*> cnp.PyArray_DATA(array) + y0 * out.row + x0 * out.column)
    return True


def fill_glyph_knockout_at(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    int width,
    int height,
    rgba,
    destination,
    backdrop,
    group_alpha,
    parent_shape,
    double shape_scale,
):
    """fill_glyph_knockout over whole planes, painting the width x height window at (ix0, iy0)."""
    cdef BytePlane target
    cdef BytePlane base
    cdef FloatPlane alpha_plane
    cdef FloatPlane shape_plane = FloatPlane(NULL, 0, 0)
    if height <= 0 or width <= 0 or not (
        _byte_plane(destination, True, iy0, ix0, height, width, &target)
        and _byte_plane(backdrop, False, iy0, ix0, height, width, &base)
        and _float_plane(group_alpha, iy0, ix0, height, width, &alpha_plane)
        and (parent_shape is None or _float_plane(parent_shape, iy0, ix0, height, width, &shape_plane))
    ):
        rows_window = slice(iy0, iy0 + height)
        columns_window = slice(ix0, ix0 + width)
        return fill_glyph_knockout(
            edges, crop_x0, crop_y1, scale, ix0, iy0, width, height, rgba,
            destination[rows_window, columns_window],
            backdrop[rows_window, columns_window],
            group_alpha[rows_window, columns_window],
            None if parent_shape is None else parent_shape[rows_window, columns_window],
            shape_scale,
        )
    rows = edge_rows(edges)
    cdef Py_ssize_t kept = _sloped_edge_count(rows)
    if kept == 0:
        return None
    _knockout_fill(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, width, height, rgba,
        target, base, alpha_plane, shape_plane, shape_scale,
    )
    return True


cdef int _knockout_fill(
    object rows,
    Py_ssize_t kept,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    int width,
    int height,
    rgba,
    BytePlane target,
    BytePlane base,
    FloatPlane alpha_plane,
    FloatPlane shape_plane,
    double shape_scale,
) except -1:
    cdef bint has_parent_shape = shape_plane.data != NULL
    cdef float red = <float> <int> rgba[0]
    cdef float green = <float> <int> rgba[1]
    cdef float blue = <float> <int> rgba[2]
    cdef int cap = <int> rgba[3]
    cdef double alpha = <double> rgba[3]
    cdef int opaque_from = 255 if cap >= 255 else 256
    cdef unsigned char opaque_red = opaque_channel(red)
    cdef unsigned char opaque_green = opaque_channel(green)
    cdef unsigned char opaque_blue = opaque_channel(blue)
    cdef Py_ssize_t stride = width + 2
    cdef double device_stack[STACK_EDGES * 4]
    cdef double acc_stack[STACK_CELLS]
    cdef double* device = _device_edges(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, device_stack, STACK_EDGES
    )
    cdef double* acc = NULL
    cdef Py_ssize_t r, c, k
    cdef double running, coverage, scaled, eff, sh, remaining, rga, ra, complete, initial, ec
    cdef double colour[3]
    cdef unsigned char raw, shape_byte, quantized
    cdef unsigned char rendered[4]
    cdef const unsigned char* element
    cdef unsigned char* destination
    cdef const unsigned char* backdrop
    cdef float* group_alpha
    cdef float* parent_shape
    cdef float ZERO = 0.0
    cdef float ONE = 1.0
    cdef float source_alpha, source_shape, previous
    try:
        acc = _zeroed_cells(height * stride, acc_stack, STACK_CELLS)
        _accumulate_device(device, kept, width, height, acc)
        with nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    destination = target.data + r * target.row + c * target.column
                    backdrop = base.data + r * base.row + c * base.column
                    group_alpha = <float*> (
                        <char*> alpha_plane.data + r * alpha_plane.row + c * alpha_plane.column
                    )
                    running += acc[r * stride + c]
                    coverage = py_min(1.0, fabs(running))
                    raw = <unsigned char> rint(coverage * alpha)
                    source_alpha = accumulate_plane(ZERO, raw, 1.0)
                    shape_byte = <unsigned char> rint(coverage * 255.0)
                    source_shape = accumulate_plane(ZERO, shape_byte, shape_scale)
                    scaled = rint(<double> source_alpha * 255.0)
                    quantized = <unsigned char> <int> scaled
                    if quantized > 0:
                        for k in range(4):
                            rendered[k] = backdrop[k * base.channel]
                        if raw != 0:
                            if raw >= opaque_from:
                                rendered[0] = opaque_red
                                rendered[1] = opaque_green
                                rendered[2] = opaque_blue
                                rendered[3] = 255
                            else:
                                blend_one(
                                    &rendered[0], &rendered[1], &rendered[2], &rendered[3],
                                    raw, cap, red, green, blue,
                                )
                        element = &rendered[0]
                    else:
                        for k in range(4):
                            rendered[k] = backdrop[k * base.channel]
                        element = &rendered[0]
                    eff = <double> quantized / 255.0
                    sh = <double> source_shape
                    if sh < 0.0:
                        sh = 0.0
                    elif sh > 1.0:
                        sh = 1.0
                    if eff > sh:
                        sh = eff
                    if sh > 0.0:
                        complete = <double> destination[3 * target.channel] / 255.0
                        initial = <double> backdrop[3 * base.channel] / 255.0
                        ec = <double> element[3] / 255.0
                        remaining = 1.0 - sh
                        rga = eff + remaining * <double> group_alpha[0]
                        ra = initial + (1.0 - initial) * rga
                        for k in range(3):
                            colour[k] = knockout_component(
                                <double> element[k] / 255.0, ec, remaining,
                                <double> destination[k * target.channel] / 255.0, complete,
                                <double> backdrop[k * base.channel] / 255.0, initial, ra,
                            )
                        for k in range(3):
                            destination[k * target.channel] = <unsigned char> unit_to_byte(colour[k])
                        destination[3 * target.channel] = <unsigned char> unit_to_byte(ra)
                        group_alpha[0] = <float> rga
                    if has_parent_shape:
                        parent_shape = <float*> (
                            <char*> shape_plane.data + r * shape_plane.row + c * shape_plane.column
                        )
                        previous = parent_shape[0]
                        parent_shape[0] = previous + (ONE - previous) * source_shape
    finally:
        if device != device_stack:
            PyMem_Free(device)
        if acc != acc_stack:
            PyMem_Free(acc)
    return 0
