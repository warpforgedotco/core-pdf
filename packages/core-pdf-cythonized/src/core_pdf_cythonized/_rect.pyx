# SPDX-License-Identifier: AGPL-3.0-only
"""Axis-aligned rectangle coverage (core_pdf.impl.render.paths).

The numpy original spent about ten array operations -- two aranges, two
minimums, two maximums, two clips, an outer product, a rint and a cast -- to
build a plane whose median size on the corpus is thirty pixels. One page
issues 18,812 of them.

At that size numpy is no faster than an interpreted loop (measured 7.1us
against 8.7us for a 6x6 plane), which is the tell: the work is scalar and the
array machinery is pure overhead, so a C loop wins outright rather than
marginally.

Coverage is separable -- the plane is the outer product of a row profile and a
column profile -- so the two profiles are computed once and multiplied per
pixel, exactly as numpy.outer did.
"""

from libc.math cimport rint
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from cpython.mem cimport PyMem_Free, PyMem_Malloc

from core_pdf_cythonized._alpha_blend cimport accumulate_plane, blend_one, opaque_channel

import numpy


cdef inline double axis_coverage(double index, double low, double high) noexcept nogil:
    # numpy: clip(minimum(index + 1, high) - maximum(index, low), 0, 1)
    cdef double upper = index + 1.0
    if high < upper:
        upper = high
    cdef double lower = index
    if low > lower:
        lower = low
    cdef double span = upper - lower
    if span < 0.0:
        return 0.0
    if span > 1.0:
        return 1.0
    return span


def rect_coverage_plane(
    Py_ssize_t ix0,
    Py_ssize_t ix1,
    Py_ssize_t iy0,
    Py_ssize_t iy1,
    double left,
    double right,
    double top,
    double bottom,
    int scale,
):
    cdef Py_ssize_t width = ix1 - ix0
    cdef Py_ssize_t height = iy1 - iy0
    if width < 0:
        width = 0
    if height < 0:
        height = 0
    plane = numpy.empty((height, width), numpy.uint8)
    if width == 0 or height == 0:
        return plane

    cdef unsigned char[:, ::1] out = plane
    cdef double* columns = <double*> PyMem_Malloc(width * sizeof(double))
    if columns == NULL:
        raise MemoryError
    cdef Py_ssize_t i, j
    cdef double row_coverage, value
    cdef double scale_value = <double> scale

    try:
        with nogil:
            for j in range(width):
                columns[j] = axis_coverage(<double> (ix0 + j), left, right)
            for i in range(height):
                row_coverage = axis_coverage(<double> (iy0 + i), top, bottom)
                for j in range(width):
                    # numpy multiplied the outer product by the scale, so the
                    # row/column product is formed first and scaled after.
                    value = rint(row_coverage * columns[j] * scale_value)
                    out[i, j] = <unsigned char> value
    finally:
        PyMem_Free(columns)
    return plane


cdef int fill_rect_pixels(
    unsigned char* base,
    Py_ssize_t row_stride,
    Py_ssize_t ix0,
    Py_ssize_t ix1,
    Py_ssize_t iy0,
    Py_ssize_t iy1,
    double left,
    double right,
    double top,
    double bottom,
    int red_byte,
    int green_byte,
    int blue_byte,
    int cap,
) noexcept nogil:
    """fill_rect_coverage into packed pixels with no plane: base is pixel (iy0, ix0).

    Returns -1 if the column table cannot be allocated, else 0. The stroke
    kernel paints its square joins and caps through this too.
    """
    cdef Py_ssize_t width = ix1 - ix0
    cdef Py_ssize_t height = iy1 - iy0
    if width <= 0 or height <= 0:
        return 0
    cdef float red = <float> red_byte
    cdef float green = <float> green_byte
    cdef float blue = <float> blue_byte
    cdef double alpha = <double> cap
    cdef int opaque_from = 255 if cap >= 255 else 256
    cdef unsigned char opaque[4]
    opaque[0] = opaque_channel(red)
    opaque[1] = opaque_channel(green)
    opaque[2] = opaque_channel(blue)
    opaque[3] = 255
    # Per column: the coverage, and the alpha byte of a fully covered row,
    # where the product is the column's coverage exactly. Most rows of a
    # large rectangle are full, so their pixels skip the multiply and the
    # rounding, and their opaque run is written as a block.
    cdef double* columns = <double*> malloc(width * (sizeof(double) + 1))
    if columns == NULL:
        return -1
    cdef unsigned char* full_raw = <unsigned char*> (columns + width)
    cdef unsigned char* row
    cdef unsigned char* pixel
    cdef bint full_row
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t run_start = 0, run_end = 0, segment_end, resume
    cdef double row_coverage
    cdef unsigned char raw
    # A translucent band over a flat backdrop repeats its inputs pixel after
    # pixel, and each result depends on its own pixel's inputs alone.
    cdef unsigned int backdrop, last_backdrop = 0
    cdef int last_raw = -1
    # Set before its first read (last_raw starts where no raw matches it),
    # which GCC cannot see.
    cdef unsigned char blended[4]
    blended[0] = blended[1] = blended[2] = blended[3] = 0
    for j in range(width):
        columns[j] = axis_coverage(<double> (ix0 + j), left, right)
        full_raw[j] = <unsigned char> rint(columns[j] * alpha)
    while run_start < width and full_raw[run_start] < opaque_from:
        run_start += 1
    run_end = run_start
    while run_end < width and full_raw[run_end] >= opaque_from:
        run_end += 1
    for k in range(run_end, width):
        if full_raw[k] >= opaque_from:
            run_start = run_end = 0
            break
    for i in range(height):
        row = base + i * row_stride
        row_coverage = axis_coverage(<double> (iy0 + i), top, bottom)
        full_row = row_coverage == 1.0
        # The columns left for the loop below: all of them, or those either
        # side of the block.
        segment_end = width
        resume = width
        if full_row and run_end > run_start:
            for j in range(run_start, run_end):
                memcpy(row + 4 * j, opaque, 4)
            segment_end = run_start
            resume = run_end
        j = 0
        while j < width:
            if j == segment_end:
                j = resume
                if j >= width:
                    break
            if full_row:
                raw = full_raw[j]
            else:
                raw = <unsigned char> rint(row_coverage * columns[j] * alpha)
            if raw != 0:
                pixel = row + 4 * j
                if raw >= opaque_from:
                    memcpy(pixel, opaque, 4)
                else:
                    memcpy(&backdrop, pixel, 4)
                    if raw != last_raw or backdrop != last_backdrop:
                        memcpy(blended, pixel, 4)
                        blend_one(
                            &blended[0], &blended[1], &blended[2], &blended[3],
                            raw, cap, red, green, blue,
                        )
                        last_raw = raw
                        last_backdrop = backdrop
                    memcpy(pixel, blended, 4)
            j += 1
    free(columns)
    return 0


def fill_rect_coverage(
    Py_ssize_t ix0,
    Py_ssize_t ix1,
    Py_ssize_t iy0,
    Py_ssize_t iy1,
    double left,
    double right,
    double top,
    double bottom,
    rgba,
    unsigned char[:, :, :] target,
    float[:, :] source_alpha,
    float[:, :] source_shape,
    double shape_scale,
):
    """A partially covered rectangle fill, fused: coverage, blend and plane records.

    fill_rect built rect_coverage_plane at the paint's alpha, blended it with
    blend_normal_alpha_array_numpy, recorded it with accumulate_source_plane
    and, for a group's shape, built the plane again at 255 and recorded that:
    five kernel calls behind Python wrappers for a plane of about thirty
    pixels. This makes the same bytes and floats in one pass -- each pixel's
    row and column coverage product scaled and rounded as
    rect_coverage_plane does, blend_one's compositing into ``target`` and
    accumulate_plane's update of each plane given, the shape at
    ``shape_scale``.

    Large rectangles are mostly fully covered rows, where each pixel's
    product is its column's coverage: those rows read the column's bytes,
    write their opaque run as a block when no plane is recorded, and loop
    only over the edges. test_3450 fills 4,728 soft-mask bands of about
    80,000 pixels; that took them from 1.1 to 0.33 ns a pixel. The blend and
    plane updates keep the last inputs and results, as flat bands repeat
    them.
    """
    cdef Py_ssize_t width = ix1 - ix0
    cdef Py_ssize_t height = iy1 - iy0
    if width <= 0 or height <= 0:
        return
    if target.shape[0] != height or target.shape[1] != width or target.shape[2] != 4:
        raise ValueError("target differs from the rectangle in shape")
    cdef bint has_alpha = source_alpha is not None
    cdef bint has_shape = source_shape is not None
    if has_alpha and (source_alpha.shape[0] != height or source_alpha.shape[1] != width):
        raise ValueError("source_alpha differs from the rectangle in shape")
    if has_shape and (source_shape.shape[0] != height or source_shape.shape[1] != width):
        raise ValueError("source_shape differs from the rectangle in shape")
    cdef int red_byte = <int> rgba[0]
    cdef int green_byte = <int> rgba[1]
    cdef int blue_byte = <int> rgba[2]
    cdef int cap = <int> rgba[3]
    cdef int status
    if target.strides[1] == 4 and target.strides[2] == 1 and not has_alpha and not has_shape:
        with nogil:
            status = fill_rect_pixels(
                &target[0, 0, 0], target.strides[0], ix0, ix1, iy0, iy1,
                left, right, top, bottom, red_byte, green_byte, blue_byte, cap,
            )
        if status < 0:
            raise MemoryError
        return
    cdef float red = <float> red_byte
    cdef float green = <float> green_byte
    cdef float blue = <float> blue_byte
    cdef double alpha = <double> cap
    cdef int opaque_from = 255 if cap >= 255 else 256
    cdef unsigned char opaque_red = opaque_channel(red)
    cdef unsigned char opaque_green = opaque_channel(green)
    cdef unsigned char opaque_blue = opaque_channel(blue)
    # Per column: the coverage, and the alpha and shape bytes of a fully
    # covered row, where the product is the column's coverage exactly.
    cdef double* columns = <double*> PyMem_Malloc(width * (sizeof(double) + 2))
    if columns == NULL:
        raise MemoryError
    cdef unsigned char* full_raw = <unsigned char*> (columns + width)
    cdef unsigned char* full_shape = full_raw + width
    cdef bint full_row
    cdef Py_ssize_t i, j
    cdef double row_coverage, product
    cdef unsigned char raw, shape
    # The blend and both plane updates keep the last inputs and what they
    # gave, as fill_rect_pixels does.
    cdef unsigned int backdrop, last_backdrop = 0
    cdef int last_raw = -1
    cdef unsigned char blended[4]
    blended[0] = blended[1] = blended[2] = blended[3] = 0
    cdef float previous, last_alpha_out = 0.0, last_shape_out = 0.0
    cdef unsigned int previous_bits, last_alpha_in = 0, last_shape_in = 0
    cdef int last_alpha_raw = -1, last_shape = -1
    try:
        with nogil:
            for j in range(width):
                columns[j] = axis_coverage(<double> (ix0 + j), left, right)
                full_raw[j] = <unsigned char> rint(columns[j] * alpha)
                full_shape[j] = <unsigned char> rint(columns[j] * 255.0)
            for i in range(height):
                row_coverage = axis_coverage(<double> (iy0 + i), top, bottom)
                full_row = row_coverage == 1.0
                for j in range(width):
                    if full_row:
                        raw = full_raw[j]
                        product = columns[j]
                    else:
                        product = row_coverage * columns[j]
                        raw = <unsigned char> rint(product * alpha)
                    if raw != 0:
                        if raw >= opaque_from:
                            target[i, j, 0] = opaque_red
                            target[i, j, 1] = opaque_green
                            target[i, j, 2] = opaque_blue
                            target[i, j, 3] = 255
                        else:
                            backdrop = (
                                <unsigned int> target[i, j, 0]
                                | (<unsigned int> target[i, j, 1] << 8)
                                | (<unsigned int> target[i, j, 2] << 16)
                                | (<unsigned int> target[i, j, 3] << 24)
                            )
                            if raw != last_raw or backdrop != last_backdrop:
                                blended[0] = target[i, j, 0]
                                blended[1] = target[i, j, 1]
                                blended[2] = target[i, j, 2]
                                blended[3] = target[i, j, 3]
                                blend_one(
                                    &blended[0], &blended[1], &blended[2], &blended[3],
                                    raw, cap, red, green, blue,
                                )
                                last_raw = raw
                                last_backdrop = backdrop
                            target[i, j, 0] = blended[0]
                            target[i, j, 1] = blended[1]
                            target[i, j, 2] = blended[2]
                            target[i, j, 3] = blended[3]
                    if has_alpha:
                        previous = source_alpha[i, j]
                        # Compared by bits, so -0.0 and 0.0 stay apart.
                        memcpy(&previous_bits, &previous, sizeof(float))
                        if raw != last_alpha_raw or previous_bits != last_alpha_in:
                            last_alpha_out = accumulate_plane(previous, raw, 1.0)
                            last_alpha_raw = raw
                            last_alpha_in = previous_bits
                        source_alpha[i, j] = last_alpha_out
                    if has_shape:
                        if full_row:
                            shape = full_shape[j]
                        else:
                            shape = <unsigned char> rint(product * 255.0)
                        previous = source_shape[i, j]
                        memcpy(&previous_bits, &previous, sizeof(float))
                        if shape != last_shape or previous_bits != last_shape_in:
                            last_shape_out = accumulate_plane(previous, shape, shape_scale)
                            last_shape = shape
                            last_shape_in = previous_bits
                        source_shape[i, j] = last_shape_out
    finally:
        PyMem_Free(columns)
