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
    cdef float red = <float> <int> rgba[0]
    cdef float green = <float> <int> rgba[1]
    cdef float blue = <float> <int> rgba[2]
    cdef int cap = <int> rgba[3]
    cdef double alpha = <double> cap
    cdef int opaque_from = 255 if cap >= 255 else 256
    cdef unsigned char opaque_red = opaque_channel(red)
    cdef unsigned char opaque_green = opaque_channel(green)
    cdef unsigned char opaque_blue = opaque_channel(blue)
    cdef double* columns = <double*> PyMem_Malloc(width * sizeof(double))
    if columns == NULL:
        raise MemoryError
    cdef Py_ssize_t i, j
    cdef double row_coverage, product
    cdef unsigned char raw, shape
    try:
        with nogil:
            for j in range(width):
                columns[j] = axis_coverage(<double> (ix0 + j), left, right)
            for i in range(height):
                row_coverage = axis_coverage(<double> (iy0 + i), top, bottom)
                for j in range(width):
                    product = row_coverage * columns[j]
                    raw = <unsigned char> rint(product * alpha)
                    if raw != 0:
                        if raw >= opaque_from:
                            target[i, j, 0] = opaque_red
                            target[i, j, 1] = opaque_green
                            target[i, j, 2] = opaque_blue
                            target[i, j, 3] = 255
                        else:
                            blend_one(
                                &target[i, j, 0], &target[i, j, 1], &target[i, j, 2],
                                &target[i, j, 3], raw, cap, red, green, blue,
                            )
                    if has_alpha:
                        source_alpha[i, j] = accumulate_plane(source_alpha[i, j], raw, 1.0)
                    if has_shape:
                        shape = <unsigned char> rint(product * 255.0)
                        source_shape[i, j] = accumulate_plane(
                            source_shape[i, j], shape, shape_scale
                        )
    finally:
        PyMem_Free(columns)
