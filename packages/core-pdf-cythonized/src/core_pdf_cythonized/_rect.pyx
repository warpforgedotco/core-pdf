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
