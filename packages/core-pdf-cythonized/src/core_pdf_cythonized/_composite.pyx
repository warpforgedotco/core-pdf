# SPDX-License-Identifier: AGPL-3.0-only
"""Elementary group compositing, opaque normal case
(core_pdf.impl.render.target.composite_nonisolated_group).

Every transparency group on a text page is an elementary knockout group around
one glyph, and a census over three corpus pages puts every one of them on the
branch this file owns: opacity exactly 1.0, blend mode normal or absent, no
soft mask. The planes are tiny -- a median of 8 to 32 pixels -- and the numpy
original spent about nine array operations on each: a float64 promotion, two
multiplies, a rint, a uint8 cast, a comparison, an any(), and a pair of
boolean-indexed gathers to copy the visible pixels across.

The views are declared with arbitrary strides on purpose. The destination is
normally a row slice of the page buffer, so it is not contiguous and cannot be
reshaped or ravelled without numpy quietly handing back a copy -- which would
take every write with it.

Two passes, because the domain check has to finish before any pixel is
written: a partial copy followed by a raised exception would leave the
destination half-composited.
"""

from libc.math cimport rint

import numpy


def composite_elementary_normal(destination, rendered, source_alpha):
    """Composite ``rendered`` over ``destination`` where coverage is non-zero.

    ``source_alpha`` holds coverage in [0, 1]; values outside it are rejected
    rather than cast, because numpy's float-to-uint8 conversion is a bare C
    cast whose out-of-range result differs between x86 and ARM. No coverage
    plane core-pdf builds can leave that range.

    Returns the effective alpha plane, quantized exactly as the original did.
    """
    cdef const float[:, :] alpha = numpy.asarray(source_alpha, dtype=numpy.float32)
    cdef unsigned char[:, :, :] dst = destination
    cdef const unsigned char[:, :, :] src = rendered

    cdef Py_ssize_t height = alpha.shape[0]
    cdef Py_ssize_t width = alpha.shape[1]
    if dst.shape[0] != height or dst.shape[1] != width or dst.shape[2] != 4:
        raise ValueError("destination must be source_alpha.shape + (4,)")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")

    effective = numpy.empty((height, width), dtype=numpy.uint8)
    cdef unsigned char[:, ::1] out = effective

    cdef Py_ssize_t y, x, c
    cdef double scaled
    cdef unsigned char quantized
    cdef bint any_visible = False

    for y in range(height):
        for x in range(width):
            # float32 widens to float64 exactly, and the original's extra
            # multiply by an opacity of exactly 1.0 cannot change a bit.
            scaled = rint(<double> alpha[y, x] * 255.0)
            if scaled < 0.0 or scaled > 255.0:
                raise ValueError("source_alpha must lie in [0, 1]")
            quantized = <unsigned char> <int> scaled
            out[y, x] = quantized
            if quantized > 0:
                any_visible = True

    if not any_visible:
        return effective

    with nogil:
        for y in range(height):
            for x in range(width):
                if out[y, x] > 0:
                    for c in range(4):
                        dst[y, x, c] = src[y, x, c]

    return effective
