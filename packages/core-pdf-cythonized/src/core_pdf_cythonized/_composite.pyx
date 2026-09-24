# SPDX-License-Identifier: AGPL-3.0-only
"""Group compositing on the normal blend mode
(core_pdf.impl.render.target.composite_nonisolated_group and
composite_masked_group).

composite_elementary_normal owns the opaque, unmasked elementary case;
composite_masked_normal owns the isolated soft-masked case. The notes below
are about the first; the second has its own docstring.

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


def composite_masked_normal(destination, rendered, double opacity, mask_alpha):
    """Composite an isolated group through a soft mask, normal blend mode.

    The numpy original quantized the effective alpha, gathered the visible
    pixels of both planes into float64 arrays through boolean masks, ran the
    normal branch of blend_channels_f64 over them, column-stacked the four
    channels and scattered them back. On test_3450 -- 5,568 such groups of
    about 80,000 pixels -- that came to roughly 40ns a pixel, nearly all of it
    in the gathers, the stack and the scatter rather than the arithmetic. This
    is the same arithmetic in one pass, in the same float64 and in the same
    order, so every intermediate lands on the same bits.

    ``mask_alpha`` must be float32, as every resolved soft mask plane is. The
    domain check -- a NaN effective alpha, whose uint8 cast numpy leaves to
    the platform -- completes before any pixel is written.

    Returns the effective alpha plane, quantized exactly as the original did.
    """
    cdef const float[:, :] mask = mask_alpha
    cdef unsigned char[:, :, :] dst = destination
    cdef const unsigned char[:, :, :] src = rendered

    cdef Py_ssize_t height = mask.shape[0]
    cdef Py_ssize_t width = mask.shape[1]
    if dst.shape[0] != height or dst.shape[1] != width or dst.shape[2] != 4:
        raise ValueError("destination must be mask_alpha.shape + (4,)")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")

    effective = numpy.empty((height, width), dtype=numpy.uint8)
    cdef unsigned char[:, ::1] out = effective

    cdef Py_ssize_t y, x
    cdef double scaled
    cdef bint any_visible = False

    for y in range(height):
        for x in range(width):
            # (alpha * opacity) * mask, left to right as numpy evaluated it,
            # with the float32 mask widened exactly to float64.
            scaled = rint(<double> src[y, x, 3] * opacity * <double> mask[y, x])
            if scaled != scaled:
                raise ValueError("effective alpha is NaN")
            if scaled < 0.0:
                scaled = 0.0
            elif scaled > 255.0:
                scaled = 255.0
            out[y, x] = <unsigned char> <int> scaled
            if out[y, x] > 0:
                any_visible = True

    if not any_visible:
        return effective

    cdef double src_a, one_minus_src_a, dst_a, out_a
    with nogil:
        for y in range(height):
            for x in range(width):
                if out[y, x] == 0:
                    continue
                src_a = <double> out[y, x] / 255.0
                one_minus_src_a = 1.0 - src_a
                dst_a = <double> dst[y, x, 3] / 255.0
                # src_a is positive here, so out_a is too: the original's
                # guard against a zero divisor and its transparent-pixel
                # override can never fire on a visible pixel.
                out_a = src_a + dst_a * one_minus_src_a
                dst[y, x, 0] = normal_channel(src[y, x, 0], dst[y, x, 0], src_a, dst_a, one_minus_src_a, out_a)
                dst[y, x, 1] = normal_channel(src[y, x, 1], dst[y, x, 1], src_a, dst_a, one_minus_src_a, out_a)
                dst[y, x, 2] = normal_channel(src[y, x, 2], dst[y, x, 2], src_a, dst_a, one_minus_src_a, out_a)
                dst[y, x, 3] = clamp_byte(rint(out_a * 255.0))

    return effective


cdef inline unsigned char normal_channel(
    unsigned char source,
    unsigned char backdrop,
    double src_a,
    double dst_a,
    double one_minus_src_a,
    double out_a,
) noexcept nogil:
    # The source colour went through /255.0 and back through *255.0 in the
    # original, which is not the identity in floating point, so it stays.
    cdef double colour = <double> source / 255.0
    return clamp_byte(
        rint(((colour * 255.0) * src_a + <double> backdrop * dst_a * one_minus_src_a) / out_a)
    )


cdef inline unsigned char clamp_byte(double value) noexcept nogil:
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return <unsigned char> <int> value
