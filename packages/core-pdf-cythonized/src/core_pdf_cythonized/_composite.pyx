# SPDX-License-Identifier: AGPL-3.0-only
"""Group compositing on the normal blend mode
(core_pdf.impl.render.target.composite_nonisolated_group and
composite_masked_group, and core_pdf.impl.render.blend's
composite_normal_group_numpy).

composite_elementary_normal owns the opaque, unmasked elementary case;
composite_masked_normal owns the isolated soft-masked case;
composite_normal_group owns an isolated, unmasked group. The notes below are
about the first; the others have their own docstrings.

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

from libc.math cimport isfinite, rint, rintf

import numpy


cdef struct Strides:
    Py_ssize_t pixel
    Py_ssize_t channel


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


def composite_normal_group(
    destination, rendered, double source_alpha_scale, double target_alpha_scale=1.0
):
    """Composite an isolated, unmasked group onto its backdrop, normal blend mode.

    The numpy original chose one of four routes by looking at the whole plane
    -- a straight copy when every source pixel is opaque and unscaled, a
    float64 blend when every backdrop pixel is opaque, a copy of the visible
    pixels when no backdrop pixel has alpha, the general float32 composite
    otherwise -- and ran each as a handful of full-plane array passes. On
    PyMuPDF test_3450 that is 2,556 groups of about 80,000 pixels, nearly all
    onto an empty backdrop. A read pass here settles the route, and a write
    pass runs it.

    Each route keeps the original's arithmetic width: the effective alpha and
    the opaque-backdrop blend in float64, because numpy promoted uint8 times a
    Python float to float64; the general route in float32, where the scale was
    cast to float32 before it multiplied. Either alpha is a function of the
    source alpha byte alone, so it is computed once per byte value into a
    table, by the same expressions. Scales must be finite -- a NaN would reach
    a float-to-byte cast whose result numpy leaves to the platform -- and the
    caller's are clamped to [0, 1]. A row's pixels must be packed, as in every
    plane core-pdf composites.
    """
    if not (isfinite(source_alpha_scale) and isfinite(target_alpha_scale)):
        raise ValueError("alpha scales must be finite")
    cdef unsigned char[:, :, :] dst = destination
    cdef const unsigned char[:, :, :] src = rendered
    cdef Py_ssize_t height = dst.shape[0]
    cdef Py_ssize_t width = dst.shape[1]
    if dst.shape[2] != 4:
        raise ValueError("destination must have four channels")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")
    if height == 0 or width == 0 or source_alpha_scale <= 0.0:
        return
    # Any strides, as the numpy original took: a group's window into the page
    # is strided by row, and a test composites through every other pixel.
    cdef Strides d = Strides(dst.strides[1], dst.strides[2])
    cdef Strides r = Strides(src.strides[1], src.strides[2])

    cdef Py_ssize_t y, x
    cdef unsigned char source_or = 0, source_and = 255
    cdef unsigned char backdrop_or = 0, backdrop_and = 255
    cdef const unsigned char *source_row
    cdef unsigned char *backdrop_row
    with nogil:
        for y in range(height):
            source_row = &src[y, 0, 0]
            backdrop_row = &dst[y, 0, 0]
            for x in range(width):
                source_or |= source_row[x * r.pixel + 3 * r.channel]
                source_and &= source_row[x * r.pixel + 3 * r.channel]
                backdrop_or |= backdrop_row[x * d.pixel + 3 * d.channel]
                backdrop_and &= backdrop_row[x * d.pixel + 3 * d.channel]
    if not source_or:
        return

    cdef double effective[256]
    cdef float general_alpha[256]
    cdef int value
    for value in range(256):
        effective[value] = effective_alpha(<unsigned char> value, source_alpha_scale, target_alpha_scale)
        general_alpha[value] = general_source_alpha(
            <unsigned char> value, source_alpha_scale, target_alpha_scale
        )

    with nogil:
        if source_alpha_scale == 1.0 and target_alpha_scale == 1.0 and source_and == 255:
            for y in range(height):
                source_row = &src[y, 0, 0]
                backdrop_row = &dst[y, 0, 0]
                for x in range(width):
                    for value in range(4):
                        backdrop_row[x * d.pixel + value * d.channel] = source_row[
                            x * r.pixel + value * r.channel
                        ]
        elif source_alpha_scale <= 1.0 and target_alpha_scale <= 1.0 and backdrop_and == 255:
            opaque_backdrop(dst, src, height, width, d, r, effective)
        elif not backdrop_or:
            empty_backdrop(dst, src, height, width, d, r, effective)
        else:
            general(dst, src, height, width, d, r, general_alpha)


cdef double effective_alpha(unsigned char alpha, double source_scale, double target_scale) noexcept nogil:
    # rint(alpha * s), then rint(* t) only when t is not 1, then clip: float64.
    cdef double value = rint(<double> alpha * source_scale)
    if target_scale != 1.0:
        value = rint(value * target_scale)
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value


cdef float general_source_alpha(unsigned char alpha, double source_scale, double target_scale) noexcept nogil:
    # The general route's source alpha: float32, the scales cast first, no clip.
    cdef float SCALE = 255.0
    cdef float value = rintf(<float> alpha * <float> source_scale)
    if target_scale != 1.0:
        value = rintf(value * <float> target_scale)
    return value / SCALE


cdef inline unsigned char clamp_to_byte(double value) noexcept nogil:
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return <unsigned char> <int> value


cdef void opaque_backdrop(
    unsigned char[:, :, :] dst,
    const unsigned char[:, :, :] src,
    Py_ssize_t height,
    Py_ssize_t width,
    Strides d,
    Strides r,
    const double *effective,
) noexcept nogil:
    # Every backdrop pixel is opaque, so alpha stays 255 and only colour moves.
    cdef Py_ssize_t y, x, c
    cdef double alpha
    cdef const unsigned char *source_pixel
    cdef unsigned char *backdrop_pixel
    for y in range(height):
        for x in range(width):
            source_pixel = &src[y, 0, 0] + x * r.pixel
            backdrop_pixel = &dst[y, 0, 0] + x * d.pixel
            alpha = effective[source_pixel[3 * r.channel]] / 255.0
            for c in range(3):
                backdrop_pixel[c * d.channel] = clamp_to_byte(
                    rint(
                        <double> source_pixel[c * r.channel] * alpha
                        + <double> backdrop_pixel[c * d.channel] * (1.0 - alpha)
                    )
                )


cdef void empty_backdrop(
    unsigned char[:, :, :] dst,
    const unsigned char[:, :, :] src,
    Py_ssize_t height,
    Py_ssize_t width,
    Strides d,
    Strides r,
    const double *effective,
) noexcept nogil:
    # No backdrop pixel has alpha: a visible source pixel is copied across with
    # its effective alpha, and the rest keep their bytes.
    cdef Py_ssize_t y, x
    cdef double alpha
    cdef const unsigned char *source_pixel
    cdef unsigned char *backdrop_pixel
    for y in range(height):
        for x in range(width):
            source_pixel = &src[y, 0, 0] + x * r.pixel
            alpha = effective[source_pixel[3 * r.channel]]
            if alpha > 0.0:
                backdrop_pixel = &dst[y, 0, 0] + x * d.pixel
                backdrop_pixel[0] = source_pixel[0]
                backdrop_pixel[d.channel] = source_pixel[r.channel]
                backdrop_pixel[2 * d.channel] = source_pixel[2 * r.channel]
                backdrop_pixel[3 * d.channel] = <unsigned char> <int> alpha


cdef void general(
    unsigned char[:, :, :] dst,
    const unsigned char[:, :, :] src,
    Py_ssize_t height,
    Py_ssize_t width,
    Strides d,
    Strides r,
    const float *source_alpha,
) noexcept nogil:
    # float32 throughout, and constants typed float so no expression widens.
    cdef float ZERO = 0.0
    cdef float ONE = 1.0
    cdef float SCALE = 255.0
    cdef Py_ssize_t y, x, c
    cdef float sa, da, oa, weight
    cdef const unsigned char *source_pixel
    cdef unsigned char *backdrop_pixel
    for y in range(height):
        for x in range(width):
            source_pixel = &src[y, 0, 0] + x * r.pixel
            sa = source_alpha[source_pixel[3 * r.channel]]
            if not sa > ZERO:
                continue
            backdrop_pixel = &dst[y, 0, 0] + x * d.pixel
            da = <float> backdrop_pixel[3 * d.channel] / SCALE
            oa = sa + da * (ONE - sa)
            for c in range(3):
                weight = <float> backdrop_pixel[c * d.channel] * da * (ONE - sa)
                backdrop_pixel[c * d.channel] = float_to_byte(
                    rintf((<float> source_pixel[c * r.channel] * sa + weight) / oa)
                )
            backdrop_pixel[3 * d.channel] = float_to_byte(rintf(oa * SCALE))


cdef inline unsigned char float_to_byte(float value) noexcept nogil:
    cdef float ZERO = 0.0
    cdef float SCALE = 255.0
    if value < ZERO:
        return 0
    if value > SCALE:
        return 255
    return <unsigned char> <int> value
