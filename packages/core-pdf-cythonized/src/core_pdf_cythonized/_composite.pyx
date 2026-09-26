# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport isfinite, rint, rintf

import numpy

from core_pdf_cythonized._byte_clamp cimport double_to_byte, float_to_byte, unit_to_byte_checked


cdef struct Strides:
    Py_ssize_t pixel
    Py_ssize_t channel


def composite_elementary_normal(destination, rendered, source_alpha):
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
    cdef unsigned char quantized
    cdef bint any_visible = False

    for y in range(height):
        for x in range(width):
            quantized = <unsigned char> unit_to_byte_checked(<double> alpha[y, x])
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


def composite_masked_normal(destination, rendered, double opacity, mask_codes, mask_table):
    cdef const unsigned char[:, :] codes = mask_codes
    cdef const float[::1] table = mask_table
    cdef unsigned char[:, :, :] dst = destination
    cdef const unsigned char[:, :, :] src = rendered

    if table.shape[0] != 256:
        raise ValueError("mask_table must hold 256 values")
    cdef Py_ssize_t height = codes.shape[0]
    cdef Py_ssize_t width = codes.shape[1]
    if dst.shape[0] != height or dst.shape[1] != width or dst.shape[2] != 4:
        raise ValueError("destination must be mask_codes.shape + (4,)")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")

    effective = numpy.empty((height, width), dtype=numpy.uint8)
    cdef unsigned char[:, ::1] out = effective

    cdef Py_ssize_t y, x
    cdef double scaled
    cdef bint any_visible = False
    cdef int last_alpha = -1, last_code = -1, alpha_in, code
    cdef unsigned char last_out = 0

    for y in range(height):
        for x in range(width):
            alpha_in = src[y, x, 3]
            code = codes[y, x]
            if alpha_in != last_alpha or code != last_code:
                scaled = rint(<double> alpha_in * opacity * <double> table[code])
                if scaled != scaled:
                    raise ValueError("effective alpha is NaN")
                last_out = double_to_byte(scaled)
                last_alpha = alpha_in
                last_code = code
            out[y, x] = last_out
            if last_out > 0:
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
                out_a = src_a + dst_a * one_minus_src_a
                dst[y, x, 0] = normal_channel(src[y, x, 0], dst[y, x, 0], src_a, dst_a, one_minus_src_a, out_a)
                dst[y, x, 1] = normal_channel(src[y, x, 1], dst[y, x, 1], src_a, dst_a, one_minus_src_a, out_a)
                dst[y, x, 2] = normal_channel(src[y, x, 2], dst[y, x, 2], src_a, dst_a, one_minus_src_a, out_a)
                dst[y, x, 3] = double_to_byte(rint(out_a * 255.0))

    return effective


cdef inline unsigned char normal_channel(
    unsigned char source,
    unsigned char backdrop,
    double src_a,
    double dst_a,
    double one_minus_src_a,
    double out_a,
) noexcept nogil:
    cdef double colour = <double> source / 255.0
    return double_to_byte(
        rint(((colour * 255.0) * src_a + <double> backdrop * dst_a * one_minus_src_a) / out_a)
    )


def composite_normal_group(
    destination,
    rendered,
    double source_alpha_scale,
    double target_alpha_scale=1.0,
    bint effective_plane=False,
):
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
    plane = None
    if height == 0 or width == 0 or source_alpha_scale <= 0.0:
        if effective_plane:
            plane = numpy.zeros((height, width), dtype=numpy.uint8)
        return plane
    cdef Strides d = Strides(dst.strides[1], dst.strides[2])
    cdef Strides r = Strides(src.strides[1], src.strides[2])

    cdef double effective[256]
    cdef float general_alpha[256]
    cdef unsigned char effective_bytes[256]
    cdef int value
    for value in range(256):
        effective_bytes[value] = effective_alpha(
            <unsigned char> value, source_alpha_scale, target_alpha_scale
        )
        effective[value] = effective_bytes[value]
        general_alpha[value] = general_source_alpha(
            <unsigned char> value, source_alpha_scale, target_alpha_scale
        )

    cdef unsigned char[:, ::1] out
    if effective_plane:
        plane = numpy.empty((height, width), dtype=numpy.uint8)
        out = plane
    cdef Py_ssize_t y, x
    cdef unsigned char alpha
    cdef unsigned char source_or = 0, source_and = 255
    cdef unsigned char backdrop_or = 0, backdrop_and = 255
    cdef const unsigned char *source_row
    cdef unsigned char *backdrop_row
    with nogil:
        for y in range(height):
            source_row = &src[y, 0, 0]
            backdrop_row = &dst[y, 0, 0]
            for x in range(width):
                alpha = source_row[x * r.pixel + 3 * r.channel]
                source_or |= alpha
                source_and &= alpha
                backdrop_or |= backdrop_row[x * d.pixel + 3 * d.channel]
                backdrop_and &= backdrop_row[x * d.pixel + 3 * d.channel]
                if effective_plane:
                    out[y, x] = effective_bytes[alpha]
    if not source_or:
        return plane

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
    return plane


cdef unsigned char effective_alpha(
    unsigned char alpha, double source_scale, double target_scale
) noexcept nogil:
    cdef double value = rint(<double> alpha * source_scale)
    if target_scale != 1.0:
        value = rint(value * target_scale)
    return double_to_byte(value)


cdef float general_source_alpha(unsigned char alpha, double source_scale, double target_scale) noexcept nogil:
    cdef float SCALE = 255.0
    cdef float value = rintf(<float> alpha * <float> source_scale)
    if target_scale != 1.0:
        value = rintf(value * <float> target_scale)
    return value / SCALE


cdef void opaque_backdrop(
    unsigned char[:, :, :] dst,
    const unsigned char[:, :, :] src,
    Py_ssize_t height,
    Py_ssize_t width,
    Strides d,
    Strides r,
    const double *effective,
) noexcept nogil:
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
                backdrop_pixel[c * d.channel] = double_to_byte(
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
