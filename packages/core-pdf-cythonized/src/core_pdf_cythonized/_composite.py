# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports.core_pdf_cythonized._byte_clamp import (
    double_to_byte,
    float_to_byte,
    unit_to_byte_checked,
)
from cython.cimports.libc.math import isfinite, rint, rintf

Strides = cython.struct(
    pixel=cython.Py_ssize_t,
    channel=cython.Py_ssize_t,
)


def composite_elementary_normal(destination, rendered, source_alpha):
    alpha: cython.const[cython.float][:, :] = numpy.asarray(source_alpha, dtype=numpy.float32)
    dst: cython.uchar[:, :, :] = destination
    src: cython.const[cython.uchar][:, :, :] = rendered

    height: cython.Py_ssize_t = alpha.shape[0]
    width: cython.Py_ssize_t = alpha.shape[1]
    if dst.shape[0] != height or dst.shape[1] != width or dst.shape[2] != 4:
        raise ValueError("destination must be source_alpha.shape + (4,)")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")

    effective = numpy.empty((height, width), dtype=numpy.uint8)
    out: cython.uchar[:, ::1] = effective

    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    c: cython.Py_ssize_t
    quantized: cython.uchar
    any_visible: cython.bint = False

    for y in range(height):
        for x in range(width):
            quantized = cython.cast(
                cython.uchar, unit_to_byte_checked(cython.cast(cython.double, alpha[y, x]))
            )
            out[y, x] = quantized
            if quantized > 0:
                any_visible = True

    if not any_visible:
        return effective

    with cython.nogil:
        for y in range(height):
            for x in range(width):
                if out[y, x] > 0:
                    for c in range(4):
                        dst[y, x, c] = src[y, x, c]

    return effective


def composite_masked_normal(destination, rendered, opacity: cython.double, mask_codes, mask_table):
    codes: cython.const[cython.uchar][:, :] = mask_codes
    table: cython.const[cython.float][::1] = mask_table
    dst: cython.uchar[:, :, :] = destination
    src: cython.const[cython.uchar][:, :, :] = rendered

    if table.shape[0] != 256:
        raise ValueError("mask_table must hold 256 values")
    height: cython.Py_ssize_t = codes.shape[0]
    width: cython.Py_ssize_t = codes.shape[1]
    if dst.shape[0] != height or dst.shape[1] != width or dst.shape[2] != 4:
        raise ValueError("destination must be mask_codes.shape + (4,)")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")

    effective = numpy.empty((height, width), dtype=numpy.uint8)
    out: cython.uchar[:, ::1] = effective

    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    scaled: cython.double
    any_visible: cython.bint = False
    last_alpha: cython.int = -1
    last_code: cython.int = -1
    alpha_in: cython.int
    code: cython.int
    last_out: cython.uchar = 0

    for y in range(height):
        for x in range(width):
            alpha_in = src[y, x, 3]
            code = codes[y, x]
            if alpha_in != last_alpha or code != last_code:
                scaled = rint(
                    cython.cast(cython.double, alpha_in)
                    * opacity
                    * cython.cast(cython.double, table[code])
                )
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

    src_a: cython.double
    one_minus_src_a: cython.double
    dst_a: cython.double
    out_a: cython.double
    with cython.nogil:
        for y in range(height):
            for x in range(width):
                if out[y, x] == 0:
                    continue
                src_a = cython.cast(cython.double, out[y, x]) / 255.0
                one_minus_src_a = 1.0 - src_a
                dst_a = cython.cast(cython.double, dst[y, x, 3]) / 255.0
                out_a = src_a + dst_a * one_minus_src_a
                dst[y, x, 0] = normal_channel(
                    src[y, x, 0], dst[y, x, 0], src_a, dst_a, one_minus_src_a, out_a
                )
                dst[y, x, 1] = normal_channel(
                    src[y, x, 1], dst[y, x, 1], src_a, dst_a, one_minus_src_a, out_a
                )
                dst[y, x, 2] = normal_channel(
                    src[y, x, 2], dst[y, x, 2], src_a, dst_a, one_minus_src_a, out_a
                )
                dst[y, x, 3] = double_to_byte(rint(out_a * 255.0))

    return effective


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def normal_channel(
    source: cython.uchar,
    backdrop: cython.uchar,
    src_a: cython.double,
    dst_a: cython.double,
    one_minus_src_a: cython.double,
    out_a: cython.double,
) -> cython.uchar:
    colour: cython.double = cython.cast(cython.double, source) / 255.0
    return double_to_byte(
        rint(
            (
                (colour * 255.0) * src_a
                + cython.cast(cython.double, backdrop) * dst_a * one_minus_src_a
            )
            / out_a
        )
    )


def composite_normal_group(
    destination,
    rendered,
    source_alpha_scale: cython.double,
    target_alpha_scale: cython.double = 1.0,
    effective_plane: cython.bint = False,
):
    if not (isfinite(source_alpha_scale) and isfinite(target_alpha_scale)):
        raise ValueError("alpha scales must be finite")
    dst: cython.uchar[:, :, :] = destination
    src: cython.const[cython.uchar][:, :, :] = rendered
    height: cython.Py_ssize_t = dst.shape[0]
    width: cython.Py_ssize_t = dst.shape[1]
    if dst.shape[2] != 4:
        raise ValueError("destination must have four channels")
    if src.shape[0] != height or src.shape[1] != width or src.shape[2] != 4:
        raise ValueError("rendered and destination must have the same shape")
    plane = None
    if height == 0 or width == 0 or source_alpha_scale <= 0.0:
        if effective_plane:
            plane = numpy.zeros((height, width), dtype=numpy.uint8)
        return plane
    d: Strides = Strides(dst.strides[1], dst.strides[2])
    r: Strides = Strides(src.strides[1], src.strides[2])

    effective = cython.declare(cython.double[256])
    general_alpha = cython.declare(cython.float[256])
    effective_bytes = cython.declare(cython.uchar[256])
    value: cython.int
    for value in range(256):
        effective_bytes[value] = effective_alpha(
            cython.cast(cython.uchar, value), source_alpha_scale, target_alpha_scale
        )
        effective[value] = effective_bytes[value]
        general_alpha[value] = general_source_alpha(
            cython.cast(cython.uchar, value), source_alpha_scale, target_alpha_scale
        )

    out: cython.uchar[:, ::1]
    if effective_plane:
        plane = numpy.empty((height, width), dtype=numpy.uint8)
        out = plane
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    alpha: cython.uchar
    source_or: cython.uchar = 0
    source_and: cython.uchar = 255
    backdrop_or: cython.uchar = 0
    backdrop_and: cython.uchar = 255
    source_row: cython.p_const_uchar
    backdrop_row: cython.p_uchar
    with cython.nogil:
        for y in range(height):
            source_row = cython.address(src[y, 0, 0])
            backdrop_row = cython.address(dst[y, 0, 0])
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

    with cython.nogil:
        if source_alpha_scale == 1.0 and target_alpha_scale == 1.0 and source_and == 255:
            for y in range(height):
                source_row = cython.address(src[y, 0, 0])
                backdrop_row = cython.address(dst[y, 0, 0])
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


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def effective_alpha(
    alpha: cython.uchar, source_scale: cython.double, target_scale: cython.double
) -> cython.uchar:
    value: cython.double = rint(cython.cast(cython.double, alpha) * source_scale)
    if target_scale != 1.0:
        value = rint(value * target_scale)
    return double_to_byte(value)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def general_source_alpha(
    alpha: cython.uchar, source_scale: cython.double, target_scale: cython.double
) -> cython.float:
    SCALE: cython.float = 255.0
    value: cython.float = rintf(
        cython.cast(cython.float, alpha) * cython.cast(cython.float, source_scale)
    )
    if target_scale != 1.0:
        value = rintf(value * cython.cast(cython.float, target_scale))
    return value / SCALE


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def opaque_backdrop(
    dst: typing.Optional[cython.uchar[:, :, :]],
    src: typing.Optional[cython.const[cython.uchar][:, :, :]],
    height: cython.Py_ssize_t,
    width: cython.Py_ssize_t,
    d: Strides,
    r: Strides,
    effective: cython.p_const_double,
) -> cython.void:
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    c: cython.Py_ssize_t
    alpha: cython.double
    source_pixel: cython.p_const_uchar
    backdrop_pixel: cython.p_uchar
    for y in range(height):
        for x in range(width):
            source_pixel = cython.address(src[y, 0, 0]) + x * r.pixel
            backdrop_pixel = cython.address(dst[y, 0, 0]) + x * d.pixel
            alpha = effective[source_pixel[3 * r.channel]] / 255.0
            for c in range(3):
                backdrop_pixel[c * d.channel] = double_to_byte(
                    rint(
                        cython.cast(cython.double, source_pixel[c * r.channel]) * alpha
                        + cython.cast(cython.double, backdrop_pixel[c * d.channel]) * (1.0 - alpha)
                    )
                )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def empty_backdrop(
    dst: typing.Optional[cython.uchar[:, :, :]],
    src: typing.Optional[cython.const[cython.uchar][:, :, :]],
    height: cython.Py_ssize_t,
    width: cython.Py_ssize_t,
    d: Strides,
    r: Strides,
    effective: cython.p_const_double,
) -> cython.void:
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    alpha: cython.double
    source_pixel: cython.p_const_uchar
    backdrop_pixel: cython.p_uchar
    for y in range(height):
        for x in range(width):
            source_pixel = cython.address(src[y, 0, 0]) + x * r.pixel
            alpha = effective[source_pixel[3 * r.channel]]
            if alpha > 0.0:
                backdrop_pixel = cython.address(dst[y, 0, 0]) + x * d.pixel
                backdrop_pixel[0] = source_pixel[0]
                backdrop_pixel[d.channel] = source_pixel[r.channel]
                backdrop_pixel[2 * d.channel] = source_pixel[2 * r.channel]
                backdrop_pixel[3 * d.channel] = cython.cast(
                    cython.uchar, cython.cast(cython.int, alpha)
                )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def general(
    dst: typing.Optional[cython.uchar[:, :, :]],
    src: typing.Optional[cython.const[cython.uchar][:, :, :]],
    height: cython.Py_ssize_t,
    width: cython.Py_ssize_t,
    d: Strides,
    r: Strides,
    source_alpha: cython.p_const_float,
) -> cython.void:
    ZERO: cython.float = 0.0
    ONE: cython.float = 1.0
    SCALE: cython.float = 255.0
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    c: cython.Py_ssize_t
    sa: cython.float
    da: cython.float
    oa: cython.float
    weight: cython.float
    source_pixel: cython.p_const_uchar
    backdrop_pixel: cython.p_uchar
    for y in range(height):
        for x in range(width):
            source_pixel = cython.address(src[y, 0, 0]) + x * r.pixel
            sa = source_alpha[source_pixel[3 * r.channel]]
            if not sa > ZERO:
                continue
            backdrop_pixel = cython.address(dst[y, 0, 0]) + x * d.pixel
            da = cython.cast(cython.float, backdrop_pixel[3 * d.channel]) / SCALE
            oa = sa + da * (ONE - sa)
            for c in range(3):
                weight = cython.cast(cython.float, backdrop_pixel[c * d.channel]) * da * (ONE - sa)
                backdrop_pixel[c * d.channel] = float_to_byte(
                    rintf(
                        (cython.cast(cython.float, source_pixel[c * r.channel]) * sa + weight) / oa
                    )
                )
            backdrop_pixel[3 * d.channel] = float_to_byte(rintf(oa * SCALE))
