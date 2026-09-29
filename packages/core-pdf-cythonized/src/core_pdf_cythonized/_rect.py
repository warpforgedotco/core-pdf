# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._alpha_blend import (
    accumulate_plane,
    blend_one,
    opaque_channel,
)
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.math import rint
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def axis_coverage(index: cython.double, low: cython.double, high: cython.double) -> cython.double:
    upper: cython.double = index + 1.0
    if high < upper:
        upper = high
    lower: cython.double = index
    if low > lower:
        lower = low
    span: cython.double = upper - lower
    if span < 0.0:
        return 0.0
    if span > 1.0:
        return 1.0
    return span


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_rect_pixels(
    base: cython.p_uchar,
    row_stride: cython.Py_ssize_t,
    ix0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    left: cython.double,
    right: cython.double,
    top: cython.double,
    bottom: cython.double,
    red_byte: cython.int,
    green_byte: cython.int,
    blue_byte: cython.int,
    cap: cython.int,
) -> cython.int:
    width: cython.Py_ssize_t = ix1 - ix0
    height: cython.Py_ssize_t = iy1 - iy0
    if width <= 0 or height <= 0:
        return 0
    red: cython.float = cython.cast(cython.float, red_byte)
    green: cython.float = cython.cast(cython.float, green_byte)
    blue: cython.float = cython.cast(cython.float, blue_byte)
    alpha: cython.double = cython.cast(cython.double, cap)
    opaque_from: cython.int = 255 if cap >= 255 else 256
    opaque = cython.declare(cython.uchar[4])
    opaque[0] = opaque_channel(red)
    opaque[1] = opaque_channel(green)
    opaque[2] = opaque_channel(blue)
    opaque[3] = 255
    columns: cython.p_double = cython.cast(
        cython.p_double,
        malloc(width * (cython.sizeof(cython.double) + 1)),
    )
    if columns == cython.NULL:
        return -1
    full_raw: cython.p_uchar = cython.cast(cython.p_uchar, columns + width)
    row: cython.p_uchar
    pixel: cython.p_uchar
    full_row: cython.bint
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    run_start: cython.Py_ssize_t = 0
    run_end: cython.Py_ssize_t = 0
    segment_end: cython.Py_ssize_t
    resume: cython.Py_ssize_t
    row_coverage: cython.double
    raw: cython.uchar
    backdrop = cython.declare(cython.uint)
    last_backdrop: cython.uint = 0
    last_raw: cython.int = -1
    blended = cython.declare(cython.uchar[4])
    blended[0] = blended[1] = blended[2] = blended[3] = 0
    for j in range(width):
        columns[j] = axis_coverage(cython.cast(cython.double, ix0 + j), left, right)
        full_raw[j] = cython.cast(cython.uchar, rint(columns[j] * alpha))
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
        row_coverage = axis_coverage(cython.cast(cython.double, iy0 + i), top, bottom)
        full_row = row_coverage == 1.0
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
                raw = cython.cast(cython.uchar, rint(row_coverage * columns[j] * alpha))
            if raw != 0:
                pixel = row + 4 * j
                if raw >= opaque_from:
                    memcpy(pixel, opaque, 4)
                else:
                    memcpy(cython.address(backdrop), pixel, 4)
                    if raw != last_raw or backdrop != last_backdrop:
                        memcpy(blended, pixel, 4)
                        blend_one(
                            cython.address(blended[0]),
                            cython.address(blended[1]),
                            cython.address(blended[2]),
                            cython.address(blended[3]),
                            raw,
                            cap,
                            red,
                            green,
                            blue,
                        )
                        last_raw = raw
                        last_backdrop = backdrop
                    memcpy(pixel, blended, 4)
            j += 1
    free(columns)
    return 0


def fill_rect_coverage(
    ix0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    left: cython.double,
    right: cython.double,
    top: cython.double,
    bottom: cython.double,
    rgba: object,
    target: cython.uchar[:, :, :],
    source_alpha: typing.Optional[cython.float[:, :]],
    source_shape: typing.Optional[cython.float[:, :]],
    shape_scale: cython.double,
) -> None:
    width: cython.Py_ssize_t = ix1 - ix0
    height: cython.Py_ssize_t = iy1 - iy0
    if width <= 0 or height <= 0:
        return
    if target.shape[0] != height or target.shape[1] != width or target.shape[2] != 4:
        raise ValueError("target differs from the rectangle in shape")
    has_alpha: cython.bint = source_alpha is not None
    has_shape: cython.bint = source_shape is not None
    if has_alpha and (source_alpha.shape[0] != height or source_alpha.shape[1] != width):
        raise ValueError("source_alpha differs from the rectangle in shape")
    if has_shape and (source_shape.shape[0] != height or source_shape.shape[1] != width):
        raise ValueError("source_shape differs from the rectangle in shape")
    red_byte: cython.int = cython.cast(cython.int, rgba[0])
    green_byte: cython.int = cython.cast(cython.int, rgba[1])
    blue_byte: cython.int = cython.cast(cython.int, rgba[2])
    cap: cython.int = cython.cast(cython.int, rgba[3])
    status: cython.int
    if target.strides[1] == 4 and target.strides[2] == 1 and not has_alpha and not has_shape:
        with cython.nogil:
            status = fill_rect_pixels(
                cython.address(target[0, 0, 0]),
                target.strides[0],
                ix0,
                ix1,
                iy0,
                iy1,
                left,
                right,
                top,
                bottom,
                red_byte,
                green_byte,
                blue_byte,
                cap,
            )
        if status < 0:
            raise MemoryError
        return
    red: cython.float = cython.cast(cython.float, red_byte)
    green: cython.float = cython.cast(cython.float, green_byte)
    blue: cython.float = cython.cast(cython.float, blue_byte)
    alpha: cython.double = cython.cast(cython.double, cap)
    opaque_from: cython.int = 255 if cap >= 255 else 256
    opaque_red: cython.uchar = opaque_channel(red)
    opaque_green: cython.uchar = opaque_channel(green)
    opaque_blue: cython.uchar = opaque_channel(blue)
    columns: cython.p_double = cython.cast(
        cython.p_double,
        PyMem_Malloc(width * (cython.sizeof(cython.double) + 2)),
    )
    if columns == cython.NULL:
        raise MemoryError
    full_raw: cython.p_uchar = cython.cast(cython.p_uchar, columns + width)
    full_shape: cython.p_uchar = full_raw + width
    full_row: cython.bint
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    row_coverage: cython.double
    product: cython.double
    raw: cython.uchar
    shape: cython.uchar
    backdrop = cython.declare(cython.uint)
    last_backdrop: cython.uint = 0
    last_raw: cython.int = -1
    blended = cython.declare(cython.uchar[4])
    blended[0] = blended[1] = blended[2] = blended[3] = 0
    previous: cython.float
    last_alpha_out: cython.float = 0.0
    last_shape_out: cython.float = 0.0
    previous_bits = cython.declare(cython.uint)
    last_alpha_in: cython.uint = 0
    last_shape_in: cython.uint = 0
    last_alpha_raw: cython.int = -1
    last_shape: cython.int = -1
    try:
        with cython.nogil:
            for j in range(width):
                columns[j] = axis_coverage(cython.cast(cython.double, ix0 + j), left, right)
                full_raw[j] = cython.cast(cython.uchar, rint(columns[j] * alpha))
                full_shape[j] = cython.cast(cython.uchar, rint(columns[j] * 255.0))
            for i in range(height):
                row_coverage = axis_coverage(cython.cast(cython.double, iy0 + i), top, bottom)
                full_row = row_coverage == 1.0
                for j in range(width):
                    if full_row:
                        raw = full_raw[j]
                        product = columns[j]
                    else:
                        product = row_coverage * columns[j]
                        raw = cython.cast(cython.uchar, rint(product * alpha))
                    if raw != 0:
                        if raw >= opaque_from:
                            target[i, j, 0] = opaque_red
                            target[i, j, 1] = opaque_green
                            target[i, j, 2] = opaque_blue
                            target[i, j, 3] = 255
                        else:
                            backdrop = (
                                cython.cast(cython.uint, target[i, j, 0])
                                | (cython.cast(cython.uint, target[i, j, 1]) << 8)
                                | (cython.cast(cython.uint, target[i, j, 2]) << 16)
                                | (cython.cast(cython.uint, target[i, j, 3]) << 24)
                            )
                            if raw != last_raw or backdrop != last_backdrop:
                                blended[0] = target[i, j, 0]
                                blended[1] = target[i, j, 1]
                                blended[2] = target[i, j, 2]
                                blended[3] = target[i, j, 3]
                                blend_one(
                                    cython.address(blended[0]),
                                    cython.address(blended[1]),
                                    cython.address(blended[2]),
                                    cython.address(blended[3]),
                                    raw,
                                    cap,
                                    red,
                                    green,
                                    blue,
                                )
                                last_raw = raw
                                last_backdrop = backdrop
                            target[i, j, 0] = blended[0]
                            target[i, j, 1] = blended[1]
                            target[i, j, 2] = blended[2]
                            target[i, j, 3] = blended[3]
                    if has_alpha:
                        previous = source_alpha[i, j]
                        memcpy(
                            cython.address(previous_bits),
                            cython.address(previous),
                            cython.sizeof(cython.float),
                        )
                        if raw != last_alpha_raw or previous_bits != last_alpha_in:
                            last_alpha_out = accumulate_plane(previous, raw, 1.0)
                            last_alpha_raw = raw
                            last_alpha_in = previous_bits
                        source_alpha[i, j] = last_alpha_out
                    if has_shape:
                        if full_row:
                            shape = full_shape[j]
                        else:
                            shape = cython.cast(cython.uchar, rint(product * 255.0))
                        previous = source_shape[i, j]
                        memcpy(
                            cython.address(previous_bits),
                            cython.address(previous),
                            cython.sizeof(cython.float),
                        )
                        if shape != last_shape or previous_bits != last_shape_in:
                            last_shape_out = accumulate_plane(previous, shape, shape_scale)
                            last_shape = shape
                            last_shape_in = previous_bits
                        source_shape[i, j] = last_shape_out
    finally:
        PyMem_Free(columns)
