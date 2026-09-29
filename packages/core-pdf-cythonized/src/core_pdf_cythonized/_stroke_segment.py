# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._pixel_blend import blend_normal_pixel, coverage_alpha

__all__ = ("stroke_segment_samples",)

OFFSETS = cython.declare(cython.double[4], [0.125, 0.375, 0.625, 0.875])


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def segment_samples(
    pixels: cython.p_uchar,
    row_stride: cython.Py_ssize_t,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    x0: cython.double,
    y0: cython.double,
    dx: cython.double,
    dy: cython.double,
    seg_len2: cython.double,
    half2: cython.double,
    projection_extension: cython.double,
    red: cython.int,
    green: cython.int,
    blue: cython.int,
    alpha: cython.int,
    allowed: cython.p_const_uchar,
    counts: cython.p_uchar,
    covered_box: cython.p_Py_ssize_t,
) -> cython.bint:
    cross_limit: cython.double = half2 * seg_len2
    px: cython.Py_ssize_t
    py: cython.Py_ssize_t
    sx: cython.Py_ssize_t
    sy: cython.Py_ssize_t
    page_x = cython.declare(cython.double[4])
    page_y = cython.declare(cython.double[4])
    offset_x: cython.double
    offset_y: cython.double
    projection: cython.double
    cross: cython.double
    covered: cython.int
    sa: cython.int
    low_x: cython.Py_ssize_t = ix1
    low_y: cython.Py_ssize_t = iy1
    high_x: cython.Py_ssize_t = ix0
    high_y: cython.Py_ssize_t = iy0
    box_width: cython.Py_ssize_t = ix1 - ix0
    for py in range(iy0, iy1):
        for sy in range(4):
            page_y[sy] = crop_y1 - (cython.cast(cython.double, py) + OFFSETS[sy]) / scale
        for px in range(ix0, ix1):
            if allowed != cython.NULL and not allowed[(py - iy0) * box_width + (px - ix0)]:
                continue
            for sx in range(4):
                page_x[sx] = crop_x0 + (cython.cast(cython.double, px) + OFFSETS[sx]) / scale
            covered = 0
            for sy in range(4):
                offset_y = page_y[sy] - y0
                for sx in range(4):
                    offset_x = page_x[sx] - x0
                    projection = offset_x * dx + offset_y * dy
                    if (
                        projection < -projection_extension
                        or projection > seg_len2 + projection_extension
                    ):
                        continue
                    cross = offset_x * dy - offset_y * dx
                    if cross * cross <= cross_limit:
                        covered += 1
            if not covered:
                continue
            if px < low_x:
                low_x = px
            if px + 1 > high_x:
                high_x = px + 1
            if py < low_y:
                low_y = py
            if py + 1 > high_y:
                high_y = py + 1
            if counts != cython.NULL:
                counts[(py - iy0) * box_width + (px - ix0)] = cython.cast(cython.uchar, covered)
                continue
            sa = coverage_alpha(alpha, covered)
            if sa <= 0:
                continue
            blend_normal_pixel(
                pixels + (py - iy0) * row_stride + (px - ix0) * 4, red, green, blue, sa
            )
    covered_box[0] = low_x
    covered_box[1] = low_y
    covered_box[2] = high_x
    covered_box[3] = high_y
    return high_x > low_x


def stroke_segment_samples(
    pixels: cython.uchar[:, :, ::1],
    origin_x: cython.Py_ssize_t,
    origin_y: cython.Py_ssize_t,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    x0: cython.double,
    y0: cython.double,
    dx: cython.double,
    dy: cython.double,
    seg_len2: cython.double,
    half2: cython.double,
    projection_extension: cython.double,
    red: cython.int,
    green: cython.int,
    blue: cython.int,
    alpha: cython.int,
    allowed: typing.Optional[cython.const[cython.uchar][::1]] = None,
    counts: typing.Optional[cython.uchar[:, ::1]] = None,
):
    if (
        ix0 < origin_x
        or iy0 < origin_y
        or ix1 > origin_x + pixels.shape[1]
        or iy1 > origin_y + pixels.shape[0]
    ):
        raise ValueError("segment box runs past the pixels given")
    box_width: cython.Py_ssize_t = ix1 - ix0
    if allowed is not None and allowed.shape[0] != box_width * (iy1 - iy0):
        raise ValueError("allowed must hold one byte per pixel of the box")
    if counts is not None and (counts.shape[0] != iy1 - iy0 or counts.shape[1] != box_width):
        raise ValueError("counts must hold one byte per pixel of the box")
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    box = cython.declare(cython.Py_ssize_t[4])
    any_covered: cython.bint
    allowed_bytes: cython.p_const_uchar = (
        cython.address(allowed[0]) if allowed is not None and allowed.shape[0] else cython.NULL
    )
    count_bytes: cython.p_uchar = (
        cython.address(counts[0, 0]) if counts is not None else cython.NULL
    )
    origin: cython.p_uchar = cython.address(pixels[iy0 - origin_y, ix0 - origin_x, 0])
    with cython.nogil:
        any_covered = segment_samples(
            origin,
            pixels.strides[0],
            ix0,
            iy0,
            ix1,
            iy1,
            crop_x0,
            crop_y1,
            scale,
            x0,
            y0,
            dx,
            dy,
            seg_len2,
            half2,
            projection_extension,
            red,
            green,
            blue,
            alpha,
            allowed_bytes,
            count_bytes,
            box,
        )
    if not any_covered:
        return None
    return box[0], box[1], box[2], box[3]
