# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._byte_clamp import float_to_byte
from cython.cimports.core_pdf_cythonized._pixel_blend import blend_normal_pixel, coverage_alpha
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min, raise_not_integral
from cython.cimports.core_pdf_cythonized._rect import fill_rect_pixels
from cython.cimports.core_pdf_cythonized._stroke import (
    CLIP_NONE,
    CLIP_ROWS,
    FAILED_INFINITY,
    FAILED_MEMORY,
    FAILED_NAN,
)
from cython.cimports.core_pdf_cythonized._stroke_segment import segment_samples
from cython.cimports.libc.math import ceil, fabs, floor, isinf, isnan, pow, rintf
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy

__all__ = ("stroke_polylines",)


Paint = cython.struct(
    pixels=cython.p_uchar,
    row_stride=cython.Py_ssize_t,
    width=cython.Py_ssize_t,
    height=cython.Py_ssize_t,
    crop_x0=cython.double,
    crop_y1=cython.double,
    scale=cython.double,
    clip_mode=cython.int,
    clip_empty=cython.bint,
    clip=cython.double[4],
    rows_origin=cython.Py_ssize_t,
    rows_count=cython.Py_ssize_t,
    row_offsets=cython.p_const_longlong,
    row_spans=cython.p_const_longlong,
    red=cython.int,
    green=cython.int,
    blue=cython.int,
    alpha=cython.int,
    exponent=cython.double,
    painted=cython.bint,
    window=cython.Py_ssize_t[4],
    error=cython.int,
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def extend(
    p: cython.pointer[Paint],
    y0: cython.Py_ssize_t,
    y1: cython.Py_ssize_t,
    x0: cython.Py_ssize_t,
    x1: cython.Py_ssize_t,
) -> cython.void:
    if not p.painted:
        p.painted = True
        p.window[0] = y0
        p.window[1] = y1
        p.window[2] = x0
        p.window[3] = x1
        return
    if y0 < p.window[0]:
        p.window[0] = y0
    if y1 > p.window[1]:
        p.window[1] = y1
    if x0 < p.window[2]:
        p.window[2] = x0
    if x1 > p.window[3]:
        p.window[3] = x1


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def checked(p: cython.pointer[Paint], value: cython.double) -> cython.bint:
    if isnan(value):
        p.error = FAILED_NAN
        return False
    if isinf(value):
        p.error = FAILED_INFINITY
        return False
    return True


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def clamped(value: cython.double, size: cython.Py_ssize_t) -> cython.Py_ssize_t:
    if value > cython.cast(cython.double, size):
        return size
    if value < 0.0:
        return 0
    return cython.cast(cython.Py_ssize_t, value)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def page_box_to_pixels(
    p: cython.pointer[Paint],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    out: cython.p_Py_ssize_t,
) -> cython.int:
    value: cython.double = (x0 - p.crop_x0) * p.scale
    if not checked(p, value):
        return -1
    out[0] = clamped(floor(value), p.width)
    value = (x1 - p.crop_x0) * p.scale
    if not checked(p, value):
        return -1
    out[2] = clamped(ceil(value), p.width)
    value = (p.crop_y1 - y1) * p.scale
    if not checked(p, value):
        return -1
    out[1] = clamped(floor(value), p.height)
    value = (p.crop_y1 - y0) * p.scale
    if not checked(p, value):
        return -1
    out[3] = clamped(ceil(value), p.height)
    if out[2] <= out[0] or out[3] <= out[1]:
        return 0
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def clip_box(p: cython.pointer[Paint], box: cython.p_double) -> cython.int:
    if p.clip_mode == CLIP_NONE:
        return 1
    box[0] = py_max(box[0], p.clip[0])
    box[1] = py_max(box[1], p.clip[1])
    box[2] = py_min(box[2], p.clip[2])
    box[3] = py_min(box[3], p.clip[3])
    if box[2] <= box[0] or box[3] <= box[1]:
        return 0
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def clipped_pixel_box(
    p: cython.pointer[Paint], box: cython.p_double, pixel_box: cython.p_Py_ssize_t
) -> cython.int:
    if p.clip_mode != CLIP_NONE and p.clip_empty:
        return 0
    if not clip_box(p, box):
        return 0
    return page_box_to_pixels(p, box[0], box[1], box[2], box[3], pixel_box)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def row_spans(
    p: cython.pointer[Paint],
    py: cython.Py_ssize_t,
    spans: cython.pp_const_longlong,
) -> cython.Py_ssize_t:
    if py < 0 or py >= p.height:
        return 0
    row: cython.Py_ssize_t = py - p.rows_origin
    if row < 0 or row >= p.rows_count:
        return 0
    spans[0] = p.row_spans + 2 * p.row_offsets[row]
    return cython.cast(cython.Py_ssize_t, p.row_offsets[row + 1] - p.row_offsets[row])


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def pixel_in_row(
    spans: cython.p_const_longlong,
    count: cython.Py_ssize_t,
    px: cython.Py_ssize_t,
) -> cython.bint:
    low: cython.Py_ssize_t = 0
    high: cython.Py_ssize_t = count
    middle: cython.Py_ssize_t
    while low < high:
        middle = (low + high) // 2
        if spans[2 * middle] < px + 1 or (
            spans[2 * middle] == px + 1 and spans[2 * middle + 1] < -1
        ):
            low = middle + 1
        else:
            high = middle
    return low > 0 and spans[2 * (low - 1)] <= px < spans[2 * (low - 1) + 1]


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def blend_pixel(
    p: cython.pointer[Paint], py: cython.Py_ssize_t, px: cython.Py_ssize_t
) -> cython.void:
    extend(p, py, py + 1, px, px + 1)
    if p.alpha <= 0:
        return
    blend_normal_pixel(p.pixels + py * p.row_stride + px * 4, p.red, p.green, p.blue, p.alpha)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def solid_blend(
    p: cython.pointer[Paint],
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
) -> cython.void:
    sa: cython.int = p.alpha
    if sa <= 0 or ix1 <= ix0 or iy1 <= iy0:
        return
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    c: cython.Py_ssize_t
    pixel: cython.p_uchar
    opaque = cython.declare(cython.uchar[4])
    if sa >= 255:
        opaque[0] = cython.cast(cython.uchar, p.red)
        opaque[1] = cython.cast(cython.uchar, p.green)
        opaque[2] = cython.cast(cython.uchar, p.blue)
        opaque[3] = cython.cast(cython.uchar, sa)
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                memcpy(p.pixels + y * p.row_stride + x * 4, opaque, 4)
        return
    any_alpha: cython.bint = False
    all_opaque: cython.bint = True
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            pixel = p.pixels + y * p.row_stride + x * 4
            if pixel[3]:
                any_alpha = True
            if pixel[3] != 255:
                all_opaque = False
    source = cython.declare(cython.int[3])
    source[0] = p.red
    source[1] = p.green
    source[2] = p.blue
    if not any_alpha:
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                pixel = p.pixels + y * p.row_stride + x * 4
                pixel[0] = cython.cast(cython.uchar, p.red)
                pixel[1] = cython.cast(cython.uchar, p.green)
                pixel[2] = cython.cast(cython.uchar, p.blue)
                pixel[3] = cython.cast(cython.uchar, sa)
        return
    source_alpha: cython.double = sa / 255.0
    inverse_source_alpha: cython.double = 1.0 - source_alpha
    scaled = cython.declare(cython.float[3])
    for c in range(3):
        scaled[c] = cython.cast(cython.float, source[c] * source_alpha)
    inverse: cython.float = cython.cast(cython.float, inverse_source_alpha)
    source_float: cython.float = cython.cast(cython.float, source_alpha)
    SCALE: cython.float = 255.0
    destination_alpha: cython.float
    output_alpha: cython.float
    value: cython.float
    if all_opaque:
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                pixel = p.pixels + y * p.row_stride + x * 4
                for c in range(3):
                    value = rintf(scaled[c] + cython.cast(cython.float, pixel[c]) * inverse)
                    pixel[c] = float_to_byte(value)
        return
    for y in range(iy0, iy1):
        for x in range(ix0, ix1):
            pixel = p.pixels + y * p.row_stride + x * 4
            destination_alpha = cython.cast(cython.float, pixel[3]) / SCALE
            output_alpha = source_float + destination_alpha * inverse
            for c in range(3):
                value = (
                    scaled[c] + cython.cast(cython.float, pixel[c]) * destination_alpha * inverse
                ) / output_alpha
                pixel[c] = float_to_byte(rintf(value))
            pixel[3] = float_to_byte(rintf(rintf(output_alpha * SCALE)))


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_rect(
    p: cython.pointer[Paint],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.int:
    box = cython.declare(cython.double[4])
    box[0] = x0
    box[1] = y0
    box[2] = x1
    box[3] = y1
    pixel_box = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = clipped_pixel_box(p, box, pixel_box)
    if found <= 0:
        return found
    ix0: cython.Py_ssize_t = pixel_box[0]
    iy0: cython.Py_ssize_t = pixel_box[1]
    ix1: cython.Py_ssize_t = pixel_box[2]
    iy1: cython.Py_ssize_t = pixel_box[3]
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    k: cython.Py_ssize_t
    count: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    spans: cython.p_const_longlong = cython.NULL
    if p.clip_mode == CLIP_ROWS:
        for y in range(iy0, iy1):
            count = row_spans(p, y, cython.address(spans))
            for k in range(count):
                start = cython.cast(cython.Py_ssize_t, spans[2 * k])
                end = cython.cast(cython.Py_ssize_t, spans[2 * k + 1])
                if start < ix0:
                    start = ix0
                if end > ix1:
                    end = ix1
                if end <= start:
                    continue
                if end - start >= 32:
                    solid_blend(p, start, y, end, y + 1)
                    extend(p, y, y + 1, start, end)
                else:
                    for x in range(start, end):
                        blend_pixel(p, y, x)
        return 0
    left: cython.double = (box[0] - p.crop_x0) * p.scale
    right: cython.double = (box[2] - p.crop_x0) * p.scale
    top: cython.double = (p.crop_y1 - box[3]) * p.scale
    bottom: cython.double = (p.crop_y1 - box[1]) * p.scale
    if not (
        left <= ix0 + 1e-9 and right >= ix1 - 1e-9 and top <= iy0 + 1e-9 and bottom >= iy1 - 1e-9
    ):
        if (
            fill_rect_pixels(
                p.pixels + iy0 * p.row_stride + ix0 * 4,
                p.row_stride,
                ix0,
                ix1,
                iy0,
                iy1,
                left,
                right,
                top,
                bottom,
                p.red,
                p.green,
                p.blue,
                p.alpha,
            )
            < 0
        ):
            p.error = FAILED_MEMORY
            return -1
    else:
        solid_blend(p, ix0, iy0, ix1, iy1)
    extend(p, iy0, iy1, ix0, ix1)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_circle(
    p: cython.pointer[Paint], cx: cython.double, cy: cython.double, radius: cython.double
) -> cython.int:
    box = cython.declare(cython.double[4])
    box[0] = cx - radius
    box[1] = cy - radius
    box[2] = cx + radius
    box[3] = cy + radius
    if not clip_box(p, box):
        return 0
    pixel_box = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = page_box_to_pixels(p, box[0], box[1], box[2], box[3], pixel_box)
    if found <= 0:
        return found
    ix0: cython.Py_ssize_t = pixel_box[0]
    iy0: cython.Py_ssize_t = pixel_box[1]
    ix1: cython.Py_ssize_t = pixel_box[2]
    iy1: cython.Py_ssize_t = pixel_box[3]
    radius2: cython.double = radius * radius
    px: cython.Py_ssize_t
    py: cython.Py_ssize_t
    k: cython.Py_ssize_t
    count: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    spans: cython.p_const_longlong = cython.NULL
    page_x: cython.double
    page_y: cython.double
    dx: cython.double
    dy: cython.double
    opaque = cython.declare(cython.uchar[4])
    opaque[0] = cython.cast(cython.uchar, p.red)
    opaque[1] = cython.cast(cython.uchar, p.green)
    opaque[2] = cython.cast(cython.uchar, p.blue)
    opaque[3] = 255
    if p.clip_mode == CLIP_ROWS:
        for py in range(iy0, iy1):
            page_y = p.crop_y1 - (cython.cast(cython.double, py) + 0.5) / p.scale
            count = row_spans(p, py, cython.address(spans))
            for k in range(count):
                start = cython.cast(cython.Py_ssize_t, spans[2 * k])
                end = cython.cast(cython.Py_ssize_t, spans[2 * k + 1])
                if start < ix0:
                    start = ix0
                if end > ix1:
                    end = ix1
                for px in range(start, end):
                    page_x = p.crop_x0 + (cython.cast(cython.double, px) + 0.5) / p.scale
                    dx = page_x - cx
                    dy = page_y - cy
                    if dx * dx + dy * dy > radius2:
                        continue
                    blend_pixel(p, py, px)
        return 0
    if p.alpha >= 255 and (ix1 - ix0) * (iy1 - iy0) > 16:
        for py in range(iy0, iy1):
            page_y = p.crop_y1 - (cython.cast(cython.double, py) + 0.5) / p.scale
            dy = page_y - cy
            for px in range(ix0, ix1):
                page_x = p.crop_x0 + (cython.cast(cython.double, px) + 0.5) / p.scale
                dx = page_x - cx
                if dx * dx + dy * dy <= radius2:
                    memcpy(p.pixels + py * p.row_stride + px * 4, opaque, 4)
        extend(p, iy0, iy1, ix0, ix1)
        return 0
    for py in range(iy0, iy1):
        page_y = p.crop_y1 - (cython.cast(cython.double, py) + 0.5) / p.scale
        for px in range(ix0, ix1):
            page_x = p.crop_x0 + (cython.cast(cython.double, px) + 0.5) / p.scale
            dx = page_x - cx
            dy = page_y - cy
            if dx * dx + dy * dy > radius2:
                continue
            blend_pixel(p, py, px)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def line_raster(
    p: cython.pointer[Paint],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    line_width: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
) -> cython.int:
    x_delta: cython.double = x1 - x0
    y_delta: cython.double = y1 - y0
    segment_length_squared: cython.double = x_delta * x_delta + y_delta * y_delta
    if segment_length_squared <= 1e-12:
        return 0
    segment_length: cython.double = pow(segment_length_squared, p.exponent)
    half: cython.double = py_max(0.5 / p.scale, line_width * 0.5)
    cap_extension: cython.double = 0.0
    width: cython.Py_ssize_t = ix1 - ix0
    height: cython.Py_ssize_t = iy1 - iy0
    x_offset: cython.p_double = cython.cast(
        cython.p_double, malloc((width + height) * cython.sizeof(cython.double))
    )
    if x_offset == cython.NULL:
        p.error = FAILED_MEMORY
        return -1
    y_offset: cython.p_double = x_offset + width
    covered: cython.p_uchar = cython.cast(cython.p_uchar, malloc(width * height))
    if covered == cython.NULL:
        free(x_offset)
        p.error = FAILED_MEMORY
        return -1
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    sx: cython.Py_ssize_t
    sy: cython.Py_ssize_t
    c: cython.Py_ssize_t
    for j in range(width):
        x_offset[j] = (p.crop_x0 + (cython.cast(cython.double, ix0 + j) + 0.125) / p.scale) - x0
    for i in range(height):
        y_offset[i] = (p.crop_y1 - (cython.cast(cython.double, iy0 + i) + 0.125) / p.scale) - y0
    sample_step: cython.double = 1.0 / (4.0 * p.scale)
    cross_limit: cython.double = half * segment_length
    projection_extension: cython.double = cap_extension * segment_length
    low_projection = cython.declare(cython.double[16])
    high_projection = cython.declare(cython.double[16])
    low_cross = cython.declare(cython.double[16])
    high_cross = cython.declare(cython.double[16])
    projection_shift: cython.double
    cross_shift: cython.double
    for sy in range(4):
        for sx in range(4):
            projection_shift = sample_step * (
                cython.cast(cython.double, sx) * x_delta - cython.cast(cython.double, sy) * y_delta
            )
            low_projection[4 * sy + sx] = -projection_extension - projection_shift
            high_projection[4 * sy + sx] = (
                segment_length_squared + projection_extension - projection_shift
            )
            cross_shift = sample_step * (
                cython.cast(cython.double, sx) * y_delta + cython.cast(cython.double, sy) * x_delta
            )
            low_cross[4 * sy + sx] = -cross_limit - cross_shift
            high_cross[4 * sy + sx] = cross_limit - cross_shift
    projection_base: cython.double
    cross_base: cython.double
    count: cython.int
    any_covered: cython.bint = False
    for i in range(height):
        for j in range(width):
            projection_base = y_offset[i] * y_delta + x_offset[j] * x_delta
            cross_base = -y_offset[i] * x_delta + x_offset[j] * y_delta
            count = 0
            for c in range(16):
                if (
                    projection_base >= low_projection[c]
                    and projection_base <= high_projection[c]
                    and cross_base >= low_cross[c]
                    and cross_base <= high_cross[c]
                ):
                    count += 1
            covered[i * width + j] = cython.cast(cython.uchar, count)
            if count:
                any_covered = True
    alpha: cython.int
    pixel: cython.p_uchar
    SCALE: cython.float = 255.0
    ONE: cython.float = 1.0
    source_fraction: cython.float
    destination_alpha: cython.float
    remaining: cython.float
    output_alpha: cython.float
    safe: cython.float
    colour = cython.declare(cython.float[3])
    colour[0] = cython.cast(cython.float, p.red)
    colour[1] = cython.cast(cython.float, p.green)
    colour[2] = cython.cast(cython.float, p.blue)
    if any_covered:
        for i in range(height):
            for j in range(width):
                alpha = coverage_alpha(p.alpha, covered[i * width + j])
                if alpha <= 0:
                    continue
                pixel = p.pixels + (iy0 + i) * p.row_stride + (ix0 + j) * 4
                if alpha >= 255:
                    pixel[0] = cython.cast(cython.uchar, p.red)
                    pixel[1] = cython.cast(cython.uchar, p.green)
                    pixel[2] = cython.cast(cython.uchar, p.blue)
                    pixel[3] = 255
                    continue
                source_fraction = cython.cast(cython.float, alpha) / SCALE
                destination_alpha = cython.cast(cython.float, pixel[3]) / SCALE
                remaining = ONE - source_fraction
                output_alpha = source_fraction + destination_alpha * remaining
                safe = output_alpha if output_alpha > 0.0 else ONE
                for c in range(3):
                    pixel[c] = float_to_byte(
                        rintf(
                            (
                                colour[c] * source_fraction
                                + cython.cast(cython.float, pixel[c])
                                * destination_alpha
                                * remaining
                            )
                            / safe
                        )
                    )
                pixel[3] = float_to_byte(rintf(output_alpha * SCALE))
    free(covered)
    free(x_offset)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_line(
    p: cython.pointer[Paint],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    line_width: cython.double,
) -> cython.int:
    dx: cython.double = x1 - x0
    dy: cython.double = y1 - y0
    half: cython.double
    if fabs(dx) <= 1e-12 or fabs(dy) <= 1e-12:
        half = py_max(0.5 / p.scale, line_width * 0.5)
        if fabs(dy) <= 1e-12:
            return fill_rect(p, py_min(x0, x1) - 0.0, y0 - half, py_max(x0, x1) + 0.0, y0 + half)
        return fill_rect(p, x0 - half, py_min(y0, y1) - 0.0, x0 + half, py_max(y0, y1) + 0.0)
    seg_len2: cython.double = dx * dx + dy * dy
    half = py_max(0.5 / p.scale, line_width * 0.5)
    if seg_len2 <= 1e-12:
        return fill_rect(p, x0 - half, y0 - half, x0 + half, y0 + half)
    seg_len: cython.double = pow(seg_len2, p.exponent)
    ux: cython.double = dx / seg_len
    uy: cython.double = dy / seg_len
    cap_extension: cython.double = 0.0
    box = cython.declare(cython.double[4])
    box[0] = py_min(x0, x1) - half - fabs(ux) * cap_extension
    box[1] = py_min(y0, y1) - half - fabs(uy) * cap_extension
    box[2] = py_max(x0, x1) + half + fabs(ux) * cap_extension
    box[3] = py_max(y0, y1) + half + fabs(uy) * cap_extension
    pixel_box = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = clipped_pixel_box(p, box, pixel_box)
    if found <= 0:
        return found
    ix0: cython.Py_ssize_t = pixel_box[0]
    iy0: cython.Py_ssize_t = pixel_box[1]
    ix1: cython.Py_ssize_t = pixel_box[2]
    iy1: cython.Py_ssize_t = pixel_box[3]
    half2: cython.double = half * half
    projection_extension: cython.double = cap_extension * seg_len
    if p.clip_mode != CLIP_ROWS and (ix1 - ix0) * (iy1 - iy0) > 64:
        if line_raster(p, x0, y0, x1, y1, line_width, ix0, iy0, ix1, iy1) < 0:
            return -1
        extend(p, iy0, iy1, ix0, ix1)
        return 0
    allowed: cython.p_uchar = cython.NULL
    box_width: cython.Py_ssize_t = ix1 - ix0
    py: cython.Py_ssize_t
    px: cython.Py_ssize_t
    count: cython.Py_ssize_t
    spans: cython.p_const_longlong = cython.NULL
    if p.clip_mode == CLIP_ROWS:
        allowed = cython.cast(cython.p_uchar, malloc(box_width * (iy1 - iy0)))
        if allowed == cython.NULL:
            p.error = FAILED_MEMORY
            return -1
        for py in range(iy0, iy1):
            count = row_spans(p, py, cython.address(spans))
            for px in range(ix0, ix1):
                allowed[(py - iy0) * box_width + (px - ix0)] = pixel_in_row(spans, count, px)
    covered_box = cython.declare(cython.Py_ssize_t[4])
    if segment_samples(
        p.pixels + iy0 * p.row_stride + ix0 * 4,
        p.row_stride,
        ix0,
        iy0,
        ix1,
        iy1,
        p.crop_x0,
        p.crop_y1,
        p.scale,
        x0,
        y0,
        dx,
        dy,
        seg_len2,
        half2,
        projection_extension,
        p.red,
        p.green,
        p.blue,
        p.alpha,
        allowed,
        cython.NULL,
        covered_box,
    ):
        extend(p, covered_box[1], covered_box[3], covered_box[0], covered_box[2])
    free(allowed)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_terminal(
    p: cython.pointer[Paint],
    x: cython.double,
    y: cython.double,
    line_width: cython.double,
    round_shape: cython.bint,
) -> cython.int:
    radius: cython.double = py_max(0.5 / p.scale, line_width * 0.5)
    if round_shape:
        return fill_circle(p, x, y, radius)
    return fill_rect(p, x - radius, y - radius, x + radius, y + radius)


Walk = cython.struct(
    xs=cython.p_const_double,
    ys=cython.p_const_double,
    bounds=cython.p_const_longlong,
    ends_differ=cython.p_const_uchar,
    coincident=cython.p_const_uchar,
    count=cython.Py_ssize_t,
    line_width=cython.double,
    native=cython.bint,
    cap_nonzero=cython.bint,
    cap_butt=cython.bint,
    cap_round=cython.bint,
    join_round=cython.bint,
)


@cython.cclass
class Callbacks:
    line: object
    join: object
    cap: object
    dot: object
    extend: object


@cython.cfunc
@cython.exceptval(-1, check=False)
def flush(p: cython.pointer[Paint], calls: typing.Optional[Callbacks]) -> cython.int:
    if p.painted:
        p.painted = False
        calls.extend(p.window[0], p.window[1], p.window[2], p.window[3])
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def failed(p: cython.pointer[Paint], calls: typing.Optional[Callbacks]) -> cython.int:
    flush(p, calls)
    if p.error in (FAILED_NAN, FAILED_INFINITY):
        return raise_not_integral(p.error == FAILED_NAN)
    raise MemoryError


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def line(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.int:
    if w.native:
        if fill_line(p, x0, y0, x1, y1, w.line_width) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    with cython.gil:
        calls.line(x0, y0, x1, y1)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def join(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    x: cython.double,
    y: cython.double,
) -> cython.int:
    if w.native:
        if fill_terminal(p, x, y, w.line_width, w.join_round) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    with cython.gil:
        calls.join(x, y)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def cap(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    x: cython.double,
    y: cython.double,
) -> cython.int:
    if w.native:
        if w.cap_butt:
            return 0
        if fill_terminal(p, x, y, w.line_width, w.cap_round) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    with cython.gil:
        calls.cap(x, y)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def walk(
    p: cython.pointer[Paint], w: cython.pointer[Walk], calls: typing.Optional[Callbacks]
) -> cython.int:
    k: cython.Py_ssize_t
    start: cython.Py_ssize_t
    n: cython.Py_ssize_t
    index: cython.Py_ssize_t
    closed: cython.bint
    same: cython.bint
    x0: cython.double
    y0: cython.double
    x1: cython.double
    y1: cython.double
    for k in range(w.count):
        start = w.bounds[3 * k]
        n = w.bounds[3 * k + 1] - start
        closed = w.bounds[3 * k + 2]
        if n < 2:
            continue
        x0 = w.xs[start]
        y0 = w.ys[start]
        x1 = w.xs[start + n - 1]
        y1 = w.ys[start + n - 1]
        if n == 2 and not closed:
            if w.coincident != cython.NULL:
                same = w.coincident[k] != 0
            else:
                same = x0 == x1 and y0 == y1
            if same:
                if w.cap_round:
                    with cython.gil:
                        flush(p, calls)
                        calls.dot(x0, y0)
                continue
            line(p, w, calls, x0, y0, x1, y1)
            if w.cap_nonzero:
                cap(p, w, calls, x0, y0)
                cap(p, w, calls, x1, y1)
            continue
        for index in range(start, start + n - 1):
            line(p, w, calls, w.xs[index], w.ys[index], w.xs[index + 1], w.ys[index + 1])
        if w.ends_differ != cython.NULL:
            same = not w.ends_differ[k]
        else:
            same = not (x0 != x1 or y0 != y1)
        if closed and not same:
            line(p, w, calls, x1, y1, x0, y0)
        for index in range(start + 1, start + n - 1):
            join(p, w, calls, w.xs[index], w.ys[index])
        if closed:
            join(p, w, calls, x0, y0)
        elif w.cap_nonzero:
            cap(p, w, calls, x0, y0)
            cap(p, w, calls, x1, y1)
    return 0


def stroke_polylines(
    pixels: cython.uchar[:, :, ::1],
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: typing.Optional[list],
    outline: cython.bint,
    ends_differ: typing.Optional[cython.const[cython.uchar][::1]],
    coincident: typing.Optional[cython.const[cython.uchar][::1]],
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    clip_mode: cython.int,
    clip_box,
    clip_empty: cython.bint,
    rows_origin: cython.Py_ssize_t,
    row_offsets: typing.Optional[cython.const[cython.longlong][::1]],
    row_spans: typing.Optional[cython.const[cython.longlong][::1]],
    line_width: cython.double,
    red: cython.int,
    green: cython.int,
    blue: cython.int,
    alpha: cython.int,
    native: cython.bint,
    cap_nonzero: cython.bint,
    cap_butt: cython.bint,
    cap_round: cython.bint,
    join_round: cython.bint,
    exponent: cython.double,
    on_line,
    on_join,
    on_cap,
    on_dot,
    on_extend,
):
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    count: cython.Py_ssize_t = len(spans)
    if ends_differ is not None and ends_differ.shape[0] != count:
        raise ValueError("ends_differ must hold one byte per span")
    if coincident is not None and coincident.shape[0] != count:
        raise ValueError("coincident must hold one byte per span")
    if clip_mode == CLIP_ROWS:
        if row_offsets is None or row_spans is None or row_offsets.shape[0] < 1:
            raise ValueError("a clip path needs its rows")
        if row_offsets[row_offsets.shape[0] - 1] * 2 > row_spans.shape[0]:
            raise ValueError("row offsets run past the spans")
    bounds: cython.pointer[cython.longlong] = cython.cast(
        cython.pointer[cython.longlong],
        malloc((3 * count + 1) * cython.sizeof(cython.longlong)),
    )
    if bounds == cython.NULL:
        raise MemoryError
    k: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    p = cython.declare(Paint)
    w = cython.declare(Walk)
    calls: Callbacks = Callbacks.__new__(Callbacks)
    calls.line = on_line
    calls.join = on_join
    calls.cap = on_cap
    calls.dot = on_dot
    calls.extend = on_extend
    try:
        for k in range(count):
            start, end, flag = spans[k]
            if start < 0 or end > xs.shape[0] or end < start:
                raise ValueError("span runs past the points")
            bounds[3 * k] = start
            bounds[3 * k + 1] = end
            bounds[3 * k + 2] = 1 if (outline or flag) else 0
        p.pixels = (
            cython.address(pixels[0, 0, 0]) if pixels.shape[0] and pixels.shape[1] else cython.NULL
        )
        p.row_stride = pixels.strides[0]
        p.width = pixels.shape[1]
        p.height = pixels.shape[0]
        p.crop_x0 = crop_x0
        p.crop_y1 = crop_y1
        p.scale = scale
        p.clip_mode = clip_mode
        p.clip_empty = clip_empty
        if clip_mode != CLIP_NONE:
            p.clip[0] = clip_box[0]
            p.clip[1] = clip_box[1]
            p.clip[2] = clip_box[2]
            p.clip[3] = clip_box[3]
        p.rows_origin = rows_origin
        if clip_mode == CLIP_ROWS:
            p.rows_count = row_offsets.shape[0] - 1
            p.row_offsets = cython.address(row_offsets[0])
            p.row_spans = cython.address(row_spans[0]) if row_spans.shape[0] else cython.NULL
        else:
            p.rows_count = 0
            p.row_offsets = cython.NULL
            p.row_spans = cython.NULL
        p.red = red
        p.green = green
        p.blue = blue
        p.alpha = alpha
        p.exponent = exponent
        p.painted = False
        p.error = 0
        w.xs = cython.address(xs[0]) if xs.shape[0] else cython.NULL
        w.ys = cython.address(ys[0]) if ys.shape[0] else cython.NULL
        w.bounds = bounds
        w.ends_differ = (
            cython.address(ends_differ[0]) if ends_differ is not None and count else cython.NULL
        )
        w.coincident = (
            cython.address(coincident[0]) if coincident is not None and count else cython.NULL
        )
        w.count = count
        w.line_width = line_width
        w.native = native
        w.cap_nonzero = cap_nonzero
        w.cap_butt = cap_butt
        w.cap_round = cap_round
        w.join_round = join_round
        with cython.nogil:
            walk(cython.address(p), cython.address(w), calls)
        flush(cython.address(p), calls)
    finally:
        free(bounds)
