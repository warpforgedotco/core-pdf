# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._alpha_blend import blend_one, opaque_channel
from cython.cimports.core_pdf_cythonized._byte_clamp import float_to_byte
from cython.cimports.core_pdf_cythonized._pixel_blend import blend_normal_pixel, coverage_alpha
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min, raise_not_integral
from cython.cimports.core_pdf_cythonized._rect import fill_rect_pixels
from cython.cimports.core_pdf_cythonized._stroke import (
    CAP_BUTT,
    CAP_ROUND,
    CAP_SQUARE,
    CLIP_NONE,
    CLIP_RECT,
    CLIP_ROWS,
    COVER_DISC,
    COVER_POLYGON,
    FAILED_INFINITY,
    FAILED_MEMORY,
    FAILED_NAN,
    JOIN_BEVEL,
    JOIN_MITER,
    JOIN_ROUND,
)
from cython.cimports.core_pdf_cythonized._stroke_segment import segment_samples
from cython.cimports.cpython.bytes import PyBytes_FromStringAndSize
from cython.cimports.libc.math import (
    ceil,
    cos,
    fabs,
    floor,
    isinf,
    isnan,
    pow,
    rint,
    rintf,
    sin,
    sqrt,
)
from cython.cimports.libc.stdlib import calloc, free, malloc, qsort, realloc
from cython.cimports.libc.string import memcpy, memset

__all__ = ("axis_aligned_polylines", "short_subpaths", "stroke_coverage", "stroke_polylines")


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
def pen_half(scale: cython.double, line_width: cython.double) -> cython.double:
    """Half the width of the pen that strokes a line width, where a unit is scale device
    pixels long, as render.paths.stroke_half_width works it out: its own, but no thinner
    than one device pixel, the width of a line width of 0 (ISO 32000-2 8.4.3.2), as Acrobat
    floors thin strokes, so that none drops out (10.7.4)."""
    return py_max(0.5 / scale, line_width * 0.5)


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
    p: cython.pointer[Paint], py: cython.Py_ssize_t, spans: cython.pp_const_longlong
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
    spans: cython.p_const_longlong, count: cython.Py_ssize_t, px: cython.Py_ssize_t
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
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def on_pixel_edge(value: cython.double) -> cython.bint:
    return fabs(value - rint(value)) <= 1e-9


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def column_samples(start: cython.double, end: cython.double, px: cython.Py_ssize_t) -> cython.uint:
    """The sample columns of pixel column px inside the device x span [start, end), as bits
    0 to 3: supersampled_coverage_plane's rule, so that a rectangle sampled here covers
    the samples a path filled there covers, and pieces that meet share none."""
    bits: cython.uint = 0
    sx: cython.int
    offset: cython.double
    for sx in range(4):
        offset = (sx + 0.5) / 4
        if ceil(start - offset) <= px and px < ceil(end - offset):
            bits |= cython.cast(cython.uint, 1) << sx
    return bits


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def row_samples(
    crop_y1: cython.double,
    scale: cython.double,
    low: cython.double,
    high: cython.double,
    py: cython.Py_ssize_t,
) -> cython.uint:
    """The sample rows of pixel row py whose page y is in [low, high), as bits 0 to 3:
    supersampled_coverage_plane's rule."""
    bits: cython.uint = 0
    sy: cython.int
    page_y: cython.double
    for sy in range(4):
        page_y = crop_y1 - (cython.cast(cython.double, py) + (sy + 0.5) / 4) / scale
        if low <= page_y and page_y < high:
            bits |= cython.cast(cython.uint, 1) << sy
    return bits


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def rect_samples(columns: cython.uint, rows: cython.uint) -> cython.uint:
    """The 4 x 4 sample bits, row sy's at 4 sy, of the samples in both columns and rows."""
    bits: cython.uint = 0
    sy: cython.int
    for sy in range(4):
        if rows & (cython.cast(cython.uint, 1) << sy):
            bits |= columns << (4 * sy)
    return bits


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_rect_samples(
    p: cython.pointer[Paint],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
) -> cython.int:
    """Paint the rectangle in the pixel box by its 4 x 4 samples under the clip path's
    rows, as fill_path paints the same rectangle there."""
    start: cython.double = (x0 - p.crop_x0) * p.scale
    end: cython.double = (x1 - p.crop_x0) * p.scale
    py: cython.Py_ssize_t
    px: cython.Py_ssize_t
    rows: cython.int
    count: cython.int
    alpha: cython.int
    span_count: cython.Py_ssize_t
    spans: cython.p_const_longlong = cython.NULL
    painted: cython.bint = False
    window = cython.declare(cython.Py_ssize_t[4])
    for py in range(iy0, iy1):
        rows = sample_count(row_samples(p.crop_y1, p.scale, y0, y1, py))
        if not rows:
            continue
        span_count = row_spans(p, py, cython.address(spans))
        for px in range(ix0, ix1):
            if not pixel_in_row(spans, span_count, px):
                continue
            count = rows * sample_count(column_samples(start, end, px))
            if not count:
                continue
            alpha = coverage_alpha(p.alpha, count)
            if alpha <= 0:
                continue
            blend_normal_pixel(p.pixels + py * p.row_stride + px * 4, p.red, p.green, p.blue, alpha)
            if not painted:
                painted = True
                window[0] = px
                window[1] = py
                window[2] = px + 1
                window[3] = py + 1
            else:
                window[0] = px if px < window[0] else window[0]
                window[2] = px + 1 if px + 1 > window[2] else window[2]
                window[3] = py + 1
    if painted:
        extend(p, window[1], window[3], window[0], window[2])
    return 0


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
        if not (
            on_pixel_edge((x0 - p.crop_x0) * p.scale)
            and on_pixel_edge((x1 - p.crop_x0) * p.scale)
            and on_pixel_edge((p.crop_y1 - y1) * p.scale)
            and on_pixel_edge((p.crop_y1 - y0) * p.scale)
        ):
            # An edge inside pixels: the 4 x 4 samples cover them, as they cover a filled
            # path's under a clip path.
            return fill_rect_samples(p, x0, y0, x1, y1, ix0, iy0, ix1, iy1)
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
    half: cython.double = pen_half(p.scale, line_width)
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
        half = pen_half(p.scale, line_width)
        if fabs(dy) <= 1e-12:
            return fill_rect(p, py_min(x0, x1) - 0.0, y0 - half, py_max(x0, x1) + 0.0, y0 + half)
        return fill_rect(p, x0 - half, py_min(y0, y1) - 0.0, x0 + half, py_max(y0, y1) + 0.0)
    seg_len2: cython.double = dx * dx + dy * dy
    half = pen_half(p.scale, line_width)
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
    radius: cython.double = pen_half(p.scale, line_width)
    if round_shape:
        return fill_circle(p, x, y, radius)
    return fill_rect(p, x - radius, y - radius, x + radius, y + radius)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fill_convex(
    p: cython.pointer[Paint], xs: cython.p_double, ys: cython.p_double, n: cython.int
) -> cython.int:
    """Fill a convex polygon of n <= 4 page-space vertices, in either order, counting the
    4 x 4 samples of each pixel inside every edge, as line_raster samples a segment."""
    box = cython.declare(cython.double[4])
    box[0] = xs[0]
    box[1] = ys[0]
    box[2] = xs[0]
    box[3] = ys[0]
    k: cython.int
    area: cython.double = 0.0
    for k in range(n):
        box[0] = py_min(box[0], xs[k])
        box[1] = py_min(box[1], ys[k])
        box[2] = py_max(box[2], xs[k])
        box[3] = py_max(box[3], ys[k])
        area += xs[k] * ys[(k + 1) % n] - xs[(k + 1) % n] * ys[k]
    if area == 0.0:
        return 0
    orientation: cython.double = 1.0 if area > 0.0 else -1.0
    # Edge k holds the samples where a[k] x + b[k] y + c[k] >= 0.
    a = cython.declare(cython.double[4])
    b = cython.declare(cython.double[4])
    c = cython.declare(cython.double[4])
    for k in range(n):
        a[k] = -(ys[(k + 1) % n] - ys[k]) * orientation
        b[k] = (xs[(k + 1) % n] - xs[k]) * orientation
        c[k] = -(a[k] * xs[k] + b[k] * ys[k])
    pixel_box = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = clipped_pixel_box(p, box, pixel_box)
    if found <= 0:
        return found
    ix0: cython.Py_ssize_t = pixel_box[0]
    iy0: cython.Py_ssize_t = pixel_box[1]
    ix1: cython.Py_ssize_t = pixel_box[2]
    iy1: cython.Py_ssize_t = pixel_box[3]
    px: cython.Py_ssize_t
    py: cython.Py_ssize_t
    sx: cython.int
    sy: cython.int
    count: cython.int
    inside: cython.bint
    alpha: cython.int
    sample_x = cython.declare(cython.double[4])
    sample_y: cython.double
    span_count: cython.Py_ssize_t = 0
    spans: cython.p_const_longlong = cython.NULL
    any_covered: cython.bint = False
    covered_box = cython.declare(cython.Py_ssize_t[4])
    for py in range(iy0, iy1):
        if p.clip_mode == CLIP_ROWS:
            span_count = row_spans(p, py, cython.address(spans))
        for px in range(ix0, ix1):
            if p.clip_mode == CLIP_ROWS and not pixel_in_row(spans, span_count, px):
                continue
            for sx in range(4):
                sample_x[sx] = (
                    p.crop_x0 + (cython.cast(cython.double, px) + 0.125 + 0.25 * sx) / p.scale
                )
            count = 0
            for sy in range(4):
                sample_y = (
                    p.crop_y1 - (cython.cast(cython.double, py) + 0.125 + 0.25 * sy) / p.scale
                )
                for sx in range(4):
                    inside = True
                    for k in range(n):
                        if a[k] * sample_x[sx] + b[k] * sample_y + c[k] < 0.0:
                            inside = False
                            break
                    if inside:
                        count += 1
            if not count:
                continue
            alpha = coverage_alpha(p.alpha, count)
            if alpha <= 0:
                continue
            blend_normal_pixel(p.pixels + py * p.row_stride + px * 4, p.red, p.green, p.blue, alpha)
            if not any_covered:
                any_covered = True
                covered_box[0] = px
                covered_box[1] = py
                covered_box[2] = px + 1
                covered_box[3] = py + 1
            else:
                if px < covered_box[0]:
                    covered_box[0] = px
                if px + 1 > covered_box[2]:
                    covered_box[2] = px + 1
                covered_box[3] = py + 1
    if any_covered:
        extend(p, covered_box[1], covered_box[3], covered_box[0], covered_box[2])
    return 0


# A stroke's coverage, built before it is painted: each pixel of the window
# (left, top, width, height) holds, as bits, the 4 x 4 samples any piece of the stroke
# covers, so that pieces which overlap or meet count each sample once. The pieces are
# worked out in stroke space, which forward maps to page space: the identity, or under a
# CTM that scales unevenly or skews, the map under which the pen is a circle.
Cover = cython.struct(
    masks=cython.p_ushort,
    left=cython.Py_ssize_t,
    top=cython.Py_ssize_t,
    width=cython.Py_ssize_t,
    height=cython.Py_ssize_t,
    crop_x0=cython.double,
    crop_y1=cython.double,
    scale=cython.double,
    transformed=cython.bint,
    forward=cython.double[4],
    inverse=cython.double[4],
)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_window(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    out: cython.p_Py_ssize_t,
) -> cython.int:
    """The window's columns out[0]:out[2] and rows out[1]:out[3] that the stroke-space box
    (x0, y0, x1, y1) reaches: 1, or 0 when it reaches none, or -1 on a NaN or infinity."""
    low_x: cython.double = x0
    low_y: cython.double = y0
    high_x: cython.double = x1
    high_y: cython.double = y1
    k: cython.int
    corner_x: cython.double
    corner_y: cython.double
    page_x: cython.double
    page_y: cython.double
    if c.transformed:
        for k in range(4):
            corner_x = x0 if k & 1 == 0 else x1
            corner_y = y0 if k & 2 == 0 else y1
            page_x = c.forward[0] * corner_x + c.forward[2] * corner_y
            page_y = c.forward[1] * corner_x + c.forward[3] * corner_y
            if k == 0:
                low_x = page_x
                high_x = page_x
                low_y = page_y
                high_y = page_y
            else:
                low_x = py_min(low_x, page_x)
                high_x = py_max(high_x, page_x)
                low_y = py_min(low_y, page_y)
                high_y = py_max(high_y, page_y)
    value: cython.double = floor((low_x - c.crop_x0) * c.scale)
    if not checked(p, value):
        return -1
    out[0] = clamped(value - c.left, c.width)
    value = ceil((high_x - c.crop_x0) * c.scale)
    if not checked(p, value):
        return -1
    out[2] = clamped(value - c.left, c.width)
    value = floor((c.crop_y1 - high_y) * c.scale)
    if not checked(p, value):
        return -1
    out[1] = clamped(value - c.top, c.height)
    value = ceil((c.crop_y1 - low_y) * c.scale)
    if not checked(p, value):
        return -1
    out[3] = clamped(value - c.top, c.height)
    if out[2] <= out[0] or out[3] <= out[1]:
        return 0
    return 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_samples(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    kind: cython.int,
    box: cython.p_double,
    a: cython.p_double,
    b: cython.p_double,
    offset: cython.p_double,
    n: cython.int,
) -> cython.int:
    """Set the bit of each sample inside a convex polygon, where
    a[k] x + b[k] y + offset[k] >= 0 for every edge k < n, or inside the disc around
    (a[0], b[0]) whose radius squared is offset[0]; box bounds the shape in stroke space.
    The samples sit where fill_convex and line_raster put them, a quarter of a pixel apart
    from an eighth of one in. A pixel whose corner samples are all inside every edge, or
    all outside one, is settled from them."""
    window = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = cover_window(p, c, box[0], box[1], box[2], box[3], window)
    if found <= 0:
        return found
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    sx: cython.int
    sy: cython.int
    k: cython.int
    bits: cython.uint
    inside: cython.bint
    whole: cython.bint
    none: cython.bint
    x: cython.double
    y: cython.double
    dx: cython.double
    dy: cython.double
    value: cython.double
    low: cython.double
    high: cython.double
    step: cython.double = 0.25 / c.scale
    left_x: cython.double
    top_y: cython.double
    # The edges as functions of the page point, and their change from one sample to the
    # next across and down.
    edge_x = cython.declare(cython.double[4])
    edge_y = cython.declare(cython.double[4])
    across = cython.declare(cython.double[4])
    down = cython.declare(cython.double[4])
    base = cython.declare(cython.double[4])
    page_x = cython.declare(cython.double[4])
    page_y = cython.declare(cython.double[4])
    if kind == COVER_POLYGON:
        for k in range(n):
            if c.transformed:
                edge_x[k] = a[k] * c.inverse[0] + b[k] * c.inverse[1]
                edge_y[k] = a[k] * c.inverse[2] + b[k] * c.inverse[3]
            else:
                edge_x[k] = a[k]
                edge_y[k] = b[k]
            across[k] = edge_x[k] * step
            down[k] = -edge_y[k] * step
    for i in range(window[1], window[3]):
        top_y = c.crop_y1 - (cython.cast(cython.double, c.top + i) + 0.125) / c.scale
        for j in range(window[0], window[2]):
            left_x = c.crop_x0 + (cython.cast(cython.double, c.left + j) + 0.125) / c.scale
            bits = 0
            if kind == COVER_POLYGON:
                whole = True
                none = False
                for k in range(n):
                    base[k] = edge_x[k] * left_x + edge_y[k] * top_y + offset[k]
                    low = base[k]
                    high = base[k]
                    for sy in range(0, 4, 3):
                        for sx in range(0, 4, 3):
                            value = base[k] + sx * across[k] + sy * down[k]
                            low = py_min(low, value)
                            high = py_max(high, value)
                    if high < 0.0:
                        none = True
                        break
                    if low < 0.0:
                        whole = False
                if none:
                    continue
                if whole:
                    bits = 0xFFFF
                else:
                    for sy in range(4):
                        for sx in range(4):
                            inside = True
                            for k in range(n):
                                if base[k] + sx * across[k] + sy * down[k] < 0.0:
                                    inside = False
                                    break
                            if inside:
                                bits |= cython.cast(cython.uint, 1) << (4 * sy + sx)
            elif not c.transformed:
                # The nearest and farthest points of the samples' square from the centre.
                dx = py_max(py_max(left_x - a[0], a[0] - (left_x + 3.0 * step)), 0.0)
                dy = py_max(py_max((top_y - 3.0 * step) - b[0], b[0] - top_y), 0.0)
                if dx * dx + dy * dy > offset[0]:
                    continue
                dx = py_max(fabs(left_x - a[0]), fabs(left_x + 3.0 * step - a[0]))
                dy = py_max(fabs(top_y - b[0]), fabs(top_y - 3.0 * step - b[0]))
                if dx * dx + dy * dy <= offset[0]:
                    bits = 0xFFFF
                else:
                    for sy in range(4):
                        for sx in range(4):
                            dx = left_x + sx * step - a[0]
                            dy = top_y - sy * step - b[0]
                            if dx * dx + dy * dy <= offset[0]:
                                bits |= cython.cast(cython.uint, 1) << (4 * sy + sx)
            else:
                for sx in range(4):
                    page_x[sx] = left_x + sx * step
                for sy in range(4):
                    page_y[sy] = top_y - sy * step
                for sy in range(4):
                    for sx in range(4):
                        x = c.inverse[0] * page_x[sx] + c.inverse[2] * page_y[sy]
                        y = c.inverse[1] * page_x[sx] + c.inverse[3] * page_y[sy]
                        dx = x - a[0]
                        dy = y - b[0]
                        if dx * dx + dy * dy <= offset[0]:
                            bits |= cython.cast(cython.uint, 1) << (4 * sy + sx)
            if bits:
                c.masks[i * c.width + j] |= cython.cast(cython.ushort, bits)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_rect(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.int:
    """Cover the page-space rectangle (x0, y0, x1, y1) by the half-open rule a filled path
    has (column_samples, row_samples), so that the stroke covers the samples the same
    shape filled as a path covers, and pieces that meet on a sample count it once."""
    if x1 <= x0 or y1 <= y0:
        return 0
    window = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = cover_window(p, c, x0, y0, x1, y1, window)
    if found <= 0:
        return found
    start: cython.double = (x0 - c.crop_x0) * c.scale
    end: cython.double = (x1 - c.crop_x0) * c.scale
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    rows: cython.uint
    bits: cython.uint
    for i in range(window[1], window[3]):
        rows = row_samples(c.crop_y1, c.scale, y0, y1, c.top + i)
        if not rows:
            continue
        for j in range(window[0], window[2]):
            bits = rect_samples(column_samples(start, end, c.left + j), rows)
            if bits:
                c.masks[i * c.width + j] |= cython.cast(cython.ushort, bits)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_convex(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    xs: cython.p_double,
    ys: cython.p_double,
    n: cython.int,
) -> cython.int:
    """Cover a convex polygon of n <= 4 stroke-space vertices, in either order: one along
    the axes of the page by cover_rect."""
    if (
        not c.transformed
        and n == 4
        and (
            (xs[0] == xs[1] and ys[1] == ys[2] and xs[2] == xs[3] and ys[3] == ys[0])
            or (ys[0] == ys[1] and xs[1] == xs[2] and ys[2] == ys[3] and xs[3] == xs[0])
        )
    ):
        return cover_rect(
            p,
            c,
            py_min(xs[0], xs[2]),
            py_min(ys[0], ys[2]),
            py_max(xs[0], xs[2]),
            py_max(ys[0], ys[2]),
        )
    box = cython.declare(cython.double[4])
    box[0] = xs[0]
    box[1] = ys[0]
    box[2] = xs[0]
    box[3] = ys[0]
    k: cython.int
    area: cython.double = 0.0
    for k in range(n):
        box[0] = py_min(box[0], xs[k])
        box[1] = py_min(box[1], ys[k])
        box[2] = py_max(box[2], xs[k])
        box[3] = py_max(box[3], ys[k])
        area += xs[k] * ys[(k + 1) % n] - xs[(k + 1) % n] * ys[k]
    if area == 0.0:
        return 0
    orientation: cython.double = 1.0 if area > 0.0 else -1.0
    a = cython.declare(cython.double[4])
    b = cython.declare(cython.double[4])
    offset = cython.declare(cython.double[4])
    for k in range(n):
        a[k] = -(ys[(k + 1) % n] - ys[k]) * orientation
        b[k] = (xs[(k + 1) % n] - xs[k]) * orientation
        offset[k] = -(a[k] * xs[k] + b[k] * ys[k])
    return cover_samples(p, c, COVER_POLYGON, box, a, b, offset, n)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_disc(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    cx: cython.double,
    cy: cython.double,
    radius: cython.double,
) -> cython.int:
    """Cover the disc of the radius around (cx, cy) in stroke space."""
    box = cython.declare(cython.double[4])
    box[0] = cx - radius
    box[1] = cy - radius
    box[2] = cx + radius
    box[3] = cy + radius
    a: cython.double = cx
    b: cython.double = cy
    radius2: cython.double = radius * radius
    return cover_samples(
        p, c, COVER_DISC, box, cython.address(a), cython.address(b), cython.address(radius2), 1
    )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_nearest(
    p: cython.pointer[Paint], c: cython.pointer[Cover], x: cython.double, y: cython.double
) -> cython.int:
    """Cover the sample nearest the stroke-space point (x, y). A dot, a subpath or dash of
    no length, or a subpath shorter than a pixel, covers it as well as the samples inside
    it, so that it never disappears (ISO 32000-2 10.7.4). The pen is a device pixel across
    or more, so a dot already holds that sample, but a butt-capped segment shorter than the
    samples' spacing can fall between them all."""
    page_x: cython.double = x
    page_y: cython.double = y
    if c.transformed:
        page_x = c.forward[0] * x + c.forward[2] * y
        page_y = c.forward[1] * x + c.forward[3] * y
    # The samples sit an eighth of a pixel in, then a quarter of a pixel apart.
    column: cython.double = floor(((page_x - c.crop_x0) * c.scale - 0.125) * 4.0 + 0.5)
    row: cython.double = floor(((c.crop_y1 - page_y) * c.scale - 0.125) * 4.0 + 0.5)
    if not checked(p, column) or not checked(p, row):
        return -1
    pixel_x: cython.double = floor(column * 0.25)
    pixel_y: cython.double = floor(row * 0.25)
    j: cython.double = pixel_x - c.left
    i: cython.double = pixel_y - c.top
    if j < 0.0 or i < 0.0 or j >= c.width or i >= c.height:
        return 0
    sx: cython.int = cython.cast(cython.int, column - pixel_x * 4.0)
    sy: cython.int = cython.cast(cython.int, row - pixel_y * 4.0)
    index: cython.Py_ssize_t = cython.cast(cython.Py_ssize_t, i) * c.width + cython.cast(
        cython.Py_ssize_t, j
    )
    c.masks[index] |= cython.cast(cython.ushort, cython.cast(cython.uint, 1) << (4 * sy + sx))
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_segment(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    half: cython.double,
) -> cython.int:
    """Cover the rectangle reaching half either side of a segment."""
    dx: cython.double = x1 - x0
    dy: cython.double = y1 - y0
    length_squared: cython.double = dx * dx + dy * dy
    if length_squared == 0.0:
        return 0
    length: cython.double = pow(length_squared, p.exponent)
    normal_x: cython.double = -dy / length * half
    normal_y: cython.double = dx / length * half
    xs = cython.declare(cython.double[4])
    ys = cython.declare(cython.double[4])
    xs[0] = x0 + normal_x
    ys[0] = y0 + normal_y
    xs[1] = x1 + normal_x
    ys[1] = y1 + normal_y
    xs[2] = x1 - normal_x
    ys[2] = y1 - normal_y
    xs[3] = x0 - normal_x
    ys[3] = y0 - normal_y
    return cover_convex(p, c, xs, ys, 4)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cover_square(
    p: cython.pointer[Paint],
    c: cython.pointer[Cover],
    x: cython.double,
    y: cython.double,
    ux: cython.double,
    uy: cython.double,
    half: cython.double,
) -> cython.int:
    """Cover the square reaching half each way around (x, y), its sides along (ux, uy)."""
    ahead_x: cython.double = ux * half
    ahead_y: cython.double = uy * half
    xs = cython.declare(cython.double[4])
    ys = cython.declare(cython.double[4])
    xs[0] = x - ahead_x - ahead_y
    ys[0] = y - ahead_y + ahead_x
    xs[1] = x + ahead_x - ahead_y
    ys[1] = y + ahead_y + ahead_x
    xs[2] = x + ahead_x + ahead_y
    ys[2] = y + ahead_y - ahead_x
    xs[3] = x - ahead_x + ahead_y
    ys[3] = y - ahead_y - ahead_x
    return cover_convex(p, c, xs, ys, 4)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def sample_count(bits: cython.uint) -> cython.uchar:
    bits = bits - ((bits >> 1) & 0x5555)
    bits = (bits & 0x3333) + ((bits >> 2) & 0x3333)
    bits = (bits + (bits >> 4)) & 0x0F0F
    return cython.cast(cython.uchar, (bits + (bits >> 8)) & 0x1F)


Walk = cython.struct(
    xs=cython.p_const_double,
    ys=cython.p_const_double,
    bounds=cython.p_const_longlong,
    count=cython.Py_ssize_t,
    line_width=cython.double,
    half=cython.double,
    native=cython.bint,
    line_cap=cython.int,
    line_join=cython.int,
    miter_limit_squared=cython.double,
    cover=cython.pointer[Cover],
    # While collecting, the pieces' edges (x0, y0, x1, y1), each piece counter-clockwise,
    # for paint_union to hand to the exact-area fill in one call.
    collect=cython.bint,
    union_edges=cython.p_double,
    union_count=cython.Py_ssize_t,
    union_capacity=cython.Py_ssize_t,
    union_box=cython.double[4],
    # The collected pieces that are rectangles along the axes, (x0, y0, x1, y1) in page
    # space, and whether any piece is not one.
    union_rects=cython.p_double,
    rect_count=cython.Py_ssize_t,
    rect_capacity=cython.Py_ssize_t,
    union_shapes=cython.bint,
    # Whether collect_box records boxes alone, not their edges: a stroke with no round cap
    # or join, whose pieces fill_rect_union fills.
    boxes_only=cython.bint,
)


@cython.cclass
class Callbacks:
    line: object
    polygon: object
    square: object
    circle: object
    dot: object
    extend: object
    union: object


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
@cython.exceptval(check=False)
def collect_edge(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.int:
    """Add the edge from (x0, y0) to (x1, y1) to the pieces' edges: -1, with p.error set
    and no exception raised, on a NaN or infinite coordinate, as the cover route fails,
    when memory runs out, or once a piece has failed; its caller raises by failed()."""
    grown: cython.p_double
    capacity: cython.Py_ssize_t
    if p.error != 0:
        # A piece already failed: the stroke is not painted.
        return -1
    if not (checked(p, x0) and checked(p, y0) and checked(p, x1) and checked(p, y1)):
        return -1
    if w.union_count == w.union_capacity:
        capacity = 64 if w.union_capacity == 0 else 2 * w.union_capacity
        grown = cython.cast(
            cython.p_double, realloc(w.union_edges, capacity * 4 * cython.sizeof(cython.double))
        )
        if grown == cython.NULL:
            p.error = FAILED_MEMORY
            return -1
        w.union_edges = grown
        w.union_capacity = capacity
    edge: cython.p_double = w.union_edges + 4 * w.union_count
    edge[0] = x0
    edge[1] = y0
    edge[2] = x1
    edge[3] = y1
    if w.union_count == 0:
        w.union_box[0] = py_min(x0, x1)
        w.union_box[1] = py_min(y0, y1)
        w.union_box[2] = py_max(x0, x1)
        w.union_box[3] = py_max(y0, y1)
    else:
        w.union_box[0] = py_min(w.union_box[0], py_min(x0, x1))
        w.union_box[1] = py_min(w.union_box[1], py_min(y0, y1))
        w.union_box[2] = py_max(w.union_box[2], py_max(x0, x1))
        w.union_box[3] = py_max(w.union_box[3], py_max(y0, y1))
    w.union_count += 1
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def collect_box(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.int:
    """Add the rectangle (x0, y0, x1, y1), x0 <= x1 and y0 <= y1, counter-clockwise from
    its lower left corner, as a closed subpath of its corners has it: under a rectangular
    clip, the part of it inside the clip, as fill_rect paints it. -1, with p.error set and
    no exception raised, as collect_edge fails."""
    if p.error != 0:
        return -1
    if not (checked(p, x0) and checked(p, y0) and checked(p, x1) and checked(p, y1)):
        return -1
    if p.clip_mode == CLIP_RECT:
        x0 = py_max(x0, p.clip[0])
        y0 = py_max(y0, p.clip[1])
        x1 = py_min(x1, p.clip[2])
        y1 = py_min(y1, p.clip[3])
        if x1 <= x0 or y1 <= y0:
            return 0
    grown: cython.p_double
    capacity: cython.Py_ssize_t
    if w.rect_count == w.rect_capacity:
        capacity = 64 if w.rect_capacity == 0 else 2 * w.rect_capacity
        grown = cython.cast(
            cython.p_double, realloc(w.union_rects, capacity * 4 * cython.sizeof(cython.double))
        )
        if grown == cython.NULL:
            p.error = FAILED_MEMORY
            return -1
        w.union_rects = grown
        w.rect_capacity = capacity
    rect: cython.p_double = w.union_rects + 4 * w.rect_count
    rect[0] = x0
    rect[1] = y0
    rect[2] = x1
    rect[3] = y1
    w.rect_count += 1
    if w.boxes_only:
        return 0
    if (
        collect_edge(p, w, x0, y0, x1, y0) < 0
        or collect_edge(p, w, x1, y0, x1, y1) < 0
        or collect_edge(p, w, x1, y1, x0, y1) < 0
        or collect_edge(p, w, x0, y1, x0, y0) < 0
    ):
        return -1
    return 0


# The most vertices a piece's polygon has, a disc's, and what cutting it to a rectangle
# can add, one for each side of the rectangle.
MAX_PIECE_VERTICES = cython.declare(cython.int, 40)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def inside_clip_side(
    side: cython.int, bound: cython.double, x: cython.double, y: cython.double
) -> cython.bint:
    """Whether (x, y) is inside the clip rectangle's side: 0 left, 1 bottom, 2 right, 3 top."""
    if side == 0:
        return x >= bound
    if side == 1:
        return y >= bound
    if side == 2:
        return x <= bound
    return y <= bound


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cut_to_clip_side(
    side: cython.int,
    bound: cython.double,
    xs: cython.p_double,
    ys: cython.p_double,
    n: cython.int,
    out_x: cython.p_double,
    out_y: cython.p_double,
) -> cython.int:
    """The polygon of n vertices cut to one side of the clip rectangle (Sutherland and
    Hodgman), into out_x and out_y: its vertex count. A crossing lies on the side's line,
    at the fraction of the edge from its previous vertex that reaches it."""
    count: cython.int = 0
    k: cython.int
    previous_x: cython.double = xs[n - 1]
    previous_y: cython.double = ys[n - 1]
    previous_inside: cython.bint = inside_clip_side(side, bound, previous_x, previous_y)
    x: cython.double
    y: cython.double
    inside: cython.bint
    t: cython.double
    for k in range(n):
        x = xs[k]
        y = ys[k]
        inside = inside_clip_side(side, bound, x, y)
        if inside != previous_inside:
            if side == 0 or side == 2:
                t = (bound - previous_x) / (x - previous_x)
                out_x[count] = bound
                out_y[count] = previous_y + t * (y - previous_y)
            else:
                t = (bound - previous_y) / (y - previous_y)
                out_x[count] = previous_x + t * (x - previous_x)
                out_y[count] = bound
            count += 1
        if inside:
            out_x[count] = x
            out_y[count] = y
            count += 1
        previous_x = x
        previous_y = y
        previous_inside = inside
    return count


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def collect_polygon(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    xs: cython.p_double,
    ys: cython.p_double,
    n: cython.int,
) -> cython.int:
    """Add the polygon of n <= 32 vertices, counter-clockwise: under a rectangular clip,
    the part of it inside the clip, cut to its left, bottom, right and top sides in turn,
    as collect_box cuts a rectangle."""
    w.union_shapes = True
    first_x = cython.declare(cython.double[40])
    first_y = cython.declare(cython.double[40])
    second_x = cython.declare(cython.double[40])
    second_y = cython.declare(cython.double[40])
    side: cython.int
    vx: cython.p_double = xs
    vy: cython.p_double = ys
    if p.clip_mode == CLIP_RECT:
        for side in range(4):
            if n < 3:
                return 0
            if side % 2 == 0:
                n = cut_to_clip_side(side, p.clip[side], vx, vy, n, first_x, first_y)
                vx = first_x
                vy = first_y
            else:
                n = cut_to_clip_side(side, p.clip[side], vx, vy, n, second_x, second_y)
                vx = second_x
                vy = second_y
    if n < 3:
        return 0
    area: cython.double = 0.0
    k: cython.int
    for k in range(n):
        area += vx[k] * vy[(k + 1) % n] - vx[(k + 1) % n] * vy[k]
    for k in range(n):
        if area >= 0.0:
            if collect_edge(p, w, vx[k], vy[k], vx[(k + 1) % n], vy[(k + 1) % n]) < 0:
                return -1
        elif collect_edge(p, w, vx[(k + 1) % n], vy[(k + 1) % n], vx[k], vy[k]) < 0:
            return -1
    return 0


# A disc is the polygon of this many sides fill_path fills for one (core_pdf's
# circle_path), its vertices from angle 0 counter-clockwise.
DISC_SIDES = cython.declare(cython.int, 32)
TAU = cython.declare(cython.double, 6.283185307179586)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def collect_disc(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    cx: cython.double,
    cy: cython.double,
    radius: cython.double,
) -> cython.int:
    """Add the disc of the radius around (cx, cy) as circle_path's polygon, whose vertices
    run counter-clockwise."""
    xs = cython.declare(cython.double[32])
    ys = cython.declare(cython.double[32])
    k: cython.int
    angle: cython.double
    xs[0] = cx + radius * cos(0.0)
    ys[0] = cy + radius * sin(0.0)
    for k in range(1, DISC_SIDES):
        angle = k * TAU / DISC_SIDES
        xs[k] = cx + radius * cos(angle)
        ys[k] = cy + radius * sin(angle)
    return collect_polygon(p, w, xs, ys, DISC_SIDES)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def collect_segment(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.int:
    """Add the rectangle around the segment, half a line width either side of it: the
    rectangle fill_line fills for a segment along an axis."""
    half: cython.double = w.half
    if y0 == y1:
        return collect_box(p, w, py_min(x0, x1) - 0.0, y0 - half, py_max(x0, x1) + 0.0, y0 + half)
    if x0 == x1:
        return collect_box(p, w, x0 - half, py_min(y0, y1) - 0.0, x0 + half, py_max(y0, y1) + 0.0)
    dx: cython.double = x1 - x0
    dy: cython.double = y1 - y0
    length: cython.double = pow(dx * dx + dy * dy, p.exponent)
    normal_x: cython.double = -dy / length * half
    normal_y: cython.double = dx / length * half
    xs = cython.declare(cython.double[4])
    ys = cython.declare(cython.double[4])
    xs[0] = x0 + normal_x
    ys[0] = y0 + normal_y
    xs[1] = x1 + normal_x
    ys[1] = y1 + normal_y
    xs[2] = x1 - normal_x
    ys[2] = y1 - normal_y
    xs[3] = x0 - normal_x
    ys[3] = y0 - normal_y
    return collect_polygon(p, w, xs, ys, 4)


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
    if w.collect:
        if collect_segment(p, w, x0, y0, x1, y1) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    if w.cover != cython.NULL:
        if cover_segment(p, w.cover, x0, y0, x1, y1, w.half) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
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
def terminal(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    x: cython.double,
    y: cython.double,
    round_shape: cython.bint,
) -> cython.int:
    """A disc, or a square a line width wide, around (x, y)."""
    found: cython.int
    if w.collect:
        if round_shape:
            found = collect_disc(p, w, x, y, w.half)
        else:
            found = collect_box(p, w, x - w.half, y - w.half, x + w.half, y + w.half)
        if found < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    if w.cover != cython.NULL:
        if round_shape:
            found = cover_disc(p, w.cover, x, y, w.half)
        else:
            found = cover_square(p, w.cover, x, y, 1.0, 0.0, w.half)
        if found < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    if w.native:
        if fill_terminal(p, x, y, w.line_width, round_shape) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    with cython.gil:
        if round_shape:
            calls.circle(x, y)
        else:
            calls.square(x, y)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def polygon(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    xs: cython.p_double,
    ys: cython.p_double,
    n: cython.int,
) -> cython.int:
    if w.collect:
        if collect_polygon(p, w, xs, ys, n) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    if w.cover != cython.NULL:
        if cover_convex(p, w.cover, xs, ys, n) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    if w.native:
        if fill_convex(p, xs, ys, n) < 0:
            with cython.gil:
                failed(p, calls)
        return 0
    with cython.gil:
        if n == 3:
            calls.polygon((xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]))
        else:
            calls.polygon((xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], xs[3], ys[3]))
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
    ax: cython.double,
    ay: cython.double,
    bx: cython.double,
    by: cython.double,
    reaches: cython.bint,
) -> cython.int:
    """The join at (x, y) of a segment arriving along the unit vector (ax, ay) and one
    leaving along (bx, by), reaches when both are at least half the pen's width long. The two
    segments' rectangles leave uncovered only the wedge on the outer side between their
    ends; a miter join fills it out to where the outer edges meet, a bevel join only to the
    line between their corners, and a round join with the disc around (x, y) (ISO 32000-2
    8.4.3.4). A miter whose length over the line width, 1 / sin(angle / 2) =
    sqrt(2 / (1 + cos)), passes the miter limit is a bevel instead (8.4.3.5)."""
    if w.line_join == JOIN_ROUND:
        return terminal(p, w, calls, x, y, True)
    cross: cython.double = ax * by - ay * bx
    if cross == 0.0:
        # Straight on, or straight back, where a miter is infinite and a bevel empty.
        return 0
    dot: cython.double = ax * bx + ay * by
    miter: cython.bint = w.line_join == JOIN_MITER and (1.0 + dot) * w.miter_limit_squared >= 2.0
    if reaches and miter and (ax == 0.0 or ay == 0.0) and (bx == 0.0 or by == 0.0):
        # A right angle between axis-aligned segments: the miter is the outer quarter of
        # the square around the corner, whose other three quarters the segments cover when
        # both reach past it. A shorter one leaves part of them out, which the square would
        # paint, so the miter is then worked out as it is at any other angle.
        return terminal(p, w, calls, x, y, False)
    # The outer side: right of the path on a left turn, left of it on a right turn.
    side: cython.double = w.half if cross > 0.0 else -w.half
    first_x: cython.double = ay * side
    first_y: cython.double = -ax * side
    second_x: cython.double = by * side
    second_y: cython.double = -bx * side
    xs = cython.declare(cython.double[4])
    ys = cython.declare(cython.double[4])
    xs[0] = x
    ys[0] = y
    xs[1] = x + first_x
    ys[1] = y + first_y
    if miter:
        xs[2] = x + (first_x + second_x) / (1.0 + dot)
        ys[2] = y + (first_y + second_y) / (1.0 + dot)
        xs[3] = x + second_x
        ys[3] = y + second_y
        return polygon(p, w, calls, xs, ys, 4)
    xs[2] = x + second_x
    ys[2] = y + second_y
    return polygon(p, w, calls, xs, ys, 3)


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def cap(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    x: cython.double,
    y: cython.double,
    ux: cython.double,
    uy: cython.double,
    reaches: cython.bint,
) -> cython.int:
    """The cap at the end (x, y) of a subpath leaving it along the unit vector (ux, uy):
    nothing, the disc around it, or the half line width square beyond it (8.4.3.3). reaches
    when the segment ending there is at least half the pen's width long."""
    if w.line_cap == CAP_BUTT:
        return 0
    if w.line_cap == CAP_ROUND:
        return terminal(p, w, calls, x, y, True)
    if reaches and (ux == 0.0 or uy == 0.0):
        # Along an axis, the square around the end: its inner half is the segment's, when
        # the segment is that long. Past a shorter one, it would paint beyond the segment's
        # other end, where the cap is not, and what it painted would hang on whether the
        # segment ran along the axis or a rounding away from it.
        return terminal(p, w, calls, x, y, False)
    normal_x: cython.double = -uy * w.half
    normal_y: cython.double = ux * w.half
    ahead_x: cython.double = ux * w.half
    ahead_y: cython.double = uy * w.half
    xs = cython.declare(cython.double[4])
    ys = cython.declare(cython.double[4])
    xs[0] = x + normal_x
    ys[0] = y + normal_y
    xs[1] = x + normal_x + ahead_x
    ys[1] = y + normal_y + ahead_y
    xs[2] = x - normal_x + ahead_x
    ys[2] = y - normal_y + ahead_y
    xs[3] = x - normal_x
    ys[3] = y - normal_y
    return polygon(p, w, calls, xs, ys, 4)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def axis_reaches(dx: cython.double, dy: cython.double, half: cython.double) -> cython.bint:
    """Whether a segment of some length (dx, dy) along an axis, whose length is then the sum
    of its sides, exactly, is at least half the pen's width long: then the square around its
    end or corner holds no more than its cap or miter adds to the segment. cap and join ask
    only of segments along an axis; in_pieces and axis_aligned_polylines ask the same."""
    return fabs(dx) + fabs(dy) >= half


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def axis_short(
    x0: cython.double, y0: cython.double, x1: cython.double, y1: cython.double, half: cython.double
) -> cython.bint:
    """Whether the segment from (x0, y0) to (x1, y1) has some length but does not reach."""
    return (x0 != x1 or y0 != y1) and not axis_reaches(x1 - x0, y1 - y0, half)


# A subpath shorter than this, in device pixels, may fall between the samples whatever
# its width: a segment across a thick pen with butt caps is a bar as thin as it is long.
SHORT_PIXELS = cython.declare(cython.double, 1.0)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def short_span(
    xs: cython.p_const_double,
    ys: cython.p_const_double,
    start: cython.Py_ssize_t,
    end: cython.Py_ssize_t,
    closed: cython.bint,
    scale: cython.double,
    forward: cython.p_const_double,
) -> cython.bint:
    """Whether the subpath of the points start to end, closed or not, has a segment of some
    length but is shorter, all its segments together and mapped by forward (a, b, c, d) or
    by nothing to the page, than SHORT_PIXELS device pixels."""
    index: cython.Py_ssize_t
    following: cython.Py_ssize_t
    dx: cython.double
    dy: cython.double
    page_x: cython.double
    page_y: cython.double
    total: cython.double = 0.0
    last: cython.Py_ssize_t = end if closed else end - 1
    limit: cython.double = SHORT_PIXELS / scale
    for index in range(start, last):
        following = index + 1 if index + 1 < end else start
        dx = xs[following] - xs[index]
        dy = ys[following] - ys[index]
        if forward != cython.NULL:
            page_x = forward[0] * dx + forward[2] * dy
            page_y = forward[1] * dx + forward[3] * dy
            dx = page_x
            dy = page_y
        total += sqrt(dx * dx + dy * dy)
        if not total < limit:
            return False
    return total > 0.0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def walk(
    p: cython.pointer[Paint], w: cython.pointer[Walk], calls: typing.Optional[Callbacks]
) -> cython.int:
    return walk_range(p, w, calls, 0, w.count)


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def walk_range(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    first: cython.Py_ssize_t,
    last: cython.Py_ssize_t,
) -> cython.int:
    """Stroke the subpaths first to last."""
    k: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t
    closed: cython.bint
    found: cython.bint
    x0: cython.double
    y0: cython.double
    x1: cython.double
    y1: cython.double
    dx: cython.double
    dy: cython.double
    length: cython.double
    ux: cython.double
    uy: cython.double
    first_x: cython.double = 0.0
    first_y: cython.double = 0.0
    last_x: cython.double = 0.0
    last_y: cython.double = 0.0
    first_reaches: cython.bint = False
    last_reaches: cython.bint = False
    reaches: cython.bint
    for k in range(first, last):
        start = w.bounds[3 * k]
        end = w.bounds[3 * k + 1]
        closed = w.bounds[3 * k + 2]
        if end - start < 2:
            continue
        x0 = w.xs[start]
        y0 = w.ys[start]
        x1 = w.xs[end - 1]
        y1 = w.ys[end - 1]
        # The segments first, then the joins and caps, each joining the directions of the
        # segments either side of it: a zero-length segment has none, and is passed over.
        found = False
        for index in range(start, end - 1):
            if w.xs[index] == w.xs[index + 1] and w.ys[index] == w.ys[index + 1]:
                continue
            found = True
            line(p, w, calls, w.xs[index], w.ys[index], w.xs[index + 1], w.ys[index + 1])
        if not found:
            # A subpath of one point: a dot with round caps, and nothing otherwise, as its
            # caps have no direction (ISO 32000-2 8.5.3.2).
            if w.line_cap == CAP_ROUND and w.collect:
                if collect_disc(p, w, x0, y0, w.half) < 0:
                    with cython.gil:
                        failed(p, calls)
            elif w.line_cap == CAP_ROUND and w.cover != cython.NULL:
                if (
                    cover_disc(p, w.cover, x0, y0, w.half) < 0
                    or cover_nearest(p, w.cover, x0, y0) < 0
                ):
                    with cython.gil:
                        failed(p, calls)
            elif w.line_cap == CAP_ROUND:
                with cython.gil:
                    flush(p, calls)
                    calls.dot(x0, y0)
            continue
        if closed and (x0 != x1 or y0 != y1):
            line(p, w, calls, x1, y1, x0, y0)
        found = False
        for index in range(start, end - 1):
            dx = w.xs[index + 1] - w.xs[index]
            dy = w.ys[index + 1] - w.ys[index]
            if dx == 0.0 and dy == 0.0:
                continue
            length = pow(dx * dx + dy * dy, p.exponent)
            ux = dx / length
            uy = dy / length
            reaches = axis_reaches(dx, dy, w.half)
            if found:
                join(
                    p,
                    w,
                    calls,
                    w.xs[index],
                    w.ys[index],
                    last_x,
                    last_y,
                    ux,
                    uy,
                    last_reaches and reaches,
                )
            else:
                first_x = ux
                first_y = uy
                first_reaches = reaches
                found = True
            last_x = ux
            last_y = uy
            last_reaches = reaches
        if closed:
            if x0 != x1 or y0 != y1:
                dx = x0 - x1
                dy = y0 - y1
                length = pow(dx * dx + dy * dy, p.exponent)
                ux = dx / length
                uy = dy / length
                reaches = axis_reaches(dx, dy, w.half)
                join(p, w, calls, x1, y1, last_x, last_y, ux, uy, last_reaches and reaches)
                last_x = ux
                last_y = uy
                last_reaches = reaches
            join(
                p,
                w,
                calls,
                x0,
                y0,
                last_x,
                last_y,
                first_x,
                first_y,
                last_reaches and first_reaches,
            )
        else:
            cap(p, w, calls, x0, y0, -first_x, -first_y, first_reaches)
            cap(p, w, calls, x1, y1, last_x, last_y, last_reaches)
        if (
            w.cover != cython.NULL
            and short_span(
                w.xs,
                w.ys,
                start,
                end,
                closed,
                p.scale,
                w.cover.forward if w.cover.transformed else cython.NULL,
            )
            # A subpath shorter than a pixel covers, as a dot does, the sample nearest the
            # middle of its points too, so that it never disappears (10.7.4).
            and cover_nearest(p, w.cover, middle(w.xs, start, end), middle(w.ys, start, end)) < 0
        ):
            with cython.gil:
                failed(p, calls)
    return 0


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def middle(
    values: cython.p_const_double, start: cython.Py_ssize_t, end: cython.Py_ssize_t
) -> cython.double:
    """The middle of the least and the greatest of values start to end."""
    low: cython.double = values[start]
    high: cython.double = values[start]
    index: cython.Py_ssize_t
    for index in range(start + 1, end):
        low = py_min(low, values[index])
        high = py_max(high, values[index])
    return (low + high) * 0.5


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def in_pieces(
    w: cython.pointer[Walk],
    first: cython.Py_ssize_t,
    last: cython.Py_ssize_t,
    opaque: cython.bint,
    scale: cython.double,
) -> cython.bint:
    """Whether painting the stroke of subpaths first to last piece by piece paints what
    painting it once would: it is one segment with butt caps, or, opaque, its pieces only
    overlap,
    every segment running along an axis and every corner a disc or the square around it,
    with no miter or bevel polygon meeting a segment's end along an edge, where two partial
    coverages painted in turn leave a lighter seam; and none of its subpaths is shorter
    than SHORT_PIXELS, which paint_once keeps from disappearing. A segment shorter than half
    the pen's width has its square caps and miters worked out as polygons, as at any angle,
    so a stroke with one paints at once unless its joins are round and its caps are not
    square."""
    k: cython.Py_ssize_t
    index: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    segments: cython.Py_ssize_t = 0
    aligned: cython.bint = True
    short: cython.bint = False
    for k in range(first, last):
        start = w.bounds[3 * k]
        end = w.bounds[3 * k + 1]
        if end - start < 2:
            continue
        if short_span(w.xs, w.ys, start, end, w.bounds[3 * k + 2], scale, cython.NULL):
            # paint_once covers the sample nearest it.
            return False
        for index in range(start, end - 1):
            if w.xs[index] == w.xs[index + 1] and w.ys[index] == w.ys[index + 1]:
                continue
            segments += 1
            if w.xs[index] != w.xs[index + 1] and w.ys[index] != w.ys[index + 1]:
                aligned = False
            if axis_short(w.xs[index], w.ys[index], w.xs[index + 1], w.ys[index + 1], w.half):
                short = True
        if w.bounds[3 * k + 2] and (w.xs[start] != w.xs[end - 1] or w.ys[start] != w.ys[end - 1]):
            segments += 1
            if w.xs[start] != w.xs[end - 1] and w.ys[start] != w.ys[end - 1]:
                aligned = False
            if axis_short(w.xs[end - 1], w.ys[end - 1], w.xs[start], w.ys[start], w.half):
                short = True
        elif w.bounds[3 * k + 2]:
            # Closed: its last point meets its first at a join.
            segments += 1
    if segments <= 1 and w.line_cap == CAP_BUTT:
        return True
    if not opaque or not aligned or w.line_join == JOIN_BEVEL:
        return False
    if short and (w.line_join == JOIN_MITER or w.line_cap == CAP_SQUARE):
        return False
    # A right angle's miter is a bevel below a miter limit of the square root of 2.
    return w.line_join != JOIN_MITER or w.miter_limit_squared >= 2.0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def paint_once(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    first: cython.Py_ssize_t,
    last: cython.Py_ssize_t,
    buffer: cython.pointer[cython.p_ushort],
    capacity: cython.p_Py_ssize_t,
) -> cython.int:
    """Paint the stroke of subpaths first to last once: count the samples of each pixel any
    of its pieces covers, each once, then blend each pixel by that coverage (ISO 32000-2
    8.5.3.2). buffer holds capacity sample masks, and grows to the stroke's box."""
    k: cython.Py_ssize_t
    index: cython.Py_ssize_t
    found: cython.bint = False
    low_x: cython.double = 0.0
    low_y: cython.double = 0.0
    high_x: cython.double = 0.0
    high_y: cython.double = 0.0
    for k in range(first, last):
        for index in range(w.bounds[3 * k], w.bounds[3 * k + 1]):
            if not found:
                low_x = w.xs[index]
                high_x = w.xs[index]
                low_y = w.ys[index]
                high_y = w.ys[index]
                found = True
                continue
            low_x = py_min(low_x, w.xs[index])
            high_x = py_max(high_x, w.xs[index])
            low_y = py_min(low_y, w.ys[index])
            high_y = py_max(high_y, w.ys[index])
    if not found:
        return 0
    # A miter's tip reaches at most the miter limit times half the line width from its
    # corner, and a projecting cap's corner the square root of 2 times.
    reach: cython.double = w.half * (
        py_max(sqrt(w.miter_limit_squared), 1.5) if w.line_join == JOIN_MITER else 1.5
    )
    box = cython.declare(cython.double[4])
    box[0] = low_x - reach
    box[1] = low_y - reach
    box[2] = high_x + reach
    box[3] = high_y + reach
    pixel_box = cython.declare(cython.Py_ssize_t[4])
    result: cython.int = clipped_pixel_box(p, box, pixel_box)
    if result < 0:
        with cython.gil:
            failed(p, calls)
    if result <= 0:
        return 0
    c = cython.declare(Cover)
    c.left = pixel_box[0]
    c.top = pixel_box[1]
    c.width = pixel_box[2] - pixel_box[0]
    c.height = pixel_box[3] - pixel_box[1]
    c.crop_x0 = p.crop_x0
    c.crop_y1 = p.crop_y1
    c.scale = p.scale
    c.transformed = False
    size: cython.Py_ssize_t = c.width * c.height
    if size > capacity[0]:
        free(buffer[0])
        buffer[0] = cython.cast(cython.p_ushort, malloc(size * cython.sizeof(cython.ushort)))
        capacity[0] = 0 if buffer[0] == cython.NULL else size
        if buffer[0] == cython.NULL:
            p.error = FAILED_MEMORY
            with cython.gil:
                failed(p, calls)
    c.masks = buffer[0]
    memset(c.masks, 0, size * cython.sizeof(cython.ushort))
    w.cover = cython.address(c)
    walk_range(p, w, calls, first, last)
    w.cover = cython.NULL
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    py: cython.Py_ssize_t
    px: cython.Py_ssize_t
    bits: cython.ushort
    alpha: cython.int
    span_count: cython.Py_ssize_t = 0
    spans: cython.p_const_longlong = cython.NULL
    painted: cython.bint = False
    window = cython.declare(cython.Py_ssize_t[4])
    for i in range(c.height):
        py = c.top + i
        if p.clip_mode == CLIP_ROWS:
            span_count = row_spans(p, py, cython.address(spans))
        for j in range(c.width):
            bits = c.masks[i * c.width + j]
            if not bits:
                continue
            px = c.left + j
            if p.clip_mode == CLIP_ROWS and not pixel_in_row(spans, span_count, px):
                continue
            alpha = coverage_alpha(p.alpha, sample_count(bits))
            if alpha <= 0:
                continue
            blend_normal_pixel(p.pixels + py * p.row_stride + px * 4, p.red, p.green, p.blue, alpha)
            if not painted:
                painted = True
                window[0] = px
                window[1] = py
                window[2] = px + 1
                window[3] = py + 1
            else:
                window[0] = px if px < window[0] else window[0]
                window[2] = px + 1 if px + 1 > window[2] else window[2]
                window[3] = py + 1
    if painted:
        extend(p, window[1], window[3], window[0], window[2])
    return 0


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def x_on_pixel_edge(p: cython.pointer[Paint], x: cython.double) -> cython.bint:
    return on_pixel_edge((x - p.crop_x0) * p.scale)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def y_on_pixel_edge(p: cython.pointer[Paint], y: cython.double) -> cython.bint:
    return on_pixel_edge((p.crop_y1 - y) * p.scale)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def segment_on_pixels(
    p: cython.pointer[Paint],
    half: cython.double,
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
) -> cython.bint:
    """Whether the rectangle around the segment, half a line width either side of it, has
    its ends along the run and its sides on pixel edges; False for a segment off the axes.
    A segment of no length has no rectangle."""
    if x0 == x1 and y0 == y1:
        return True
    if y0 == y1:
        return (
            x_on_pixel_edge(p, x0)
            and x_on_pixel_edge(p, x1)
            and y_on_pixel_edge(p, y0 - half)
            and y_on_pixel_edge(p, y0 + half)
        )
    if x0 == x1:
        return (
            y_on_pixel_edge(p, y0)
            and y_on_pixel_edge(p, y1)
            and x_on_pixel_edge(p, x0 - half)
            and x_on_pixel_edge(p, x0 + half)
        )
    return False


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def pieces_on_pixels(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    first: cython.Py_ssize_t,
    last: cython.Py_ssize_t,
) -> cython.bint:
    """Whether painting the stroke of subpaths first to last piece by piece, as in_pieces
    allows, covers each pixel once: it is one segment with butt caps, a single piece; or
    its pieces run along the axes and every edge of them falls on a pixel edge, each
    segment's ends along its run and its sides half a line width either way, and the sides
    of each square cap and of each corner's square, half a line width around its point, so
    that no two pieces share a partly covered pixel. Pieces meeting inside a pixel would
    blend it in turn, leaving a seam where they abut and a darker edge where they overlap,
    and a disc always meets the rectangle it caps inside pixels: paint_once covers such a
    stroke's samples once instead."""
    k: cython.Py_ssize_t
    index: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    closed: cython.bint
    segments: cython.Py_ssize_t = 0
    for k in range(first, last):
        start = w.bounds[3 * k]
        end = w.bounds[3 * k + 1]
        for index in range(start, end - 1):
            if w.xs[index] != w.xs[index + 1] or w.ys[index] != w.ys[index + 1]:
                segments += 1
        if w.bounds[3 * k + 2] and end - start >= 2:
            segments += 1
    if segments <= 1 and w.line_cap == CAP_BUTT:
        return True
    if w.line_cap == CAP_ROUND or w.line_join == JOIN_ROUND:
        return False
    half: cython.double = w.half
    x: cython.double
    y: cython.double
    for k in range(first, last):
        start = w.bounds[3 * k]
        end = w.bounds[3 * k + 1]
        closed = w.bounds[3 * k + 2] != 0
        for index in range(start, end):
            x = w.xs[index]
            y = w.ys[index]
            if index + 1 < end:
                if not segment_on_pixels(p, half, x, y, w.xs[index + 1], w.ys[index + 1]):
                    return False
            elif closed and not segment_on_pixels(p, half, x, y, w.xs[start], w.ys[start]):
                return False
            # A corner's square, or a square cap's at an open end.
            if (
                closed or (index != start and index != end - 1) or w.line_cap == CAP_SQUARE
            ) and not (
                x_on_pixel_edge(p, x - half)
                and x_on_pixel_edge(p, x + half)
                and y_on_pixel_edge(p, y - half)
                and y_on_pixel_edge(p, y + half)
            ):
                return False
    return True


# fill_rect_union works on the stack for up to this many rectangles, over up to this
# many pixel columns: its stack arrays' sizes follow from them.
SMALL_UNION = cython.declare(cython.int, 32)
SMALL_WIDTH = cython.declare(cython.int, 512)
# A rectangle in device space, left to right and top to bottom.
DeviceRect = cython.struct(
    left=cython.double, right=cython.double, top=cython.double, bottom=cython.double
)
Span = cython.struct(left=cython.double, right=cython.double)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compare_tops(
    first: cython.pointer[cython.const[cython.void]],
    second: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    a: cython.double = cython.cast(cython.pointer[cython.const[DeviceRect]], first).top
    b: cython.double = cython.cast(cython.pointer[cython.const[DeviceRect]], second).top
    return (a > b) - (a < b)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compare_lefts(
    first: cython.pointer[cython.const[cython.void]],
    second: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    a: cython.double = cython.cast(cython.pointer[cython.const[Span]], first).left
    b: cython.double = cython.cast(cython.pointer[cython.const[Span]], second).left
    return (a > b) - (a < b)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compare_doubles(
    first: cython.pointer[cython.const[cython.void]],
    second: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    a: cython.double = cython.cast(cython.pointer[cython.const[cython.double]], first)[0]
    b: cython.double = cython.cast(cython.pointer[cython.const[cython.double]], second)[0]
    return (a > b) - (a < b)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def sort_doubles(values: cython.p_double, count: cython.Py_ssize_t) -> cython.void:
    """Sort values ascending: by insertion for the few a pixel row holds, else qsort."""
    if count > 16:
        qsort(values, count, cython.sizeof(cython.double), compare_doubles)
        return
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    value: cython.double
    for i in range(1, count):
        value = values[i]
        j = i - 1
        while j >= 0 and values[j] > value:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = value


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def sort_spans(spans: cython.pointer[Span], count: cython.Py_ssize_t) -> cython.void:
    """Sort spans by their left ends: by insertion for a few, else qsort."""
    if count > 16:
        qsort(spans, count, cython.sizeof(Span), compare_lefts)
        return
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    span: Span
    for i in range(1, count):
        span = spans[i]
        j = i - 1
        while j >= 0 and spans[j].left > span.left:
            spans[j + 1] = spans[j]
            j -= 1
        spans[j + 1] = span


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def sort_tops(rects: cython.pointer[DeviceRect], count: cython.Py_ssize_t) -> cython.void:
    """Sort rectangles by their tops: by insertion for a few, else qsort."""
    if count > 16:
        qsort(rects, count, cython.sizeof(DeviceRect), compare_tops)
        return
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    rect: DeviceRect
    for i in range(1, count):
        rect = rects[i]
        j = i - 1
        while j >= 0 and rects[j].top > rect.top:
            rects[j + 1] = rects[j]
            j -= 1
        rects[j + 1] = rect


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def span_coverage(index: cython.double, low: cython.double, high: cython.double) -> cython.double:
    """The share of pixel index that [low, high) covers along one axis, as fill_rect_pixels
    measures it."""
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
def fill_rect_union(p: cython.pointer[Paint], w: cython.pointer[Walk]) -> cython.int:
    """Fill the union of the collected rectangles, opaque or at p.alpha in the Normal mode,
    by its exact area: each pixel row is cut into bands at the rectangles' tops and
    bottoms inside it, each band's rectangles' spans are merged, and a pixel's coverage is
    the sum over the bands of the band's height times the share of the pixel its merged
    spans cover. A pixel one rectangle covers alone gets the coverage, and the colour,
    fill_rect_pixels gives it. -1 when memory runs out."""
    n: cython.Py_ssize_t = w.rect_count
    if n == 0:
        return 0
    k: cython.Py_ssize_t
    box = cython.declare(cython.double[4])
    box[0] = w.union_rects[0]
    box[1] = w.union_rects[1]
    box[2] = w.union_rects[2]
    box[3] = w.union_rects[3]
    for k in range(1, n):
        box[0] = py_min(box[0], w.union_rects[4 * k])
        box[1] = py_min(box[1], w.union_rects[4 * k + 1])
        box[2] = py_max(box[2], w.union_rects[4 * k + 2])
        box[3] = py_max(box[3], w.union_rects[4 * k + 3])
    pixel_box = cython.declare(cython.Py_ssize_t[4])
    found: cython.int = clipped_pixel_box(p, box, pixel_box)
    if found <= 0:
        return found
    ix0: cython.Py_ssize_t = pixel_box[0]
    iy0: cython.Py_ssize_t = pixel_box[1]
    ix1: cython.Py_ssize_t = pixel_box[2]
    iy1: cython.Py_ssize_t = pixel_box[3]
    width: cython.Py_ssize_t = ix1 - ix0
    # A stroke of a few pieces, a table cell's frame, works on the stack.
    small_rects = cython.declare(cython.double[128])
    small_active = cython.declare(cython.Py_ssize_t[32])
    small_spans = cython.declare(cython.double[64])
    small_cuts = cython.declare(cython.double[66])
    small_coverage = cython.declare(cython.double[512])
    heap: cython.bint = n > SMALL_UNION
    wide: cython.bint = width > SMALL_WIDTH
    rects: cython.pointer[DeviceRect] = cython.cast(cython.pointer[DeviceRect], small_rects)
    active: cython.p_Py_ssize_t = small_active
    spans: cython.pointer[Span] = cython.cast(cython.pointer[Span], small_spans)
    cuts: cython.p_double = small_cuts
    coverage: cython.p_double = small_coverage
    if heap:
        rects = cython.cast(cython.pointer[DeviceRect], malloc(n * cython.sizeof(DeviceRect)))
        active = cython.cast(cython.p_Py_ssize_t, malloc(n * cython.sizeof(cython.Py_ssize_t)))
        spans = cython.cast(cython.pointer[Span], malloc(n * cython.sizeof(Span)))
        cuts = cython.cast(cython.p_double, malloc((2 * n + 2) * cython.sizeof(cython.double)))
    if wide:
        coverage = cython.cast(cython.p_double, calloc(width, cython.sizeof(cython.double)))
    else:
        memset(coverage, 0, width * cython.sizeof(cython.double))
    if (
        rects == cython.NULL
        or active == cython.NULL
        or spans == cython.NULL
        or cuts == cython.NULL
        or coverage == cython.NULL
    ):
        if heap:
            free(rects)
            free(active)
            free(spans)
            free(cuts)
        if wide:
            free(coverage)
        p.error = FAILED_MEMORY
        return -1
    for k in range(n):
        rects[k].left = (w.union_rects[4 * k] - p.crop_x0) * p.scale
        rects[k].right = (w.union_rects[4 * k + 2] - p.crop_x0) * p.scale
        rects[k].top = (p.crop_y1 - w.union_rects[4 * k + 3]) * p.scale
        rects[k].bottom = (p.crop_y1 - w.union_rects[4 * k + 1]) * p.scale
    sort_tops(rects, n)
    alpha: cython.double = cython.cast(cython.double, p.alpha)
    opaque_from: cython.int = 255 if p.alpha >= 255 else 256
    red: cython.float = cython.cast(cython.float, p.red)
    green: cython.float = cython.cast(cython.float, p.green)
    blue: cython.float = cython.cast(cython.float, p.blue)
    opaque = cython.declare(cython.uchar[4])
    opaque[0] = opaque_channel(red)
    opaque[1] = opaque_channel(green)
    opaque[2] = opaque_channel(blue)
    opaque[3] = 255
    next_rect: cython.Py_ssize_t = 0
    active_count: cython.Py_ssize_t = 0
    kept: cython.Py_ssize_t
    cut_count: cython.Py_ssize_t
    span_count: cython.Py_ssize_t
    merged: cython.Py_ssize_t
    a: cython.Py_ssize_t
    c: cython.Py_ssize_t
    py: cython.Py_ssize_t
    px: cython.Py_ssize_t
    first: cython.Py_ssize_t
    end: cython.Py_ssize_t
    low_px: cython.Py_ssize_t
    high_px: cython.Py_ssize_t
    row_top: cython.double
    row_bottom: cython.double
    band_top: cython.double
    band_bottom: cython.double
    height: cython.double
    left: cython.double
    right: cython.double
    value: cython.double
    raw: cython.int
    pixel: cython.p_uchar
    rect: cython.pointer[DeviceRect]
    painted: cython.bint = False
    window = cython.declare(cython.Py_ssize_t[4])
    for py in range(iy0, iy1):
        row_top = cython.cast(cython.double, py)
        row_bottom = row_top + 1.0
        while next_rect < n and rects[next_rect].top < row_bottom:
            active[active_count] = next_rect
            active_count += 1
            next_rect += 1
        kept = 0
        for a in range(active_count):
            if rects[active[a]].bottom > row_top:
                active[kept] = active[a]
                kept += 1
        active_count = kept
        if active_count == 0:
            continue
        # The row cut at the tops and bottoms inside it.
        cuts[0] = row_top
        cuts[1] = row_bottom
        cut_count = 2
        for a in range(active_count):
            rect = cython.address(rects[active[a]])
            if row_top < rect.top < row_bottom:
                cuts[cut_count] = rect.top
                cut_count += 1
            if row_top < rect.bottom < row_bottom:
                cuts[cut_count] = rect.bottom
                cut_count += 1
        sort_doubles(cuts, cut_count)
        low_px = ix1
        high_px = ix0
        for c in range(cut_count - 1):
            band_top = cuts[c]
            band_bottom = cuts[c + 1]
            if band_bottom <= band_top:
                continue
            height = band_bottom - band_top
            span_count = 0
            for a in range(active_count):
                rect = cython.address(rects[active[a]])
                if rect.top <= band_top and rect.bottom >= band_bottom and rect.right > rect.left:
                    spans[span_count].left = rect.left
                    spans[span_count].right = rect.right
                    span_count += 1
            if span_count == 0:
                continue
            sort_spans(spans, span_count)
            merged = 0
            for a in range(1, span_count):
                if spans[a].left <= spans[merged].right:
                    if spans[a].right > spans[merged].right:
                        spans[merged].right = spans[a].right
                else:
                    merged += 1
                    spans[merged] = spans[a]
            for a in range(merged + 1):
                left = spans[a].left
                right = spans[a].right
                first = cython.cast(cython.Py_ssize_t, floor(left))
                end = cython.cast(cython.Py_ssize_t, ceil(right))
                if first < ix0:
                    first = ix0
                if end > ix1:
                    end = ix1
                for px in range(first, end):
                    coverage[px - ix0] += height * span_coverage(
                        cython.cast(cython.double, px), left, right
                    )
                if first < low_px:
                    low_px = first
                if end > high_px:
                    high_px = end
        for px in range(low_px, high_px):
            value = coverage[px - ix0]
            if value <= 0.0:
                continue
            coverage[px - ix0] = 0.0
            if value > 1.0:
                value = 1.0
            raw = cython.cast(cython.int, cython.cast(cython.uchar, rint(value * alpha)))
            if raw == 0:
                continue
            pixel = p.pixels + py * p.row_stride + px * 4
            if raw >= opaque_from:
                memcpy(pixel, opaque, 4)
            else:
                blend_one(
                    cython.address(pixel[0]),
                    cython.address(pixel[1]),
                    cython.address(pixel[2]),
                    cython.address(pixel[3]),
                    raw,
                    p.alpha,
                    red,
                    green,
                    blue,
                )
            if not painted:
                painted = True
                window[0] = px
                window[1] = py
                window[2] = px + 1
                window[3] = py + 1
            else:
                window[0] = px if px < window[0] else window[0]
                window[2] = px + 1 if px + 1 > window[2] else window[2]
                window[3] = py + 1
    if heap:
        free(rects)
        free(active)
        free(spans)
        free(cuts)
    if wide:
        free(coverage)
    if painted:
        extend(p, window[1], window[3], window[0], window[2])
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def paint_union(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    first: cython.Py_ssize_t,
    last: cython.Py_ssize_t,
) -> cython.int:
    """Paint the stroke of subpaths first to last, whose pieces in_pieces allows but meet
    inside pixels, once: collect its pieces' rectangles, squares, polygons and discs (as
    circle_path's polygons), each counter-clockwise so that the nonzero rule fills their
    union. Rectangles alone, fill_rect_union fills by their exact area here; with any other
    shape among them, calls.union takes their edges and the box they span in one call,
    which fills them by their exact area as fill_path fills the same pieces as one path."""
    w.collect = True
    w.union_count = 0
    w.rect_count = 0
    w.union_shapes = False
    w.boxes_only = w.line_cap != CAP_ROUND and w.line_join != JOIN_ROUND
    result: cython.int = walk_range(p, w, calls, first, last)
    if result >= 0 and p.error == 0 and w.boxes_only and w.union_shapes:
        # A polygon after all: collected again, every piece's edges in order.
        w.union_count = 0
        w.rect_count = 0
        w.boxes_only = False
        result = walk_range(p, w, calls, first, last)
    w.collect = False
    w.boxes_only = False
    if result < 0 or p.error != 0:
        return -1
    if w.union_count == 0 and w.rect_count == 0:
        return 0
    if not w.union_shapes:
        # Rectangles alone: their union filled here.
        if fill_rect_union(p, w) < 0:
            with cython.gil:
                failed(p, calls)
            return -1
        return 0
    with cython.gil:
        flush(p, calls)
        calls.union(
            PyBytes_FromStringAndSize(
                cython.cast(cython.p_char, w.union_edges),
                w.union_count * 4 * cython.sizeof(cython.double),
            ),
            (w.union_box[0], w.union_box[1], w.union_box[2], w.union_box[3]),
        )
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(-1, check=False)
def walk_once(
    p: cython.pointer[Paint],
    w: cython.pointer[Walk],
    calls: typing.Optional[Callbacks],
    ends: cython.p_longlong,
    groups: cython.Py_ssize_t,
    buffer: cython.pointer[cython.p_ushort],
    capacity: cython.p_Py_ssize_t,
) -> cython.int:
    """Paint each group of subpaths, the strokes ending at ends, once, each in turn, by
    one of three routes. A group in_pieces allows, opaque along the axes, whose pieces
    share no partly covered pixel (pieces_on_pixels), piece by piece (walk_range), each
    piece by its exact area. One whose pieces meet inside pixels, without a clip path or
    under a rectangular one, by paint_union: their union in one call, by its exact area,
    as a filled path's. Every other group, and any under a clip path, by paint_once: its
    pieces' 4 x 4 samples covered once, as a filled path's are under a clip path, in
    buffer, which holds capacity sample masks and which the caller frees, an error raised
    or not."""
    group: cython.Py_ssize_t
    first: cython.Py_ssize_t = 0
    last: cython.Py_ssize_t
    result: cython.int = 0
    pieces: cython.bint
    for group in range(groups):
        last = ends[group]
        pieces = in_pieces(w, first, last, p.alpha >= 255, p.scale)
        if pieces and pieces_on_pixels(p, w, first, last):
            result = walk_range(p, w, calls, first, last)
        elif pieces and p.clip_mode != CLIP_ROWS and calls.union is not None:
            result = paint_union(p, w, calls, first, last)
        else:
            result = paint_once(p, w, calls, first, last, buffer, capacity)
        if result < 0:
            break
        first = last
    return result


def stroke_polylines(
    pixels: cython.uchar[:, :, ::1],
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: typing.Optional[list],
    outline: cython.bint,
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
    line_cap: cython.int,
    line_join: cython.int,
    miter_limit: cython.double,
    exponent: cython.double,
    once: cython.bint,
    group_ends: typing.Optional[list],
    on_line,
    on_polygon,
    on_square,
    on_circle,
    on_dot,
    on_extend,
    on_union=None,
):
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    count: cython.Py_ssize_t = len(spans)
    if clip_mode == CLIP_ROWS:
        if row_offsets is None or row_spans is None or row_offsets.shape[0] < 1:
            raise ValueError("a clip path needs its rows")
        if row_offsets[row_offsets.shape[0] - 1] * 2 > row_spans.shape[0]:
            raise ValueError("row offsets run past the spans")
    if once and not native:
        raise ValueError("a stroke painted once is painted natively")
    groups: cython.Py_ssize_t = 1 if group_ends is None else len(group_ends)
    bounds: cython.pointer[cython.longlong] = cython.cast(
        cython.pointer[cython.longlong],
        malloc((3 * count + groups + 1) * cython.sizeof(cython.longlong)),
    )
    if bounds == cython.NULL:
        raise MemoryError
    ends: cython.p_longlong = bounds + 3 * count
    # paint_once's sample masks, grown to each stroke's box and freed here.
    buffer: cython.p_ushort = cython.NULL
    capacity: cython.Py_ssize_t = 0
    k: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    p = cython.declare(Paint)
    w = cython.declare(Walk)
    w.collect = False
    w.union_edges = cython.NULL
    w.union_count = 0
    w.union_capacity = 0
    w.union_rects = cython.NULL
    w.rect_count = 0
    w.rect_capacity = 0
    w.union_shapes = False
    w.boxes_only = False
    calls: Callbacks = Callbacks.__new__(Callbacks)
    calls.line = on_line
    calls.polygon = on_polygon
    calls.square = on_square
    calls.circle = on_circle
    calls.dot = on_dot
    calls.extend = on_extend
    calls.union = on_union
    try:
        for k in range(count):
            start, end, flag = spans[k]
            if start < 0 or end > xs.shape[0] or end < start:
                raise ValueError("span runs past the points")
            bounds[3 * k] = start
            bounds[3 * k + 1] = end
            bounds[3 * k + 2] = 1 if (outline or flag) else 0
        if group_ends is None:
            ends[0] = count
        else:
            start = 0
            for k in range(groups):
                end = group_ends[k]
                if end < start or end > count:
                    raise ValueError("groups run past the subpaths")
                ends[k] = end
                start = end
            if groups and start != count:
                raise ValueError("groups leave subpaths out")
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
        w.count = count
        w.line_width = line_width
        w.half = pen_half(scale, line_width)
        w.native = native
        w.line_cap = line_cap
        w.line_join = line_join
        w.miter_limit_squared = miter_limit * miter_limit
        w.cover = cython.NULL
        with cython.nogil:
            if once:
                walk_once(
                    cython.address(p),
                    cython.address(w),
                    calls,
                    ends,
                    groups,
                    cython.address(buffer),
                    cython.address(capacity),
                )
            else:
                walk(cython.address(p), cython.address(w), calls)
        flush(cython.address(p), calls)
    finally:
        free(w.union_edges)
        free(w.union_rects)
        free(buffer)
        free(bounds)


def stroke_coverage(
    counts: cython.uchar[:, ::1],
    left: cython.Py_ssize_t,
    top: cython.Py_ssize_t,
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: list,
    outline: cython.bint,
    squares: cython.const[cython.double][::1],
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    matrix: typing.Optional[tuple],
    line_width: cython.double,
    line_cap: cython.int,
    line_join: cython.int,
    miter_limit: cython.double,
    exponent: cython.double,
) -> bool:
    """Count, in counts, the 4 x 4 samples of each pixel of the window at (left, top) that
    the stroke of the subpaths covers: its segments, joins and caps, and the square around
    each point of squares, (x, y, ux, uy) with the path along (ux, uy) there, as a dash of
    zero length with projecting caps paints, each such square, and each dot of round caps,
    covering at least the sample nearest its centre. A sample two pieces cover counts once, as a
    stroke is one shape painted once (ISO 32000-2 8.5.3.2). The points are in stroke space,
    which matrix, (a, b, c, d) or None for the identity, maps to page space; the line width
    is a diameter of the pen there. True if any sample is covered."""
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    if squares.shape[0] % 4:
        raise ValueError("squares hold four numbers each")
    count: cython.Py_ssize_t = len(spans)
    height: cython.Py_ssize_t = counts.shape[0]
    width: cython.Py_ssize_t = counts.shape[1]
    c = cython.declare(Cover)
    c.transformed = matrix is not None
    smallest: cython.double = 1.0
    determinant: cython.double
    if c.transformed:
        c.forward[0] = matrix[0]
        c.forward[1] = matrix[1]
        c.forward[2] = matrix[2]
        c.forward[3] = matrix[3]
        determinant = c.forward[0] * c.forward[3] - c.forward[1] * c.forward[2]
        if determinant == 0.0 or isnan(determinant) or isinf(determinant):
            raise ValueError("the stroke matrix is not invertible")
        c.inverse[0] = c.forward[3] / determinant
        c.inverse[1] = -c.forward[1] / determinant
        c.inverse[2] = -c.forward[2] / determinant
        c.inverse[3] = c.forward[0] / determinant
        # The matrix's smaller singular value: the least it shrinks a length by, which
        # pen_half widens a thin pen by to keep it a device pixel wide on the page in
        # every direction.
        total: cython.double = (
            c.forward[0] * c.forward[0]
            + c.forward[1] * c.forward[1]
            + c.forward[2] * c.forward[2]
            + c.forward[3] * c.forward[3]
        )
        largest: cython.double = sqrt(
            (total + sqrt(py_max(0.0, total * total - 4.0 * determinant * determinant))) * 0.5
        )
        smallest = fabs(determinant) / largest
    c.left = left
    c.top = top
    c.width = width
    c.height = height
    c.crop_x0 = crop_x0
    c.crop_y1 = crop_y1
    c.scale = scale
    c.masks = cython.NULL
    if width <= 0 or height <= 0:
        return False
    bounds: cython.pointer[cython.longlong] = cython.cast(
        cython.pointer[cython.longlong], malloc((3 * count + 1) * cython.sizeof(cython.longlong))
    )
    if bounds == cython.NULL:
        raise MemoryError
    c.masks = cython.cast(cython.p_ushort, calloc(width * height, cython.sizeof(cython.ushort)))
    if c.masks == cython.NULL:
        free(bounds)
        raise MemoryError
    k: cython.Py_ssize_t
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    found: cython.int = 0
    covered: cython.bint = False
    number: cython.uchar
    p = cython.declare(Paint)
    w = cython.declare(Walk)
    w.collect = False
    w.union_edges = cython.NULL
    w.union_count = 0
    w.union_capacity = 0
    w.union_rects = cython.NULL
    w.rect_count = 0
    w.rect_capacity = 0
    w.union_shapes = False
    w.boxes_only = False
    no_calls: typing.Optional[Callbacks] = None
    try:
        for k in range(count):
            start, end, flag = spans[k]
            if start < 0 or end > xs.shape[0] or end < start:
                raise ValueError("span runs past the points")
            bounds[3 * k] = start
            bounds[3 * k + 1] = end
            bounds[3 * k + 2] = 1 if (outline or flag) else 0
        p.scale = scale
        p.exponent = exponent
        p.painted = False
        p.error = 0
        w.xs = cython.address(xs[0]) if xs.shape[0] else cython.NULL
        w.ys = cython.address(ys[0]) if ys.shape[0] else cython.NULL
        w.bounds = bounds
        w.count = count
        w.line_width = line_width
        w.half = pen_half(scale * smallest, line_width)
        w.native = True
        w.line_cap = line_cap
        w.line_join = line_join
        w.miter_limit_squared = miter_limit * miter_limit
        w.cover = cython.address(c)
        with cython.nogil:
            walk(cython.address(p), cython.address(w), no_calls)
            for k in range(0, squares.shape[0], 4):
                found = cover_square(
                    cython.address(p),
                    cython.address(c),
                    squares[k],
                    squares[k + 1],
                    squares[k + 2],
                    squares[k + 3],
                    w.half,
                )
                if found >= 0:
                    found = cover_nearest(
                        cython.address(p), cython.address(c), squares[k], squares[k + 1]
                    )
                if found < 0:
                    break
        if found < 0:
            failed(cython.address(p), no_calls)
        for i in range(height):
            for j in range(width):
                number = sample_count(c.masks[i * width + j])
                counts[i, j] = number
                if number:
                    covered = True
    finally:
        free(c.masks)
        free(bounds)
    return covered


def short_subpaths(
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: list,
    outline: cython.bint,
    scale: cython.double,
) -> bool:
    """Whether any of the subpaths, in page space, is shorter than a device pixel, scale of
    them to a unit, but not of no length: one that may fall between the samples, which
    stroke_coverage keeps from disappearing."""
    k: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    for k in range(len(spans)):
        start, end, flag = spans[k]
        if start < 0 or end > xs.shape[0] or end > ys.shape[0] or end < start:
            raise ValueError("span runs past the points")
        if end - start < 2:
            continue
        if short_span(
            cython.address(xs[0]),
            cython.address(ys[0]),
            start,
            end,
            outline or flag,
            scale,
            cython.NULL,
        ):
            return True
    return False


def axis_aligned_polylines(
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: list,
    outline: cython.bint,
    shortest: cython.double,
) -> bool:
    """Whether every segment of the subpaths, the one closing a closed subpath included,
    runs along an axis or has no length, and none of some length is shorter than shortest,
    measured as in_pieces measures it."""
    k: cython.Py_ssize_t
    index: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    for k in range(len(spans)):
        start, end, flag = spans[k]
        if start < 0 or end > xs.shape[0] or end > ys.shape[0] or end < start:
            raise ValueError("span runs past the points")
        if end - start < 2:
            continue
        for index in range(start, end - 1):
            if xs[index] != xs[index + 1] and ys[index] != ys[index + 1]:
                return False
            if axis_short(xs[index], ys[index], xs[index + 1], ys[index + 1], shortest):
                return False
        if outline or flag:
            if xs[start] != xs[end - 1] and ys[start] != ys[end - 1]:
                return False
            if axis_short(xs[end - 1], ys[end - 1], xs[start], ys[start], shortest):
                return False
    return True
