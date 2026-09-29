# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports.core_pdf_cythonized._pixel_blend import (
    MODE_NORMAL,
    blend_mode_pixel,
    blend_normal_pixel,
    plane_accumulate,
)
from cython.cimports.libc.math import fabs, isfinite, pow

__all__ = ("shading_blend", "shading_t", "shading_values")


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def axial_t(
    c: cython.p_const_double,
    px: cython.double,
    py: cython.double,
    t: cython.p_double,
) -> cython.bint:
    dx: cython.double = c[2] - c[0]
    dy: cython.double = c[3] - c[1]
    denom: cython.double = dx * dx + dy * dy
    if denom <= 1e-12:
        return False
    t[0] = ((px - c[0]) * dx + (py - c[1]) * dy) / denom
    return True


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def radial_t(
    c: cython.p_const_double,
    px: cython.double,
    py: cython.double,
    half: cython.double,
    t: cython.p_double,
) -> cython.bint:
    dx: cython.double = c[3] - c[0]
    dy: cython.double = c[4] - c[1]
    dr: cython.double = c[5] - c[2]
    qx: cython.double = px - c[0]
    qy: cython.double = py - c[1]
    a: cython.double = dx * dx + dy * dy - dr * dr
    b: cython.double = -2.0 * (qx * dx + qy * dy + c[2] * dr)
    cc: cython.double = qx * qx + qy * qy - c[2] * c[2]
    disc: cython.double
    root: cython.double
    t0: cython.double
    t1: cython.double
    valid0: cython.bint
    valid1: cython.bint
    in0: cython.bint
    in1: cython.bint
    if fabs(a) <= 1e-12:
        if fabs(b) <= 1e-12:
            return False
        t[0] = -cc / b
        return True
    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        return False
    root = pow(disc, half)
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    valid0 = isfinite(t0)
    valid1 = isfinite(t1)
    if not valid0 and not valid1:
        return False
    in0 = valid0 and 0.0 <= t0 <= 1.0
    in1 = valid1 and 0.0 <= t1 <= 1.0
    if in0 or in1:
        if in0 and in1:
            t[0] = t1 if t1 > t0 else t0
        else:
            t[0] = t0 if in0 else t1
        return True
    if valid0 and valid1:
        t[0] = t1 if fabs(t1 - 0.5) < fabs(t0 - 0.5) else t0
    else:
        t[0] = t0 if valid0 else t1
    return True


def shading_t(kind: cython.int, coords, px: cython.double, py: cython.double, half: cython.double):
    c = cython.declare(cython.double[6])
    t = cython.declare(cython.double)
    i: cython.Py_ssize_t
    need: cython.Py_ssize_t = 4 if kind == 2 else 6
    if len(coords) < need:
        raise ValueError("too few shading coordinates")
    for i in range(need):
        c[i] = coords[i]
    found = (
        axial_t(c, px, py, cython.address(t))
        if kind == 2
        else radial_t(c, px, py, half, cython.address(t))
    )
    return t if found else None


def shading_values(
    kind: cython.int,
    coords,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    allowed: cython.const[cython.uchar][:, ::1],
    extend0: cython.bint,
    extend1: cython.bint,
    domain0: cython.double,
    domain_span: cython.double,
    half: cython.double,
):
    height: cython.Py_ssize_t = iy1 - iy0
    width: cython.Py_ssize_t = ix1 - ix0
    if allowed.shape[0] != height or allowed.shape[1] != width:
        raise ValueError("allowed differs from the box in shape")
    c = cython.declare(cython.double[6])
    need: cython.Py_ssize_t = 4 if kind == 2 else 6
    if len(coords) < need:
        raise ValueError("too few shading coordinates")
    i: cython.Py_ssize_t
    for i in range(need):
        c[i] = coords[i]
    values_array = numpy.zeros((height, width), dtype=numpy.float64)
    painted_array = numpy.zeros((height, width), dtype=numpy.uint8)
    values: cython.double[:, ::1] = values_array
    painted: cython.uchar[:, ::1] = painted_array
    r: cython.Py_ssize_t
    col: cython.Py_ssize_t
    page_x: cython.double
    page_y: cython.double
    t = cython.declare(cython.double)
    found: cython.bint
    with cython.nogil:
        for r in range(height):
            page_y = crop_y1 - (cython.cast(cython.double, iy0 + r) + 0.5) / scale
            for col in range(width):
                if not allowed[r, col]:
                    continue
                page_x = crop_x0 + (cython.cast(cython.double, ix0 + col) + 0.5) / scale
                if kind == 2:
                    found = axial_t(c, page_x, page_y, cython.address(t))
                else:
                    found = radial_t(c, page_x, page_y, half, cython.address(t))
                if not found:
                    continue
                if t < 0.0:
                    if not extend0:
                        continue
                    t = 0.0
                elif t > 1.0:
                    if not extend1:
                        continue
                    t = 1.0
                values[r, col] = domain0 + t * domain_span
                painted[r, col] = 1
    return values_array, painted_array


def shading_blend(
    pixels: cython.uchar[:, :, ::1],
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    painted: cython.const[cython.uchar][:, ::1],
    color_index: cython.const[cython.longlong][::1],
    colors: cython.const[cython.int][:, ::1],
    mode: cython.int,
    revised: cython.bint,
    alpha_plane: typing.Optional[cython.float[:, :]],
    shape_plane: typing.Optional[cython.float[:, :]],
    shape_source: cython.double,
    stop_at_visible: cython.bint,
):
    height: cython.Py_ssize_t = painted.shape[0]
    width: cython.Py_ssize_t = painted.shape[1]
    if iy0 < 0 or ix0 < 0 or iy0 + height > pixels.shape[0] or ix0 + width > pixels.shape[1]:
        raise ValueError("box runs past the pixels")
    if colors.shape[1] != 4:
        raise ValueError("colors must be (n, 4)")
    has_alpha: cython.bint = alpha_plane is not None
    has_shape: cython.bint = shape_plane is not None
    if has_alpha and (
        alpha_plane.shape[0] != pixels.shape[0] or alpha_plane.shape[1] != pixels.shape[1]
    ):
        raise ValueError("alpha_plane differs from the pixels in shape")
    if has_shape and (
        shape_plane.shape[0] != pixels.shape[0] or shape_plane.shape[1] != pixels.shape[1]
    ):
        raise ValueError("shape_plane differs from the pixels in shape")
    all_extend: cython.bint = has_shape or not has_alpha
    r: cython.Py_ssize_t
    col: cython.Py_ssize_t
    k: cython.Py_ssize_t = 0
    count: cython.Py_ssize_t = color_index.shape[0]
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    low_x: cython.Py_ssize_t = width
    low_y: cython.Py_ssize_t = height
    high_x: cython.Py_ssize_t = -1
    high_y: cython.Py_ssize_t = -1
    which: cython.longlong
    palette: cython.Py_ssize_t = colors.shape[0]
    sa: cython.int
    stopped: cython.bint = False
    with cython.nogil:
        for r in range(height):
            if k >= count or stopped:
                break
            for col in range(width):
                if not painted[r, col]:
                    continue
                if k >= count:
                    break
                which = color_index[k]
                k += 1
                if which < 0 or which >= palette:
                    with cython.gil:
                        raise ValueError("colour index out of range")
                sa = colors[which, 3]
                y = iy0 + r
                x = ix0 + col
                if has_shape:
                    shape_plane[y, x] = plane_accumulate(shape_plane[y, x], shape_source)
                if all_extend or sa > 0:
                    if col < low_x:
                        low_x = col
                    if col > high_x:
                        high_x = col
                    if r < low_y:
                        low_y = r
                    if r > high_y:
                        high_y = r
                if sa <= 0:
                    continue
                if has_alpha:
                    alpha_plane[y, x] = plane_accumulate(alpha_plane[y, x], sa / 255.0)
                if stop_at_visible:
                    stopped = True
                    break
                if mode == MODE_NORMAL:
                    blend_normal_pixel(
                        cython.address(pixels[y, x, 0]),
                        colors[which, 0],
                        colors[which, 1],
                        colors[which, 2],
                        sa,
                    )
                else:
                    blend_mode_pixel(
                        cython.address(pixels[y, x, 0]),
                        colors[which, 0],
                        colors[which, 1],
                        colors[which, 2],
                        sa,
                        mode,
                        revised,
                    )
    window = None if high_x < 0 else (ix0 + low_x, iy0 + low_y, ix0 + high_x + 1, iy0 + high_y + 1)
    return window, stopped
