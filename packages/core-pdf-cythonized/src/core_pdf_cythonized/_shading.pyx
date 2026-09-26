# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport fabs, isfinite, pow

import numpy

from core_pdf_cythonized._pixel_blend cimport (
    MODE_NORMAL,
    blend_mode_pixel,
    blend_normal_pixel,
    plane_accumulate,
)

__all__ = ("shading_blend", "shading_t", "shading_values")

cdef inline bint axial_t(const double* c, double px, double py, double* t) noexcept nogil:
    cdef double dx = c[2] - c[0]
    cdef double dy = c[3] - c[1]
    cdef double denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return False
    t[0] = ((px - c[0]) * dx + (py - c[1]) * dy) / denom
    return True


cdef inline bint radial_t(
    const double* c, double px, double py, double half, double* t
) noexcept nogil:
    cdef double dx = c[3] - c[0]
    cdef double dy = c[4] - c[1]
    cdef double dr = c[5] - c[2]
    cdef double qx = px - c[0]
    cdef double qy = py - c[1]
    cdef double a = dx * dx + dy * dy - dr * dr
    cdef double b = -2.0 * (qx * dx + qy * dy + c[2] * dr)
    cdef double cc = qx * qx + qy * qy - c[2] * c[2]
    cdef double disc, root, t0, t1, best
    cdef bint valid0, valid1, in0, in1
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


def shading_t(int kind, coords, double px, double py, double half):
    cdef double c[6]
    cdef double t
    cdef Py_ssize_t i, need = 4 if kind == 2 else 6
    if len(coords) < need:
        raise ValueError("too few shading coordinates")
    for i in range(need):
        c[i] = coords[i]
    found = axial_t(c, px, py, &t) if kind == 2 else radial_t(c, px, py, half, &t)
    return t if found else None


def shading_values(
    int kind,
    coords,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    Py_ssize_t ix1,
    Py_ssize_t iy1,
    const unsigned char[:, ::1] allowed,
    bint extend0,
    bint extend1,
    double domain0,
    double domain_span,
    double half,
):
    cdef Py_ssize_t height = iy1 - iy0, width = ix1 - ix0
    if allowed.shape[0] != height or allowed.shape[1] != width:
        raise ValueError("allowed differs from the box in shape")
    cdef double c[6]
    cdef Py_ssize_t need = 4 if kind == 2 else 6
    if len(coords) < need:
        raise ValueError("too few shading coordinates")
    cdef Py_ssize_t i
    for i in range(need):
        c[i] = coords[i]
    values_array = numpy.zeros((height, width), dtype=numpy.float64)
    painted_array = numpy.zeros((height, width), dtype=numpy.uint8)
    cdef double[:, ::1] values = values_array
    cdef unsigned char[:, ::1] painted = painted_array
    cdef Py_ssize_t r, col
    cdef double page_x, page_y, t
    cdef bint found
    with nogil:
        for r in range(height):
            page_y = crop_y1 - (<double> (iy0 + r) + 0.5) / scale
            for col in range(width):
                if not allowed[r, col]:
                    continue
                page_x = crop_x0 + (<double> (ix0 + col) + 0.5) / scale
                if kind == 2:
                    found = axial_t(c, page_x, page_y, &t)
                else:
                    found = radial_t(c, page_x, page_y, half, &t)
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
    unsigned char[:, :, ::1] pixels,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    const unsigned char[:, ::1] painted,
    const long long[::1] color_index,
    const int[:, ::1] colors,
    int mode,
    bint revised,
    float[:, :] alpha_plane,
    float[:, :] shape_plane,
    double shape_source,
    bint stop_at_visible,
):
    cdef Py_ssize_t height = painted.shape[0], width = painted.shape[1]
    if iy0 < 0 or ix0 < 0 or iy0 + height > pixels.shape[0] or ix0 + width > pixels.shape[1]:
        raise ValueError("box runs past the pixels")
    if colors.shape[1] != 4:
        raise ValueError("colors must be (n, 4)")
    cdef bint has_alpha = alpha_plane is not None
    cdef bint has_shape = shape_plane is not None
    if has_alpha and (
        alpha_plane.shape[0] != pixels.shape[0] or alpha_plane.shape[1] != pixels.shape[1]
    ):
        raise ValueError("alpha_plane differs from the pixels in shape")
    if has_shape and (
        shape_plane.shape[0] != pixels.shape[0] or shape_plane.shape[1] != pixels.shape[1]
    ):
        raise ValueError("shape_plane differs from the pixels in shape")
    cdef bint all_extend = has_shape or not has_alpha
    cdef Py_ssize_t r, col, k = 0, count = color_index.shape[0], y, x
    cdef Py_ssize_t low_x = width, low_y = height, high_x = -1, high_y = -1
    cdef long long which
    cdef Py_ssize_t palette = colors.shape[0]
    cdef int sa
    cdef bint stopped = False
    with nogil:
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
                    with gil:
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
                        &pixels[y, x, 0], colors[which, 0], colors[which, 1], colors[which, 2], sa
                    )
                else:
                    blend_mode_pixel(
                        &pixels[y, x, 0],
                        colors[which, 0],
                        colors[which, 1],
                        colors[which, 2],
                        sa,
                        mode,
                        revised,
                    )
    window = None if high_x < 0 else (ix0 + low_x, iy0 + low_y, ix0 + high_x + 1, iy0 + high_y + 1)
    return window, stopped
