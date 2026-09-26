# SPDX-License-Identifier: AGPL-3.0-only
"""Axial and radial shading fills (core_pdf.impl.render.target.paint_shading).

paint_shading walked the shading's pixel box in Python: for each pixel the
clip allowed, axial_shading_t or radial_shading_t for its centre, the extend
rules, the domain value, a colour from a dictionary memo, and blend_px. A
page-filling gradient is a few hundred thousand pixels at about 5 us each.

shading_values is the per-pixel parameter, and shading_blend the blend and
the group plane records, as blend_px makes them for every blend mode it
knows; the colour of each distinct value stays in Python, where the
shading's function is. Every
step is the Python's, in its order and in double: the pixel centres, the
axial projection, the radial quadratic -- its square root taken with pow,
as ``disc**0.5`` is -- with the exponent passed in, not written, so the
compiler cannot turn it into sqrt, which differs by an ulp now and then --
its roots filtered for finiteness, the largest root in
[0, 1] or else the root nearest 0.5, the first of two equal ones -- the
extend clamps, ``domain0 + t * span``. setup.py builds with
-ffp-contract=off.
"""

from libc.math cimport fabs, isfinite, pow

import numpy

from core_pdf_cythonized._pixel_blend cimport blend_normal_pixel, clamp_byte, plane_accumulate

__all__ = ("shading_blend", "shading_t", "shading_values")

# The blend modes blend_px treats apart; any other composites as normal.
cdef enum:
    MODE_NORMAL = 0
    MODE_MULTIPLY = 1
    MODE_SCREEN = 2
    MODE_COLOR_DODGE = 3
    MODE_COLOR_BURN = 4


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
        # max(in_range): the first of the largest.
        if in0 and in1:
            t[0] = t1 if t1 > t0 else t0
        else:
            t[0] = t0 if in0 else t1
        return True
    # min(valid, key=abs(t - 0.5)): the first of the nearest.
    if valid0 and valid1:
        t[0] = t1 if fabs(t1 - 0.5) < fabs(t0 - 0.5) else t0
    else:
        t[0] = t0 if valid0 else t1
    return True


def shading_t(int kind, coords, double px, double py, double half):
    """The shading parameter at one point before the extend rules, or None.

    ``kind`` is 2 for axial and 3 for radial; ``half`` is 0.5, as for
    shading_values.
    """
    cdef double c[6]
    cdef double t
    cdef Py_ssize_t i, need = 4 if kind == 2 else 6
    if len(coords) < need:
        raise ValueError("too few shading coordinates")
    for i in range(need):
        c[i] = coords[i]
    found = axial_t(c, px, py, &t) if kind == 2 else radial_t(c, px, py, half, &t)
    return t if found else None


cdef inline double color_dodge(double backdrop, double source, bint revised) noexcept nogil:
    # blend_component's ColorDodge.
    if revised and backdrop == 0.0:
        return 0.0
    if backdrop >= 1.0 - source:
        return 1.0
    return backdrop / (1.0 - source)


cdef inline double color_burn(double backdrop, double source, bint revised) noexcept nogil:
    # blend_component's ColorBurn.
    if revised and backdrop == 1.0:
        return 1.0
    if 1.0 - backdrop >= source:
        return 0.0
    return 1.0 - (1.0 - backdrop) / source


cdef inline void blend_mode_pixel(
    unsigned char* pixel, int red, int green, int blue, int sa, int mode, bint revised
) noexcept nogil:
    # blend_px for multiply, screen, color dodge and color burn, in double as
    # Python computes it: the source adjusted against the backdrop, then the
    # general compositing, with no opaque shortcut.
    cdef double src_a = sa / 255.0
    cdef double dst_a = pixel[3] / 255.0
    cdef double source[3]
    cdef double backdrop
    cdef int k
    cdef int destination[3]
    source[0] = red / 255.0
    source[1] = green / 255.0
    source[2] = blue / 255.0
    for k in range(3):
        destination[k] = pixel[k]
        backdrop = destination[k] / 255.0
        if mode == MODE_MULTIPLY:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * (source[k] * backdrop)
        elif mode == MODE_SCREEN:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * (
                1.0 - (1.0 - source[k]) * (1.0 - backdrop)
            )
        elif mode == MODE_COLOR_DODGE:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * color_dodge(backdrop, source[k], revised)
        else:
            source[k] = source[k] * (1.0 - dst_a) + dst_a * color_burn(backdrop, source[k], revised)
    cdef double out_a = src_a + dst_a * (1.0 - src_a)
    for k in range(3):
        pixel[k] = <unsigned char> clamp_byte(
            ((source[k] * 255.0) * src_a + destination[k] * dst_a * (1.0 - src_a)) / out_a
        )
    pixel[3] = <unsigned char> clamp_byte(out_a * 255.0)


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
    """Each allowed pixel's shading value, where the shading paints it.

    ``kind`` is 2 for axial, 3 for radial; ``allowed`` holds one byte per
    pixel of the box [iy0, iy1) x [ix0, ix1), non-zero where the clip lets
    it through. ``half`` is 0.5, the exponent of the radial square root:
    ``disc**0.5`` is libm's pow, which differs from sqrt by an ulp now and
    then, and a compiler that saw a constant 0.5 would call sqrt instead.
    Returns (values, painted): float64 values and a uint8 mask of the pixels
    painted, both the box's shape.
    """
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
    """Blend the first len(color_index) painted pixels of the box, as blend_px does.

    ``color_index`` holds, in row-major order of the painted pixels, the row
    of ``colors`` (r, g, b, a) each is painted with; it may stop short of
    the last. ``mode`` is 0 for any blend blend_px composites as normal, 1
    multiply, 2 screen, 3 color dodge, 4 color burn; ``revised`` is
    blend_component's revised-blending flag. Each pixel is first recorded,
    as blend_px records it, into ``shape_plane`` (with ``shape_source``) and,
    when its alpha is not zero, into ``alpha_plane``. With
    ``stop_at_visible``, the first pixel of non-zero alpha is recorded and
    then not blended, and the scan stops there: blend_px raises at that
    point when it cannot tell which blending rules apply.

    Returns (window, stopped): (x0, y0, x1, y1), half-open, of the pixels
    that extend the paint window -- every one without an alpha plane or with
    a shape plane, as blend_px extends it before it looks at alpha; only
    those of non-zero alpha with just an alpha plane -- or None; and whether
    the scan stopped at a visible pixel.
    """
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
