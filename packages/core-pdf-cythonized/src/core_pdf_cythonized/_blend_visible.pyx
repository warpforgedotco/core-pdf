# SPDX-License-Identifier: AGPL-3.0-only

from libc.math cimport rint

import numpy


cdef enum:
    NORMAL = 0
    MULTIPLY = 1
    SCREEN = 2


cdef inline unsigned char channel_byte(double value) noexcept nogil:
    if value < 0.0:
        value = 0.0
    elif value > 255.0:
        value = 255.0
    return <unsigned char>value


cdef inline double blended(double source, double backdrop, double dst_a, int mode) noexcept nogil:
    if mode == MULTIPLY:
        return source * (1.0 - dst_a) + dst_a * (source * (backdrop / 255.0))
    if mode == SCREEN:
        return source * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - source) * (1.0 - backdrop / 255.0))
    return source


def blend_visible_rgba(
    unsigned char[:, :, :] destination,
    const unsigned char[:, :] visible,
    red,
    green,
    blue,
    const double[:] alpha,
    int mode,
):
    """Composite one source colour per visible pixel over an RGBA destination.

    red, green and blue are each a float or one value per visible pixel, in the
    row-major order of the mask; alpha is one value per visible pixel. mode is
    0 (normal), 1 (multiply) or 2 (screen). The arithmetic is float64 in the
    order render_blend.blend_channels_f64 computes it.
    """
    cdef Py_ssize_t height = visible.shape[0], width = visible.shape[1]
    if destination.shape[0] != height or destination.shape[1] != width or destination.shape[2] < 4:
        raise ValueError("destination does not match the visibility mask")
    if mode < NORMAL or mode > SCREEN:
        raise ValueError("unsupported blend mode")
    cdef const double[:] red_values
    cdef const double[:] green_values
    cdef const double[:] blue_values
    cdef bint red_array = isinstance(red, numpy.ndarray)
    cdef bint green_array = isinstance(green, numpy.ndarray)
    cdef bint blue_array = isinstance(blue, numpy.ndarray)
    cdef double red_constant = 0.0, green_constant = 0.0, blue_constant = 0.0
    if red_array:
        red_values = red
    else:
        red_constant = red
    if green_array:
        green_values = green
    else:
        green_constant = green
    if blue_array:
        blue_values = blue
    else:
        blue_constant = blue
    cdef Py_ssize_t count = 0, y, x, index = 0
    for y in range(height):
        for x in range(width):
            if visible[y, x]:
                count += 1
    if alpha.shape[0] != count or (
        (red_array and red_values.shape[0] != count)
        or (green_array and green_values.shape[0] != count)
        or (blue_array and blue_values.shape[0] != count)
    ):
        raise ValueError("colour and alpha values do not match the visible pixels")
    cdef double src_r, src_g, src_b, src_a, one_minus_src_a
    cdef double dr, dg, db, da, dst_a, out_a, safe_out_a
    with nogil:
        for y in range(height):
            for x in range(width):
                if not visible[y, x]:
                    continue
                src_r = red_values[index] if red_array else red_constant
                src_g = green_values[index] if green_array else green_constant
                src_b = blue_values[index] if blue_array else blue_constant
                src_a = alpha[index]
                index += 1
                dr = destination[y, x, 0]
                dg = destination[y, x, 1]
                db = destination[y, x, 2]
                da = destination[y, x, 3]
                one_minus_src_a = 1.0 - src_a
                dst_a = da / 255.0
                src_r = blended(src_r, dr, dst_a, mode)
                src_g = blended(src_g, dg, dst_a, mode)
                src_b = blended(src_b, db, dst_a, mode)
                out_a = src_a + dst_a * one_minus_src_a
                if out_a <= 0.0:
                    destination[y, x, 0] = 0
                    destination[y, x, 1] = 0
                    destination[y, x, 2] = 0
                    destination[y, x, 3] = 0
                    continue
                safe_out_a = out_a if out_a > 0.0 else 1.0
                destination[y, x, 0] = channel_byte(
                    rint(((src_r * 255.0) * src_a + dr * dst_a * one_minus_src_a) / safe_out_a)
                )
                destination[y, x, 1] = channel_byte(
                    rint(((src_g * 255.0) * src_a + dg * dst_a * one_minus_src_a) / safe_out_a)
                )
                destination[y, x, 2] = channel_byte(
                    rint(((src_b * 255.0) * src_a + db * dst_a * one_minus_src_a) / safe_out_a)
                )
                destination[y, x, 3] = channel_byte(rint(out_a * 255.0))
