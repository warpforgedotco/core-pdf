# SPDX-License-Identifier: AGPL-3.0-only

import cython
import numpy
from cython.cimports.core_pdf_cythonized._pixel_blend import MODE_COUNT, MODE_NORMAL, blend_rgb
from cython.cimports.libc.math import rint


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def channel_byte(value: cython.double) -> cython.uchar:
    if value < 0.0:
        value = 0.0
    elif value > 255.0:
        value = 255.0
    return cython.cast(cython.uchar, value)


def blend_visible_rgba(
    destination: cython.uchar[:, :, :],
    visible: cython.const[cython.uchar][:, :],
    red,
    green,
    blue,
    alpha: cython.const[cython.double][:],
    mode: cython.int,
    revised: cython.bint = True,
):
    """Composite one source colour per visible pixel over an RGBA destination.

    red, green and blue are each a float or one value per visible pixel, in the
    row-major order of the mask; alpha is one value per visible pixel. mode is a
    blend code, 0 to 17 (core_pdf.impl.render.blend.BlendOp); revised selects the
    PDF 2.0 dodge and burn endpoints. The arithmetic is float64 in the order
    core_pdf.impl.render.blend.blend_channels_f64 computes it.
    """
    height: cython.Py_ssize_t = visible.shape[0]
    width: cython.Py_ssize_t = visible.shape[1]
    if destination.shape[0] != height or destination.shape[1] != width or destination.shape[2] < 4:
        raise ValueError("destination does not match the visibility mask")
    if mode < MODE_NORMAL or mode >= MODE_COUNT:
        raise ValueError("unsupported blend mode")
    red_values: cython.const[cython.double][:]
    green_values: cython.const[cython.double][:]
    blue_values: cython.const[cython.double][:]
    red_array: cython.bint = isinstance(red, numpy.ndarray)
    green_array: cython.bint = isinstance(green, numpy.ndarray)
    blue_array: cython.bint = isinstance(blue, numpy.ndarray)
    red_constant: cython.double = 0.0
    green_constant: cython.double = 0.0
    blue_constant: cython.double = 0.0
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
    count: cython.Py_ssize_t = 0
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    index: cython.Py_ssize_t = 0
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
    src_r: cython.double
    src_g: cython.double
    src_b: cython.double
    src_a: cython.double
    one_minus_src_a: cython.double
    dr: cython.double
    dg: cython.double
    db: cython.double
    da: cython.double
    dst_a: cython.double
    out_a: cython.double
    safe_out_a: cython.double
    backdrop = cython.declare(cython.double[3])
    source = cython.declare(cython.double[3])
    blended = cython.declare(cython.double[3])
    with cython.nogil:
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
                if mode != MODE_NORMAL:
                    backdrop[0] = dr / 255.0
                    backdrop[1] = dg / 255.0
                    backdrop[2] = db / 255.0
                    source[0] = src_r
                    source[1] = src_g
                    source[2] = src_b
                    blend_rgb(backdrop, source, blended, mode, revised)
                    src_r = src_r * (1.0 - dst_a) + dst_a * blended[0]
                    src_g = src_g * (1.0 - dst_a) + dst_a * blended[1]
                    src_b = src_b * (1.0 - dst_a) + dst_a * blended[2]
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
