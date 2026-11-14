# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports.libc.math import rint

__all__ = ("SPACE_CMYK", "SPACE_GRAY", "SPACE_RGB", "luminosity_channel")

# The space codes: what a mask raster's colour components hold
# (core_pdf.impl.render.raster_spaces): sRGB, a gray level in all three, or a CMYK group's
# two passes, the complements of C, M and Y and the complement of K.
SPACE_RGB = 0
SPACE_GRAY = 1
SPACE_CMYK = 2


@cython.cfunc
@cython.inline
@cython.nogil
def luminosity_byte(value: cython.double) -> cython.uchar:
    scaled: cython.double = rint(value * 255.0)
    if scaled <= 0.0:
        return 0
    if scaled >= 255.0:
        return 255
    return cython.cast(cython.uchar, scaled)


def luminosity_channel(
    pixels: cython.const[cython.uchar][:, :, ::1],
    black: typing.Optional[cython.const[cython.uchar][:, :, ::1]],
    space: cython.int,
    presence: cython.bint,
):
    """Each pixel's luminosity as a byte (ISO 32000-2 11.5.3), as
    core_pdf.impl.spec.s_11_transparency.soft_masks.device_luminosity computes it on the
    components the bytes stand for, and which byte values occur."""
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    if space < 0 or space > 2:
        raise ValueError("unknown luminosity space")
    height: cython.Py_ssize_t = pixels.shape[0]
    width: cython.Py_ssize_t = pixels.shape[1]
    if space == 2:
        if black is None:
            raise ValueError("a CMYK raster needs its black pass")
        if black.shape[0] != height or black.shape[1] != width or black.shape[2] != 4:
            raise ValueError("the black pass must match the raster")
    luminosity = numpy.empty((height, width), dtype=numpy.uint8)
    out: cython.uchar[:, ::1] = luminosity
    present = numpy.zeros(256, dtype=numpy.bool_) if presence else None
    marks: cython.uchar[::1]
    seen = cython.declare(cython.uchar[256])
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    k: cython.Py_ssize_t
    value: cython.uchar
    cyan: cython.double
    magenta: cython.double
    yellow: cython.double
    key: cython.double
    for k in range(256):
        seen[k] = 0
    with cython.nogil:
        for y in range(height):
            for x in range(width):
                if space == 1:
                    value = pixels[y, x, 0]
                elif space == 0:
                    value = luminosity_byte(
                        0.3 * (pixels[y, x, 0] / 255.0)
                        + 0.59 * (pixels[y, x, 1] / 255.0)
                        + 0.11 * (pixels[y, x, 2] / 255.0)
                    )
                else:
                    cyan = (255 - pixels[y, x, 0]) / 255.0
                    magenta = (255 - pixels[y, x, 1]) / 255.0
                    yellow = (255 - pixels[y, x, 2]) / 255.0
                    key = (
                        0.3 * cyan + 0.59 * magenta + 0.11 * yellow + (255 - black[y, x, 0]) / 255.0
                    )
                    value = luminosity_byte(1.0 - (1.0 if key > 1.0 else key))
                out[y, x] = value
                seen[value] = 1
    if presence:
        marks = present.view(numpy.uint8)
        for k in range(256):
            marks[k] = seen[k]
    return luminosity, present
