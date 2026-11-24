# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.libc.math import rint

# UNIT_FROM_BYTE[b] is b / 255.0, looked up instead of divided for every byte.
UNIT_FROM_BYTE = cython.declare(cython.double[256])
for _byte in range(256):
    UNIT_FROM_BYTE[_byte] = cython.cast(cython.double, _byte) / 255.0

# How the element's plate is found from the backdrop's, below (core_pdf.impl.render.raster_spots):
# a paint naming the plate takes its layer's tints, one that does not overprint erases it, one
# that overprints keeps it, and a group takes its own plate.
NAMED = 0
ERASED = 1
KEPT = 2
GROUP = 3

__all__ = ("ERASED", "GROUP", "KEPT", "NAMED", "apply_spot_plate", "composite_spot_plate")


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def clamp_unit(value: cython.double) -> cython.double:
    if not value >= 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
@cython.nogil
def ratio(numerator: cython.double, denominator: cython.double) -> cython.double:
    if denominator <= 0.0:
        return 0.0
    return clamp_unit(numerator / denominator)


@cython.boundscheck(False)
@cython.wraparound(False)
def composite_spot_plate(
    values: cython.uchar[:, :, :],
    below: typing.Optional[cython.uchar[:, :, :]],
    inner: typing.Optional[cython.uchar[:, :, :]],
    previous: typing.Optional[cython.uchar[:, :, :]],
    layer: typing.Optional[cython.uchar[:, :, :]],
    marked: cython.int,
    source: cython.uchar[:, :],
    result: cython.uchar[:, :],
    current: cython.uchar[:, :],
    backdrop_alpha: cython.uchar[:, :],
    element_alpha: cython.uchar[:, :],
    group_alpha: typing.Optional[cython.float[:, :]],
    cover: typing.Optional[cython.float[:, :]],
    mode: cython.int,
) -> None:
    """Composite one plate over a box into values, a byte per component (0 to 255 ink).

    below is the plate of the backdrop the element composites with (None: empty), inner the
    group's own (GROUP), layer the paint's colour, whose components it does not mark (bits of
    marked) hold the complements of its tints (NAMED; one component on a K pass's plate).
    source is the alpha the element composited with, element_alpha the alpha after it
    composited with that backdrop, backdrop_alpha the backdrop's, a byte each. A group's
    share is source / group_alpha where it gives one, else source / element_alpha. With
    cover, the knockout shape, the element replaces previous, the plate before, by it: (Pe ae
    + (1 - shape) (Pc ac - Pb ab)) / ar, current being ac and result ar. Every value is
    clamped to 0 to 1 and rounded to a byte.
    """
    rows: cython.Py_ssize_t = values.shape[0]
    cols: cython.Py_ssize_t = values.shape[1]
    channels: cython.Py_ssize_t = values.shape[2]
    has_below: cython.bint = below is not None
    has_inner: cython.bint = inner is not None
    has_previous: cython.bint = previous is not None
    has_layer: cython.bint = layer is not None
    has_group: cython.bint = group_alpha is not None
    knockout: cython.bint = cover is not None
    if knockout and not has_previous:
        raise ValueError("a knockout plate needs the plate before")
    if mode == NAMED and not has_layer:
        raise ValueError("a paint naming the plate needs its layer")
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    s: cython.double
    ea: cython.double
    ba: cython.double
    share: cython.double
    kept: cython.double
    b: cython.double
    e: cython.double
    t: cython.double
    sh: cython.double
    ar: cython.double
    ca: cython.double
    p: cython.double
    with cython.nogil:
        for i in range(rows):
            for j in range(cols):
                s = UNIT_FROM_BYTE[source[i, j]]
                ea = UNIT_FROM_BYTE[element_alpha[i, j]]
                ba = UNIT_FROM_BYTE[backdrop_alpha[i, j]]
                if mode == 3 and has_group:  # GROUP
                    share = ratio(s, cython.cast(cython.double, group_alpha[i, j]))
                else:
                    share = ratio(s, ea)
                kept = ratio(ba, ea) if ea > 0.0 else 1.0
                sh = 0.0
                if knockout:
                    sh = clamp_unit(cython.cast(cython.double, cover[i, j]))
                    if s > sh:
                        sh = s
                ar = UNIT_FROM_BYTE[result[i, j]]
                ca = UNIT_FROM_BYTE[current[i, j]]
                for k in range(channels):
                    b = UNIT_FROM_BYTE[below[i, j, k]] if has_below else 0.0
                    if mode == 0:  # NAMED
                        if marked & (1 << k):
                            t = 0.0
                        else:
                            t = 1.0 - UNIT_FROM_BYTE[layer[i, j, k]]
                        e = b + share * (t - b)
                    elif mode == 1:  # ERASED
                        e = b * (1.0 - share)
                    elif mode == 2:  # KEPT
                        e = b * kept
                    else:
                        e = b + share * ((UNIT_FROM_BYTE[inner[i, j, k]] if has_inner else 0.0) - b)
                    if knockout:
                        p = UNIT_FROM_BYTE[previous[i, j, k]]
                        if sh > 0.0:
                            if ar > 0.0:
                                e = (e * ea + (1.0 - sh) * (p * ca - b * ba)) / ar
                            else:
                                e = 0.0
                        else:
                            e = p
                    values[i, j, k] = cython.cast(cython.uchar, rint(clamp_unit(e) * 255.0))


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_spot_plate(pixels: cython.uchar[:, :, :], ink: cython.uchar[:, :, :]) -> None:
    """Print a plate's ink, one component (for all three) or three, over pixels, whose first
    three components hold the complements of the process inks: max(c - ink, 0)."""
    rows: cython.Py_ssize_t = pixels.shape[0]
    cols: cython.Py_ssize_t = pixels.shape[1]
    channels: cython.Py_ssize_t = ink.shape[2]
    if ink.shape[0] != rows or ink.shape[1] != cols or (channels != 1 and channels != 3):
        raise ValueError("ink must have pixels' rows and columns, and one or three components")
    single: cython.bint = channels == 1
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    c: cython.int
    with cython.nogil:
        for i in range(rows):
            for j in range(cols):
                for k in range(3):
                    c = pixels[i, j, k] - ink[i, j, 0 if single else k]
                    pixels[i, j, k] = cython.cast(cython.uchar, c if c > 0 else 0)
