# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._byte_clamp import round_to_byte
from cython.cimports.core_pdf_cythonized._pixel_blend import (
    MODE_NORMAL,
    blend_mode_pixel,
    blend_normal_pixel,
    coverage_alpha,
    plane_accumulate,
)

__all__ = ("blend_coverage_counts",)


def blend_coverage_counts(
    pixels: cython.uchar[:, :, ::1],
    counts: cython.const[cython.uchar][:, ::1],
    left: cython.Py_ssize_t,
    top: cython.Py_ssize_t,
    allowed: typing.Optional[cython.const[cython.uchar][::1]],
    red: cython.int,
    green: cython.int,
    blue: cython.int,
    alpha: cython.int,
    source_alpha: typing.Optional[cython.float[:, :]],
    source_shape: typing.Optional[cython.float[:, :]],
    track_shape: cython.bint,
    shape_alpha: cython.double,
    mode: cython.int = MODE_NORMAL,
    revised: cython.bint = True,
    stop_at_visible: cython.bint = False,
):
    rows: cython.Py_ssize_t = counts.shape[0]
    cols: cython.Py_ssize_t = counts.shape[1]
    if left < 0 or top < 0 or top + rows > pixels.shape[0] or left + cols > pixels.shape[1]:
        raise ValueError("counts run past the pixels given")
    masked: cython.bint = allowed is not None
    if masked and allowed.shape[0] != rows * cols:
        raise ValueError("allowed must hold one byte per count")
    has_alpha: cython.bint = source_alpha is not None
    has_shape: cython.bint = source_shape is not None
    if has_alpha and (
        source_alpha.shape[0] != pixels.shape[0] or source_alpha.shape[1] != pixels.shape[1]
    ):
        raise ValueError("source_alpha differs from the pixels in shape")
    if has_shape and (
        source_shape.shape[0] != pixels.shape[0] or source_shape.shape[1] != pixels.shape[1]
    ):
        raise ValueError("source_shape differs from the pixels in shape")
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    covered: cython.int
    sa: cython.int
    shape: cython.int
    low_x: cython.Py_ssize_t = left + cols
    low_y: cython.Py_ssize_t = top + rows
    high_x: cython.Py_ssize_t = left
    high_y: cython.Py_ssize_t = top
    stopped: cython.bint = False
    with cython.nogil:
        for r in range(rows):
            if stopped:
                break
            y = top + r
            for c in range(cols):
                covered = counts[r, c]
                if not covered:
                    continue
                if masked and not allowed[r * cols + c]:
                    continue
                x = left + c
                sa = coverage_alpha(alpha, covered)
                if has_shape:
                    shape = (
                        round_to_byte(cython.cast(cython.double, 255 * covered) / 16.0)
                        if track_shape
                        else 255
                    )
                    source_shape[y, x] = plane_accumulate(
                        source_shape[y, x], (shape / 255.0) * shape_alpha
                    )
                if has_shape or not has_alpha or sa > 0:
                    if x < low_x:
                        low_x = x
                    if x + 1 > high_x:
                        high_x = x + 1
                    if y < low_y:
                        low_y = y
                    if y + 1 > high_y:
                        high_y = y + 1
                if sa <= 0:
                    continue
                if has_alpha:
                    source_alpha[y, x] = plane_accumulate(source_alpha[y, x], sa / 255.0)
                if stop_at_visible:
                    stopped = True
                    break
                if mode == MODE_NORMAL:
                    blend_normal_pixel(cython.address(pixels[y, x, 0]), red, green, blue, sa)
                else:
                    blend_mode_pixel(
                        cython.address(pixels[y, x, 0]), red, green, blue, sa, mode, revised
                    )
    if high_x <= low_x:
        return None, stopped
    return (low_x, low_y, high_x, high_y), stopped
