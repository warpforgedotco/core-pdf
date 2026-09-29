# SPDX-License-Identifier: AGPL-3.0-only

import cython
import numpy

__all__ = ("alpha_channel", "interleave_soft_mask", "sample_opaque_pixels")


def sample_opaque_pixels(
    target: cython.uchar[:, :, :],
    source: cython.const[cython.uchar][:, :, ::1],
    source_y: cython.const[cython.Py_ssize_t][::1],
    source_x: cython.const[cython.Py_ssize_t][::1],
    valid_rows: cython.const[cython.uchar][::1],
    valid_columns: cython.const[cython.uchar][::1],
    transposed: cython.bint,
):
    rows: cython.Py_ssize_t = target.shape[0]
    columns: cython.Py_ssize_t = target.shape[1]
    channels: cython.Py_ssize_t = source.shape[2]
    if target.shape[2] != 4:
        raise ValueError("target must be RGBA")
    if channels != 1 and channels < 3:
        raise ValueError("source must have one channel or at least three")
    if valid_rows.shape[0] != rows or valid_columns.shape[0] != columns:
        raise ValueError("validity masks differ from the target in size")
    row_count: cython.Py_ssize_t = columns if transposed else rows
    column_count: cython.Py_ssize_t = rows if transposed else columns
    if source_y.shape[0] != row_count or source_x.shape[0] != column_count:
        raise ValueError("sample indices differ from the target in size")
    height: cython.Py_ssize_t = source.shape[0]
    width: cython.Py_ssize_t = source.shape[1]
    i: cython.Py_ssize_t
    for i in range(row_count):
        if not 0 <= source_y[i] < height:
            raise IndexError("sample row out of range")
    for i in range(column_count):
        if not 0 <= source_x[i] < width:
            raise IndexError("sample column out of range")
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    sy: cython.Py_ssize_t
    sx: cython.Py_ssize_t
    sample: cython.p_const_uchar
    pixel: cython.p_uchar
    if target.strides[2] != 1:
        raise ValueError("target channels must be contiguous")
    with cython.nogil:
        for r in range(rows):
            if not valid_rows[r]:
                continue
            for c in range(columns):
                if not valid_columns[c]:
                    continue
                if transposed:
                    sy = source_y[c]
                    sx = source_x[r]
                else:
                    sy = source_y[r]
                    sx = source_x[c]
                sample = cython.address(source[sy, sx, 0])
                pixel = cython.address(target[r, c, 0])
                if channels == 1:
                    pixel[0] = sample[0]
                    pixel[1] = sample[0]
                    pixel[2] = sample[0]
                else:
                    pixel[0] = sample[0]
                    pixel[1] = sample[1]
                    pixel[2] = sample[2]
                pixel[3] = 255


def interleave_soft_mask(
    raster: cython.const[cython.uchar][:, :, :],
    channels: cython.Py_ssize_t,
    mask: cython.const[cython.uchar][:, :],
    mask_rows: cython.const[cython.Py_ssize_t][::1],
    mask_columns: cython.const[cython.Py_ssize_t][::1],
):
    rows: cython.Py_ssize_t = raster.shape[0]
    cols: cython.Py_ssize_t = raster.shape[1]
    if channels < 0 or channels > raster.shape[2]:
        raise ValueError("channels exceeds the raster's")
    if mask_rows.shape[0] != rows or mask_columns.shape[0] != cols:
        raise ValueError("mask indices differ from the raster in size")
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    for i in range(rows):
        if not 0 <= mask_rows[i] < mask.shape[0]:
            raise IndexError("mask row out of range")
    for j in range(cols):
        if not 0 <= mask_columns[j] < mask.shape[1]:
            raise IndexError("mask column out of range")
    output = numpy.empty((rows, cols, channels + 1), dtype=numpy.uint8)
    out: cython.uchar[:, :, ::1] = output
    row: cython.Py_ssize_t
    with cython.nogil:
        for i in range(rows):
            row = mask_rows[i]
            for j in range(cols):
                for k in range(channels):
                    out[i, j, k] = raster[i, j, k]
                out[i, j, channels] = mask[row, mask_columns[j]]
    return output


def alpha_channel(pixels: cython.const[cython.uchar][:, :, ::1], presence: cython.bint):
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    height: cython.Py_ssize_t = pixels.shape[0]
    width: cython.Py_ssize_t = pixels.shape[1]
    alpha = numpy.empty((height, width), dtype=numpy.uint8)
    out: cython.uchar[:, ::1] = alpha
    present = numpy.zeros(256, dtype=numpy.bool_) if presence else None
    marks: cython.uchar[::1]
    seen = cython.declare(cython.uchar[256])
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    k: cython.Py_ssize_t
    value: cython.uchar
    for k in range(256):
        seen[k] = 0
    with cython.nogil:
        for y in range(height):
            for x in range(width):
                value = pixels[y, x, 3]
                out[y, x] = value
                seen[value] = 1
    if presence:
        marks = present.view(numpy.uint8)
        for k in range(256):
            marks[k] = seen[k]
    return alpha, present
