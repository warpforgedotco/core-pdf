# SPDX-License-Identifier: AGPL-3.0-only

import numpy

__all__ = ("alpha_channel", "interleave_soft_mask", "sample_opaque_pixels")


def sample_opaque_pixels(
    unsigned char[:, :, :] target,
    const unsigned char[:, :, ::1] source,
    const Py_ssize_t[::1] source_y,
    const Py_ssize_t[::1] source_x,
    const unsigned char[::1] valid_rows,
    const unsigned char[::1] valid_columns,
    bint transposed,
):
    cdef Py_ssize_t rows = target.shape[0]
    cdef Py_ssize_t columns = target.shape[1]
    cdef Py_ssize_t channels = source.shape[2]
    if target.shape[2] != 4:
        raise ValueError("target must be RGBA")
    if channels != 1 and channels < 3:
        raise ValueError("source must have one channel or at least three")
    if valid_rows.shape[0] != rows or valid_columns.shape[0] != columns:
        raise ValueError("validity masks differ from the target in size")
    cdef Py_ssize_t row_count = columns if transposed else rows
    cdef Py_ssize_t column_count = rows if transposed else columns
    if source_y.shape[0] != row_count or source_x.shape[0] != column_count:
        raise ValueError("sample indices differ from the target in size")
    cdef Py_ssize_t height = source.shape[0]
    cdef Py_ssize_t width = source.shape[1]
    cdef Py_ssize_t i
    for i in range(row_count):
        if not 0 <= source_y[i] < height:
            raise IndexError("sample row out of range")
    for i in range(column_count):
        if not 0 <= source_x[i] < width:
            raise IndexError("sample column out of range")
    cdef Py_ssize_t r, c, sy, sx
    cdef const unsigned char* sample
    cdef unsigned char* pixel
    if target.strides[2] != 1:
        raise ValueError("target channels must be contiguous")
    with nogil:
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
                sample = &source[sy, sx, 0]
                pixel = &target[r, c, 0]
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
    const unsigned char[:, :, :] raster,
    Py_ssize_t channels,
    const unsigned char[:, :] mask,
    const Py_ssize_t[::1] mask_rows,
    const Py_ssize_t[::1] mask_columns,
):
    cdef Py_ssize_t rows = raster.shape[0], cols = raster.shape[1]
    if channels < 0 or channels > raster.shape[2]:
        raise ValueError("channels exceeds the raster's")
    if mask_rows.shape[0] != rows or mask_columns.shape[0] != cols:
        raise ValueError("mask indices differ from the raster in size")
    cdef Py_ssize_t i, j, k
    for i in range(rows):
        if not 0 <= mask_rows[i] < mask.shape[0]:
            raise IndexError("mask row out of range")
    for j in range(cols):
        if not 0 <= mask_columns[j] < mask.shape[1]:
            raise IndexError("mask column out of range")
    output = numpy.empty((rows, cols, channels + 1), dtype=numpy.uint8)
    cdef unsigned char[:, :, ::1] out = output
    cdef Py_ssize_t row
    with nogil:
        for i in range(rows):
            row = mask_rows[i]
            for j in range(cols):
                for k in range(channels):
                    out[i, j, k] = raster[i, j, k]
                out[i, j, channels] = mask[row, mask_columns[j]]
    return output


def alpha_channel(const unsigned char[:, :, ::1] pixels, bint presence):
    if pixels.shape[2] != 4:
        raise ValueError("pixels must be RGBA")
    cdef Py_ssize_t height = pixels.shape[0], width = pixels.shape[1]
    alpha = numpy.empty((height, width), dtype=numpy.uint8)
    cdef unsigned char[:, ::1] out = alpha
    present = numpy.zeros(256, dtype=numpy.bool_) if presence else None
    cdef unsigned char[::1] marks
    cdef unsigned char seen[256]
    cdef Py_ssize_t y, x, k
    cdef unsigned char value
    for k in range(256):
        seen[k] = 0
    with nogil:
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
