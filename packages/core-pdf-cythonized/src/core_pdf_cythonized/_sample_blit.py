# SPDX-License-Identifier: AGPL-3.0-only

import cython
import numpy

__all__ = (
    "alpha_channel",
    "bilinear_opaque_pixels",
    "bilinear_samples",
    "interleave_soft_mask",
    "sample_opaque_pixels",
)


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


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def bilinear_byte(
    source: cython.const[cython.uchar][:, :, ::1],
    top_row: cython.Py_ssize_t,
    bottom_row: cython.Py_ssize_t,
    fy: cython.double,
    left: cython.Py_ssize_t,
    right: cython.Py_ssize_t,
    fx: cython.double,
    k: cython.Py_ssize_t,
) -> cython.uchar:
    top: cython.double = (
        source[top_row, left, k] + (source[top_row, right, k] - source[top_row, left, k]) * fx
    )
    bottom: cython.double = (
        source[bottom_row, left, k]
        + (source[bottom_row, right, k] - source[bottom_row, left, k]) * fx
    )
    value: cython.double = top + (bottom - top) * fy + 0.5
    if value < 0.0:
        value = 0.0
    elif value > 255.0:
        value = 255.0
    return cython.cast(cython.uchar, value)


def check_bilinear_indices(
    source: cython.const[cython.uchar][:, :, ::1],
    rows0: cython.const[cython.Py_ssize_t][::1],
    rows1: cython.const[cython.Py_ssize_t][::1],
    row_weights: cython.const[cython.double][::1],
    columns0: cython.const[cython.Py_ssize_t][::1],
    columns1: cython.const[cython.Py_ssize_t][::1],
    column_weights: cython.const[cython.double][::1],
) -> None:
    rows: cython.Py_ssize_t = rows0.shape[0]
    columns: cython.Py_ssize_t = columns0.shape[0]
    height: cython.Py_ssize_t = source.shape[0]
    width: cython.Py_ssize_t = source.shape[1]
    if rows1.shape[0] != rows or row_weights.shape[0] != rows:
        raise ValueError("row indices and weights differ in size")
    if columns1.shape[0] != columns or column_weights.shape[0] != columns:
        raise ValueError("column indices and weights differ in size")
    i: cython.Py_ssize_t
    for i in range(rows):
        if not (0 <= rows0[i] < height and 0 <= rows1[i] < height):
            raise IndexError("sample row out of range")
    for i in range(columns):
        if not (0 <= columns0[i] < width and 0 <= columns1[i] < width):
            raise IndexError("sample column out of range")


def bilinear_samples(
    source: cython.const[cython.uchar][:, :, ::1],
    rows0: cython.const[cython.Py_ssize_t][::1],
    rows1: cython.const[cython.Py_ssize_t][::1],
    row_weights: cython.const[cython.double][::1],
    columns0: cython.const[cython.Py_ssize_t][::1],
    columns1: cython.const[cython.Py_ssize_t][::1],
    column_weights: cython.const[cython.double][::1],
):
    """The samples of source at each pair of a row and a column, each interpolated between
    two source rows (rows0[i] and rows1[i], weighted row_weights[i] towards the second) and
    two source columns likewise, rounded half up to bytes."""
    check_bilinear_indices(source, rows0, rows1, row_weights, columns0, columns1, column_weights)
    rows: cython.Py_ssize_t = rows0.shape[0]
    columns: cython.Py_ssize_t = columns0.shape[0]
    channels: cython.Py_ssize_t = source.shape[2]
    output = numpy.empty((rows, columns, channels), dtype=numpy.uint8)
    out: cython.uchar[:, :, ::1] = output
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    k: cython.Py_ssize_t
    with cython.nogil:
        for r in range(rows):
            for c in range(columns):
                for k in range(channels):
                    out[r, c, k] = bilinear_byte(
                        source,
                        rows0[r],
                        rows1[r],
                        row_weights[r],
                        columns0[c],
                        columns1[c],
                        column_weights[c],
                        k,
                    )
    return output


def bilinear_opaque_pixels(
    target: cython.uchar[:, :, :],
    source: cython.const[cython.uchar][:, :, ::1],
    rows0: cython.const[cython.Py_ssize_t][::1],
    rows1: cython.const[cython.Py_ssize_t][::1],
    row_weights: cython.const[cython.double][::1],
    columns0: cython.const[cython.Py_ssize_t][::1],
    columns1: cython.const[cython.Py_ssize_t][::1],
    column_weights: cython.const[cython.double][::1],
):
    """Write bilinear_samples of a grey or RGB source into the RGBA target, opaque."""
    check_bilinear_indices(source, rows0, rows1, row_weights, columns0, columns1, column_weights)
    rows: cython.Py_ssize_t = rows0.shape[0]
    columns: cython.Py_ssize_t = columns0.shape[0]
    channels: cython.Py_ssize_t = source.shape[2]
    if target.shape[2] != 4 or target.strides[2] != 1:
        raise ValueError("target must be RGBA with contiguous channels")
    if target.shape[0] != rows or target.shape[1] != columns:
        raise ValueError("sample indices differ from the target in size")
    if channels != 1 and channels != 3:
        raise ValueError("source must have one channel or three")
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    k: cython.Py_ssize_t
    pixel: cython.p_uchar
    with cython.nogil:
        for r in range(rows):
            for c in range(columns):
                pixel = cython.address(target[r, c, 0])
                for k in range(channels):
                    pixel[k] = bilinear_byte(
                        source,
                        rows0[r],
                        rows1[r],
                        row_weights[r],
                        columns0[c],
                        columns1[c],
                        column_weights[c],
                        k,
                    )
                if channels == 1:
                    pixel[1] = pixel[0]
                    pixel[2] = pixel[0]
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
