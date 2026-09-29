# SPDX-License-Identifier: AGPL-3.0-only

import cython
import numpy
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.string import memset

__all__ = ("box_downsample_blocks",)


def box_downsample_blocks(
    grid: cython.const[cython.uchar][:, :, :],
    row_edges: cython.const[cython.longlong][:],
    column_edges: cython.const[cython.longlong][:],
):
    target_height: cython.Py_ssize_t = row_edges.shape[0] - 1
    target_width: cython.Py_ssize_t = column_edges.shape[0] - 1
    channels: cython.Py_ssize_t = grid.shape[2]
    if target_height < 1 or target_width < 1:
        raise ValueError("downsampling needs at least one block each way")
    if row_edges[target_height] > grid.shape[0] or column_edges[target_width] > grid.shape[1]:
        raise ValueError("block edges run past the image")
    output = numpy.empty((target_height, target_width, channels), dtype=numpy.uint8)
    out: cython.uchar[:, :, ::1] = output
    row_width: cython.Py_ssize_t = target_width * channels
    sums: cython.p_uint = cython.cast(
        cython.p_uint, PyMem_Malloc(row_width * cython.sizeof(cython.uint))
    )
    if sums == cython.NULL:
        raise MemoryError()
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    k: cython.Py_ssize_t
    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    x0: cython.Py_ssize_t
    x1: cython.Py_ssize_t
    base: cython.Py_ssize_t
    rows_in_block: cython.longlong
    count: cython.longlong
    packed: cython.bint = grid.strides[2] == 1 and grid.strides[1] == channels
    row: cython.p_const_uchar
    p: cython.p_const_uchar
    s0: cython.uint
    s1: cython.uint
    s2: cython.uint
    s3: cython.uint
    try:
        with cython.nogil:
            for r in range(target_height):
                memset(sums, 0, row_width * cython.sizeof(cython.uint))
                for y in range(row_edges[r], row_edges[r + 1]):
                    if packed and grid.shape[1] > 0:
                        row = cython.address(grid[y, 0, 0])
                        for c in range(target_width):
                            x0 = column_edges[c]
                            x1 = column_edges[c + 1]
                            base = c * channels
                            p = row + x0 * channels
                            if channels == 1:
                                s0 = 0
                                for x in range(x1 - x0):
                                    s0 += p[x]
                                sums[base] += s0
                            elif channels == 3:
                                s0 = 0
                                s1 = 0
                                s2 = 0
                                for x in range(x1 - x0):
                                    s0 += p[3 * x]
                                    s1 += p[3 * x + 1]
                                    s2 += p[3 * x + 2]
                                sums[base] += s0
                                sums[base + 1] += s1
                                sums[base + 2] += s2
                            elif channels == 4:
                                s0 = 0
                                s1 = 0
                                s2 = 0
                                s3 = 0
                                for x in range(x1 - x0):
                                    s0 += p[4 * x]
                                    s1 += p[4 * x + 1]
                                    s2 += p[4 * x + 2]
                                    s3 += p[4 * x + 3]
                                sums[base] += s0
                                sums[base + 1] += s1
                                sums[base + 2] += s2
                                sums[base + 3] += s3
                            else:
                                for x in range(x1 - x0):
                                    for k in range(channels):
                                        sums[base + k] += p[x * channels + k]
                        continue
                    for c in range(target_width):
                        x0 = column_edges[c]
                        x1 = column_edges[c + 1]
                        base = c * channels
                        for x in range(x0, x1):
                            for k in range(channels):
                                sums[base + k] += grid[y, x, k]
                rows_in_block = row_edges[r + 1] - row_edges[r]
                for c in range(target_width):
                    count = rows_in_block * (column_edges[c + 1] - column_edges[c])
                    if count < 1:
                        count = 1
                    base = c * channels
                    for k in range(channels):
                        out[r, c, k] = cython.cast(
                            cython.uchar, cython.cast(cython.longlong, sums[base + k]) // count
                        )
    finally:
        PyMem_Free(sums)
    return output
