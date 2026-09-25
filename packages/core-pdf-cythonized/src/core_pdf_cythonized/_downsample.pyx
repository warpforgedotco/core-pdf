# SPDX-License-Identifier: AGPL-3.0-only
"""Box downsampling of an image to its device size (core_pdf.impl.render.target).

An image drawn smaller than its pixel size is averaged over rectangular
blocks before it is blitted. numpy did that with two add.reduceat passes
widening to uint32: 376 ms to take a 5086x6570 RGB scan to 788x788 on
SCORE-Bench mgm_895557014, most of it reduceat's per-element casting.

This walks the source once, row by row, adding each row into its target
row's accumulators, then divides. It is integer arithmetic throughout, so
there is no rounding to reproduce -- only numpy's widths: sums are uint32 and
wrap exactly as reduceat's did, and the quotient is a floor division by the
block's pixel count, truncated to uint8. The block edges are computed by
the caller, as (i * source) // target, and passed in.
"""

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.string cimport memset

import numpy

__all__ = ("box_downsample_blocks",)


def box_downsample_blocks(
    const unsigned char[:, :, :] grid,
    const long long[:] row_edges,
    const long long[:] column_edges,
):
    """Average `grid` over the blocks between consecutive edges; uint8 (rows, cols, channels)."""
    cdef Py_ssize_t target_height = row_edges.shape[0] - 1
    cdef Py_ssize_t target_width = column_edges.shape[0] - 1
    cdef Py_ssize_t channels = grid.shape[2]
    if target_height < 1 or target_width < 1:
        raise ValueError("downsampling needs at least one block each way")
    if row_edges[target_height] > grid.shape[0] or column_edges[target_width] > grid.shape[1]:
        raise ValueError("block edges run past the image")
    output = numpy.empty((target_height, target_width, channels), dtype=numpy.uint8)
    cdef unsigned char[:, :, ::1] out = output
    cdef Py_ssize_t row_width = target_width * channels
    cdef unsigned int *sums = <unsigned int *> PyMem_Malloc(row_width * sizeof(unsigned int))
    if sums == NULL:
        raise MemoryError()
    cdef Py_ssize_t r, c, k, y, x, x0, x1, base
    cdef long long rows_in_block, count
    try:
        with nogil:
            for r in range(target_height):
                memset(sums, 0, row_width * sizeof(unsigned int))
                for y in range(row_edges[r], row_edges[r + 1]):
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
                        out[r, c, k] = <unsigned char> (<long long> sums[base + k] // count)
    finally:
        PyMem_Free(sums)
    return output
