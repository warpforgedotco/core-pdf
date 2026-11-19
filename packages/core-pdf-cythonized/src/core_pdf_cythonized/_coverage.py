# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports import numpy as cnp
from cython.cimports.core_pdf_cythonized._alpha_blend import (
    accumulate_plane,
    blend_one,
    opaque_channel,
)
from cython.cimports.core_pdf_cythonized._byte_clamp import unit_to_byte
from cython.cimports.core_pdf_cythonized._coverage import (
    DENSE_CELLS,
    STACK_CELLS,
    STACK_EDGE_VALUES,
    STACK_EDGES,
    STACK_PIECES,
    TALLY_STACK,
)
from cython.cimports.core_pdf_cythonized._knockout_math import knockout_component
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.math import INFINITY, NAN, ceil, fabs, floor, rint
from cython.cimports.libc.stdlib import qsort
from cython.cimports.libc.string import memcpy, memset

# UNIT_FROM_BYTE[b] is b / 255.0, the same double division, looked up instead of
# recomputed for every byte channel of every pixel.
UNIT_FROM_BYTE = cython.declare(cython.double[256])
for _byte in range(256):
    UNIT_FROM_BYTE[_byte] = cython.cast(cython.double, _byte) / 255.0

cnp.import_array()


BytePlane = cython.struct(
    data=cython.p_uchar, row=cython.Py_ssize_t, column=cython.Py_ssize_t, channel=cython.Py_ssize_t
)


FloatPlane = cython.struct(data=cython.p_float, row=cython.Py_ssize_t, column=cython.Py_ssize_t)


@cython.cfunc
def edge_rows(edges: object) -> object:
    if not (
        cnp.PyArray_Check(edges)
        and cnp.PyArray_TYPE(cython.cast(cnp.ndarray, edges)) == cnp.NPY_FLOAT64
        and cnp.PyArray_NDIM(cython.cast(cnp.ndarray, edges)) == 2
        and cnp.PyArray_ISCARRAY_RO(cython.cast(cnp.ndarray, edges))
    ):
        edges = numpy.ascontiguousarray(edges, dtype=numpy.float64)
        if cnp.PyArray_NDIM(cython.cast(cnp.ndarray, edges)) != 2:
            _require_edge_matrix(edges)
    # The kernels read each edge as four consecutive doubles.
    if (
        cnp.PyArray_DIM(cython.cast(cnp.ndarray, edges), 0) != 0
        and cnp.PyArray_DIM(cython.cast(cnp.ndarray, edges), 1) != 4
    ):
        raise ValueError("edges must have four columns")
    return edges


@cython.cfunc
@cython.exceptval(-1, check=False)
def _require_edge_matrix(_view: typing.Optional[cython.const[cython.double][:, ::1]]) -> cython.int:
    return 0


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def edge_data(rows: object) -> cython.p_const_double:
    return cython.cast(cython.p_const_double, cnp.PyArray_DATA(cython.cast(cnp.ndarray, rows)))


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def edge_count(rows: object) -> cython.Py_ssize_t:
    return cnp.PyArray_DIM(cython.cast(cnp.ndarray, rows), 0)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def edge_width(rows: object) -> cython.Py_ssize_t:
    return cnp.PyArray_DIM(cython.cast(cnp.ndarray, rows), 1)


@cython.cfunc
@cython.exceptval(check=True)
def _zeroed_cells(
    cells: cython.Py_ssize_t,
    scratch: cython.p_double = cython.NULL,
    scratch_cells: cython.Py_ssize_t = 0,
) -> cython.p_double:
    """cells zeroed doubles: scratch when they fit in it, else a PyMem block."""
    acc: cython.p_double
    if cells <= scratch_cells:
        acc = scratch
    else:
        acc = cython.cast(
            cython.p_double,
            PyMem_Malloc((cells if cells > 0 else 1) * cython.sizeof(cython.double)),
        )
        if acc == cython.NULL:
            raise MemoryError
    if cells > 0:
        memset(acc, 0, cells * cython.sizeof(cython.double))
    return acc


def signed_area_coverage(edges, width: cython.int, height: cython.int):
    """The coverage of device edges over a width x height window, as fill_glyph_coverage
    measures it."""
    if height <= 0 or width <= 0 or edges.size == 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)
    view: cython.double[:, ::1] = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    if view.shape[1] != 4:
        raise ValueError("edges must have four columns")
    return _engine_plane(cython.address(view[0, 0]), view.shape[0], width, height, 0)[0]


def glyph_coverage_plane(
    edges,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.double,
    iy0: cython.double,
    width: cython.int,
    height: cython.int,
):
    """area_coverage_plane's plane alone."""
    found = area_coverage_plane(edges, crop_x0, crop_y1, scale, ix0, iy0, width, height)
    return None if found is None else found[0]


@cython.cfunc
@cython.exceptval(check=False)
def _sloped_edge_count(rows: object) -> cython.Py_ssize_t:
    view: cython.p_const_double = edge_data(rows)
    count: cython.Py_ssize_t = edge_count(rows)
    width: cython.Py_ssize_t = edge_width(rows)
    i: cython.Py_ssize_t
    kept: cython.Py_ssize_t = 0
    for i in range(count):
        if view[i * width + 1] != view[i * width + 3]:
            kept += 1
    return kept


@cython.cfunc
@cython.exceptval(check=True)
def _device_edges(
    rows: object,
    kept: cython.Py_ssize_t,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.double,
    iy0: cython.double,
    scratch: cython.p_double = cython.NULL,
    scratch_edges: cython.Py_ssize_t = 0,
) -> cython.p_double:
    """The kept edges in device space: in scratch when they fit, else a PyMem block."""
    view: cython.p_const_double = edge_data(rows)
    count: cython.Py_ssize_t = edge_count(rows)
    width: cython.Py_ssize_t = edge_width(rows)
    device: cython.p_double
    if kept <= scratch_edges:
        device = scratch
    else:
        device = cython.cast(cython.p_double, PyMem_Malloc(kept * 4 * cython.sizeof(cython.double)))
        if device == cython.NULL:
            raise MemoryError
    i: cython.Py_ssize_t
    out: cython.Py_ssize_t = 0
    row: cython.p_const_double
    sy: cython.double
    ey: cython.double
    for i in range(count):
        row = view + i * width
        sy = row[1]
        ey = row[3]
        if sy == ey:
            continue
        device[out] = (row[0] - crop_x0) * scale - ix0
        device[out + 1] = (crop_y1 - sy) * scale - iy0
        device[out + 2] = (row[2] - crop_x0) * scale - ix0
        device[out + 3] = (crop_y1 - ey) * scale - iy0
        out += 4
    return device


# The row engine: a fill's coverage one pixel row at a time, from the edges that reach the
# row. A window of at most DENSE_CELLS cells is summed densely at once, as glyphs are; a
# larger one row by row over the cells its pieces touch, so its memory is the edges and one
# row and its work the pieces and the painted runs. Each cell's sum is added in the order the
# dense sums added it.
#
# A pixel's sum of signed areas is its coverage only where the path's winding inside it is
# 0 or one sign of 1: regions of +1 and -1 cancel, and a region of 2 counts twice. Each row
# is split at the ends of its edges into bands where the edges keep their order; a row where
# they cross, or where a band has a region of winding beyond 1 or of both signs, is measured
# exactly instead: split as well where its edges cross, each band's edges between a region
# inside the path and one outside it are the outline there, and the signed areas of those
# pieces alone, each turned to add inside, give the coverage.

RowEdge = cython.struct(
    top_x=cython.double,
    bottom_x=cython.double,
    top_y=cython.double,
    bottom_y=cython.double,
    slope=cython.double,
    direction=cython.double,
    first_row=cython.Py_ssize_t,
    last_row=cython.Py_ssize_t,
    # Consecutive edges of one direction, each starting where the last ended, share a chain:
    # a monotone stretch of outline.
    chain=cython.Py_ssize_t,
)
# A chain: its edges' first and last index, which run down the window when direction is 1
# and up it when -1, the rows it spans, and a cursor on the edge last read.
Chain = cython.struct(
    first=cython.Py_ssize_t,
    last=cython.Py_ssize_t,
    direction=cython.int,
    top=cython.double,
    bottom=cython.double,
    cursor=cython.Py_ssize_t,
    bottom_cursor=cython.Py_ssize_t,
    xa=cython.double,
    xb=cython.double,
    index=cython.Py_ssize_t,
)
# A piece's right weight, added to the cell after its own once every left weight is in.
Entry = cython.struct(column=cython.Py_ssize_t, value=cython.double)
Run = cython.struct(start=cython.Py_ssize_t, end=cython.Py_ssize_t, coverage=cython.double)
# An edge across a band: its x at the band's top (or middle) and bottom.
Span = cython.struct(
    xa=cython.double, xb=cython.double, direction=cython.int, index=cython.Py_ssize_t
)
Rows = cython.struct(
    edges=cython.pointer[RowEdge],
    count=cython.Py_ssize_t,
    order=cython.p_Py_ssize_t,
    active=cython.p_Py_ssize_t,
    active_count=cython.Py_ssize_t,
    next_edge=cython.Py_ssize_t,
    width=cython.Py_ssize_t,
    height=cython.Py_ssize_t,
    spans=cython.pointer[Span],
    chains=cython.pointer[Chain],
    chain_count=cython.Py_ssize_t,
    band=cython.p_Py_ssize_t,
    events=cython.p_double,
    # Per row: whether it is measured exactly, and whether a band over it has a region of
    # winding 1 and of -1.
    mixed=cython.p_uchar,
    positive=cython.p_uchar,
    negative=cython.p_uchar,
    ys=cython.p_double,
    ys_capacity=cython.Py_ssize_t,
    entries=cython.pointer[Entry],
    entry_capacity=cython.Py_ssize_t,
    runs=cython.pointer[Run],
    run_capacity=cython.Py_ssize_t,
    run_count=cython.Py_ssize_t,
    acc=cython.p_double,
    exact_rows=cython.Py_ssize_t,
    dense=cython.bint,
    cells=cython.p_double,
    cells_heap=cython.bint,
    ordered=cython.bint,
    block=cython.p_char,
    # Whether _mark_rows ran out of work and left every row after to be supersampled.
    exhausted=cython.bint,
    # The edges crossing the last supersampled line, in order of x there, and its y.
    sampled=cython.pointer[Span],
    sampled_count=cython.Py_ssize_t,
    sampled_y=cython.double,
    block_bytes=cython.Py_ssize_t,
    acc_cells=cython.Py_ssize_t,
    heap_cells=cython.Py_ssize_t,
)

# Blocks a fill borrows and gives back, kept between fills so a glyph allocates nothing: the
# kernels hold the GIL and never nest.
SPARE_BLOCK = cython.declare(cython.p_char, cython.NULL)
SPARE_BLOCK_BYTES = cython.declare(cython.Py_ssize_t, 0)
SPARE_YS = cython.declare(cython.p_double, cython.NULL)
SPARE_YS_CAPACITY = cython.declare(cython.Py_ssize_t, 0)
SPARE_ENTRIES = cython.declare(cython.pointer[Entry], cython.NULL)
SPARE_ENTRY_CAPACITY = cython.declare(cython.Py_ssize_t, 0)
SPARE_RUNS = cython.declare(cython.pointer[Run], cython.NULL)
SPARE_RUN_CAPACITY = cython.declare(cython.Py_ssize_t, 0)
SPARE_ACC = cython.declare(cython.p_double, cython.NULL)
SPARE_ACC_CELLS = cython.declare(cython.Py_ssize_t, 0)
SPARE_CELLS = cython.declare(cython.p_double, cython.NULL)
SPARE_CELLS_COUNT = cython.declare(cython.Py_ssize_t, 0)
# A spare block past this many bytes is freed rather than kept.
SPARE_LIMIT = cython.declare(cython.Py_ssize_t, 1 << 22)

# _rows_begin's mode bit for an outline whose windings never mix, which skips _mark_rows.
SIMPLE_OUTLINE = cython.declare(cython.int, 4)


class SimpleEdges(numpy.ndarray):
    """Edge rows of an outline whose windings are 0 or one sign of 1 everywhere, under any
    placement, as windings_mix found: the fills skip looking for rows to measure exactly."""

    __slots__ = ()


# The dense sums take a window's edges only when every y is within this, as a cast of a
# row from a larger one would not be defined.
DENSE_REACH = cython.declare(cython.double, 1e15)
# Past this many items, the sorts go to qsort rather than insertion.
SORT_SMALL = cython.declare(cython.Py_ssize_t, 16)
# _mark_rows stops looking past this much work per edge and per row, counting the chains
# of each band and the edges its crossing walks read, as a path of tens of thousands of
# edges crossing each row would cost; the rest of the window's rows are then supersampled,
# which keeps their windings at a cost bounded by their edges.
SWEEP_WORK_PER_EDGE = cython.declare(cython.Py_ssize_t, 64)
SWEEP_WORK_PER_ROW = cython.declare(cython.Py_ssize_t, 16)
# A row's flag in rows.mixed when _mark_rows ran out of work before it: supersampled.
SAMPLED_ROW = cython.declare(cython.uchar, 2)
# A row to measure exactly with more active edges than this is supersampled instead.
EXACT_EDGES = cython.declare(cython.Py_ssize_t, 256)
# Each side of a supersampled row's grid of samples per pixel.
ROW_SAMPLES = cython.declare(cython.Py_ssize_t, 4)
OUTLINE_ROWS = cython.declare(cython.Py_ssize_t, 1)
# How far outside two edges a worked-out crossing may fall and still be taken at their end.
TOUCH = cython.declare(cython.double, 1e-7)
# How far two chains' x may overlap from rounding, relative to x, before they count as crossed.
CROSS_SLACK = cython.declare(cython.double, 1e-9)


@cython.cfunc
@cython.exceptval(-1, check=False)
def _accumulate_edges(
    edges: cython.pointer[RowEdge],
    count: cython.Py_ssize_t,
    width: cython.int,
    acc: cython.p_double,
) -> cython.int:
    """The dense sums of the edge table's pieces over a window width wide, in acc; the
    table's rows are already clamped to the window's height."""
    stride: cython.Py_ssize_t = width + 2

    # Each piece adds its left weight to its cell now and its right weight to the
    # next cell after every left weight is in, the order the sums were defined in.
    # The right weights wait in stack buffers, moved to the heap past STACK_PIECES.
    idx_stack = cython.declare(cython.Py_ssize_t[STACK_PIECES])
    right_stack = cython.declare(cython.double[STACK_PIECES])
    capacity: cython.Py_ssize_t = STACK_PIECES
    pieces: cython.Py_ssize_t = 0
    idx: cython.p_Py_ssize_t = idx_stack
    right_w: cython.p_double = right_stack

    i: cython.Py_ssize_t
    r: cython.Py_ssize_t
    first_row: cython.Py_ssize_t
    last_row: cython.Py_ssize_t
    column: cython.Py_ssize_t
    column_count: cython.Py_ssize_t
    column_offset: cython.Py_ssize_t
    direction: cython.double
    top_x: cython.double
    top_y: cython.double
    bottom_y: cython.double
    slope: cython.double
    row_top: cython.double
    row_bottom: cython.double
    row_height: cython.double
    entry_x: cython.double
    exit_x: cython.double
    left_x: cython.double
    right_x: cython.double
    walk_left: cython.double
    walk_right: cython.double
    floor_left: cython.double
    column_start: cython.double
    fragment_left: cython.double
    fragment_right: cython.double
    piece_span: cython.double
    share: cython.double
    fragment_height: cython.double
    midpoint: cython.double
    offset_in_cell: cython.double
    signed_height: cython.double
    vertical: cython.bint
    new_idx: cython.p_Py_ssize_t
    new_right: cython.p_double
    edge: cython.pointer[RowEdge]

    try:
        for i in range(count):
            # The table's values are the ones these sums worked out from the edge's points,
            # and its rows, clamped to the window, give the same pieces.
            edge = edges + i
            direction = edge.direction
            top_x = edge.top_x
            top_y = edge.top_y
            bottom_y = edge.bottom_y
            slope = edge.slope
            first_row = edge.first_row
            last_row = edge.last_row

            for r in range(first_row, last_row):
                row_top = py_max(cython.cast(cython.double, r), top_y)
                row_bottom = py_min(r + 1.0, bottom_y)
                row_height = row_bottom - row_top
                if row_height <= 0.0:
                    continue
                entry_x = top_x + (row_top - top_y) * slope
                exit_x = top_x + (row_bottom - top_y) * slope
                left_x = py_min(exit_x, entry_x)
                right_x = py_max(exit_x, entry_x)
                walk_left = py_min(width + 1.0, py_max(-1.0, left_x))
                walk_right = py_min(width + 1.0, py_max(-1.0, right_x))
                floor_left = floor(walk_left)
                column_count = cython.cast(cython.Py_ssize_t, floor(walk_right) - floor_left + 1.0)
                piece_span = right_x - left_x
                vertical = piece_span <= 0.0

                for column_offset in range(column_count):
                    column_start = floor_left + column_offset
                    fragment_left = left_x if column_offset == 0 else column_start
                    fragment_right = (
                        right_x if column_offset == column_count - 1 else column_start + 1.0
                    )
                    if not (vertical or fragment_right > fragment_left):
                        continue
                    share = 1.0 if vertical else (fragment_right - fragment_left) / piece_span
                    fragment_height = row_height * share
                    midpoint = (fragment_left + fragment_right) * 0.5
                    column = cython.cast(
                        cython.Py_ssize_t,
                        py_min(cython.cast(cython.double, width), py_max(0.0, floor(midpoint))),
                    )
                    offset_in_cell = py_min(1.0, py_max(0.0, midpoint - column))
                    signed_height = direction * fragment_height

                    if pieces == capacity:
                        new_idx = cython.cast(
                            cython.p_Py_ssize_t,
                            PyMem_Malloc(2 * capacity * cython.sizeof(cython.Py_ssize_t)),
                        )
                        new_right = cython.cast(
                            cython.p_double,
                            PyMem_Malloc(2 * capacity * cython.sizeof(cython.double)),
                        )
                        if new_idx == cython.NULL or new_right == cython.NULL:
                            PyMem_Free(new_idx)
                            PyMem_Free(new_right)
                            raise MemoryError
                        memcpy(new_idx, idx, capacity * cython.sizeof(cython.Py_ssize_t))
                        memcpy(new_right, right_w, capacity * cython.sizeof(cython.double))
                        if idx != idx_stack:
                            PyMem_Free(idx)
                            PyMem_Free(right_w)
                        idx = new_idx
                        right_w = new_right
                        capacity *= 2
                    column = r * stride + column
                    acc[column] += signed_height * (1.0 - offset_in_cell)
                    idx[pieces] = column
                    right_w[pieces] = signed_height * offset_in_cell
                    pieces += 1

        with cython.nogil:
            for i in range(pieces):
                acc[idx[i] + 1] += right_w[i]
    finally:
        if idx != idx_stack:
            PyMem_Free(idx)
            PyMem_Free(right_w)
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def _accumulate_device(
    e: cython.p_const_double,
    count: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    acc: cython.p_double,
) -> cython.int:
    """The dense sums of the device edges' pieces over a width x height window, in acc,
    as _accumulate_edges gives them from the edge table: for an outline known simple, which
    needs no table."""
    stride: cython.Py_ssize_t = width + 2

    # Each piece adds its left weight to its cell now and its right weight to the
    # next cell after every left weight is in, the order the sums were defined in.
    # The right weights wait in stack buffers, moved to the heap past STACK_PIECES.
    idx_stack = cython.declare(cython.Py_ssize_t[STACK_PIECES])
    right_stack = cython.declare(cython.double[STACK_PIECES])
    capacity: cython.Py_ssize_t = STACK_PIECES
    pieces: cython.Py_ssize_t = 0
    idx: cython.p_Py_ssize_t = idx_stack
    right_w: cython.p_double = right_stack

    i: cython.Py_ssize_t
    r: cython.Py_ssize_t
    first_row: cython.Py_ssize_t
    last_row: cython.Py_ssize_t
    column: cython.Py_ssize_t
    column_count: cython.Py_ssize_t
    column_offset: cython.Py_ssize_t
    sx: cython.double
    sy: cython.double
    ex: cython.double
    ey: cython.double
    direction: cython.double
    top_x: cython.double
    top_y: cython.double
    bottom_x: cython.double
    bottom_y: cython.double
    slope: cython.double
    row_top: cython.double
    row_bottom: cython.double
    row_height: cython.double
    entry_x: cython.double
    exit_x: cython.double
    left_x: cython.double
    right_x: cython.double
    walk_left: cython.double
    walk_right: cython.double
    floor_left: cython.double
    column_start: cython.double
    fragment_left: cython.double
    fragment_right: cython.double
    piece_span: cython.double
    share: cython.double
    fragment_height: cython.double
    midpoint: cython.double
    offset_in_cell: cython.double
    signed_height: cython.double
    vertical: cython.bint
    new_idx: cython.p_Py_ssize_t
    new_right: cython.p_double

    try:
        for i in range(count):
            sx = e[i * 4]
            sy = e[i * 4 + 1]
            ex = e[i * 4 + 2]
            ey = e[i * 4 + 3]
            if sy == ey:
                continue
            if ey > sy:
                direction = 1.0
                top_x = sx
                top_y = sy
                bottom_x = ex
                bottom_y = ey
            else:
                direction = -1.0
                top_x = ex
                top_y = ey
                bottom_x = sx
                bottom_y = sy
            slope = (bottom_x - top_x) / (bottom_y - top_y)

            first_row = cython.cast(cython.Py_ssize_t, py_max(floor(top_y), 0.0))
            last_row = cython.cast(
                cython.Py_ssize_t, py_min(ceil(bottom_y), cython.cast(cython.double, height))
            )

            for r in range(first_row, last_row):
                row_top = py_max(cython.cast(cython.double, r), top_y)
                row_bottom = py_min(r + 1.0, bottom_y)
                row_height = row_bottom - row_top
                if row_height <= 0.0:
                    continue
                entry_x = top_x + (row_top - top_y) * slope
                exit_x = top_x + (row_bottom - top_y) * slope
                left_x = py_min(exit_x, entry_x)
                right_x = py_max(exit_x, entry_x)
                walk_left = py_min(width + 1.0, py_max(-1.0, left_x))
                walk_right = py_min(width + 1.0, py_max(-1.0, right_x))
                floor_left = floor(walk_left)
                column_count = cython.cast(cython.Py_ssize_t, floor(walk_right) - floor_left + 1.0)
                piece_span = right_x - left_x
                vertical = piece_span <= 0.0

                for column_offset in range(column_count):
                    column_start = floor_left + column_offset
                    fragment_left = left_x if column_offset == 0 else column_start
                    fragment_right = (
                        right_x if column_offset == column_count - 1 else column_start + 1.0
                    )
                    if not (vertical or fragment_right > fragment_left):
                        continue
                    share = 1.0 if vertical else (fragment_right - fragment_left) / piece_span
                    fragment_height = row_height * share
                    midpoint = (fragment_left + fragment_right) * 0.5
                    column = cython.cast(
                        cython.Py_ssize_t,
                        py_min(cython.cast(cython.double, width), py_max(0.0, floor(midpoint))),
                    )
                    offset_in_cell = py_min(1.0, py_max(0.0, midpoint - column))
                    signed_height = direction * fragment_height

                    if pieces == capacity:
                        new_idx = cython.cast(
                            cython.p_Py_ssize_t,
                            PyMem_Malloc(2 * capacity * cython.sizeof(cython.Py_ssize_t)),
                        )
                        new_right = cython.cast(
                            cython.p_double,
                            PyMem_Malloc(2 * capacity * cython.sizeof(cython.double)),
                        )
                        if new_idx == cython.NULL or new_right == cython.NULL:
                            PyMem_Free(new_idx)
                            PyMem_Free(new_right)
                            raise MemoryError
                        memcpy(new_idx, idx, capacity * cython.sizeof(cython.Py_ssize_t))
                        memcpy(new_right, right_w, capacity * cython.sizeof(cython.double))
                        if idx != idx_stack:
                            PyMem_Free(idx)
                            PyMem_Free(right_w)
                        idx = new_idx
                        right_w = new_right
                        capacity *= 2
                    column = r * stride + column
                    acc[column] += signed_height * (1.0 - offset_in_cell)
                    idx[pieces] = column
                    right_w[pieces] = signed_height * offset_in_cell
                    pieces += 1

        with cython.nogil:
            for i in range(pieces):
                acc[idx[i] + 1] += right_w[i]
    finally:
        if idx != idx_stack:
            PyMem_Free(idx)
            PyMem_Free(right_w)
    return 0


@cython.cfunc
@cython.exceptval(check=False)
def _rows_clear(rows: cython.pointer[Rows]) -> cython.void:
    """rows with no blocks, so _rows_free can run whatever _rows_begin reached."""
    rows.edges = cython.NULL
    rows.block = cython.NULL
    rows.block_bytes = 0
    rows.ys = cython.NULL
    rows.entries = cython.NULL
    rows.runs = cython.NULL
    rows.acc = cython.NULL
    rows.acc_cells = 0
    rows.cells = cython.NULL
    rows.cells_heap = False
    rows.heap_cells = 0
    rows.dense = False
    rows.count = 0
    rows.active_count = 0
    rows.next_edge = 0
    rows.run_count = 0
    rows.exact_rows = 0
    rows.exhausted = False
    rows.sampled_count = 0
    rows.sampled_y = -INFINITY
    rows.entry_capacity = 0
    rows.run_capacity = 0
    rows.ys_capacity = 0


@cython.cfunc
@cython.exceptval(check=False)
def _rows_free(rows: cython.pointer[Rows]) -> cython.void:
    """Give rows' blocks back as spares, or free them."""
    global SPARE_BLOCK, SPARE_BLOCK_BYTES, SPARE_YS, SPARE_YS_CAPACITY, SPARE_ENTRIES
    global SPARE_ENTRY_CAPACITY, SPARE_RUNS, SPARE_RUN_CAPACITY, SPARE_ACC, SPARE_ACC_CELLS
    global SPARE_CELLS, SPARE_CELLS_COUNT
    if SPARE_BLOCK == cython.NULL and rows.block_bytes <= SPARE_LIMIT:
        SPARE_BLOCK = rows.block
        SPARE_BLOCK_BYTES = rows.block_bytes
    else:
        PyMem_Free(rows.block)
    if SPARE_YS == cython.NULL and rows.ys_capacity * 8 <= SPARE_LIMIT:
        SPARE_YS = rows.ys
        SPARE_YS_CAPACITY = rows.ys_capacity
    else:
        PyMem_Free(rows.ys)
    if SPARE_ENTRIES == cython.NULL and rows.entry_capacity * cython.sizeof(Entry) <= SPARE_LIMIT:
        SPARE_ENTRIES = rows.entries
        SPARE_ENTRY_CAPACITY = rows.entry_capacity
    else:
        PyMem_Free(rows.entries)
    if SPARE_RUNS == cython.NULL and rows.run_capacity * cython.sizeof(Run) <= SPARE_LIMIT:
        SPARE_RUNS = rows.runs
        SPARE_RUN_CAPACITY = rows.run_capacity
    else:
        PyMem_Free(rows.runs)
    if SPARE_ACC == cython.NULL and rows.acc_cells * 8 <= SPARE_LIMIT:
        SPARE_ACC = rows.acc
        SPARE_ACC_CELLS = rows.acc_cells
    else:
        PyMem_Free(rows.acc)
    if rows.cells_heap:
        if SPARE_CELLS == cython.NULL and rows.heap_cells * 8 <= SPARE_LIMIT:
            SPARE_CELLS = rows.cells
            SPARE_CELLS_COUNT = rows.heap_cells
        else:
            PyMem_Free(rows.cells)


@cython.cfunc
@cython.exceptval(-1, check=False)
def _rows_begin(
    rows: cython.pointer[Rows],
    device: cython.p_const_double,
    kept: cython.Py_ssize_t,
    width: cython.Py_ssize_t,
    height: cython.Py_ssize_t,
    cells_stack: cython.p_double,
    stack_cells: cython.Py_ssize_t,
    mode: cython.int,
) -> cython.int:
    """rows over the kept device edges of a width x height window; _rows_free frees it,
    whether or not this succeeds. The dense sums go in cells_stack when they fit. mode 1 asks
    for the dense sums and 2 for the rows, where each can be had, so both can be checked."""
    rows.width = width
    rows.height = height
    cells: cython.Py_ssize_t = kept if kept > 0 else 1
    per_edge: cython.Py_ssize_t = (
        cython.sizeof(RowEdge)
        + 2 * cython.sizeof(cython.Py_ssize_t)
        + 2 * cython.sizeof(Span)
        + cython.sizeof(Chain)
        + cython.sizeof(cython.Py_ssize_t)
        + 2 * cython.sizeof(cython.double)
    )
    global SPARE_BLOCK, SPARE_BLOCK_BYTES, SPARE_YS, SPARE_YS_CAPACITY, SPARE_ENTRIES
    global SPARE_ENTRY_CAPACITY, SPARE_RUNS, SPARE_RUN_CAPACITY, SPARE_CELLS, SPARE_CELLS_COUNT
    block: cython.p_char
    needed: cython.Py_ssize_t = cells * per_edge + 3 * (height + 1)
    if SPARE_BLOCK != cython.NULL and SPARE_BLOCK_BYTES >= needed:
        block = SPARE_BLOCK
        rows.block_bytes = SPARE_BLOCK_BYTES
        SPARE_BLOCK = cython.NULL
    else:
        block = cython.cast(cython.p_char, PyMem_Malloc(needed))
        if block == cython.NULL:
            raise MemoryError
        rows.block_bytes = needed
    rows.block = block
    rows.ys = SPARE_YS
    rows.ys_capacity = SPARE_YS_CAPACITY
    SPARE_YS = cython.NULL
    rows.entries = SPARE_ENTRIES
    rows.entry_capacity = SPARE_ENTRY_CAPACITY
    SPARE_ENTRIES = cython.NULL
    rows.runs = SPARE_RUNS
    rows.run_capacity = SPARE_RUN_CAPACITY
    SPARE_RUNS = cython.NULL
    rows.edges = cython.cast(cython.pointer[RowEdge], block)
    block += cells * cython.sizeof(RowEdge)
    rows.order = cython.cast(cython.p_Py_ssize_t, block)
    block += cells * cython.sizeof(cython.Py_ssize_t)
    rows.active = cython.cast(cython.p_Py_ssize_t, block)
    block += cells * cython.sizeof(cython.Py_ssize_t)
    rows.spans = cython.cast(cython.pointer[Span], block)
    block += cells * cython.sizeof(Span)
    rows.sampled = cython.cast(cython.pointer[Span], block)
    block += cells * cython.sizeof(Span)
    rows.chains = cython.cast(cython.pointer[Chain], block)
    block += cells * cython.sizeof(Chain)
    rows.band = cython.cast(cython.p_Py_ssize_t, block)
    block += cells * cython.sizeof(cython.Py_ssize_t)
    rows.events = cython.cast(cython.p_double, block)
    block += 2 * cells * cython.sizeof(cython.double)
    rows.mixed = cython.cast(cython.p_uchar, block)
    rows.positive = rows.mixed + (height + 1)
    rows.negative = rows.positive + (height + 1)
    i: cython.Py_ssize_t
    out: cython.Py_ssize_t = 0
    sx: cython.double
    sy: cython.double
    ex: cython.double
    ey: cython.double
    first: cython.double
    last: cython.double
    edge: cython.pointer[RowEdge]
    dense: cython.bint = mode != 2 and (mode == 1 or (width + 2) * height <= DENSE_CELLS)
    last_x: cython.double = 0.0
    last_y: cython.double = 0.0
    last_direction: cython.double = 0.0
    chain: cython.Py_ssize_t = -1
    for i in range(kept):
        sx = device[i * 4]
        sy = device[i * 4 + 1]
        ex = device[i * 4 + 2]
        ey = device[i * 4 + 3]
        if not (-DENSE_REACH <= sy <= DENSE_REACH and -DENSE_REACH <= ey <= DENSE_REACH):
            dense = False
        # An edge with a NaN or infinite coordinate reaches no row or column the sums can
        # measure, and would leave the sorts without an order.
        if sy == ey or not (
            sy - sy == 0.0 and ey - ey == 0.0 and sx - sx == 0.0 and ex - ex == 0.0
        ):
            continue
        edge = rows.edges + out
        if ey > sy:
            edge.direction = 1.0
            edge.top_x = sx
            edge.bottom_x = ex
            edge.top_y = sy
            edge.bottom_y = ey
            edge.slope = (ex - sx) / (ey - sy)
        else:
            edge.direction = -1.0
            edge.top_x = ex
            edge.bottom_x = sx
            edge.top_y = ey
            edge.bottom_y = sy
            edge.slope = (sx - ex) / (sy - ey)
        if edge.slope - edge.slope != 0.0:
            continue
        first = py_min(py_max(floor(edge.top_y), 0.0), cython.cast(cython.double, height))
        last = py_max(py_min(ceil(edge.bottom_y), cython.cast(cython.double, height)), 0.0)
        edge.first_row = cython.cast(cython.Py_ssize_t, first)
        edge.last_row = cython.cast(cython.Py_ssize_t, last)
        if not (out > 0 and sx == last_x and sy == last_y and edge.direction == last_direction):
            chain += 1
        edge.chain = chain
        last_x = ex
        last_y = ey
        last_direction = edge.direction
        out += 1
    rows.count = out
    rows.ordered = False
    if mode & SIMPLE_OUTLINE:
        memset(rows.mixed, 0, 3 * (height + 1))
    else:
        _mark_rows(rows)
    if dense:
        rows.dense = True
        if (width + 2) * height <= stack_cells:
            rows.cells = cells_stack
            memset(rows.cells, 0, (width + 2) * height * cython.sizeof(cython.double))
        else:
            if SPARE_CELLS != cython.NULL and SPARE_CELLS_COUNT >= (width + 2) * height:
                rows.cells = SPARE_CELLS
                rows.heap_cells = SPARE_CELLS_COUNT
                SPARE_CELLS = cython.NULL
                memset(rows.cells, 0, (width + 2) * height * cython.sizeof(cython.double))
            else:
                rows.cells = _zeroed_cells((width + 2) * height)
                rows.heap_cells = (width + 2) * height
            rows.cells_heap = True
        _accumulate_edges(rows.edges, rows.count, width, rows.cells)
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def _row_buffers(rows: cython.pointer[Rows]) -> cython.int:
    """rows' one-row cells and pieces, made the first time a row needs them."""
    global SPARE_ACC, SPARE_ACC_CELLS
    if rows.acc != cython.NULL:
        return 0
    if SPARE_ACC != cython.NULL and SPARE_ACC_CELLS >= rows.width + 2:
        rows.acc = SPARE_ACC
        rows.acc_cells = SPARE_ACC_CELLS
        SPARE_ACC = cython.NULL
        memset(rows.acc, 0, (rows.width + 2) * cython.sizeof(cython.double))
    else:
        rows.acc = _zeroed_cells(rows.width + 2)
        rows.acc_cells = rows.width + 2
    if rows.entries == cython.NULL:
        rows.entry_capacity = 64
        rows.entries = cython.cast(
            cython.pointer[Entry], PyMem_Malloc(rows.entry_capacity * cython.sizeof(Entry))
        )
        if rows.entries == cython.NULL:
            rows.entry_capacity = 0
            raise MemoryError
    return 0


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
def _emit(
    rows: cython.pointer[Rows],
    start: cython.Py_ssize_t,
    end: cython.Py_ssize_t,
    coverage: cython.double,
) -> cython.int:
    grown: cython.pointer[Run]
    last: cython.pointer[Run]
    capacity: cython.Py_ssize_t
    if end <= start:
        return 0
    if rows.run_count:
        last = rows.runs + rows.run_count - 1
        if last.end == start and last.coverage == coverage:
            last.end = end
            return 0
    if rows.run_count == rows.run_capacity:
        capacity = 2 * rows.run_capacity if rows.run_capacity else 64
        grown = cython.cast(cython.pointer[Run], PyMem_Malloc(capacity * cython.sizeof(Run)))
        if grown == cython.NULL:
            raise MemoryError
        if rows.run_count:
            memcpy(grown, rows.runs, rows.run_count * cython.sizeof(Run))
        PyMem_Free(rows.runs)
        rows.runs = grown
        rows.run_capacity = capacity
    rows.runs[rows.run_count].start = start
    rows.runs[rows.run_count].end = end
    rows.runs[rows.run_count].coverage = coverage
    rows.run_count += 1
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def _order_edges(rows: cython.pointer[Rows]) -> cython.int:
    """rows.order, the edges in order of their first row, then of index, made the first time
    a row reads the active list: a counting sort over the rows."""
    if rows.ordered:
        return 0
    height: cython.Py_ssize_t = rows.height
    edges: cython.pointer[RowEdge] = rows.edges
    tally_stack = cython.declare(cython.Py_ssize_t[TALLY_STACK])
    tally: cython.p_Py_ssize_t = tally_stack
    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    total: cython.Py_ssize_t = 0
    if height + 1 > TALLY_STACK:
        tally = cython.cast(
            cython.p_Py_ssize_t, PyMem_Malloc((height + 1) * cython.sizeof(cython.Py_ssize_t))
        )
        if tally == cython.NULL:
            raise MemoryError
    memset(tally, 0, (height + 1) * cython.sizeof(cython.Py_ssize_t))
    for i in range(rows.count):
        tally[edges[i].first_row] += 1
    for k in range(height + 1):
        total += tally[k]
        tally[k] = total - tally[k]
    for i in range(rows.count):
        rows.order[tally[edges[i].first_row]] = i
        tally[edges[i].first_row] += 1
    if tally != tally_stack:
        PyMem_Free(tally)
    rows.ordered = True
    return 0


@cython.cfunc
@cython.exceptval(check=False)
def _advance(rows: cython.pointer[Rows], r: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """Bring the active list to row r, rows rising: the edges that reach it, in index
    order; their count."""
    edges: cython.pointer[RowEdge] = rows.edges
    active: cython.p_Py_ssize_t = rows.active
    index: cython.Py_ssize_t
    position: cython.Py_ssize_t
    a: cython.Py_ssize_t
    keep: cython.Py_ssize_t = 0
    while rows.next_edge < rows.count and edges[rows.order[rows.next_edge]].first_row <= r:
        index = rows.order[rows.next_edge]
        rows.next_edge += 1
        if edges[index].last_row <= r:
            continue
        position = rows.active_count
        while position > 0 and active[position - 1] > index:
            active[position] = active[position - 1]
            position -= 1
        active[position] = index
        rows.active_count += 1
    for a in range(rows.active_count):
        if edges[active[a]].last_row > r:
            active[keep] = active[a]
            keep += 1
    rows.active_count = keep
    return keep


@cython.cfunc
@cython.exceptval(-1, check=False)
def _reserve_ys(rows: cython.pointer[Rows], needed: cython.Py_ssize_t) -> cython.int:
    """Room for needed band edges in rows.ys, keeping those already there."""
    grown: cython.p_double
    if needed <= rows.ys_capacity:
        return 0
    grown = cython.cast(cython.p_double, PyMem_Malloc(needed * cython.sizeof(cython.double)))
    if grown == cython.NULL:
        raise MemoryError
    if rows.ys_capacity:
        memcpy(grown, rows.ys, rows.ys_capacity * cython.sizeof(cython.double))
    PyMem_Free(rows.ys)
    rows.ys = grown
    rows.ys_capacity = needed
    return 0


@cython.cfunc
@cython.exceptval(check=False)
def _sort_doubles(values: cython.p_double, count: cython.Py_ssize_t) -> cython.void:
    a: cython.Py_ssize_t
    k: cython.Py_ssize_t
    value: cython.double
    for a in range(1, count):
        value = values[a]
        k = a
        while k > 0 and values[k - 1] > value:
            values[k] = values[k - 1]
            k -= 1
        values[k] = value


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _compare_spans(
    left: cython.pointer[cython.const[cython.void]],
    right: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    """By xa, then xb, then index: the order _sort_spans gives."""
    a: cython.pointer[cython.const[Span]] = cython.cast(cython.pointer[cython.const[Span]], left)
    b: cython.pointer[cython.const[Span]] = cython.cast(cython.pointer[cython.const[Span]], right)
    if a.xa != b.xa:
        return -1 if a.xa < b.xa else 1
    if a.xb != b.xb:
        return -1 if a.xb < b.xb else 1
    return (a.index > b.index) - (a.index < b.index)


@cython.cfunc
@cython.exceptval(check=False)
def _sort_spans(spans: cython.pointer[Span], count: cython.Py_ssize_t) -> cython.void:
    """By xa, then xb, then index: a total order, so the same on every platform; qsort past
    SORT_SMALL spans."""
    a: cython.Py_ssize_t
    k: cython.Py_ssize_t
    span: Span
    if count > SORT_SMALL:
        qsort(spans, count, cython.sizeof(Span), _compare_spans)
        return
    for a in range(1, count):
        span = spans[a]
        k = a
        while k > 0 and (
            spans[k - 1].xa > span.xa
            or (
                spans[k - 1].xa == span.xa
                and (
                    spans[k - 1].xb > span.xb
                    or (spans[k - 1].xb == span.xb and spans[k - 1].index > span.index)
                )
            )
        ):
            spans[k] = spans[k - 1]
            k -= 1
        spans[k] = span


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def _edge_x(edge: cython.pointer[RowEdge], y: cython.double) -> cython.double:
    """The edge's x at y, its end's own x at either end, so edges that meet agree there."""
    if y == edge.bottom_y:
        return edge.bottom_x
    if y == edge.top_y:
        return edge.top_x
    return edge.top_x + (y - edge.top_y) * edge.slope


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def _chain_x(
    rows: cython.pointer[Rows],
    chain: cython.pointer[Chain],
    cursor: cython.p_Py_ssize_t,
    y: cython.double,
) -> cython.double:
    """The x of chain at y, y rising between calls with one cursor: the cursor moves down the
    chain's edges to the one that reaches y."""
    edge: cython.pointer[RowEdge] = rows.edges + cursor[0]
    while edge.bottom_y < y and cursor[0] != chain.last:
        cursor[0] += chain.direction
        edge = rows.edges + cursor[0]
    return _edge_x(edge, y)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def _mark_span(
    flags: cython.p_uchar,
    height: cython.Py_ssize_t,
    y0: cython.double,
    y1: cython.double,
    value: cython.uchar = 1,
) -> cython.void:
    """Set the flag of every row that overlaps y0..y1 to value (1 unless given)."""
    first: cython.double = py_max(floor(y0), 0.0)
    last: cython.double = py_min(ceil(y1), cython.cast(cython.double, height))
    r: cython.Py_ssize_t
    if not (last > first):
        return
    for r in range(cython.cast(cython.Py_ssize_t, first), cython.cast(cython.Py_ssize_t, last)):
        flags[r] = value


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def _pair_crossing(
    rows: cython.pointer[Rows],
    left: cython.pointer[Chain],
    right: cython.pointer[Chain],
    ya: cython.double,
    yb: cython.double,
    work: cython.p_Py_ssize_t,
) -> cython.double:
    """Where left, ordered before right at ya, passes it in ya..yb: the y of the last end of
    either's edges before the first where left's x exceeds right's, or NaN when it never
    does. Their cursors stand on the edges that reach ya."""
    edges: cython.pointer[RowEdge] = rows.edges
    li: cython.Py_ssize_t = left.cursor
    ri: cython.Py_ssize_t = right.cursor
    el: cython.pointer[RowEdge]
    er: cython.pointer[RowEdge]
    y: cython.double = ya
    below: cython.double
    xl: cython.double
    # Ordered at ya, the pair is read at each next end of either's edges, where the edge
    # ending there gives its end's own x, as the next edge gives at its top.
    while y < yb:
        el = edges + li
        er = edges + ri
        work[0] += 1
        below = py_min(py_min(el.bottom_y, er.bottom_y), yb)
        xl = _edge_x(el, below)
        # A gap within rounding of the coordinates is chains that meet, not cross.
        if xl - _edge_x(er, below) > CROSS_SLACK * py_max(1.0, fabs(xl)):
            return y
        y = below
        if el.bottom_y <= y and li != left.last:
            li += left.direction
        if er.bottom_y <= y and ri != right.last:
            ri += right.direction
    return NAN


@cython.cfunc
@cython.exceptval(check=False)
def _mark_rows(rows: cython.pointer[Rows]) -> cython.void:
    """Find the rows to measure exactly, in one sweep down the window. The edges join into
    chains, monotone stretches of outline; between the chains' ends, the chains keep a set,
    and, ordered by x at the band's top (then its middle), must have regions of winding 0 or
    one sign of 1 between them, and no two neighbours may cross anywhere in the band: each
    pair is walked down the band at every end of either's edges, where the gap between them,
    piecewise linear, takes its extremes. A band that breaks either marks its rows (from the
    crossing on), and so does a row under bands of both signs. Past its budget of work, which
    counts only the bands and the edges walked, the sweep leaves every row after to be
    supersampled (SAMPLED_ROW)."""
    edges: cython.pointer[RowEdge] = rows.edges
    chains: cython.pointer[Chain] = rows.chains
    band: cython.p_Py_ssize_t = rows.band
    events: cython.p_double = rows.events
    height: cython.Py_ssize_t = rows.height
    count: cython.Py_ssize_t = 0
    event_count: cython.Py_ssize_t = 0
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    m: cython.Py_ssize_t
    e: cython.Py_ssize_t
    winding: cython.int
    sign: cython.int
    edge: cython.pointer[RowEdge]
    chain: cython.pointer[Chain]
    ya: cython.double
    yb: cython.double
    y: cython.double
    bad: cython.bint
    earliest: cython.double
    middle: cython.double
    memset(rows.mixed, 0, 3 * (height + 1))
    for i in range(rows.count):
        edge = edges + i
        if count and edge.chain == edges[i - 1].chain:
            chain = chains + count - 1
            chain.last = i
            chain.top = py_min(chain.top, edge.top_y)
            chain.bottom = py_max(chain.bottom, edge.bottom_y)
        else:
            chain = chains + count
            chain.first = i
            chain.last = i
            chain.direction = 1 if edge.direction > 0.0 else -1
            chain.top = edge.top_y
            chain.bottom = edge.bottom_y
            chain.index = count
            count += 1
    rows.chain_count = count
    for i in range(count):
        chain = chains + i
        # A chain running up the window is read from its last edge.
        if chain.direction < 0:
            chain.first, chain.last = chain.last, chain.first
        chain.cursor = chain.first
        chain.bottom_cursor = chain.first
        events[event_count] = chain.top
        events[event_count + 1] = chain.bottom
        event_count += 2
    if event_count > 32:
        qsort(events, event_count, cython.sizeof(cython.double), _compare_doubles)
    else:
        _sort_doubles(events, event_count)
    # The chains in order of their top: they join the band list as the sweep reaches them,
    # and leave it at their bottom; the list keeps the last band's order.
    by_top: cython.p_Py_ssize_t = rows.order
    spans: cython.pointer[Span] = rows.spans
    for i in range(count):
        spans[i].xa = chains[i].top
        spans[i].xb = 0.0
        spans[i].index = i
    _sort_spans(spans, count)
    for i in range(count):
        by_top[i] = spans[i].index
    work: cython.Py_ssize_t = 0
    budget: cython.Py_ssize_t = (
        SWEEP_WORK_PER_EDGE * rows.count + SWEEP_WORK_PER_ROW * height + 4096
    )
    added: cython.Py_ssize_t
    next_top: cython.Py_ssize_t = 0
    m = 0
    for e in range(event_count - 1):
        if not (events[e + 1] > events[e]):
            continue
        k = 0
        for i in range(m):
            if chains[band[i]].bottom > events[e]:
                band[k] = band[i]
                k += 1
        m = k
        added = 0
        while next_top < count and chains[by_top[next_top]].top <= events[e]:
            if chains[by_top[next_top]].bottom > events[e]:
                band[m] = by_top[next_top]
                m += 1
                added += 1
            next_top += 1
        if m == 0 or events[e + 1] <= 0.0 or events[e] >= height:
            continue
        work += m
        if work > budget:
            # Unchecked rows are not taken as safe: every row from here is supersampled, by
            # winding, at a cost bounded by its edges.
            _mark_span(
                rows.mixed, height, events[e], cython.cast(cython.double, height), SAMPLED_ROW
            )
            rows.exhausted = True
            break
        # Only the band's part over the window matters.
        ya = py_max(events[e], 0.0)
        yb = py_min(events[e + 1], cython.cast(cython.double, height))
        middle = (ya + yb) * 0.5
        for i in range(m):
            chain = chains + band[i]
            chain.xa = _chain_x(rows, chain, cython.address(chain.cursor), ya)
            chain.xb = _chain_x(rows, chain, cython.address(chain.bottom_cursor), middle)
        # The band's chains in order of x at its top, then at its middle (chains that meet at
        # the top part below it), then of index: the last band's order with the new chains
        # slotted in, or a fresh sort for many new ones.
        if added > SORT_SMALL:
            for i in range(m):
                spans[i].xa = chains[band[i]].xa
                spans[i].xb = chains[band[i]].xb
                spans[i].index = band[i]
            qsort(spans, m, cython.sizeof(Span), _compare_spans)
            for i in range(m):
                band[i] = spans[i].index
        for i in range(1, m):
            k = band[i]
            j = i
            while j > 0 and (
                chains[band[j - 1]].xa > chains[k].xa
                or (
                    chains[band[j - 1]].xa == chains[k].xa
                    and (
                        chains[band[j - 1]].xb > chains[k].xb
                        or (
                            chains[band[j - 1]].xb == chains[k].xb
                            and chains[band[j - 1]].index > chains[k].index
                        )
                    )
                )
            ):
                band[j] = band[j - 1]
                j -= 1
            band[j] = k
        bad = False
        sign = 0
        winding = 0
        for i in range(m - 1):
            winding += chains[band[i]].direction
            # Every region counts, even one whose chains meet at the band's ends: chains
            # need not be straight between them.
            if winding == 0:
                continue
            if winding > 1 or winding < -1 or (sign != 0 and winding != sign):
                bad = True
                break
            sign = winding
        if bad:
            _mark_span(rows.mixed, height, ya, yb)
            continue
        if sign > 0:
            _mark_span(rows.positive, height, ya, yb)
        elif sign < 0:
            _mark_span(rows.negative, height, ya, yb)
        # Neighbours must not cross: once two do, the regions between them change for the rest
        # of the band, from the earliest crossing of any pair.
        earliest = yb
        for i in range(m - 1):
            y = _pair_crossing(
                rows, chains + band[i], chains + band[i + 1], ya, yb, cython.address(work)
            )
            if y < earliest:
                earliest = y
        if earliest < yb:
            _mark_span(rows.mixed, height, earliest, yb)
        if work > budget:
            _mark_span(rows.mixed, height, ya, cython.cast(cython.double, height), SAMPLED_ROW)
            rows.exhausted = True
            break
    for i in range(height):
        if rows.positive[i] and rows.negative[i] and rows.mixed[i] == 0:
            rows.mixed[i] = 1


@cython.cfunc
@cython.exceptval(-1, check=False)
def _add_piece(
    rows: cython.pointer[Rows],
    edge: cython.pointer[RowEdge],
    band_top: cython.double,
    band_bottom: cython.double,
    direction: cython.double,
    count: cython.Py_ssize_t,
    bounds: cython.p_Py_ssize_t,
) -> cython.Py_ssize_t:
    """Add edge's pieces between band_top and band_bottom, turned to direction, to rows.acc
    (left weights) and rows.entries from count (right weights); the new count of entries.
    bounds holds the lowest and highest cell touched."""
    width: cython.Py_ssize_t = rows.width
    acc: cython.p_double = rows.acc
    entries: cython.pointer[Entry] = rows.entries
    grown: cython.pointer[Entry]
    column: cython.Py_ssize_t
    column_count: cython.Py_ssize_t
    column_offset: cython.Py_ssize_t
    row_height: cython.double = band_bottom - band_top
    entry_x: cython.double
    exit_x: cython.double
    left_x: cython.double
    right_x: cython.double
    walk_left: cython.double
    walk_right: cython.double
    floor_left: cython.double
    column_start: cython.double
    fragment_left: cython.double
    fragment_right: cython.double
    piece_span: cython.double
    share: cython.double
    fragment_height: cython.double
    midpoint: cython.double
    offset_in_cell: cython.double
    signed_height: cython.double
    vertical: cython.bint
    if row_height <= 0.0:
        return count
    entry_x = edge.top_x + (band_top - edge.top_y) * edge.slope
    exit_x = edge.top_x + (band_bottom - edge.top_y) * edge.slope
    left_x = py_min(exit_x, entry_x)
    right_x = py_max(exit_x, entry_x)
    walk_left = py_min(width + 1.0, py_max(-1.0, left_x))
    walk_right = py_min(width + 1.0, py_max(-1.0, right_x))
    floor_left = floor(walk_left)
    column_count = cython.cast(cython.Py_ssize_t, floor(walk_right) - floor_left + 1.0)
    piece_span = right_x - left_x
    vertical = piece_span <= 0.0
    for column_offset in range(column_count):
        column_start = floor_left + column_offset
        fragment_left = left_x if column_offset == 0 else column_start
        fragment_right = right_x if column_offset == column_count - 1 else column_start + 1.0
        if not (vertical or fragment_right > fragment_left):
            continue
        share = 1.0 if vertical else (fragment_right - fragment_left) / piece_span
        fragment_height = row_height * share
        midpoint = (fragment_left + fragment_right) * 0.5
        column = cython.cast(
            cython.Py_ssize_t,
            py_min(cython.cast(cython.double, width), py_max(0.0, floor(midpoint))),
        )
        offset_in_cell = py_min(1.0, py_max(0.0, midpoint - column))
        signed_height = direction * fragment_height
        if count + 1 > rows.entry_capacity:
            grown = cython.cast(
                cython.pointer[Entry], PyMem_Malloc(2 * rows.entry_capacity * cython.sizeof(Entry))
            )
            if grown == cython.NULL:
                raise MemoryError
            memcpy(grown, entries, count * cython.sizeof(Entry))
            PyMem_Free(entries)
            rows.entries = grown
            entries = grown
            rows.entry_capacity *= 2
        # The left weight goes into the cell now, the right weight into the next once every
        # left weight is in, the order the dense sums added them in.
        acc[column] += signed_height * (1.0 - offset_in_cell)
        entries[count].column = column + 1
        entries[count].value = signed_height * offset_in_cell
        count += 1
        if column < bounds[0]:
            bounds[0] = column
        if column + 1 > bounds[1]:
            bounds[1] = column + 1
    return count


@cython.cfunc
@cython.exceptval(-1, check=False)
def _emit_cells(
    rows: cython.pointer[Rows], count: cython.Py_ssize_t, bounds: cython.p_Py_ssize_t
) -> cython.int:
    """Add the right weights, then emit rows.acc's running sums as runs, zeroing it."""
    acc: cython.p_double = rows.acc
    width: cython.Py_ssize_t = rows.width
    low: cython.Py_ssize_t = bounds[0]
    high: cython.Py_ssize_t = bounds[1]
    last: cython.Py_ssize_t
    a: cython.Py_ssize_t
    column: cython.Py_ssize_t
    run_start: cython.Py_ssize_t
    running: cython.double = 0.0
    cell: cython.double
    coverage: cython.double = 0.0
    next_coverage: cython.double
    for a in range(count):
        acc[rows.entries[a].column] += rows.entries[a].value
    # Cells outside low..high are 0: the running sum is 0 before them and constant after.
    last = width - 1 if high >= width else high
    run_start = low
    # A cell no piece touched leaves the sum, and so the coverage, as it was: the run goes on.
    for column in range(low, last + 1):
        cell = acc[column]
        if cell != 0.0 or column == low:
            running += cell
            acc[column] = 0.0
            next_coverage = py_min(1.0, fabs(running))
            if column == low:
                coverage = next_coverage
            elif next_coverage != coverage:
                _emit(rows, run_start, column, coverage)
                run_start = column
                coverage = next_coverage
    if last >= low:
        _emit(rows, run_start, last + 1, coverage)
    for column in range(last + 1, high + 1):
        acc[column] = 0.0
    if last + 1 < width:
        _emit(rows, last + 1, width, py_min(1.0, fabs(running)))
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def _exact_row(rows: cython.pointer[Rows], r: cython.Py_ssize_t) -> cython.int:
    """Row r's runs measured exactly. The row's edges join into its chains' stretches; split
    into bands at the stretches' ends and at every crossing of two edges of different chains,
    each band's stretches keep their order, and those between a region inside the path and
    one outside it are its outline there: their edges' pieces in the band alone, each turned
    to add inside, give the coverage."""
    edges: cython.pointer[RowEdge] = rows.edges
    active: cython.p_Py_ssize_t = rows.active
    keep: cython.Py_ssize_t = rows.active_count
    stretches: cython.pointer[Chain] = rows.chains
    spans: cython.pointer[Span] = rows.spans
    order: cython.p_Py_ssize_t = rows.band
    top: cython.double = cython.cast(cython.double, r)
    bottom: cython.double = top + 1.0
    bounds = cython.declare(cython.Py_ssize_t[2])
    count: cython.Py_ssize_t = 0
    bands: cython.Py_ssize_t = 2
    m: cython.Py_ssize_t
    a: cython.Py_ssize_t
    j: cython.Py_ssize_t
    b: cython.Py_ssize_t
    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    entries: cython.Py_ssize_t = 0
    winding: cython.int
    inside: cython.bint
    ei: cython.pointer[RowEdge]
    ej: cython.pointer[RowEdge]
    edge: cython.pointer[RowEdge]
    stretch: cython.pointer[Chain]
    y: cython.double
    ya: cython.double
    yb: cython.double
    middle: cython.double
    xa: cython.double
    xb: cython.double
    low_y: cython.double
    high_y: cython.double
    _row_buffers(rows)
    # The row's stretches: runs of its active edges (in index order) of one chain.
    for a in range(keep):
        edge = edges + active[a]
        ya = py_max(top, edge.top_y)
        yb = py_min(bottom, edge.bottom_y)
        xa = edge.top_x + (ya - edge.top_y) * edge.slope
        xb = edge.top_x + (yb - edge.top_y) * edge.slope
        spans[a].xa = py_min(xa, xb)
        spans[a].xb = py_max(xa, xb)
        spans[a].index = a
        spans[a].direction = cython.cast(cython.int, count)
        if count and stretches[count - 1].index == edge.chain:
            stretch = stretches + count - 1
            stretch.top = py_min(stretch.top, ya)
            stretch.bottom = py_max(stretch.bottom, yb)
            stretch.last += 1
            spans[a].direction = cython.cast(cython.int, count - 1)
        else:
            stretch = stretches + count
            stretch.first = a
            stretch.last = 1
            stretch.top = ya
            stretch.bottom = yb
            stretch.direction = 1 if edge.direction > 0.0 else -1
            stretch.index = edge.chain
            count += 1
    _reserve_ys(rows, 2 * count + 2)
    rows.ys[0] = top
    rows.ys[1] = bottom
    for a in range(count):
        if stretches[a].top > top:
            rows.ys[bands] = stretches[a].top
            bands += 1
        if stretches[a].bottom < bottom:
            rows.ys[bands] = stretches[a].bottom
            bands += 1
    # Crossings of edges of different stretches, found among the edges whose x ranges in the
    # row overlap: sorted by the low end, each meets only those that start before it ends.
    _sort_spans(spans, keep)
    for a in range(keep):
        ei = edges + active[spans[a].index]
        for j in range(a + 1, keep):
            if spans[j].xa > spans[a].xb:
                break
            if spans[j].direction == spans[a].direction:
                continue
            ej = edges + active[spans[j].index]
            if ei.slope == ej.slope:
                continue
            y = (ej.top_x - ei.top_x + ei.top_y * ei.slope - ej.top_y * ej.slope) / (
                ei.slope - ej.slope
            )
            # A crossing at an edge's end counts too, where one stretch may touch another and
            # turn back; one worked out a hair outside the edges is taken at their end.
            low_y = py_max(ei.top_y, ej.top_y)
            high_y = py_min(ei.bottom_y, ej.bottom_y)
            if low_y - TOUCH <= y < low_y:
                y = low_y
            elif high_y < y <= high_y + TOUCH:
                y = high_y
            if top < y < bottom and low_y <= y <= high_y:
                if bands >= rows.ys_capacity:
                    _reserve_ys(rows, 2 * rows.ys_capacity + 2)
                rows.ys[bands] = y
                bands += 1
    if bands > 32:
        qsort(rows.ys, bands, cython.sizeof(cython.double), _compare_doubles)
    else:
        _sort_doubles(rows.ys, bands)
    bounds[0] = rows.width + 1
    bounds[1] = -1
    for b in range(bands - 1):
        ya = rows.ys[b]
        yb = rows.ys[b + 1]
        if not (yb > ya):
            continue
        middle = (ya + yb) * 0.5
        m = 0
        for a in range(count):
            stretch = stretches + a
            if stretch.top <= ya and stretch.bottom >= yb:
                stretch.xa = _stretch_middle(rows, stretch, middle)
                order[m] = a
                m += 1
        if m > SORT_SMALL:
            for i in range(m):
                spans[i].xa = stretches[order[i]].xa
                spans[i].xb = cython.cast(cython.double, stretches[order[i]].index)
                spans[i].index = order[i]
            qsort(spans, m, cython.sizeof(Span), _compare_spans)
            for i in range(m):
                order[i] = spans[i].index
        else:
            for i in range(1, m):
                k = order[i]
                j = i
                while j > 0 and (
                    stretches[order[j - 1]].xa > stretches[k].xa
                    or (
                        stretches[order[j - 1]].xa == stretches[k].xa
                        and stretches[order[j - 1]].index > stretches[k].index
                    )
                ):
                    order[j] = order[j - 1]
                    j -= 1
                order[j] = k
        winding = 0
        for i in range(m):
            stretch = stretches + order[i]
            inside = winding != 0
            winding += stretch.direction
            if (winding != 0) == inside:
                continue
            for a in range(stretch.first, stretch.first + stretch.last):
                edge = edges + active[a]
                entries = _add_piece(
                    rows,
                    edge,
                    py_max(ya, edge.top_y),
                    py_min(yb, edge.bottom_y),
                    -1.0 if inside else 1.0,
                    entries,
                    cython.address(bounds[0]),
                )
    rows.exact_rows += 1
    return _emit_cells(rows, entries, cython.address(bounds[0]))


@cython.cfunc
@cython.exceptval(check=False)
def _stretch_middle(
    rows: cython.pointer[Rows], stretch: cython.pointer[Chain], y: cython.double
) -> cython.double:
    """The x of stretch at y, from the first of its edges that reaches y."""
    edge: cython.pointer[RowEdge]
    a: cython.Py_ssize_t
    for a in range(stretch.first, stretch.first + stretch.last):
        edge = rows.edges + rows.active[a]
        if edge.top_y <= y <= edge.bottom_y:
            return edge.top_x + (y - edge.top_y) * edge.slope
    edge = rows.edges + rows.active[stretch.first]
    return edge.top_x + (y - edge.top_y) * edge.slope


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _compare_doubles(
    left: cython.pointer[cython.const[cython.void]],
    right: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    a: cython.double = cython.cast(cython.pointer[cython.const[cython.double]], left)[0]
    b: cython.double = cython.cast(cython.pointer[cython.const[cython.double]], right)[0]
    return (a > b) - (a < b)


@cython.cfunc
@cython.exceptval(-1, check=False)
def _sum_row(rows: cython.pointer[Rows], r: cython.Py_ssize_t) -> cython.int:
    """Row r's runs from the sums of its edges' pieces, as the dense sums gave them."""
    edges: cython.pointer[RowEdge] = rows.edges
    bounds = cython.declare(cython.Py_ssize_t[2])
    entries: cython.Py_ssize_t = 0
    a: cython.Py_ssize_t
    edge: cython.pointer[RowEdge]
    _row_buffers(rows)
    bounds[0] = rows.width + 1
    bounds[1] = -1
    for a in range(rows.active_count):
        edge = edges + rows.active[a]
        entries = _add_piece(
            rows,
            edge,
            py_max(cython.cast(cython.double, r), edge.top_y),
            py_min(r + 1.0, edge.bottom_y),
            edge.direction,
            entries,
            cython.address(bounds[0]),
        )
    return _emit_cells(rows, entries, cython.address(bounds[0]))


@cython.cfunc
@cython.exceptval(-1, check=False)
def _sampled_row(rows: cython.pointer[Rows], r: cython.Py_ssize_t) -> cython.int:
    """Row r's runs from a ROW_SAMPLES x ROW_SAMPLES grid of samples per pixel at the centres
    of its cells, each inside where the nonzero winding of the active edges crossing its
    line left of it is: correct for any windings. Each line's crossings are the last line's,
    still crossing, then the edges that start between them, put in order of x by insertion
    from the last line's order, or sorted afresh when many are new."""
    edges: cython.pointer[RowEdge] = rows.edges
    spans: cython.pointer[Span] = rows.sampled
    width: cython.Py_ssize_t = rows.width
    columns: cython.Py_ssize_t = width * ROW_SAMPLES
    samples: cython.double = cython.cast(cython.double, ROW_SAMPLES)
    unit: cython.double = 1.0 / (samples * samples)
    acc: cython.p_double
    edge: cython.pointer[RowEdge]
    line: cython.Py_ssize_t
    a: cython.Py_ssize_t
    k: cython.Py_ssize_t
    count: cython.Py_ssize_t
    added: cython.Py_ssize_t
    winding: cython.int
    first: cython.Py_ssize_t
    end: cython.Py_ssize_t
    column: cython.Py_ssize_t
    y: cython.double
    previous: cython.double
    span: Span
    _row_buffers(rows)
    acc = rows.acc
    for line in range(ROW_SAMPLES):
        y = r + (line + 0.5) / samples
        previous = rows.sampled_y
        count = 0
        for a in range(rows.sampled_count):
            edge = edges + spans[a].index
            if edge.top_y <= y < edge.bottom_y:
                spans[count].index = spans[a].index
                spans[count].direction = spans[a].direction
                spans[count].xa = edge.top_x + (y - edge.top_y) * edge.slope
                spans[count].xb = 0.0
                count += 1
        added = 0
        for a in range(rows.active_count):
            edge = edges + rows.active[a]
            if previous < edge.top_y <= y < edge.bottom_y:
                spans[count].xa = edge.top_x + (y - edge.top_y) * edge.slope
                spans[count].xb = 0.0
                spans[count].direction = cython.cast(cython.int, edge.direction)
                spans[count].index = rows.active[a]
                count += 1
                added += 1
        if added > SORT_SMALL:
            _sort_spans(spans, count)
        else:
            for a in range(1, count):
                span = spans[a]
                k = a
                while k > 0 and (
                    spans[k - 1].xa > span.xa
                    or (spans[k - 1].xa == span.xa and spans[k - 1].index > span.index)
                ):
                    spans[k] = spans[k - 1]
                    k -= 1
                spans[k] = span
        rows.sampled_count = count
        rows.sampled_y = y
        winding = 0
        for a in range(count - 1):
            winding += spans[a].direction
            if winding == 0:
                continue
            # The columns of samples whose centres lie in xa..next xa.
            first = cython.cast(
                cython.Py_ssize_t, py_max(py_min(ceil(spans[a].xa * samples - 0.5), columns), 0.0)
            )
            end = cython.cast(
                cython.Py_ssize_t,
                py_max(py_min(ceil(spans[a + 1].xa * samples - 0.5), columns), 0.0),
            )
            for column in range(first, end):
                acc[column // ROW_SAMPLES] += unit
    for column in range(width):
        _emit(rows, column, column + 1, py_min(1.0, acc[column]))
        acc[column] = 0.0
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def _rows_next(rows: cython.pointer[Rows], r: cython.Py_ssize_t, emit: cython.bint) -> cython.int:
    """Row r's runs in rows.runs, rows rising, and 0; or 1 when emit is false and the row is
    the dense sums', in rows.cells, for the caller to read."""
    rows.run_count = 0
    if rows.mixed[r] or not rows.dense:
        # Only these rows read the active list, which catches up with the rows skipped.
        _order_edges(rows)
        if _advance(rows, r) == 0:
            return 0
        # A row of many edges is supersampled too: its exact measure looks for crossings
        # among every pair of them.
        if rows.mixed[r] == SAMPLED_ROW or (rows.mixed[r] and rows.active_count > EXACT_EDGES):
            return _sampled_row(rows, r)
        if rows.mixed[r]:
            return _exact_row(rows, r)
        return _sum_row(rows, r)
    if not emit:
        return 1
    width: cython.Py_ssize_t = rows.width
    cells: cython.p_double = rows.cells + r * (width + 2)
    running: cython.double = 0.0
    column: cython.Py_ssize_t
    for column in range(width):
        running += cells[column]
        _emit(rows, column, column + 1, py_min(1.0, fabs(running)))
    return 0


def windings_mix(edges) -> bool:
    """Whether the path of these edges has, anywhere, regions of winding beyond 1 or of both
    signs, or neighbouring chains that cross, as _mark_rows finds them with the outline's
    whole height in one row; True too when the sweep ran out of work. False means the
    windings are 0 or one sign of 1 everywhere, which no invertible placement changes."""
    array = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(array)
    if kept == 0:
        return False
    view: cython.p_const_double = edge_data(array)
    stride: cython.Py_ssize_t = edge_width(array)
    i: cython.Py_ssize_t
    low: cython.double = view[1]
    high: cython.double = view[1]
    for i in range(edge_count(array)):
        low = py_min(low, py_min(view[i * stride + 1], view[i * stride + 3]))
        high = py_max(high, py_max(view[i * stride + 1], view[i * stride + 3]))
    if not (high > low) or high - low - (high - low) != 0.0:
        return True
    scale: cython.double = OUTLINE_ROWS / (high - low)
    device_stack = cython.declare(cython.double[STACK_EDGE_VALUES])
    device: cython.p_double = _device_edges(
        array, kept, 0.0, high, scale, 0.0, 0.0, device_stack, STACK_EDGES
    )
    rows = cython.declare(Rows)
    r: cython.Py_ssize_t
    mixed: cython.bint = False
    _rows_clear(cython.address(rows))
    try:
        # The outline's whole height maps to one row, so every band's sign lands in it and
        # _mark_rows marks it mixed for bands of both signs: one sign over all the outline,
        # as a rotated placement needs, not only within each row of one placement.
        _rows_begin(cython.address(rows), device, kept, 1, OUTLINE_ROWS, cython.NULL, 0, 2)
        mixed = rows.exhausted
        for r in range(OUTLINE_ROWS):
            if rows.mixed[r]:
                mixed = True
    finally:
        _rows_free(cython.address(rows))
        if device != device_stack:
            PyMem_Free(device)
    return mixed


def area_coverage_plane(
    edges,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.double,
    iy0: cython.double,
    width: cython.int,
    height: cython.int,
    mode: cython.int = 0,
):
    """The coverage fill_glyph_coverage paints, as a height x width plane of doubles, and the
    number of rows measured exactly; None when no edge slopes. mode 1 sums the window
    densely and 2 row by row, as a small and a large window are, where each can be had."""
    rows_array = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows_array)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64), 0
    device: cython.p_double = _device_edges(rows_array, kept, crop_x0, crop_y1, scale, ix0, iy0)
    try:
        return _engine_plane(device, kept, width, height, mode)
    finally:
        PyMem_Free(device)


@cython.cfunc
def _engine_plane(
    device: cython.p_const_double,
    kept: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    mode: cython.int,
) -> object:
    plane = numpy.zeros((height, width), numpy.float64)
    out: cython.double[:, ::1] = plane
    rows = cython.declare(Rows)
    r: cython.Py_ssize_t
    k: cython.Py_ssize_t
    c: cython.Py_ssize_t
    exact: cython.Py_ssize_t = 0
    _rows_clear(cython.address(rows))
    try:
        _rows_begin(cython.address(rows), device, kept, width, height, cython.NULL, 0, mode)
        for r in range(height):
            _rows_next(cython.address(rows), r, True)
            for k in range(rows.run_count):
                for c in range(rows.runs[k].start, rows.runs[k].end):
                    out[r, c] = rows.runs[k].coverage
        exact = rows.exact_rows
    finally:
        _rows_free(cython.address(rows))
    return plane, exact


def fill_glyph_coverage(
    edges,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    rgba,
    target: cython.uchar[:, :, :],
    source_alpha: typing.Optional[cython.float[:, :]],
    source_shape: typing.Optional[cython.float[:, :]],
    shape_scale: cython.double,
):
    rows = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return True
    if target.shape[0] != height or target.shape[1] != width or target.shape[2] != 4:
        raise ValueError("target differs from the plane in shape")
    has_alpha: cython.bint = source_alpha is not None
    has_shape: cython.bint = source_shape is not None
    if has_alpha and (source_alpha.shape[0] != height or source_alpha.shape[1] != width):
        raise ValueError("source_alpha differs from the plane in shape")
    if has_shape and (source_shape.shape[0] != height or source_shape.shape[1] != width):
        raise ValueError("source_shape differs from the plane in shape")
    pixels: BytePlane = BytePlane(
        cython.address(target[0, 0, 0]), target.strides[0], target.strides[1], target.strides[2]
    )
    alpha_plane: FloatPlane = FloatPlane(cython.NULL, 0, 0)
    shape_plane: FloatPlane = FloatPlane(cython.NULL, 0, 0)
    if has_alpha:
        alpha_plane = FloatPlane(
            cython.address(source_alpha[0, 0]), source_alpha.strides[0], source_alpha.strides[1]
        )
    if has_shape:
        shape_plane = FloatPlane(
            cython.address(source_shape[0, 0]), source_shape.strides[0], source_shape.strides[1]
        )
    _coverage_fill(
        rows,
        kept,
        crop_x0,
        crop_y1,
        scale,
        ix0,
        iy0,
        width,
        height,
        rgba,
        pixels,
        alpha_plane,
        shape_plane,
        shape_scale,
    )
    return True


def fill_glyph_coverage_at(
    edges,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    rgba,
    target,
    source_alpha,
    source_shape,
    shape_scale: cython.double,
):
    """fill_glyph_coverage over whole planes, painting the width x height window at (ix0, iy0)."""
    pixels = cython.declare(BytePlane)
    alpha_plane: FloatPlane = FloatPlane(cython.NULL, 0, 0)
    shape_plane: FloatPlane = FloatPlane(cython.NULL, 0, 0)
    if (
        height <= 0
        or width <= 0
        or not (
            _byte_plane(target, True, iy0, ix0, height, width, cython.address(pixels))
            and (
                source_alpha is None
                or _float_plane(source_alpha, iy0, ix0, height, width, cython.address(alpha_plane))
            )
            and (
                source_shape is None
                or _float_plane(source_shape, iy0, ix0, height, width, cython.address(shape_plane))
            )
        )
    ):
        rows_window = slice(iy0, iy0 + height)
        columns_window = slice(ix0, ix0 + width)
        return fill_glyph_coverage(
            edges,
            crop_x0,
            crop_y1,
            scale,
            ix0,
            iy0,
            width,
            height,
            rgba,
            target[rows_window, columns_window],
            None if source_alpha is None else source_alpha[rows_window, columns_window],
            None if source_shape is None else source_shape[rows_window, columns_window],
            shape_scale,
        )
    rows = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows)
    if kept == 0:
        return None
    _coverage_fill(
        rows,
        kept,
        crop_x0,
        crop_y1,
        scale,
        ix0,
        iy0,
        width,
        height,
        rgba,
        pixels,
        alpha_plane,
        shape_plane,
        shape_scale,
    )
    return True


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def _paint_coverage(
    pixels: BytePlane,
    alpha_plane: FloatPlane,
    shape_plane: FloatPlane,
    r: cython.Py_ssize_t,
    c: cython.Py_ssize_t,
    raw: cython.uchar,
    shape: cython.uchar,
    cap: cython.int,
    opaque_from: cython.int,
    red: cython.float,
    green: cython.float,
    blue: cython.float,
    opaque_red: cython.uchar,
    opaque_green: cython.uchar,
    opaque_blue: cython.uchar,
    has_alpha: cython.bint,
    has_shape: cython.bint,
    shape_scale: cython.double,
) -> cython.void:
    """One pixel of a fill at alpha raw, and the source planes at raw and shape."""
    channel: cython.Py_ssize_t = pixels.channel
    pixel: cython.p_uchar
    cell: cython.p_float
    if raw != 0:
        pixel = pixels.data + r * pixels.row + c * pixels.column
        if raw >= opaque_from:
            pixel[0] = opaque_red
            pixel[channel] = opaque_green
            pixel[2 * channel] = opaque_blue
            pixel[3 * channel] = 255
        else:
            blend_one(
                cython.address(pixel[0]),
                cython.address(pixel[channel]),
                cython.address(pixel[2 * channel]),
                cython.address(pixel[3 * channel]),
                raw,
                cap,
                red,
                green,
                blue,
            )
    if has_alpha:
        cell = cython.cast(
            cython.p_float,
            cython.cast(cython.p_char, alpha_plane.data)
            + r * alpha_plane.row
            + c * alpha_plane.column,
        )
        cell[0] = accumulate_plane(cell[0], raw, 1.0)
    if has_shape:
        cell = cython.cast(
            cython.p_float,
            cython.cast(cython.p_char, shape_plane.data)
            + r * shape_plane.row
            + c * shape_plane.column,
        )
        cell[0] = accumulate_plane(cell[0], shape, shape_scale)


@cython.cfunc
@cython.exceptval(check=False)
def _dense_ready(device: cython.p_const_double, kept: cython.Py_ssize_t) -> cython.bint:
    """Whether the dense sums take these device edges as the engine's table would: every
    coordinate finite, every y near enough to cast, and every sloped edge's slope finite."""
    i: cython.Py_ssize_t
    sx: cython.double
    sy: cython.double
    ex: cython.double
    ey: cython.double
    slope: cython.double
    for i in range(kept):
        sx = device[i * 4]
        sy = device[i * 4 + 1]
        ex = device[i * 4 + 2]
        ey = device[i * 4 + 3]
        if not (
            -DENSE_REACH <= sy <= DENSE_REACH
            and -DENSE_REACH <= ey <= DENSE_REACH
            and sx - sx == 0.0
            and ex - ex == 0.0
        ):
            return False
        if sy != ey:
            slope = (ex - sx) / (ey - sy)
            if slope - slope != 0.0:
                return False
    return True


@cython.cfunc
@cython.exceptval(-1, check=False)
def _coverage_fill(
    rows: object,
    kept: cython.Py_ssize_t,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    rgba: object,
    pixels: BytePlane,
    alpha_plane: FloatPlane,
    shape_plane: FloatPlane,
    shape_scale: cython.double,
) -> cython.int:
    has_alpha: cython.bint = alpha_plane.data != cython.NULL
    has_shape: cython.bint = shape_plane.data != cython.NULL
    red: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[0]))
    green: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[1]))
    blue: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[2]))
    cap: cython.int = cython.cast(cython.int, rgba[3])
    alpha: cython.double = cython.cast(cython.double, rgba[3])
    opaque_from: cython.int = 255 if cap >= 255 else 256
    opaque_red: cython.uchar = opaque_channel(red)
    opaque_green: cython.uchar = opaque_channel(green)
    opaque_blue: cython.uchar = opaque_channel(blue)
    device_stack = cython.declare(cython.double[STACK_EDGE_VALUES])
    device: cython.p_double = _device_edges(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, device_stack, STACK_EDGES
    )
    engine = cython.declare(Rows)
    cells_stack = cython.declare(cython.double[STACK_CELLS])
    k: cython.Py_ssize_t
    row_cells: cython.p_double
    running: cython.double
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    coverage: cython.double
    raw: cython.uchar
    shape: cython.uchar
    channel: cython.Py_ssize_t = pixels.channel
    pixel: cython.p_uchar
    cell: cython.p_float
    simple: cython.bint = type(rows) is SimpleEdges
    acc: cython.p_double = cython.NULL
    # An outline whose windings never mix needs no edge table: the dense sums, from the
    # device edges, as main summed them, when the engine would keep the same edges.
    simple = (
        simple
        and (cython.cast(cython.Py_ssize_t, width) + 2) * height <= DENSE_CELLS
        and _dense_ready(device, kept)
    )
    if not simple:
        _rows_clear(cython.address(engine))
    try:
        if simple:
            acc = _zeroed_cells(
                (cython.cast(cython.Py_ssize_t, width) + 2) * height, cells_stack, STACK_CELLS
            )
            _accumulate_device(device, kept, width, height, acc)
        else:
            _rows_begin(
                cython.address(engine),
                device,
                kept,
                width,
                height,
                cells_stack,
                STACK_CELLS,
                SIMPLE_OUTLINE if type(rows) is SimpleEdges else 0,
            )
        for r in range(height):
            if simple or _rows_next(cython.address(engine), r, False):
                row_cells = (acc if simple else engine.cells) + r * (width + 2)
                running = 0.0
                # The per-pixel loop main ran over the dense sums.
                for c in range(width):
                    running += row_cells[c]
                    coverage = py_min(1.0, fabs(running))
                    raw = cython.cast(cython.uchar, rint(coverage * alpha))
                    if raw != 0:
                        pixel = pixels.data + r * pixels.row + c * pixels.column
                        if raw >= opaque_from:
                            pixel[0] = opaque_red
                            pixel[channel] = opaque_green
                            pixel[2 * channel] = opaque_blue
                            pixel[3 * channel] = 255
                        else:
                            blend_one(
                                cython.address(pixel[0]),
                                cython.address(pixel[channel]),
                                cython.address(pixel[2 * channel]),
                                cython.address(pixel[3 * channel]),
                                raw,
                                cap,
                                red,
                                green,
                                blue,
                            )
                    if has_alpha:
                        cell = cython.cast(
                            cython.p_float,
                            cython.cast(cython.p_char, alpha_plane.data)
                            + r * alpha_plane.row
                            + c * alpha_plane.column,
                        )
                        cell[0] = accumulate_plane(cell[0], raw, 1.0)
                    if has_shape:
                        shape = cython.cast(cython.uchar, rint(coverage * 255.0))
                        cell = cython.cast(
                            cython.p_float,
                            cython.cast(cython.p_char, shape_plane.data)
                            + r * shape_plane.row
                            + c * shape_plane.column,
                        )
                        cell[0] = accumulate_plane(cell[0], shape, shape_scale)
                continue
            for k in range(engine.run_count):
                coverage = engine.runs[k].coverage
                raw = cython.cast(cython.uchar, rint(coverage * alpha))
                shape = cython.cast(cython.uchar, rint(coverage * 255.0)) if has_shape else 0
                # A pixel this run leaves at zero alpha and shape keeps every plane as it is.
                if raw == 0 and shape == 0:
                    continue
                for c in range(engine.runs[k].start, engine.runs[k].end):
                    _paint_coverage(
                        pixels,
                        alpha_plane,
                        shape_plane,
                        r,
                        c,
                        raw,
                        shape,
                        cap,
                        opaque_from,
                        red,
                        green,
                        blue,
                        opaque_red,
                        opaque_green,
                        opaque_blue,
                        has_alpha,
                        has_shape,
                        shape_scale,
                    )
    finally:
        if device != device_stack:
            PyMem_Free(device)
        if acc != cells_stack:
            PyMem_Free(acc)
        if not simple:
            _rows_free(cython.address(engine))
    return 0


def fill_glyph_knockout(
    edges,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    rgba,
    destination: cython.uchar[:, :, :],
    backdrop: cython.const[cython.uchar][:, :, :],
    group_alpha: cython.float[:, :],
    parent_shape: typing.Optional[cython.float[:, :]],
    shape_scale: cython.double,
):
    rows = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return True
    if destination.shape[0] != height or destination.shape[1] != width or destination.shape[2] != 4:
        raise ValueError("destination differs from the plane in shape")
    if backdrop.shape[0] != height or backdrop.shape[1] != width or backdrop.shape[2] != 4:
        raise ValueError("backdrop differs from the plane in shape")
    if group_alpha.shape[0] != height or group_alpha.shape[1] != width:
        raise ValueError("group_alpha differs from the plane in shape")
    has_parent_shape: cython.bint = parent_shape is not None
    if has_parent_shape and (parent_shape.shape[0] != height or parent_shape.shape[1] != width):
        raise ValueError("parent_shape differs from the plane in shape")
    target: BytePlane = BytePlane(
        cython.address(destination[0, 0, 0]),
        destination.strides[0],
        destination.strides[1],
        destination.strides[2],
    )
    base: BytePlane = BytePlane(
        cython.cast(cython.p_uchar, cython.address(backdrop[0, 0, 0])),
        backdrop.strides[0],
        backdrop.strides[1],
        backdrop.strides[2],
    )
    alpha_plane: FloatPlane = FloatPlane(
        cython.address(group_alpha[0, 0]), group_alpha.strides[0], group_alpha.strides[1]
    )
    shape_plane: FloatPlane = FloatPlane(cython.NULL, 0, 0)
    if has_parent_shape:
        shape_plane = FloatPlane(
            cython.address(parent_shape[0, 0]), parent_shape.strides[0], parent_shape.strides[1]
        )
    _knockout_fill(
        rows,
        kept,
        crop_x0,
        crop_y1,
        scale,
        ix0,
        iy0,
        width,
        height,
        rgba,
        target,
        base,
        alpha_plane,
        shape_plane,
        shape_scale,
    )
    return True


@cython.cfunc
@cython.exceptval(check=False)
def _byte_plane(
    plane: object,
    writable: cython.bint,
    y0: cython.Py_ssize_t,
    x0: cython.Py_ssize_t,
    height: cython.int,
    width: cython.int,
    out: cython.pointer[BytePlane],
) -> cython.bint:
    if not cnp.PyArray_Check(plane):
        return False
    array: cnp.ndarray = cython.cast(cnp.ndarray, plane)
    if (
        cnp.PyArray_TYPE(array) != cnp.NPY_UINT8
        or cnp.PyArray_NDIM(array) != 3
        or cnp.PyArray_DIM(array, 2) != 4
        or (writable and not cnp.PyArray_ISWRITEABLE(array))
        or y0 < 0
        or x0 < 0
        or y0 + height > cnp.PyArray_DIM(array, 0)
        or x0 + width > cnp.PyArray_DIM(array, 1)
    ):
        return False
    out.row = cnp.PyArray_STRIDE(array, 0)
    out.column = cnp.PyArray_STRIDE(array, 1)
    out.channel = cnp.PyArray_STRIDE(array, 2)
    out.data = cython.cast(cython.p_uchar, cnp.PyArray_DATA(array)) + y0 * out.row + x0 * out.column
    return True


@cython.cfunc
@cython.exceptval(check=False)
def _float_plane(
    plane: object,
    y0: cython.Py_ssize_t,
    x0: cython.Py_ssize_t,
    height: cython.int,
    width: cython.int,
    out: cython.pointer[FloatPlane],
) -> cython.bint:
    if not cnp.PyArray_Check(plane):
        return False
    array: cnp.ndarray = cython.cast(cnp.ndarray, plane)
    if (
        cnp.PyArray_TYPE(array) != cnp.NPY_FLOAT32
        or cnp.PyArray_NDIM(array) != 2
        or not cnp.PyArray_ISWRITEABLE(array)
        or not cnp.PyArray_ISALIGNED(array)
        or not cnp.PyArray_ISNOTSWAPPED(array)
        or y0 < 0
        or x0 < 0
        or y0 + height > cnp.PyArray_DIM(array, 0)
        or x0 + width > cnp.PyArray_DIM(array, 1)
    ):
        return False
    out.row = cnp.PyArray_STRIDE(array, 0)
    out.column = cnp.PyArray_STRIDE(array, 1)
    out.data = cython.cast(
        cython.p_float,
        cython.cast(cython.p_char, cnp.PyArray_DATA(array)) + y0 * out.row + x0 * out.column,
    )
    return True


def fill_glyph_knockout_at(
    edges,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    rgba,
    destination,
    backdrop,
    group_alpha,
    parent_shape,
    shape_scale: cython.double,
):
    """fill_glyph_knockout over whole planes, painting the width x height window at (ix0, iy0)."""
    target = cython.declare(BytePlane)
    base = cython.declare(BytePlane)
    alpha_plane = cython.declare(FloatPlane)
    shape_plane: FloatPlane = FloatPlane(cython.NULL, 0, 0)
    if (
        height <= 0
        or width <= 0
        or not (
            _byte_plane(destination, True, iy0, ix0, height, width, cython.address(target))
            and _byte_plane(backdrop, False, iy0, ix0, height, width, cython.address(base))
            and _float_plane(group_alpha, iy0, ix0, height, width, cython.address(alpha_plane))
            and (
                parent_shape is None
                or _float_plane(parent_shape, iy0, ix0, height, width, cython.address(shape_plane))
            )
        )
    ):
        rows_window = slice(iy0, iy0 + height)
        columns_window = slice(ix0, ix0 + width)
        return fill_glyph_knockout(
            edges,
            crop_x0,
            crop_y1,
            scale,
            ix0,
            iy0,
            width,
            height,
            rgba,
            destination[rows_window, columns_window],
            backdrop[rows_window, columns_window],
            group_alpha[rows_window, columns_window],
            None if parent_shape is None else parent_shape[rows_window, columns_window],
            shape_scale,
        )
    rows = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows)
    if kept == 0:
        return None
    _knockout_fill(
        rows,
        kept,
        crop_x0,
        crop_y1,
        scale,
        ix0,
        iy0,
        width,
        height,
        rgba,
        target,
        base,
        alpha_plane,
        shape_plane,
        shape_scale,
    )
    return True


@cython.cfunc
@cython.exceptval(-1, check=False)
def _knockout_fill(
    rows: object,
    kept: cython.Py_ssize_t,
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    rgba: object,
    target: BytePlane,
    base: BytePlane,
    alpha_plane: FloatPlane,
    shape_plane: FloatPlane,
    shape_scale: cython.double,
) -> cython.int:

    has_parent_shape: cython.bint = shape_plane.data != cython.NULL
    red: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[0]))
    green: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[1]))
    blue: cython.float = cython.cast(cython.float, cython.cast(cython.int, rgba[2]))
    cap: cython.int = cython.cast(cython.int, rgba[3])
    alpha: cython.double = cython.cast(cython.double, rgba[3])
    opaque_from: cython.int = 255 if cap >= 255 else 256
    opaque_red: cython.uchar = opaque_channel(red)
    opaque_green: cython.uchar = opaque_channel(green)
    opaque_blue: cython.uchar = opaque_channel(blue)
    device_stack = cython.declare(cython.double[STACK_EDGE_VALUES])
    device: cython.p_double = _device_edges(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, device_stack, STACK_EDGES
    )
    engine = cython.declare(Rows)
    cells_stack = cython.declare(cython.double[STACK_CELLS])
    run: cython.Py_ssize_t
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    k: cython.Py_ssize_t
    coverage: cython.double
    scaled: cython.double
    eff: cython.double
    sh: cython.double
    remaining: cython.double
    rga: cython.double
    ra: cython.double
    complete: cython.double
    initial: cython.double
    ec: cython.double
    colour = cython.declare(cython.double[3])
    raw: cython.uchar
    shape_byte: cython.uchar
    quantized: cython.uchar
    rendered = cython.declare(cython.uchar[4])
    element: cython.p_const_uchar
    destination: cython.p_uchar
    backdrop: cython.p_const_uchar
    group_alpha: cython.p_float
    parent_shape: cython.p_float
    ZERO: cython.float = 0.0
    ONE: cython.float = 1.0
    source_alpha: cython.float
    source_shape: cython.float
    previous: cython.float
    # As in _coverage_fill: an outline whose windings never mix is summed densely from the
    # device edges, each pixel a run of its own, when the engine would keep the same edges.
    simple: cython.bint = (
        type(rows) is SimpleEdges
        and (cython.cast(cython.Py_ssize_t, width) + 2) * height <= DENSE_CELLS
        and _dense_ready(device, kept)
    )
    acc: cython.p_double = cython.NULL
    row_cells: cython.p_double
    running: cython.double = 0.0
    segments: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    _rows_clear(cython.address(engine))
    try:
        if simple:
            acc = _zeroed_cells(
                (cython.cast(cython.Py_ssize_t, width) + 2) * height, cells_stack, STACK_CELLS
            )
            _accumulate_device(device, kept, width, height, acc)
        else:
            _rows_begin(
                cython.address(engine),
                device,
                kept,
                width,
                height,
                cells_stack,
                STACK_CELLS,
                SIMPLE_OUTLINE if type(rows) is SimpleEdges else 0,
            )
        for r in range(height):
            if simple:
                row_cells = acc + r * (width + 2)
                running = 0.0
                segments = width
            else:
                _rows_next(cython.address(engine), r, True)
                segments = engine.run_count
            for run in range(segments):
                if simple:
                    running += row_cells[run]
                    coverage = py_min(1.0, fabs(running))
                    start = run
                    end = run + 1
                else:
                    coverage = engine.runs[run].coverage
                    start = engine.runs[run].start
                    end = engine.runs[run].end
                # A pixel this run leaves at zero alpha and shape is left as it is.
                if (
                    cython.cast(cython.uchar, rint(coverage * alpha)) == 0
                    and cython.cast(cython.uchar, rint(coverage * 255.0)) == 0
                ):
                    continue
                for c in range(start, end):
                    destination = target.data + r * target.row + c * target.column
                    backdrop = base.data + r * base.row + c * base.column
                    group_alpha = cython.cast(
                        cython.p_float,
                        cython.cast(cython.p_char, alpha_plane.data)
                        + r * alpha_plane.row
                        + c * alpha_plane.column,
                    )
                    raw = cython.cast(cython.uchar, rint(coverage * alpha))
                    source_alpha = accumulate_plane(ZERO, raw, 1.0)
                    shape_byte = cython.cast(cython.uchar, rint(coverage * 255.0))
                    source_shape = accumulate_plane(ZERO, shape_byte, shape_scale)
                    scaled = rint(cython.cast(cython.double, source_alpha) * 255.0)
                    quantized = cython.cast(cython.uchar, cython.cast(cython.int, scaled))
                    if quantized > 0:
                        for k in range(4):
                            rendered[k] = backdrop[k * base.channel]
                        if raw != 0:
                            if raw >= opaque_from:
                                rendered[0] = opaque_red
                                rendered[1] = opaque_green
                                rendered[2] = opaque_blue
                                rendered[3] = 255
                            else:
                                blend_one(
                                    cython.address(rendered[0]),
                                    cython.address(rendered[1]),
                                    cython.address(rendered[2]),
                                    cython.address(rendered[3]),
                                    raw,
                                    cap,
                                    red,
                                    green,
                                    blue,
                                )
                        element = cython.address(rendered[0])
                    else:
                        for k in range(4):
                            rendered[k] = backdrop[k * base.channel]
                        element = cython.address(rendered[0])
                    eff = UNIT_FROM_BYTE[quantized]
                    sh = cython.cast(cython.double, source_shape)
                    if sh < 0.0:
                        sh = 0.0
                    elif sh > 1.0:
                        sh = 1.0
                    if eff > sh:
                        sh = eff
                    if sh > 0.0:
                        complete = UNIT_FROM_BYTE[destination[3 * target.channel]]
                        initial = UNIT_FROM_BYTE[backdrop[3 * base.channel]]
                        ec = UNIT_FROM_BYTE[element[3]]
                        remaining = 1.0 - sh
                        rga = eff + remaining * cython.cast(cython.double, group_alpha[0])
                        ra = initial + (1.0 - initial) * rga
                        for k in range(3):
                            colour[k] = knockout_component(
                                UNIT_FROM_BYTE[element[k]],
                                ec,
                                remaining,
                                UNIT_FROM_BYTE[destination[k * target.channel]],
                                complete,
                                UNIT_FROM_BYTE[backdrop[k * base.channel]],
                                initial,
                                ra,
                            )
                        for k in range(3):
                            destination[k * target.channel] = cython.cast(
                                cython.uchar, unit_to_byte(colour[k])
                            )
                        destination[3 * target.channel] = cython.cast(
                            cython.uchar, unit_to_byte(ra)
                        )
                        group_alpha[0] = cython.cast(cython.float, rga)
                    if has_parent_shape:
                        parent_shape = cython.cast(
                            cython.p_float,
                            cython.cast(cython.p_char, shape_plane.data)
                            + r * shape_plane.row
                            + c * shape_plane.column,
                        )
                        previous = parent_shape[0]
                        parent_shape[0] = previous + (ONE - previous) * source_shape
    finally:
        if device != device_stack:
            PyMem_Free(device)
        if acc != cython.NULL and acc != cells_stack:
            PyMem_Free(acc)
        _rows_free(cython.address(engine))
    return 0
