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
    SCRATCH_DOUBLES,
    SCRATCH_EDGES,
    SCRATCH_ENTRIES,
    SCRATCH_RUNS,
    STACK_EDGE_VALUES,
    STACK_EDGES,
    STACK_PIECES,
    TALLY_STACK,
)
from cython.cimports.core_pdf_cythonized._knockout_math import knockout_component
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.math import ceil, fabs, floor, rint
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
def _coverage_from_device(
    e: cython.p_const_double, count: cython.Py_ssize_t, width: cython.int, height: cython.int
) -> object:
    stride: cython.Py_ssize_t = width + 2
    result = numpy.zeros((height, width), numpy.float64)
    out: cython.double[:, ::1] = result
    acc: cython.p_double = _zeroed_cells(height * stride)
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    running: cython.double
    try:
        _accumulate_device(e, count, width, height, acc)
        with cython.nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    running += acc[r * stride + c]
                    out[r, c] = py_min(1.0, fabs(running))
    finally:
        PyMem_Free(acc)
    return result


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


@cython.cfunc
@cython.exceptval(-1, check=False)
def _accumulate_device(
    e: cython.p_const_double,
    count: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
    acc: cython.p_double,
) -> cython.int:
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


def signed_area_coverage(edges, width: cython.int, height: cython.int):
    if height <= 0 or width <= 0 or edges.size == 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)

    view: cython.double[:, ::1] = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    if view.shape[1] != 4:
        raise ValueError("edges must have four columns")
    return _coverage_from_device(cython.address(view[0, 0]), view.shape[0], width, height)


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
    rows = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows)
    if kept == 0:
        return None
    if height <= 0 or width <= 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)
    device: cython.p_double = _device_edges(rows, kept, crop_x0, crop_y1, scale, ix0, iy0)
    try:
        return _coverage_from_device(device, kept, width, height)
    finally:
        PyMem_Free(device)


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


# The row engine: a fill's coverage one pixel row at a time, as runs of pixels that share it,
# from the edges that reach the row. Its memory is bounded by the edges and one row's width,
# never by the pixel box, and a row costs its edges' pieces and the cells they touch.
#
# Where a path's regions of winding +1 and -1 meet in a pixel their signed areas cancel, and
# where a region of winding 2 meets one of 0 its area counts twice. A row whose sums leave
# 0..1 (or 0..-1) has such regions, and is measured instead along FALLBACK_LINES lines, each
# nonzero span's length exact.

RowEdge = cython.struct(
    top_x=cython.double,
    top_y=cython.double,
    bottom_y=cython.double,
    slope=cython.double,
    direction=cython.double,
    first_row=cython.Py_ssize_t,
    last_row=cython.Py_ssize_t,
)
# A piece's right weight, added to the cell after its own once every left weight is in.
Entry = cython.struct(column=cython.Py_ssize_t, value=cython.double)
Run = cython.struct(start=cython.Py_ssize_t, end=cython.Py_ssize_t, coverage=cython.double)
Cross = cython.struct(x=cython.double, direction=cython.int)
EdgeKey = cython.struct(key=cython.Py_ssize_t, index=cython.Py_ssize_t)
Rows = cython.struct(
    edges=cython.pointer[RowEdge],
    count=cython.Py_ssize_t,
    order=cython.p_Py_ssize_t,
    active=cython.p_Py_ssize_t,
    active_count=cython.Py_ssize_t,
    next_edge=cython.Py_ssize_t,
    width=cython.Py_ssize_t,
    height=cython.Py_ssize_t,
    entries=cython.pointer[Entry],
    entry_capacity=cython.Py_ssize_t,
    crosses=cython.pointer[Cross],
    runs=cython.pointer[Run],
    run_capacity=cython.Py_ssize_t,
    run_count=cython.Py_ssize_t,
    full=cython.p_int,
    partial=cython.p_double,
    fallback_rows=cython.Py_ssize_t,
    acc=cython.p_double,
    acc_heap=cython.bint,
    block_heap=cython.bint,
    entries_heap=cython.bint,
    runs_heap=cython.bint,
    # A small window's cells, summed densely for the whole window at once, as glyphs are.
    dense=cython.bint,
    cells=cython.p_double,
    cells_heap=cython.bint,
    device=cython.p_const_double,
    kept=cython.Py_ssize_t,
    scratch=cython.p_double,
    ready=cython.bint,
)

# How far a row's sums may stray past 0..1 from rounding before it counts as mixed.
MIXED_SLACK = cython.declare(cython.double, 1e-7)
# The dense sums take a window's edges only when every y is within this, as a cast of a
# row from a larger one would not be defined.
DENSE_REACH = cython.declare(cython.double, 1e15)
FALLBACK_LINES = cython.declare(cython.int, 16)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _compare_crosses(
    left: cython.pointer[cython.const[cython.void]],
    right: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    a: cython.double = cython.cast(cython.pointer[cython.const[Cross]], left).x
    b: cython.double = cython.cast(cython.pointer[cython.const[Cross]], right).x
    return (a > b) - (a < b)


@cython.cfunc
@cython.exceptval(-1, check=False)
def _rows_begin(
    rows: cython.pointer[Rows],
    device: cython.p_const_double,
    kept: cython.Py_ssize_t,
    width: cython.Py_ssize_t,
    height: cython.Py_ssize_t,
    scratch: cython.p_double,
    mode: cython.int,
) -> cython.int:
    """rows over the kept device edges of a width x height window, in scratch (of
    SCRATCH_DOUBLES doubles, or NULL) where they fit; _rows_free frees it, whether or not
    this succeeds. A window of at most DENSE_CELLS cells whose edges are all finite and near
    is summed densely, as glyphs are, its edges read only for a row that takes the fallback;
    any other goes row by row. mode 1 asks for the dense sums and 2 for the rows, where
    they can be had, so both can be checked."""
    rows.active_count = 0
    rows.next_edge = 0
    rows.width = width
    rows.height = height
    rows.run_count = 0
    rows.fallback_rows = 0
    rows.device = device
    rows.kept = kept
    rows.scratch = scratch
    i: cython.Py_ssize_t
    y: cython.double
    stride: cython.Py_ssize_t = width + 2
    entry_bytes: cython.Py_ssize_t = SCRATCH_ENTRIES * cython.sizeof(Entry)
    run_bytes: cython.Py_ssize_t = SCRATCH_RUNS * cython.sizeof(Run)
    if scratch != cython.NULL:
        rows.runs = cython.cast(
            cython.pointer[Run],
            cython.cast(cython.p_char, scratch) + _scratch_edge_bytes() + entry_bytes,
        )
        rows.run_capacity = SCRATCH_RUNS
    else:
        rows.run_capacity = 64
        rows.runs = cython.cast(
            cython.pointer[Run], PyMem_Malloc(rows.run_capacity * cython.sizeof(Run))
        )
        if rows.runs == cython.NULL:
            raise MemoryError
        rows.runs_heap = True
    dense: cython.bint = mode != 2 and (mode == 1 or stride * height <= DENSE_CELLS)
    if dense:
        for i in range(kept):
            y = device[i * 4 + 1]
            if not (-DENSE_REACH <= y <= DENSE_REACH):
                dense = False
                break
            y = device[i * 4 + 3]
            if not (-DENSE_REACH <= y <= DENSE_REACH):
                dense = False
                break
    if not dense:
        return _rows_edges(rows, True)
    rows.dense = True
    tail: cython.Py_ssize_t = _scratch_edge_bytes() + entry_bytes + run_bytes
    if scratch != cython.NULL and tail + stride * height * cython.sizeof(
        cython.double
    ) <= SCRATCH_DOUBLES * cython.sizeof(cython.double):
        rows.cells = cython.cast(cython.p_double, cython.cast(cython.p_char, scratch) + tail)
        memset(rows.cells, 0, stride * height * cython.sizeof(cython.double))
    else:
        rows.cells = _zeroed_cells(stride * height)
        rows.cells_heap = True
    _accumulate_device(device, kept, width, height, rows.cells)
    return 0


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def _scratch_edge_bytes() -> cython.Py_ssize_t:
    return SCRATCH_EDGES * (
        cython.sizeof(RowEdge)
        + 2 * cython.sizeof(cython.Py_ssize_t)
        + cython.sizeof(Cross)
        + cython.sizeof(EdgeKey)
    )


@cython.cfunc
@cython.exceptval(-1, check=False)
def _rows_edges(rows: cython.pointer[Rows], scratch_acc: cython.bint) -> cython.int:
    """rows' edge table, active list and one row's cells, for going row by row or for the
    fallback; the cells go in scratch only when scratch_acc, as the dense sums use it."""
    device: cython.p_const_double = rows.device
    kept: cython.Py_ssize_t = rows.kept
    width: cython.Py_ssize_t = rows.width
    height: cython.Py_ssize_t = rows.height
    scratch: cython.p_double = rows.scratch
    rows.ready = True
    per_edge: cython.Py_ssize_t = (
        cython.sizeof(RowEdge)
        + 2 * cython.sizeof(cython.Py_ssize_t)
        + cython.sizeof(Cross)
        + cython.sizeof(EdgeKey)
    )
    edge_bytes: cython.Py_ssize_t = SCRATCH_EDGES * per_edge
    entry_bytes: cython.Py_ssize_t = SCRATCH_ENTRIES * cython.sizeof(Entry)
    run_bytes: cython.Py_ssize_t = SCRATCH_RUNS * cython.sizeof(Run)
    fits: cython.bint = scratch != cython.NULL
    cells: cython.Py_ssize_t = kept if kept > 0 else 1
    block: cython.p_char
    if fits and cells <= SCRATCH_EDGES:
        block = cython.cast(cython.p_char, scratch)
        cells = SCRATCH_EDGES
    else:
        block = cython.cast(cython.p_char, PyMem_Malloc(cells * per_edge))
        if block == cython.NULL:
            raise MemoryError
        rows.block_heap = True
    rows.edges = cython.cast(cython.pointer[RowEdge], block)
    block += cells * cython.sizeof(RowEdge)
    rows.order = cython.cast(cython.p_Py_ssize_t, block)
    block += cells * cython.sizeof(cython.Py_ssize_t)
    rows.active = cython.cast(cython.p_Py_ssize_t, block)
    block += cells * cython.sizeof(cython.Py_ssize_t)
    rows.crosses = cython.cast(cython.pointer[Cross], block)
    block += cells * cython.sizeof(Cross)
    keys: cython.pointer[EdgeKey] = cython.cast(cython.pointer[EdgeKey], block)
    if fits:
        rows.entries = cython.cast(
            cython.pointer[Entry], cython.cast(cython.p_char, scratch) + edge_bytes
        )
        rows.entry_capacity = SCRATCH_ENTRIES
    else:
        rows.entry_capacity = 64
        rows.entries = cython.cast(
            cython.pointer[Entry], PyMem_Malloc(rows.entry_capacity * cython.sizeof(Entry))
        )
        if rows.entries == cython.NULL:
            raise MemoryError
        rows.entries_heap = True
    acc_bytes: cython.Py_ssize_t = (width + 2) * cython.sizeof(cython.double)
    if (
        fits
        and scratch_acc
        and edge_bytes + entry_bytes + run_bytes + acc_bytes
        <= SCRATCH_DOUBLES * cython.sizeof(cython.double)
    ):
        rows.acc = cython.cast(
            cython.p_double,
            cython.cast(cython.p_char, scratch) + edge_bytes + entry_bytes + run_bytes,
        )
    else:
        rows.acc = cython.cast(cython.p_double, PyMem_Malloc(acc_bytes))
        if rows.acc == cython.NULL:
            raise MemoryError
        rows.acc_heap = True
    memset(rows.acc, 0, acc_bytes)
    k: cython.Py_ssize_t
    i: cython.Py_ssize_t
    out: cython.Py_ssize_t = 0
    sx: cython.double
    sy: cython.double
    ex: cython.double
    ey: cython.double
    first: cython.double
    last: cython.double
    edge: cython.pointer[RowEdge]
    for i in range(kept):
        sx = device[i * 4]
        sy = device[i * 4 + 1]
        ex = device[i * 4 + 2]
        ey = device[i * 4 + 3]
        # An edge with a NaN or infinite y reaches no row the sums can measure.
        if sy == ey or not (sy - sy == 0.0 and ey - ey == 0.0):
            continue
        edge = rows.edges + out
        if ey > sy:
            edge.direction = 1.0
            edge.top_x = sx
            edge.top_y = sy
            edge.bottom_y = ey
            edge.slope = (ex - sx) / (ey - sy)
        else:
            edge.direction = -1.0
            edge.top_x = ex
            edge.top_y = ey
            edge.bottom_y = sy
            edge.slope = (sx - ex) / (sy - ey)
        first = py_min(py_max(floor(edge.top_y), 0.0), cython.cast(cython.double, height))
        last = py_max(py_min(ceil(edge.bottom_y), cython.cast(cython.double, height)), 0.0)
        edge.first_row = cython.cast(cython.Py_ssize_t, first)
        edge.last_row = cython.cast(cython.Py_ssize_t, last)
        keys[out].key = edge.first_row
        keys[out].index = out
        out += 1
    rows.count = out
    # The edges in order of their first row, then of index: a counting sort over the rows.
    tally_stack = cython.declare(cython.Py_ssize_t[TALLY_STACK])
    tally: cython.p_Py_ssize_t = tally_stack
    if height + 1 > TALLY_STACK:
        tally = cython.cast(
            cython.p_Py_ssize_t, PyMem_Malloc((height + 1) * cython.sizeof(cython.Py_ssize_t))
        )
        if tally == cython.NULL:
            raise MemoryError
    memset(tally, 0, (height + 1) * cython.sizeof(cython.Py_ssize_t))
    for i in range(out):
        tally[keys[i].key] += 1
    total: cython.Py_ssize_t = 0
    for k in range(height + 1):
        total += tally[k]
        tally[k] = total - tally[k]
    for i in range(out):
        rows.order[tally[keys[i].key]] = keys[i].index
        tally[keys[i].key] += 1
    if tally != tally_stack:
        PyMem_Free(tally)
    return 0


@cython.cfunc
@cython.exceptval(check=False)
def _rows_clear(rows: cython.pointer[Rows]) -> cython.void:
    """rows with no blocks, so _rows_free can run whatever _rows_init reached."""
    rows.edges = cython.NULL
    rows.order = cython.NULL
    rows.active = cython.NULL
    rows.crosses = cython.NULL
    rows.entries = cython.NULL
    rows.runs = cython.NULL
    rows.full = cython.NULL
    rows.partial = cython.NULL
    rows.acc = cython.NULL
    rows.acc_heap = False
    rows.cells = cython.NULL
    rows.cells_heap = False
    rows.dense = False
    rows.ready = False
    rows.count = 0
    rows.block_heap = False
    rows.entries_heap = False
    rows.runs_heap = False


@cython.cfunc
@cython.exceptval(check=False)
def _rows_free(rows: cython.pointer[Rows]) -> cython.void:
    if rows.block_heap:
        PyMem_Free(rows.edges)
    if rows.entries_heap:
        PyMem_Free(rows.entries)
    if rows.runs_heap:
        PyMem_Free(rows.runs)
    if rows.acc_heap:
        PyMem_Free(rows.acc)
    if rows.cells_heap:
        PyMem_Free(rows.cells)
    PyMem_Free(rows.full)
    PyMem_Free(rows.partial)


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
    if end <= start:
        return 0
    if rows.run_count:
        last = rows.runs + rows.run_count - 1
        if last.end == start and last.coverage == coverage:
            last.end = end
            return 0
    if rows.run_count == rows.run_capacity:
        grown = cython.cast(
            cython.pointer[Run], PyMem_Malloc(2 * rows.run_capacity * cython.sizeof(Run))
        )
        if grown == cython.NULL:
            raise MemoryError
        memcpy(grown, rows.runs, rows.run_capacity * cython.sizeof(Run))
        if rows.runs_heap:
            PyMem_Free(rows.runs)
        rows.runs = grown
        rows.runs_heap = True
        rows.run_capacity *= 2
    rows.runs[rows.run_count].start = start
    rows.runs[rows.run_count].end = end
    rows.runs[rows.run_count].coverage = coverage
    rows.run_count += 1
    return 0


@cython.cfunc
@cython.exceptval(check=False)
def _crossings(rows: cython.pointer[Rows], y: cython.double) -> cython.Py_ssize_t:
    """The active edges' crossings of the line y, in order of x."""
    count: cython.Py_ssize_t = 0
    a: cython.Py_ssize_t
    k: cython.Py_ssize_t
    edge: cython.pointer[RowEdge]
    cross: Cross
    for a in range(rows.active_count):
        edge = rows.edges + rows.active[a]
        if edge.top_y <= y < edge.bottom_y:
            rows.crosses[count].x = edge.top_x + (y - edge.top_y) * edge.slope
            rows.crosses[count].direction = 1 if edge.direction > 0.0 else -1
            count += 1
    if count > 16:
        qsort(rows.crosses, count, cython.sizeof(Cross), _compare_crosses)
    else:
        for a in range(1, count):
            cross = rows.crosses[a]
            k = a
            while k > 0 and rows.crosses[k - 1].x > cross.x:
                rows.crosses[k] = rows.crosses[k - 1]
                k -= 1
            rows.crosses[k] = cross
    return count


@cython.cfunc
@cython.exceptval(-1, check=False)
def _fallback_row(rows: cython.pointer[Rows], r: cython.Py_ssize_t) -> cython.int:
    """Row r's runs measured along FALLBACK_LINES lines, each nonzero span's length exact."""
    width: cython.Py_ssize_t = rows.width
    if rows.full == cython.NULL:
        rows.full = cython.cast(cython.p_int, PyMem_Malloc((width + 1) * cython.sizeof(cython.int)))
        rows.partial = cython.cast(
            cython.p_double, PyMem_Malloc((width + 1) * cython.sizeof(cython.double))
        )
        if rows.full == cython.NULL or rows.partial == cython.NULL:
            raise MemoryError
        memset(rows.full, 0, (width + 1) * cython.sizeof(cython.int))
        memset(rows.partial, 0, (width + 1) * cython.sizeof(cython.double))
    full: cython.p_int = rows.full
    partial: cython.p_double = rows.partial
    right: cython.double = cython.cast(cython.double, width)
    low: cython.Py_ssize_t = width
    high: cython.Py_ssize_t = 0
    line: cython.int
    i: cython.Py_ssize_t
    c: cython.Py_ssize_t
    count: cython.Py_ssize_t
    winding: cython.int
    start: cython.double
    end: cython.double
    first: cython.Py_ssize_t
    last: cython.Py_ssize_t
    running: cython.int
    coverage: cython.double
    for line in range(FALLBACK_LINES):
        count = _crossings(rows, cython.cast(cython.double, r) + (line + 0.5) / FALLBACK_LINES)
        winding = 0
        for i in range(count - 1):
            winding += rows.crosses[i].direction
            if winding == 0:
                continue
            start = rows.crosses[i].x
            end = rows.crosses[i + 1].x
            if start < 0.0:
                start = 0.0
            if end > right:
                end = right
            if not (end > start):
                continue
            first = cython.cast(cython.Py_ssize_t, floor(start))
            last = cython.cast(cython.Py_ssize_t, floor(end))
            if first == last:
                partial[first] += end - start
            else:
                partial[first] += (first + 1.0) - start
                full[first + 1] += 1
                full[last] -= 1
                if last < width:
                    partial[last] += end - last
            if first < low:
                low = first
            if last >= high:
                high = last + 1 if last < width else width
    rows.fallback_rows += 1
    running = 0
    for c in range(low, high):
        running += full[c]
        coverage = py_min(1.0, (cython.cast(cython.double, running) + partial[c]) / FALLBACK_LINES)
        full[c] = 0
        partial[c] = 0.0
        if coverage > 0.0:
            _emit(rows, c, c + 1, coverage)
    if low < high and high <= width:
        full[high] = 0
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
def _dense_row(rows: cython.pointer[Rows], r: cython.Py_ssize_t, emit: cython.bint) -> cython.int:
    """1 when row r's dense sums show no mixed windings and emit is false: the caller reads
    them from rows.cells. Otherwise 0, with the row's runs in rows.runs: the sums', or the
    fallback's when they show mixed windings."""
    width: cython.Py_ssize_t = rows.width
    cells: cython.p_double = rows.cells + r * (width + 2)
    running: cython.double = 0.0
    highest: cython.double = 0.0
    lowest: cython.double = 0.0
    column: cython.Py_ssize_t
    rows.run_count = 0
    for column in range(width):
        running += cells[column]
        if running > highest:
            highest = running
        if running < lowest:
            lowest = running
    if (
        highest > 1.0 + MIXED_SLACK
        or lowest < -1.0 - MIXED_SLACK
        or (highest > MIXED_SLACK and lowest < -MIXED_SLACK)
    ):
        if not rows.ready:
            _rows_edges(rows, False)
        _advance(rows, r)
        _fallback_row(rows, r)
        return 0
    if not emit:
        return 1
    running = 0.0
    for column in range(width):
        running += cells[column]
        _emit(rows, column, column + 1, py_min(1.0, fabs(running)))
    return 0


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
def _rows_next(rows: cython.pointer[Rows], r: cython.Py_ssize_t, emit: cython.bint) -> cython.int:
    """Row r's runs, in rows.runs, rows rising, and 0; or 1 when the row is the dense sums'
    in rows.cells and emit is false."""
    if rows.dense:
        return _dense_row(rows, r, emit)
    _row(rows, r)
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def _row(rows: cython.pointer[Rows], r: cython.Py_ssize_t) -> cython.int:
    """Row r's runs, in rows.runs: each pixel's coverage, the sum of the signed areas of the
    edges' pieces in it, as the dense sums gave it, or the fallback's."""
    edges: cython.pointer[RowEdge] = rows.edges
    active: cython.p_Py_ssize_t = rows.active
    width: cython.Py_ssize_t = rows.width
    entries: cython.pointer[Entry]
    a: cython.Py_ssize_t
    keep: cython.Py_ssize_t
    count: cython.Py_ssize_t = 0
    edge: cython.pointer[RowEdge]
    grown: cython.pointer[Entry]
    column: cython.Py_ssize_t
    column_count: cython.Py_ssize_t
    column_offset: cython.Py_ssize_t
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
    running: cython.double = 0.0
    vertical: cython.bint
    rows.run_count = 0
    keep = _advance(rows, r)
    if keep == 0:
        return 0

    entries = rows.entries
    acc: cython.p_double = rows.acc
    low: cython.Py_ssize_t = width + 1
    high: cython.Py_ssize_t = -1
    highest: cython.double = 0.0
    lowest: cython.double = 0.0
    run_start: cython.Py_ssize_t
    cell: cython.double
    coverage: cython.double
    next_coverage: cython.double
    last: cython.Py_ssize_t
    for a in range(keep):
        edge = edges + active[a]
        row_top = py_max(cython.cast(cython.double, r), edge.top_y)
        row_bottom = py_min(r + 1.0, edge.bottom_y)
        row_height = row_bottom - row_top
        if row_height <= 0.0:
            continue
        entry_x = edge.top_x + (row_top - edge.top_y) * edge.slope
        exit_x = edge.top_x + (row_bottom - edge.top_y) * edge.slope
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
            signed_height = edge.direction * fragment_height
            if count + 1 > rows.entry_capacity:
                grown = cython.cast(
                    cython.pointer[Entry],
                    PyMem_Malloc(2 * rows.entry_capacity * cython.sizeof(Entry)),
                )
                if grown == cython.NULL:
                    raise MemoryError
                memcpy(grown, entries, count * cython.sizeof(Entry))
                if rows.entries_heap:
                    PyMem_Free(entries)
                rows.entries = grown
                rows.entries_heap = True
                entries = grown
                rows.entry_capacity *= 2
            # The left weight goes into the cell now, the right weight into the next
            # once every left weight is in, the order the dense sums added them in.
            acc[column] += signed_height * (1.0 - offset_in_cell)
            entries[count].column = column + 1
            entries[count].value = signed_height * offset_in_cell
            count += 1
            if column < low:
                low = column
            if column + 1 > high:
                high = column + 1
    for a in range(count):
        acc[entries[a].column] += entries[a].value
    # Cells outside low..high are 0: the running sum is 0 before them and constant after.
    if high >= width:
        last = width - 1
    else:
        last = high
    # A cell no piece touched leaves the sum, and so the coverage, as it was: the run goes on.
    run_start = low
    coverage = 0.0
    for column in range(low, last + 1):
        cell = acc[column]
        if cell != 0.0 or column == low:
            running += cell
            acc[column] = 0.0
            if running > highest:
                highest = running
            if running < lowest:
                lowest = running
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
    # A row whose sums leave 0..1 or 0..-1 has regions of both windings, or of a winding
    # beyond 1, which the sums cancel or count twice: it is measured along lines instead.
    if (
        highest > 1.0 + MIXED_SLACK
        or lowest < -1.0 - MIXED_SLACK
        or (highest > MIXED_SLACK and lowest < -MIXED_SLACK)
    ):
        rows.run_count = 0
        return _fallback_row(rows, r)
    if last + 1 < width:
        _emit(rows, last + 1, width, py_min(1.0, fabs(running)))
    return 0


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
    number of rows that took the fallback; None when no edge slopes. mode 1 sums the window
    densely and 2 row by row, as a small and a large window are, where each can be had."""
    rows_array = edge_rows(edges)
    kept: cython.Py_ssize_t = _sloped_edge_count(rows_array)
    if kept == 0:
        return None
    plane = numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)
    if height <= 0 or width <= 0:
        return plane, 0
    out: cython.double[:, ::1] = plane
    device: cython.p_double = _device_edges(rows_array, kept, crop_x0, crop_y1, scale, ix0, iy0)
    rows = cython.declare(Rows)
    r: cython.Py_ssize_t
    k: cython.Py_ssize_t
    c: cython.Py_ssize_t
    fallback: cython.Py_ssize_t = 0
    _rows_clear(cython.address(rows))
    try:
        _rows_begin(cython.address(rows), device, kept, width, height, cython.NULL, mode)
        for r in range(height):
            _rows_next(cython.address(rows), r, True)
            for k in range(rows.run_count):
                for c in range(rows.runs[k].start, rows.runs[k].end):
                    out[r, c] = rows.runs[k].coverage
        fallback = rows.fallback_rows
    finally:
        _rows_free(cython.address(rows))
        PyMem_Free(device)
    return plane, fallback


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
    scratch = cython.declare(cython.double[SCRATCH_DOUBLES])
    k: cython.Py_ssize_t
    row_cells: cython.p_double
    running: cython.double
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    coverage: cython.double
    raw: cython.uchar
    shape: cython.uchar
    _rows_clear(cython.address(engine))
    try:
        _rows_begin(cython.address(engine), device, kept, width, height, scratch, 0)
        for r in range(height):
            if _rows_next(cython.address(engine), r, False):
                row_cells = engine.cells + r * (width + 2)
                running = 0.0
                for c in range(width):
                    running += row_cells[c]
                    coverage = py_min(1.0, fabs(running))
                    raw = cython.cast(cython.uchar, rint(coverage * alpha))
                    shape = cython.cast(cython.uchar, rint(coverage * 255.0)) if has_shape else 0
                    if raw != 0 or shape != 0:
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
    scratch = cython.declare(cython.double[SCRATCH_DOUBLES])
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
    _rows_clear(cython.address(engine))
    try:
        _rows_begin(cython.address(engine), device, kept, width, height, scratch, 0)
        for r in range(height):
            _rows_next(cython.address(engine), r, True)
            for run in range(engine.run_count):
                coverage = engine.runs[run].coverage
                # A pixel this run leaves at zero alpha and shape is left as it is.
                if (
                    cython.cast(cython.uchar, rint(coverage * alpha)) == 0
                    and cython.cast(cython.uchar, rint(coverage * 255.0)) == 0
                ):
                    continue
                for c in range(engine.runs[run].start, engine.runs[run].end):
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
        _rows_free(cython.address(engine))
    return 0
