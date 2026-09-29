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
    STACK_CELLS,
    STACK_EDGE_VALUES,
    STACK_EDGES,
    STACK_PIECES,
)
from cython.cimports.core_pdf_cythonized._knockout_math import knockout_component
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.math import ceil, fabs, floor, rint
from cython.cimports.libc.string import memcpy, memset

cnp.import_array()


BytePlane = cython.struct(
    data=cython.p_uchar,
    row=cython.Py_ssize_t,
    column=cython.Py_ssize_t,
    channel=cython.Py_ssize_t,
)


FloatPlane = cython.struct(
    data=cython.p_float,
    row=cython.Py_ssize_t,
    column=cython.Py_ssize_t,
)


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
    return cython.cast(
        cython.p_const_double,
        cnp.PyArray_DATA(cython.cast(cnp.ndarray, rows)),
    )


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
    e: cython.p_const_double,
    count: cython.Py_ssize_t,
    width: cython.int,
    height: cython.int,
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
    stride: cython.Py_ssize_t = width + 2
    device_stack = cython.declare(cython.double[STACK_EDGE_VALUES])
    acc_stack = cython.declare(cython.double[STACK_CELLS])
    device: cython.p_double = _device_edges(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, device_stack, STACK_EDGES
    )
    acc: cython.p_double = cython.NULL
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    channel: cython.Py_ssize_t = pixels.channel
    running: cython.double
    coverage: cython.double
    raw: cython.uchar
    shape: cython.uchar
    pixel: cython.p_uchar
    cell: cython.p_float
    try:
        acc = _zeroed_cells(height * stride, acc_stack, STACK_CELLS)
        _accumulate_device(device, kept, width, height, acc)
        with cython.nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    running += acc[r * stride + c]
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
    finally:
        if device != device_stack:
            PyMem_Free(device)
        if acc != acc_stack:
            PyMem_Free(acc)
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
    stride: cython.Py_ssize_t = width + 2
    device_stack = cython.declare(cython.double[STACK_EDGE_VALUES])
    acc_stack = cython.declare(cython.double[STACK_CELLS])
    device: cython.p_double = _device_edges(
        rows, kept, crop_x0, crop_y1, scale, ix0, iy0, device_stack, STACK_EDGES
    )
    acc: cython.p_double = cython.NULL
    r: cython.Py_ssize_t
    c: cython.Py_ssize_t
    k: cython.Py_ssize_t
    running: cython.double
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
    try:
        acc = _zeroed_cells(height * stride, acc_stack, STACK_CELLS)
        _accumulate_device(device, kept, width, height, acc)
        with cython.nogil:
            for r in range(height):
                running = 0.0
                for c in range(width):
                    destination = target.data + r * target.row + c * target.column
                    backdrop = base.data + r * base.row + c * base.column
                    group_alpha = cython.cast(
                        cython.p_float,
                        cython.cast(cython.p_char, alpha_plane.data)
                        + r * alpha_plane.row
                        + c * alpha_plane.column,
                    )
                    running += acc[r * stride + c]
                    coverage = py_min(1.0, fabs(running))
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
                    eff = cython.cast(cython.double, quantized) / 255.0
                    sh = cython.cast(cython.double, source_shape)
                    if sh < 0.0:
                        sh = 0.0
                    elif sh > 1.0:
                        sh = 1.0
                    if eff > sh:
                        sh = eff
                    if sh > 0.0:
                        complete = (
                            cython.cast(cython.double, destination[3 * target.channel]) / 255.0
                        )
                        initial = cython.cast(cython.double, backdrop[3 * base.channel]) / 255.0
                        ec = cython.cast(cython.double, element[3]) / 255.0
                        remaining = 1.0 - sh
                        rga = eff + remaining * cython.cast(cython.double, group_alpha[0])
                        ra = initial + (1.0 - initial) * rga
                        for k in range(3):
                            colour[k] = knockout_component(
                                cython.cast(cython.double, element[k]) / 255.0,
                                ec,
                                remaining,
                                cython.cast(cython.double, destination[k * target.channel]) / 255.0,
                                complete,
                                cython.cast(cython.double, backdrop[k * base.channel]) / 255.0,
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
        if acc != acc_stack:
            PyMem_Free(acc)
    return 0
