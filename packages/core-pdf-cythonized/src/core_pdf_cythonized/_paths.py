# SPDX-License-Identifier: AGPL-3.0-only

import typing
from math import hypot as math_hypot

import cython
import numpy
from cython.cimports import numpy as cnp
from cython.cimports.core_pdf_cythonized._ndarray import empty_float64, float64_data
from cython.cimports.core_pdf_cythonized._pymath import check_integral, py_max, py_min
from cython.cimports.cpython import array
from cython.cimports.cpython.bytearray import PyByteArray_AS_STRING, PyByteArray_GET_SIZE
from cython.cimports.cpython.bytes import PyBytes_AS_STRING, PyBytes_GET_SIZE
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Realloc
from cython.cimports.libc.math import ceil, fabs

cnp.import_array()


INF = cython.declare(cython.double, float("inf"))


Points = cython.struct(
    x=cython.p_double,
    y=cython.p_double,
    count=cython.Py_ssize_t,
    capacity=cython.Py_ssize_t,
)


@cython.cfunc
@cython.exceptval(-1, check=False)
def grow(points: cython.pointer[Points], needed: cython.Py_ssize_t) -> cython.int:
    capacity: cython.Py_ssize_t = points.capacity
    if points.count + needed <= capacity:
        return 0
    while capacity < points.count + needed:
        capacity = capacity * 2 if capacity else 64
    x: cython.p_double = cython.cast(
        cython.p_double, PyMem_Realloc(points.x, capacity * cython.sizeof(cython.double))
    )
    if x == cython.NULL:
        raise MemoryError
    points.x = x
    y: cython.p_double = cython.cast(
        cython.p_double, PyMem_Realloc(points.y, capacity * cython.sizeof(cython.double))
    )
    if y == cython.NULL:
        raise MemoryError
    points.y = y
    points.capacity = capacity
    return 0


@cython.cclass
class PathBuilder:
    points: Points
    starts: list
    closed: list
    current_start: cython.Py_ssize_t
    current_closed: cython.bint
    has_subpath: cython.bint

    def __cinit__(self):
        self.points.x = cython.NULL
        self.points.y = cython.NULL
        self.points.count = 0
        self.points.capacity = 0
        self.starts = []
        self.closed = []
        self.has_subpath = False

    def __dealloc__(self):
        PyMem_Free(self.points.x)
        PyMem_Free(self.points.y)

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def finish_subpath(self) -> cython.int:
        if self.has_subpath:
            self.starts.append(self.current_start)
            self.closed.append(self.current_closed)
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def move_to(self, x: cython.double, y: cython.double) -> cython.int:
        self.finish_subpath()
        self.current_start = self.points.count
        self.current_closed = False
        self.has_subpath = True
        return self.append(x, y)

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def line_to(self, x: cython.double, y: cython.double) -> cython.int:
        if not self.has_subpath:
            return self.move_to(x, y)
        if not self.current_closed:
            return self.append(x, y)
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def close(self) -> cython.int:
        if self.has_subpath and self.points.count - self.current_start > 1:
            self.current_closed = True
        return 0

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def rect(
        self, x: cython.double, y: cython.double, w: cython.double, h: cython.double
    ) -> cython.int:
        self.finish_subpath()
        self.current_start = self.points.count
        self.has_subpath = True
        self.append(x, y)
        self.append(x + w, y)
        self.append(x + w, y + h)
        self.append(x, y + h)
        self.current_closed = True
        return 0

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-1, check=False)
    def append(self, x: cython.double, y: cython.double) -> cython.int:
        grow(cython.address(self.points), 1)
        self.points.x[self.points.count] = x
        self.points.y[self.points.count] = y
        self.points.count += 1
        return 0


@cython.cfunc
@cython.exceptval(check=False)
def add_subpath_box(
    px: cython.p_const_double,
    py: cython.p_const_double,
    start: cython.Py_ssize_t,
    end: cython.Py_ssize_t,
    have_box: cython.bint,
    box: cython.p_double,
) -> cython.bint:
    sx0: cython.double = INF
    sy0: cython.double = INF
    sx1: cython.double = -INF
    sy1: cython.double = -INF
    x: cython.double
    y: cython.double
    i: cython.Py_ssize_t
    for i in range(start, end):
        x = px[i]
        y = py[i]
        if x < sx0:
            sx0 = x
        if x > sx1:
            sx1 = x
        if y < sy0:
            sy0 = y
        if y > sy1:
            sy1 = y
    if sx0 > sx1:
        return have_box
    if not have_box:
        box[0] = sx0
        box[1] = sy0
        box[2] = sx1
        box[3] = sy1
        return True
    box[0] = py_min(box[0], sx0)
    box[1] = py_min(box[1], sy0)
    box[2] = py_max(box[2], sx1)
    box[3] = py_max(box[3], sy1)
    return True


@cython.cfunc
@cython.exceptval(-1, check=False)
def add_curve(path: PathBuilder, values: cython.p_const_double, hypot) -> cython.int:
    x0: cython.double = values[0]
    y0: cython.double = values[1]
    x1: cython.double = values[2]
    y1: cython.double = values[3]
    x2: cython.double = values[4]
    y2: cython.double = values[5]
    x3: cython.double = values[6]
    y3: cython.double = values[7]
    scale: cython.double = py_max(
        py_max(
            cython.cast(cython.double, hypot(values[8], values[9])),
            cython.cast(cython.double, hypot(values[10], values[11])),
        ),
        1.0,
    )
    control_len: cython.double = (
        cython.cast(cython.double, hypot(x1 - x0, y1 - y0))
        + cython.cast(cython.double, hypot(x2 - x1, y2 - y1))
        + cython.cast(cython.double, hypot(x3 - x2, y3 - y2))
    )
    flatness_value: cython.double = values[12] if values[12] != 0.0 else 0.25
    flatness: cython.double = py_max(0.1, flatness_value)
    steps: cython.double = ceil(control_len * scale / (flatness * 8.0))
    check_integral(steps)
    steps = steps if steps < 128 else 128
    segments: cython.Py_ssize_t = cython.cast(cython.Py_ssize_t, steps) if steps > 4 else 4
    previous_x: cython.double = x0
    previous_y: cython.double = y0
    segment_step: cython.double = 1.0 / segments
    t: cython.double
    mt: cython.double
    mt2: cython.double
    t2: cython.double
    b0: cython.double
    b1: cython.double
    b2: cython.double
    b3: cython.double
    x: cython.double
    y: cython.double
    i: cython.Py_ssize_t
    for i in range(1, segments + 1):
        t = i * segment_step
        mt = 1.0 - t
        mt2 = mt * mt
        t2 = t * t
        b0 = mt2 * mt
        b1 = 3.0 * mt2 * t
        b2 = 3.0 * mt * t2
        b3 = t2 * t
        x = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
        y = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
        if not path.has_subpath:
            path.move_to(previous_x, previous_y)
        path.line_to(x, y)
        previous_x = x
        previous_y = y
    return 0


def flatten_path_commands(
    ops,
    coords,
    matrix,
    hypot,
    line_rows: typing.Optional[array.array],
    line_width: cython.double,
):
    op_data: cython.p_const_uchar
    op_count: cython.Py_ssize_t
    # The raw-pointer path holds no buffer export, so it is taken only when the one
    # callback made during the walk, hypot, is math.hypot and cannot resize them.
    if hypot is not math_hypot:
        return flatten_path_command_views(ops, coords, matrix, hypot, line_rows, line_width)
    if type(ops) is bytearray:
        op_data = cython.cast(cython.p_const_uchar, PyByteArray_AS_STRING(ops))
        op_count = PyByteArray_GET_SIZE(ops)
    elif type(ops) is bytes:
        op_data = cython.cast(cython.p_const_uchar, PyBytes_AS_STRING(ops))
        op_count = PyBytes_GET_SIZE(ops)
    else:
        return flatten_path_command_views(ops, coords, matrix, hypot, line_rows, line_width)
    if (
        type(coords) is not array.array
        or cython.cast(array.array, coords).ob_descr.typecode != b"d"
    ):
        return flatten_path_command_views(ops, coords, matrix, hypot, line_rows, line_width)
    return _flatten_path_commands(
        op_data,
        op_count,
        cython.cast(array.array, coords).data.as_doubles,
        len(coords),
        matrix,
        hypot,
        line_rows,
        line_width,
    )


def flatten_path_command_views(
    ops: cython.const[cython.uchar][::1],
    coords: cython.const[cython.double][::1],
    matrix,
    hypot,
    line_rows: typing.Optional[array.array],
    line_width: cython.double,
):
    return _flatten_path_commands(
        cython.address(ops[0]) if ops.shape[0] else cython.NULL,
        ops.shape[0],
        cython.address(coords[0]) if coords.shape[0] else cython.NULL,
        coords.shape[0],
        matrix,
        hypot,
        line_rows,
        line_width,
    )


@cython.cfunc
def _flatten_path_commands(
    ops: cython.p_const_uchar,
    op_count: cython.Py_ssize_t,
    coords: cython.p_const_double,
    available: cython.Py_ssize_t,
    matrix,
    hypot,
    line_rows: typing.Optional[array.array],
    line_width: cython.double,
) -> tuple:
    path: PathBuilder = PathBuilder()
    op_index: cython.Py_ssize_t
    at: cython.Py_ssize_t = 0
    op: cython.uchar
    for op_index in range(op_count):
        op = ops[op_index]
        if op == 109:
            if at + 2 > available:
                raise ValueError("path coordinates run short")
            path.move_to(coords[at], coords[at + 1])
            at += 2
        elif op == 108:
            if at + 2 > available:
                raise ValueError("path coordinates run short")
            path.line_to(coords[at], coords[at + 1])
            at += 2
        elif op == 104:
            path.close()
        elif op == 114:
            if at + 4 > available:
                raise ValueError("path coordinates run short")
            path.rect(coords[at], coords[at + 1], coords[at + 2], coords[at + 3])
            at += 4
        elif op == 99:
            if at + 13 > available:
                raise ValueError("path coordinates run short")
            add_curve(path, cython.address(coords[at]), hypot)
            at += 13
        else:
            raise ValueError(f"unknown path operator {op}")
    path.finish_subpath()

    count: cython.Py_ssize_t = path.points.count
    px: cython.p_double = path.points.x
    py: cython.p_double = path.points.y
    a: cython.double
    b: cython.double
    c: cython.double
    d: cython.double
    e: cython.double
    f: cython.double
    x: cython.double
    y: cython.double
    i: cython.Py_ssize_t
    if matrix is not None:
        a, b, c, d, e, f = matrix
        for i in range(count):
            x = px[i]
            y = py[i]
            px[i] = x * a + y * c + e
            py[i] = x * b + y * d + f

    xs = empty_float64(count)
    ys = empty_float64(count)
    out_x: cython.p_double = float64_data(xs)
    out_y: cython.p_double = float64_data(ys)
    for i in range(count):
        out_x[i] = px[i]
        out_y[i] = py[i]

    starts: list = path.starts
    closed: list = path.closed
    subpath_count: cython.Py_ssize_t = len(starts)
    spans: list = []
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t
    line_count: cython.Py_ssize_t = 0
    has_segments: cython.bint = False
    have_box: cython.bint = False
    box = cython.declare(cython.double[4])
    for index in range(subpath_count):
        start = starts[index]
        end = starts[index + 1] if index + 1 < subpath_count else count
        spans.append((start, end, closed[index]))
        if end - start > 1:
            has_segments = True
        for i in range(start + 1, end):
            if fabs(px[i] - px[i - 1]) > 0.01 or fabs(py[i] - py[i - 1]) > 0.01:
                line_count += 1
        have_box = add_subpath_box(px, py, start, end, have_box, box)

    row: cython.Py_ssize_t
    out: cython.p_double
    if line_rows is not None and line_count:
        row = len(line_rows)
        array.resize_smart(line_rows, row + line_count * 5)
        out = line_rows.data.as_doubles + row
        for index in range(subpath_count):
            start = starts[index]
            end = starts[index + 1] if index + 1 < subpath_count else count
            for i in range(start + 1, end):
                if fabs(px[i] - px[i - 1]) > 0.01 or fabs(py[i] - py[i - 1]) > 0.01:
                    out[0] = px[i - 1]
                    out[1] = py[i - 1]
                    out[2] = px[i]
                    out[3] = py[i]
                    out[4] = line_width
                    out += 5

    bbox = (box[0], box[1], box[2], box[3]) if have_box else None
    return xs, ys, spans, bbox, has_segments


def path_bounds(
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: typing.Optional[list],
):
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    have_box: cython.bint = False
    has_segments: cython.bint = False
    box = cython.declare(cython.double[4])
    for start, end, _ in spans:
        if start < 0 or end > xs.shape[0] or end < start:
            raise ValueError("span runs past the points")
        if end <= start:
            continue
        if end - start > 1:
            has_segments = True
        have_box = add_subpath_box(
            cython.address(xs[0]), cython.address(ys[0]), start, end, have_box, box
        )
    bbox = (box[0], box[1], box[2], box[3]) if have_box else None
    return bbox, has_segments


def fill_edge_rows(
    xs: cython.const[cython.double][::1],
    ys: cython.const[cython.double][::1],
    spans: typing.Optional[list],
):
    """The fill edges of flattened subpaths, one (x0, y0, x1, y1) row each.

    Every subpath of two or more points contributes its consecutive edges and a
    closing edge back to its first point; the closing edge is dropped when it
    has no length, and no other edge is.
    """
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    count: cython.Py_ssize_t = xs.shape[0]
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t
    total: cython.Py_ssize_t = 0
    row: cython.Py_ssize_t = 0
    for start, end, _ in spans:
        if end - start < 2:
            continue
        if start < 0 or end > count:
            raise ValueError("span runs past the points")
        total += end - start
    rows = numpy.empty((total, 4), dtype=numpy.float64)
    out: cython.double[:, ::1] = rows
    for start, end, _ in spans:
        if end - start < 2:
            continue
        for index in range(start, end - 1):
            out[row, 0] = xs[index]
            out[row, 1] = ys[index]
            out[row, 2] = xs[index + 1]
            out[row, 3] = ys[index + 1]
            row += 1
        if xs[end - 1] != xs[start] or ys[end - 1] != ys[start]:
            out[row, 0] = xs[end - 1]
            out[row, 1] = ys[end - 1]
            out[row, 2] = xs[start]
            out[row, 3] = ys[start]
            row += 1
    return rows if row == total else rows[:row].copy()
