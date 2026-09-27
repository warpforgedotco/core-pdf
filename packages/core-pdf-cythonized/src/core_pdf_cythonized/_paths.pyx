# SPDX-License-Identifier: AGPL-3.0-only

cimport numpy as cnp
from cpython cimport array
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_GET_SIZE
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.math cimport ceil, fabs

from core_pdf_cythonized._ndarray cimport empty_float64, float64_data
from core_pdf_cythonized._pymath cimport check_integral, py_max, py_min

import numpy

cnp.import_array()


cdef double INF = float("inf")


cdef struct Points:
    double *x
    double *y
    Py_ssize_t count
    Py_ssize_t capacity


cdef int grow(Points *points, Py_ssize_t needed) except -1:
    cdef Py_ssize_t capacity = points.capacity
    if points.count + needed <= capacity:
        return 0
    while capacity < points.count + needed:
        capacity = capacity * 2 if capacity else 64
    cdef double *x = <double *> PyMem_Realloc(points.x, capacity * sizeof(double))
    if x == NULL:
        raise MemoryError
    points.x = x
    cdef double *y = <double *> PyMem_Realloc(points.y, capacity * sizeof(double))
    if y == NULL:
        raise MemoryError
    points.y = y
    points.capacity = capacity
    return 0


cdef class PathBuilder:
    cdef Points points
    cdef list starts
    cdef list closed
    cdef Py_ssize_t current_start
    cdef bint current_closed
    cdef bint has_subpath

    def __cinit__(self):
        self.points.x = NULL
        self.points.y = NULL
        self.points.count = 0
        self.points.capacity = 0
        self.starts = []
        self.closed = []
        self.has_subpath = False

    def __dealloc__(self):
        PyMem_Free(self.points.x)
        PyMem_Free(self.points.y)

    cdef int finish_subpath(self) except -1:
        if self.has_subpath:
            self.starts.append(self.current_start)
            self.closed.append(self.current_closed)
        return 0

    cdef int move_to(self, double x, double y) except -1:
        self.finish_subpath()
        self.current_start = self.points.count
        self.current_closed = False
        self.has_subpath = True
        return self.append(x, y)

    cdef int line_to(self, double x, double y) except -1:
        if not self.has_subpath:
            return self.move_to(x, y)
        if not self.current_closed:
            return self.append(x, y)
        return 0

    cdef int close(self) except -1:
        if self.has_subpath and self.points.count - self.current_start > 1:
            self.current_closed = True
        return 0

    cdef int rect(self, double x, double y, double w, double h) except -1:
        self.finish_subpath()
        self.current_start = self.points.count
        self.has_subpath = True
        self.append(x, y)
        self.append(x + w, y)
        self.append(x + w, y + h)
        self.append(x, y + h)
        self.current_closed = True
        return 0

    cdef inline int append(self, double x, double y) except -1:
        grow(&self.points, 1)
        self.points.x[self.points.count] = x
        self.points.y[self.points.count] = y
        self.points.count += 1
        return 0


cdef bint add_subpath_box(
    const double* px, const double* py, Py_ssize_t start, Py_ssize_t end, bint have_box, double* box
) noexcept:
    cdef double sx0 = INF, sy0 = INF, sx1 = -INF, sy1 = -INF
    cdef double x, y
    cdef Py_ssize_t i
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


cdef int add_curve(PathBuilder path, const double *values, hypot) except -1:
    cdef double x0 = values[0], y0 = values[1], x1 = values[2], y1 = values[3]
    cdef double x2 = values[4], y2 = values[5], x3 = values[6], y3 = values[7]
    cdef double scale = py_max(
        py_max(<double> hypot(values[8], values[9]), <double> hypot(values[10], values[11])),
        1.0,
    )
    cdef double control_len = (
        <double> hypot(x1 - x0, y1 - y0)
        + <double> hypot(x2 - x1, y2 - y1)
        + <double> hypot(x3 - x2, y3 - y2)
    )
    cdef double flatness_value = values[12] if values[12] != 0.0 else 0.25
    cdef double flatness = py_max(0.1, flatness_value)
    cdef double steps = ceil(control_len * scale / (flatness * 8.0))
    check_integral(steps)
    steps = steps if steps < 128 else 128
    cdef Py_ssize_t segments = <Py_ssize_t> steps if steps > 4 else 4
    cdef double previous_x = x0, previous_y = y0
    cdef double segment_step = 1.0 / segments
    cdef double t, mt, mt2, t2, b0, b1, b2, b3, x, y
    cdef Py_ssize_t i
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


def flatten_path_commands(ops, coords, matrix, hypot, array.array line_rows, double line_width):
    cdef const unsigned char* op_data
    cdef Py_ssize_t op_count
    if type(ops) is bytearray:
        op_data = <const unsigned char*> PyByteArray_AS_STRING(ops)
        op_count = PyByteArray_GET_SIZE(ops)
    elif type(ops) is bytes:
        op_data = <const unsigned char*> PyBytes_AS_STRING(ops)
        op_count = PyBytes_GET_SIZE(ops)
    else:
        return flatten_path_command_views(ops, coords, matrix, hypot, line_rows, line_width)
    if type(coords) is not array.array or (<array.array> coords).ob_descr.typecode != c'd':
        return flatten_path_command_views(ops, coords, matrix, hypot, line_rows, line_width)
    return _flatten_path_commands(
        op_data,
        op_count,
        (<array.array> coords).data.as_doubles,
        len(coords),
        matrix,
        hypot,
        line_rows,
        line_width,
    )


def flatten_path_command_views(
    const unsigned char[::1] ops,
    const double[::1] coords,
    matrix,
    hypot,
    array.array line_rows,
    double line_width,
):
    return _flatten_path_commands(
        &ops[0] if ops.shape[0] else NULL,
        ops.shape[0],
        &coords[0] if coords.shape[0] else NULL,
        coords.shape[0],
        matrix,
        hypot,
        line_rows,
        line_width,
    )


cdef tuple _flatten_path_commands(
    const unsigned char* ops,
    Py_ssize_t op_count,
    const double* coords,
    Py_ssize_t available,
    matrix,
    hypot,
    array.array line_rows,
    double line_width,
):
    cdef PathBuilder path = PathBuilder()
    cdef Py_ssize_t op_index, at = 0
    cdef unsigned char op
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
            add_curve(path, &coords[at], hypot)
            at += 13
        else:
            raise ValueError(f"unknown path operator {op}")
    path.finish_subpath()

    cdef Py_ssize_t count = path.points.count
    cdef double *px = path.points.x
    cdef double *py = path.points.y
    cdef double a, b, c, d, e, f, x, y
    cdef Py_ssize_t i
    if matrix is not None:
        a, b, c, d, e, f = matrix
        for i in range(count):
            x = px[i]
            y = py[i]
            px[i] = x * a + y * c + e
            py[i] = x * b + y * d + f

    xs = empty_float64(count)
    ys = empty_float64(count)
    cdef double* out_x = float64_data(xs)
    cdef double* out_y = float64_data(ys)
    for i in range(count):
        out_x[i] = px[i]
        out_y[i] = py[i]

    cdef list starts = path.starts
    cdef list closed = path.closed
    cdef Py_ssize_t subpath_count = len(starts)
    cdef list spans = []
    cdef Py_ssize_t start, end, index, line_count = 0
    cdef bint has_segments = False
    cdef bint have_box = False
    cdef double box[4]
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

    cdef Py_ssize_t row
    cdef double* out
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


def path_bounds(const double[::1] xs, const double[::1] ys, list spans):
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    cdef Py_ssize_t start, end
    cdef bint have_box = False, has_segments = False
    cdef double box[4]
    for start, end, _ in spans:
        if start < 0 or end > xs.shape[0] or end < start:
            raise ValueError("span runs past the points")
        if end <= start:
            continue
        if end - start > 1:
            has_segments = True
        have_box = add_subpath_box(&xs[0], &ys[0], start, end, have_box, box)
    bbox = (box[0], box[1], box[2], box[3]) if have_box else None
    return bbox, has_segments


def fill_edge_rows(const double[::1] xs, const double[::1] ys, list spans):
    """The fill edges of flattened subpaths, one (x0, y0, x1, y1) row each.

    Every subpath of two or more points contributes its consecutive edges and a
    closing edge back to its first point; the closing edge is dropped when it
    has no length, and no other edge is.
    """
    if xs.shape[0] != ys.shape[0]:
        raise ValueError("xs and ys differ in length")
    cdef Py_ssize_t count = xs.shape[0]
    cdef Py_ssize_t start, end, index, total = 0, row = 0
    for start, end, _ in spans:
        if end - start < 2:
            continue
        if start < 0 or end > count:
            raise ValueError("span runs past the points")
        total += end - start
    rows = numpy.empty((total, 4), dtype=numpy.float64)
    cdef double[:, ::1] out = rows
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
