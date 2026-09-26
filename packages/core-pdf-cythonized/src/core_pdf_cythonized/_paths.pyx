# SPDX-License-Identifier: AGPL-3.0-only
"""Path flattening for capture (core_pdf.impl.capture.recording.paint_path).

Every painted path was flattened into CapturedSubpath point lists, moved
through the CTM into another set of lists, walked again for its stroke lines
-- one CapturedLine object per segment -- and walked a fourth time, much later,
when table detection turned those objects back into arrays. On PyMuPDF
test_3806 that is 733,122 line objects for one page, and the four walks were
about 40% of its extraction.

This does all of it in one pass over the path's commands and returns arrays:
the transformed points, a (start, end, closed) span per subpath, the bounding
box, whether the path has a segment at all, and the stroke-line endpoints.
Extraction needs nothing else, so a captured path defers its point lists until
something -- the renderer, OCR -- asks for them.

Each step is the original's, in the original's order. The subpath rules are
CapturedPath's: a line_to after a close is dropped, a close needs two points,
and a curve on an empty path starts one at its first control point. Curves use
the original's Bernstein expansion, and the segment count goes through the
same math.hypot -- passed in, and called, rather than reimplemented, because
CPython's hypot is its own algorithm and C's may land one ULP away and change
a count. The box follows points_bbox and union_bbox exactly, including which
of two equal values survives, which matters only for signed zeros.
"""

from cpython cimport array
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
from libc.math cimport ceil, fabs, isinf, isnan

import numpy


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
    # CapturedPath's move_to / line_to / close / rect, over flat storage. Only
    # the last subpath ever receives points, so each subpath is a contiguous
    # run and a (start, closed) pair per subpath is the whole structure.
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


cdef int check_integral(double value) except -1:
    # What math.ceil raises converting the value to an int.
    if isnan(value):
        raise ValueError("cannot convert float NaN to integer")
    if isinf(value):
        raise OverflowError("cannot convert float infinity to integer")
    return 0


cdef inline double python_max(double left, double right) noexcept:
    # max(left, right): the left one unless the right is strictly greater.
    return right if right > left else left


cdef inline double python_min(double left, double right) noexcept:
    return right if right < left else left


cdef bint add_subpath_box(
    const double* px, const double* py, Py_ssize_t start, Py_ssize_t end, bint have_box, double* box
) noexcept:
    # points_bbox for one subpath -- NaNs never pass its comparisons -- then
    # union_bbox into box. Returns whether box now holds one.
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
    box[0] = python_min(box[0], sx0)
    box[1] = python_min(box[1], sy0)
    box[2] = python_max(box[2], sx1)
    box[3] = python_max(box[3], sy1)
    return True


cdef int add_curve(PathBuilder path, const double *values, hypot) except -1:
    # values: the start point, two controls and the end point, then the
    # linear part of the curve's CTM (a, b, c, d) and its flatness.
    cdef double x0 = values[0], y0 = values[1], x1 = values[2], y1 = values[3]
    cdef double x2 = values[4], y2 = values[5], x3 = values[6], y3 = values[7]
    cdef double scale = python_max(
        python_max(<double> hypot(values[8], values[9]), <double> hypot(values[10], values[11])),
        1.0,
    )
    cdef double control_len = (
        <double> hypot(x1 - x0, y1 - y0)
        + <double> hypot(x2 - x1, y2 - y1)
        + <double> hypot(x3 - x2, y3 - y2)
    )
    # flatness or 0.25: either zero is falsy, a NaN is not.
    cdef double flatness_value = values[12] if values[12] != 0.0 else 0.25
    cdef double flatness = python_max(0.1, flatness_value)
    cdef double steps = ceil(control_len * scale / (flatness * 8.0))
    check_integral(steps)
    # max(4, min(128, steps)), each keeping its first argument on a tie.
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


def flatten_path_commands(
    const unsigned char[::1] ops,
    const double[::1] coords,
    matrix,
    hypot,
    array.array line_rows,
    double line_width,
):
    """Flatten, transform and summarize a path, as capture needs it.

    ``ops`` and ``coords`` are a PdfPath's operator bytes and numbers, laid
    out as core_pdf_spec's PATH_OPERAND_COUNTS says; ``matrix`` is the
    six-number CTM to apply to every point, or None to leave them as
    flattened; ``hypot`` is math.hypot.

    Returns ``(xs, ys, spans, bbox, has_segments)``: float64 point columns,
    a ``(start, end, closed)`` span per subpath, the bounding box or None, and
    whether any subpath has two points.

    The stroke lines -- every consecutive pair of points within a subpath
    that moves by more than 0.01 on either axis -- are appended to
    ``line_rows``, an ``array('d')``, as ``x0, y0, x1, y1, line_width`` rows;
    it is the page's one line table, so a path adds no array of its own. With
    ``line_rows`` None they are not collected.
    """
    cdef PathBuilder path = PathBuilder()
    cdef Py_ssize_t op_index, at = 0, available = coords.shape[0]
    cdef unsigned char op
    for op_index in range(ops.shape[0]):
        op = ops[op_index]
        if op == 109:  # m
            if at + 2 > available:
                raise ValueError("path coordinates run short")
            path.move_to(coords[at], coords[at + 1])
            at += 2
        elif op == 108:  # l
            if at + 2 > available:
                raise ValueError("path coordinates run short")
            path.line_to(coords[at], coords[at + 1])
            at += 2
        elif op == 104:  # h
            path.close()
        elif op == 114:  # re
            if at + 4 > available:
                raise ValueError("path coordinates run short")
            path.rect(coords[at], coords[at + 1], coords[at + 2], coords[at + 3])
            at += 4
        elif op == 99:  # c
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

    xs = numpy.empty(count, dtype=numpy.float64)
    ys = numpy.empty(count, dtype=numpy.float64)
    cdef double[::1] out_x = xs
    cdef double[::1] out_y = ys
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
    """CapturedPath.bbox() and has_segments() of the subpaths `spans` cut from the columns.

    A deferred path answers both without building its subpaths: bbox_union
    over each subpath's points_bbox, as add_subpath_box computes it for
    flatten_path_commands, and whether any subpath has two points.
    """
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
