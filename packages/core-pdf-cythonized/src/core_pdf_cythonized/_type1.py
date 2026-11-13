# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
from cython.cimports.core_pdf_cythonized._type1 import T1_MAX_DEPTH, T1_STACK, fabs, sqrt
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc, PyMem_RawFree, PyMem_RawRealloc
from cython.cimports.libc.math import pow

__all__ = ("decrypt_type1", "type1_glyph_bounds", "type1_glyph_contours")


def decrypt_type1(data: cython.const[cython.uchar][::1], key: cython.int):
    n: cython.Py_ssize_t = data.shape[0]
    i: cython.Py_ssize_t
    result = bytearray(n)
    out: cython.uchar[::1] = result
    state: cython.uint = key & 0xFFFF
    cipher: cython.uint
    with cython.nogil:
        for i in range(n):
            cipher = data[i]
            out[i] = cython.cast(cython.uchar, (cipher ^ (state >> 8)))
            state = ((cipher + state) * 52845 + 22719) & 0xFFFF
    return bytes(result)


# type1_glyph_bounds mirrors fontTools' T1CharString.draw into a BoundsPen behind
# a TransformPen, which core_pdf.impl.fonts.program_type1 keeps for every input
# declined here: the arithmetic below is theirs, operation for operation, so the
# bounds are the same doubles. type1_glyph_contours runs the same interpreter
# collecting the outline instead, as core's Type1Outline records it and
# recording_to_contours flattens it.

T1_OK = cython.declare(cython.int, 0)
T1_STOP = cython.declare(cython.int, 1)
T1_DECLINE = cython.declare(cython.int, -1)
QUADRATIC_EPSILON = cython.declare(cython.double, 1e-10)
SIGN_BIT = cython.declare(cython.longlong, 1 << 31)
WORD_RANGE = cython.declare(cython.longlong, 1 << 32)


T1State = cython.struct(
    stack=cython.double[256],
    sp=cython.int,
    x=cython.double,
    y=cython.double,
    saw_move=cython.bint,
    flexing=cython.bint,
    pen_x=cython.double,
    pen_y=cython.double,
    has_bounds=cython.bint,
    min_x=cython.double,
    min_y=cython.double,
    max_x=cython.double,
    max_y=cython.double,
    m=cython.double[6],
    subrs=cython.pp_const_uchar,
    subr_lengths=cython.p_Py_ssize_t,
    subr_count=cython.Py_ssize_t,
    # Collecting contours: the points, where the open contour starts, and where each
    # closed contour ends.
    collect=cython.bint,
    open_contour=cython.bint,
    failed=cython.bint,
    xs=cython.p_double,
    ys=cython.p_double,
    point_count=cython.Py_ssize_t,
    point_capacity=cython.Py_ssize_t,
    contour_start=cython.Py_ssize_t,
    ends=cython.p_Py_ssize_t,
    end_count=cython.Py_ssize_t,
    end_capacity=cython.Py_ssize_t,
)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def collect_point(s: cython.pointer[T1State], x: cython.double, y: cython.double) -> cython.void:
    capacity: cython.Py_ssize_t
    xs: cython.p_double
    ys: cython.p_double
    if s.failed:
        return
    if s.point_count == s.point_capacity:
        capacity = s.point_capacity * 2 if s.point_capacity else 64
        xs = cython.cast(
            cython.p_double, PyMem_RawRealloc(s.xs, capacity * cython.sizeof(cython.double))
        )
        if xs == cython.NULL:
            s.failed = True
            return
        s.xs = xs
        ys = cython.cast(
            cython.p_double, PyMem_RawRealloc(s.ys, capacity * cython.sizeof(cython.double))
        )
        if ys == cython.NULL:
            s.failed = True
            return
        s.ys = ys
        s.point_capacity = capacity
    s.xs[s.point_count] = x
    s.ys[s.point_count] = y
    s.point_count += 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def close_contour(s: cython.pointer[T1State]) -> cython.void:
    # recording_to_contours' close_contour: the start point again unless it ends there.
    start: cython.Py_ssize_t = s.contour_start
    last: cython.Py_ssize_t
    capacity: cython.Py_ssize_t
    ends: cython.p_Py_ssize_t
    if not s.open_contour:
        return
    s.open_contour = False
    if s.failed or s.point_count == start:
        return
    last = s.point_count - 1
    if s.xs[start] != s.xs[last] or s.ys[start] != s.ys[last]:
        collect_point(s, s.xs[start], s.ys[start])
    if s.end_count == s.end_capacity:
        capacity = s.end_capacity * 2 if s.end_capacity else 16
        ends = cython.cast(
            cython.p_Py_ssize_t,
            PyMem_RawRealloc(s.ends, capacity * cython.sizeof(cython.Py_ssize_t)),
        )
        if ends == cython.NULL:
            s.failed = True
            return
        s.ends = ends
        s.end_capacity = capacity
    s.ends[s.end_count] = s.point_count
    s.end_count += 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def end_path(s: cython.pointer[T1State]) -> cython.void:
    # Type1Outline's end of a path: a closePath or endPath after a move closes the contour.
    if s.collect and s.saw_move:
        close_contour(s)
    s.saw_move = False


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def add_point(s: cython.pointer[T1State], x: cython.double, y: cython.double) -> cython.void:
    if not s.has_bounds:
        s.min_x = x
        s.min_y = y
        s.max_x = x
        s.max_y = y
        s.has_bounds = True
        return
    if x < s.min_x:
        s.min_x = x
    if y < s.min_y:
        s.min_y = y
    if x > s.max_x:
        s.max_x = x
    if y > s.max_y:
        s.max_y = y


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def transform(
    s: cython.pointer[T1State], x: cython.double, y: cython.double, out: cython.p_double
) -> cython.void:
    out[0] = s.m[0] * x + s.m[2] * y + s.m[4]
    out[1] = s.m[1] * x + s.m[3] * y + s.m[5]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def quadratic_roots(
    a: cython.double, b: cython.double, c: cython.double, roots: cython.p_double
) -> cython.int:
    dd: cython.double
    rdd: cython.double
    if fabs(a) < QUADRATIC_EPSILON:
        if fabs(b) < QUADRATIC_EPSILON:
            return 0
        roots[0] = -c / b
        return 1
    dd = b * b - 4.0 * a * c
    if dd >= 0.0:
        rdd = sqrt(dd)
        roots[0] = (-b + rdd) / 2.0 / a
        roots[1] = (-b - rdd) / 2.0 / a
        return 2
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cubic_bounds(
    x1: cython.double,
    y1: cython.double,
    x2: cython.double,
    y2: cython.double,
    x3: cython.double,
    y3: cython.double,
    x4: cython.double,
    y4: cython.double,
    out: cython.p_double,
) -> cython.void:
    cx: cython.double = (x2 - x1) * 3.0
    cy: cython.double = (y2 - y1) * 3.0
    bx: cython.double = (x3 - x2) * 3.0 - cx
    by: cython.double = (y3 - y2) * 3.0 - cy
    ax: cython.double = x4 - x1 - cx - bx
    ay: cython.double = y4 - y1 - cy - by
    solved = cython.declare(cython.double[2])
    roots = cython.declare(cython.double[4])
    count: cython.int = 0
    found: cython.int
    i: cython.int
    t: cython.double
    px: cython.double
    py: cython.double
    found = quadratic_roots(ax * 3.0, bx * 2.0, cx, solved)
    for i in range(found):
        if 0 <= solved[i] < 1:
            roots[count] = solved[i]
            count += 1
    found = quadratic_roots(ay * 3.0, by * 2.0, cy, solved)
    for i in range(found):
        if 0 <= solved[i] < 1:
            roots[count] = solved[i]
            count += 1
    first: cython.bint = True
    for i in range(count + 2):
        if i < count:
            t = roots[i]
            px = ax * t * t * t + bx * t * t + cx * t + x1
            py = ay * t * t * t + by * t * t + cy * t + y1
        elif i == count:
            px = x1
            py = y1
        else:
            px = x4
            py = y4
        if first:
            out[0] = px
            out[1] = py
            out[2] = px
            out[3] = py
            first = False
            continue
        if px < out[0]:
            out[0] = px
        if py < out[1]:
            out[1] = py
        if px > out[2]:
            out[2] = px
        if py > out[3]:
            out[3] = py


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def pen_move(s: cython.pointer[T1State]) -> cython.void:
    p = cython.declare(cython.double[2])
    if s.collect:
        close_contour(s)
        s.contour_start = s.point_count
        s.open_contour = True
        collect_point(s, s.x, s.y)
        return
    transform(s, s.x, s.y, p)
    add_point(s, p[0], p[1])
    s.pen_x = p[0]
    s.pen_y = p[1]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def pen_line(s: cython.pointer[T1State]) -> cython.void:
    p = cython.declare(cython.double[2])
    if s.collect:
        collect_point(s, s.x, s.y)
        return
    transform(s, s.x, s.y, p)
    add_point(s, p[0], p[1])
    s.pen_x = p[0]
    s.pen_y = p[1]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def r_move(s: cython.pointer[T1State], dx: cython.double, dy: cython.double) -> cython.void:
    s.x = s.x + dx
    s.y = s.y + dy
    pen_move(s)
    s.saw_move = True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def r_line(s: cython.pointer[T1State], dx: cython.double, dy: cython.double) -> cython.void:
    if not s.saw_move:
        r_move(s, 0.0, 0.0)
    s.x = s.x + dx
    s.y = s.y + dy
    pen_line(s)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def collect_curve(
    s: cython.pointer[T1State],
    dx1: cython.double,
    dy1: cython.double,
    dx2: cython.double,
    dy2: cython.double,
    dx3: cython.double,
    dy3: cython.double,
) -> cython.void:
    # The curve's three points, then flatten_cubic's eight: cubic_point at i / 8.
    x0: cython.double = s.x
    y0: cython.double = s.y
    s.x = s.x + dx1
    s.y = s.y + dy1
    x1: cython.double = s.x
    y1: cython.double = s.y
    s.x = s.x + dx2
    s.y = s.y + dy2
    x2: cython.double = s.x
    y2: cython.double = s.y
    s.x = s.x + dx3
    s.y = s.y + dy3
    x3: cython.double = s.x
    y3: cython.double = s.y
    i: cython.int
    t: cython.double
    mt: cython.double
    mt3: cython.double
    t3: cython.double
    mt2t: cython.double
    mtt2: cython.double
    for i in range(1, 9):
        t = i / 8.0
        mt = 1.0 - t
        mt3 = pow(mt, 3.0)
        t3 = pow(t, 3.0)
        mt2t = 3.0 * mt * mt * t
        mtt2 = 3.0 * mt * t * t
        collect_point(
            s,
            mt3 * x0 + mt2t * x1 + mtt2 * x2 + t3 * x3,
            mt3 * y0 + mt2t * y1 + mtt2 * y2 + t3 * y3,
        )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def r_curve(
    s: cython.pointer[T1State],
    dx1: cython.double,
    dy1: cython.double,
    dx2: cython.double,
    dy2: cython.double,
    dx3: cython.double,
    dy3: cython.double,
) -> cython.void:
    c1 = cython.declare(cython.double[2])
    c2 = cython.declare(cython.double[2])
    p3 = cython.declare(cython.double[2])
    box = cython.declare(cython.double[4])
    if not s.saw_move:
        r_move(s, 0.0, 0.0)
    if s.collect:
        collect_curve(s, dx1, dy1, dx2, dy2, dx3, dy3)
        return
    s.x = s.x + dx1
    s.y = s.y + dy1
    transform(s, s.x, s.y, c1)
    s.x = s.x + dx2
    s.y = s.y + dy2
    transform(s, s.x, s.y, c2)
    s.x = s.x + dx3
    s.y = s.y + dy3
    transform(s, s.x, s.y, p3)
    add_point(s, p3[0], p3[1])
    if not (s.min_x <= c1[0] <= s.max_x and s.min_y <= c1[1] <= s.max_y) or not (
        s.min_x <= c2[0] <= s.max_x and s.min_y <= c2[1] <= s.max_y
    ):
        cubic_bounds(s.pen_x, s.pen_y, c1[0], c1[1], c2[0], c2[1], p3[0], p3[1], box)
        if box[0] < s.min_x:
            s.min_x = box[0]
        if box[1] < s.min_y:
            s.min_y = box[1]
        if box[2] > s.max_x:
            s.max_x = box[2]
        if box[3] > s.max_y:
            s.max_y = box[3]
    s.pen_x = p3[0]
    s.pen_y = p3[1]


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def push_value(s: cython.pointer[T1State], value: cython.double) -> cython.int:
    if s.sp >= T1_STACK:
        return T1_DECLINE
    s.stack[s.sp] = value
    s.sp += 1
    return T1_OK


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def alternating_lines(s: cython.pointer[T1State], horizontal: cython.bint) -> cython.int:
    i: cython.int
    for i in range(s.sp):
        if horizontal:
            r_line(s, s.stack[i], 0.0)
        else:
            r_line(s, 0.0, s.stack[i])
        horizontal = not horizontal
    s.sp = 0
    return T1_OK


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def rr_curves(s: cython.pointer[T1State]) -> cython.int:
    i: cython.int
    a: cython.p_double = s.stack
    if s.sp % 6:
        return T1_DECLINE
    for i in range(0, s.sp, 6):
        r_curve(s, a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4], a[i + 5])
    s.sp = 0
    return T1_OK


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def alternating_curves(s: cython.pointer[T1State], vertical_first: cython.bint) -> cython.int:
    i: cython.int = 0
    remaining: cython.int
    a: cython.p_double = s.stack
    last: cython.double
    vertical: cython.bint = vertical_first
    while i < s.sp:
        if s.sp - i < 4:
            return T1_DECLINE
        remaining = s.sp - i - 4
        last = a[i + 4] if remaining == 1 else 0.0
        if vertical:
            r_curve(s, 0.0, a[i], a[i + 1], a[i + 2], a[i + 3], last)
        else:
            r_curve(s, a[i], 0.0, a[i + 1], a[i + 2], last, a[i + 3])
        i += 5 if remaining == 1 else 4
        vertical = not vertical
    s.sp = 0
    return T1_OK


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def do_flex(s: cython.pointer[T1State]) -> cython.int:
    final_x: cython.double
    final_y: cython.double
    v = cython.declare(cython.double[17])
    i: cython.int
    base: cython.int
    if s.sp < 17:
        return T1_DECLINE
    base = s.sp - 17
    for i in range(17):
        v[i] = s.stack[base + i]
    s.sp = base
    # v: rpx rpy bcp1x bcp1y bcp2x bcp2y p2x p2y bcp3x bcp3y bcp4x bcp4y p3x p3y height
    #    finalx finaly
    final_x = v[15]
    final_y = v[16]
    if (
        push_value(s, v[2] + v[0])
        or push_value(s, v[3] + v[1])
        or push_value(s, v[4])
        or push_value(s, v[5])
        or push_value(s, v[6])
        or push_value(s, v[7])
    ):
        return T1_DECLINE
    if rr_curves(s) != T1_OK:
        return T1_DECLINE
    for i in range(8, 14):
        if push_value(s, v[i]) != T1_OK:
            return T1_DECLINE
    if rr_curves(s) != T1_OK:
        return T1_DECLINE
    if push_value(s, final_x) or push_value(s, final_y):
        return T1_DECLINE
    return T1_OK


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def subr_index(s: cython.pointer[T1State], out: cython.p_Py_ssize_t) -> cython.int:
    value: cython.double
    if s.sp < 1:
        return T1_DECLINE
    s.sp -= 1
    value = s.stack[s.sp]
    if (
        value != cython.cast(cython.double, cython.cast(cython.longlong, value))
        or value < 0
        or value >= s.subr_count
    ):
        return T1_DECLINE
    out[0] = cython.cast(cython.Py_ssize_t, value)
    return T1_OK


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def execute(
    s: cython.pointer[T1State],
    program: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    depth: cython.int,
) -> cython.int:
    pos: cython.Py_ssize_t = 0
    index = cython.declare(cython.Py_ssize_t)
    op: cython.int
    status: cython.int
    b0: cython.uint
    b1: cython.uint
    word: cython.longlong
    num1: cython.double
    num2: cython.double
    other: cython.double
    count: cython.double
    if depth > T1_MAX_DEPTH:
        return T1_DECLINE
    while pos < length:
        b0 = program[pos]
        pos += 1
        if b0 >= 32:
            if b0 <= 246:
                status = push_value(
                    s, cython.cast(cython.double, (cython.cast(cython.int, b0) - 139))
                )
            elif b0 <= 250:
                if pos >= length:
                    return T1_DECLINE
                b1 = program[pos]
                pos += 1
                status = push_value(
                    s,
                    cython.cast(
                        cython.double,
                        (
                            (cython.cast(cython.int, b0) - 247) * 256
                            + cython.cast(cython.int, b1)
                            + 108
                        ),
                    ),
                )
            elif b0 <= 254:
                if pos >= length:
                    return T1_DECLINE
                b1 = program[pos]
                pos += 1
                status = push_value(
                    s,
                    cython.cast(
                        cython.double,
                        (
                            -(cython.cast(cython.int, b0) - 251) * 256
                            - cython.cast(cython.int, b1)
                            - 108
                        ),
                    ),
                )
            else:
                if pos + 4 > length:
                    return T1_DECLINE
                word = (
                    (cython.cast(cython.longlong, program[pos]) << 24)
                    | (cython.cast(cython.longlong, program[pos + 1]) << 16)
                    | (cython.cast(cython.longlong, program[pos + 2]) << 8)
                    | cython.cast(cython.longlong, program[pos + 3])
                )
                if word >= SIGN_BIT:
                    word -= WORD_RANGE
                pos += 4
                status = push_value(s, cython.cast(cython.double, word))
            if status != T1_OK:
                return T1_DECLINE
            continue
        if b0 == 12:
            if pos >= length:
                return T1_DECLINE
            op = 1200 + program[pos]
            pos += 1
        else:
            op = b0
        if op == 1 or op == 3 or op == 1200 or op == 1201 or op == 1202 or op == 1207:
            s.sp = 0
        elif op == 4:
            if s.flexing:
                if push_value(s, 0.0) != T1_OK or s.sp < 2:
                    return T1_DECLINE
                num1 = s.stack[s.sp - 1]
                s.stack[s.sp - 1] = s.stack[s.sp - 2]
                s.stack[s.sp - 2] = num1
                continue
            end_path(s)
            if s.sp < 1:
                return T1_DECLINE
            num1 = s.stack[0]
            s.sp = 0
            r_move(s, 0.0, num1)
        elif op == 5:
            if s.sp % 2:
                return T1_DECLINE
            for index in range(0, s.sp, 2):
                r_line(s, s.stack[index], s.stack[index + 1])
            s.sp = 0
        elif op == 6 or op == 7:
            alternating_lines(s, op == 6)
        elif op == 8:
            if rr_curves(s) != T1_OK:
                return T1_DECLINE
        elif op == 9:
            end_path(s)
        elif op == 10:
            if subr_index(s, cython.address(index)) != T1_OK:
                return T1_DECLINE
            status = execute(s, s.subrs[index], s.subr_lengths[index], depth + 1)
            if status == T1_DECLINE:
                return T1_DECLINE
        elif op == 11 or op == 1217:
            pass
        elif op == 13:
            if s.sp != 2:
                return T1_DECLINE
            s.x = s.stack[0]
            s.sp = 0
        elif op == 14:
            end_path(s)
        elif op == 21:
            if s.flexing:
                continue
            end_path(s)
            if s.sp < 2:
                return T1_DECLINE
            num1 = s.stack[0]
            num2 = s.stack[1]
            s.sp = 0
            r_move(s, num1, num2)
        elif op == 22:
            if s.flexing:
                if push_value(s, 0.0) != T1_OK:
                    return T1_DECLINE
                continue
            end_path(s)
            if s.sp < 1:
                return T1_DECLINE
            num1 = s.stack[0]
            s.sp = 0
            r_move(s, num1, 0.0)
        elif op == 30 or op == 31:
            if alternating_curves(s, op == 30) != T1_OK:
                return T1_DECLINE
        elif op == 1212:
            if s.sp < 2:
                return T1_DECLINE
            num2 = s.stack[s.sp - 1]
            num1 = s.stack[s.sp - 2]
            if num2 == 0.0:
                return T1_DECLINE
            s.sp -= 2
            if push_value(s, num1 / num2) != T1_OK:
                return T1_DECLINE
        elif op == 1216:
            if s.sp < 2:
                return T1_DECLINE
            other = s.stack[s.sp - 1]
            count = s.stack[s.sp - 2]
            s.sp -= 2
            if other == 0.0 and count == 3.0:
                if do_flex(s) != T1_OK:
                    return T1_DECLINE
                s.flexing = False
            elif other == 1.0 and count == 0.0:
                s.flexing = True
        elif op == 1233:
            if s.sp != 2:
                return T1_DECLINE
            s.x = s.stack[0]
            s.y = s.stack[1]
            s.sp = 0
        elif op == 1206:
            return T1_DECLINE
        else:
            return T1_STOP
    return T1_OK


def type1_glyph_bounds(
    charstring: typing.Optional[bytes],
    subrs: typing.Optional[tuple],
    matrix: typing.Optional[tuple],
):
    """Bounds of a Type 1 charstring drawn through matrix, as fontTools' BoundsPen
    behind a TransformPen gives them.

    Returns the (x_min, y_min, x_max, y_max) tuple, () when nothing is drawn, or
    None for input it does not own -- seac, a malformed operand count, a subr
    index it cannot resolve as Python would, division by zero, a truncated
    number -- which the caller draws in Python.
    """
    state = cython.declare(T1State)
    count: cython.Py_ssize_t = len(subrs)
    i: cython.Py_ssize_t
    status: cython.int
    subr_data: cython.pp_const_uchar = cython.NULL
    subr_lengths: cython.p_Py_ssize_t = cython.NULL
    if len(matrix) != 6:
        return None
    for i in range(6):
        state.m[i] = matrix[i]
    state.sp = 0
    state.x = 0.0
    state.y = 0.0
    state.saw_move = False
    state.flexing = False
    state.pen_x = 0.0
    state.pen_y = 0.0
    state.has_bounds = False
    state.min_x = state.min_y = state.max_x = state.max_y = 0.0
    state.collect = False
    state.subr_count = count
    if count:
        subr_data = cython.cast(
            cython.pp_const_uchar, PyMem_Malloc(count * cython.sizeof(cython.p_const_uchar))
        )
        subr_lengths = cython.cast(
            cython.p_Py_ssize_t, PyMem_Malloc(count * cython.sizeof(cython.Py_ssize_t))
        )
        if subr_data == cython.NULL or subr_lengths == cython.NULL:
            PyMem_Free(subr_data)
            PyMem_Free(subr_lengths)
            raise MemoryError()
    try:
        for i in range(count):
            subr = subrs[i]
            if type(subr) is not bytes:
                return None
            subr_data[i] = cython.cast(cython.p_const_uchar, cython.cast(bytes, subr))
            subr_lengths[i] = len(cython.cast(bytes, subr))
        state.subrs = subr_data
        state.subr_lengths = subr_lengths
        status = execute(
            cython.address(state), cython.cast(cython.p_const_uchar, charstring), len(charstring), 0
        )
    finally:
        PyMem_Free(subr_data)
        PyMem_Free(subr_lengths)
    if status == T1_DECLINE:
        return None
    if not state.has_bounds:
        return ()
    return (state.min_x, state.min_y, state.max_x, state.max_y)


def type1_glyph_contours(
    charstring: typing.Optional[bytes],
    subrs: typing.Optional[tuple],
    matrix: typing.Optional[tuple],
):
    """A Type 1 charstring's contours, as core's Type1Outline records them,
    recording_to_contours flattens them (each curve in eight steps, each contour closed
    back to its start, those under three points dropped) and transform_contours maps them
    through the font matrix into a 1000-unit em.

    Returns a tuple of contours, each a tuple of (x, y) points, or None for input it
    declines as type1_glyph_bounds does, which the caller draws in Python.
    """
    state = cython.declare(T1State)
    count: cython.Py_ssize_t = len(subrs)
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    status: cython.int
    subr_data: cython.pp_const_uchar = cython.NULL
    subr_lengths: cython.p_Py_ssize_t = cython.NULL
    if len(matrix) != 6:
        return None
    a: cython.double = matrix[0]
    b: cython.double = matrix[1]
    c: cython.double = matrix[2]
    d: cython.double = matrix[3]
    e: cython.double = matrix[4]
    f: cython.double = matrix[5]
    state.sp = 0
    state.x = 0.0
    state.y = 0.0
    state.saw_move = False
    state.flexing = False
    state.pen_x = 0.0
    state.pen_y = 0.0
    state.has_bounds = False
    state.collect = True
    state.open_contour = False
    state.failed = False
    state.xs = cython.NULL
    state.ys = cython.NULL
    state.point_count = 0
    state.point_capacity = 0
    state.contour_start = 0
    state.ends = cython.NULL
    state.end_count = 0
    state.end_capacity = 0
    state.subr_count = count
    if count:
        subr_data = cython.cast(
            cython.pp_const_uchar, PyMem_Malloc(count * cython.sizeof(cython.p_const_uchar))
        )
        subr_lengths = cython.cast(
            cython.p_Py_ssize_t, PyMem_Malloc(count * cython.sizeof(cython.Py_ssize_t))
        )
        if subr_data == cython.NULL or subr_lengths == cython.NULL:
            PyMem_Free(subr_data)
            PyMem_Free(subr_lengths)
            raise MemoryError()
    contours: list = []
    try:
        for i in range(count):
            subr = subrs[i]
            if type(subr) is not bytes:
                return None
            subr_data[i] = cython.cast(cython.p_const_uchar, cython.cast(bytes, subr))
            subr_lengths[i] = len(cython.cast(bytes, subr))
        state.subrs = subr_data
        state.subr_lengths = subr_lengths
        status = execute(
            cython.address(state), cython.cast(cython.p_const_uchar, charstring), len(charstring), 0
        )
        if status == T1_DECLINE:
            return None
        end_path(cython.address(state))
        if state.failed:
            raise MemoryError()
        start: cython.Py_ssize_t = 0
        end: cython.Py_ssize_t
        x: cython.double
        y: cython.double
        for i in range(state.end_count):
            end = state.ends[i]
            if end - start >= 3:
                points: list = []
                for j in range(start, end):
                    x = state.xs[j]
                    y = state.ys[j]
                    points.append(((x * a + y * c + e) * 1000.0, (x * b + y * d + f) * 1000.0))
                contours.append(tuple(points))
            start = end
    finally:
        PyMem_Free(subr_data)
        PyMem_Free(subr_lengths)
        PyMem_RawFree(state.xs)
        PyMem_RawFree(state.ys)
        PyMem_RawFree(state.ends)
    return tuple(contours)
