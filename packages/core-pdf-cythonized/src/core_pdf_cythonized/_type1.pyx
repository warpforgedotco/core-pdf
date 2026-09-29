# SPDX-License-Identifier: AGPL-3.0-only

from cpython.mem cimport PyMem_Free, PyMem_Malloc

__all__ = ("decrypt_type1", "type1_glyph_bounds")


def decrypt_type1(const unsigned char[::1] data, int key):
    cdef Py_ssize_t n = data.shape[0], i
    result = bytearray(n)
    cdef unsigned char[::1] out = result
    cdef unsigned int state = key & 0xFFFF
    cdef unsigned int cipher
    with nogil:
        for i in range(n):
            cipher = data[i]
            out[i] = <unsigned char> (cipher ^ (state >> 8))
            state = ((cipher + state) * 52845 + 22719) & 0xFFFF
    return bytes(result)


# type1_glyph_bounds mirrors fontTools' T1CharString.draw into a BoundsPen behind
# a TransformPen, which core_pdf.impl.fonts_program_type1 keeps for every input
# declined here: the arithmetic below is theirs, operation for operation, so the
# bounds are the same doubles.

cdef extern from "math.h" nogil:
    double sqrt(double)
    double fabs(double)

cdef enum:
    T1_STACK = 256
    T1_MAX_DEPTH = 64

cdef int T1_OK = 0
cdef int T1_STOP = 1
cdef int T1_DECLINE = -1
cdef double QUADRATIC_EPSILON = 1e-10
cdef long long SIGN_BIT = 1LL << 31
cdef long long WORD_RANGE = 1LL << 32


cdef struct T1State:
    double stack[256]
    int sp
    double x
    double y
    bint saw_move
    bint flexing
    double pen_x
    double pen_y
    bint has_bounds
    double min_x
    double min_y
    double max_x
    double max_y
    double m[6]
    const unsigned char **subrs
    Py_ssize_t *subr_lengths
    Py_ssize_t subr_count


cdef inline void add_point(T1State *s, double x, double y) noexcept nogil:
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


cdef inline void transform(T1State *s, double x, double y, double *out) noexcept nogil:
    out[0] = s.m[0] * x + s.m[2] * y + s.m[4]
    out[1] = s.m[1] * x + s.m[3] * y + s.m[5]


cdef int quadratic_roots(double a, double b, double c, double *roots) noexcept nogil:
    cdef double dd, rdd
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


cdef void cubic_bounds(double x1, double y1, double x2, double y2, double x3, double y3,
                       double x4, double y4, double *out) noexcept nogil:
    cdef double cx = (x2 - x1) * 3.0
    cdef double cy = (y2 - y1) * 3.0
    cdef double bx = (x3 - x2) * 3.0 - cx
    cdef double by = (y3 - y2) * 3.0 - cy
    cdef double ax = x4 - x1 - cx - bx
    cdef double ay = y4 - y1 - cy - by
    cdef double solved[2]
    cdef double roots[4]
    cdef int count = 0, found, i
    cdef double t, px, py
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
    cdef bint first = True
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


cdef void pen_move(T1State *s) noexcept nogil:
    cdef double p[2]
    transform(s, s.x, s.y, p)
    add_point(s, p[0], p[1])
    s.pen_x = p[0]
    s.pen_y = p[1]


cdef void pen_line(T1State *s) noexcept nogil:
    cdef double p[2]
    transform(s, s.x, s.y, p)
    add_point(s, p[0], p[1])
    s.pen_x = p[0]
    s.pen_y = p[1]


cdef void r_move(T1State *s, double dx, double dy) noexcept nogil:
    s.x = s.x + dx
    s.y = s.y + dy
    pen_move(s)
    s.saw_move = True


cdef void r_line(T1State *s, double dx, double dy) noexcept nogil:
    if not s.saw_move:
        r_move(s, 0.0, 0.0)
    s.x = s.x + dx
    s.y = s.y + dy
    pen_line(s)


cdef void r_curve(T1State *s, double dx1, double dy1, double dx2, double dy2,
                  double dx3, double dy3) noexcept nogil:
    cdef double c1[2]
    cdef double c2[2]
    cdef double p3[2]
    cdef double box[4]
    if not s.saw_move:
        r_move(s, 0.0, 0.0)
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
    if not (
        s.min_x <= c1[0] <= s.max_x and s.min_y <= c1[1] <= s.max_y
    ) or not (
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


cdef inline int push_value(T1State *s, double value) noexcept nogil:
    if s.sp >= T1_STACK:
        return T1_DECLINE
    s.stack[s.sp] = value
    s.sp += 1
    return T1_OK


cdef int alternating_lines(T1State *s, bint horizontal) noexcept nogil:
    cdef int i
    for i in range(s.sp):
        if horizontal:
            r_line(s, s.stack[i], 0.0)
        else:
            r_line(s, 0.0, s.stack[i])
        horizontal = not horizontal
    s.sp = 0
    return T1_OK


cdef int rr_curves(T1State *s) noexcept nogil:
    cdef int i
    cdef double *a = s.stack
    if s.sp % 6:
        return T1_DECLINE
    for i in range(0, s.sp, 6):
        r_curve(s, a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4], a[i + 5])
    s.sp = 0
    return T1_OK


cdef int alternating_curves(T1State *s, bint vertical_first) noexcept nogil:
    cdef int i = 0, remaining
    cdef double *a = s.stack
    cdef double last
    cdef bint vertical = vertical_first
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


cdef int do_flex(T1State *s) noexcept nogil:
    cdef double *a
    cdef double final_x, final_y
    cdef double v[17]
    cdef int i, base
    if s.sp < 17:
        return T1_DECLINE
    base = s.sp - 17
    for i in range(17):
        v[i] = s.stack[base + i]
    s.sp = base
    # v: rpx rpy bcp1x bcp1y bcp2x bcp2y p2x p2y bcp3x bcp3y bcp4x bcp4y p3x p3y height finalx finaly
    final_x = v[15]
    final_y = v[16]
    if (
        push_value(s, v[2] + v[0]) or push_value(s, v[3] + v[1])
        or push_value(s, v[4]) or push_value(s, v[5])
        or push_value(s, v[6]) or push_value(s, v[7])
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


cdef int subr_index(T1State *s, Py_ssize_t *out) noexcept nogil:
    cdef double value
    if s.sp < 1:
        return T1_DECLINE
    s.sp -= 1
    value = s.stack[s.sp]
    if value != <double> <long long> value or value < 0 or value >= s.subr_count:
        return T1_DECLINE
    out[0] = <Py_ssize_t> value
    return T1_OK


cdef int execute(T1State *s, const unsigned char *program, Py_ssize_t length,
                 int depth) noexcept nogil:
    cdef Py_ssize_t pos = 0, index
    cdef int op, status
    cdef unsigned int b0, b1
    cdef long long word
    cdef double num1, num2, other, count
    if depth > T1_MAX_DEPTH:
        return T1_DECLINE
    while pos < length:
        b0 = program[pos]
        pos += 1
        if b0 >= 32:
            if b0 <= 246:
                status = push_value(s, <double> (<int> b0 - 139))
            elif b0 <= 250:
                if pos >= length:
                    return T1_DECLINE
                b1 = program[pos]
                pos += 1
                status = push_value(s, <double> ((<int> b0 - 247) * 256 + <int> b1 + 108))
            elif b0 <= 254:
                if pos >= length:
                    return T1_DECLINE
                b1 = program[pos]
                pos += 1
                status = push_value(s, <double> (-(<int> b0 - 251) * 256 - <int> b1 - 108))
            else:
                if pos + 4 > length:
                    return T1_DECLINE
                word = (
                    (<long long> program[pos] << 24) | (<long long> program[pos + 1] << 16)
                    | (<long long> program[pos + 2] << 8) | <long long> program[pos + 3]
                )
                if word >= SIGN_BIT:
                    word -= WORD_RANGE
                pos += 4
                status = push_value(s, <double> word)
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
            s.saw_move = False
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
            s.saw_move = False
        elif op == 10:
            if subr_index(s, &index) != T1_OK:
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
            s.saw_move = False
        elif op == 21:
            if s.flexing:
                continue
            s.saw_move = False
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
            s.saw_move = False
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


def type1_glyph_bounds(bytes charstring, tuple subrs, tuple matrix):
    """Bounds of a Type 1 charstring drawn through matrix, as fontTools' BoundsPen
    behind a TransformPen gives them.

    Returns the (x_min, y_min, x_max, y_max) tuple, () when nothing is drawn, or
    None for input it does not own -- seac, a malformed operand count, a subr
    index it cannot resolve as Python would, division by zero, a truncated
    number -- which the caller draws in Python.
    """
    cdef T1State state
    cdef Py_ssize_t count = len(subrs), i
    cdef int status
    cdef const unsigned char **subr_data = NULL
    cdef Py_ssize_t *subr_lengths = NULL
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
    state.subr_count = count
    if count:
        subr_data = <const unsigned char **> PyMem_Malloc(count * sizeof(const unsigned char *))
        subr_lengths = <Py_ssize_t *> PyMem_Malloc(count * sizeof(Py_ssize_t))
        if subr_data == NULL or subr_lengths == NULL:
            PyMem_Free(subr_data)
            PyMem_Free(subr_lengths)
            raise MemoryError()
    try:
        for i in range(count):
            subr = subrs[i]
            if type(subr) is not bytes:
                return None
            subr_data[i] = <const unsigned char *> (<bytes> subr)
            subr_lengths[i] = len(<bytes> subr)
        state.subrs = subr_data
        state.subr_lengths = subr_lengths
        status = execute(&state, <const unsigned char *> charstring, len(charstring), 0)
    finally:
        PyMem_Free(subr_data)
        PyMem_Free(subr_lengths)
    if status == T1_DECLINE:
        return None
    if not state.has_bounds:
        return ()
    return (state.min_x, state.min_y, state.max_x, state.max_y)
