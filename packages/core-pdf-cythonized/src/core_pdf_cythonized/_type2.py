# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._bezier import (
    CUBIC_SAMPLE_CAPACITY,
    extrema,
    sample_times_c,
)
from cython.cimports.libc.math import fabs, isfinite, pow, sqrt
from cython.cimports.libc.stdlib import free, malloc, realloc

MAX_STACK = cython.declare(cython.int, 48)
TRANSIENT_SIZE = cython.declare(cython.int, 32)
MAX_SUBR_DEPTH = cython.declare(cython.int, 10)
RANDOM_INITIAL_STATE = cython.declare(cython.uint, 0x1234ABCD)

INF = cython.declare(cython.double, float("inf"))


Ctx = cython.struct(
    stack=cython.double[48],
    sp=cython.int,
    transient=cython.double[32],
    stem_count=cython.int,
    width_resolved=cython.bint,
    x=cython.double,
    y=cython.double,
    pts=cython.p_double,
    npts=cython.Py_ssize_t,
    pts_cap=cython.Py_ssize_t,
    cur_start=cython.Py_ssize_t,
    spans=cython.p_Py_ssize_t,
    nspans=cython.Py_ssize_t,
    spans_cap=cython.Py_ssize_t,
    cmin_x=cython.double,
    cmin_y=cython.double,
    cmax_x=cython.double,
    cmax_y=cython.double,
    chas=cython.bint,
    bmin_x=cython.double,
    bmin_y=cython.double,
    bmax_x=cython.double,
    bmax_y=cython.double,
    bhas=cython.bint,
    rnd=cython.uint,
    flatten=cython.bint,
    retain=cython.bint,
    lsub=cython.pp_const_uchar,
    llen=cython.p_Py_ssize_t,
    nl=cython.Py_ssize_t,
    gsub=cython.pp_const_uchar,
    glen=cython.p_Py_ssize_t,
    ng=cython.Py_ssize_t,
    lbias=cython.int,
    gbias=cython.int,
    has_seac=cython.bint,
    seac_base=cython.int,
    seac_accent=cython.int,
    seac_dx=cython.double,
    seac_dy=cython.double,
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def subr_bias(count: cython.Py_ssize_t) -> cython.int:
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def push(c: cython.pointer[Ctx], value: cython.double) -> cython.int:
    if c.sp >= MAX_STACK or not isfinite(value):
        return -1
    c.stack[c.sp] = value
    c.sp += 1
    return 0


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def require_integer(value: cython.double, out: cython.p_int) -> cython.int:
    integer: cython.int = cython.cast(cython.int, value)
    if value != cython.cast(cython.double, integer):
        return -1
    out[0] = integer
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def grow_points(c: cython.pointer[Ctx]) -> cython.int:
    cap: cython.Py_ssize_t = c.pts_cap * 2 if c.pts_cap else 256
    grown: cython.p_double = cython.cast(
        cython.p_double, realloc(c.pts, cap * 2 * cython.sizeof(cython.double))
    )
    if grown == cython.NULL:
        return -1
    c.pts = grown
    c.pts_cap = cap
    return 0


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def record_point(c: cython.pointer[Ctx], px: cython.double, py: cython.double) -> cython.int:
    if c.retain:
        if c.npts >= c.pts_cap and grow_points(c) != 0:
            return -1
        c.pts[2 * c.npts] = px
        c.pts[2 * c.npts + 1] = py
        c.npts += 1
    if px < c.cmin_x:
        c.cmin_x = px
    if px > c.cmax_x:
        c.cmax_x = px
    if py < c.cmin_y:
        c.cmin_y = py
    if py > c.cmax_y:
        c.cmax_y = py
    c.chas = True
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def add_span(
    c: cython.pointer[Ctx], start: cython.Py_ssize_t, end: cython.Py_ssize_t
) -> cython.int:
    cap: cython.Py_ssize_t
    grown: cython.p_Py_ssize_t
    if c.nspans * 2 >= c.spans_cap:
        cap = c.spans_cap * 2 if c.spans_cap else 64
        grown = cython.cast(
            cython.p_Py_ssize_t, realloc(c.spans, cap * cython.sizeof(cython.Py_ssize_t))
        )
        if grown == cython.NULL:
            return -1
        c.spans = grown
        c.spans_cap = cap
    c.spans[2 * c.nspans] = start
    c.spans[2 * c.nspans + 1] = end
    c.nspans += 1
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def flush_contour(c: cython.pointer[Ctx]) -> cython.int:
    if c.retain and c.npts > c.cur_start:
        if add_span(c, c.cur_start, c.npts) != 0:
            return -1
        c.cur_start = c.npts
    if c.chas:
        if c.cmin_x < c.bmin_x:
            c.bmin_x = c.cmin_x
        if c.cmin_y < c.bmin_y:
            c.bmin_y = c.cmin_y
        if c.cmax_x > c.bmax_x:
            c.bmax_x = c.cmax_x
        if c.cmax_y > c.bmax_y:
            c.bmax_y = c.cmax_y
        c.bhas = True
        c.cmin_x = INF
        c.cmin_y = INF
        c.cmax_x = -INF
        c.cmax_y = -INF
        c.chas = False
    return 0


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def cubic_point(
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    x2: cython.double,
    y2: cython.double,
    x3: cython.double,
    y3: cython.double,
    t: cython.double,
    out: cython.p_double,
) -> cython.void:
    mt: cython.double = 1.0 - t
    mt3: cython.double = pow(mt, 3.0)
    t3: cython.double = pow(t, 3.0)
    mt2t: cython.double = 3.0 * mt * mt * t
    mtt2: cython.double = 3.0 * mt * t * t
    out[0] = mt3 * x0 + mt2t * x1 + mtt2 * x2 + t3 * x3
    out[1] = mt3 * y0 + mt2t * y1 + mtt2 * y2 + t3 * y3


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def do_move(c: cython.pointer[Ctx], dx: cython.double, dy: cython.double) -> cython.int:
    if flush_contour(c) != 0:
        return -1
    c.x += dx
    c.y += dy
    return record_point(c, c.x, c.y)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def do_line(c: cython.pointer[Ctx], dx: cython.double, dy: cython.double) -> cython.int:
    c.x += dx
    c.y += dy
    return record_point(c, c.x, c.y)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def do_curve(
    c: cython.pointer[Ctx],
    dx1: cython.double,
    dy1: cython.double,
    dx2: cython.double,
    dy2: cython.double,
    dx3: cython.double,
    dy3: cython.double,
) -> cython.int:
    x0: cython.double = c.x
    y0: cython.double = c.y
    x1: cython.double = x0 + dx1
    y1: cython.double = y0 + dy1
    x2: cython.double = x1 + dx2
    y2: cython.double = y1 + dy2
    x3: cython.double = x2 + dx3
    y3: cython.double = y2 + dy3
    buf = cython.declare(cython.double[CUBIC_SAMPLE_CAPACITY])
    ex = cython.declare(cython.double[2])
    pt = cython.declare(cython.double[2])
    count: cython.int
    n: cython.int = 0
    i: cython.int
    if c.flatten:
        count = sample_times_c(x0, y0, x1, y1, x2, y2, x3, y3, buf)
        for i in range(count):
            cubic_point(x0, y0, x1, y1, x2, y2, x3, y3, buf[i], pt)
            if record_point(c, pt[0], pt[1]) != 0:
                return -1
    else:
        extrema(x0, x1, x2, x3, ex, cython.address(n))
        for i in range(n):
            cubic_point(x0, y0, x1, y1, x2, y2, x3, y3, ex[i], pt)
            if record_point(c, pt[0], pt[1]) != 0:
                return -1
        extrema(y0, y1, y2, y3, ex, cython.address(n))
        for i in range(n):
            cubic_point(x0, y0, x1, y1, x2, y2, x3, y3, ex[i], pt)
            if record_point(c, pt[0], pt[1]) != 0:
                return -1
        if record_point(c, x3, y3) != 0:
            return -1
    c.x = x3
    c.y = y3
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def parse_number(
    p: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    pos: cython.Py_ssize_t,
    value: cython.p_double,
    next_pos: cython.p_Py_ssize_t,
) -> cython.int:
    b0: cython.int
    raw: cython.int
    if pos < 0 or pos >= length:
        return -1
    b0 = p[pos]
    if 32 <= b0 <= 246:
        value[0] = cython.cast(cython.double, b0 - 139)
        next_pos[0] = pos + 1
        return 0
    if 247 <= b0 <= 250:
        if pos + 1 >= length:
            return -1
        value[0] = cython.cast(cython.double, (b0 - 247) * 256 + p[pos + 1] + 108)
        next_pos[0] = pos + 2
        return 0
    if 251 <= b0 <= 254:
        if pos + 1 >= length:
            return -1
        value[0] = cython.cast(cython.double, -(b0 - 251) * 256 - p[pos + 1] - 108)
        next_pos[0] = pos + 2
        return 0
    if b0 == 28:
        if pos + 3 > length:
            return -1
        raw = (p[pos + 1] << 8) | p[pos + 2]
        if raw >= 0x8000:
            raw -= 0x10000
        value[0] = cython.cast(cython.double, raw)
        next_pos[0] = pos + 3
        return 0
    if b0 == 255:
        if pos + 5 > length:
            return -1
        raw = (p[pos + 1] << 24) | (p[pos + 2] << 16) | (p[pos + 3] << 8) | p[pos + 4]
        value[0] = (cython.cast(cython.double, raw)) / 65536.0
        next_pos[0] = pos + 5
        return 0
    return -1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def do_flex(c: cython.pointer[Ctx], operator: cython.int) -> cython.int:
    s: cython.p_double = c.stack
    dy6: cython.double
    dx6: cython.double
    dx: cython.double
    dy: cython.double
    if operator == 34:
        if c.sp != 7:
            return -1
        if do_curve(c, s[0], 0.0, s[1], s[2], s[3], 0.0) != 0:
            return -1
        return do_curve(c, s[4], 0.0, s[5], -s[2], s[6], 0.0)
    if operator == 35:
        if c.sp != 13:
            return -1
        if do_curve(c, s[0], s[1], s[2], s[3], s[4], s[5]) != 0:
            return -1
        return do_curve(c, s[6], s[7], s[8], s[9], s[10], s[11])
    if operator == 36:
        if c.sp != 9:
            return -1
        dy6 = -(s[1] + s[3] + s[7])
        if do_curve(c, s[0], s[1], s[2], s[3], s[4], 0.0) != 0:
            return -1
        return do_curve(c, s[5], 0.0, s[6], s[7], s[8], dy6)
    if operator == 37:
        if c.sp != 11:
            return -1
        dx = s[0] + s[2] + s[4] + s[6] + s[8]
        dy = s[1] + s[3] + s[5] + s[7] + s[9]
        if fabs(dx) > fabs(dy):
            dx6 = s[10]
            dy6 = -dy
        else:
            dx6 = -dx
            dy6 = s[10]
        if do_curve(c, s[0], s[1], s[2], s[3], s[4], s[5]) != 0:
            return -1
        return do_curve(c, s[6], s[7], s[8], s[9], dx6, dy6)
    return -1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def escaped(c: cython.pointer[Ctx], operator: cython.int) -> cython.int:
    first: cython.double
    second: cython.double
    v1: cython.double
    v2: cython.double
    ch1: cython.double
    ch2: cython.double
    index = cython.declare(cython.int)
    count = cython.declare(cython.int)
    shift = cython.declare(cython.int)
    i: cython.int
    rolled = cython.declare(cython.double[48])
    if operator == 0:
        c.sp = 0
        return 0
    if operator in {3, 4, 15, 10, 11, 12, 24}:
        if c.sp < 2:
            return -1
        second = c.stack[c.sp - 1]
        first = c.stack[c.sp - 2]
        c.sp -= 2
        if operator == 3:
            return push(c, 1.0 if (first != 0.0 and second != 0.0) else 0.0)
        if operator == 4:
            return push(c, 1.0 if (first != 0.0 or second != 0.0) else 0.0)
        if operator == 15:
            return push(c, 1.0 if first == second else 0.0)
        if operator == 10:
            return push(c, first + second)
        if operator == 11:
            return push(c, first - second)
        if operator == 12:
            if second == 0.0:
                return -1
            return push(c, first / second)
        return push(c, first * second)
    if operator in {5, 9, 14, 26, 18}:
        if c.sp < 1:
            return -1
        first = c.stack[c.sp - 1]
        c.sp -= 1
        if operator == 18:
            return 0
        if operator == 5:
            return push(c, 1.0 if first == 0.0 else 0.0)
        if operator == 9:
            return push(c, fabs(first))
        if operator == 14:
            return push(c, -first)
        if first < 0.0:
            return -1
        return push(c, sqrt(first))
    if operator == 20:
        if c.sp < 2:
            return -1
        if require_integer(c.stack[c.sp - 1], cython.address(index)) != 0:
            return -1
        c.sp -= 1
        first = c.stack[c.sp - 1]
        c.sp -= 1
        if index < 0 or index >= TRANSIENT_SIZE:
            return -1
        c.transient[index] = first
        return 0
    if operator == 21:
        if c.sp < 1:
            return -1
        if require_integer(c.stack[c.sp - 1], cython.address(index)) != 0:
            return -1
        c.sp -= 1
        if index < 0 or index >= TRANSIENT_SIZE:
            return -1
        return push(c, c.transient[index])
    if operator == 22:
        if c.sp < 4:
            return -1
        v2 = c.stack[c.sp - 1]
        v1 = c.stack[c.sp - 2]
        ch2 = c.stack[c.sp - 3]
        ch1 = c.stack[c.sp - 4]
        c.sp -= 4
        return push(c, ch1 if v1 <= v2 else ch2)
    if operator == 23:
        c.rnd = (
            cython.cast(cython.uint, 1103515245) * c.rnd + cython.cast(cython.uint, 12345)
        ) & 0x7FFFFFFF
        return push(
            c, (cython.cast(cython.double, c.rnd + 1)) / cython.cast(cython.double, 0x80000000)
        )
    if operator == 27:
        if c.sp < 1:
            return -1
        return push(c, c.stack[c.sp - 1])
    if operator == 28:
        if c.sp < 2:
            return -1
        first = c.stack[c.sp - 1]
        c.stack[c.sp - 1] = c.stack[c.sp - 2]
        c.stack[c.sp - 2] = first
        return 0
    if operator == 29:
        if c.sp < 1:
            return -1
        if require_integer(c.stack[c.sp - 1], cython.address(index)) != 0:
            return -1
        c.sp -= 1
        if index < 0:
            index = 0
        if index >= c.sp:
            return -1
        return push(c, c.stack[c.sp - index - 1])
    if operator == 30:
        if c.sp < 2:
            return -1
        if require_integer(c.stack[c.sp - 1], cython.address(shift)) != 0:
            return -1
        c.sp -= 1
        if require_integer(c.stack[c.sp - 1], cython.address(count)) != 0:
            return -1
        c.sp -= 1
        if count < 0 or count > c.sp:
            return -1
        if count:
            shift = shift % count
            if shift < 0:
                shift += count
            if shift:
                for i in range(count):
                    rolled[i] = c.stack[c.sp - count + i]
                for i in range(count):
                    c.stack[c.sp - count + i] = rolled[(i - shift + count) % count]
        return 0
    if 34 <= operator <= 37:
        if not c.chas:
            return -1
        if do_flex(c, operator) != 0:
            return -1
        c.sp = 0
        return 0
    return -1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def execute(  # noqa: C901 - one dispatch loop over the Type 2 operators, as in the .pyx
    c: cython.pointer[Ctx],
    program: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    depth: cython.int,
) -> cython.int:
    pos: cython.Py_ssize_t = 0
    next_pos = cython.declare(cython.Py_ssize_t)
    byte: cython.int
    escaped_operator: cython.int
    operand_count: cython.int
    mask_bytes: cython.int
    i: cython.int
    n: cython.int
    subr_index = cython.declare(cython.int)
    base_code = cython.declare(cython.int)
    accent_code = cython.declare(cython.int)
    value = cython.declare(cython.double)
    displacement: cython.double
    dx: cython.double
    dy: cython.double
    first_offset: cython.double
    first: cython.double
    dx2: cython.double
    dy2: cython.double
    last: cython.double
    dx1: cython.double
    dy1: cython.double
    dx3: cython.double
    dy3: cython.double
    horizontal: cython.bint
    args = cython.declare(cython.double[48])
    nargs: cython.int
    head: cython.int
    result: cython.int

    if depth > MAX_SUBR_DEPTH:
        return -1

    while pos < length:
        byte = program[pos]
        if byte > 31 or byte == 28 or byte == 255:
            if (
                parse_number(program, length, pos, cython.address(value), cython.address(next_pos))
                != 0
            ):
                return -1
            pos = next_pos
            if push(c, value) != 0:
                return -1
            continue
        pos += 1

        if byte in {1, 3, 18, 23}:
            operand_count = c.sp
            if not c.width_resolved and operand_count % 2:
                operand_count -= 1
            if operand_count < 2 or operand_count % 2:
                return -1
            c.stem_count += operand_count // 2
            if c.stem_count > 96:
                return -1
            c.width_resolved = True
            c.sp = 0
        elif byte in {4, 22}:
            if c.sp == 1:
                displacement = c.stack[0]
            elif not c.width_resolved and c.sp == 2:
                displacement = c.stack[1]
            else:
                return -1
            c.width_resolved = True
            if byte == 4:
                result = do_move(c, 0.0, displacement)
            else:
                result = do_move(c, displacement, 0.0)
            if result != 0:
                return -1
            c.sp = 0
        elif byte == 5:
            if not c.chas or c.sp < 2 or c.sp % 2:
                return -1
            for i in range(0, c.sp - 1, 2):
                if do_line(c, c.stack[i], c.stack[i + 1]) != 0:
                    return -1
            c.sp = 0
        elif byte in {6, 7}:
            if not c.chas or c.sp == 0:
                return -1
            horizontal = byte == 6
            for i in range(c.sp):
                if horizontal:
                    result = do_line(c, c.stack[i], 0.0)
                else:
                    result = do_line(c, 0.0, c.stack[i])
                if result != 0:
                    return -1
                horizontal = not horizontal
            c.sp = 0
        elif byte == 8:
            if not c.chas or c.sp < 6 or c.sp % 6:
                return -1
            for i in range(0, c.sp - 5, 6):
                if (
                    do_curve(
                        c,
                        c.stack[i],
                        c.stack[i + 1],
                        c.stack[i + 2],
                        c.stack[i + 3],
                        c.stack[i + 4],
                        c.stack[i + 5],
                    )
                    != 0
                ):
                    return -1
            c.sp = 0
        elif byte in {10, 29}:
            if c.sp == 0:
                return -1
            if require_integer(c.stack[c.sp - 1], cython.address(subr_index)) != 0:
                return -1
            c.sp -= 1
            if byte == 10:
                subr_index += c.lbias
                if subr_index < 0 or subr_index >= c.nl:
                    return -1
                result = execute(c, c.lsub[subr_index], c.llen[subr_index], depth + 1)
            else:
                subr_index += c.gbias
                if subr_index < 0 or subr_index >= c.ng:
                    return -1
                result = execute(c, c.gsub[subr_index], c.glen[subr_index], depth + 1)
            if result != 1:
                return result
        elif byte == 11:
            return 1
        elif byte == 12:
            if pos >= length:
                return -1
            escaped_operator = program[pos]
            pos += 1
            if escaped(c, escaped_operator) != 0:
                return -1
        elif byte == 14:
            nargs = c.sp
            head = 0
            if not c.width_resolved:
                if nargs == 1 or nargs == 5:
                    head = 1
                    nargs -= 1
                elif nargs != 0 and nargs != 4:
                    return -1
                c.width_resolved = True
            elif nargs != 0 and nargs != 4:
                return -1
            for i in range(nargs):
                args[i] = c.stack[head + i]
            c.sp = 0
            if flush_contour(c) != 0:
                return -1
            if nargs:
                if require_integer(args[2], cython.address(base_code)) != 0:
                    return -1
                if require_integer(args[3], cython.address(accent_code)) != 0:
                    return -1
                c.has_seac = True
                c.seac_base = base_code
                c.seac_accent = accent_code
                c.seac_dx = args[0]
                c.seac_dy = args[1]
            return 0
        elif byte in {19, 20}:
            operand_count = c.sp
            if not c.width_resolved and operand_count % 2:
                operand_count -= 1
            if operand_count % 2:
                return -1
            c.stem_count += operand_count // 2
            if c.stem_count <= 0 or c.stem_count > 96:
                return -1
            mask_bytes = (c.stem_count + 7) // 8
            if pos + mask_bytes > length:
                return -1
            c.width_resolved = True
            c.sp = 0
            pos += mask_bytes
        elif byte == 21:
            if c.sp == 2:
                dx = c.stack[0]
                dy = c.stack[1]
            elif not c.width_resolved and c.sp == 3:
                dx = c.stack[1]
                dy = c.stack[2]
            else:
                return -1
            c.width_resolved = True
            if do_move(c, dx, dy) != 0:
                return -1
            c.sp = 0
        elif byte == 24:
            if not c.chas or c.sp < 8 or (c.sp - 2) % 6:
                return -1
            n = c.sp - 2
            for i in range(0, n - 5, 6):
                if (
                    do_curve(
                        c,
                        c.stack[i],
                        c.stack[i + 1],
                        c.stack[i + 2],
                        c.stack[i + 3],
                        c.stack[i + 4],
                        c.stack[i + 5],
                    )
                    != 0
                ):
                    return -1
            if do_line(c, c.stack[c.sp - 2], c.stack[c.sp - 1]) != 0:
                return -1
            c.sp = 0
        elif byte == 25:
            if not c.chas or c.sp < 8 or (c.sp - 6) % 2:
                return -1
            n = c.sp - 6
            for i in range(0, n - 1, 2):
                if do_line(c, c.stack[i], c.stack[i + 1]) != 0:
                    return -1
            if (
                do_curve(
                    c,
                    c.stack[c.sp - 6],
                    c.stack[c.sp - 5],
                    c.stack[c.sp - 4],
                    c.stack[c.sp - 3],
                    c.stack[c.sp - 2],
                    c.stack[c.sp - 1],
                )
                != 0
            ):
                return -1
            c.sp = 0
        elif byte in {26, 27}:
            if not c.chas or c.sp < 4 or (c.sp % 4 != 0 and c.sp % 4 != 1):
                return -1
            head = 0
            first_offset = 0.0
            if c.sp % 2:
                first_offset = c.stack[0]
                head = 1
            n = c.sp - head
            for i in range(0, n - 3, 4):
                first = c.stack[head + i]
                dx2 = c.stack[head + i + 1]
                dy2 = c.stack[head + i + 2]
                last = c.stack[head + i + 3]
                if byte == 26:
                    result = do_curve(c, first_offset, first, dx2, dy2, 0.0, last)
                else:
                    result = do_curve(c, first, first_offset, dx2, dy2, last, 0.0)
                if result != 0:
                    return -1
                first_offset = 0.0
            c.sp = 0
        elif byte in {30, 31}:
            if not c.chas or c.sp < 4 or (c.sp % 4 != 0 and c.sp % 4 != 1):
                return -1
            horizontal = byte == 31
            nargs = c.sp
            for i in range(nargs):
                args[i] = c.stack[i]
            c.sp = 0
            head = 0
            while nargs - head >= 4:
                if horizontal:
                    dx1 = args[head]
                    head += 1
                    dy1 = 0.0
                    dx2 = args[head]
                    head += 1
                    dy2 = args[head]
                    head += 1
                    dy3 = args[head]
                    head += 1
                    if nargs - head == 1:
                        dx3 = args[head]
                        head += 1
                    else:
                        dx3 = 0.0
                else:
                    dx1 = 0.0
                    dy1 = args[head]
                    head += 1
                    dx2 = args[head]
                    head += 1
                    dy2 = args[head]
                    head += 1
                    dx3 = args[head]
                    head += 1
                    if nargs - head == 1:
                        dy3 = args[head]
                        head += 1
                    else:
                        dy3 = 0.0
                if do_curve(c, dx1, dy1, dx2, dy2, dx3, dy3) != 0:
                    return -1
                horizontal = not horizontal
            c.sp = 0
        else:
            return -1
    return 1


def type2_glyph_geometry(
    charstring: bytes,
    local_subrs: tuple,
    global_subrs: tuple,
    flatten: cython.bint = True,
    retain_contours: cython.bint = True,
):
    c = cython.declare(Ctx)
    i: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    j: cython.Py_ssize_t
    item: bytes
    status: cython.int
    contours: list = []
    contour: list

    c.sp = 0
    c.stem_count = 0
    c.width_resolved = False
    c.x = 0.0
    c.y = 0.0
    c.pts = cython.NULL
    c.npts = 0
    c.pts_cap = 0
    c.cur_start = 0
    c.spans = cython.NULL
    c.nspans = 0
    c.spans_cap = 0
    c.cmin_x = INF
    c.cmin_y = INF
    c.cmax_x = -INF
    c.cmax_y = -INF
    c.chas = False
    c.bmin_x = INF
    c.bmin_y = INF
    c.bmax_x = -INF
    c.bmax_y = -INF
    c.bhas = False
    c.rnd = RANDOM_INITIAL_STATE
    c.flatten = flatten
    c.retain = retain_contours
    c.has_seac = False
    c.seac_base = 0
    c.seac_accent = 0
    c.seac_dx = 0.0
    c.seac_dy = 0.0
    for i in range(TRANSIENT_SIZE):
        c.transient[i] = 0.0

    c.nl = len(local_subrs)
    c.ng = len(global_subrs)
    c.lbias = subr_bias(c.nl)
    c.gbias = subr_bias(c.ng)
    c.lsub = cython.cast(
        cython.pp_const_uchar,
        malloc((c.nl if c.nl else 1) * cython.sizeof(cython.p_void)),
    )
    c.llen = cython.cast(
        cython.p_Py_ssize_t,
        malloc((c.nl if c.nl else 1) * cython.sizeof(cython.Py_ssize_t)),
    )
    c.gsub = cython.cast(
        cython.pp_const_uchar,
        malloc((c.ng if c.ng else 1) * cython.sizeof(cython.p_void)),
    )
    c.glen = cython.cast(
        cython.p_Py_ssize_t,
        malloc((c.ng if c.ng else 1) * cython.sizeof(cython.Py_ssize_t)),
    )
    if (
        c.lsub == cython.NULL
        or c.llen == cython.NULL
        or c.gsub == cython.NULL
        or c.glen == cython.NULL
    ):
        free(c.lsub)
        free(c.llen)
        free(c.gsub)
        free(c.glen)
        raise MemoryError
    try:
        for i in range(c.nl):
            item = local_subrs[i]
            c.lsub[i] = cython.cast(cython.p_const_uchar, item)
            c.llen[i] = len(item)
        for i in range(c.ng):
            item = global_subrs[i]
            c.gsub[i] = cython.cast(cython.p_const_uchar, item)
            c.glen[i] = len(item)

        status = execute(
            cython.address(c),
            cython.cast(cython.p_const_uchar, charstring),
            len(charstring),
            0,
        )
        if status == 1:
            flush_contour(cython.address(c))

        if c.retain:
            for i in range(c.nspans):
                start = c.spans[2 * i]
                end = c.spans[2 * i + 1]
                contour = [None] * (end - start)
                for j in range(start, end):
                    contour[j - start] = (c.pts[2 * j], c.pts[2 * j + 1])
                contours.append(contour)
    finally:
        free(c.lsub)
        free(c.llen)
        free(c.gsub)
        free(c.glen)
        free(c.pts)
        free(c.spans)

    bbox = (c.bmin_x, c.bmin_y, c.bmax_x, c.bmax_y) if c.bhas else None
    seac = (c.seac_base, c.seac_accent, c.seac_dx, c.seac_dy) if c.has_seac else None
    return contours, bbox, seac, status != -1
