# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._truetype import (
    CUBIC,
    DECLINED,
    DONE,
    HAVE_SCALE,
    INSTRUCTIONS,
    MORE,
    NO_MEMORY,
    ON_CURVE,
    REPEAT,
    TWO_BY_TWO,
    WORDS,
    X_SAME,
    X_SHORT,
    XY_SCALE,
    XY_VALUES,
    Y_SAME,
    Y_SHORT,
)
from cython.cimports.libc.stdlib import free, malloc, realloc

__all__ = ("truetype_contours",)

MAX_DEPTH = cython.declare(cython.int, 32)


Num = cython.struct(
    v=cython.double,
    integer=cython.bint,
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def as_int(value: cython.double) -> Num:
    n = cython.declare(Num)
    n.v = value + 0.0
    n.integer = True
    return n


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def as_float(value: cython.double) -> Num:
    n = cython.declare(Num)
    n.v = value
    n.integer = False
    return n


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def mul(a: Num, b: Num) -> Num:
    if a.integer and b.integer:
        return as_int(a.v * b.v)
    return as_float(a.v * b.v)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def add(a: Num, b: Num) -> Num:
    if a.integer and b.integer:
        return as_int(a.v + b.v)
    return as_float(a.v + b.v)


Transform = cython.struct(
    t=Num[6],
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def is_identity(m: cython.pointer[Transform]) -> cython.bint:
    return (
        m.t[0].v == 1.0
        and m.t[1].v == 0.0
        and m.t[2].v == 0.0
        and m.t[3].v == 1.0
        and m.t[4].v == 0.0
        and m.t[5].v == 0.0
    )


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def apply(
    m: cython.pointer[Transform],
    x: Num,
    y: Num,
    out_x: cython.pointer[Num],
    out_y: cython.pointer[Num],
) -> cython.void:
    out_x[0] = add(add(mul(m.t[0], x), mul(m.t[2], y)), m.t[4])
    out_y[0] = add(add(mul(m.t[1], x), mul(m.t[3], y)), m.t[5])


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def compose(outer: cython.pointer[Transform], other: cython.pointer[Transform]) -> Transform:
    r = cython.declare(Transform)
    xx1: Num = other.t[0]
    xy1: Num = other.t[1]
    yx1: Num = other.t[2]
    yy1: Num = other.t[3]
    dx1: Num = other.t[4]
    dy1: Num = other.t[5]
    xx2: Num = outer.t[0]
    xy2: Num = outer.t[1]
    yx2: Num = outer.t[2]
    yy2: Num = outer.t[3]
    dx2: Num = outer.t[4]
    dy2: Num = outer.t[5]
    r.t[0] = add(mul(xx1, xx2), mul(xy1, yx2))
    r.t[1] = add(mul(xx1, xy2), mul(xy1, yy2))
    r.t[2] = add(mul(yx1, xx2), mul(yy1, yx2))
    r.t[3] = add(mul(yx1, xy2), mul(yy1, yy2))
    r.t[4] = add(add(mul(xx2, dx1), mul(yx2, dy1)), dx2)
    r.t[5] = add(add(mul(xy2, dx1), mul(yy2, dy1)), dy2)
    return r


Recording = cython.struct(
    points=cython.p_double,
    count=cython.Py_ssize_t,
    capacity=cython.Py_ssize_t,
    ends=cython.p_Py_ssize_t,
    contours=cython.Py_ssize_t,
    contour_capacity=cython.Py_ssize_t,
    contour_start=cython.Py_ssize_t,
    open_contour=cython.bint,
    have_current=cython.bint,
    current_x=cython.double,
    current_y=cython.double,
    start_x=cython.double,
    start_y=cython.double,
    failed=cython.bint,
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def push(r: cython.pointer[Recording], x: cython.double, y: cython.double) -> cython.void:
    grown: cython.p_double
    if r.count == r.capacity:
        grown = cython.cast(
            cython.p_double,
            realloc(r.points, (2 * r.capacity + 64) * 2 * cython.sizeof(cython.double)),
        )
        if grown == cython.NULL:
            r.failed = True
            return
        r.points = grown
        r.capacity = 2 * r.capacity + 64
    r.points[2 * r.count] = x
    r.points[2 * r.count + 1] = y
    r.count += 1


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def finish(r: cython.pointer[Recording]) -> cython.void:
    grown: cython.p_Py_ssize_t
    first: cython.Py_ssize_t = r.contour_start
    if r.count > first and (
        r.points[2 * first] != r.points[2 * (r.count - 1)]
        or r.points[2 * first + 1] != r.points[2 * (r.count - 1) + 1]
    ):
        push(r, r.points[2 * first], r.points[2 * first + 1])
    if r.contours == r.contour_capacity:
        grown = cython.cast(
            cython.p_Py_ssize_t,
            realloc(r.ends, (2 * r.contour_capacity + 16) * cython.sizeof(cython.Py_ssize_t)),
        )
        if grown == cython.NULL:
            r.failed = True
            return
        r.ends = grown
        r.contour_capacity = 2 * r.contour_capacity + 16
    r.ends[r.contours] = r.count
    r.contours += 1
    r.contour_start = r.count
    r.open_contour = False


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def move_to(r: cython.pointer[Recording], x: Num, y: Num) -> cython.void:
    if r.open_contour and r.count > r.contour_start:
        finish(r)
    r.contour_start = r.count
    r.open_contour = True
    r.start_x = x.v
    r.start_y = y.v
    r.current_x = x.v
    r.current_y = y.v
    r.have_current = True
    push(r, x.v, y.v)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def line_to(r: cython.pointer[Recording], x: Num, y: Num) -> cython.void:
    if not r.have_current:
        return
    r.current_x = x.v
    r.current_y = y.v
    push(r, x.v, y.v)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def flatten_quadratic(
    r: cython.pointer[Recording],
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    x2: cython.double,
    y2: cython.double,
) -> cython.void:
    i: cython.int
    t: cython.double
    mt: cython.double
    for i in range(1, 7):
        t = cython.cast(cython.double, i) / 6.0
        mt = 1.0 - t
        push(
            r,
            mt * mt * x0 + 2.0 * mt * t * x1 + t * t * x2,
            mt * mt * y0 + 2.0 * mt * t * y1 + t * t * y2,
        )


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def q_curve_to(
    r: cython.pointer[Recording],
    xs: cython.p_double,
    ys: cython.p_double,
    n: cython.Py_ssize_t,
) -> cython.void:
    if not r.have_current or n == 0:
        return
    end_x: cython.double = xs[n - 1]
    end_y: cython.double = ys[n - 1]
    start_x: cython.double = r.current_x
    start_y: cython.double = r.current_y
    segment_x: cython.double
    segment_y: cython.double
    i: cython.Py_ssize_t
    if n == 1:
        push(r, end_x, end_y)
    else:
        for i in range(n - 1):
            if i == n - 2:
                segment_x = end_x
                segment_y = end_y
            else:
                segment_x = (xs[i] + xs[i + 1]) * 0.5
                segment_y = (ys[i] + ys[i + 1]) * 0.5
            flatten_quadratic(r, start_x, start_y, xs[i], ys[i], segment_x, segment_y)
            start_x = segment_x
            start_y = segment_y
    r.current_x = end_x
    r.current_y = end_y


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def close_path(r: cython.pointer[Recording]) -> cython.void:
    if r.open_contour and r.count > r.contour_start:
        finish(r)
    r.open_contour = False
    r.have_current = False


Font = cython.struct(
    glyf=cython.p_const_uchar,
    glyf_length=cython.Py_ssize_t,
    loca=cython.p_const_longlong,
    loca_count=cython.Py_ssize_t,
    lsb=cython.p_const_longlong,
    glyph_count=cython.Py_ssize_t,
)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def read_u16(
    data: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    pos: cython.Py_ssize_t,
) -> cython.int:
    if pos < 0 or pos + 2 > length:
        return -1
    return (data[pos] << 8) | data[pos + 1]


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def read_i16(data: cython.p_const_uchar, pos: cython.Py_ssize_t) -> cython.int:
    value: cython.int = (data[pos] << 8) | data[pos + 1]
    return value - 65536 if value >= 32768 else value


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def glyph_bytes(
    font: cython.pointer[Font],
    gid: cython.Py_ssize_t,
    data: cython.pp_const_uchar,
    length: cython.p_Py_ssize_t,
) -> cython.int:
    if gid < 0 or gid >= font.glyph_count or gid + 1 >= font.loca_count:
        return DECLINED
    start: cython.longlong = font.loca[gid]
    end: cython.longlong = font.loca[gid + 1]
    if end <= start or start >= font.glyf_length:
        length[0] = 0
        data[0] = font.glyf
        return DONE
    data[0] = font.glyf + start
    length[0] = cython.cast(cython.Py_ssize_t, end - start)
    return DONE


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def draw_simple(
    r: cython.pointer[Recording],
    data: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    contour_count: cython.int,
    offset: cython.double,
    transformed: cython.bint,
    m: cython.pointer[Transform],
) -> cython.int:
    pos: cython.Py_ssize_t = 10
    i: cython.Py_ssize_t
    previous: cython.Py_ssize_t = -1
    value: cython.int
    if pos + 2 * contour_count + 2 > length:
        return DECLINED
    for i in range(contour_count):
        value = read_u16(data, length, pos + 2 * i)
        if value <= previous:
            return DECLINED
        previous = value
    n: cython.Py_ssize_t = previous + 1
    ends: cython.p_Py_ssize_t = cython.cast(
        cython.p_Py_ssize_t, malloc(contour_count * cython.sizeof(cython.Py_ssize_t))
    )
    flags: cython.p_uchar = cython.cast(cython.p_uchar, malloc(n))
    xs: cython.p_double = cython.cast(cython.p_double, malloc(4 * n * cython.sizeof(cython.double)))
    px: cython.pointer[Num] = cython.cast(cython.pointer[Num], malloc(2 * n * cython.sizeof(Num)))
    status: cython.int = NO_MEMORY
    if ends != cython.NULL and flags != cython.NULL and xs != cython.NULL and px != cython.NULL:
        for i in range(contour_count):
            ends[i] = read_u16(data, length, pos + 2 * i)
        status = simple_body(
            r, data, length, contour_count, offset, transformed, m, n, ends, flags, xs, px
        )
    free(ends)
    free(flags)
    free(xs)
    free(px)
    return status


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def simple_body(
    r: cython.pointer[Recording],
    data: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    contour_count: cython.int,
    offset: cython.double,
    transformed: cython.bint,
    m: cython.pointer[Transform],
    n: cython.Py_ssize_t,
    ends: cython.p_Py_ssize_t,
    flags: cython.p_uchar,
    xs: cython.p_double,
    px: cython.pointer[Num],
) -> cython.int:
    ys: cython.p_double = xs + n
    qx: cython.p_double = xs + 2 * n
    qy: cython.p_double = xs + 3 * n
    py: cython.pointer[Num] = px + n
    pos: cython.Py_ssize_t = 10 + 2 * contour_count
    instruction_length: cython.int = read_i16(data, pos)
    if instruction_length < 0:
        return DECLINED
    pos += 2 + instruction_length
    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    _k: cython.Py_ssize_t
    j: cython.Py_ssize_t = 0
    repeat: cython.Py_ssize_t
    x_length: cython.Py_ssize_t = 0
    y_length: cython.Py_ssize_t = 0
    flag: cython.uchar
    x_pos: cython.Py_ssize_t
    y_pos: cython.Py_ssize_t
    start: cython.Py_ssize_t
    count: cython.Py_ssize_t
    first_on: cython.Py_ssize_t
    index: cython.Py_ssize_t
    run: cython.Py_ssize_t
    end_point: cython.Py_ssize_t
    x: cython.double
    y: cython.double
    tx = cython.declare(Num)
    ty = cython.declare(Num)
    while True:
        if pos >= length:
            return DECLINED
        flag = data[pos]
        pos += 1
        repeat = 1
        if flag & REPEAT:
            if pos >= length:
                return DECLINED
            repeat = data[pos] + 1
            pos += 1
        for _k in range(repeat):
            if j >= n:
                return DECLINED
            flags[j] = flag
            j += 1
            if flag & X_SHORT:
                x_length += 1
            elif not (flag & X_SAME):
                x_length += 2
            if flag & Y_SHORT:
                y_length += 1
            elif not (flag & Y_SAME):
                y_length += 2
        if j >= n:
            break
    if pos + x_length + y_length > length:
        return DECLINED
    x_pos = pos
    y_pos = pos + x_length
    x = 0.0
    y = 0.0
    for i in range(n):
        flag = flags[i]
        if flag & CUBIC:
            return DECLINED
        if flag & X_SHORT:
            if flag & X_SAME:
                x += data[x_pos]
            else:
                x -= data[x_pos]
            x_pos += 1
        elif not (flag & X_SAME):
            x += read_i16(data, x_pos)
            x_pos += 2
        if flag & Y_SHORT:
            if flag & Y_SAME:
                y += data[y_pos]
            else:
                y -= data[y_pos]
            y_pos += 1
        elif not (flag & Y_SAME):
            y += read_i16(data, y_pos)
            y_pos += 2
        xs[i] = x + offset
        ys[i] = y
    start = 0
    for k in range(contour_count):
        end_point = ends[k] + 1
        count = end_point - start
        first_on = -1
        for i in range(count):
            index = start + i
            px[i] = (
                as_int(xs[index])
                if xs[index] == cython.cast(cython.double, cython.cast(cython.longlong, xs[index]))
                else as_float(xs[index])
            )
            py[i] = (
                as_int(ys[index])
                if ys[index] == cython.cast(cython.double, cython.cast(cython.longlong, ys[index]))
                else as_float(ys[index])
            )
            if transformed:
                apply(m, px[i], py[i], cython.address(tx), cython.address(ty))
                px[i] = tx
                py[i] = ty
            if first_on < 0 and flags[index] & ON_CURVE:
                first_on = i
        if first_on < 0:
            close_path(r)
            start = end_point
            continue
        move_to(r, px[first_on], py[first_on])
        i = first_on + 1
        run = 0
        while run < count:
            j = 0
            while not (flags[start + (i + j) % count] & ON_CURVE):
                j += 1
            if j == 0:
                if count - run > 1:
                    line_to(r, px[i % count], py[i % count])
            else:
                for index in range(j + 1):
                    qx[index] = px[(i + index) % count].v
                    qy[index] = py[(i + index) % count].v
                q_curve_to(r, qx, qy, j + 1)
            run += j + 1
            i += j + 1
        close_path(r)
        start = end_point
    if r.failed:
        return NO_MEMORY
    return DONE


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def draw_glyph(
    r: cython.pointer[Recording],
    font: cython.pointer[Font],
    gid: cython.Py_ssize_t,
    top: cython.bint,
    transformed: cython.bint,
    m: cython.pointer[Transform],
    depth: cython.int,
) -> cython.int:
    if depth > MAX_DEPTH:
        return DECLINED
    data = cython.declare(cython.p_const_uchar)
    length = cython.declare(cython.Py_ssize_t)
    if glyph_bytes(font, gid, cython.address(data), cython.address(length)) != DONE:
        return DECLINED
    if length == 0:
        return DONE
    if length < 10:
        return DECLINED
    contour_count: cython.int = read_i16(data, 0)
    x_min: cython.int = read_i16(data, 2)
    offset: cython.double = 0.0
    if contour_count == 0:
        return DONE
    if contour_count > 0:
        if top:
            offset = cython.cast(cython.double, font.lsb[gid] - x_min)
        return draw_simple(r, data, length, contour_count, offset, transformed, m)
    if contour_count != -1:
        return DECLINED
    return draw_composite(r, font, data, length, transformed, m, depth)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def draw_composite(
    r: cython.pointer[Recording],
    font: cython.pointer[Font],
    data: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    transformed: cython.bint,
    m: cython.pointer[Transform],
    depth: cython.int,
) -> cython.int:
    capacity: cython.Py_ssize_t = 8
    components: cython.pointer[Transform] = cython.cast(
        cython.pointer[Transform], malloc(capacity * cython.sizeof(Transform))
    )
    indexes: cython.p_Py_ssize_t = cython.cast(
        cython.p_Py_ssize_t, malloc(capacity * cython.sizeof(cython.Py_ssize_t))
    )
    status: cython.int = NO_MEMORY
    if components != cython.NULL and indexes != cython.NULL:
        status = composite_body(
            r,
            font,
            data,
            length,
            transformed,
            m,
            depth,
            cython.address(components),
            cython.address(indexes),
            capacity,
        )
    free(components)
    free(indexes)
    return status


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def composite_body(
    r: cython.pointer[Recording],
    font: cython.pointer[Font],
    data: cython.p_const_uchar,
    length: cython.Py_ssize_t,
    transformed: cython.bint,
    m: cython.pointer[Transform],
    depth: cython.int,
    components_out: cython.pointer[cython.pointer[Transform]],
    indexes_out: cython.pointer[cython.p_Py_ssize_t],
    capacity: cython.Py_ssize_t,
) -> cython.int:
    pos: cython.Py_ssize_t = 10
    component_count: cython.Py_ssize_t = 0
    index: cython.Py_ssize_t
    grown_components: cython.pointer[Transform]
    grown_indexes: cython.p_Py_ssize_t
    flags: cython.int
    component: cython.int
    status: cython.int
    more: cython.bint = True
    instructions: cython.bint = False
    c: cython.pointer[Transform]
    combined: Transform
    while more:
        if pos + 4 > length:
            return DECLINED
        flags = read_u16(data, length, pos)
        component = read_u16(data, length, pos + 2)
        pos += 4
        if component >= font.glyph_count or not (flags & XY_VALUES):
            return DECLINED
        if component_count == capacity:
            capacity *= 2
            grown_components = cython.cast(
                cython.pointer[Transform],
                realloc(components_out[0], capacity * cython.sizeof(Transform)),
            )
            if grown_components == cython.NULL:
                return NO_MEMORY
            components_out[0] = grown_components
            grown_indexes = cython.cast(
                cython.p_Py_ssize_t,
                realloc(indexes_out[0], capacity * cython.sizeof(cython.Py_ssize_t)),
            )
            if grown_indexes == cython.NULL:
                return NO_MEMORY
            indexes_out[0] = grown_indexes
        c = cython.address(components_out[0][component_count])
        indexes_out[0][component_count] = component
        component_count += 1
        if flags & WORDS:
            if pos + 4 > length:
                return DECLINED
            c.t[4] = as_int(read_i16(data, pos))
            c.t[5] = as_int(read_i16(data, pos + 2))
            pos += 4
        else:
            if pos + 2 > length:
                return DECLINED
            c.t[4] = as_int(cython.cast(cython.schar, data[pos]))
            c.t[5] = as_int(cython.cast(cython.schar, data[pos + 1]))
            pos += 2
        c.t[0] = as_int(1.0)
        c.t[1] = as_int(0.0)
        c.t[2] = as_int(0.0)
        c.t[3] = as_int(1.0)
        if flags & HAVE_SCALE:
            if pos + 2 > length:
                return DECLINED
            c.t[0] = as_float(read_i16(data, pos) / 16384.0)
            c.t[3] = c.t[0]
            pos += 2
        elif flags & XY_SCALE:
            if pos + 4 > length:
                return DECLINED
            c.t[0] = as_float(read_i16(data, pos) / 16384.0)
            c.t[3] = as_float(read_i16(data, pos + 2) / 16384.0)
            pos += 4
        elif flags & TWO_BY_TWO:
            if pos + 8 > length:
                return DECLINED
            c.t[0] = as_float(read_i16(data, pos) / 16384.0)
            c.t[1] = as_float(read_i16(data, pos + 2) / 16384.0)
            c.t[2] = as_float(read_i16(data, pos + 4) / 16384.0)
            c.t[3] = as_float(read_i16(data, pos + 6) / 16384.0)
            pos += 8
        more = (flags & MORE) != 0
        if flags & INSTRUCTIONS:
            instructions = True
    if instructions and pos + 2 > length:
        return DECLINED
    for index in range(component_count):
        c = cython.address(components_out[0][index])
        if transformed:
            combined = compose(m, c)
        else:
            combined = c[0]
        status = draw_glyph(
            r,
            font,
            indexes_out[0][index],
            False,
            not is_identity(cython.address(combined)),
            cython.address(combined),
            depth + 1,
        )
        if status != DONE:
            return status
    return DONE


def truetype_contours(
    glyf: cython.const[cython.uchar][::1],
    loca: cython.const[cython.longlong][::1],
    lsb: cython.const[cython.longlong][::1],
    glyph_count: cython.Py_ssize_t,
    gid: cython.Py_ssize_t,
    scale: cython.double,
):
    if lsb.shape[0] < glyph_count:
        raise ValueError("lsb must hold one bearing per glyph")
    font = cython.declare(Font)
    font.glyf = cython.address(glyf[0]) if glyf.shape[0] else cython.NULL
    font.glyf_length = glyf.shape[0]
    font.loca = cython.address(loca[0]) if loca.shape[0] else cython.NULL
    font.loca_count = loca.shape[0]
    font.lsb = cython.address(lsb[0]) if lsb.shape[0] else cython.NULL
    font.glyph_count = glyph_count
    r = cython.declare(Recording)
    r.points = cython.NULL
    r.count = 0
    r.capacity = 0
    r.ends = cython.NULL
    r.contours = 0
    r.contour_capacity = 0
    r.contour_start = 0
    r.open_contour = False
    r.have_current = False
    r.failed = False
    identity = cython.declare(Transform)
    status: cython.int
    try:
        with cython.nogil:
            status = draw_glyph(
                cython.address(r),
                cython.address(font),
                gid,
                True,
                False,
                cython.address(identity),
                0,
            )
            if status == DONE and r.open_contour and r.count > r.contour_start:
                finish(cython.address(r))
            if r.failed:
                status = NO_MEMORY
        if status == NO_MEMORY:
            raise MemoryError
        if status != DONE:
            return None
        contours = []
        start = 0
        for k in range(r.contours):
            end = r.ends[k]
            if end - start >= 3:
                if scale == 1.0:
                    contours.append(
                        tuple([(r.points[2 * i], r.points[2 * i + 1]) for i in range(start, end)])
                    )
                else:
                    contours.append(
                        tuple(
                            [
                                (r.points[2 * i] * scale, r.points[2 * i + 1] * scale)
                                for i in range(start, end)
                            ]
                        )
                    )
            start = end
        return tuple(contours)
    finally:
        free(r.points)
        free(r.ends)
