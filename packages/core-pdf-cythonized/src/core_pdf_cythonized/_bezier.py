# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._bezier import CUBIC_SAMPLE_CAPACITY
from cython.cimports.libc.math import fabs, sqrt

CUBIC_FLATNESS = cython.declare(cython.double, 0.25)
CUBIC_MAX_DEPTH = cython.declare(cython.int, 12)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def flat(
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    x2: cython.double,
    y2: cython.double,
    x3: cython.double,
    y3: cython.double,
) -> cython.bint:
    dx: cython.double = x3 - x0
    dy: cython.double = y3 - y0
    chord: cython.double = dx * dx + dy * dy
    tol: cython.double = CUBIC_FLATNESS * CUBIC_FLATNESS
    a: cython.double
    b: cython.double
    c1: cython.double
    c2: cython.double
    if chord <= 1e-18:
        a = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)
        b = (x2 - x0) * (x2 - x0) + (y2 - y0) * (y2 - y0)
        return (a if a > b else b) <= tol
    c1 = dx * (y1 - y0) - dy * (x1 - x0)
    c2 = dx * (y2 - y0) - dy * (x2 - x0)
    a = c1 * c1
    b = c2 * c2
    return (a if a > b else b) <= tol * chord


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def extrema(
    p0: cython.double,
    p1: cython.double,
    p2: cython.double,
    p3: cython.double,
    out: cython.p_double,
    n: cython.p_int,
) -> cython.void:
    n[0] = 0
    if (p0 <= p1 <= p2 <= p3) or (p0 >= p1 >= p2 >= p3):
        return
    a: cython.double = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    b: cython.double = 2.0 * (p0 - 2.0 * p1 + p2)
    c: cython.double = p1 - p0
    eps: cython.double = 1e-12
    root: cython.double
    disc: cython.double
    rd: cython.double
    r0: cython.double
    r1: cython.double
    if fabs(a) <= eps:
        if fabs(b) <= eps:
            return
        root = -c / b
        if 0.0 < root < 1.0:
            out[0] = root
            n[0] = 1
        return
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return
    rd = sqrt(disc)
    r0 = (-b - rd) / (2.0 * a)
    r1 = (-b + rd) / (2.0 * a)
    if 0.0 < r0 < 1.0:
        out[n[0]] = r0
        n[0] += 1
    if 0.0 < r1 < 1.0 and r1 != r0:
        out[n[0]] = r1
        n[0] += 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def rec(
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    x2: cython.double,
    y2: cython.double,
    x3: cython.double,
    y3: cython.double,
    t0: cython.double,
    t1: cython.double,
    depth: cython.int,
    buf: cython.p_double,
    count: cython.p_int,
    capacity: cython.int,
) -> cython.void:
    if count[0] >= capacity:
        return
    if depth >= CUBIC_MAX_DEPTH or flat(x0, y0, x1, y1, x2, y2, x3, y3):
        buf[count[0]] = t1
        count[0] += 1
        return
    ax: cython.double = (x0 + x1) / 2.0
    ay: cython.double = (y0 + y1) / 2.0
    bx: cython.double = (x1 + x2) / 2.0
    by: cython.double = (y1 + y2) / 2.0
    cx: cython.double = (x2 + x3) / 2.0
    cy: cython.double = (y2 + y3) / 2.0
    dx: cython.double = (ax + bx) / 2.0
    dy: cython.double = (ay + by) / 2.0
    ex: cython.double = (bx + cx) / 2.0
    ey: cython.double = (by + cy) / 2.0
    mx: cython.double = (dx + ex) / 2.0
    my: cython.double = (dy + ey) / 2.0
    tm: cython.double = (t0 + t1) / 2.0
    rec(x0, y0, ax, ay, dx, dy, mx, my, t0, tm, depth + 1, buf, count, capacity)
    rec(mx, my, ex, ey, cx, cy, x3, y3, tm, t1, depth + 1, buf, count, capacity)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def sample_times_c(
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    x2: cython.double,
    y2: cython.double,
    x3: cython.double,
    y3: cython.double,
    out: cython.p_double,
) -> cython.int:
    ex = cython.declare(cython.double[2])
    count: cython.int = 0
    n: cython.int = 0
    i: cython.int
    j: cython.int
    key: cython.double
    out[count] = 1.0
    count += 1
    extrema(x0, x1, x2, x3, ex, cython.address(n))
    for i in range(n):
        out[count] = ex[i]
        count += 1
    extrema(y0, y1, y2, y3, ex, cython.address(n))
    for i in range(n):
        out[count] = ex[i]
        count += 1
    rec(
        x0,
        y0,
        x1,
        y1,
        x2,
        y2,
        x3,
        y3,
        0.0,
        1.0,
        0,
        out,
        cython.address(count),
        CUBIC_SAMPLE_CAPACITY,
    )
    for i in range(1, count):
        key = out[i]
        j = i - 1
        while j >= 0 and out[j] > key:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = key
    j = 0
    for i in range(count):
        if i == 0 or out[i] != out[j - 1]:
            out[j] = out[i]
            j += 1
    return j


def cubic_sample_times(p0: tuple, p1: tuple, p2: tuple, p3: tuple):
    x0: cython.double = p0[0]
    y0: cython.double = p0[1]
    x1: cython.double = p1[0]
    y1: cython.double = p1[1]
    x2: cython.double = p2[0]
    y2: cython.double = p2[1]
    x3: cython.double = p3[0]
    y3: cython.double = p3[1]
    buf = cython.declare(cython.double[CUBIC_SAMPLE_CAPACITY])
    count: cython.int = sample_times_c(x0, y0, x1, y1, x2, y2, x3, y3, buf)
    return tuple([buf[i] for i in range(count)])
