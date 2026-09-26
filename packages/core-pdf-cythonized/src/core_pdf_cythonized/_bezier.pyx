# SPDX-License-Identifier: AGPL-3.0-only
"""Cubic bezier flattening for Type 2 charstring outlines (_type2.pyx).

_type2 flattens each curve through sample_times_c; cubic_sample_times returns
the same times as a tuple. The extrema match
core_adobe_fonts.cff.charstrings.cubic_extrema_times exactly. The recursion
carries eight doubles instead of four coordinate tuples, which is the whole
point: no Python object touches the inner loop.
"""

from libc.math cimport sqrt, fabs

cdef double CUBIC_FLATNESS = 0.25
cdef int CUBIC_MAX_DEPTH = 12

cdef inline bint flat(double x0, double y0, double x1, double y1,
                      double x2, double y2, double x3, double y3) noexcept nogil:
    cdef double dx = x3 - x0, dy = y3 - y0
    cdef double chord = dx*dx + dy*dy
    cdef double tol = CUBIC_FLATNESS * CUBIC_FLATNESS
    cdef double a, b, c1, c2
    if chord <= 1e-18:
        a = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)
        b = (x2-x0)*(x2-x0) + (y2-y0)*(y2-y0)
        return (a if a > b else b) <= tol
    c1 = dx*(y1-y0) - dy*(x1-x0)
    c2 = dx*(y2-y0) - dy*(x2-x0)
    a = c1*c1; b = c2*c2
    return (a if a > b else b) <= tol * chord

cdef void extrema(double p0, double p1, double p2, double p3,
                  double* out, int* n) noexcept nogil:
    n[0] = 0
    if (p0 <= p1 <= p2 <= p3) or (p0 >= p1 >= p2 >= p3):
        return
    cdef double a = -p0 + 3.0*p1 - 3.0*p2 + p3
    cdef double b = 2.0*(p0 - 2.0*p1 + p2)
    cdef double c = p1 - p0
    cdef double eps = 1e-12, root, disc, rd, r0, r1
    if fabs(a) <= eps:
        if fabs(b) <= eps:
            return
        root = -c / b
        if 0.0 < root < 1.0:
            out[0] = root; n[0] = 1
        return
    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return
    rd = sqrt(disc)
    r0 = (-b - rd) / (2.0*a)
    r1 = (-b + rd) / (2.0*a)
    if 0.0 < r0 < 1.0:
        out[n[0]] = r0; n[0] += 1
    if 0.0 < r1 < 1.0 and r1 != r0:
        out[n[0]] = r1; n[0] += 1

cdef void rec(double x0, double y0, double x1, double y1, double x2, double y2,
              double x3, double y3, double t0, double t1, int depth,
              double* buf, int* count, int capacity) noexcept nogil:
    # Depth is capped, so leaves <= 2**CUBIC_MAX_DEPTH and the buffer is
    # provably large enough. The guard is still here because this runs with
    # boundscheck off: a miscount would corrupt memory, not raise.
    if count[0] >= capacity:
        return
    if depth >= CUBIC_MAX_DEPTH or flat(x0,y0,x1,y1,x2,y2,x3,y3):
        buf[count[0]] = t1; count[0] += 1
        return
    cdef double ax = (x0+x1)/2.0, ay = (y0+y1)/2.0
    cdef double bx = (x1+x2)/2.0, by = (y1+y2)/2.0
    cdef double cx = (x2+x3)/2.0, cy = (y2+y3)/2.0
    cdef double dx = (ax+bx)/2.0, dy = (ay+by)/2.0
    cdef double ex = (bx+cx)/2.0, ey = (by+cy)/2.0
    cdef double mx = (dx+ex)/2.0, my = (dy+ey)/2.0
    cdef double tm = (t0+t1)/2.0
    rec(x0,y0, ax,ay, dx,dy, mx,my, t0, tm, depth+1, buf, count, capacity)
    rec(mx,my, ex,ey, cx,cy, x3,y3, tm, t1, depth+1, buf, count, capacity)

cdef int sample_times_c(double x0, double y0, double x1, double y1,
                        double x2, double y2, double x3, double y3,
                        double* out) noexcept nogil:
    """Fill `out` with the sorted, de-duplicated sample times. Returns the count.

    `out` must hold CUBIC_SAMPLE_CAPACITY doubles. This is what
    cubic_sample_times returns as a tuple, and what _type2 consumes directly;
    the sort-then-drop-neighbours is the C spelling of sorted(set(...)), which
    agrees for these values because none of them can be NaN.
    """
    cdef double ex[2]
    cdef int count = 0, n = 0, i, j
    cdef double key
    out[count] = 1.0; count += 1
    extrema(x0, x1, x2, x3, ex, &n)
    for i in range(n):
        out[count] = ex[i]; count += 1
    extrema(y0, y1, y2, y3, ex, &n)
    for i in range(n):
        out[count] = ex[i]; count += 1
    rec(x0,y0,x1,y1,x2,y2,x3,y3, 0.0, 1.0, 0, out, &count, CUBIC_SAMPLE_CAPACITY)
    # Insertion sort: the count is a handful for a typical glyph curve, and it
    # avoids a qsort callback in the inner loop.
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
            out[j] = out[i]; j += 1
    return j


def cubic_sample_times(tuple p0, tuple p1, tuple p2, tuple p3):
    cdef double x0=p0[0], y0=p0[1], x1=p1[0], y1=p1[1]
    cdef double x2=p2[0], y2=p2[1], x3=p3[0], y3=p3[1]
    cdef double buf[CUBIC_SAMPLE_CAPACITY]
    cdef int count = sample_times_c(x0,y0,x1,y1,x2,y2,x3,y3, buf)
    cdef int i
    return tuple([buf[i] for i in range(count)])
