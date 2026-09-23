# SPDX-License-Identifier: AGPL-3.0-only
"""Type 2 charstring interpretation and glyph geometry, fused.

Interpreting a glyph's charstring was 12% of a profiled page extract, spread
evenly across two halves that could not be compiled apart: the interpreter
(core_adobe_fonts.cff.charstrings) drove a pen of Python closures
(core_pdf.impl.fonts.font_program) through one call per operand, one per
outline point and one per curve. Measured over 459 corpus charstrings the
cost split roughly execute 22%, record_point 18%, curve 15%, push 11%,
cubic_point 9%, parse_number 8%, min/max/len 18% -- no single piece worth
compiling, and the callbacks are most of what the halves cost each other.

Fusing them is the same move the knockout kernel made, and for the same
reason: the callbacks cannot be removed while the callee stays in Python.

Unlike that one the algorithm has NOT been moved out of the standards
package. core_adobe_fonts.cff.charstrings keeps execute_type2_charstring as
its published Type 2 API, and its conformance suite keeps driving it. That
suite asserts pen event sequences -- that hvcurveto alternates its operands,
that an invalid escaped operator is rejected -- and those are properties a
flattened contour does not pin, so rewriting it against this kernel would
have cost real coverage in a standards package to satisfy the letter of the
"delete the Python it replaced" rule.

What that rule exists to prevent is a compiled path and an interpreted path
both live and drifting apart. That is not the case here: core reaches Type 2
geometry only through this kernel, and the interpreter in core_adobe_fonts is
not a fallback for it. The kernel is pinned to that interpreter's behaviour
by golden vectors generated from it, exactly as the other kernels are.

Composite (seac) glyphs are the one thing this does not finish: resolving an
accent needs the font's charset, which is the caller's. endchar with
arguments returns the request instead, and the caller runs the components
back through this same function -- one interpreter, not two.

Float semantics must match CPython exactly. cubic_point uses pow() rather
than repeated multiplication because Python's ** on floats calls libm pow,
and the two disagree in the last place; the golden vectors are what says so.
"""

from libc.math cimport fabs, isfinite, pow, sqrt
from libc.stdlib cimport free, malloc, realloc

from core_pdf_cythonized._bezier cimport extrema, sample_times_c

cdef int MAX_STACK = 48
cdef int TRANSIENT_SIZE = 32
cdef int MAX_SUBR_DEPTH = 10
cdef unsigned int RANDOM_INITIAL_STATE = 0x1234ABCD

cdef double INF = float("inf")


cdef struct Ctx:
    double stack[48]
    int sp
    double transient[32]
    int stem_count
    bint width_resolved
    double x
    double y

    double *pts
    Py_ssize_t npts
    Py_ssize_t pts_cap
    Py_ssize_t cur_start
    Py_ssize_t *spans
    Py_ssize_t nspans
    Py_ssize_t spans_cap

    double cmin_x, cmin_y, cmax_x, cmax_y
    bint chas
    double bmin_x, bmin_y, bmax_x, bmax_y
    bint bhas

    unsigned int rnd
    bint flatten
    bint retain

    const unsigned char **lsub
    Py_ssize_t *llen
    Py_ssize_t nl
    const unsigned char **gsub
    Py_ssize_t *glen
    Py_ssize_t ng
    int lbias
    int gbias

    bint has_seac
    int seac_base
    int seac_accent
    double seac_dx
    double seac_dy


cdef inline int subr_bias(Py_ssize_t count) noexcept nogil:
    if count < 1240:
        return 107
    if count < 33900:
        return 1131
    return 32768


cdef inline int push(Ctx *c, double value) noexcept nogil:
    if c.sp >= MAX_STACK or not isfinite(value):
        return -1
    c.stack[c.sp] = value
    c.sp += 1
    return 0


cdef inline int require_integer(double value, int *out) noexcept nogil:
    cdef int integer = <int> value
    if value != <double> integer:
        return -1
    out[0] = integer
    return 0


cdef int grow_points(Ctx *c) noexcept nogil:
    cdef Py_ssize_t cap = c.pts_cap * 2 if c.pts_cap else 256
    cdef double *grown = <double *> realloc(c.pts, cap * 2 * sizeof(double))
    if grown == NULL:
        return -1
    c.pts = grown
    c.pts_cap = cap
    return 0


cdef inline int record_point(Ctx *c, double px, double py) noexcept nogil:
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


cdef int add_span(Ctx *c, Py_ssize_t start, Py_ssize_t end) noexcept nogil:
    cdef Py_ssize_t cap
    cdef Py_ssize_t *grown
    if c.nspans * 2 >= c.spans_cap:
        cap = c.spans_cap * 2 if c.spans_cap else 64
        grown = <Py_ssize_t *> realloc(c.spans, cap * sizeof(Py_ssize_t))
        if grown == NULL:
            return -1
        c.spans = grown
        c.spans_cap = cap
    c.spans[2 * c.nspans] = start
    c.spans[2 * c.nspans + 1] = end
    c.nspans += 1
    return 0


cdef int flush_contour(Ctx *c) noexcept nogil:
    # Two independent tests, exactly as the Python pen had them: the contour is
    # only recorded when points were retained, but the bounding box is merged
    # whenever any point was seen -- which is the bounds_only path.
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


cdef inline void cubic_point(double x0, double y0, double x1, double y1,
                             double x2, double y2, double x3, double y3,
                             double t, double *out) noexcept nogil:
    cdef double mt = 1.0 - t
    cdef double mt3 = pow(mt, 3.0)
    cdef double t3 = pow(t, 3.0)
    cdef double mt2t = 3.0 * mt * mt * t
    cdef double mtt2 = 3.0 * mt * t * t
    out[0] = mt3 * x0 + mt2t * x1 + mtt2 * x2 + t3 * x3
    out[1] = mt3 * y0 + mt2t * y1 + mtt2 * y2 + t3 * y3


cdef int do_move(Ctx *c, double dx, double dy) noexcept nogil:
    if flush_contour(c) != 0:
        return -1
    c.x += dx
    c.y += dy
    return record_point(c, c.x, c.y)


cdef inline int do_line(Ctx *c, double dx, double dy) noexcept nogil:
    c.x += dx
    c.y += dy
    return record_point(c, c.x, c.y)


cdef int do_curve(Ctx *c, double dx1, double dy1, double dx2, double dy2,
                  double dx3, double dy3) noexcept nogil:
    cdef double x0 = c.x, y0 = c.y
    cdef double x1 = x0 + dx1, y1 = y0 + dy1
    cdef double x2 = x1 + dx2, y2 = y1 + dy2
    cdef double x3 = x2 + dx3, y3 = y2 + dy3
    cdef double buf[4101]
    cdef double ex[2]
    cdef double pt[2]
    cdef int count, n = 0, i
    if c.flatten:
        count = sample_times_c(x0, y0, x1, y1, x2, y2, x3, y3, buf)
        for i in range(count):
            cubic_point(x0, y0, x1, y1, x2, y2, x3, y3, buf[i], pt)
            if record_point(c, pt[0], pt[1]) != 0:
                return -1
    else:
        extrema(x0, x1, x2, x3, ex, &n)
        for i in range(n):
            cubic_point(x0, y0, x1, y1, x2, y2, x3, y3, ex[i], pt)
            if record_point(c, pt[0], pt[1]) != 0:
                return -1
        extrema(y0, y1, y2, y3, ex, &n)
        for i in range(n):
            cubic_point(x0, y0, x1, y1, x2, y2, x3, y3, ex[i], pt)
            if record_point(c, pt[0], pt[1]) != 0:
                return -1
        if record_point(c, x3, y3) != 0:
            return -1
    c.x = x3
    c.y = y3
    return 0


cdef int parse_number(const unsigned char *p, Py_ssize_t length, Py_ssize_t pos,
                      double *value, Py_ssize_t *next_pos) noexcept nogil:
    cdef int b0, b1
    cdef int raw
    if pos < 0 or pos >= length:
        return -1
    b0 = p[pos]
    if 32 <= b0 <= 246:
        value[0] = <double> (b0 - 139)
        next_pos[0] = pos + 1
        return 0
    if 247 <= b0 <= 250:
        if pos + 1 >= length:
            return -1
        value[0] = <double> ((b0 - 247) * 256 + p[pos + 1] + 108)
        next_pos[0] = pos + 2
        return 0
    if 251 <= b0 <= 254:
        if pos + 1 >= length:
            return -1
        value[0] = <double> (-(b0 - 251) * 256 - p[pos + 1] - 108)
        next_pos[0] = pos + 2
        return 0
    if b0 == 28:
        if pos + 3 > length:
            return -1
        raw = (p[pos + 1] << 8) | p[pos + 2]
        if raw >= 0x8000:
            raw -= 0x10000
        value[0] = <double> raw
        next_pos[0] = pos + 3
        return 0
    if b0 == 255:
        if pos + 5 > length:
            return -1
        raw = (p[pos + 1] << 24) | (p[pos + 2] << 16) | (p[pos + 3] << 8) | p[pos + 4]
        value[0] = (<double> raw) / 65536.0
        next_pos[0] = pos + 5
        return 0
    return -1


cdef int do_flex(Ctx *c, int operator) noexcept nogil:
    cdef double *s = c.stack
    cdef double dy6, dx6, dx, dy
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


cdef int escaped(Ctx *c, int operator) noexcept nogil:
    cdef double first, second, v1, v2, ch1, ch2
    cdef int index, count, shift, i
    cdef double rolled[48]
    if operator == 0:
        c.sp = 0
        return 0
    if operator == 3 or operator == 4 or operator == 15 or operator == 10 \
            or operator == 11 or operator == 12 or operator == 24:
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
    if operator == 5 or operator == 9 or operator == 14 or operator == 26 or operator == 18:
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
        if require_integer(c.stack[c.sp - 1], &index) != 0:
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
        if require_integer(c.stack[c.sp - 1], &index) != 0:
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
        c.rnd = (1103515245u * c.rnd + 12345u) & 0x7FFFFFFF
        return push(c, (<double> (c.rnd + 1)) / <double> 0x80000000)
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
        if require_integer(c.stack[c.sp - 1], &index) != 0:
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
        if require_integer(c.stack[c.sp - 1], &shift) != 0:
            return -1
        c.sp -= 1
        if require_integer(c.stack[c.sp - 1], &count) != 0:
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


cdef int execute(Ctx *c, const unsigned char *program, Py_ssize_t length,
                 int depth) noexcept nogil:
    """1 = ran to the end, 0 = endchar, -1 = invalid charstring."""
    cdef Py_ssize_t pos = 0, next_pos
    cdef int byte, escaped_operator, operand_count, mask_bytes, i, n, subr_index
    cdef int base_code, accent_code
    cdef double value, displacement, dx, dy, first_offset, first, dx2, dy2, last
    cdef double dx1, dy1, dx3, dy3
    cdef bint horizontal
    cdef double args[48]
    cdef int nargs, head
    cdef int result

    if depth > MAX_SUBR_DEPTH:
        return -1

    while pos < length:
        byte = program[pos]
        if byte > 31 or byte == 28 or byte == 255:
            if parse_number(program, length, pos, &value, &next_pos) != 0:
                return -1
            pos = next_pos
            if push(c, value) != 0:
                return -1
            continue
        pos += 1

        if byte == 1 or byte == 3 or byte == 18 or byte == 23:
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
        elif byte == 4 or byte == 22:
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
        elif byte == 6 or byte == 7:
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
                if do_curve(c, c.stack[i], c.stack[i + 1], c.stack[i + 2],
                            c.stack[i + 3], c.stack[i + 4], c.stack[i + 5]) != 0:
                    return -1
            c.sp = 0
        elif byte == 10 or byte == 29:
            if c.sp == 0:
                return -1
            if require_integer(c.stack[c.sp - 1], &subr_index) != 0:
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
                if require_integer(args[2], &base_code) != 0:
                    return -1
                if require_integer(args[3], &accent_code) != 0:
                    return -1
                c.has_seac = True
                c.seac_base = base_code
                c.seac_accent = accent_code
                c.seac_dx = args[0]
                c.seac_dy = args[1]
            return 0
        elif byte == 19 or byte == 20:
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
                if do_curve(c, c.stack[i], c.stack[i + 1], c.stack[i + 2],
                            c.stack[i + 3], c.stack[i + 4], c.stack[i + 5]) != 0:
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
            if do_curve(c, c.stack[c.sp - 6], c.stack[c.sp - 5], c.stack[c.sp - 4],
                        c.stack[c.sp - 3], c.stack[c.sp - 2], c.stack[c.sp - 1]) != 0:
                return -1
            c.sp = 0
        elif byte == 26 or byte == 27:
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
        elif byte == 30 or byte == 31:
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
                    dx1 = args[head]; head += 1
                    dy1 = 0.0
                    dx2 = args[head]; head += 1
                    dy2 = args[head]; head += 1
                    dy3 = args[head]; head += 1
                    if nargs - head == 1:
                        dx3 = args[head]; head += 1
                    else:
                        dx3 = 0.0
                else:
                    dx1 = 0.0
                    dy1 = args[head]; head += 1
                    dx2 = args[head]; head += 1
                    dy2 = args[head]; head += 1
                    dx3 = args[head]; head += 1
                    if nargs - head == 1:
                        dy3 = args[head]; head += 1
                    else:
                        dy3 = 0.0
                if do_curve(c, dx1, dy1, dx2, dy2, dx3, dy3) != 0:
                    return -1
                horizontal = not horizontal
            c.sp = 0
        else:
            return -1
    return 1


def type2_glyph_geometry(bytes charstring, tuple local_subrs, tuple global_subrs,
                         bint flatten=True, bint retain_contours=True):
    """Interpret one Type 2 charstring into contours and a bounding box.

    Returns ``(contours, bbox, seac, valid)``. ``seac`` is ``None`` unless the
    glyph ended with a composite request, in which case it is
    ``(base_code, accent_code, dx, dy)`` for the caller to resolve.

    An invalid charstring does not raise: interpretation stops and whatever
    was completed before that point is returned, which is what the Python
    original did by swallowing the exception around its pen. ``valid`` reports
    whether it stopped that way, so a conformance test can still assert that a
    malformed program is rejected rather than quietly producing geometry --
    that property is why the interpreter had a raising contract at all, and it
    would otherwise have been lost in the move.
    """
    cdef Ctx c
    cdef Py_ssize_t i, start, end, j
    cdef bytes item
    cdef int status
    cdef list contours = []
    cdef list contour

    c.sp = 0
    c.stem_count = 0
    c.width_resolved = False
    c.x = 0.0
    c.y = 0.0
    c.pts = NULL
    c.npts = 0
    c.pts_cap = 0
    c.cur_start = 0
    c.spans = NULL
    c.nspans = 0
    c.spans_cap = 0
    c.cmin_x = INF; c.cmin_y = INF; c.cmax_x = -INF; c.cmax_y = -INF
    c.chas = False
    c.bmin_x = INF; c.bmin_y = INF; c.bmax_x = -INF; c.bmax_y = -INF
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
    c.lsub = <const unsigned char **> malloc((c.nl if c.nl else 1) * sizeof(void *))
    c.llen = <Py_ssize_t *> malloc((c.nl if c.nl else 1) * sizeof(Py_ssize_t))
    c.gsub = <const unsigned char **> malloc((c.ng if c.ng else 1) * sizeof(void *))
    c.glen = <Py_ssize_t *> malloc((c.ng if c.ng else 1) * sizeof(Py_ssize_t))
    if c.lsub == NULL or c.llen == NULL or c.gsub == NULL or c.glen == NULL:
        free(c.lsub); free(c.llen); free(c.gsub); free(c.glen)
        raise MemoryError
    try:
        # The tuples are held by the caller for the whole call, so borrowing
        # their buffers is safe and keeps the interpreter off Python objects.
        for i in range(c.nl):
            item = local_subrs[i]
            c.lsub[i] = <const unsigned char *> item
            c.llen[i] = len(item)
        for i in range(c.ng):
            item = global_subrs[i]
            c.gsub[i] = <const unsigned char *> item
            c.glen[i] = len(item)

        status = execute(&c, <const unsigned char *> charstring, len(charstring), 0)
        if status == 1:
            flush_contour(&c)

        if c.retain:
            for i in range(c.nspans):
                start = c.spans[2 * i]
                end = c.spans[2 * i + 1]
                contour = [None] * (end - start)
                for j in range(start, end):
                    contour[j - start] = (c.pts[2 * j], c.pts[2 * j + 1])
                contours.append(contour)
    finally:
        free(c.lsub); free(c.llen); free(c.gsub); free(c.glen)
        free(c.pts); free(c.spans)

    bbox = (c.bmin_x, c.bmin_y, c.bmax_x, c.bmax_y) if c.bhas else None
    seac = (c.seac_base, c.seac_accent, c.seac_dx, c.seac_dy) if c.has_seac else None
    return contours, bbox, seac, status != -1
