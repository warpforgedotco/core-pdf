# SPDX-License-Identifier: AGPL-3.0-only
"""TrueType glyph outlines from the glyf table (core_pdf.impl.fonts.font_program).

A TrueType glyph's contours were fontTools': the whole glyf table split into
a Glyph per glyph on first use, the glyph expanded, drawn through a
DecomposingRecordingPen -- components decomposed through TransformPens --
and the recording flattened by recording_to_contours. About 245 us a glyph,
and 6.5 s of a 400-document render sweep.

This reads the glyph's bytes through loca and does what that pipeline did,
step for step:

- Glyph.decompileCoordinates: the flags with their repeats, the byte and
  short deltas with their signs, the running sums;
- GlyphCoordinates' slices, which hand back integral values as ints;
- Glyph.draw: the top-level ``lsb - xMin`` offset, each contour rotated to
  end on an on-curve point, lines, quadratic runs, and all-off-curve
  contours, which the recording drops;
- composites: each component's glyph index, offsets and 2.14 scales,
  DecomposingPen.addComponent's identity test, and TransformPen's point
  transform and composition;
- recording_to_contours: quadratic runs split at implied on-curve midpoints
  and flattened in six steps, contours closed and those under three points
  dropped; then scale_contours.

Python arithmetic is kept, not just its results: a value that fontTools
holds as an int stays exact and never becomes -0.0, and one that meets a
float becomes a double, as Python's int and float rules make it.

Anything fontTools would treat differently -- data that runs short, a glyph
index past the glyph order, a component placed by point matching, cubic
flags, a component nested past 32 levels -- returns None, and the caller
draws that glyph with fontTools. The caller checks, once per font, the
table-wide conditions fontTools' decompile imposes.
"""

from libc.stdlib cimport free, malloc, realloc

__all__ = ("truetype_contours",)

cdef enum:
    ON_CURVE = 0x01
    X_SHORT = 0x02
    Y_SHORT = 0x04
    REPEAT = 0x08
    X_SAME = 0x10
    Y_SAME = 0x20
    CUBIC = 0x80

cdef enum:
    WORDS = 0x0001
    XY_VALUES = 0x0002
    HAVE_SCALE = 0x0008
    MORE = 0x0020
    XY_SCALE = 0x0040
    TWO_BY_TWO = 0x0080
    INSTRUCTIONS = 0x0100

cdef enum:
    DONE = 0
    DECLINED = 1
    NO_MEMORY = 2

cdef int MAX_DEPTH = 32


cdef struct Num:
    # A Python number: the value, and whether Python holds it as an int.
    double v
    bint integer


cdef inline Num as_int(double value) noexcept nogil:
    cdef Num n
    n.v = value + 0.0  # an int has no negative zero
    n.integer = True
    return n


cdef inline Num as_float(double value) noexcept nogil:
    cdef Num n
    n.v = value
    n.integer = False
    return n


cdef inline Num mul(Num a, Num b) noexcept nogil:
    if a.integer and b.integer:
        return as_int(a.v * b.v)
    return as_float(a.v * b.v)


cdef inline Num add(Num a, Num b) noexcept nogil:
    if a.integer and b.integer:
        return as_int(a.v + b.v)
    return as_float(a.v + b.v)


cdef struct Transform:
    Num t[6]  # xx, xy, yx, yy, dx, dy


cdef inline bint is_identity(Transform* m) noexcept nogil:
    # transformation != Identity, compared by value.
    return (
        m.t[0].v == 1.0 and m.t[1].v == 0.0 and m.t[2].v == 0.0
        and m.t[3].v == 1.0 and m.t[4].v == 0.0 and m.t[5].v == 0.0
    )


cdef inline void apply(Transform* m, Num x, Num y, Num* out_x, Num* out_y) noexcept nogil:
    # Transform.transformPoint: (xx * x + yx * y + dx, xy * x + yy * y + dy).
    out_x[0] = add(add(mul(m.t[0], x), mul(m.t[2], y)), m.t[4])
    out_y[0] = add(add(mul(m.t[1], x), mul(m.t[3], y)), m.t[5])


cdef inline Transform compose(Transform* outer, Transform* other) noexcept nogil:
    # Transform.transform(other) with self the outer TransformPen's.
    cdef Transform r
    cdef Num xx1 = other.t[0], xy1 = other.t[1], yx1 = other.t[2], yy1 = other.t[3]
    cdef Num dx1 = other.t[4], dy1 = other.t[5]
    cdef Num xx2 = outer.t[0], xy2 = outer.t[1], yx2 = outer.t[2], yy2 = outer.t[3]
    cdef Num dx2 = outer.t[4], dy2 = outer.t[5]
    r.t[0] = add(mul(xx1, xx2), mul(xy1, yx2))
    r.t[1] = add(mul(xx1, xy2), mul(xy1, yy2))
    r.t[2] = add(mul(yx1, xx2), mul(yy1, yx2))
    r.t[3] = add(mul(yx1, xy2), mul(yy1, yy2))
    r.t[4] = add(add(mul(xx2, dx1), mul(yx2, dy1)), dx2)
    r.t[5] = add(add(mul(xy2, dx1), mul(yy2, dy1)), dy2)
    return r


cdef struct Recording:
    # recording_to_contours' state, and the contours it has made.
    double* points  # x, y pairs
    Py_ssize_t count
    Py_ssize_t capacity
    Py_ssize_t* ends  # each finished contour's end in points
    Py_ssize_t contours
    Py_ssize_t contour_capacity
    Py_ssize_t contour_start  # where the open contour starts
    bint open_contour
    bint have_current
    double current_x
    double current_y
    double start_x
    double start_y
    bint failed


cdef inline void push(Recording* r, double x, double y) noexcept nogil:
    cdef double* grown
    if r.count == r.capacity:
        grown = <double*> realloc(r.points, (2 * r.capacity + 64) * 2 * sizeof(double))
        if grown == NULL:
            r.failed = True
            return
        r.points = grown
        r.capacity = 2 * r.capacity + 64
    r.points[2 * r.count] = x
    r.points[2 * r.count + 1] = y
    r.count += 1


cdef inline void finish(Recording* r) noexcept nogil:
    # close_contour, then the contour is done.
    cdef Py_ssize_t* grown
    cdef Py_ssize_t first = r.contour_start
    if r.count > first and (
        r.points[2 * first] != r.points[2 * (r.count - 1)]
        or r.points[2 * first + 1] != r.points[2 * (r.count - 1) + 1]
    ):
        push(r, r.points[2 * first], r.points[2 * first + 1])
    if r.contours == r.contour_capacity:
        grown = <Py_ssize_t*> realloc(r.ends, (2 * r.contour_capacity + 16) * sizeof(Py_ssize_t))
        if grown == NULL:
            r.failed = True
            return
        r.ends = grown
        r.contour_capacity = 2 * r.contour_capacity + 16
    r.ends[r.contours] = r.count
    r.contours += 1
    r.contour_start = r.count
    r.open_contour = False


cdef inline void move_to(Recording* r, Num x, Num y) noexcept nogil:
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


cdef inline void line_to(Recording* r, Num x, Num y) noexcept nogil:
    if not r.have_current:
        return
    r.current_x = x.v
    r.current_y = y.v
    push(r, x.v, y.v)


cdef inline void flatten_quadratic(
    Recording* r, double x0, double y0, double x1, double y1, double x2, double y2
) noexcept nogil:
    cdef int i
    cdef double t, mt
    for i in range(1, 7):
        t = <double> i / 6.0
        mt = 1.0 - t
        push(
            r,
            mt * mt * x0 + 2.0 * mt * t * x1 + t * t * x2,
            mt * mt * y0 + 2.0 * mt * t * y1 + t * t * y2,
        )


cdef inline void q_curve_to(Recording* r, double* xs, double* ys, Py_ssize_t n) noexcept nogil:
    # append_quadratic for n points, the last on-curve.
    if not r.have_current or n == 0:
        return
    cdef double end_x = xs[n - 1], end_y = ys[n - 1]
    cdef double start_x = r.current_x, start_y = r.current_y
    cdef double segment_x, segment_y
    cdef Py_ssize_t i
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


cdef inline void close_path(Recording* r) noexcept nogil:
    if r.open_contour and r.count > r.contour_start:
        finish(r)
    r.open_contour = False
    r.have_current = False


cdef struct Font:
    const unsigned char* glyf
    Py_ssize_t glyf_length
    const long long* loca
    Py_ssize_t loca_count  # entries in loca
    const long long* lsb
    Py_ssize_t glyph_count


cdef inline int read_u16(const unsigned char* data, Py_ssize_t length, Py_ssize_t pos) noexcept nogil:
    if pos < 0 or pos + 2 > length:
        return -1
    return (data[pos] << 8) | data[pos + 1]


cdef inline int read_i16(const unsigned char* data, Py_ssize_t pos) noexcept nogil:
    cdef int value = (data[pos] << 8) | data[pos + 1]
    return value - 65536 if value >= 32768 else value


cdef int glyph_bytes(Font* font, Py_ssize_t gid, const unsigned char** data, Py_ssize_t* length) noexcept nogil:
    # The glyph's slice of glyf, or DECLINED where fontTools' glyfTable[name]
    # would not find it.
    if gid < 0 or gid >= font.glyph_count or gid + 1 >= font.loca_count:
        return DECLINED
    cdef long long start = font.loca[gid], end = font.loca[gid + 1]
    if end <= start or start >= font.glyf_length:
        length[0] = 0
        data[0] = font.glyf
        return DONE
    data[0] = font.glyf + start
    length[0] = <Py_ssize_t> (end - start)
    return DONE


cdef int draw_simple(
    Recording* r,
    const unsigned char* data,
    Py_ssize_t length,
    int contour_count,
    double offset,
    bint transformed,
    Transform* m,
) noexcept nogil:
    # Glyph.decompileCoordinates and Glyph.draw for a simple glyph.
    cdef Py_ssize_t pos = 10, i, previous = -1
    cdef int value
    if pos + 2 * contour_count + 2 > length:
        return DECLINED
    for i in range(contour_count):
        value = read_u16(data, length, pos + 2 * i)
        if value <= previous:
            return DECLINED
        previous = value
    cdef Py_ssize_t n = previous + 1
    cdef Py_ssize_t* ends = <Py_ssize_t*> malloc(contour_count * sizeof(Py_ssize_t))
    cdef unsigned char* flags = <unsigned char*> malloc(n)
    cdef double* xs = <double*> malloc(4 * n * sizeof(double))
    cdef Num* px = <Num*> malloc(2 * n * sizeof(Num))
    cdef int status = NO_MEMORY
    if ends != NULL and flags != NULL and xs != NULL and px != NULL:
        for i in range(contour_count):
            ends[i] = read_u16(data, length, pos + 2 * i)
        status = simple_body(r, data, length, contour_count, offset, transformed, m, n, ends, flags, xs, px)
    free(ends)
    free(flags)
    free(xs)
    free(px)
    return status


cdef int simple_body(
    Recording* r,
    const unsigned char* data,
    Py_ssize_t length,
    int contour_count,
    double offset,
    bint transformed,
    Transform* m,
    Py_ssize_t n,
    Py_ssize_t* ends,
    unsigned char* flags,
    double* xs,
    Num* px,
) noexcept nogil:
    cdef double* ys = xs + n
    cdef double* qx = xs + 2 * n
    cdef double* qy = xs + 3 * n
    cdef Num* py = px + n
    cdef Py_ssize_t pos = 10 + 2 * contour_count
    cdef int instruction_length = read_i16(data, pos)
    if instruction_length < 0:
        return DECLINED
    pos += 2 + instruction_length
    cdef Py_ssize_t i, k, j = 0, repeat, x_length = 0, y_length = 0
    cdef unsigned char flag
    cdef Py_ssize_t x_pos, y_pos, start, count, first_on, index, run, end_point
    cdef double x, y
    cdef Num tx, ty
    # The flags and their repeats.
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
        for k in range(repeat):
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
    # The deltas, their signs, and the running sums.
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
    # Glyph.draw, a contour at a time.
    start = 0
    for k in range(contour_count):
        end_point = ends[k] + 1
        count = end_point - start
        # The slice's points, each an int when integral, then transformed.
        first_on = -1
        for i in range(count):
            index = start + i
            px[i] = as_int(xs[index]) if xs[index] == <double> <long long> xs[index] else as_float(xs[index])
            py[i] = as_int(ys[index]) if ys[index] == <double> <long long> ys[index] else as_float(ys[index])
            if transformed:
                apply(m, px[i], py[i], &tx, &ty)
                px[i] = tx
                py[i] = ty
            if first_on < 0 and flags[index] & ON_CURVE:
                first_on = i
        if first_on < 0:
            # All off-curve: a qCurveTo with no current point, then closePath.
            close_path(r)
            start = end_point
            continue
        # Rotated to end on the first on-curve point, which is the moveTo.
        move_to(r, px[first_on], py[first_on])
        i = first_on + 1
        run = 0
        while run < count:
            # The distance to the next on-curve point, from i (mod count).
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


cdef int draw_glyph(
    Recording* r, Font* font, Py_ssize_t gid, bint top, bint transformed, Transform* m, int depth
) noexcept nogil:
    if depth > MAX_DEPTH:
        return DECLINED
    cdef const unsigned char* data
    cdef Py_ssize_t length
    if glyph_bytes(font, gid, &data, &length) != DONE:
        return DECLINED
    if length == 0:
        return DONE
    if length < 10:
        return DECLINED
    cdef int contour_count = read_i16(data, 0)
    cdef int x_min = read_i16(data, 2)
    cdef double offset = 0.0
    if contour_count == 0:
        return DONE
    if contour_count > 0:
        if top:
            offset = <double> (font.lsb[gid] - x_min)
        return draw_simple(r, data, length, contour_count, offset, transformed, m)
    if contour_count != -1:
        return DECLINED
    return draw_composite(r, font, data, length, transformed, m, depth)


cdef int draw_composite(
    Recording* r,
    Font* font,
    const unsigned char* data,
    Py_ssize_t length,
    bint transformed,
    Transform* m,
    int depth,
) noexcept nogil:
    # Every component is read before any is drawn, as decompileComponents
    # reads them when the glyph is expanded; then Glyph.draw adds each.
    cdef Py_ssize_t capacity = 8
    cdef Transform* components = <Transform*> malloc(capacity * sizeof(Transform))
    cdef Py_ssize_t* indexes = <Py_ssize_t*> malloc(capacity * sizeof(Py_ssize_t))
    cdef int status = NO_MEMORY
    if components != NULL and indexes != NULL:
        status = composite_body(r, font, data, length, transformed, m, depth, &components, &indexes, capacity)
    free(components)
    free(indexes)
    return status


cdef int composite_body(
    Recording* r,
    Font* font,
    const unsigned char* data,
    Py_ssize_t length,
    bint transformed,
    Transform* m,
    int depth,
    Transform** components_out,
    Py_ssize_t** indexes_out,
    Py_ssize_t capacity,
) noexcept nogil:
    cdef Py_ssize_t pos = 10, component_count = 0, index
    cdef Transform* grown_components
    cdef Py_ssize_t* grown_indexes
    cdef int flags, component, status
    cdef bint more = True, instructions = False
    cdef Transform* c
    cdef Transform combined
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
            grown_components = <Transform*> realloc(components_out[0], capacity * sizeof(Transform))
            if grown_components == NULL:
                return NO_MEMORY
            components_out[0] = grown_components
            grown_indexes = <Py_ssize_t*> realloc(indexes_out[0], capacity * sizeof(Py_ssize_t))
            if grown_indexes == NULL:
                return NO_MEMORY
            indexes_out[0] = grown_indexes
        c = &components_out[0][component_count]
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
            c.t[4] = as_int(<signed char> data[pos])
            c.t[5] = as_int(<signed char> data[pos + 1])
            pos += 2
        # getComponentInfo: (1, 0, 0, 1, x, y), or the 2.14 matrix with its
        # int zeros off the diagonal.
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
        c = &components_out[0][index]
        if transformed:
            # TransformPen.addComponent composes, then the recording pen
            # decomposes with the result.
            combined = compose(m, c)
        else:
            combined = c[0]
        status = draw_glyph(
            r, font, indexes_out[0][index], False, not is_identity(&combined), &combined, depth + 1
        )
        if status != DONE:
            return status
    return DONE


def truetype_contours(
    const unsigned char[::1] glyf,
    const long long[::1] loca,
    const long long[::1] lsb,
    Py_ssize_t glyph_count,
    Py_ssize_t gid,
    double scale,
):
    """fonttools_contours(font, gid), scaled as normalized_glyph_contours scales it.

    ``loca`` holds the glyph offsets, ``lsb`` each glyph's left side bearing
    from hmtx, ``glyph_count`` the length of the glyph order. Returns the
    contours as tuples of (x, y) points, or None where fontTools has to
    draw the glyph.
    """
    if lsb.shape[0] < glyph_count:
        raise ValueError("lsb must hold one bearing per glyph")
    cdef Font font
    font.glyf = &glyf[0] if glyf.shape[0] else NULL
    font.glyf_length = glyf.shape[0]
    font.loca = &loca[0] if loca.shape[0] else NULL
    font.loca_count = loca.shape[0]
    font.lsb = &lsb[0] if lsb.shape[0] else NULL
    font.glyph_count = glyph_count
    cdef Recording r
    r.points = NULL
    r.count = 0
    r.capacity = 0
    r.ends = NULL
    r.contours = 0
    r.contour_capacity = 0
    r.contour_start = 0
    r.open_contour = False
    r.have_current = False
    r.failed = False
    cdef Transform identity
    cdef int status
    try:
        with nogil:
            status = draw_glyph(&r, &font, gid, True, False, &identity, 0)
            if status == DONE and r.open_contour and r.count > r.contour_start:
                finish(&r)
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
