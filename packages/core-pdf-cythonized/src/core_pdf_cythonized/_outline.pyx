# SPDX-License-Identifier: AGPL-3.0-only
"""Glyph outline edge construction (core_pdf.impl.render.commands).

transformed_outline is the largest single item left in rendering, about 20% of
it across the corpus. Only part of that is worth compiling, and measuring said
which part:

    points via tolist() + zip()  ->  C loop     0.72x   (slower)
    edges via column_stack       ->  C loop     2.01x

Building Python tuples in C loses to CPython's zip, so the point lists stay in
the caller. This takes only the numeric half: it decides which spans survive,
and writes every edge of every surviving span into one array in a single pass.
The original allocated a column_stack per span, a separate 1x4 array for each
closing edge, and then concatenated the lot.

The span bookkeeping is returned rather than recomputed, so the caller builds
point lists for exactly the spans that were kept, with the same adjusted ends.
"""

from libc.math cimport isnan

import numpy


def outline_edges(double[::1] xs, double[::1] ys, spans):
    """Edges for all surviving spans, plus which spans survived.

    Returns (edges, kept, dropped) where `kept` is a list of (start, end) with
    end already adjusted for a duplicated closing point, and `dropped` is True
    if any span was too short to contribute.
    """
    cdef Py_ssize_t start, end, i, w = 0, total = 0
    cdef bint dropped = False
    cdef bint closes
    cdef list kept = []

    for span in spans:
        start = span[0]
        end = span[1]
        # The original built the point list first, dropped a duplicated final
        # point, and only then checked the length -- so a two-point span whose
        # ends coincide collapses to one point and is dropped, not kept.
        if end > start and xs[start] == xs[end - 1] and ys[start] == ys[end - 1]:
            end -= 1
        if end - start >= 2:
            closes = xs[start] != xs[end - 1] or ys[start] != ys[end - 1]
            kept.append((start, end, closes))
            total += (end - start - 1) + (1 if closes else 0)
        else:
            dropped = True

    if not kept:
        return None, kept, dropped

    edges = numpy.empty((total, 4), numpy.float64)
    cdef double[:, ::1] out = edges
    for span in kept:
        start = span[0]
        end = span[1]
        closes = span[2]
        for i in range(start, end - 1):
            out[w, 0] = xs[i]
            out[w, 1] = ys[i]
            out[w, 2] = xs[i + 1]
            out[w, 3] = ys[i + 1]
            w += 1
        if closes:
            out[w, 0] = xs[end - 1]
            out[w, 1] = ys[end - 1]
            out[w, 2] = xs[start]
            out[w, 3] = ys[start]
            w += 1
    return edges, kept, dropped


cdef tuple _outline_edges(const double* xs, const double* ys, spans):
    # outline_edges over raw columns whose spans all lie inside them, so
    # translated_outline_edges reaches it without a Python call and two more
    # buffer acquisitions per glyph. Inside those bounds every index below is
    # one outline_edges reads the same element with; the caller sends any
    # other span to outline_edges itself, with its wraparound and errors.
    cdef Py_ssize_t start, end, i, total = 0
    cdef bint dropped = False
    cdef bint closes
    cdef list kept = []

    for span in spans:
        start = span[0]
        end = span[1]
        if end > start and xs[start] == xs[end - 1] and ys[start] == ys[end - 1]:
            end -= 1
        if end - start >= 2:
            closes = xs[start] != xs[end - 1] or ys[start] != ys[end - 1]
            kept.append((start, end, closes))
            total += (end - start - 1) + (1 if closes else 0)
        else:
            dropped = True

    if not kept:
        return None, kept, dropped

    edges = numpy.empty((total, 4), numpy.float64)
    cdef double[:, ::1] out_view = edges
    cdef double* out = &out_view[0, 0]
    for span in kept:
        start = span[0]
        end = span[1]
        closes = span[2]
        for i in range(start, end - 1):
            out[0] = xs[i]
            out[1] = ys[i]
            out[2] = xs[i + 1]
            out[3] = ys[i + 1]
            out += 4
        if closes:
            out[0] = xs[end - 1]
            out[1] = ys[end - 1]
            out[2] = xs[start]
            out[3] = ys[start]
            out += 4
    return edges, kept, dropped


cdef bint _spans_inside(spans, Py_ssize_t count) except -1:
    cdef Py_ssize_t start, end
    for span in spans:
        start = span[0]
        end = span[1]
        if start < 0 or end > count:
            return False
    return True


cdef enum:
    HAS_NAN = 1
    POSITIVE_ZERO = 2
    NEGATIVE_ZERO = 4


cdef int column_range(
    const double* values, Py_ssize_t count, double* low, double* high
) noexcept nogil:
    """The first-found minimum and maximum of a non-empty column, and flags.

    Each bound is kept as `if value < low: low = value` keeps it, from the
    first element, so equal values resolve as a one-sided scan resolves them.
    The flags say whether the column holds a NaN, a positive zero or a
    negative zero, for column_extreme to settle the cases numpy decides.
    """
    cdef Py_ssize_t i
    cdef double value, lo = values[0], hi = values[0]
    cdef int flags = 0
    for i in range(count):
        value = values[i]
        if value != value:
            flags |= HAS_NAN
        elif value == 0.0:
            flags |= POSITIVE_ZERO if 1.0 / value > 0.0 else NEGATIVE_ZERO
        if value < lo:
            lo = value
        if value > hi:
            hi = value
    low[0] = lo
    high[0] = hi
    return flags


cdef object column_extreme(
    const double* values, Py_ssize_t count, object column, bint maximum, double bound, int flags
):
    """numpy's column.min() or column.max() as a float, for a non-empty column.

    A NaN anywhere is the answer, as numpy propagates it: the first one, as
    a scan that stops at it returns. Two zeros of opposite sign compare
    equal, and which of them numpy's reduction returns is its own business,
    so a column holding both is handed to numpy. Otherwise it is `bound`,
    which column_range found by the comparisons a one-sided scan makes; a
    NaN never enters them, since it compares false.
    """
    cdef Py_ssize_t i
    if flags & HAS_NAN:
        for i in range(count):
            if values[i] != values[i]:
                return values[i]
    if (flags & POSITIVE_ZERO) and (flags & NEGATIVE_ZERO):
        return float(column.max() if maximum else column.min())
    return bound


def translated_outline_edges(double[::1] linear_x, double[::1] linear_y, double e, double f, spans):
    """outline_edges over the columns linear_x + e and linear_y + f, with their bounds.

    transformed_outline added the translation to the cached linear columns
    with numpy, called outline_edges, and took the bounds with four numpy
    reductions -- six array operations around one kernel, per glyph drawn.
    This makes the translated columns, the edges and the bounds in one call:
    each column entry is the same double sum, the edges come from
    outline_edges on those columns, and each bound is what numpy's min or max
    of the column gives.

    Returns (column_x, column_y, edges, kept, dropped, bounds), where edges is
    None as outline_edges returns it, and bounds is (min x, min y, max x,
    max y), or None when there are no edges.
    """
    cdef Py_ssize_t count = linear_x.shape[0], i
    if linear_y.shape[0] != count:
        raise ValueError("columns differ in length")
    column_x = numpy.empty(count, numpy.float64)
    column_y = numpy.empty(count, numpy.float64)
    cdef double[::1] xs = column_x
    cdef double[::1] ys = column_y
    for i in range(count):
        xs[i] = linear_x[i] + e
        ys[i] = linear_y[i] + f
    if count and _spans_inside(spans, count):
        edges, kept, dropped = _outline_edges(&xs[0], &ys[0], spans)
    else:
        edges, kept, dropped = outline_edges(xs, ys, spans)
    if edges is None:
        return column_x, column_y, None, kept, dropped, None
    cdef double low_x, high_x, low_y, high_y
    cdef int flags_x, flags_y
    with nogil:
        flags_x = column_range(&xs[0], count, &low_x, &high_x)
        flags_y = column_range(&ys[0], count, &low_y, &high_y)
    bounds = (
        column_extreme(&xs[0], count, column_x, False, low_x, flags_x),
        column_extreme(&ys[0], count, column_y, False, low_y, flags_y),
        column_extreme(&xs[0], count, column_x, True, high_x, flags_x),
        column_extreme(&ys[0], count, column_y, True, high_y, flags_y),
    )
    return column_x, column_y, edges, kept, dropped, bounds
