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


cdef object column_extreme(double[::1] values, object column, bint maximum):
    """numpy's column.min() or column.max() as a float, for a non-empty column.

    A NaN anywhere is the answer, as numpy propagates it. Two zeros of
    opposite sign compare equal, and which of them numpy's reduction returns
    is its own business, so a column holding both is handed to numpy.
    """
    cdef Py_ssize_t i, count = values.shape[0]
    cdef double best = values[0], value
    cdef bint positive_zero = False, negative_zero = False
    for i in range(count):
        value = values[i]
        if isnan(value):
            return value
        if value == 0.0:
            if 1.0 / value > 0.0:
                positive_zero = True
            else:
                negative_zero = True
        if (value > best) if maximum else (value < best):
            best = value
    if positive_zero and negative_zero:
        return float(column.max() if maximum else column.min())
    return best


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
    edges, kept, dropped = outline_edges(xs, ys, spans)
    if edges is None:
        return column_x, column_y, None, kept, dropped, None
    bounds = (
        column_extreme(xs, column_x, False),
        column_extreme(ys, column_y, False),
        column_extreme(xs, column_x, True),
        column_extreme(ys, column_y, True),
    )
    return column_x, column_y, edges, kept, dropped, bounds
