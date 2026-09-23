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
