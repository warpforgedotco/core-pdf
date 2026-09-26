# SPDX-License-Identifier: AGPL-3.0-only
"""4x4 supersampled path coverage (core_pdf.impl.render.target.fill_path).

A path fill that glyph_coverage_plane cannot take -- in practice an even-odd
fill, whose winding the signed-area accumulation does not model -- was
covered by sampling: four sample rows a pixel row, the edge crossings of each
sorted into spans by the fill rule, and four sample columns turned into
per-pixel counts through a running sum of deltas. PyMuPDF test_3806, the
slowest page in the corpus, is 16,262 even-odd fills, and this loop was 4.7 s
of its 8.9 s rasterize: 5.6 million calls to math.ceil and 3.9 million each
to min and max, on numbers.

The kernel is that loop, term for term. The expressions below are the ones
both of the original's crossing routines computed -- the numpy one for eight
or more edges and the scalar one below that agreed exactly -- so a sample
lands on the same bits and a span on the same pixel. What the original
blended row by row, the caller now blends as one plane: a row with no
coverage was skipped, and blending or recording zero coverage changes
nothing, so trimming the plane to its first and last covered rows is the
whole of the difference.

math.ceil raised on a span edge that was not finite. The kernel raises the
same exceptions, before returning anything rather than partway down the
fill; capture rejects non-finite operands, so only a transform overflowing
could produce one.
"""

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.math cimport ceil
from libc.stdlib cimport qsort

from core_pdf_cythonized._pymath cimport check_integral

import numpy

cdef enum:
    SAMPLES = 4


cdef struct Crossing:
    double x
    int direction


cdef int compare_crossings(const void *left, const void *right) noexcept nogil:
    cdef double a = (<const Crossing *> left).x
    cdef double b = (<const Crossing *> right).x
    return (a > b) - (a < b)


cdef int add_span(
    double start_x,
    double end_x,
    double crop_x0,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t ix1,
    int *deltas,
) except -1:
    # The pixel columns whose four sample columns fall inside the span.
    cdef double span_start = (start_x - crop_x0) * scale
    cdef double span_end = (end_x - crop_x0) * scale
    cdef double offset, first, last
    cdef Py_ssize_t sx, start, end
    for sx in range(SAMPLES):
        offset = (sx + 0.5) / SAMPLES
        first = ceil(span_start - offset)
        last = ceil(span_end - offset)
        # In the original's order: the start column is converted first.
        check_integral(first)
        check_integral(last)
        # Clamp while still a double: an unclamped ceil can be far outside
        # what a Py_ssize_t holds.
        start = ix0 if first <= ix0 else (ix1 if first >= ix1 else <Py_ssize_t> first)
        end = ix1 if last >= ix1 else (ix0 if last <= ix0 else <Py_ssize_t> last)
        if end > start:
            deltas[start - ix0] += 1
            deltas[end - ix0] -= 1
    return 0


cdef int add_sample_row(
    Crossing *crossings,
    Py_ssize_t count,
    bint evenodd,
    double crop_x0,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t ix1,
    int *deltas,
) except -1:
    # fill_path_crossing_spans, feeding each span straight to add_span.
    cdef double first, second, previous
    cdef Py_ssize_t index
    cdef int winding, delta
    cdef bint started
    if count == 0:
        return 0
    if count == 2:
        first = crossings[0].x
        second = crossings[1].x
        if second < first:
            first, second = second, first
        if second > first:
            add_span(first, second, crop_x0, scale, ix0, ix1, deltas)
        return 0
    qsort(crossings, count, sizeof(Crossing), compare_crossings)
    if evenodd:
        index = 0
        while index + 1 < count:
            if crossings[index + 1].x > crossings[index].x:
                add_span(crossings[index].x, crossings[index + 1].x, crop_x0, scale, ix0, ix1, deltas)
            index += 2
        return 0
    winding = 0
    started = False
    previous = 0.0
    index = 0
    while index < count:
        first = crossings[index].x
        if started and winding != 0 and first > previous:
            add_span(previous, first, crop_x0, scale, ix0, ix1, deltas)
        delta = 0
        while index < count and crossings[index].x == first:
            delta += crossings[index].direction
            index += 1
        winding += delta
        previous = first
        started = True
    return 0


def supersampled_coverage_plane(
    edges,
    double crop_x0,
    double crop_y1,
    double scale,
    Py_ssize_t ix0,
    Py_ssize_t iy0,
    Py_ssize_t ix1,
    Py_ssize_t iy1,
    bint evenodd,
):
    """Sample counts, 0 to 16, for the pixels of a box, under a fill rule.

    ``edges`` holds rows of page-space x0, y0, x1, y1. Returns
    ``(counts, first_row)``, where ``counts`` covers the rows from
    ``iy0 + first_row`` through the last row with any coverage and the full
    width ``ix1 - ix0``; or ``None`` when no row is covered, which includes
    every path whose edges are all horizontal.
    """
    cdef const double[:, ::1] view = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    if view.shape[0] and view.shape[1] != 4:
        raise ValueError("edges must have four columns")
    cdef Py_ssize_t total = view.shape[0]
    cdef Py_ssize_t width = ix1 - ix0
    cdef Py_ssize_t height = iy1 - iy0
    cdef Py_ssize_t kept = 0
    cdef Py_ssize_t i
    for i in range(total):
        if view[i, 1] != view[i, 3]:
            kept += 1
    if kept == 0 or width <= 0 or height <= 0:
        return None

    counts = numpy.zeros((height, width), dtype=numpy.uint8)
    cdef unsigned char[:, ::1] out = counts
    # Per kept edge: x0, y0, x1, y1, low, high -- the original's segments.
    cdef double *segments = <double *> PyMem_Malloc(kept * 6 * sizeof(double))
    cdef Crossing *crossings = <Crossing *> PyMem_Malloc(kept * sizeof(Crossing))
    cdef int *deltas = <int *> PyMem_Malloc((width + 1) * sizeof(int))
    if segments == NULL or crossings == NULL or deltas == NULL:
        PyMem_Free(segments)
        PyMem_Free(crossings)
        PyMem_Free(deltas)
        raise MemoryError

    cdef Py_ssize_t kept_index = 0, row, sample_row, column, crossing_count
    cdef Py_ssize_t first_row = -1, last_row = -1
    cdef double page_y, y0, y1
    cdef double *segment
    cdef int running
    cdef bint covered
    try:
        for i in range(total):
            y0 = view[i, 1]
            y1 = view[i, 3]
            if y0 == y1:
                continue
            segment = segments + kept_index * 6
            segment[0] = view[i, 0]
            segment[1] = y0
            segment[2] = view[i, 2]
            segment[3] = y1
            segment[4] = y1 if y1 < y0 else y0
            segment[5] = y0 if y0 > y1 else y1
            kept_index += 1

        for row in range(height):
            for column in range(width + 1):
                deltas[column] = 0
            for sample_row in range(SAMPLES):
                page_y = crop_y1 - (<double> (iy0 + row) + (sample_row + 0.5) / SAMPLES) / scale
                crossing_count = 0
                for i in range(kept):
                    segment = segments + i * 6
                    if not (segment[4] <= page_y < segment[5]):
                        continue
                    crossings[crossing_count].x = segment[0] + (
                        (page_y - segment[1]) / (segment[3] - segment[1]) * (segment[2] - segment[0])
                    )
                    crossings[crossing_count].direction = 1 if segment[3] > segment[1] else -1
                    crossing_count += 1
                add_sample_row(crossings, crossing_count, evenodd, crop_x0, scale, ix0, ix1, deltas)
            running = 0
            covered = False
            for column in range(width):
                running += deltas[column]
                out[row, column] = <unsigned char> running
                if running:
                    covered = True
            if covered:
                if first_row < 0:
                    first_row = row
                last_row = row
    finally:
        PyMem_Free(segments)
        PyMem_Free(crossings)
        PyMem_Free(deltas)

    if first_row < 0:
        return None
    return counts[first_row : last_row + 1], first_row
