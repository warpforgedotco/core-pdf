# SPDX-License-Identifier: AGPL-3.0-only

import cython
import numpy
from cython.cimports.core_pdf_cythonized._pymath import check_integral
from cython.cimports.core_pdf_cythonized._supersample import SAMPLES
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc
from cython.cimports.libc.math import ceil
from cython.cimports.libc.stdlib import qsort

Crossing = cython.struct(x=cython.double, direction=cython.int)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compare_crossings(
    left: cython.pointer[cython.const[cython.void]],
    right: cython.pointer[cython.const[cython.void]],
) -> cython.int:
    a: cython.double = cython.cast(cython.pointer[cython.const[Crossing]], left).x
    b: cython.double = cython.cast(cython.pointer[cython.const[Crossing]], right).x
    return (a > b) - (a < b)


@cython.cfunc
@cython.exceptval(-1, check=False)
def add_span(
    start_x: cython.double,
    end_x: cython.double,
    crop_x0: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    deltas: cython.p_int,
) -> cython.int:
    span_start: cython.double = (start_x - crop_x0) * scale
    span_end: cython.double = (end_x - crop_x0) * scale
    offset: cython.double
    first: cython.double
    last: cython.double
    sx: cython.Py_ssize_t
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    for sx in range(SAMPLES):
        offset = (sx + 0.5) / SAMPLES
        first = ceil(span_start - offset)
        last = ceil(span_end - offset)
        check_integral(first)
        check_integral(last)
        start = (
            ix0
            if first <= ix0
            else (ix1 if first >= ix1 else cython.cast(cython.Py_ssize_t, first))
        )
        end = ix1 if last >= ix1 else (ix0 if last <= ix0 else cython.cast(cython.Py_ssize_t, last))
        if end > start:
            deltas[start - ix0] += 1
            deltas[end - ix0] -= 1
    return 0


@cython.cfunc
@cython.exceptval(-1, check=False)
def add_sample_row(
    crossings: cython.pointer[Crossing],
    count: cython.Py_ssize_t,
    evenodd: cython.bint,
    crop_x0: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    deltas: cython.p_int,
) -> cython.int:
    first: cython.double
    second: cython.double
    previous: cython.double
    index: cython.Py_ssize_t
    winding: cython.int
    delta: cython.int
    started: cython.bint
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
    qsort(crossings, count, cython.sizeof(Crossing), compare_crossings)
    if evenodd:
        index = 0
        while index + 1 < count:
            if crossings[index + 1].x > crossings[index].x:
                add_span(
                    crossings[index].x, crossings[index + 1].x, crop_x0, scale, ix0, ix1, deltas
                )
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
    crop_x0: cython.double,
    crop_y1: cython.double,
    scale: cython.double,
    ix0: cython.Py_ssize_t,
    iy0: cython.Py_ssize_t,
    ix1: cython.Py_ssize_t,
    iy1: cython.Py_ssize_t,
    evenodd: cython.bint,
):
    view: cython.const[cython.double][:, ::1] = numpy.ascontiguousarray(edges, dtype=numpy.float64)
    if view.shape[0] and view.shape[1] != 4:
        raise ValueError("edges must have four columns")
    total: cython.Py_ssize_t = view.shape[0]
    width: cython.Py_ssize_t = ix1 - ix0
    height: cython.Py_ssize_t = iy1 - iy0
    kept: cython.Py_ssize_t = 0
    i: cython.Py_ssize_t
    for i in range(total):
        if view[i, 1] != view[i, 3]:
            kept += 1
    if kept == 0 or width <= 0 or height <= 0:
        return None

    counts = numpy.zeros((height, width), dtype=numpy.uint8)
    out: cython.uchar[:, ::1] = counts
    segments: cython.p_double = cython.cast(
        cython.p_double, PyMem_Malloc(kept * 6 * cython.sizeof(cython.double))
    )
    crossings: cython.pointer[Crossing] = cython.cast(
        cython.pointer[Crossing], PyMem_Malloc(kept * cython.sizeof(Crossing))
    )
    deltas: cython.p_int = cython.cast(
        cython.p_int, PyMem_Malloc((width + 1) * cython.sizeof(cython.int))
    )
    if segments == cython.NULL or crossings == cython.NULL or deltas == cython.NULL:
        PyMem_Free(segments)
        PyMem_Free(crossings)
        PyMem_Free(deltas)
        raise MemoryError

    kept_index: cython.Py_ssize_t = 0
    row: cython.Py_ssize_t
    sample_row: cython.Py_ssize_t
    column: cython.Py_ssize_t
    crossing_count: cython.Py_ssize_t
    first_row: cython.Py_ssize_t = -1
    last_row: cython.Py_ssize_t = -1
    page_y: cython.double
    y0: cython.double
    y1: cython.double
    segment: cython.p_double
    running: cython.int
    covered: cython.bint
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
                page_y = (
                    crop_y1
                    - (cython.cast(cython.double, iy0 + row) + (sample_row + 0.5) / SAMPLES) / scale
                )
                crossing_count = 0
                for i in range(kept):
                    segment = segments + i * 6
                    if not (segment[4] <= page_y < segment[5]):
                        continue
                    crossings[crossing_count].x = segment[0] + (
                        (page_y - segment[1])
                        / (segment[3] - segment[1])
                        * (segment[2] - segment[0])
                    )
                    crossings[crossing_count].direction = 1 if segment[3] > segment[1] else -1
                    crossing_count += 1
                add_sample_row(crossings, crossing_count, evenodd, crop_x0, scale, ix0, ix1, deltas)
            running = 0
            covered = False
            for column in range(width):
                running += deltas[column]
                out[row, column] = cython.cast(cython.uchar, running)
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
