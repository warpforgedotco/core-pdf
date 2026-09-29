# SPDX-License-Identifier: AGPL-3.0-only

from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.ref cimport Py_INCREF
from libc.math cimport isfinite, rint

cdef enum:
    GRID_WIDTH = 18
    GRID_HEIGHT = 24

cdef double CELL_X_SCALE = GRID_WIDTH - 1
cdef double CELL_Y_SCALE = GRID_HEIGHT - 1


def cell_distance_map(tuple cells):
    """The city-block distance from each cell of the 18 x 24 feature grid to the nearest
    of cells, capped at 42, row-major. Cells outside the grid are ignored."""
    if not cells:
        return ()
    cdef long long distances[GRID_HEIGHT][GRID_WIDTH]
    cdef long long limit = GRID_WIDTH + GRID_HEIGHT, best
    cdef Py_ssize_t x, y
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            distances[y][x] = limit
    for cell in cells:
        x = cell[0]
        y = cell[1]
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            distances[y][x] = 0
    for y in range(GRID_HEIGHT):
        best = distances[y][0]
        for x in range(1, GRID_WIDTH):
            best = best + 1 if best + 1 < distances[y][x] else distances[y][x]
            distances[y][x] = best
        best = distances[y][GRID_WIDTH - 1]
        for x in range(GRID_WIDTH - 2, -1, -1):
            best = best + 1 if best + 1 < distances[y][x] else distances[y][x]
            distances[y][x] = best
    for x in range(GRID_WIDTH):
        best = distances[0][x]
        for y in range(1, GRID_HEIGHT):
            best = best + 1 if best + 1 < distances[y][x] else distances[y][x]
            distances[y][x] = best
        best = distances[GRID_HEIGHT - 1][x]
        for y in range(GRID_HEIGHT - 2, -1, -1):
            best = best + 1 if best + 1 < distances[y][x] else distances[y][x]
            distances[y][x] = best
    result = PyTuple_New(GRID_WIDTH * GRID_HEIGHT)
    cdef object value
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            value = distances[y][x]
            Py_INCREF(value)
            PyTuple_SET_ITEM(result, y * GRID_WIDTH + x, value)
    return result


def glyph_feature_cells(contours):
    """The occupied cells of glyph contours scaled onto the 18 x 24 feature grid,
    sorted by column then row, with the unclamped-to-one bbox width and height.

    Returns () when there are no points, or None for a non-finite coordinate,
    which the caller measures in Python.
    """
    cdef bint occupied[GRID_WIDTH][GRID_HEIGHT]
    cdef Py_ssize_t total = 0, index = 0, x, y
    cdef double min_x = 0.0, min_y = 0.0, max_x = 0.0, max_y = 0.0, px, py, width, height
    cdef double *xs
    cdef double *ys
    for contour in contours:
        total += len(contour)
    if total == 0:
        return ()
    xs = <double *>PyMem_Malloc(total * sizeof(double))
    ys = <double *>PyMem_Malloc(total * sizeof(double))
    if xs == NULL or ys == NULL:
        PyMem_Free(xs)
        PyMem_Free(ys)
        raise MemoryError()
    try:
        for contour in contours:
            for point in contour:
                px, py = point
                if not isfinite(px) or not isfinite(py):
                    return None
                xs[index] = px
                ys[index] = py
                index += 1
        min_x = max_x = xs[0]
        min_y = max_y = ys[0]
        for index in range(1, total):
            if xs[index] < min_x:
                min_x = xs[index]
            if xs[index] > max_x:
                max_x = xs[index]
            if ys[index] < min_y:
                min_y = ys[index]
            if ys[index] > max_y:
                max_y = ys[index]
        width = max_x - min_x
        if width < 1.0:
            width = 1.0
        height = max_y - min_y
        if height < 1.0:
            height = 1.0
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                occupied[x][y] = False
        for index in range(total):
            x = <Py_ssize_t>rint((xs[index] - min_x) / width * CELL_X_SCALE)
            y = <Py_ssize_t>rint((ys[index] - min_y) / height * CELL_Y_SCALE)
            occupied[x][y] = True
    finally:
        PyMem_Free(xs)
        PyMem_Free(ys)
    cells = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if occupied[x][y]:
                cells.append((x, y))
    return (tuple(cells), width, height)
