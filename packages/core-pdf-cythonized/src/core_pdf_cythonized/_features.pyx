# SPDX-License-Identifier: AGPL-3.0-only

from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.ref cimport Py_INCREF

cdef enum:
    GRID_WIDTH = 18
    GRID_HEIGHT = 24


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
