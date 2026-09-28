# SPDX-License-Identifier: AGPL-3.0-only

import random

import numpy

from core_pdf_cythonized import cell_distance_map

WIDTH = 18
HEIGHT = 24


def numpy_cell_distance_map(cells):
    if not cells:
        return ()
    distances = numpy.full((HEIGHT, WIDTH), WIDTH + HEIGHT, dtype=numpy.int64)
    for x, y in cells:
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            distances[y, x] = 0
    for axis, extent in ((1, WIDTH), (0, HEIGHT)):
        offsets = numpy.arange(extent, dtype=numpy.int64)
        if axis == 0:
            offsets = offsets[:, None]
        forward = numpy.minimum.accumulate(distances - offsets, axis=axis) + offsets
        flip = (slice(None, None, -1),) if axis == 0 else (slice(None), slice(None, None, -1))
        backward = numpy.minimum.accumulate((distances + offsets)[flip], axis=axis)[flip] - offsets
        distances = numpy.minimum(forward, backward)
    return tuple(distances.reshape(-1).tolist())


def test_compiled_transform_matches_the_array_passes():
    rng = random.Random(0)
    for _ in range(5000):
        cells = tuple(
            sorted({(rng.randint(-3, 20), rng.randint(-3, 26)) for _ in range(rng.randint(0, 80))})
        )
        assert cell_distance_map(cells) == numpy_cell_distance_map(cells)
