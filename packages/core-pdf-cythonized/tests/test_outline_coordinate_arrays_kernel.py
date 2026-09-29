# SPDX-License-Identifier: AGPL-3.0-only

import random

import numpy

from core_pdf_cythonized import outline_coordinate_arrays


def python_outline_arrays(contours):
    xs, ys, spans = [], [], []
    for contour in contours:
        if len(contour) < 2:
            continue
        start = len(xs)
        for x, y in contour:
            xs.append(x)
            ys.append(y)
        spans.append((start, len(xs)))
    if not spans:
        return None
    return (
        numpy.asarray(xs, dtype=numpy.float64),
        numpy.asarray(ys, dtype=numpy.float64),
        tuple(spans),
    )


def test_compiled_columns_match_the_list_build():
    rng = random.Random(0)
    for _ in range(3000):
        contours = tuple(
            tuple(
                (rng.uniform(-1e4, 1e4), rng.choice((rng.uniform(-1e4, 1e4), rng.randint(-9, 9))))
                for _ in range(rng.randint(0, 12))
            )
            for _ in range(rng.randint(0, 5))
        )
        expected = python_outline_arrays(contours)
        actual = outline_coordinate_arrays(contours)
        if expected is None:
            assert actual is None
            continue
        assert actual is not None
        assert actual[0].dtype == numpy.float64
        assert actual[0].tolist() == expected[0].tolist()
        assert actual[1].tolist() == expected[1].tolist()
        assert actual[2] == expected[2]


def test_single_point_contours_are_skipped():
    assert outline_coordinate_arrays(((), ((1.0, 1.0),))) is None
    arrays = outline_coordinate_arrays((((1.0, 1.0),), ((0.0, 1.0), (2.0, 3.0))))
    assert arrays is not None
    xs, ys, spans = arrays
    assert xs.tolist() == [0.0, 2.0]
    assert ys.tolist() == [1.0, 3.0]
    assert spans == ((0, 2),)
