"""A deferred path answers as its built subpaths do, without building them."""

import random

import numpy
import pytest

from core_pdf.impl.capture.records import CapturedPath

RECT = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
SHAPES = [
    [RECT],
    [RECT + [(0.0, 0.0)]],
    [[(5.0, 5.0)], RECT],
    [RECT, [(5.0, 5.0)]],
    [RECT, RECT],
    [[(0.0, 0.0), (4.0, 0.0), (4.0, 2.0)]],
    [[(0.0, 0.0), (4.0, 1.0), (4.0, 2.0), (0.0, 2.0)]],
    [[(0.0, 0.0), (4.0, 0.0), (0.0, 2.0), (4.0, 2.0)]],
    [[(0.0, 0.0), (0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]],
    [RECT + [(0.0, 0.0), (4.0, 0.0)]],
    [[(1.0, 1.0)]],
    [],
    [[(0.0, 0.0), (-4.0, 0.0), (-4.0, -2.0), (0.0, -2.0)]],
    [[(0.0, 0.0), (0.0, 2.0), (4.0, 2.0), (4.0, 0.0)], [(1.0, 1.0)]],
]


def deferred(subpaths, flags, *, outline, built):
    xs = [x for points in subpaths for x, _ in points]
    ys = [y for points in subpaths for _, y in points]
    spans, start = [], 0
    for points, flag in zip(subpaths, flags, strict=True):
        spans.append((start, start + len(points), flag))
        start += len(points)
    columns = numpy.asarray(xs, dtype=numpy.float64), numpy.asarray(ys, dtype=numpy.float64)
    if outline:
        path = CapturedPath.deferred_outline(*columns, spans)
    else:
        path = CapturedPath.deferred_flattened(*columns, spans, None, bool(spans))
    if built:
        path.subpaths  # noqa: B018 -- reading it builds the subpaths
    return path


@pytest.mark.parametrize("shape", range(len(SHAPES)))
@pytest.mark.parametrize("outline", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_deferred_and_built_agree(shape, outline, seed):
    subpaths = SHAPES[shape]
    flags = [random.Random(seed * 31 + index).random() < 0.5 for index in range(len(subpaths))]
    lazy = deferred(subpaths, flags, outline=outline, built=False)
    expected = deferred(subpaths, flags, outline=outline, built=True).axis_aligned_rect()
    if outline and (len(subpaths) != 1 or len(subpaths[0]) != 4):
        # An outline's spans are final, so anything but one of four points
        # is no rectangle, whatever its built points would say.
        expected = None
    assert lazy.axis_aligned_rect() == expected
    assert lazy._deferred is not None  # answered without building the subpaths


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("outline", [False, True])
def test_the_edge_array_is_fill_edges(seed, outline):
    rng = random.Random(seed)
    subpaths = []
    for _ in range(rng.randint(0, 5)):
        points = [
            (float(rng.randint(-3, 3)), rng.choice([0.0, -0.0, 1.5, float(rng.randint(0, 4))]))
            for _ in range(rng.randint(1, 6))
        ]
        if rng.random() < 0.4:
            points.append(points[0])
        subpaths.append(points)
    flags = [rng.random() < 0.5 for _ in subpaths]
    lazy = deferred(subpaths, flags, outline=outline, built=False)
    array = lazy.fill_edge_array()
    assert lazy._deferred is not None
    expected = deferred(subpaths, flags, outline=outline, built=True).fill_edges()
    assert array is not None
    assert array.shape == (len(expected), 4)
    assert [tuple(row) for row in array.tolist()] == expected


def test_a_built_path_has_no_edge_array():
    assert CapturedPath([]).fill_edge_array() is None
