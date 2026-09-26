"""A deferred path answers as its built subpaths do, without building them."""

import math
import random

import numpy
import pytest

from core_pdf.impl.capture_records import CapturedPath

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


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("outline", [False, True])
def test_bounds_come_from_the_columns_as_the_subpaths_give_them(seed, outline):
    rng = random.Random(seed)
    values = [0.0, -0.0, 1.5, -2.0, math.inf, -math.inf, math.nan, 3.25]
    subpaths = [
        [(rng.choice(values), rng.choice(values)) for _ in range(rng.randint(1, 5))]
        for _ in range(rng.randint(0, 4))
    ]
    flags = [rng.random() < 0.5 for _ in subpaths]
    lazy = deferred(subpaths, flags, outline=outline, built=False)
    lazy._summary = None  # as an outline or a coalesced path has it
    built = deferred(subpaths, flags, outline=outline, built=True)
    # repr keeps NaN in its place and a zero's sign, which == does not.
    assert repr(lazy.bbox()) == repr(built.bbox())
    assert lazy.has_segments() == built.has_segments()
    assert lazy._deferred is not None


def test_coalesced_strokes_keep_their_columns():
    first = deferred([[(0.0, 0.0), (1.0, 1.0)]], [False], outline=False, built=False)
    second = deferred(
        [[(2.0, 2.0)], [(3.0, 3.0), (4.0, -1.0)]], [False, True], outline=False, built=False
    )
    merged = first.coalesced_with(second)
    assert merged is not None
    assert merged.subpath_count() == 3
    assert merged.bbox() == (0.0, -1.0, 4.0, 3.0)
    expected = CapturedPath([*first.subpaths, *second.subpaths])
    assert [(s.points, s.closed) for s in merged.subpaths] == [
        (s.points, s.closed) for s in expected.subpaths
    ]
    outline = deferred([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]], [True], outline=True, built=False)
    assert first.coalesced_with(outline) is None
