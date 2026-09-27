# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import outline_edges, translated_outline_edges

GOLDEN_PATH = Path(__file__).parent / "outline_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
REAL = GOLDEN["real"]
SYNTHETIC = GOLDEN["synthetic"]


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(REAL) == 150
    assert len(SYNTHETIC) == 6


@pytest.mark.parametrize("index", range(len(REAL)))
def test_kernel_reproduces_numpy_edges_bitwise(index):
    case = REAL[index]
    edges, kept, dropped = outline_edges(case["x"], case["y"], case["spans"])
    assert edges is not None
    assert edges.shape == case["edges"].shape
    assert numpy.array_equal(edges, case["edges"])


@pytest.mark.parametrize("index", range(len(SYNTHETIC)))
def test_synthetic_span_shapes(index):
    xs, ys, spans = SYNTHETIC[index]
    xs = numpy.ascontiguousarray(xs, dtype=numpy.float64)
    ys = numpy.ascontiguousarray(ys, dtype=numpy.float64)
    edges, kept, dropped = outline_edges(xs, ys, spans)
    if edges is None:
        assert not kept
        assert dropped
        return
    expected = sum((end - start - 1) + (1 if closes else 0) for start, end, closes in kept)
    assert edges.shape == (expected, 4)
    for start, end, closes in kept:
        assert end - start >= 2
        assert closes == (xs[start] != xs[end - 1] or ys[start] != ys[end - 1])


def test_a_two_point_span_with_equal_ends_is_dropped_not_kept():
    xs = numpy.array([4.0, 4.0])
    ys = numpy.array([9.0, 9.0])
    edges, kept, dropped = outline_edges(xs, ys, [(0, 2)])
    assert edges is None
    assert kept == []
    assert dropped


@pytest.mark.parametrize("span", [(0, 3), (-1, 2)])
def test_a_span_outside_the_columns_is_refused(span):
    xs = numpy.array([0.0, 1.0])
    ys = numpy.array([0.0, 1.0])
    with pytest.raises(IndexError):
        outline_edges(xs, ys, [span])
    with pytest.raises(IndexError):
        translated_outline_edges(xs, ys, 0.0, 0.0, [span])
