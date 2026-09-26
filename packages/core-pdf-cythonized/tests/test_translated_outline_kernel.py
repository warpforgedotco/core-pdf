# SPDX-License-Identifier: AGPL-3.0-only

"""Translated outline edges: what transformed_outline's numpy steps made.

transformed_outline added the translation to the cached linear columns with
numpy, handed the sums to outline_edges and took the bounds as float(column
.min()) and .max(). The reference here is exactly that, over the golden
outlines at several translations, plus columns that hold a NaN or zeros of
both signs -- the two cases where a hand-written minimum could part from
numpy's.
"""

import gzip
import math
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import outline_edges, translated_outline_edges

GOLDEN_PATH = Path(__file__).parent / "outline_golden.pkl.gz"
REAL = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))["real"]
TRANSLATIONS = ((0.0, 0.0), (12.5, -3.25), (-0.1, 1e6), (-0.0, -0.0))


def reference(linear_x, linear_y, e, f, spans):
    column_x = linear_x + e
    column_y = linear_y + f
    edges, kept, dropped = outline_edges(column_x, column_y, spans)
    bounds = (
        None
        if edges is None
        else (
            float(column_x.min()),
            float(column_y.min()),
            float(column_x.max()),
            float(column_y.max()),
        )
    )
    return column_x, column_y, edges, kept, dropped, bounds


def same(left, right):
    if isinstance(left, numpy.ndarray) or isinstance(right, numpy.ndarray):
        return numpy.array_equal(left, right, equal_nan=True)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return left == right and math.copysign(1.0, left) == math.copysign(1.0, right)
    return left == right


@pytest.mark.parametrize("index", range(len(REAL)))
def test_translated_edges_and_bounds_match_numpy(index):
    case = REAL[index]
    linear_x = numpy.ascontiguousarray(case["x"], dtype=numpy.float64)
    linear_y = numpy.ascontiguousarray(case["y"], dtype=numpy.float64)
    for e, f in TRANSLATIONS:
        got = translated_outline_edges(linear_x, linear_y, e, f, case["spans"])
        want = reference(linear_x, linear_y, e, f, case["spans"])
        assert same(got, want)


@pytest.mark.parametrize(
    ("xs", "ys"),
    [
        ([0.0, -0.0, 3.0, 1.0], [1.0, 2.0, 0.0, 5.0]),
        ([-0.0, 2.0, 3.0, 1.0], [0.0, -0.0, 1.0, 5.0]),
        ([1.0, math.nan, 3.0, 1.5], [1.0, 2.0, 3.0, 4.0]),
        ([1.0, 2.0, 3.0, 1.5], [math.nan, 2.0, 3.0, 4.0]),
    ],
)
def test_nans_and_signed_zeros_bound_as_numpy_does(xs, ys):
    linear_x = numpy.array(xs, dtype=numpy.float64)
    linear_y = numpy.array(ys, dtype=numpy.float64)
    spans = [(0, 4)]
    # A -0.0 translation keeps both zeros: 0.0 + -0.0 is 0.0, -0.0 + -0.0 is -0.0.
    got = translated_outline_edges(linear_x, linear_y, -0.0, -0.0, spans)
    want = reference(linear_x, linear_y, -0.0, -0.0, spans)
    assert same(got, want)
