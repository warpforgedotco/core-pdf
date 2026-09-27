# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import signed_area_coverage

GOLDEN_PATH = Path(__file__).parent / "coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 306
    assert any(case["edges"].shape[0] == 0 for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_output_bitwise(index):
    case = GOLDEN[index]
    got = signed_area_coverage(case["edges"], case["width"], case["height"])
    assert got.shape == case["expected"].shape
    assert numpy.array_equal(got, case["expected"])


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 4), (4, 0), (-1, 4), (4, -1)],
)
def test_degenerate_dimensions_yield_an_empty_buffer(width, height):
    edges = numpy.array([[0.0, 0.0, 4.0, 4.0]])
    out = signed_area_coverage(edges, width, height)
    assert out.shape == (max(height, 0), max(width, 0))


def test_coverage_is_bounded_to_unit_range():
    for case in GOLDEN:
        out = signed_area_coverage(case["edges"], case["width"], case["height"])
        assert out.size == 0 or (out.min() >= 0.0 and out.max() <= 1.0)
