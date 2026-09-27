# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import fill_edge_rows

GOLDEN_PATH = Path(__file__).parent / "fill_edge_rows_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 809
    assert sum(case["origin"] == "corpus" for case in GOLDEN) == 800
    assert any(len(case["spans"]) > 1 for case in GOLDEN)
    assert any(len(case["expected"]) == 0 for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_golden_vector(index):
    case = GOLDEN[index]
    rows = fill_edge_rows(case["xs"], case["ys"], case["spans"])
    assert rows.dtype == numpy.float64
    assert rows.shape == case["expected"].shape
    assert rows.tobytes() == case["expected"].tobytes()


def test_only_a_zero_length_closing_edge_is_dropped():
    xs = numpy.array([0.0, 0.0, 1.0, 0.0])
    ys = numpy.array([0.0, 0.0, 1.0, 0.0])
    rows = fill_edge_rows(xs, ys, [(0, 4, True)])
    assert rows.tolist() == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]


def test_spans_past_the_points_are_refused():
    xs = numpy.zeros(3)
    with pytest.raises(ValueError, match="span runs past"):
        fill_edge_rows(xs, xs, [(0, 4, False)])
    with pytest.raises(ValueError, match="differ in length"):
        fill_edge_rows(xs, numpy.zeros(2), [])
