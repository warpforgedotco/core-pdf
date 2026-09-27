# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import blend_visible_rgba

GOLDEN_PATH = Path(__file__).parent / "blend_visible_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 1100
    assert {case["mode"] for case in GOLDEN} == {0, 1, 2}
    assert {case["origin"] for case in GOLDEN} == {"corpus", "fuzz"}
    assert any(not isinstance(case["red"], numpy.ndarray) for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_golden_vector(index):
    case = GOLDEN[index]
    destination = case["destination"].copy()
    blend_visible_rgba(
        destination,
        case["visible"].view(numpy.uint8),
        case["red"],
        case["green"],
        case["blue"],
        case["alpha"],
        case["mode"],
    )
    assert destination.tobytes() == case["expected"].tobytes()


def test_writes_through_a_strided_window():
    page = numpy.zeros((4, 5, 4), dtype=numpy.uint8)
    visible = numpy.ones((2, 2), dtype=numpy.bool_)
    blend_visible_rgba(page[1:3, 2:4], visible.view(numpy.uint8), 1.0, 0.0, 0.0, numpy.ones(4), 0)
    assert page[1:3, 2:4].tolist() == [[[255, 0, 0, 255]] * 2] * 2
    assert page.sum() == 4 * 510


def test_mismatched_values_are_refused():
    page = numpy.zeros((2, 2, 4), dtype=numpy.uint8)
    visible = numpy.ones((2, 2), dtype=numpy.uint8)
    with pytest.raises(ValueError, match="do not match"):
        blend_visible_rgba(page, visible, 0.0, 0.0, 0.0, numpy.ones(3), 0)
    with pytest.raises(ValueError, match="unsupported"):
        blend_visible_rgba(page, visible, 0.0, 0.0, 0.0, numpy.ones(4), 3)
