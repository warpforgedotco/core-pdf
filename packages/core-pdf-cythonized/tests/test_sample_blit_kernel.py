# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import sample_opaque_pixels

GOLDEN_PATH = Path(__file__).parent / "sample_blit_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 120
    assert {case["source"].shape[2] for case in GOLDEN} == {1, 3, 4}
    assert any(case["transposed"] for case in GOLDEN)
    assert any(not case["valid_rows"].all() for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_writes_what_the_numpy_gather_wrote(index):
    case = GOLDEN[index]
    target = case["before"].copy()
    rows, columns = len(case["valid_rows"]), len(case["valid_columns"])
    sample_opaque_pixels(
        target[1 : 1 + rows, 2 : 2 + columns],
        case["source"],
        case["source_y"],
        case["source_x"],
        case["valid_rows"].view(numpy.uint8),
        case["valid_columns"].view(numpy.uint8),
        case["transposed"],
    )
    assert numpy.array_equal(target, case["expected"])


def test_an_index_outside_the_source_is_refused():
    target = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    source = numpy.zeros((2, 2, 3), dtype=numpy.uint8)
    ones = numpy.ones(1, dtype=numpy.uint8)
    with pytest.raises(IndexError):
        sample_opaque_pixels(
            target,
            source,
            numpy.array([2], dtype=numpy.intp),
            numpy.array([0], dtype=numpy.intp),
            ones,
            ones,
            False,
        )


def test_a_two_channel_source_is_refused():
    target = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    source = numpy.zeros((1, 1, 2), dtype=numpy.uint8)
    index = numpy.zeros(1, dtype=numpy.intp)
    ones = numpy.ones(1, dtype=numpy.uint8)
    with pytest.raises(ValueError, match="one channel or at least three"):
        sample_opaque_pixels(target, source, index, index, ones, ones, False)
