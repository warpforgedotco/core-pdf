# SPDX-License-Identifier: AGPL-3.0-only

"""Source-plane accumulation: bitwise agreement with RasterTarget.record_plane.

source_plane_golden.pkl.gz holds 1,500 calls sampled from the 78,244 that
rendering three pages each of five corpus documents made, with the plane
window numpy produced; 400 synthetic windows over random, degenerate and
out-of-range plane values and scales other than 1; and every coverage byte
against eight awkward plane values. Each case is the numpy result of
plane + (1 - plane) * (coverage / 255.0 * scale), so the float32/float64
promotion it goes through is pinned, not just the formula.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import accumulate_source_plane

GOLDEN_PATH = Path(__file__).parent / "source_plane_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 1916
    assert {case["source"] for case in GOLDEN} == {"corpus", "synthetic", "every-byte"}
    assert any(case["scale"] != 1.0 for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_bitwise(index):
    case = GOLDEN[index]
    plane = case["plane"].copy()
    accumulate_source_plane(plane, case["coverage"], case["scale"])
    assert plane.dtype == numpy.float32
    assert plane.tobytes() == case["expected"].tobytes()


def test_a_window_of_a_larger_plane_is_updated_in_place():
    page = numpy.zeros((6, 8), dtype=numpy.float32)
    coverage = numpy.full((2, 3), 255, dtype=numpy.uint8)
    accumulate_source_plane(page[2:4, 3:6], coverage, 1.0)
    expected = numpy.zeros((6, 8), dtype=numpy.float32)
    expected[2:4, 3:6] = 1.0
    assert numpy.array_equal(page, expected)


def test_mismatched_windows_are_rejected():
    with pytest.raises(ValueError, match="differ in shape"):
        accumulate_source_plane(
            numpy.zeros((2, 2), dtype=numpy.float32), numpy.zeros((2, 3), dtype=numpy.uint8), 1.0
        )
