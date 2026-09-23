# SPDX-License-Identifier: AGPL-3.0-only

"""Rectangle coverage: bitwise agreement with the numpy original.

rect_coverage_golden.pkl.gz holds 1,200 argument sets captured while
rasterizing four corpus pages, plus 209 synthetic ones covering branches real
pages do not reach: empty spans, fully covered planes, sub-pixel slivers,
zero-area and inverted bounds, negative origins, and a zero scale.

The kernel exploits separability -- the plane is the outer product of a row
profile and a column profile -- and forms the row/column product before
applying the scale, because numpy.outer did the multiply first and scaled
afterwards. Scaling each profile instead would round differently.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import rect_coverage_plane

GOLDEN_PATH = Path(__file__).parent / "rect_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 1409
    assert any(case["args"][0] == case["args"][1] for case in GOLDEN)  # empty span
    assert any(case["args"][8] == 0 for case in GOLDEN)  # zero scale


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_output_bitwise(index):
    case = GOLDEN[index]
    got = rect_coverage_plane(*case["args"])
    assert got.shape == case["expected"].shape
    assert numpy.array_equal(got, case["expected"])


@pytest.mark.parametrize(
    ("ix0", "ix1", "iy0", "iy1"),
    [(3, 3, 0, 4), (0, 4, 3, 3), (5, 2, 0, 4), (0, 4, 5, 2)],
)
def test_empty_or_inverted_spans_give_an_empty_plane(ix0, ix1, iy0, iy1):
    plane = rect_coverage_plane(ix0, ix1, iy0, iy1, 0.0, 9.0, 0.0, 9.0, 255)
    assert plane.size == 0
    assert plane.shape == (max(iy1 - iy0, 0), max(ix1 - ix0, 0))


def test_full_coverage_saturates_and_no_coverage_is_zero():
    covered = rect_coverage_plane(0, 3, 0, 3, -9.0, 9.0, -9.0, 9.0, 255)
    assert numpy.array_equal(covered, numpy.full((3, 3), 255, numpy.uint8))
    missed = rect_coverage_plane(0, 3, 0, 3, 50.0, 60.0, 50.0, 60.0, 255)
    assert numpy.array_equal(missed, numpy.zeros((3, 3), numpy.uint8))


def test_render_target_uses_the_kernel():
    from core_pdf.impl.render import paths, target

    assert target.rect_coverage_plane is rect_coverage_plane
    assert not hasattr(paths, "rect_coverage_plane")
