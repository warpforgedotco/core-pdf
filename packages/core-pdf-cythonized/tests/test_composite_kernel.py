# SPDX-License-Identifier: AGPL-3.0-only

"""Elementary compositing: agreement with the numpy branch it replaced.

elementary_composite_golden.pkl.gz holds every argument of the opaque-normal
branch on three corpus pages -- the destination before the call, the rendered
group, the coverage plane -- with the destination afterwards and the effective
alpha the original returned.

The destination at the real call site is a row slice of the page buffer and so
is not contiguous. That is the one thing a port here can get wrong silently:
reshaping or ravelling such an array gives back a copy, the kernel writes into
the copy, and the page comes out unchanged. test_writes_through_a_strided_view
is the guard.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import composite_elementary_normal

GOLDEN_PATH = Path(__file__).parent / "elementary_composite_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 142
    sizes = [case["source_alpha"].size for case in GOLDEN]
    assert min(sizes) <= 4
    assert max(sizes) > 1000


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_the_numpy_branch_bitwise(index):
    case = GOLDEN[index]
    destination = case["destination"].copy()
    effective = composite_elementary_normal(destination, case["rendered"], case["source_alpha"])
    assert numpy.array_equal(destination, case["after"])
    assert numpy.array_equal(effective, case["effective_alpha"])
    assert effective.dtype == numpy.uint8


def test_writes_through_a_strided_view():
    case = GOLDEN[0]
    height, width = case["source_alpha"].shape
    page = numpy.zeros((height + 6, width + 9, 4), dtype=numpy.uint8)
    page[3 : 3 + height, 4 : 4 + width] = case["destination"]
    view = page[3 : 3 + height, 4 : 4 + width]
    assert not view.flags["C_CONTIGUOUS"]
    composite_elementary_normal(view, case["rendered"], case["source_alpha"])
    assert numpy.array_equal(page[3 : 3 + height, 4 : 4 + width], case["after"])
    # Nothing outside the window moved.
    assert not page[:3].any()
    assert not page[:, :4].any()


def test_zero_coverage_leaves_the_destination_alone():
    destination = numpy.full((2, 3, 4), 7, dtype=numpy.uint8)
    rendered = numpy.full((2, 3, 4), 200, dtype=numpy.uint8)
    alpha = numpy.zeros((2, 3), dtype=numpy.float32)
    effective = composite_elementary_normal(destination, rendered, alpha)
    assert not effective.any()
    assert (destination == 7).all()


def test_partial_coverage_copies_only_the_covered_pixels():
    destination = numpy.full((1, 3, 4), 7, dtype=numpy.uint8)
    rendered = numpy.full((1, 3, 4), 200, dtype=numpy.uint8)
    alpha = numpy.array([[0.0, 0.5, 1.0]], dtype=numpy.float32)
    effective = composite_elementary_normal(destination, rendered, alpha)
    # 0.5 * 255 is 127.5, and rint rounds halves to even.
    assert effective.tolist() == [[0, 128, 255]]
    assert destination[0, 0].tolist() == [7, 7, 7, 7]
    assert destination[0, 1].tolist() == [200, 200, 200, 200]


def test_coverage_outside_the_unit_interval_is_rejected():
    destination = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    rendered = numpy.full((1, 2, 4), 9, dtype=numpy.uint8)
    for bad in (-0.01, 1.01):
        alpha = numpy.array([[0.5, bad]], dtype=numpy.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            composite_elementary_normal(destination, rendered, alpha)
        # The domain check runs to completion before any pixel is written.
        assert not destination.any()


def test_mismatched_shapes_are_rejected():
    alpha = numpy.zeros((2, 3), dtype=numpy.float32)
    with pytest.raises(ValueError):
        composite_elementary_normal(
            numpy.zeros((2, 3, 3), dtype=numpy.uint8),
            numpy.zeros((2, 3, 3), dtype=numpy.uint8),
            alpha,
        )
    with pytest.raises(ValueError):
        composite_elementary_normal(
            numpy.zeros((2, 3, 4), dtype=numpy.uint8),
            numpy.zeros((2, 4, 4), dtype=numpy.uint8),
            alpha,
        )


def test_render_target_uses_the_kernel():
    pytest.importorskip("core_pdf")
    from core_pdf.impl import render_target as target

    assert target.composite_elementary_normal is composite_elementary_normal
