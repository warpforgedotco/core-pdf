# SPDX-License-Identifier: AGPL-3.0-only

"""The fused knockout wrapper against the core implementation it replaced.

knockout_golden.pkl.gz holds buffers captured while rasterizing three corpus
pages, with the output produced by core's composite_knockout_group before it
was deleted. That function built a mask, took three boolean fancy-index
copies, called into spec and scattered the result back; the kernel does all of
it in one pass, so this pins the fused version to the staged one.

The ISO 32000-2 11.4.x conformance tests for the algorithm itself live beside
this file in test_knockout_groups.py, moved here from core-pdf-spec along with
the function.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import composite_knockout_group

GOLDEN_PATH = Path(__file__).parent / "knockout_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 120


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_fused_wrapper_reproduces_staged_output_bitwise(index):
    case = GOLDEN[index]
    destination = case["destination"].copy()
    group_alpha = case["group_alpha"].copy()
    composite_knockout_group(
        destination,
        case["backdrop"],
        case["element"],
        group_alpha,
        case["element_alpha"],
        case["shape"],
    )
    assert numpy.array_equal(destination, case["expected_destination"])
    assert numpy.array_equal(group_alpha, case["expected_group_alpha"])


def test_pixels_outside_the_shape_are_left_alone():
    """The original masked on `shape > 0` after folding in element alpha."""
    destination = numpy.full((2, 2, 4), 7, numpy.uint8)
    before = destination.copy()
    group_alpha = numpy.zeros((2, 2), numpy.float32)
    composite_knockout_group(
        destination,
        numpy.zeros((2, 2, 4), numpy.uint8),
        numpy.full((2, 2, 4), 200, numpy.uint8),
        group_alpha,
        numpy.zeros((2, 2), numpy.uint8),
        numpy.zeros((2, 2), numpy.float32),
    )
    assert numpy.array_equal(destination, before)


def test_render_target_uses_the_kernel():
    # These wire-up assertions need the consumer installed. The kernel tests
    # otherwise stand alone, so cibuildwheel can run the golden vectors
    # against a freshly built wheel with nothing else present.
    pytest.importorskip("core_pdf")
    from core_pdf.impl.render import target

    assert target.composite_knockout_group is composite_knockout_group


def test_spec_no_longer_owns_the_algorithm():
    pytest.importorskip("core_pdf_spec")
    from core_pdf_spec.s_11_transparency import groups

    assert not hasattr(groups, "composite_knockout_element")
    assert "composite_knockout_element" not in groups.__all__
    # remove_group_backdrop and its helper stay behind.
    assert hasattr(groups, "remove_group_backdrop")
