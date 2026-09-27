# SPDX-License-Identifier: AGPL-3.0-only


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
    pytest.importorskip("core_pdf")
    from core_pdf.impl import render_target as target

    assert target.composite_knockout_group is composite_knockout_group


def test_spec_no_longer_owns_the_algorithm():
    pytest.importorskip("core_pdf_spec")
    from core_pdf_spec.s_11_transparency import groups

    assert "composite_knockout_element" not in groups.__all__
    assert hasattr(groups, "remove_group_backdrop")
