# SPDX-License-Identifier: AGPL-3.0-only

"""Supersampled coverage: agreement with the fill_path loop it replaced.

supersampled_coverage_golden.pkl.gz was generated from the deltas branch of
core_pdf.impl.render.target.RasterTarget.fill_path before it was deleted. The
generator first proved its transcription of that loop against every one of
the 15,426 calls a corpus sweep sent through the live code, row by row, and
only then used it. It holds:

- corpus: every call from the documents other than PyMuPDF test_3806 that
  reach the branch, and an even spread by size of test_3806's own 15,407.
  All of them are even-odd fills.
- synthetic: under both fill rules and three device mappings, the shapes the
  corpus does not reach -- a hole wound against and with the outer contour,
  five- and nine-point stars, overlapping and coincident contours, a comb,
  slivers and sub-pixel triangles, a path far past the box, one of only
  horizontal edges -- and forty random polygons with 3 to 20 edges, on both
  sides of the eight edges where the original switched to numpy crossings.

Expected results are stored as the kernel returns them: the count plane
trimmed to its first and last covered rows, with the offset of the first, or
None where the original covered no row.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import supersampled_coverage_plane

GOLDEN_PATH = Path(__file__).parent / "supersampled_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def run(case):
    ix0, iy0, ix1, iy1 = case["box"]
    return supersampled_coverage_plane(
        numpy.asarray(case["edges"], dtype=numpy.float64).reshape(-1, 4),
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        ix0,
        iy0,
        ix1,
        iy1,
        case["fill_rule"] == "evenodd",
    )


def square(x0, y0, x1, y1):
    return numpy.array(
        [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)],
        dtype=numpy.float64,
    )


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 319
    origins = [case["origin"] for case in GOLDEN]
    assert sum(origin.startswith("synthetic/") for origin in origins) == 118
    assert "corpus/test_3806.pdf" in origins
    assert {case["fill_rule"] for case in GOLDEN} == {"evenodd", "nonzero"}
    assert any(case["expected"] is None for case in GOLDEN)
    edge_counts = {len(case["edges"]) for case in GOLDEN}
    assert min(edge_counts) < 8 <= max(edge_counts)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_the_loop(index):
    case = GOLDEN[index]
    got = run(case)
    if case["expected"] is None:
        assert got is None
        return
    plane, first_row = got
    expected_plane, expected_first_row = case["expected"]
    assert first_row == expected_first_row
    assert plane.dtype == numpy.uint8
    assert numpy.array_equal(plane, expected_plane)


def test_fill_rules_differ_only_where_winding_does():
    # A hole wound the same way as its outer contour: nonzero fills it and
    # even-odd does not.
    edges = numpy.concatenate([square(0, 0, 8, 8), square(2, 2, 6, 6)])
    evenodd = supersampled_coverage_plane(edges, 0.0, 8.0, 1.0, 0, 0, 8, 8, True)
    nonzero = supersampled_coverage_plane(edges, 0.0, 8.0, 1.0, 0, 0, 8, 8, False)
    assert evenodd is not None
    assert nonzero is not None
    assert evenodd[0][4, 4] == 0
    assert nonzero[0][4, 4] == 16
    assert evenodd[0][0, 0] == nonzero[0][0, 0] == 16


def test_uncovered_rows_are_trimmed():
    # Covers only device rows 3 and 4 of a ten-row box.
    sampled = supersampled_coverage_plane(square(0, 5, 10, 7), 0.0, 10.0, 1.0, 0, 0, 10, 10, True)
    assert sampled is not None
    plane, first_row = sampled
    assert first_row == 3
    assert plane.shape == (2, 10)
    assert (plane == 16).all()


def test_nothing_to_cover():
    horizontal = numpy.array([(0, 1, 5, 1), (5, 2, 0, 2)], dtype=numpy.float64)
    assert supersampled_coverage_plane(horizontal, 0.0, 5.0, 1.0, 0, 0, 5, 5, True) is None
    outside = square(20, 20, 30, 30)
    assert supersampled_coverage_plane(outside, 0.0, 5.0, 1.0, 0, 0, 5, 5, True) is None
    assert supersampled_coverage_plane(square(0, 0, 5, 5), 0.0, 5.0, 1.0, 0, 0, 0, 5, True) is None


def test_non_finite_spans_raise_what_math_ceil_raised():
    # A crossing at an infinite x is NaN (inf + t * (inf - inf)) and forms no
    # span, in the original too. A span edge goes non-finite in the device
    # mapping instead: a scale that overflows it, or a NaN origin.
    edges = square(0, 0, 4, 4)
    with pytest.raises(OverflowError):
        supersampled_coverage_plane(edges, 0.0, 2.0, 1e308, 0, 0, 4, 4, True)
    with pytest.raises(ValueError):
        supersampled_coverage_plane(edges, numpy.nan, 4.0, 1.0, 0, 0, 4, 4, False)


def test_edges_must_have_four_columns():
    with pytest.raises(ValueError, match="four columns"):
        supersampled_coverage_plane(numpy.zeros((3, 6)), 0.0, 4.0, 1.0, 0, 0, 4, 4, True)


def test_render_target_uses_the_kernel():
    pytest.importorskip("core_pdf")
    from core_pdf.impl.render import target

    assert target.supersampled_coverage_plane is supersampled_coverage_plane
