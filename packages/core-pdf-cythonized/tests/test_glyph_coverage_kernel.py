# SPDX-License-Identifier: AGPL-3.0-only

"""Fused glyph coverage: agreement with the numpy prep it replaced.

glyph_coverage_golden.pkl.gz holds the arguments of every fast-path fill on
three corpus pages -- the page-space edge array, the crop origin, the scale,
the plane's pixel origin and its size -- together with the coverage plane the
original produced by building a device-space copy with twelve numpy operations
and handing it to signed_area_coverage.

The transform is four affine expressions per edge, so fusing it into the
kernel's own read of each edge is value-for-value identical rather than merely
close: the assertions below demand array_equal, not allclose.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import glyph_coverage_plane, signed_area_coverage

GOLDEN_PATH = Path(__file__).parent / "glyph_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def plane(case):
    return glyph_coverage_plane(
        case["src"],
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        case["ix0"],
        case["iy0"],
        case["w"],
        case["h"],
    )


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 152
    assert min(len(case["src"]) for case in GOLDEN) < 20
    assert max(len(case["src"]) for case in GOLDEN) > 200


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_prep_bitwise(index):
    case = GOLDEN[index]
    got = plane(case)
    assert got is not None
    assert got.shape == case["cov"].shape
    assert numpy.array_equal(got, case["cov"])


def test_flat_edges_only_returns_none():
    # Every edge horizontal in page space: the original's sloped mask was empty
    # and it returned before touching the buffer.
    edges = numpy.array([[0.0, 4.0, 9.0, 4.0], [9.0, 7.0, 0.0, 7.0]])
    assert glyph_coverage_plane(edges, 0.0, 10.0, 1.0, 0.0, 0.0, 9, 10) is None


def test_flat_edges_are_skipped_not_counted():
    sloped = numpy.array([[0.0, 0.0, 0.0, 10.0], [0.0, 10.0, 5.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    flat = numpy.array([[0.0, 3.0, 5.0, 3.0]])
    mixed = numpy.concatenate([sloped, flat])
    with_flat = glyph_coverage_plane(mixed, 0.0, 10.0, 1.0, 0.0, 0.0, 5, 10)
    without = glyph_coverage_plane(sloped, 0.0, 10.0, 1.0, 0.0, 0.0, 5, 10)
    assert with_flat is not None
    assert without is not None
    assert numpy.array_equal(with_flat, without)


def test_it_agrees_with_the_device_space_entry_point():
    # The same edges, transformed by hand, must reach the shared core with the
    # same values -- this is the seam the fusion removed.
    src = numpy.array([[1.0, 2.0, 3.0, 9.0], [3.0, 9.0, 6.0, 2.0], [6.0, 2.0, 1.0, 2.0]])
    crop_x0, crop_y1, scale, ix0, iy0 = 0.5, 11.0, 1.5, 2.0, 3.0
    device = numpy.empty(src.shape, dtype=numpy.float64)
    device[:, 0] = (src[:, 0] - crop_x0) * scale - ix0
    device[:, 1] = (crop_y1 - src[:, 1]) * scale - iy0
    device[:, 2] = (src[:, 2] - crop_x0) * scale - ix0
    device[:, 3] = (crop_y1 - src[:, 3]) * scale - iy0
    fused = glyph_coverage_plane(src, crop_x0, crop_y1, scale, ix0, iy0, 8, 12)
    assert fused is not None
    assert numpy.array_equal(fused, signed_area_coverage(device, 8, 12))


def test_an_empty_plane_matches_the_device_space_entry_point():
    src = numpy.array([[0.0, 0.0, 0.0, 10.0], [0.0, 10.0, 5.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    empty = glyph_coverage_plane(src, 0.0, 10.0, 1.0, 0.0, 0.0, 0, 0)
    assert empty is not None
    assert empty.shape == (0, 0)


def test_render_target_uses_the_kernel():
    pytest.importorskip("core_pdf")
    from core_pdf.impl.render import target

    assert target.glyph_coverage_plane is glyph_coverage_plane
