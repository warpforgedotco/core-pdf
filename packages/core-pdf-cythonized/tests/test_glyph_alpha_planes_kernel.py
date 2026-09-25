# SPDX-License-Identifier: AGPL-3.0-only

"""Quantized glyph coverage: the bytes fill_path used to make with numpy.

fill_path quantized glyph_coverage_plane's float64 plane with
numpy.rint(coverage * alpha).astype(uint8), and at 255 for a group's shape
plane. glyph_alpha_planes writes both from the running sum. The reference
here is exactly that composition, over the glyph coverage golden cases --
glyph_coverage_plane itself is pinned by them in test_glyph_coverage_kernel --
at the alphas a paint can have, including a non-integral one.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import glyph_alpha_planes, glyph_coverage_plane

GOLDEN_PATH = Path(__file__).parent / "glyph_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
ALPHAS = (255, 254, 128, 77, 1, 0, 127.5)


def coverage_of(case):
    coverage = glyph_coverage_plane(
        case["src"],
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        case["ix0"],
        case["iy0"],
        case["w"],
        case["h"],
    )
    assert coverage is not None
    return coverage


def planes_of(case, alpha, want_shape):
    planes = glyph_alpha_planes(
        case["src"],
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        case["ix0"],
        case["iy0"],
        case["w"],
        case["h"],
        alpha,
        want_shape,
    )
    assert planes is not None
    return planes


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_planes_are_the_numpy_quantization_of_the_coverage(index):
    case = GOLDEN[index]
    coverage = coverage_of(case)
    for alpha in ALPHAS:
        alpha_plane, shape_plane = planes_of(case, alpha, True)
        assert alpha_plane.dtype == numpy.uint8
        assert numpy.array_equal(alpha_plane, numpy.rint(coverage * alpha).astype(numpy.uint8))
        assert shape_plane is not None
        assert numpy.array_equal(shape_plane, numpy.rint(coverage * 255).astype(numpy.uint8))


def test_no_shape_plane_unless_asked():
    assert planes_of(GOLDEN[0], 255, False)[1] is None


def test_flat_edges_only_return_none():
    edges = numpy.array([[0.0, 4.0, 9.0, 4.0], [9.0, 7.0, 0.0, 7.0]])
    assert glyph_alpha_planes(edges, 0.0, 10.0, 1.0, 0.0, 0.0, 9, 10, 255, True) is None


@pytest.mark.parametrize(("width", "height"), [(0, 4), (4, 0), (-1, 3)])
def test_degenerate_planes_are_empty(width, height):
    edges = numpy.array([[0.0, 0.0, 4.0, 4.0]])
    planes = glyph_alpha_planes(edges, 0.0, 10.0, 1.0, 0.0, 0.0, width, height, 255, True)
    assert planes is not None
    alpha_plane, shape_plane = planes
    assert shape_plane is not None
    assert alpha_plane.shape == shape_plane.shape == (max(height, 0), max(width, 0))


def test_render_target_uses_the_kernel():
    # glyph_coverage_plane stays as the pinned reference; fill_path reaches the
    # shared accumulation through this quantized front end.
    pytest.importorskip("core_pdf")
    from core_pdf.impl.render import target

    assert target.glyph_alpha_planes is glyph_alpha_planes
