# SPDX-License-Identifier: AGPL-3.0-only

"""The fused glyph fill: what fill_path's four steps made, in one pass.

fill_path quantized glyph_coverage_plane's plane with
numpy.rint(coverage * alpha).astype(uint8) -- and at 255 for the shape --
blended the first with blend_normal_alpha_array_numpy and recorded each into
its group plane with accumulate_source_plane. The reference here is that
pipeline, each step pinned by its own golden vectors, over the glyph coverage
golden cases, at opaque and translucent paints, with and without planes, into
a window of a larger buffer whose surroundings must come through untouched.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import (
    accumulate_source_plane,
    blend_normal_alpha_array_numpy,
    fill_glyph_coverage,
    glyph_coverage_plane,
)

GOLDEN_PATH = Path(__file__).parent / "glyph_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
PAINTS = ((0, 0, 0, 255), (200, 30, 90, 255), (10, 250, 40, 128), (255, 255, 255, 1), (3, 4, 5, 0))


def pipeline(case, rgba, target, source_alpha, source_shape, shape_scale):
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
    if coverage is None:
        return None
    alpha_plane = numpy.rint(coverage * rgba[3]).astype(numpy.uint8)
    blend_normal_alpha_array_numpy(target, rgba, alpha_plane)
    if source_alpha is not None:
        accumulate_source_plane(source_alpha, alpha_plane, 1.0)
    if source_shape is not None:
        shape_plane = numpy.rint(coverage * 255).astype(numpy.uint8)
        accumulate_source_plane(source_shape, shape_plane, shape_scale)
    return True


def fused(case, rgba, target, source_alpha, source_shape, shape_scale):
    return fill_glyph_coverage(
        case["src"],
        case["crop_x0"],
        case["crop_y1"],
        case["scale"],
        int(case["ix0"]),
        int(case["iy0"]),
        case["w"],
        case["h"],
        rgba,
        target,
        source_alpha,
        source_shape,
        shape_scale,
    )


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_one_pass_matches_the_four_steps(index):
    case = GOLDEN[index]
    width, height = case["w"], case["h"]
    rng = numpy.random.default_rng(index)
    for rgba in PAINTS:
        for planes in range(4):
            backdrop = rng.integers(0, 256, (height + 2, width + 3, 4), dtype=numpy.uint8)
            alpha = rng.random((height + 2, width + 3)).astype(numpy.float32)
            shape = rng.random((height + 2, width + 3)).astype(numpy.float32)
            results = []
            for run in (pipeline, fused):
                target = backdrop.copy()
                alpha_plane = alpha.copy() if planes & 1 else None
                shape_plane = shape.copy() if planes & 2 else None
                window = (slice(1, 1 + height), slice(2, 2 + width))
                returned = run(
                    case,
                    rgba,
                    target[window],
                    alpha_plane[window] if alpha_plane is not None else None,
                    shape_plane[window] if shape_plane is not None else None,
                    0.37,
                )
                results.append((returned, target, alpha_plane, shape_plane))
            (want, *want_arrays), (got, *got_arrays) = results
            assert got == want
            for expected, actual in zip(want_arrays, got_arrays, strict=True):
                assert (expected is None) == (actual is None)
                if expected is not None and actual is not None:
                    assert numpy.array_equal(expected, actual)


def test_flat_edges_only_return_none():
    edges = numpy.array([[0.0, 4.0, 9.0, 4.0], [9.0, 7.0, 0.0, 7.0]])
    target = numpy.zeros((10, 9, 4), dtype=numpy.uint8)
    assert (
        fill_glyph_coverage(
            edges, 0.0, 10.0, 1.0, 0, 0, 9, 10, (0, 0, 0, 255), target, None, None, 1.0
        )
        is None
    )
    assert not target.any()


def test_a_mismatched_target_is_refused():
    edges = numpy.array([[0.0, 0.0, 4.0, 4.0]])
    target = numpy.zeros((3, 3, 4), dtype=numpy.uint8)
    with pytest.raises(ValueError, match="target"):
        fill_glyph_coverage(
            edges, 0.0, 10.0, 1.0, 0, 0, 4, 4, (0, 0, 0, 255), target, None, None, 1.0
        )
