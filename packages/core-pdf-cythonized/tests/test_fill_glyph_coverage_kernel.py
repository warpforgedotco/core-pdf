# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import (
    accumulate_source_plane,
    blend_normal_alpha_array_numpy,
    fill_glyph_coverage,
    fill_glyph_coverage_at,
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


def optional_plane(rng, shape, present):
    return rng.random(shape, dtype=numpy.float32) if present else None


def copied_plane(plane):
    return None if plane is None else plane.copy()


def assert_same_plane(have, want):
    if want is None:
        assert have is None
    else:
        assert have is not None
        assert have.tobytes() == want.tobytes()


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_whole_plane_entry_matches_the_windowed_one(index):
    case = GOLDEN[index]
    height, width = case["h"], case["w"]
    ix0, iy0 = int(case["ix0"]), int(case["iy0"])
    rng = numpy.random.default_rng(2000 + index)
    arguments = (case["src"], case["crop_x0"], case["crop_y1"], case["scale"], ix0, iy0)
    rows = slice(iy0, iy0 + height)
    columns = slice(ix0, ix0 + width)
    for rgba in PAINTS[:3]:
        for with_alpha, with_shape in ((False, False), (True, False), (True, True)):
            shape = (max(iy0, 0) + height + 2, max(ix0, 0) + width + 2)
            target = rng.integers(0, 256, (*shape, 4), dtype=numpy.uint8)
            source_alpha = optional_plane(rng, shape, with_alpha)
            source_shape = optional_plane(rng, shape, with_shape)
            whole_target = target.copy()
            whole_alpha = copied_plane(source_alpha)
            whole_shape = copied_plane(source_shape)
            try:
                expected = fill_glyph_coverage(
                    *arguments,
                    width,
                    height,
                    rgba,
                    target[rows, columns],
                    None if source_alpha is None else source_alpha[rows, columns],
                    None if source_shape is None else source_shape[rows, columns],
                    0.75,
                )
            except ValueError:
                with pytest.raises(ValueError):
                    fill_glyph_coverage_at(
                        *arguments,
                        width,
                        height,
                        rgba,
                        whole_target,
                        whole_alpha,
                        whole_shape,
                        0.75,
                    )
                continue
            got = fill_glyph_coverage_at(
                *arguments, width, height, rgba, whole_target, whole_alpha, whole_shape, 0.75
            )
            assert got == expected
            assert whole_target.tobytes() == target.tobytes()
            assert_same_plane(whole_alpha, source_alpha)
            assert_same_plane(whole_shape, source_shape)
