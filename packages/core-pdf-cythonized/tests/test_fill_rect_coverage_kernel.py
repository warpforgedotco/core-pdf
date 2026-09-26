# SPDX-License-Identifier: AGPL-3.0-only

"""The fused rectangle fill: what fill_rect's five steps made, in one pass.

fill_rect built rect_coverage_plane at the paint's alpha, blended it with
blend_normal_alpha_array_numpy, recorded it with accumulate_source_plane, and
built and recorded the plane again at 255 for a group's shape. The reference
here is that pipeline, each kernel pinned by its own golden vectors, over the
rectangles of the rect coverage golden cases, into a window of a larger
buffer whose surroundings must come through untouched.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import (
    accumulate_source_plane,
    blend_normal_alpha_array_numpy,
    fill_rect_coverage,
    rect_coverage_plane,
)

GOLDEN_PATH = Path(__file__).parent / "rect_coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
# Every eighth rectangle keeps the run short; each is tried at five paints
# and four plane combinations.
CASES = GOLDEN[::8]
PAINTS = ((0, 0, 0, 255), (200, 30, 90, 255), (10, 250, 40, 128), (255, 255, 255, 1), (3, 4, 5, 0))


def pipeline(box, rgba, target, source_alpha, source_shape, shape_scale):
    ix0, ix1, iy0, iy1, left, right, top, bottom = box
    alpha_plane = rect_coverage_plane(ix0, ix1, iy0, iy1, left, right, top, bottom, rgba[3])
    blend_normal_alpha_array_numpy(target, rgba, alpha_plane)
    if source_alpha is not None:
        accumulate_source_plane(source_alpha, alpha_plane, 1.0)
    if source_shape is not None:
        shape_plane = rect_coverage_plane(ix0, ix1, iy0, iy1, left, right, top, bottom, 255)
        accumulate_source_plane(source_shape, shape_plane, shape_scale)


def fused(box, rgba, target, source_alpha, source_shape, shape_scale):
    ix0, ix1, iy0, iy1, left, right, top, bottom = box
    fill_rect_coverage(
        ix0,
        ix1,
        iy0,
        iy1,
        left,
        right,
        top,
        bottom,
        rgba,
        target,
        source_alpha,
        source_shape,
        shape_scale,
    )


@pytest.mark.parametrize("index", range(len(CASES)))
def test_one_pass_matches_the_five_steps(index):
    box = tuple(CASES[index]["args"][:8])
    ix0, ix1, iy0, iy1 = box[:4]
    width, height = max(0, ix1 - ix0), max(0, iy1 - iy0)
    if not width or not height:
        pytest.skip("an empty rectangle paints nothing")
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
                run(
                    box,
                    rgba,
                    target[window],
                    alpha_plane[window] if alpha_plane is not None else None,
                    shape_plane[window] if shape_plane is not None else None,
                    0.37,
                )
                results.append((target, alpha_plane, shape_plane))
            for expected, actual in zip(results[0], results[1], strict=True):
                assert (expected is None) == (actual is None)
                if expected is not None and actual is not None:
                    assert numpy.array_equal(expected, actual)


def test_a_mismatched_target_is_refused():
    target = numpy.zeros((2, 2, 4), dtype=numpy.uint8)
    with pytest.raises(ValueError, match="target"):
        fill_rect_coverage(0, 3, 0, 3, 0.0, 3.0, 0.0, 3.0, (0, 0, 0, 255), target, None, None, 1.0)


@pytest.mark.parametrize("seed", range(24))
def test_flat_bands_match_the_five_steps(seed):
    # Large rectangles with fractional edges over a flat backdrop and flat
    # planes: fully covered rows, the opaque block they write, and runs of
    # repeated inputs, which the kernel does not recompute.
    rng = numpy.random.default_rng(seed)
    width, height = int(rng.integers(3, 60)), int(rng.integers(3, 40))
    ix0, iy0 = int(rng.integers(0, 5)), int(rng.integers(0, 5))
    left = ix0 + rng.choice([0.0, 0.25, 0.5, 0.9])
    right = ix0 + width - rng.choice([0.0, 0.3, 0.75])
    top = iy0 + rng.choice([0.0, 0.4])
    bottom = iy0 + height - rng.choice([0.0, 0.6])
    box = (ix0, ix0 + width, iy0, iy0 + height, left, right, top, bottom)
    rgba = PAINTS[seed % len(PAINTS)]
    backdrop = numpy.empty((height + 2, width + 3, 4), dtype=numpy.uint8)
    backdrop[...] = rng.integers(0, 256, 4, dtype=numpy.uint8)
    backdrop[0, 0] = 7
    planes = seed % 4
    alpha = numpy.full((height + 2, width + 3), 0.25, dtype=numpy.float32)
    shape = numpy.full((height + 2, width + 3), -0.0, dtype=numpy.float32)
    shape[1::2] = 0.0
    results = []
    for run in (pipeline, fused):
        for packed in (True, False):
            if run is pipeline and not packed:
                continue
            target = backdrop.copy()
            if not packed:
                # Channels apart from each other, so the kernel's generic path runs.
                spread = numpy.zeros((height + 2, width + 3, 8), dtype=numpy.uint8)
                spread[..., ::2] = target
                target = spread[..., ::2]
            alpha_plane = alpha.copy() if planes & 1 else None
            shape_plane = shape.copy() if planes & 2 else None
            window = (slice(1, 1 + height), slice(2, 2 + width))
            run(
                box,
                rgba,
                target[window],
                alpha_plane[window] if alpha_plane is not None else None,
                shape_plane[window] if shape_plane is not None else None,
                0.37,
            )
            results.append(
                (
                    numpy.ascontiguousarray(target).tobytes(),
                    None if alpha_plane is None else alpha_plane.tobytes(),
                    None if shape_plane is None else shape_plane.tobytes(),
                )
            )
    assert results[0] == results[1] == results[2]
