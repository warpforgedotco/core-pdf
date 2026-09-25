# SPDX-License-Identifier: AGPL-3.0-only

"""An elementary group into its knockout parent: the three steps, fused.

composite_group built the element as a copy of the parent's initial backdrop,
ran composite_elementary_normal over it, then composite_knockout_group, then
parent_shape += (1.0 - parent_shape) * shape in numpy. The reference is that
sequence, both kernels pinned by their own golden vectors, over random windows
of larger buffers: coverage that is zero, partial and full, shapes below and
above the element's alpha, with and without a parent shape plane.
"""

import numpy
import pytest

from core_pdf_cythonized import (
    composite_elementary_knockout,
    composite_elementary_normal,
    composite_knockout_group,
)


def planes(rng, rows, cols):
    source_alpha = rng.random((rows, cols)).astype(numpy.float32)
    source_alpha[rng.random((rows, cols)) < 0.3] = 0.0
    source_alpha[rng.random((rows, cols)) < 0.1] = 1.0
    return {
        "destination": rng.integers(0, 256, (rows + 2, cols + 3, 4), dtype=numpy.uint8),
        "backdrop": rng.integers(0, 256, (rows + 2, cols + 3, 4), dtype=numpy.uint8),
        "rendered": rng.integers(0, 256, (rows, cols, 4), dtype=numpy.uint8),
        "source_alpha": source_alpha,
        "group_alpha": rng.random((rows + 2, cols + 3)).astype(numpy.float32),
        "shape": (rng.random((rows, cols)) * 1.2 - 0.1).astype(numpy.float32),
        "parent_shape": rng.random((rows + 2, cols + 3)).astype(numpy.float32),
    }


WINDOW = (slice(1, -1), slice(2, -1))


def steps(p, with_parent_shape):
    initial = p["backdrop"][WINDOW]
    element = initial.copy()
    effective = composite_elementary_normal(element, p["rendered"], p["source_alpha"])
    composite_knockout_group(
        p["destination"][WINDOW], initial, element, p["group_alpha"][WINDOW], effective, p["shape"]
    )
    if with_parent_shape:
        parent_shape = p["parent_shape"][WINDOW]
        parent_shape += (1.0 - parent_shape) * p["shape"]


def fused(p, with_parent_shape):
    composite_elementary_knockout(
        p["destination"][WINDOW],
        p["backdrop"][WINDOW],
        p["rendered"],
        p["source_alpha"],
        p["group_alpha"][WINDOW],
        p["shape"],
        p["parent_shape"][WINDOW] if with_parent_shape else None,
    )


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("with_parent_shape", [False, True])
def test_one_pass_matches_the_three_steps(seed, with_parent_shape):
    rng = numpy.random.default_rng(seed)
    rows, cols = int(rng.integers(1, 20)), int(rng.integers(1, 20))
    original = planes(rng, rows, cols)
    want = {name: value.copy() for name, value in original.items()}
    got = {name: value.copy() for name, value in original.items()}
    steps(want, with_parent_shape)
    fused(got, with_parent_shape)
    for name in original:
        assert numpy.array_equal(got[name], want[name]), name


def test_out_of_range_alpha_is_refused_before_anything_is_written():
    rng = numpy.random.default_rng(1)
    p = planes(rng, 4, 4)
    p["source_alpha"][3, 3] = 1.5
    before = p["destination"].copy()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fused(p, True)
    assert numpy.array_equal(p["destination"], before)
