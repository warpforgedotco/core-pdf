# SPDX-License-Identifier: AGPL-3.0-only

"""A soft mask's alpha plane and the bytes it holds, as numpy took them.

resolve_soft_mask copied ``view[..., 3]`` and marked ``present[alpha] =
True``; alpha_channel does both in one pass. Both originals are numpy
one-liners, so they serve as the reference directly.
"""

import numpy
import pytest

from core_pdf_cythonized import alpha_channel


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("presence", [False, True])
def test_the_plane_and_its_values_are_numpys(seed, presence):
    rng = numpy.random.default_rng(seed)
    height, width = rng.integers(1, 40, size=2)
    pixels = rng.integers(0, 256, size=(height, width, 4), dtype=numpy.uint8)
    if seed % 3 == 0:
        # A mask plane is mostly a few values, like a rendered group's alpha.
        pixels[..., 3] = rng.choice(numpy.array([0, 128, 255], dtype=numpy.uint8), (height, width))
    alpha, present = alpha_channel(pixels, presence)
    expected = pixels[..., 3].copy()
    assert alpha.flags["C_CONTIGUOUS"]
    assert alpha.dtype == numpy.uint8
    assert numpy.array_equal(alpha, expected)
    if not presence:
        assert present is None
        return
    assert present is not None
    marks = numpy.zeros(256, dtype=numpy.bool_)
    marks[expected] = True
    assert present.dtype == numpy.bool_
    assert numpy.array_equal(present, marks)


def test_an_empty_plane_holds_nothing():
    alpha, present = alpha_channel(numpy.zeros((0, 5, 4), dtype=numpy.uint8), True)
    assert alpha.shape == (0, 5)
    assert present is not None
    assert not present.any()


def test_only_rgba_is_taken():
    with pytest.raises(ValueError, match="RGBA"):
        alpha_channel(numpy.zeros((2, 2, 3), dtype=numpy.uint8), True)
