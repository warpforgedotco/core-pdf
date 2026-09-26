# SPDX-License-Identifier: AGPL-3.0-only

"""Normal-mode alpha compositing: bitwise agreement with the numpy original.

blend_golden.pkl.gz holds buffers captured while rasterizing three corpus
pages, each paired with the output the numpy implementation produced before it
was deleted.

Bitwise is achievable here only because the kernel stays in float32. numpy
promoted the uint8 buffers to float32 and did every step there; computing in
double and narrowing at the end lands on different bytes for some pixels.
That is what the typed float constants in the kernel are protecting.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import blend_normal_alpha_array_numpy

GOLDEN_PATH = Path(__file__).parent / "blend_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_both_call_shapes():
    assert len(GOLDEN) == 270
    ranks = {case["target"].ndim for case in GOLDEN}
    # target.py blends whole rectangles and single scanlines; the scanline
    # shape is not reached by the pages these buffers came from, so those
    # cases are single rows sliced out of the rectangular ones.
    assert ranks == {2, 3}


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_output_bitwise(index):
    case = GOLDEN[index]
    target = case["target"].copy()
    blend_normal_alpha_array_numpy(target, tuple(case["rgba"]), case["alpha"])
    assert numpy.array_equal(target, case["expected"])


def test_zero_alpha_pixels_are_left_untouched():
    """The original's final copyto carried a `where=alpha > 0` mask."""
    target = numpy.arange(2 * 3 * 4, dtype=numpy.uint8).reshape(2, 3, 4)
    before = target.copy()
    alpha = numpy.zeros((2, 3), numpy.uint8)
    alpha[1, 1] = 200
    blend_normal_alpha_array_numpy(target, (10, 20, 30, 255), alpha)
    untouched = numpy.ones((2, 3), bool)
    untouched[1, 1] = False
    assert numpy.array_equal(target[untouched], before[untouched])
    assert not numpy.array_equal(target[1, 1], before[1, 1])


def test_unsupported_rank_is_rejected_rather_than_silently_wrong():
    target = numpy.zeros((2, 2, 2, 4), numpy.uint8)
    with pytest.raises(ValueError):
        blend_normal_alpha_array_numpy(target, (1, 2, 3, 4), numpy.ones((2, 2, 2), numpy.uint8))


def test_empty_buffer_is_a_no_op():
    target = numpy.zeros((0, 0, 4), numpy.uint8)
    blend_normal_alpha_array_numpy(target, (1, 2, 3, 4), numpy.zeros((0, 0), numpy.uint8))


def test_render_target_uses_the_kernel():
    # These wire-up assertions need the consumer installed. The kernel tests
    # otherwise stand alone, so cibuildwheel can run the golden vectors
    # against a freshly built wheel with nothing else present.
    pytest.importorskip("core_pdf")
    from core_pdf.impl import render_target as target

    assert target.blend_normal_alpha_array_numpy is blend_normal_alpha_array_numpy


def reference_pixel(destination, raw, rgba):
    """blend_one step by step in float32, as the C kernel computes it."""
    f = numpy.float32
    zero, one, scale = f(0.0), f(1.0), f(255.0)
    sa = f(min(raw, int(rgba[3]))) / scale
    da = f(destination[3]) / scale
    oa = sa + da * (one - sa)
    safe = oa if oa > zero else one
    weight = da * (one - sa)
    out = []
    for source, channel in zip(rgba[:3], destination[:3], strict=True):
        value = numpy.rint((f(source) * sa + f(channel) * weight) / safe)
        out.append(int(min(max(value, zero), scale)))
    value = numpy.rint(oa * scale)
    out.append(int(min(max(value, zero), scale)))
    return out


@pytest.mark.parametrize("cap", [0, 128, 254, 255, 300])
@pytest.mark.parametrize("colour", [(0, 128, 255), (-5, 300, 17), (3, 200, 255)])
def test_full_coverage_matches_the_general_arithmetic(cap, colour):
    # Opaque colour at full coverage takes a shortcut in the kernel; it must
    # land where blend_one's float arithmetic would, for every destination.
    rng = numpy.random.default_rng(cap)
    destination = rng.integers(0, 256, size=(256, 3, 4), dtype=numpy.uint8)
    destination[:, :, 3] = numpy.arange(256, dtype=numpy.uint8)[:, None]
    coverage = numpy.zeros((256, 3), dtype=numpy.uint8)
    coverage[:, 0] = 255
    coverage[:, 1] = 254
    rgba = (*colour, cap)
    expected = destination.copy()
    for i in range(256):
        for j in range(3):
            if coverage[i, j]:
                expected[i, j] = reference_pixel(destination[i, j], int(coverage[i, j]), rgba)
    blended = destination.copy()
    blend_normal_alpha_array_numpy(blended, rgba, coverage)
    assert numpy.array_equal(blended, expected)
    row = destination[:, 0].copy()
    blend_normal_alpha_array_numpy(row, rgba, coverage[:, 0].copy())
    assert numpy.array_equal(row, expected[:, 0])
