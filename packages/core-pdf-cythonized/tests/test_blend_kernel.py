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
    from core_pdf.impl.render import blend, target

    assert target.blend_normal_alpha_array_numpy is blend_normal_alpha_array_numpy
    assert not hasattr(blend, "blend_normal_alpha_array_numpy")
