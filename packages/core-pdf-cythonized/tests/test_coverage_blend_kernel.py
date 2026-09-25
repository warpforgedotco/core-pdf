# SPDX-License-Identifier: AGPL-3.0-only

"""Coverage counts blended pixel by pixel: agreement with fill_path's Python fallback.

coverage_blend_golden.pkl.gz holds 584 fills sampled from the 1,150 that
seven corpus pages sent through fill_path's per-pixel loop under clips that
are not rectangles: the 4x4 counts, the clip mask pixel_in_clip gave, and
the pixel and group-plane windows and paint window before and after the real
loop ran. The corpus paints them in opaque colour, so it also holds 600
synthetic cases over translucent and zero alpha, every combination of the
two group planes, track_shape on and off, shape_alpha other than 1 and
empty, missing and existing paint windows -- from a verbatim copy of the
loop that first had to reproduce all 584 captures, planes and window too.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import blend_coverage_counts

GOLDEN_PATH = Path(__file__).parent / "coverage_blend_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def copy_or_none(plane):
    return None if plane is None else plane.copy()


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 1184
    synthetic = [case for case in GOLDEN if case["source"] == "synthetic"]
    combinations = {
        (c["alpha_before"] is not None, c["shape_before"] is not None) for c in synthetic
    }
    assert combinations == {(True, True), (True, False), (False, True), (False, False)}
    assert {case["rgba"][3] for case in synthetic} >= {0, 1, 128, 255}


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_the_python_loop(index):
    case = GOLDEN[index]
    pixels = case["pixels_before"].copy()
    alpha = copy_or_none(case["alpha_before"])
    shape = copy_or_none(case["shape_before"])
    red, green, blue, alpha_value = case["rgba"]
    touched = blend_coverage_counts(
        pixels,
        numpy.ascontiguousarray(case["counts"]),
        0,
        0,
        case["allowed"],
        red,
        green,
        blue,
        alpha_value,
        alpha,
        shape,
        case["track_shape"],
        case["shape_alpha"],
    )
    assert numpy.array_equal(pixels, case["pixels_after"])
    for got, want in ((alpha, case["alpha_after"]), (shape, case["shape_after"])):
        assert (got is None) == (want is None)
        if got is not None:
            assert got.tobytes() == want.tobytes()
    window = None if case["window_before"] is None else list(case["window_before"])
    if window is not None and touched is not None:
        top, left = case["rows"][0], case["cols"][0]
        x0, y0, x1, y1 = touched
        box = [top + y0, top + y1, left + x0, left + x1]
        window = (
            [
                min(window[0], box[0]),
                max(window[1], box[1]),
                min(window[2], box[2]),
                max(window[3], box[3]),
            ]
            if window
            else box
        )
    assert window == case["window_after"]


def test_counts_past_the_pixels_are_rejected():
    with pytest.raises(ValueError, match="run past"):
        blend_coverage_counts(
            numpy.zeros((2, 2, 4), dtype=numpy.uint8),
            numpy.ones((2, 3), dtype=numpy.uint8),
            0,
            0,
            None,
            0,
            0,
            0,
            255,
            None,
            None,
            False,
            1.0,
        )
