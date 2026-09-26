# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import stroke_segment_samples

GOLDEN_PATH = Path(__file__).parent / "stroke_segment_golden.pkl.gz"
RECORDED = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
GOLDEN = [case for case in RECORDED if not case["round_cap"]]


def run(case, pixels=None):
    ix0, iy0, ix1, iy1 = case["box"]
    crop_x0, crop_y1 = case["crop"]
    x0, y0, _, _ = case["segment"]
    dx, dy = case["dxdy"]
    red, green, blue, alpha = case["rgba"]
    pixels = case["before"].copy() if pixels is None else pixels
    covered = stroke_segment_samples(
        pixels,
        ix0,
        iy0,
        ix0,
        iy0,
        ix1,
        iy1,
        crop_x0,
        crop_y1,
        case["scale"],
        x0,
        y0,
        dx,
        dy,
        case["seg_len2"],
        case["half2"],
        case["projection_extension"],
        red,
        green,
        blue,
        alpha,
        case["allowed"],
    )
    return pixels, covered


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(RECORDED) == 5200
    assert any(case["allowed"] is not None for case in GOLDEN)
    assert any(case["allowed"] is None for case in GOLDEN)
    synthetic = [case for case in GOLDEN if case["source"] == "synthetic"]
    assert {case["line_cap"] for case in synthetic} == {0, 2}
    assert {case["rgba"][3] for case in synthetic} >= {0, 1, 128, 254, 255}


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_the_python_loop_bytewise(index):
    case = GOLDEN[index]
    pixels, covered = run(case)
    assert numpy.array_equal(pixels, case["expected"])
    changed = numpy.argwhere((pixels != case["before"]).any(axis=2))
    if len(changed):
        assert covered is not None
        ix0, iy0 = case["box"][:2]
        x0, y0, x1, y1 = covered
        assert (changed[:, 1] + ix0 >= x0).all()
        assert (changed[:, 1] + ix0 < x1).all()
        assert (changed[:, 0] + iy0 >= y0).all()
        assert (changed[:, 0] + iy0 < y1).all()


def test_a_box_past_the_pixels_is_rejected():
    case = GOLDEN[0]
    ix0, iy0, ix1, iy1 = case["box"]
    small = numpy.zeros((max(1, iy1 - iy0 - 1), ix1 - ix0, 4), dtype=numpy.uint8)
    if iy1 - iy0 > 1:
        with pytest.raises(ValueError, match="runs past"):
            run(case, small)


def test_a_mask_of_the_wrong_size_is_rejected():
    case = next(case for case in GOLDEN if case["allowed"] is not None)
    with pytest.raises(ValueError, match="one byte per pixel"):
        run({**case, "allowed": case["allowed"] + b"\x00"})
