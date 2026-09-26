# SPDX-License-Identifier: AGPL-3.0-only

"""Soft mask interleaving: the array core_pdf's apply_soft_mask built with numpy.

soft_mask_interleave_golden.pkl.gz was produced by apply_soft_mask before the
kernel replaced its gather and copies: grey and RGB rasters with and without
an alpha channel of their own (which the mask replaces), masks smaller and
larger than the image, and rasters that were strided views of wider arrays.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import interleave_soft_mask

GOLDEN_PATH = Path(__file__).parent / "soft_mask_interleave_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def nearest(size, source):
    return numpy.minimum(source - 1, (numpy.arange(size) * source) // size).astype(numpy.intp)


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 72
    assert any(case["strided"] for case in GOLDEN)
    assert {case["raster"].shape[2] for case in GOLDEN} == {1, 2, 3, 4}


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_builds_what_numpy_built(index):
    case = GOLDEN[index]
    raster = case["raster"]
    if case["strided"]:
        wide = numpy.zeros((raster.shape[0], raster.shape[1] + 5, raster.shape[2]), numpy.uint8)
        wide[:, 2 : 2 + raster.shape[1]] = raster
        raster = wide[:, 2 : 2 + case["raster"].shape[1]]
    colour = 1 if case["color_model"] == "gray" else 3
    mask = case["mask"]
    out = interleave_soft_mask(
        raster,
        colour,
        mask[:, :, 0],
        nearest(raster.shape[0], mask.shape[0]),
        nearest(raster.shape[1], mask.shape[1]),
    )
    assert out.dtype == numpy.uint8
    assert numpy.array_equal(out, case["expected"])


def test_an_index_outside_the_mask_is_refused():
    raster = numpy.zeros((1, 1, 3), dtype=numpy.uint8)
    mask = numpy.zeros((2, 2), dtype=numpy.uint8)
    with pytest.raises(IndexError):
        interleave_soft_mask(
            raster, 3, mask, numpy.array([2], dtype=numpy.intp), numpy.array([0], numpy.intp)
        )


def test_the_image_path_uses_the_kernel():
    pytest.importorskip("core_pdf")
    from core_pdf.impl.graphics import images

    assert images.interleave_soft_mask is interleave_soft_mask
