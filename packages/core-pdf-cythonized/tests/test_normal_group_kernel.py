# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import composite_normal_group

GOLDEN_PATH = Path(__file__).parent / "normal_group_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 455
    routes = {case["route"] for case in GOLDEN}
    assert routes == {"noop", "copy", "opaque-backdrop", "empty-backdrop", "general"}
    origins = {case["origin"].split("/")[0] for case in GOLDEN}
    assert {"corpus-window", "synthetic", "ties"} <= origins
    scales = {(case["source_alpha_scale"], case["target_alpha_scale"]) for case in GOLDEN}
    assert (1.4, 1.0) in scales
    assert (0.7, 1.3) in scales


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_the_numpy_function(index):
    case = GOLDEN[index]
    destination = case["destination"].copy()
    composite_normal_group(
        destination, case["source"], case["source_alpha_scale"], case["target_alpha_scale"]
    )
    assert numpy.array_equal(destination, case["after"])


def test_writes_through_a_strided_view():
    case = next(case for case in GOLDEN if case["route"] == "general")
    height, width = case["destination"].shape[:2]
    page = numpy.full((height + 4, width + 6, 4), 3, dtype=numpy.uint8)
    page[2 : 2 + height, 3 : 3 + width] = case["destination"]
    view = page[2 : 2 + height, 3 : 3 + width]
    assert not view.flags["C_CONTIGUOUS"]
    composite_normal_group(
        view, case["source"], case["source_alpha_scale"], case["target_alpha_scale"]
    )
    assert numpy.array_equal(page[2 : 2 + height, 3 : 3 + width], case["after"])
    assert (page[:2] == 3).all()
    assert (page[:, :3] == 3).all()


def test_non_finite_scales_are_rejected_before_any_write():
    destination = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    source = numpy.full((1, 2, 4), 200, dtype=numpy.uint8)
    for scales in ((numpy.nan, 1.0), (1.0, numpy.inf)):
        with pytest.raises(ValueError, match="finite"):
            composite_normal_group(destination, source, *scales)
    assert not destination.any()


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        composite_normal_group(
            numpy.zeros((2, 3, 3), dtype=numpy.uint8),
            numpy.zeros((2, 3, 3), dtype=numpy.uint8),
            1.0,
        )
    with pytest.raises(ValueError):
        composite_normal_group(
            numpy.zeros((2, 3, 4), dtype=numpy.uint8),
            numpy.zeros((2, 4, 4), dtype=numpy.uint8),
            1.0,
        )


@pytest.mark.parametrize("index", range(0, len(GOLDEN), 7))
@pytest.mark.parametrize("scale", [1.0, 0.6, 1.4, 0.0, -0.5, 2e-3])
def test_the_effective_plane_is_composite_group_intos(index, scale):
    case = GOLDEN[index]
    source = case["source"]
    expected = numpy.clip(
        numpy.rint(source[..., 3].astype(numpy.float64) * scale), 0.0, 255.0
    ).astype(numpy.uint8)
    destination = case["destination"].copy()
    plane = composite_normal_group(destination, source, scale, 1.0, True)
    assert plane is not None
    assert plane.dtype == numpy.uint8
    assert numpy.array_equal(plane, expected)
    alone = case["destination"].copy()
    assert composite_normal_group(alone, source, scale) is None
    assert numpy.array_equal(alone, destination)
