# SPDX-License-Identifier: AGPL-3.0-only

"""Box downsampling: byte-for-byte agreement with the numpy reduceat original.

box_downsample_golden.pkl.gz holds 344 cases computed by the numpy
implementation before it stopped being used for uint8 images: 36 crops of
the images four corpus pages downsample (a 33-megapixel scan among them),
300 random grids of one, three and four channels with target sizes above,
at and below the source, and edge shapes -- single pixels, one-row and
one-column strips, a zero target either way.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import box_downsample_blocks

GOLDEN_PATH = Path(__file__).parent / "box_downsample_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def downsample(grid, target_width, target_height):
    """The caller's edge computation and early returns, as target.box_downsample has them."""
    source_height, source_width, channels = grid.shape
    if target_width <= 0 or target_height <= 0:
        return grid.reshape(-1), source_width, source_height
    if source_width <= target_width and source_height <= target_height:
        return grid.reshape(-1), source_width, source_height
    target_width = min(target_width, source_width)
    target_height = min(target_height, source_height)
    rows = (numpy.arange(target_height + 1, dtype=numpy.int64) * source_height) // target_height
    columns = (numpy.arange(target_width + 1, dtype=numpy.int64) * source_width) // target_width
    return box_downsample_blocks(grid, rows, columns).reshape(-1), target_width, target_height


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 344
    assert {case["source"] for case in GOLDEN} == {"corpus-crop", "synthetic", "edge"}
    assert {case["grid"].shape[2] for case in GOLDEN} == {1, 3, 4}


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_bytewise(index):
    case = GOLDEN[index]
    expected, width, height = case["expected"]
    got, got_width, got_height = downsample(case["grid"], *case["target"])
    assert (got_width, got_height) == (width, height)
    assert numpy.array_equal(got, expected)


def test_a_strided_view_is_read_without_copying_first():
    grid = numpy.arange(6 * 8 * 3, dtype=numpy.uint8).reshape(6, 8, 3)
    view = grid[:, ::2]
    rows = numpy.array([0, 3, 6], dtype=numpy.int64)
    columns = numpy.array([0, 2, 4], dtype=numpy.int64)
    assert numpy.array_equal(
        box_downsample_blocks(view, rows, columns),
        box_downsample_blocks(numpy.ascontiguousarray(view), rows, columns),
    )


def test_edges_past_the_image_are_rejected():
    grid = numpy.zeros((2, 2, 1), dtype=numpy.uint8)
    with pytest.raises(ValueError, match="past the image"):
        box_downsample_blocks(grid, numpy.array([0, 3]), numpy.array([0, 2]))


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
def test_packed_rows_sum_as_strided_ones_do(channels):
    rng = numpy.random.default_rng(channels)
    grid = rng.integers(0, 256, size=(37, 53, channels), dtype=numpy.uint8)
    padded = numpy.zeros((37, 53, channels + 1), dtype=numpy.uint8)
    padded[:, :, :channels] = grid
    strided = padded[:, :, :channels]
    rows = (numpy.arange(8, dtype=numpy.int64) * 37) // 7
    columns = (numpy.arange(12, dtype=numpy.int64) * 53) // 11
    assert numpy.array_equal(
        box_downsample_blocks(grid, rows, columns), box_downsample_blocks(strided, rows, columns)
    )
