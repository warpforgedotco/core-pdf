# SPDX-License-Identifier: AGPL-3.0-only

"""Distinct uint16 rows and the uint8 scatter back.

No golden vectors here: nothing is computed on colour values. The contract
is structural -- the distinct rows, indexed by the returned inverse, are the
input exactly, in first-seen order -- and is checked directly.
"""

import numpy
import pytest

from core_pdf_cythonized import distinct_uint16_rows, gather_uint8_rows


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("palette", [1, 7, 300, 70_000])
def test_distinct_rows_rebuild_the_input_in_first_seen_order(channels, palette):
    rng = numpy.random.default_rng(palette * 10 + channels)
    colours = rng.integers(0, 65536, size=(palette, channels), dtype=numpy.uint16)
    samples = colours[rng.integers(0, palette, size=200_000)]
    result = distinct_uint16_rows(samples, 1 << 30)
    assert result is not None
    distinct, inverse = result
    assert inverse.dtype == numpy.uint32
    assert numpy.array_equal(distinct[inverse], samples)
    _, first = numpy.unique(samples, axis=0, return_index=True)
    assert numpy.array_equal(distinct, samples[numpy.sort(first)])


def test_extreme_keys_are_distinct_rows():
    samples = numpy.array(
        [[0, 0, 0, 0], [65535] * 4, [0, 0, 0, 1], [1, 0, 0, 0], [65535] * 4, [0, 0, 0, 0]],
        dtype=numpy.uint16,
    )
    result = distinct_uint16_rows(samples, 100)
    assert result is not None
    distinct, inverse = result
    assert distinct.tolist() == [[0, 0, 0, 0], [65535] * 4, [0, 0, 0, 1], [1, 0, 0, 0]]
    assert inverse.tolist() == [0, 1, 2, 3, 1, 0]


def test_too_many_distinct_rows_gives_up():
    samples = numpy.arange(3000, dtype=numpy.uint16).reshape(-1, 3)
    assert distinct_uint16_rows(samples, 999) is None
    assert distinct_uint16_rows(samples, 1000) is not None


def test_growth_keeps_every_row():
    samples = numpy.arange(200_000 * 2, dtype=numpy.uint32).astype(numpy.uint16).reshape(-1, 2)
    result = distinct_uint16_rows(samples, 1 << 30)
    assert result is not None
    distinct, inverse = result
    assert numpy.array_equal(distinct[inverse], samples)


def test_gather_matches_numpy_take():
    rng = numpy.random.default_rng(3)
    table = rng.integers(0, 256, size=(50, 3), dtype=numpy.uint8)
    indices = rng.integers(0, 50, size=10_000).astype(numpy.uint32)
    assert numpy.array_equal(gather_uint8_rows(table, indices), table[indices])
    with pytest.raises(IndexError):
        gather_uint8_rows(table, numpy.array([50], dtype=numpy.uint32))


@pytest.mark.parametrize("channels", [0, 5])
def test_unsupported_widths_are_rejected(channels):
    with pytest.raises(ValueError, match="one to four"):
        distinct_uint16_rows(numpy.zeros((3, channels), dtype=numpy.uint16), 10)
