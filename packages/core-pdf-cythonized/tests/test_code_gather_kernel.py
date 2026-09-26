# SPDX-License-Identifier: AGPL-3.0-only

"""A one-component image's codes: which occur, and each pixel's converted row.

convert_distinct_codes marked the codes with ``present[codes] = True`` and
scattered with ``numpy.take(table, codes, axis=0)``; both are numpy
expressions, so they are the reference here directly.
"""

import numpy
import pytest

from core_pdf_cythonized import code_presence, gather_uint8_codes


@pytest.mark.parametrize("seed", range(10))
def test_presence_and_largest_are_numpys(seed):
    rng = numpy.random.default_rng(seed)
    high = [2, 16, 256, 4096, 65536][seed % 5]
    codes = rng.integers(0, high, size=int(rng.integers(1, 5000)), dtype=numpy.uint16)
    present, largest = code_presence(codes)
    expected = numpy.zeros(65536, dtype=numpy.bool_)
    expected[codes] = True
    assert present.dtype == numpy.bool_
    assert numpy.array_equal(present, expected)
    assert largest == int(codes.max())


def test_no_codes_have_no_largest():
    present, largest = code_presence(numpy.zeros(0, dtype=numpy.uint16))
    assert largest == -1
    assert not present.any()


@pytest.mark.parametrize("columns", [1, 2, 3, 4, 5])
def test_the_gather_is_take(columns):
    rng = numpy.random.default_rng(columns)
    table = rng.integers(0, 256, size=(300, columns), dtype=numpy.uint8)
    codes = rng.integers(0, 300, size=2000, dtype=numpy.uint16)
    assert numpy.array_equal(gather_uint8_codes(table, codes), numpy.take(table, codes, axis=0))


def test_a_code_past_the_table_is_refused():
    table = numpy.zeros((4, 3), dtype=numpy.uint8)
    with pytest.raises(IndexError):
        gather_uint8_codes(table, numpy.array([1, 4], dtype=numpy.uint16))
