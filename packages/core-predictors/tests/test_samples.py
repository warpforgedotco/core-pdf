# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, cast

import numpy
import pytest

from core_predictors.samples import uint8_view


@pytest.mark.parametrize(("offset", "count"), [(-1, -1), (4, -1), (0, 4), (2, 2)])
def test_array_view_rejects_out_of_bounds_ranges(offset, count):
    with pytest.raises(ValueError):
        uint8_view(numpy.array([1, 2, 3], dtype=numpy.uint8), offset=offset, count=count)


def test_mutable_matrix_view_shares_storage_and_honors_offset_and_count():
    source = numpy.array([[1, 2], [3, 4]], dtype=numpy.uint8)
    view = uint8_view(source, offset=1, count=2)
    view[:] = [8, 9]
    assert source.tolist() == [[1, 8], [9, 4]]
    with pytest.raises(TypeError):
        uint8_view(source, count=cast(Any, 1.5))


@pytest.mark.parametrize("layout", ["contiguous", "strided", "converted"])
def test_unbounded_array_view_preserves_logical_element_order(layout):
    source = numpy.array([[1, 2], [3, 4]], dtype=numpy.uint8)
    if layout == "strided":
        source = source.T
    elif layout == "converted":
        source = source.astype(numpy.uint16)
    expected = [int(value) for value in source.flat]
    view = uint8_view(source)
    assert view.tolist() == expected
    assert view.dtype == numpy.uint8
    assert view.flags.c_contiguous
    assert uint8_view(source, offset=source.size).size == 0
