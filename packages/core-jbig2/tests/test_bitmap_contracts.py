# SPDX-License-Identifier: AGPL-3.0-only
"""JBIG2 packed composition agrees with a byte-independent pixel oracle."""

from typing import Any, cast

import numpy
import pytest

from core_jbig2 import bitmap as kernels


def compose_pixels(rows, width, x, y, image_width, image_height, stride, initial, operator):
    result = bytearray(initial)
    for row_index, row in enumerate(rows):
        for column in range(width):
            target_x, target_y = x + column, y + row_index
            if not (0 <= target_x < image_width and 0 <= target_y < image_height):
                continue
            source_bit = (row[column // 8] >> (7 - column % 8)) & 1
            offset, bit = target_y * stride + target_x // 8, 7 - target_x % 8
            old_bit = (result[offset] >> bit) & 1
            new_bit = old_bit | source_bit if operator == 0 else old_bit ^ source_bit
            result[offset] = (result[offset] & ~(1 << bit)) | (new_bit << bit)
    return result


@pytest.mark.parametrize("operator", [0, 2])
@pytest.mark.parametrize("width", [1, 7, 8, 9, 16, 31, 64])
@pytest.mark.parametrize("position", [(0, 0), (8, 1), (3, -1), (-3, 2), (-8, -2), (70, 0), (0, 6)])
def test_scalar_numpy_and_dispatch_match_pixel_composition(operator, width, position):
    stride, image_width, image_height = 10, 67, 5
    initial = bytes((index * 37 + 91) % 256 for index in range(stride * image_height))
    row_bytes = (width + 7) // 8
    packed = bytes((index * 53 + 167) % 256 for index in range(row_bytes * 4))
    rows = [packed[start : start + row_bytes] for start in range(0, len(packed), row_bytes)]
    x, y = position
    expected = compose_pixels(
        rows, width, x, y, image_width, image_height, stride, initial, operator
    )
    scalar, bulk, dispatched = (bytearray(initial) for _ in range(3))
    kernels.internal_compose_packed_bitmap_scalar(
        rows, width, x, y, image_width, image_height, stride, scalar, operator
    )
    kernels.internal_compose_packed_bitmap_numpy(
        packed, 4, row_bytes, width, x, y, image_width, image_height, stride, bulk, operator
    )
    kernels.compose_packed_bitmap_data(
        packed, 4, width, x, y, image_width, image_height, stride, dispatched, operator
    )
    assert scalar == expected
    assert bulk == expected
    assert dispatched == expected


@pytest.mark.parametrize("width", [9, 64])
@pytest.mark.parametrize(
    "representation", ["bytes", "bytearray", "memoryview", "array", "strided", "uint16"]
)
def test_composition_normalizes_buffers_and_ignores_incomplete_rows(width, representation):
    row_bytes = (width + 7) // 8
    data = bytes(range(1, row_bytes * 2 + 2))
    match representation:
        case "bytes":
            source = data
        case "bytearray":
            source = bytearray(data)
        case "memoryview":
            source = memoryview(data)
        case "array":
            source = numpy.frombuffer(data, dtype=numpy.uint8)
        case "strided":
            source = numpy.repeat(numpy.frombuffer(data, dtype=numpy.uint8), 2)[::2]
        case "uint16":
            source = numpy.frombuffer(data, dtype=numpy.uint8).astype(numpy.uint16)
    output = bytearray(row_bytes * 4)
    kernels.compose_packed_bitmap_data(source, 4, width, 0, 0, width, 4, row_bytes, output, 0)
    rows = [data[:row_bytes], data[row_bytes : row_bytes * 2]]
    assert output == compose_pixels(rows, width, 0, 0, width, 4, row_bytes, bytes(len(output)), 0)


@pytest.mark.parametrize(("count", "width", "data"), [(0, 8, b"x"), (1, 0, b"x"), (1, 16, b"x")])
def test_empty_or_incomplete_bitmap_preserves_destination(count, width, data):
    output = bytearray(b"ab")
    kernels.compose_packed_bitmap_data(data, count, width, 0, 0, 8, 2, 1, output, 0)
    assert output == b"ab"


@pytest.mark.parametrize(("offset", "count"), [(-1, -1), (4, -1), (0, 4), (2, 2)])
def test_array_view_rejects_out_of_bounds_ranges(offset, count):
    with pytest.raises(ValueError):
        kernels.internal_uint8_view(
            numpy.array([1, 2, 3], dtype=numpy.uint8), offset=offset, count=count
        )


def test_mutable_matrix_view_shares_storage_and_honors_offset_and_count():
    source = numpy.array([[1, 2], [3, 4]], dtype=numpy.uint8)
    view = kernels.internal_uint8_view(source, offset=1, count=2)
    view[:] = [8, 9]
    assert source.tolist() == [[1, 8], [9, 4]]
    matrix = kernels.uint8_matrix_view(source, 2, 2)
    matrix[0, 0] = 7
    assert source[0, 0] == 7
    with pytest.raises(TypeError):
        kernels.internal_uint8_view(source, count=cast(Any, 1.5))


@pytest.mark.parametrize("layout", ["contiguous", "strided", "converted"])
def test_unbounded_array_view_preserves_logical_element_order(layout):
    source = numpy.array([[1, 2], [3, 4]], dtype=numpy.uint8)
    if layout == "strided":
        source = source.T
    elif layout == "converted":
        source = source.astype(numpy.uint16)
    expected = [int(value) for value in source.flat]
    view = kernels.internal_uint8_view(source)
    assert view.tolist() == expected
    assert view.dtype == numpy.uint8
    assert view.flags.c_contiguous
    assert kernels.internal_uint8_view(source, offset=source.size).size == 0
