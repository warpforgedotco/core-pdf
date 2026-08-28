from __future__ import annotations

import numpy

from core_pdf.impl.spec.s_07_filters.jbig2.bitmap_kernels import (
    compose_packed_bitmap_data,
)
from core_pdf.impl.spec.s_07_filters.jbig2.codec import jbig2_bitmap_to_pdf_image


def test_packed_compositor_handles_clipping_and_xor() -> None:
    rows = [b"\xff\x80\x00", b"\x81\x00\x00", b"\x00\x7f\x80"]
    packed = b"".join(rows)
    width = 17
    height = 5
    stride = (width + 7) // 8
    initial = bytearray(stride * height)

    compose_packed_bitmap_data(packed, len(rows), 17, -3, 1, width, height, stride, initial, 0)
    expected = bytearray(stride * height)
    for row_index, source in enumerate(rows):
        y = row_index + 1
        for column in range(17):
            if source[column >> 3] & (0x80 >> (column & 7)):
                x = column - 3
                if 0 <= x < width:
                    expected[y * stride + (x >> 3)] |= 0x80 >> (x & 7)
    assert initial == expected

    compose_packed_bitmap_data(packed, len(rows), 17, -3, 1, width, height, stride, initial, 2)
    assert initial == bytearray(stride * height)


def test_jbig2_polarity_conversion_preserves_exact_bytes() -> None:
    data = numpy.arange(8192, dtype=numpy.uint8).tobytes()

    assert jbig2_bitmap_to_pdf_image(data) == bytes(value ^ 0xFF for value in data)


def test_packed_compositor_accepts_numpy_row_matrix_with_clipping() -> None:
    rows = numpy.zeros((8, 16), dtype=numpy.uint8)
    rows[:, 0] = 0xFF
    width = 128
    height = 8
    stride = width // 8
    initial = bytearray(stride * height)

    compose_packed_bitmap_data(rows, 8, width, 0, 0, width, height, stride, initial, 0)

    numpy.testing.assert_array_equal(
        numpy.frombuffer(initial, dtype=numpy.uint8).reshape(height, stride),
        rows,
    )


def test_packed_compositor_uses_aligned_partial_byte_operations() -> None:
    width = 129
    height = 8
    region_x = 8
    image_width = 160
    image_stride = (image_width + 7) // 8
    rows = numpy.zeros((height, (width + 7) // 8), dtype=numpy.uint8)
    rows[:, :-1] = 0xA5
    rows[:, -1] = 0xE0
    initial = bytearray([0x5A] * (image_stride * height))
    expected = bytearray(initial)

    compose_packed_bitmap_data(
        rows,
        height,
        width,
        region_x,
        0,
        image_width,
        height,
        image_stride,
        initial,
        2,
    )
    for row_index in range(height):
        for column in range(width):
            if rows[row_index, column >> 3] & (0x80 >> (column & 7)):
                x = region_x + column
                expected[row_index * image_stride + (x >> 3)] ^= 0x80 >> (x & 7)

    assert initial == expected
