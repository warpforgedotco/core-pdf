"""Packed bitmap composition kernels for JBIG2 region decoding."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl._impl.runtime.array_views import uint8_view


def uint8_matrix_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    """Return a validated mutable/read-only matrix view over packed rows."""
    return uint8_view(buffer, count=rows * columns).reshape(rows, columns)


# NumPy's packed-bit path amortizes its view setup above a small region.  The
# aligned case is already faster at 64 pixels, while scalar composition remains
# cheaper for tiny regions and avoids unpacking temporary bit arrays.
PACKED_COMPOSE_NUMPY_THRESHOLD = 64


def internal_compose_packed_bitmap_scalar(
    rows: list[bytes],
    region_width: int,
    region_x: int,
    region_y: int,
    image_width: int,
    image_height: int,
    image_stride: int,
    image_data: bytearray,
    operator: int,
) -> None:
    for row_index, source in enumerate(rows):
        y = region_y + row_index
        if y < 0 or y >= image_height:
            continue
        for col in range(region_width):
            if not source[col >> 3] & (0x80 >> (col & 7)):
                continue
            x = region_x + col
            if x < 0 or x >= image_width:
                continue
            index = y * image_stride + (x >> 3)
            mask = 0x80 >> (x & 7)
            if operator == 0:
                image_data[index] |= mask
            else:
                image_data[index] ^= mask


def internal_compose_packed_bitmap_numpy(
    packed_bitmap: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    row_count: int,
    row_byte_length: int,
    region_width: int,
    region_x: int,
    region_y: int,
    image_width: int,
    image_height: int,
    image_stride: int,
    image_data: bytearray,
    operator: int,
) -> None:
    first_row = max(0, -region_y)
    last_row = min(row_count, image_height - region_y)
    first_col = max(0, -region_x)
    last_col = min(region_width, image_width - region_x)
    if first_row >= last_row or first_col >= last_col:
        return

    source = uint8_view(
        packed_bitmap,
        count=(last_row - first_row) * row_byte_length,
        offset=first_row * row_byte_length,
    ).reshape(last_row - first_row, row_byte_length)

    bit_count = last_col - first_col
    destination_x = region_x + first_col
    if first_col & 7 == 0 and destination_x & 7 == 0:
        source_start_byte = first_col >> 3
        destination_start_byte = destination_x >> 3
        byte_count = (bit_count + 7) >> 3
        source_bytes = source[:, source_start_byte : source_start_byte + byte_count]
        target = uint8_matrix_view(image_data, image_height, image_stride)
        destination = target[
            region_y + first_row : region_y + last_row,
            destination_start_byte : destination_start_byte + byte_count,
        ]
        if bit_count & 7:
            last_mask = numpy.uint8((0xFF << (8 - (bit_count & 7))) & 0xFF)
            if operator == 0:
                destination[:, :-1] |= source_bytes[:, :-1]
                destination[:, -1] |= source_bytes[:, -1] & last_mask
            else:
                destination[:, :-1] ^= source_bytes[:, :-1]
                destination[:, -1] ^= source_bytes[:, -1] & last_mask
        elif operator == 0:
            destination |= source_bytes
        else:
            destination ^= source_bytes
        return

    source_bits = numpy.unpackbits(source, axis=1, bitorder="big")
    source_bits = source_bits[:, first_col:last_col]

    start_byte = destination_x >> 3
    end_byte = (destination_x + source_bits.shape[1] + 7) >> 3
    target = uint8_matrix_view(image_data, image_height, image_stride)
    destination = target[
        region_y + first_row : region_y + last_row,
        start_byte:end_byte,
    ]
    destination_bits = numpy.unpackbits(destination, axis=1, bitorder="big")
    bit_offset = destination_x & 7
    target_bits = destination_bits[:, bit_offset : bit_offset + source_bits.shape[1]]
    if operator == 0:
        target_bits |= source_bits
    else:
        target_bits ^= source_bits
    destination[:] = numpy.packbits(destination_bits, axis=1, bitorder="big")


def compose_packed_bitmap_data(
    packed_bitmap: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    row_count: int,
    region_width: int,
    region_x: int,
    region_y: int,
    image_width: int,
    image_height: int,
    image_stride: int,
    image_data: bytearray,
    operator: int,
) -> None:
    """Composite contiguous packed one-bit rows into an image buffer."""
    if row_count <= 0 or region_width <= 0:
        return
    row_byte_length = max(1, (region_width + 7) // 8)
    packed_bytes = (
        packed_bitmap.nbytes if isinstance(packed_bitmap, numpy.ndarray) else len(packed_bitmap)
    )
    available_rows = min(row_count, packed_bytes // row_byte_length)
    if available_rows <= 0:
        return
    if region_width * available_rows < PACKED_COMPOSE_NUMPY_THRESHOLD:
        if isinstance(packed_bitmap, numpy.ndarray):
            packed_rows = uint8_matrix_view(packed_bitmap, available_rows, row_byte_length)
            rows = [bytes(row) for row in packed_rows[:available_rows]]
        else:
            rows = [
                bytes(packed_bitmap[row * row_byte_length : (row + 1) * row_byte_length])
                for row in range(available_rows)
            ]
        internal_compose_packed_bitmap_scalar(
            rows,
            region_width,
            region_x,
            region_y,
            image_width,
            image_height,
            image_stride,
            image_data,
            operator,
        )
        return
    internal_compose_packed_bitmap_numpy(
        packed_bitmap,
        available_rows,
        row_byte_length,
        region_width,
        region_x,
        region_y,
        image_width,
        image_height,
        image_stride,
        image_data,
        operator,
    )


__all__ = ("compose_packed_bitmap_data", "uint8_matrix_view", "uint8_view")
