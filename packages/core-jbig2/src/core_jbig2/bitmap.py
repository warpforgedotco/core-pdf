# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from operator import index
from typing import Any

import numpy


def uint8_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    *,
    count: int = -1,
    offset: int = 0,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    if isinstance(buffer, numpy.ndarray):
        buffer = numpy.ascontiguousarray(buffer, dtype=numpy.uint8).reshape(-1)
    return numpy.frombuffer(buffer, dtype=numpy.uint8, count=index(count), offset=index(offset))


def uint8_matrix_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    return uint8_view(buffer, count=rows * columns).reshape(rows, columns)


PACKED_COMPOSE_NUMPY_THRESHOLD = 64


def compose_packed_bitmap_scalar(
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


def compose_packed_bitmap_numpy(
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

    combine = numpy.bitwise_or if operator == 0 else numpy.bitwise_xor
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
            combine(destination[:, :-1], source_bytes[:, :-1], out=destination[:, :-1])
            combine(destination[:, -1], source_bytes[:, -1] & last_mask, out=destination[:, -1])
        else:
            combine(destination, source_bytes, out=destination)
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
    combine(target_bits, source_bits, out=target_bits)
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
    if row_count <= 0 or region_width <= 0:
        return
    row_byte_length = max(1, (region_width + 7) // 8)
    packed_bytes = (
        packed_bitmap.size if isinstance(packed_bitmap, numpy.ndarray) else len(packed_bitmap)
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
        compose_packed_bitmap_scalar(
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
    compose_packed_bitmap_numpy(
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


def invert_packed_bitmap(data: bytes | bytearray) -> bytes:
    if isinstance(data, bytearray):
        image = uint8_view(data)
        numpy.bitwise_xor(image, 0xFF, out=image)
        return bytes(data)
    if len(data) < 4096:
        return bytes(byte ^ 0xFF for byte in data)
    return numpy.bitwise_xor(uint8_view(data), 0xFF).tobytes()


__all__ = (
    "uint8_matrix_view",
    "PACKED_COMPOSE_NUMPY_THRESHOLD",
    "compose_packed_bitmap_data",
    "invert_packed_bitmap",
)
