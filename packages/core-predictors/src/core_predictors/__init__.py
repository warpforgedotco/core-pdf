# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from operator import index
from typing import Any

import numpy

__all__ = (
    "PredictorError",
    "UnsupportedPngFilterError",
    "png_predict",
    "tiff_predict",
    "tiff_predict_bits",
    "uint8_view",
    "unpack_subbyte_rows",
)


class PredictorError(ValueError):
    pass


class UnsupportedPngFilterError(PredictorError):
    pass


def unpack_subbyte_rows(
    packed_rows: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    samples_per_row: int,
    bits_per_component: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    binary = numpy.unpackbits(packed_rows, axis=1, bitorder="big")[
        :, : samples_per_row * bits_per_component
    ]
    groups = binary.reshape(len(packed_rows), samples_per_row, bits_per_component)
    shifts = numpy.arange(bits_per_component - 1, -1, -1, dtype=numpy.uint8)
    return numpy.sum(groups << shifts, axis=2, dtype=numpy.uint8)


def uint8_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    *,
    count: int = -1,
    offset: int = 0,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    count = index(count)
    offset = index(offset)
    if isinstance(buffer, numpy.ndarray):
        view = numpy.ascontiguousarray(buffer, dtype=numpy.uint8).reshape(-1)
        if offset < 0 or offset > view.size:
            raise ValueError("offset must be non-negative and no greater than buffer length")
        if count > view.size - offset:
            raise ValueError("buffer is smaller than requested size")
        if offset:
            view = view[offset:]
        if count >= 0:
            view = view[:count]
        return view
    return numpy.frombuffer(buffer, dtype=numpy.uint8, count=count, offset=offset)


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
) -> bytes:
    if bits_per_component not in {1, 2, 4, 8, 16}:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    bytes_per_pixel = max(1, (colors * bits_per_component + 7) // 8)
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    n = len(data)
    if n % (row_length + 1):
        raise PredictorError("truncated PNG predictor row")
    out = bytearray((n // (row_length + 1)) * row_length)
    out_view = numpy.frombuffer(out, dtype=numpy.uint8)
    previous = memoryview(bytes(row_length))
    bpp = bytes_per_pixel
    rl = row_length
    first = min(bpp, rl)
    for row_index, start in enumerate(range(0, n, rl + 1)):
        filter_type = data[start]
        pos = start + 1
        out_pos = row_index * rl
        if filter_type == 0:
            row: bytes | bytearray | memoryview | numpy.ndarray = memoryview(data)[pos : pos + rl]
        elif filter_type == 1:
            row_array = uint8_view(data, count=rl, offset=pos).copy()
            for offset in range(first):
                row_array[offset::bpp] = numpy.cumsum(row_array[offset::bpp], dtype=numpy.uint8)
            row = row_array
        elif filter_type == 2:
            row = uint8_view(data, count=rl, offset=pos) + uint8_view(previous)
        elif filter_type == 3:
            row_bytes = bytearray(data[pos : pos + rl])
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + (previous[i] >> 1)) & 0xFF
            for i in range(bpp, rl):
                row_bytes[i] = (row_bytes[i] + ((row_bytes[i - bpp] + previous[i]) >> 1)) & 0xFF
            row = row_bytes
        elif filter_type == 4:
            row_bytes = bytearray(data[pos : pos + rl])
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + previous[i]) & 0xFF
            for i in range(bpp, rl):
                left, up, up_left = (
                    row_bytes[i - bpp],
                    previous[i],
                    previous[i - bpp],
                )
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    row_bytes[i] = (row_bytes[i] + left) & 0xFF
                elif pb <= pc:
                    row_bytes[i] = (row_bytes[i] + up) & 0xFF
                else:
                    row_bytes[i] = (row_bytes[i] + up_left) & 0xFF
            row = row_bytes
        else:
            raise UnsupportedPngFilterError(f"Unsupported PNG predictor filter {filter_type}")
        out_view[out_pos : out_pos + rl] = row
        previous = memoryview(out)[out_pos : out_pos + rl]
    return bytes(out)


def tiff_predict_words(
    data: bytes | memoryview, columns: int, colors: int, dtype: numpy.dtype
) -> bytes:
    bytes_per_row = colors * columns * dtype.itemsize
    if bytes_per_row <= 0:
        return b""
    complete = (len(data) // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    rows = numpy.frombuffer(data, dtype=dtype, count=complete // dtype.itemsize).reshape(
        -1, columns, colors
    )
    return (
        numpy.cumsum(rows, axis=1, dtype=dtype.newbyteorder("="))
        .astype(dtype, copy=False)
        .tobytes()
    )


def tiff_predict_bits(data: bytes | memoryview, columns: int, colors: int, bits: int) -> bytes:
    sample_count = colors * columns
    row_byte_length = max(1, (sample_count * bits + 7) // 8)
    complete_rows = len(data) // row_byte_length
    if complete_rows == 0:
        return b""
    encoded = numpy.frombuffer(
        data,
        dtype=numpy.uint8,
        count=complete_rows * row_byte_length,
    )
    samples = unpack_subbyte_rows(
        encoded.reshape(complete_rows, row_byte_length), sample_count, bits
    ).reshape(complete_rows, columns, colors)
    accumulated = numpy.cumsum(samples, axis=1, dtype=numpy.uint8)
    decoded = accumulated & numpy.uint8((1 << bits) - 1)
    flat = decoded.reshape(complete_rows, sample_count)
    samples_per_byte = 8 // bits
    padding = (-sample_count) % samples_per_byte
    if padding:
        flat = numpy.pad(flat, ((0, 0), (0, padding)))
    expanded = (flat[:, :, None] >> numpy.arange(bits - 1, -1, -1, dtype=numpy.uint8)) & 1
    return numpy.packbits(expanded.reshape(complete_rows, -1), axis=1, bitorder="big").tobytes()


def tiff_predict(
    data: bytes | memoryview, *, columns: int, colors: int, bits_per_component: int
) -> bytes:
    if bits_per_component == 8:
        return tiff_predict_words(data, columns, colors, numpy.dtype("u1"))
    if bits_per_component == 16:
        return tiff_predict_words(data, columns, colors, numpy.dtype(">u2"))
    if bits_per_component not in {1, 2, 4}:
        raise PredictorError(f"invalid TIFF predictor bits {bits_per_component}")
    return tiff_predict_bits(data, columns, colors, bits_per_component)
