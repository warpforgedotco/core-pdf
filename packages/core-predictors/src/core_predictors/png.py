# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy

from core_predictors.errors import PredictorError, UnsupportedPngFilterError
from core_predictors.samples import uint8_view


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


__all__ = ("png_predict",)
