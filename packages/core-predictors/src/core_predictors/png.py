# SPDX-License-Identifier: AGPL-3.0-only
"""PNG (3rd edition) section 9 filter reconstruction over byte rows.

Each encoded row starts with a filter-type byte (0 None, 1 Sub, 2 Up,
3 Average, 4 Paeth) followed by ``row_length`` filtered bytes. The Paeth
predictor breaks ties in the prescribed order: left, above, upper-left.
"""

from __future__ import annotations

import numpy

from core_predictors.errors import PredictorError, UnsupportedPngFilterError


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
    for row_index, start in enumerate(range(0, n, rl + 1)):
        filter_type = data[start]
        pos = start + 1
        out_pos = row_index * rl
        if filter_type == 0:
            raw_row = memoryview(data)[pos : pos + rl]
            out_view[out_pos : out_pos + rl] = numpy.frombuffer(raw_row, dtype=numpy.uint8)
            previous = raw_row
            continue
        if filter_type == 1:
            row_array = numpy.frombuffer(data, dtype=numpy.uint8, count=rl, offset=pos).copy()
            for offset in range(min(bpp, len(row_array))):
                row_array[offset::bpp] = numpy.cumsum(
                    row_array[offset::bpp],
                    dtype=numpy.uint16,
                ).astype(numpy.uint8, copy=False)
            row: bytes | bytearray | memoryview | numpy.ndarray = row_array
        elif filter_type == 2:
            row_array = numpy.frombuffer(data, dtype=numpy.uint8, count=rl, offset=pos).copy()
            row_array[:] = (
                row_array.astype(numpy.uint16) + numpy.frombuffer(previous, dtype=numpy.uint8)
            ).astype(numpy.uint8)
            row = row_array
        elif filter_type == 3:
            row_bytes = bytearray(data[pos : pos + rl])
            n_row = len(row_bytes)
            first = min(bpp, n_row)
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + (previous[i] >> 1)) & 0xFF
            for i in range(bpp, n_row):
                row_bytes[i] = (row_bytes[i] + ((row_bytes[i - bpp] + previous[i]) >> 1)) & 0xFF
            row = row_bytes
        elif filter_type == 4:
            row_bytes = bytearray(data[pos : pos + rl])
            n_row = len(row_bytes)
            first = min(bpp, n_row)
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + previous[i]) & 0xFF
            for i in range(bpp, n_row):
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
