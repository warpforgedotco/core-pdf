# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy

from core_predictors.errors import PredictorError
from core_predictors.samples import unpack_subbyte_rows


def internal_tiff_predict_words(
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


def tiff_predict_8(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    return internal_tiff_predict_words(data, columns, colors, numpy.dtype("u1"))


def tiff_predict_16(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    return internal_tiff_predict_words(data, columns, colors, numpy.dtype(">u2"))


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
        return tiff_predict_8(data, columns, colors)
    if bits_per_component == 16:
        return tiff_predict_16(data, columns, colors)
    if bits_per_component not in {1, 2, 4}:
        raise PredictorError(f"invalid TIFF predictor bits {bits_per_component}")
    return tiff_predict_bits(data, columns, colors, bits_per_component)


__all__ = (
    "tiff_predict_8",
    "tiff_predict_16",
    "tiff_predict_bits",
    "tiff_predict",
)
