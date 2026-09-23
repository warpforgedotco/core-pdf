# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from operator import index
from typing import Any

import numpy

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import (
    FilterParseError,
    FilterUnsupportedError,
    PredictorError,
    UnsupportedPngFilterError,
)
from core_pdf_spec.samples import unpack_subbyte_rows

# The sample depths the PNG and TIFF predictors accept (ISO 32000-2 7.4.4.4).
# Callers gate on these before framing rows; the kernels raise PredictorError
# for anything else.
SUPPORTED_PREDICTOR_BITS = frozenset({1, 2, 4, 8, 16})
SUBBYTE_PREDICTOR_BITS = frozenset({1, 2, 4})


def uint8_view(
    buffer: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
    *,
    count: int = -1,
    offset: int = 0,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
    if isinstance(buffer, numpy.ndarray):
        buffer = numpy.ascontiguousarray(buffer, dtype=numpy.uint8).reshape(-1)
    return numpy.frombuffer(buffer, dtype=numpy.uint8, count=index(count), offset=index(offset))


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
) -> bytes:
    if bits_per_component not in SUPPORTED_PREDICTOR_BITS:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    bpp = max(1, (colors * bits_per_component + 7) // 8)
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    n = len(data)
    if n % (row_length + 1):
        raise PredictorError("truncated PNG predictor row")
    out = bytearray((n // (row_length + 1)) * row_length)
    out_view = numpy.frombuffer(out, dtype=numpy.uint8)
    previous = memoryview(bytes(row_length))
    first = min(bpp, row_length)
    for row_index, start in enumerate(range(0, n, row_length + 1)):
        filter_type = data[start]
        pos = start + 1
        out_pos = row_index * row_length
        if filter_type == 0:
            row: bytearray | memoryview | numpy.ndarray = memoryview(data)[pos : pos + row_length]
        elif filter_type == 1:
            row_array = uint8_view(data, count=row_length, offset=pos).copy()
            for offset in range(first):
                row_array[offset::bpp] = numpy.cumsum(row_array[offset::bpp], dtype=numpy.uint8)
            row = row_array
        elif filter_type == 2:
            row = uint8_view(data, count=row_length, offset=pos) + uint8_view(previous)
        elif filter_type == 3:
            row_bytes = bytearray(data[pos : pos + row_length])
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + (previous[i] >> 1)) & 0xFF
            for i in range(bpp, row_length):
                row_bytes[i] = (row_bytes[i] + ((row_bytes[i - bpp] + previous[i]) >> 1)) & 0xFF
            row = row_bytes
        elif filter_type == 4:
            row_bytes = bytearray(data[pos : pos + row_length])
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + previous[i]) & 0xFF
            for i in range(bpp, row_length):
                left, up, up_left = (
                    row_bytes[i - bpp],
                    previous[i],
                    previous[i - bpp],
                )
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    nearest = left
                elif pb <= pc:
                    nearest = up
                else:
                    nearest = up_left
                row_bytes[i] = (row_bytes[i] + nearest) & 0xFF
            row = row_bytes
        else:
            raise UnsupportedPngFilterError(f"Unsupported PNG predictor filter {filter_type}")
        out_view[out_pos : out_pos + row_length] = row
        previous = memoryview(out)[out_pos : out_pos + row_length]
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
    if bits_per_component not in SUBBYTE_PREDICTOR_BITS:
        raise PredictorError(f"invalid TIFF predictor bits {bits_per_component}")
    return tiff_predict_bits(data, columns, colors, bits_per_component)


def apply_tiff_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = (params.columns * params.colors * params.bits_per_component + 7) // 8
        if row_length and len(data) % row_length:
            raise FilterParseError("truncated TIFF predictor row")
    try:
        return tiff_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
        )
    except PredictorError as exc:
        raise FilterParseError(str(exc)) from exc


def apply_png_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = (params.columns * params.colors * params.bits_per_component + 7) // 8
        stride = row_length + 1
        if len(data) % stride:
            raise FilterParseError("truncated PNG predictor row")
    try:
        return png_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
        )
    except UnsupportedPngFilterError as exc:
        raise FilterUnsupportedError(str(exc)) from exc
    except PredictorError as exc:
        raise FilterParseError(str(exc)) from exc


def apply_predictor(data: bytes | memoryview, parms: object) -> bytes:
    if parms is None or parms == {}:
        return bytes(data)
    params = parms if type(parms) is FilterParams else FilterParams.from_parms(parms)
    predictor = params.predictor
    if predictor == 1:
        return bytes(data)
    if predictor == 2:
        return apply_tiff_predictor(data, params)
    if predictor >= 10:
        return apply_png_predictor(data, params)
    raise FilterParseError(f"invalid stream predictor {predictor}")


__all__ = (
    "SUBBYTE_PREDICTOR_BITS",
    "SUPPORTED_PREDICTOR_BITS",
    "png_predict",
    "tiff_predict",
    "tiff_predict_bits",
    "apply_tiff_predictor",
    "apply_png_predictor",
    "apply_predictor",
)
