# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any

lazy import numpy

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.errors import (
    FilterParseError,
    FilterUnsupportedError,
    PredictorError,
    UnsupportedPngFilterError,
)
lazy from core_pdf.impl.jbig2.bitmap import uint8_view
lazy from core_pdf.impl.spec.samples import unpack_subbyte_rows

SUPPORTED_PREDICTOR_BITS = frozenset({1, 2, 4, 8, 16})
SUBBYTE_PREDICTOR_BITS = frozenset({1, 2, 4})


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
    if n:
        rows = uint8_view(data, count=n).reshape(-1, row_length + 1)
        if (rows[:, 0] == 2).all():
            return numpy.cumsum(rows[:, 1:], axis=0, dtype=numpy.uint8).tobytes()
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


type SampleArray = numpy.ndarray[Any, Any]


def tiff_predict_words_with(
    data: bytes | memoryview,
    columns: int,
    colors: int,
    dtype: numpy.dtype[Any],
    accumulate: Callable[[SampleArray], SampleArray],
) -> bytes:
    """Undo TIFF predictor 2 on whole-byte samples; ``accumulate`` sums each row's samples
    along axis 1 of a (rows, columns, colors) array.
    """
    bytes_per_row = colors * columns * dtype.itemsize
    if bytes_per_row <= 0:
        return b""
    complete = (len(data) // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    rows = numpy.frombuffer(data, dtype=dtype, count=complete // dtype.itemsize).reshape(
        -1, columns, colors
    )
    return accumulate(rows).tobytes()


def tiff_predict_bits_with(
    data: bytes | memoryview,
    columns: int,
    colors: int,
    bits: int,
    unpack: Callable[[SampleArray, int, int], SampleArray],
    accumulate: Callable[[SampleArray], SampleArray],
    pack: Callable[[SampleArray, int], bytes],
) -> bytes:
    """Undo TIFF predictor 2 on 1-, 2- or 4-bit samples. ``unpack`` turns (rows, row bytes)
    into one byte per sample, ``accumulate`` sums along axis 1 of (rows, columns, colors), and
    ``pack`` packs (rows, byte-aligned samples) back to bits.
    """
    sample_count = colors * columns
    row_byte_length = max(1, (sample_count * bits + 7) // 8)
    complete_rows = len(data) // row_byte_length
    if complete_rows == 0:
        return b""
    encoded = numpy.frombuffer(
        data,
        dtype=numpy.uint8,
        count=complete_rows * row_byte_length,
    ).reshape(complete_rows, row_byte_length)
    samples = unpack(encoded, sample_count, bits).reshape(complete_rows, columns, colors)
    decoded = accumulate(samples) & numpy.uint8((1 << bits) - 1)
    flat = decoded.reshape(complete_rows, sample_count)
    padding = (-sample_count) % (8 // bits)
    if padding:
        flat = numpy.pad(flat, ((0, 0), (0, padding)))
    return pack(flat, bits)


def cumsum_rows(samples: SampleArray) -> SampleArray:
    dtype = samples.dtype
    return numpy.cumsum(samples, axis=1, dtype=dtype.newbyteorder("=")).astype(dtype, copy=False)


def pack_bit_samples(flat: SampleArray, bits: int) -> bytes:
    expanded = (flat[:, :, None] >> numpy.arange(bits - 1, -1, -1, dtype=numpy.uint8)) & 1
    return numpy.packbits(expanded.reshape(len(flat), -1), axis=1, bitorder="big").tobytes()


def tiff_predict_words(
    data: bytes | memoryview, columns: int, colors: int, dtype: numpy.dtype[Any]
) -> bytes:
    return tiff_predict_words_with(data, columns, colors, dtype, cumsum_rows)


def tiff_predict_bits(data: bytes | memoryview, columns: int, colors: int, bits: int) -> bytes:
    return tiff_predict_bits_with(
        data, columns, colors, bits, unpack_subbyte_rows, cumsum_rows, pack_bit_samples
    )


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


def png_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    return png_predict(
        data,
        columns=params.columns,
        colors=params.colors,
        bits_per_component=params.bits_per_component,
    )


def tiff_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    return tiff_predict(
        data,
        columns=params.columns,
        colors=params.colors,
        bits_per_component=params.bits_per_component,
    )


PredictFn = Callable[[bytes | memoryview, FilterParams], bytes]


def apply_tiff_predictor(
    data: bytes | memoryview, params: FilterParams, *, predict: PredictFn = tiff_predictor
) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = (params.columns * params.colors * params.bits_per_component + 7) // 8
        if row_length and len(data) % row_length:
            raise FilterParseError("truncated TIFF predictor row")
    try:
        return predict(data, params)
    except PredictorError as exc:
        raise FilterParseError(str(exc)) from exc


def apply_png_predictor(
    data: bytes | memoryview, params: FilterParams, *, predict: PredictFn = png_predictor
) -> bytes:
    if params.bits_per_component in SUPPORTED_PREDICTOR_BITS:
        if not data:
            return b""
        row_length = (params.columns * params.colors * params.bits_per_component + 7) // 8
        stride = row_length + 1
        if len(data) % stride and not params.damaged_rows_before_error:
            raise FilterParseError("truncated PNG predictor row")
    try:
        return predict(data, params)
    except UnsupportedPngFilterError as exc:
        raise FilterUnsupportedError(str(exc)) from exc
    except PredictorError as exc:
        raise FilterParseError(str(exc)) from exc


def apply_predictor(
    data: bytes | memoryview,
    parms: object,
    *,
    png: PredictFn = png_predictor,
    tiff: PredictFn = tiff_predictor,
) -> bytes:
    if parms is None or parms == {}:
        return bytes(data)
    params = parms if isinstance(parms, FilterParams) else FilterParams.from_parms(parms)
    predictor = params.predictor
    if predictor == 1:
        return bytes(data)
    if predictor == 2:
        return apply_tiff_predictor(data, params, predict=tiff)
    if predictor >= 10:
        return apply_png_predictor(data, params, predict=png)
    raise FilterParseError(f"invalid stream predictor {predictor}")


__all__ = (
    "SUBBYTE_PREDICTOR_BITS",
    "SUPPORTED_PREDICTOR_BITS",
    "png_predict",
    "tiff_predict",
    "tiff_predict_bits",
    "png_predictor",
    "tiff_predictor",
    "apply_tiff_predictor",
    "apply_png_predictor",
    "apply_predictor",
)
