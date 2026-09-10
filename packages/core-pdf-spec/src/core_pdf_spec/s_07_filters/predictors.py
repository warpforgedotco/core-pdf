# SPDX-License-Identifier: AGPL-3.0-only
"""TIFF and PNG predictors (7.4.4.4): kernels plus the FilterParams-aware wrappers."""

from __future__ import annotations

import numpy

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError


class PredictorError(ValueError):
    """Invalid predictor parameters or data."""


class UnsupportedPngFilterError(PredictorError):
    """Unsupported PNG predictor row filter."""


def tiff_predict_8(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns
    if bytes_per_row <= 0:
        return b""
    complete = (len(data) // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    rows = numpy.frombuffer(data, dtype=numpy.uint8, count=complete).reshape(
        -1,
        columns,
        colors,
    )
    return numpy.cumsum(rows, axis=1, dtype=numpy.uint8).tobytes()


def tiff_predict_16(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns * 2
    if bytes_per_row <= 0:
        return b""
    complete = (len(data) // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    rows = numpy.frombuffer(data, dtype=">u2", count=complete // 2).reshape(
        -1,
        columns,
        colors,
    )
    return numpy.cumsum(rows, axis=1, dtype=numpy.uint16).astype(">u2").tobytes()


def tiff_predict_bits(data: bytes | memoryview, columns: int, colors: int, bits: int) -> bytes:
    """Undo TIFF differences modulo the sample depth, preserving row alignment."""
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
    binary = numpy.unpackbits(
        encoded.reshape(complete_rows, row_byte_length), axis=1, bitorder="big"
    )[:, : sample_count * bits]
    groups = binary.reshape(complete_rows, sample_count, bits)
    samples = numpy.sum(
        groups << numpy.arange(bits - 1, -1, -1, dtype=numpy.uint8), axis=2, dtype=numpy.uint8
    ).reshape(complete_rows, columns, colors)
    # uint8 accumulation wraps modulo 256, and 2**bits divides 256 for every
    # width here, so masking once at the end agrees with masking every step.
    accumulated = numpy.cumsum(samples, axis=1, dtype=numpy.uint8)
    decoded = accumulated & numpy.uint8((1 << bits) - 1)
    flat = decoded.reshape(complete_rows, sample_count)
    # packints_encode packs the whole array as one bitstream, so pad each row
    # out to a byte boundary first to keep rows byte-aligned as TIFF requires.
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
    out_pos = 0
    pos = 0
    previous: bytes | bytearray | memoryview | numpy.ndarray = bytearray(row_length)
    bpp = bytes_per_pixel
    rl = row_length
    while pos < n:
        if pos + 1 > n:
            break
        filter_type = data[pos]
        pos += 1
        if pos + rl > n:
            break
        if filter_type == 0:
            raw_row = memoryview(data)[pos : pos + rl]
            pos += rl
            out_view[out_pos : out_pos + rl] = numpy.frombuffer(raw_row, dtype=numpy.uint8)
            out_pos += rl
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
            previous_array = previous.tobytes() if isinstance(previous, numpy.ndarray) else previous
            first = min(bpp, n_row)
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + (previous_array[i] >> 1)) & 0xFF
            for i in range(bpp, n_row):
                row_bytes[i] = (
                    row_bytes[i] + ((row_bytes[i - bpp] + previous_array[i]) >> 1)
                ) & 0xFF
            row = row_bytes
        elif filter_type == 4:
            row_bytes = bytearray(data[pos : pos + rl])
            n_row = len(row_bytes)
            previous_array = previous.tobytes() if isinstance(previous, numpy.ndarray) else previous
            first = min(bpp, n_row)
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + previous_array[i]) & 0xFF
            for i in range(bpp, n_row):
                left, up, up_left = (
                    row_bytes[i - bpp],
                    previous_array[i],
                    previous_array[i - bpp],
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
        pos += rl
        out_view[out_pos : out_pos + rl] = row
        out_pos += rl
        previous = row
    return bytes(out[:out_pos])


SUPPORTED_PREDICTOR_BITS = frozenset({1, 2, 4, 8, 16})


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
    "PredictorError",
    "UnsupportedPngFilterError",
    "tiff_predict_8",
    "tiff_predict_16",
    "tiff_predict_bits",
    "tiff_predict",
    "png_predict",
    "SUPPORTED_PREDICTOR_BITS",
    "apply_tiff_predictor",
    "apply_png_predictor",
    "apply_predictor",
)
