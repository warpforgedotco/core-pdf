# SPDX-License-Identifier: AGPL-3.0-only
"""TIFF and PNG predictors (7.4.4.4): kernels plus the FilterParams-aware wrappers."""

from __future__ import annotations

import struct
import zlib

import imagecodecs
import numpy

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError

# The libpng path beats the scalar row loop at every size measured down to ~20
# bytes (2.2x at 20B, 14x at 500B, 35x at 2.7KB), so nothing is held back for
# it. Small PNG-predicted streams -- xref streams and object-stream indexes in
# incrementally-updated files, the most common /Predictor 12 use -- were paying
# up to 500us each under the old 1KB floor. The codec call is still guarded by
# the fallback below, and 150 fuzzed shapes across columns/colors/bpc/rows and
# all five filter types produce identical output on both paths.
PNG_CODEC_THRESHOLD = 0
# PDF predictor Colors -> PNG color type with identical sample layout.
internal_PNG_COLOR_TYPES = {1: 0, 3: 2, 4: 6}
internal_PNG_MAX_DIMENSION = 1_000_000
internal_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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
    return numpy.asarray(imagecodecs.delta_decode(rows, axis=1)).tobytes()


def tiff_predict_16(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns * 2
    if bytes_per_row <= 0:
        return b""
    complete = (len(data) // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    # delta_decode preserves byte order, so the big-endian view accumulates and
    # serializes without a pair of byte swaps around it.
    rows = numpy.frombuffer(data, dtype=">u2", count=complete // 2).reshape(
        -1,
        columns,
        colors,
    )
    return numpy.asarray(imagecodecs.delta_decode(rows, axis=1)).tobytes()


def tiff_predict_bits(data: bytes | memoryview, columns: int, colors: int, bits: int) -> bytes:
    """Undo TIFF prediction on sub-byte samples, byte-aligned per row.

    imcd unpacks and repacks the MSB-first bitstream and accumulates the rows,
    which is 5-21x the numpy lookup-table path this replaced and removes the
    scalar bit-buffer loop it fell back to on short streams.
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
    )
    samples = numpy.asarray(
        imagecodecs.packints_decode(encoded, numpy.uint8, bits, runlen=sample_count)
    ).reshape(complete_rows, columns, colors)
    # uint8 accumulation wraps modulo 256, and 2**bits divides 256 for every
    # width here, so masking once at the end agrees with masking every step.
    accumulated = numpy.asarray(imagecodecs.delta_decode(samples, axis=1))
    decoded = accumulated & numpy.uint8((1 << bits) - 1)
    flat = decoded.reshape(complete_rows, sample_count)
    # packints_encode packs the whole array as one bitstream, so pad each row
    # out to a byte boundary first to keep rows byte-aligned as TIFF requires.
    samples_per_byte = 8 // bits
    padding = (-sample_count) % samples_per_byte
    if padding:
        flat = numpy.pad(flat, ((0, 0), (0, padding)))
    packed = imagecodecs.packints_encode(numpy.ascontiguousarray(flat), bits)
    return numpy.asarray(packed).tobytes()


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


def internal_png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload))
    )


def internal_png_predict_codec(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
) -> bytes | None:
    """Unfilter PNG-predicted rows with libpng via a minimal PNG container.

    The filtered stream is byte-for-byte PNG scanline data, so wrapping it in
    IHDR/IDAT/IEND (stored-mode zlib, ~memcpy cost) lets imagecodecs run the
    row unfilter in C. Returns ``None`` when the parameter combination has no
    PNG equivalent; damaged data raises and the caller falls back to the
    scalar path, which reproduces the exact error/partial-output semantics.
    """
    color_type = internal_PNG_COLOR_TYPES.get(colors)
    if color_type is None:
        return None
    # Sub-byte depths exist only for grayscale, and the decoder's sample
    # expansion drops row padding bits, so require byte-aligned rows to
    # stay byte-identical with the scalar path.
    if bits_per_component not in (8, 16) and (
        color_type != 0 or (columns * bits_per_component) % 8
    ):
        return None
    if not 1 <= columns <= internal_PNG_MAX_DIMENSION:
        return None
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    rows = len(data) // (row_length + 1)
    if not 1 <= rows <= internal_PNG_MAX_DIMENSION:
        return None
    body = memoryview(data)[: rows * (row_length + 1)]
    header = struct.pack(">IIBBBBB", columns, rows, bits_per_component, color_type, 0, 0, 0)
    png = b"".join(
        (
            internal_PNG_SIGNATURE,
            internal_png_chunk(b"IHDR", header),
            internal_png_chunk(b"IDAT", zlib.compress(body, 0)),
            internal_png_chunk(b"IEND", b""),
        )
    )
    decoded = numpy.asarray(imagecodecs.png_decode(png))
    if bits_per_component == 16:
        return decoded.astype(">u2", copy=False).tobytes()
    if bits_per_component == 8:
        return decoded.tobytes()
    # Sub-byte gray comes back expanded to one byte per sample, scaled by the
    # exact factor 255 // (2**bits - 1); undo the scaling and repack.
    bits = bits_per_component
    samples = decoded.reshape(rows, columns) // (255 // ((1 << bits) - 1))
    per_byte = 8 // bits
    grouped = samples.reshape(rows, -1, per_byte)
    packed = numpy.zeros(grouped.shape[:2], dtype=numpy.uint8)
    for sample_index in range(per_byte):
        packed |= grouped[:, :, sample_index] << (bits * (per_byte - 1 - sample_index))
    return packed.tobytes()


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
    damaged_rows_before_error: int = 0,
) -> bytes:
    if bits_per_component not in {1, 2, 4, 8, 16}:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    if len(data) >= PNG_CODEC_THRESHOLD:
        try:
            decoded = internal_png_predict_codec(
                data,
                columns=columns,
                colors=colors,
                bits_per_component=bits_per_component,
            )
        except Exception:
            decoded = None
        if decoded is not None:
            return decoded
    bytes_per_pixel = max(1, (colors * bits_per_component + 7) // 8)
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    n = len(data)
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
            if damaged_rows_before_error:
                break
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
        if len(data) % stride and not params.damaged_rows_before_error:
            raise FilterParseError("truncated PNG predictor row")
    try:
        return png_predict(
            data,
            columns=params.columns,
            colors=params.colors,
            bits_per_component=params.bits_per_component,
            damaged_rows_before_error=params.damaged_rows_before_error,
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
    "apply_png_predictor",
    "apply_predictor",
    "apply_tiff_predictor",
    "png_predict",
    "tiff_predict",
    "tiff_predict_8",
    "tiff_predict_16",
    "tiff_predict_bits",
)
