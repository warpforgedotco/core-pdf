# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Callable
from typing import NoReturn

import imagecodecs
import numpy


class CodecParseError(ValueError):
    pass


class CodecUnsupportedError(ValueError):
    pass


def env_int(name: str, default: int) -> int:
    configured = os.environ.get(name)
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return default


def raise_codec_error(
    data: bytes | memoryview,
    exc: BaseException,
    *,
    check: Callable[[bytes | memoryview], object],
    name: str,
) -> NoReturn:
    if not data:
        raise CodecParseError(f"invalid {name} stream") from exc
    try:
        valid = bool(check(data))
    except Exception:
        valid = False
    if not valid:
        raise CodecParseError(f"invalid {name} stream") from exc
    raise CodecUnsupportedError(f"unsupported {name} stream") from exc


def normalize_imagecodecs_array(
    decoded: object,
    *,
    name: str,
    allow_float: bool = False,
    preserve_uint16: bool = False,
) -> numpy.ndarray:
    array = numpy.asarray(decoded)
    if array.ndim not in {2, 3}:
        raise CodecUnsupportedError(f"{name} decoder returned an unsupported shape")
    if array.dtype.kind not in ({"u", "i", "f"} if allow_float else {"u", "i"}):
        raise CodecUnsupportedError(f"{name} decoder returned an unsupported dtype")
    if array.ndim == 3 and array.shape[2] <= 0:
        raise CodecUnsupportedError(f"{name} decoder returned zero channels")
    if preserve_uint16 and array.dtype == numpy.uint16:
        return numpy.ascontiguousarray(array)
    if array.dtype != numpy.uint8:
        if array.dtype.kind == "f":
            array = numpy.clip(numpy.rint(array), 0, 255).astype(numpy.uint8, copy=False)
        else:
            item_bits = max(8, array.dtype.itemsize * 8)
            shift = max(0, item_bits - 8)
            clipped = numpy.clip(array, 0, None)
            array = (
                clipped.astype(numpy.uint64, copy=False) >> shift if shift else clipped
            ).astype(numpy.uint8, copy=False)
    return numpy.ascontiguousarray(array)


def decode_jpeg_image(
    data: bytes | memoryview, *, out: numpy.ndarray | None = None
) -> numpy.ndarray:
    try:
        decoded = imagecodecs.jpeg_decode(data, out=out)
    except Exception as exc:  # pragma: no cover
        raise_codec_error(data, exc, check=imagecodecs.jpeg_check, name="JPEG")
    return normalize_imagecodecs_array(decoded, name="JPEG")


def decode_jpx_image(
    data: bytes | memoryview,
    *,
    out: numpy.ndarray | None = None,
    preserve_precision: bool = False,
) -> numpy.ndarray:
    try:
        decoded = imagecodecs.jpeg2k_decode(
            data,
            out=out,
            numthreads=jpx_thread_count(),
        )
    except Exception as exc:  # pragma: no cover
        raise_codec_error(data, exc, check=imagecodecs.jpeg2k_check, name="JPX")
    return normalize_imagecodecs_array(
        decoded, name="JPX", allow_float=True, preserve_uint16=preserve_precision
    )


def jpx_thread_count() -> int:
    return min(4, env_int("CORE_PDF_JPX_THREADS", max(1, min(4, os.cpu_count() or 1))))


def decode_ccitt_fax_image(
    data: bytes | memoryview,
    *,
    width: int,
    height: int,
    group4: bool,
    t4options: int = 0,
    out: numpy.ndarray | None = None,
) -> numpy.ndarray[tuple[int, int], numpy.dtype[numpy.uint8]]:
    decoder_check = imagecodecs.ccittfax4_check if group4 else imagecodecs.ccittfax3_check
    try:
        if group4:
            decoded = imagecodecs.ccittfax4_decode(
                data,
                height=height,
                width=width,
                out=out,
            )
        else:
            decoded = imagecodecs.ccittfax3_decode(
                data,
                height=height,
                width=width,
                t4options=t4options,
                out=out,
            )
    except Exception as exc:  # pragma: no cover
        raise_codec_error(data, exc, check=decoder_check, name="CCITT")
    array = numpy.asarray(decoded)
    if array.ndim != 2 or array.shape[1] != width or array.dtype != numpy.uint8:
        raise CodecUnsupportedError("CCITT decoder returned an unsupported image")
    return array


PNG_COLOR_TYPES = {1: 0, 3: 2, 4: 6}
PNG_MAX_DIMENSION = 1_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload))
    )


def png_predict_codec(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
) -> bytes | None:
    color_type = PNG_COLOR_TYPES.get(colors)
    if color_type is None:
        return None
    if bits_per_component not in (8, 16) and (
        color_type != 0 or (columns * bits_per_component) % 8
    ):
        return None
    if not 1 <= columns <= PNG_MAX_DIMENSION:
        return None
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    rows = len(data) // (row_length + 1)
    if not 1 <= rows <= PNG_MAX_DIMENSION:
        return None
    body = memoryview(data)[: rows * (row_length + 1)]
    header = struct.pack(">IIBBBBB", columns, rows, bits_per_component, color_type, 0, 0, 0)
    png = b"".join(
        (
            PNG_SIGNATURE,
            png_chunk(b"IHDR", header),
            png_chunk(b"IDAT", zlib.compress(body, 0)),
            png_chunk(b"IEND", b""),
        )
    )
    decoded = numpy.asarray(imagecodecs.png_decode(png))
    if bits_per_component == 16:
        return decoded.astype(">u2", copy=False).tobytes()
    if bits_per_component == 8:
        return decoded.tobytes()
    bits = bits_per_component
    samples = decoded.reshape(rows, columns) // (255 // ((1 << bits) - 1))
    per_byte = 8 // bits
    grouped = samples.reshape(rows, -1, per_byte)
    packed = numpy.zeros(grouped.shape[:2], dtype=numpy.uint8)
    for sample_index in range(per_byte):
        packed |= grouped[:, :, sample_index] << (bits * (per_byte - 1 - sample_index))
    return packed.tobytes()


def tiff_predict_words(
    data: bytes | memoryview, columns: int, colors: int, dtype: str, sample_bytes: int
) -> bytes:
    bytes_per_row = colors * columns * sample_bytes
    if bytes_per_row <= 0:
        return b""
    complete = (len(data) // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    rows = numpy.frombuffer(data, dtype=dtype, count=complete // sample_bytes).reshape(
        -1,
        columns,
        colors,
    )
    return numpy.asarray(imagecodecs.delta_decode(rows, axis=1)).tobytes()


def tiff_predict_codec(
    data: bytes | memoryview, *, columns: int, colors: int, bits_per_component: int
) -> bytes | None:
    # Mirrors png_predict_codec: returns None when imagecodecs cannot take
    # this shape, so the caller has one decision to make rather than a
    # second copy of the bit-width table plus a bare except.
    try:
        if bits_per_component == 8:
            return tiff_predict_words(data, columns, colors, "u1", 1)
        if bits_per_component == 16:
            return tiff_predict_words(data, columns, colors, ">u2", 2)
        if bits_per_component in {1, 2, 4}:
            return tiff_predict_bits(data, columns, colors, bits_per_component)
    except Exception:
        return None
    return None


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
    samples = numpy.asarray(
        imagecodecs.packints_decode(encoded, numpy.uint8, bits, runlen=sample_count)
    ).reshape(complete_rows, columns, colors)
    accumulated = numpy.asarray(imagecodecs.delta_decode(samples, axis=1))
    decoded = accumulated & numpy.uint8((1 << bits) - 1)
    flat = decoded.reshape(complete_rows, sample_count)
    samples_per_byte = 8 // bits
    padding = (-sample_count) % samples_per_byte
    if padding:
        flat = numpy.pad(flat, ((0, 0), (0, padding)))
    packed = imagecodecs.packints_encode(numpy.ascontiguousarray(flat), bits)
    return numpy.asarray(packed).tobytes()
