# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import os
from collections.abc import Callable
from typing import NoReturn

import imagecodecs
import numpy

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf.impl.spec.s_07_filters.jbig2.codec import (
    Jbig2ParseError,
    Jbig2UnsupportedError,
    assemble_embedded_jbig2,
    decode_embedded_jbig2,
    parse_jbig2_file,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import coerce_to_bytes, is_pdf_null


def raise_pdf_parse(exc: BaseException) -> NoReturn:
    raise FilterParseError(str(exc)) from exc


def raise_pdf_unsupported(exc: BaseException) -> NoReturn:
    raise FilterUnsupportedError(str(exc)) from exc


def internal_raise_codec_error(
    data: bytes | memoryview,
    exc: BaseException,
    *,
    check: Callable[[bytes | memoryview], object],
    name: str,
) -> NoReturn:
    if not data:
        raise FilterParseError(f"invalid {name} stream") from exc
    try:
        valid = bool(check(data))
    except Exception:
        valid = False
    if not valid:
        raise FilterParseError(f"invalid {name} stream") from exc
    raise FilterUnsupportedError(f"unsupported {name} stream") from exc


def internal_normalize_imagecodecs_array(
    decoded: object,
    *,
    name: str,
    allow_float: bool = False,
) -> numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]]:
    array = numpy.asarray(decoded)
    if array.ndim not in {2, 3}:
        raise FilterUnsupportedError(f"{name} decoder returned an unsupported shape")
    if array.dtype.kind not in ({"u", "i", "f"} if allow_float else {"u", "i"}):
        raise FilterUnsupportedError(f"{name} decoder returned an unsupported dtype")
    if array.ndim == 3 and array.shape[2] <= 0:
        raise FilterUnsupportedError(f"{name} decoder returned zero channels")
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


def decode_ccitt_fax_image(
    data: bytes | memoryview,
    parms: FilterParams,
    *,
    out: numpy.ndarray | None = None,
) -> numpy.ndarray[tuple[int, int], numpy.dtype[numpy.uint8]]:
    width = parms.columns if parms.has_columns else 1728
    height = parms.rows
    decoder_check = imagecodecs.ccittfax4_check if parms.k < 0 else imagecodecs.ccittfax3_check
    try:
        if parms.k < 0:
            decoded = imagecodecs.ccittfax4_decode(
                data,
                height=height,
                width=width,
                out=out,
            )
        else:
            t4options = 1 if parms.k > 0 else 0
            if parms.encoded_byte_align:
                t4options |= 4
            decoded = imagecodecs.ccittfax3_decode(
                data,
                height=height,
                width=width,
                t4options=t4options,
                out=out,
            )
    except Exception as exc:  # pragma: no cover - C-extension integration boundary
        internal_raise_codec_error(data, exc, check=decoder_check, name="CCITT")
    array = numpy.asarray(decoded)
    if array.ndim != 2 or array.shape[1] != width or array.dtype != numpy.uint8:
        raise FilterUnsupportedError("CCITT decoder returned an unsupported image")
    # imagecodecs uses zero for white and one for black; PDF samples use the
    # inverse polarity. The decoder contract is binary uint8, so invert in place
    # when an output buffer was supplied and avoid a second image allocation.
    numpy.bitwise_xor(array, 1, out=array)
    numpy.multiply(array, 255, out=array)
    return array


def decode_ccitt_fax(data: bytes, parms: object) -> bytes:
    params = parms if type(parms) is FilterParams else FilterParams.from_parms(parms)
    decoded = decode_ccitt_fax_image(data, params)
    return numpy.packbits(decoded != 0, axis=1, bitorder="big").tobytes()


def decode_jpeg_image(
    data: bytes | memoryview, *, out: numpy.ndarray | None = None
) -> numpy.ndarray:
    try:
        decoded = imagecodecs.jpeg_decode(data, out=out)
    except Exception as exc:  # pragma: no cover - C-extension integration boundary
        internal_raise_codec_error(data, exc, check=imagecodecs.jpeg_check, name="JPEG")
    return internal_normalize_imagecodecs_array(decoded, name="JPEG")


def decode_jpeg(data: bytes, parms: object) -> bytes:
    return decode_jpeg_image(data).tobytes()


def decode_jpx_image(
    data: bytes | memoryview, *, out: numpy.ndarray | None = None
) -> numpy.ndarray:
    try:
        decoded = imagecodecs.jpeg2k_decode(
            data,
            out=out,
            numthreads=internal_jpx_thread_count(),
        )
    except Exception as exc:  # pragma: no cover - C-extension integration boundary
        internal_raise_codec_error(data, exc, check=imagecodecs.jpeg2k_check, name="JPX")
    return internal_normalize_imagecodecs_array(decoded, name="JPX", allow_float=True)


def internal_jpx_thread_count() -> int:
    configured = os.environ.get("CORE_PDF_JPX_THREADS")
    if configured:
        try:
            return max(1, min(4, int(configured)))
        except ValueError:
            pass
    return max(1, min(4, os.cpu_count() or 1))


def decode_jpx(data: bytes, parms: object) -> bytes:
    return decode_jpx_image(data).tobytes()


def decode_jbig2(data: bytes, parms: object) -> bytes:
    if type(parms) is FilterParams:
        params = parms
    else:
        try:
            params = FilterParams.from_parms(parms)
        except ValueError as exc:
            raise FilterParseError("invalid JBIG2 parameters") from exc

    globals_obj = params.jbig2_globals

    if is_pdf_null(globals_obj):
        globals_data = b""
    else:
        try:
            globals_data = coerce_to_bytes(globals_obj)
        except TypeError as exc:
            raise FilterParseError("invalid JBIG2 globals") from exc

    if data is None:
        data = b""

    try:
        assembled = assemble_embedded_jbig2(globals_data, data)
        decoded = decode_embedded_jbig2(assembled)
        if decoded:
            return decoded
        parse_jbig2_file(assembled)
    except Jbig2UnsupportedError as exc:
        raise_pdf_unsupported(exc)
    except Jbig2ParseError as exc:
        raise_pdf_parse(exc)
    raise FilterUnsupportedError("JBIG2 stream could not be decoded")


def decode_crypt(data: bytes, parms: object) -> bytes:
    # Stream cryptography is applied by the document security handler before
    # ordinary filters.  /Crypt remains in the decode pipeline only to retain
    # the original dictionary and filter ordering.
    if is_pdf_null(parms):
        return data
    if not isinstance(parms, dict):
        raise FilterParseError("invalid Crypt filter params")
    return data
