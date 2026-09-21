# SPDX-License-Identifier: AGPL-3.0-only
"""Compose PDF image-filter parameters with the installed codec backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl._impl.graphics.decode_compat import FilterParams
from core_pdf.impl._impl.graphics.jbig2_recovery import RecoveryJBIG2PageDecoder
from core_pdf.impl._impl.model.pdf_values import is_pdf_null
from core_pdf.impl._impl.runtime import codec_backends
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.jbig2 import decode_jbig2 as decode_strict_jbig2


def internal_decode(
    call: Callable[..., numpy.ndarray[Any, Any]], *args: Any, **kwargs: Any
) -> numpy.ndarray[Any, Any]:
    try:
        return call(*args, **kwargs)
    except codec_backends.CodecParseError as exc:
        raise FilterParseError(str(exc)) from exc
    except codec_backends.CodecUnsupportedError as exc:
        raise FilterUnsupportedError(str(exc)) from exc


def decode_jpeg_image(
    data: bytes | memoryview, *, out: numpy.ndarray[Any, Any] | None = None
) -> numpy.ndarray[Any, Any]:
    return internal_decode(codec_backends.decode_jpeg_image, data, out=out)


def decode_jpx_image(
    data: bytes | memoryview,
    *,
    out: numpy.ndarray[Any, Any] | None = None,
    preserve_precision: bool = False,
) -> numpy.ndarray[Any, Any]:
    return internal_decode(
        codec_backends.decode_jpx_image, data, out=out, preserve_precision=preserve_precision
    )


def decode_jpeg(data: bytes, parms: object) -> bytes:
    return decode_jpeg_image(data).tobytes()


def decode_jpx(data: bytes, parms: object) -> bytes:
    return decode_jpx_image(data).tobytes()


def decode_ccitt_fax_image(
    data: bytes | memoryview, parms: FilterParams, *, out: numpy.ndarray[Any, Any] | None = None
) -> numpy.ndarray[Any, Any]:
    array = internal_decode(
        codec_backends.decode_ccitt_fax_image,
        data,
        width=parms.columns if parms.has_columns else 1728,
        height=parms.rows,
        group4=parms.k < 0,
        t4options=(1 if parms.k > 0 else 0) | (4 if parms.encoded_byte_align else 0),
        out=out,
    )
    if not parms.black_is_1:
        numpy.bitwise_xor(array, 1, out=array)
    numpy.multiply(array, 255, out=array)
    return array


def decode_ccitt_fax(data: bytes, parms: object) -> bytes:
    params = parms if type(parms) is FilterParams else FilterParams.from_parms(parms)
    return numpy.packbits(
        decode_ccitt_fax_image(data, params) != 0, axis=1, bitorder="big"
    ).tobytes()


def decode_crypt(data: bytes, parms: object) -> bytes:
    # Stream cryptography is applied by the document security handler before
    # ordinary filters.  /Crypt remains in the decode pipeline only to retain
    # the original dictionary and filter ordering.
    if is_pdf_null(parms):
        return data
    if not isinstance(parms, dict):
        raise FilterParseError("invalid Crypt filter params")
    return data


def decode_jbig2(data: bytes, parms: object) -> bytes:
    params = parms if isinstance(parms, FilterParams) else FilterParams.from_parms(parms)
    return decode_strict_jbig2(data, params, decoder_type=RecoveryJBIG2PageDecoder)
