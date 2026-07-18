# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from typing import NoReturn

from core_ccitt import (
    CcittParseError,
    CcittUnsupportedError,
)
from core_ccitt import (
    decode_ccitt_fax as decode_ccitt_impl,
)
from core_jbig2 import (
    Jbig2ParseError,
    Jbig2UnsupportedError,
    assemble_embedded_jbig2,
    decode_embedded_jbig2,
    parse_jbig2_file,
)
from core_jpeg.api import decode_dct
from core_jpeg.api import decode_jpx as decode_jpx_impl
from core_jpeg.errors import JpegParseError, JpegUnsupportedError

from core_pdf.impl.engine.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.engine.spec.s_07_objects.coercion import coerce_to_bytes, is_pdf_null
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError


def raise_pdf_parse(exc: BaseException) -> NoReturn:
    raise PdfParseError(str(exc)) from exc


def raise_pdf_unsupported(exc: BaseException) -> NoReturn:
    raise PdfUnsupportedError(str(exc)) from exc


def decode_ccitt_fax(data: bytes, parms: object) -> bytes:
    params = parms if type(parms) is FilterParams else FilterParams.from_parms(parms)
    k = params.k
    try:
        return decode_ccitt_impl(
            data,
            columns=params.columns if params.has_columns else 1728,
            rows=params.rows,
            byte_align=params.encoded_byte_align,
            k=k,
        )
    except CcittUnsupportedError as exc:
        raise_pdf_unsupported(exc)
    except CcittParseError as exc:
        raise_pdf_parse(exc)


def decode_jpeg(data: bytes, parms: object) -> bytes:
    try:
        return decode_dct(data)
    except JpegUnsupportedError as exc:
        raise_pdf_unsupported(exc)
    except JpegParseError as exc:
        raise_pdf_parse(exc)
    except (ValueError, TypeError, struct.error, AssertionError) as exc:
        raise PdfParseError("invalid JPEG stream") from exc


def decode_jpx(data: bytes, parms: object) -> bytes:
    try:
        return decode_jpx_impl(
            data,
            apply_embedded_color=not jpx_parent_uses_explicit_colorspace(parms),
        )
    except JpegUnsupportedError as exc:
        raise_pdf_unsupported(exc)
    except JpegParseError as exc:
        raise_pdf_parse(exc)
    except (ValueError, TypeError, struct.error, AssertionError) as exc:
        raise PdfParseError("invalid JPX stream") from exc


def jpx_parent_uses_explicit_colorspace(parms: object) -> bool:
    return lookup_dict_key(parms, "ColorSpace") is not None


def decode_jbig2(data: bytes, parms: object) -> bytes:
    if type(parms) is FilterParams:
        params = parms
    else:
        try:
            params = FilterParams.from_parms(parms)
        except ValueError as exc:
            raise PdfParseError("invalid JBIG2 parameters") from exc

    globals_obj = params.jbig2_globals

    if is_pdf_null(globals_obj):
        globals_data = b""
    else:
        try:
            globals_data = coerce_to_bytes(globals_obj)
        except TypeError as exc:
            raise PdfParseError("invalid JBIG2 globals") from exc

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
    raise PdfUnsupportedError("JBIG2 stream could not be decoded")


def decode_crypt(data: bytes, parms: object) -> bytes:
    # Stream cryptography is applied by the document security handler before
    # ordinary filters.  /Crypt remains in the decode pipeline only to retain
    # the original dictionary and filter ordering.
    if is_pdf_null(parms):
        return data
    if not isinstance(parms, dict):
        raise PdfParseError("invalid Crypt filter params")
    return data
