# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from typing import NoReturn

from core_pdf.impl.third_party.ccitt import (
    CcittParseError,
    CcittUnsupportedError,
    decode_ccitt_fax as decode_ccitt_impl,
)
from core_pdf.impl.third_party.jbig2 import (
    Jbig2ParseError,
    Jbig2UnsupportedError,
    assemble_embedded_jbig2,
    decode_embedded_jbig2,
    parse_jbig2_file,
)
from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    coerce_to_bytes,
    is_pdf_null,
    normalize_pdf_name,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.engine.spec.s_07_security.aes import AES
from core_pdf.impl.engine.spec.s_07_security.rc4 import CryptRC4
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_jpeg.api import decode_dct, decode_jpx as decode_jpx_impl
from core_jpeg.errors import JpegParseError, JpegUnsupportedError

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
    if not isinstance(parms, dict):
        if is_pdf_null(parms):
            raise PdfParseError("missing Crypt params")
        raise PdfParseError("invalid Crypt filter params")

    cfm_raw = lookup_dict_key(parms, "CFM")
    if is_pdf_null(cfm_raw):
        raise PdfParseError("invalid Crypt filter CFM")

    cfm = normalize_pdf_name(cfm_raw)
    if cfm is None:
        raise PdfParseError("invalid Crypt filter CFM")
    if cfm in {"None", "Identity"}:
        return data

    key = lookup_dict_key(parms, "Key")
    if is_pdf_null(key):
        key = lookup_dict_key(parms, "CryptKey")
    if is_pdf_null(key):
        raise PdfParseError("Crypt filter missing key")
    try:
        key_bytes = coerce_to_bytes(key)
    except TypeError as exc:
        raise PdfParseError("invalid Crypt filter key type") from exc

    if cfm in {"V2", "RC4"}:
        return CryptRC4(key_bytes).decrypt(data)
    if cfm in {"AESV2", "AESV3"}:
        if len(data) < 16:
            raise PdfParseError("invalid Crypt filter AES stream")
        initialization_vector = data[:16]
        ciphertext = data[16:]
        try:
            cipher = AES(key_bytes)
        except ValueError as exc:
            raise PdfParseError("invalid Crypt filter key") from exc
        try:
            return cipher.decrypt_cbc(initialization_vector, ciphertext, padding=True)
        except ValueError as exc:
            raise PdfParseError("invalid Crypt filter stream") from exc
    raise PdfUnsupportedError(f"Unsupported crypt filter method {cfm}")
