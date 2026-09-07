# SPDX-License-Identifier: AGPL-3.0-only
"""Normative JBIG2 stream decoding and PDF global segment handling."""

from __future__ import annotations

from typing import NoReturn

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


def decode_jbig2(data: bytes, parms: object) -> bytes:
    if isinstance(parms, FilterParams):
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
        # ISO 32000-1 Table 12 types JBIG2Globals as a *stream* -- "Global
        # segments shall be placed in this stream" -- so once DecodeParms is
        # resolved this is a stream object, not bytes. s_07_filters sits below
        # s_07_syntax in the layer contract and so cannot name PdfStream;
        # unwrap the decoded bytes structurally instead.
        stream_data = getattr(globals_obj, "data", None)
        if isinstance(stream_data, (bytes, bytearray, memoryview)):
            globals_obj = stream_data
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
