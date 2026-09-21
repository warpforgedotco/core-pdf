# SPDX-License-Identifier: AGPL-3.0-only
"""JBIG2Decode over DecodeParms (ISO 32000-1 7.4.7) using the T.88 kernel.

The decoder lives in ``core_jbig2``. This module resolves ``JBIG2Globals``,
maps kernel errors onto filter errors, and converts T.88 polarity (1 = black)
into the ImageMask/DeviceGray convention PDF requires (1 = white).
"""

from __future__ import annotations

from core_jbig2.bitmap import invert_packed_bitmap
from core_jbig2.codec import (
    JBIG2PageDecoder,
    Jbig2ParseError,
    Jbig2UnsupportedError,
    parse_embedded_segments,
)
from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_syntax_primitives.coercion import coerce_to_bytes, is_pdf_null


def decode_jbig2(
    data: bytes,
    parms: object,
    *,
    decoder_type: type[JBIG2PageDecoder] = JBIG2PageDecoder,
) -> bytes:
    """Decode a PDF JBIG2 stream with strict defaults and a fresh page decoder."""
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

    try:
        decoder = decoder_type()
        for segment in parse_embedded_segments(globals_data + data):
            decoder.decode_segment(segment)
        # ISO 32000-1 7.4.7: the filter delivers 1 bits as black pixels, so
        # the decoded bitmap is inverted for the DeviceGray/ImageMask sense.
        return invert_packed_bitmap(decoder.finish())
    except Jbig2UnsupportedError as exc:
        raise FilterUnsupportedError(str(exc)) from exc
    except Jbig2ParseError as exc:
        raise FilterParseError(str(exc)) from exc


__all__ = ("decode_jbig2",)
