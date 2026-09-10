# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility decoding for malformed and mislabeled PDF stream data."""

from __future__ import annotations

import binascii
import zlib

import imagecodecs

import core_pdf_spec.s_07_filters.codecs as strict
from core_pdf.impl._impl.graphics.decode_compat import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_syntax_primitives.content_operators import PDF_CONTENT_OPERATOR_BYTES
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import (
    DELIMITERS,
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
)

ASCII_HEX_DIGITS = b"0123456789ABCDEFabcdef"
ASCII_HEX_INVALID_BYTES = bytes(byte for byte in range(256) if byte not in ASCII_HEX_DIGITS)
MIN_TRUNCATED_RAW_FLATE_BYTES = 8


def apply_ascii_hex(data: bytes, parms: object) -> bytes:
    terminator = data.find(b">")
    if terminator >= 0:
        data = data[:terminator]
    filtered = data.translate(None, ASCII_HEX_INVALID_BYTES)
    if len(filtered) & 1:
        filtered += b"0"
    return binascii.unhexlify(filtered)


def apply_run_length(data: bytes, parms: object) -> bytes:
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        length = data[i]
        i += 1
        if length == 128:
            break
        if length < 128:
            run = length + 1
            if i + run > n:
                out.extend(data[i:n])
                break
            out.extend(data[i : i + run])
            i += run
            continue
        run = 257 - length
        if i >= n:
            break
        out.extend(data[i : i + 1] * run)
        i += 1
    return bytes(out)


def apply_ascii85(data: bytes | memoryview, parms: object) -> bytes:
    clean = bytes(data).lstrip(WHITESPACE)
    if clean.startswith(b"<~"):
        clean = clean[2:]
    if b"~>" not in clean:
        clean += b"~>"
    return strict.apply_ascii85(clean, parms)


def apply_lzw(data: bytes | memoryview, parms: object) -> bytes:
    params = parms if isinstance(parms, FilterParams) else FilterParams.from_parms(parms)
    if params.early_change == 1 and imagecodecs.LZW.available:
        try:
            return bytes(imagecodecs.lzw_decode(data))
        except Exception as exc:
            raise ValueError("invalid LZW stream") from exc
    try:
        return strict.apply_lzw(data, params)
    except strict.IncompleteLzwError as exc:
        return exc.decoded


def apply_flate(data: bytes, parms: object) -> bytes:
    if not data:
        return b""
    candidates: tuple[int, ...]
    if len(data) >= 2:
        cmf = data[0]
        flg = data[1]
        if cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0:
            candidates = (zlib.MAX_WBITS, zlib.MAX_WBITS | 32, -15)
        elif cmf == 0x1F and flg == 0x8B:
            candidates = (zlib.MAX_WBITS | 32, zlib.MAX_WBITS, -15)
        else:
            candidates = (-15, zlib.MAX_WBITS | 32, zlib.MAX_WBITS)
    else:
        candidates = (-15, zlib.MAX_WBITS | 32, zlib.MAX_WBITS)

    tried_default = candidates[0] == zlib.MAX_WBITS
    if tried_default:
        try:
            return zlib.decompress(data, zlib.MAX_WBITS)
        except zlib.error:
            pass
        try:
            return bytes(imagecodecs.zlib_decode(data))
        except Exception:
            # imagecodecs raises its own error types; any failure here just means
            # this decoder cannot handle the stream, so try the wbits candidates.
            pass

    for wbits in candidates:
        if not (tried_default and wbits == zlib.MAX_WBITS):
            try:
                return zlib.decompress(data, wbits)
            except zlib.error:
                pass
        recovered = recover_flate(data, wbits)
        if recovered is not None:
            return recovered

    if looks_like_pdf_content_stream(data):
        return bytes(data)
    raise FilterParseError("invalid FlateDecode stream")


def recover_flate(data: bytes, wbits: int = zlib.MAX_WBITS) -> bytes | None:
    try:
        decoder = zlib.decompressobj(wbits)
        decoded = decoder.decompress(data) + decoder.flush()
        if wbits < 0 and not decoder.eof and len(data) < MIN_TRUNCATED_RAW_FLATE_BYTES:
            return None
        return decoded
    except zlib.error:
        if wbits > 0 and len(data) > 6:
            cmf = data[0]
            flg = data[1]
            if cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0:
                try:
                    return zlib.decompress(data[2:-4], -15)
                except zlib.error:
                    return None
        return None


def looks_like_pdf_content_stream(data: bytes | memoryview) -> bool:
    data_len = len(data)
    scan_limit = min(data_len, 1024)
    source_bytes = full_source_bytes(data)
    raw = source_bytes[:scan_limit] if source_bytes is not None else bytes(data[:scan_limit])
    end = len(raw)
    pos = 0
    token_count = 0
    container_depth = 0
    while pos < end and token_count < 64:
        byte = raw[pos]
        if WS_TABLE[byte]:
            pos += 1
            continue
        if byte == 37:
            pos = skip_comment(raw, pos, end)
            continue
        if byte == 40:
            pos = skip_literal_string(raw, pos, end)
            continue
        if byte == 60:
            if pos + 1 < end and raw[pos + 1] == 60:
                container_depth += 1
                pos += 2
            else:
                pos = skip_hex_string(raw, pos, end)
            continue
        if byte == 62 and pos + 1 < end and raw[pos + 1] == 62:
            container_depth = max(0, container_depth - 1)
            pos += 2
            continue
        if byte == 91:
            container_depth += 1
            pos += 1
            continue
        if byte == 93:
            container_depth = max(0, container_depth - 1)
            pos += 1
            continue
        if byte == 47:
            pos += 1
            while pos < end:
                byte = raw[pos]
                if SEPARATOR_TABLE[byte]:
                    break
                pos += 1
            continue
        if byte in DELIMITERS:
            pos += 1
            continue
        start = pos
        while pos < end:
            byte = raw[pos]
            if SEPARATOR_TABLE[byte]:
                break
            pos += 1
        token_count += 1
        if container_depth == 0 and raw[start:pos] in PDF_CONTENT_OPERATOR_BYTES:
            return True
    return False
