# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import zlib

from core_pdf.impl.engine.spec.s_07_filters.codecs import PDF_WHITESPACE_TABLE
from core_pdf.impl.exceptions import PdfParseError

PDF_CONTENT_OPERATORS = {
    b"b",
    b"b*",
    b"B",
    b"B*",
    b"BI",
    b"BT",
    b"c",
    b"cm",
    b"CS",
    b"cs",
    b"d",
    b"Do",
    b"ET",
    b"f",
    b"F",
    b"f*",
    b"gs",
    b"h",
    b"i",
    b"ID",
    b"j",
    b"J",
    b"l",
    b"m",
    b"M",
    b"n",
    b"q",
    b"Q",
    b"re",
    b"ri",
    b"s",
    b"S",
    b"SC",
    b"sc",
    b"SCN",
    b"scn",
    b"sh",
    b"T*",
    b"TD",
    b"Td",
    b"Tf",
    b"Tj",
    b"TJ",
    b"Tm",
    b"Tr",
    b"Ts",
    b"Tw",
    b"Tz",
    b"v",
    b"w",
    b"W",
    b"W*",
    b"y",
    b"'",
    b'"',
}
PDF_CONTENT_DELIMITERS = b"()<>[]{}/%"


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

    for wbits in candidates:
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            recovered = recover_flate(data, wbits)
            if recovered:
                return recovered

    if looks_like_pdf_content_stream(data):
        return bytes(data)
    raise PdfParseError("invalid FlateDecode stream")


def recover_flate(data: bytes, wbits: int = zlib.MAX_WBITS) -> bytes:
    try:
        decoder = zlib.decompressobj(wbits)
        return decoder.decompress(data) + decoder.flush()
    except zlib.error:
        if wbits > 0 and len(data) > 6:
            cmf = data[0]
            flg = data[1]
            if cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0:
                try:
                    return zlib.decompress(data[2:-4], -15)
                except zlib.error:
                    return b""
        return b""


def looks_like_pdf_content_stream(data: bytes | memoryview) -> bool:
    view = data if type(data) is memoryview else memoryview(data)
    end = len(view)
    pos = 0
    token_count = 0
    scan_limit = min(end, 1024)
    while pos < scan_limit and token_count < 64:
        byte = view[pos]
        if PDF_WHITESPACE_TABLE[byte]:
            pos += 1
            continue
        if byte == 37:
            newline = bytes(view[pos:scan_limit]).find(b"\n")
            if newline < 0:
                return False
            pos += newline + 1
            continue
        if byte in PDF_CONTENT_DELIMITERS:
            pos += 1
            continue
        start = pos
        while pos < scan_limit:
            byte = view[pos]
            if PDF_WHITESPACE_TABLE[byte] or byte in PDF_CONTENT_DELIMITERS:
                break
            pos += 1
        token = bytes(view[start:pos])
        token_count += 1
        if token in PDF_CONTENT_OPERATORS:
            return True
    return False
