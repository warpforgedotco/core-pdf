# SPDX-License-Identifier: AGPL-3.0-only
"""Stream codecs: ASCIIHex, ASCII85, Flate, LZW, and run-length."""

from __future__ import annotations

import binascii
import struct
import zlib

import imagecodecs

from core_pdf.impl.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import (
    DELIMITERS,
    PDF_CONTENT_OPERATOR_BYTES,
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
)

ASCII_HEX_DIGITS = b"0123456789ABCDEFabcdef"
ASCII_HEX_INVALID_BYTES = bytes(byte for byte in range(256) if byte not in ASCII_HEX_DIGITS)
ASCII85_DIGITS = bytes(range(33, 118))
ASCII85_PACK_QUAD = struct.Struct(">I").pack_into
ASCII85_MAX = 0xFFFFFFFF


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


class BitReader:
    __slots__ = ("data", "pos", "length", "buffer", "bits_in_buffer")

    def __init__(self, data: bytes | memoryview):
        self.data = data
        self.pos = 0
        self.length = len(data)
        self.buffer = 0
        self.bits_in_buffer = 0

    def read_bits(self, n: int) -> int | None:
        if self.bits_in_buffer >= n:
            self.bits_in_buffer -= n
            value = (self.buffer >> self.bits_in_buffer) & ((1 << n) - 1)
            self.buffer &= (1 << self.bits_in_buffer) - 1 if self.bits_in_buffer else 0
            return value

        if self.pos >= self.length:
            return None

        data = self.data
        pos = self.pos
        length = self.length
        buffer = self.buffer
        bits_in_buffer = self.bits_in_buffer

        while bits_in_buffer < n and pos < length:
            buffer = (buffer << 8) | data[pos]
            bits_in_buffer += 8
            pos += 1

        self.pos = pos
        self.buffer = buffer
        self.bits_in_buffer = bits_in_buffer
        if bits_in_buffer < n:
            return None

        self.bits_in_buffer -= n
        value = (self.buffer >> self.bits_in_buffer) & ((1 << n) - 1)
        self.buffer &= (1 << self.bits_in_buffer) - 1 if self.bits_in_buffer else 0
        return value


def apply_lzw(data: bytes | memoryview, parms: object) -> bytes:
    params = parms if type(parms) is FilterParams else FilterParams.from_parms(parms)
    ec = params.early_change

    if ec == 1 and imagecodecs.LZW.available:
        try:
            return bytes(imagecodecs.lzw_decode(data))
        except Exception as exc:  # pragma: no cover - C-extension integration boundary
            raise ValueError("invalid LZW stream") from exc

    code_size = 9
    next_code = 258

    table: list[bytes] = [bytes([i]) for i in range(256)] + [b""] * (4096 - 256)

    reader = BitReader(data)
    out = bytearray()
    prev: bytes | None = None

    out_extend = out.extend

    while True:
        code = reader.read_bits(code_size)
        if code is None:
            break
        if code == 256:
            code_size = 9
            next_code = 258
            prev = None
            continue
        if code == 257:
            break

        if code < next_code:
            entry = table[code]
        elif code == next_code and prev is not None:
            entry = prev + prev[:1]
        else:
            if out:
                break
            raise ValueError(f"invalid LZW code: {code}")

        out_extend(entry)
        if prev is not None and next_code < 4096:
            table[next_code] = prev + entry[:1]
            next_code += 1
            if next_code == (1 << code_size) - ec and code_size < 12:
                code_size += 1
        prev = entry
    return bytes(out)


def apply_ascii85(data: bytes | memoryview, parms: object) -> bytes:
    try:
        clean = bytes(data).lstrip(WHITESPACE)
        if clean.startswith(b"<~"):
            clean = clean[2:]
        clean = clean.translate(None, WHITESPACE)
        terminator = clean.find(b"~>")
        if terminator >= 0:
            clean = clean[:terminator]
        clean_len = len(clean)

        acc = 0
        digits = 0
        zero_quad = b"\x00\x00\x00\x00"
        has_z = 122 in clean
        if has_z:
            decoded = bytearray()
            append = decoded.extend
            for byte in clean:
                if byte == 122:
                    if digits:
                        raise ValueError("z inside Ascii85 5-tuple")
                    append(zero_quad)
                    continue
                if 33 <= byte <= 117:
                    acc = acc * 85 + (byte - 33)
                    digits += 1
                    if digits == 5:
                        if acc > ASCII85_MAX:
                            raise ValueError("Ascii85 overflow")
                        append(acc.to_bytes(4, "big"))
                        acc = 0
                        digits = 0
                    continue
                raise ValueError(f"Non-Ascii85 digit found: {chr(byte)}")
        else:
            decoded = bytearray(((clean_len + 4) // 5) * 4)
            out_pos = 0
            pack_quad = ASCII85_PACK_QUAD
            full_end = clean_len - clean_len % 5
            pos = 0
            invalid_digits = clean.translate(None, ASCII85_DIGITS)
            # Keep short streams on the allocation-free scalar path.  For large,
            # validated streams, five ASCII85 digits form an independent numeric
            # group and can be decoded in bulk.
            if not invalid_digits and full_end >= 4096:
                import numpy

                groups = numpy.frombuffer(clean[:full_end], dtype=numpy.uint8).reshape(-1, 5)
                values = groups.astype(numpy.uint64) - 33
                accumulators = (
                    values[:, 0] * 52200625
                    + values[:, 1] * 614125
                    + values[:, 2] * 7225
                    + values[:, 3] * 85
                    + values[:, 4]
                )
                if bool(numpy.any(accumulators > ASCII85_MAX)):
                    raise ValueError("Ascii85 overflow")
                encoded = accumulators.astype(">u4", copy=False).tobytes()
                decoded[: len(encoded)] = encoded
                out_pos = len(encoded)
                pos = full_end
            if invalid_digits:
                while pos < full_end:
                    byte0 = clean[pos]
                    byte1 = clean[pos + 1]
                    byte2 = clean[pos + 2]
                    byte3 = clean[pos + 3]
                    byte4 = clean[pos + 4]
                    if not (
                        33 <= byte0 <= 117
                        and 33 <= byte1 <= 117
                        and 33 <= byte2 <= 117
                        and 33 <= byte3 <= 117
                        and 33 <= byte4 <= 117
                    ):
                        for byte in (byte0, byte1, byte2, byte3, byte4):
                            if not 33 <= byte <= 117:
                                raise ValueError(f"Non-Ascii85 digit found: {chr(byte)}")
                    acc = (
                        (byte0 - 33) * 52200625
                        + (byte1 - 33) * 614125
                        + (byte2 - 33) * 7225
                        + (byte3 - 33) * 85
                        + byte4
                        - 33
                    )
                    if acc > ASCII85_MAX:
                        raise ValueError("Ascii85 overflow")
                    pack_quad(decoded, out_pos, acc)
                    out_pos += 4
                    pos += 5
            else:
                while pos < full_end:
                    byte0 = clean[pos]
                    byte1 = clean[pos + 1]
                    byte2 = clean[pos + 2]
                    byte3 = clean[pos + 3]
                    byte4 = clean[pos + 4]
                    acc = (
                        (byte0 - 33) * 52200625
                        + (byte1 - 33) * 614125
                        + (byte2 - 33) * 7225
                        + (byte3 - 33) * 85
                        + byte4
                        - 33
                    )
                    if acc > ASCII85_MAX:
                        raise ValueError("Ascii85 overflow")
                    pack_quad(decoded, out_pos, acc)
                    out_pos += 4
                    pos += 5

            digits = clean_len - full_end
            if digits:
                if digits == 1:
                    raise ValueError("invalid final Ascii85 tuple")
                acc = 0
                if invalid_digits:
                    for byte in clean[full_end:]:
                        if 33 <= byte <= 117:
                            acc = acc * 85 + (byte - 33)
                            continue
                        raise ValueError(f"Non-Ascii85 digit found: {chr(byte)}")
                else:
                    for byte in clean[full_end:]:
                        acc = acc * 85 + (byte - 33)
                for count in range(5 - digits):
                    acc = acc * 85 + 84
                if acc > ASCII85_MAX:
                    raise ValueError("Ascii85 overflow")
                pack_quad(decoded, out_pos, acc)
                out_pos += digits - 1
            return bytes(decoded[:out_pos])

        if digits:
            if digits == 1:
                raise ValueError("invalid final Ascii85 tuple")
            for ignored in range(5 - digits):
                acc = acc * 85 + 84
            if acc > ASCII85_MAX:
                raise ValueError("Ascii85 overflow")
            append(acc.to_bytes(4, "big")[: digits - 1])
        return bytes(decoded)
    except (ValueError, binascii.Error) as exc:
        raise FilterParseError("invalid ASCII85Decode stream") from exc


# Incomplete raw Deflate has no signature, so a few arbitrary bytes can decode
# to garbage without the decoder ever reaching an end-of-stream marker.
MIN_TRUNCATED_RAW_FLATE_BYTES = 8


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
