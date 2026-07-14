# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import binascii
import struct

from core_pdf.impl.engine.spec.s_07_filters.decode_spec import FilterParams
from core_pdf.impl.exceptions import PdfParseError

PDF_WHITESPACE_TABLE = bytes([1 if i in b"\x00\t\n\x0c\r " else 0 for i in range(256)])
PDF_WHITESPACE_BYTES = b"\x00\t\n\x0c\r "
ASCII85_DIGITS = bytes(range(33, 118))
ASCII85_PACK_QUAD = struct.Struct(">I").pack_into


def apply_ascii_hex(data: bytes, parms: object) -> bytes:
    filtered = bytearray()
    ws = PDF_WHITESPACE_TABLE
    for byte in data:
        if ws[byte]:
            continue
        if byte == 62:
            break
        if not (48 <= byte <= 57 or 65 <= byte <= 70 or 97 <= byte <= 102):
            continue
        filtered.append(byte)
    if len(filtered) & 1:
        filtered.append(48)
    try:
        return binascii.unhexlify(filtered)
    except binascii.Error as exc:
        raise PdfParseError("invalid ASCIIHexDecode stream") from exc


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
    ec = (
        parms.early_change
        if type(parms) is FilterParams
        else FilterParams.from_parms(parms).early_change
    )

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
        view = data if type(data) is memoryview else memoryview(data)
        start = 0
        end = len(view)

        while start < end and PDF_WHITESPACE_TABLE[view[start]]:
            start += 1
        while end > start and PDF_WHITESPACE_TABLE[view[end - 1]]:
            end -= 1

        if end - start >= 2 and view[start] == 60 and view[start + 1] == 126:
            start += 2

        clean = bytes(view[start:end]).translate(None, PDF_WHITESPACE_BYTES)
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
                        if acc <= 0xFFFFFFFF:
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
                    if acc <= 0xFFFFFFFF:
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
                    if acc <= 0xFFFFFFFF:
                        pack_quad(decoded, out_pos, acc)
                        out_pos += 4
                    pos += 5

            digits = clean_len - full_end
            if digits:
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
                if acc <= 0xFFFFFFFF:
                    pack_quad(decoded, out_pos, acc)
                    out_pos += digits - 1
            return bytes(decoded[:out_pos])

        if digits:
            for ignored in range(5 - digits):
                acc = acc * 85 + 84
            if acc <= 0xFFFFFFFF:
                append(acc.to_bytes(4, "big")[: digits - 1])
        return bytes(decoded)
    except (ValueError, binascii.Error) as exc:
        raise PdfParseError("invalid ASCII85Decode stream") from exc
