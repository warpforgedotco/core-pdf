# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import binascii
import struct
import zlib

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_syntax_primitives.tokens import WHITESPACE

ASCII85_DIGITS = bytes(range(33, 118))
ASCII85_PACK_QUAD = struct.Struct(">I").pack_into
ASCII85_MAX = 0xFFFFFFFF


class IncompleteLzwError(ValueError):
    def __init__(self, message: str, decoded: bytes) -> None:
        super().__init__(message)
        self.decoded = decoded


def apply_ascii_hex(data: bytes, parms: object) -> bytes:
    terminator = data.find(b">")
    if terminator < 0:
        raise FilterParseError("missing ASCIIHex end marker")
    filtered = data[:terminator].translate(None, WHITESPACE)
    if len(filtered) & 1:
        filtered += b"0"
    return binascii.unhexlify(filtered)


def apply_flate(data: bytes, parms: object) -> bytes:
    try:
        return zlib.decompress(data)
    except zlib.error as exc:
        raise FilterParseError("invalid FlateDecode stream") from exc


def apply_run_length(data: bytes, parms: object) -> bytes:
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        length = data[i]
        i += 1
        if length == 128:
            return bytes(out)
        if length < 128:
            run = length + 1
            if i + run > n:
                raise FilterParseError("truncated RunLength stream")
            out.extend(data[i : i + run])
            i += run
            continue
        run = 257 - length
        if i >= n:
            raise FilterParseError("truncated RunLength stream")
        out.extend(data[i : i + 1] * run)
        i += 1
    raise FilterParseError("missing RunLength end marker")


class BitReader:
    __slots__ = ("data", "pos", "length", "buffer", "bits_in_buffer")

    def __init__(self, data: bytes | memoryview):
        self.data = data
        self.pos = 0
        self.length = len(data)
        self.buffer = 0
        self.bits_in_buffer = 0

    def read_bits(self, n: int) -> int | None:
        if self.bits_in_buffer < n:
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
    params = parms if isinstance(parms, FilterParams) else FilterParams.from_parms(parms)
    ec = params.early_change

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
            raise IncompleteLzwError("truncated LZW stream", bytes(out))
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
                raise IncompleteLzwError("invalid LZW code", bytes(out))
            raise ValueError(f"invalid LZW code: {code}")

        out_extend(entry)
        if prev is not None and next_code < 4096:
            table[next_code] = prev + entry[:1]
            next_code += 1
            if next_code == (1 << code_size) - ec and code_size < 12:
                code_size += 1
        prev = entry
    return bytes(out)


def ascii85_tail(accumulator: int, digits: int) -> bytes:
    if not digits:
        return b""
    if digits == 1:
        raise ValueError("invalid final Ascii85 tuple")
    for _ in range(5 - digits):
        accumulator = accumulator * 85 + 84
    if accumulator > ASCII85_MAX:
        raise ValueError("Ascii85 overflow")
    return accumulator.to_bytes(4, "big")[: digits - 1]


def apply_ascii85(data: bytes | memoryview, parms: object) -> bytes:
    try:
        clean = bytes(data).translate(None, WHITESPACE)
        terminator = clean.find(b"~>")
        if terminator < 0:
            raise ValueError("missing ASCII85 end marker")
        clean = clean[:terminator]
        invalid = clean.translate(None, ASCII85_DIGITS + b"z")
        if invalid:
            raise ValueError(f"Non-Ascii85 digit found: {chr(invalid[0])}")

        acc = 0
        digits = 0
        if 122 in clean:
            decoded = bytearray()
            for byte in clean:
                if byte == 122:
                    if digits:
                        raise ValueError("z inside Ascii85 5-tuple")
                    decoded.extend(b"\x00\x00\x00\x00")
                    continue
                acc = acc * 85 + (byte - 33)
                digits += 1
                if digits == 5:
                    if acc > ASCII85_MAX:
                        raise ValueError("Ascii85 overflow")
                    decoded.extend(acc.to_bytes(4, "big"))
                    acc = 0
                    digits = 0
            decoded.extend(ascii85_tail(acc, digits))
            return bytes(decoded)

        clean_len = len(clean)
        decoded = bytearray(((clean_len + 4) // 5) * 4)
        out_pos = 0
        full_end = clean_len - clean_len % 5
        pos = 0
        if full_end >= 4096:
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
        pack_quad = ASCII85_PACK_QUAD
        while pos < full_end:
            acc = (
                (clean[pos] - 33) * 52200625
                + (clean[pos + 1] - 33) * 614125
                + (clean[pos + 2] - 33) * 7225
                + (clean[pos + 3] - 33) * 85
                + clean[pos + 4]
                - 33
            )
            if acc > ASCII85_MAX:
                raise ValueError("Ascii85 overflow")
            pack_quad(decoded, out_pos, acc)
            out_pos += 4
            pos += 5
        acc = 0
        for byte in clean[full_end:]:
            acc = acc * 85 + (byte - 33)
        tail = ascii85_tail(acc, clean_len - full_end)
        decoded[out_pos : out_pos + len(tail)] = tail
        return bytes(decoded[: out_pos + len(tail)])
    except (ValueError, binascii.Error) as exc:
        raise FilterParseError("invalid ASCII85Decode stream") from exc


__all__ = (
    "ASCII85_DIGITS",
    "ASCII85_PACK_QUAD",
    "ASCII85_MAX",
    "IncompleteLzwError",
    "apply_ascii_hex",
    "apply_flate",
    "apply_run_length",
    "BitReader",
    "apply_lzw",
    "apply_ascii85",
)
