# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError


class JpegBitReader:
    __slots__ = ("data", "pos", "buffer", "bits_left", "len")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.buffer = 0
        self.bits_left = 0
        self.len = len(data)

    def fill_byte(self) -> None:
        data = self.data
        pos = self.pos
        if pos >= self.len:
            raise JpegUnsupportedError("Unexpected end of JPEG scan data")
        byte = data[pos]
        pos += 1
        if byte == 0xFF and pos < self.len and data[pos] == 0x00:
            pos += 1
        self.pos = pos
        self.buffer = (self.buffer << 8) | byte
        self.bits_left += 8

    def get_bits(self, n: int) -> int:
        if n == 0:
            return 0
        while self.bits_left < n and self.pos < self.len:
            self.fill_byte()
        bits_left = self.bits_left
        if bits_left < n:
            missing = n - bits_left
            value = ((self.buffer << missing) | ((1 << missing) - 1)) & ((1 << n) - 1)
            self.bits_left = 0
            self.buffer = 0
            return value
        bits_left -= n
        value = (self.buffer >> bits_left) & ((1 << n) - 1)
        self.bits_left = bits_left
        if bits_left:
            self.buffer &= (1 << bits_left) - 1
        else:
            self.buffer = 0
        return value

    def peek_bits(self, n: int) -> int:
        while self.bits_left < n and self.pos < self.len:
            self.fill_byte()
        bits_left = self.bits_left
        if bits_left < n:
            missing = n - bits_left
            return ((self.buffer << missing) | ((1 << missing) - 1)) & ((1 << n) - 1)
        return (self.buffer >> (bits_left - n)) & ((1 << n) - 1)

    def drop_bits(self, n: int) -> None:
        if n >= self.bits_left:
            self.bits_left = 0
            self.buffer = 0
            return
        bits_left = self.bits_left - n
        self.bits_left = bits_left
        if bits_left:
            self.buffer &= (1 << bits_left) - 1
        else:
            self.buffer = 0

    def get_bit(self) -> int:
        bits_left = self.bits_left
        if bits_left == 0:
            if self.pos >= self.len:
                raise JpegParseError("unexpected end of JPEG scan data")
            self.fill_byte()
            bits_left = self.bits_left
        bits_left -= 1
        bit = (self.buffer >> bits_left) & 1
        self.bits_left = bits_left
        if bits_left:
            self.buffer &= (1 << bits_left) - 1
        else:
            self.buffer = 0
        return bit
