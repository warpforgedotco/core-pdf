# SPDX-License-Identifier: AGPL-3.0-only
"""JBIG2 segment records, bitmap storage, and binary fields."""

from __future__ import annotations

from dataclasses import dataclass

MAX_BITMAP_PIXELS = 1 << 28
MAX_SYMBOLS = 100_000
MAX_INSTANCES = 1_000_000
MAX_INTEGER_RUNS = 100_000
MAX_REFERENCES = 100_000


class Jbig2Error(Exception):
    """Base error for JBIG2 codec failures."""


class Jbig2ParseError(Jbig2Error):
    """Raised when JBIG2 bytes are malformed."""


class Jbig2UnsupportedError(Jbig2Error):
    """Raised when valid JBIG2 data uses unsupported features."""


@dataclass(slots=True)
class JBIG2Segment:
    number: int
    flags: int
    retention_flags: int
    page_association: int
    data: bytes
    referred_to_segments: tuple[int, ...] = ()

    @property
    def segment_type(self) -> int:
        return self.flags & 0x3F


@dataclass(slots=True)
class JBIG2PageInfo:
    width: int
    height: int
    x_resolution: int
    y_resolution: int
    flags: int


@dataclass(slots=True)
class JBIG2Region:
    width: int
    height: int
    x: int
    y: int
    flags: int
    raw: bytes


@dataclass(slots=True)
class JBIG2SegmentHeader:
    number: int
    flags: int
    retention_flags: int
    referred_to_count: int
    referred_to_segments: list[int]
    page_association: int
    data_length: int
    header_length: int


@dataclass(slots=True)
class JBIG2Image:
    width: int
    height: int
    stride: int
    data: bytearray

    @classmethod
    def create(cls, width: int, height: int, *, allow_empty: bool = False) -> "JBIG2Image":
        if width < 0 or height < 0 or (not allow_empty and not width * height):
            raise Jbig2ParseError("invalid JBIG2 image dimensions")
        if width * height > MAX_BITMAP_PIXELS:
            raise Jbig2UnsupportedError("JBIG2 bitmap exceeds supported pixel limit")
        stride = (width + 7) // 8
        return cls(width=width, height=height, stride=stride, data=bytearray(stride * height))

    def fill(self, value: int) -> None:
        fill_byte = 0xFF if value else 0x00
        self.data[:] = bytes([fill_byte]) * len(self.data)


def read_be_u32(data: bytes, pos: int) -> int:
    return (data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3]


def read_be_i32(data: bytes, pos: int) -> int:
    value = read_be_u32(data, pos)
    return value - 0x100000000 if value & 0x80000000 else value


def read_be_i8(data: bytes, pos: int) -> int:
    value = data[pos]
    return value - 0x100 if value & 0x80 else value
