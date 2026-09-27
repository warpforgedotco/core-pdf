# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import ClassVar

from core_jbig2.bitmap import (
    compose_packed_bitmap_data,
    uint8_matrix_view,
)
from core_records import (
    GeneratedRecord,
    RecordType,
    ReplaceFields,
    ReprFields,
)

JBIG2_PAGE_INFO = 48
JBIG2_END_OF_FILE = 51
JBIG2_INTERMEDIATE_TEXT_REGION = 4
JBIG2_IMMEDIATE_TEXT_REGION = 6
JBIG2_IMMEDIATE_LOSSLESS_TEXT_REGION = 7
JBIG2_IMMEDIATE_GENERIC_REGION = 38
JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION = 39


class Jbig2Error(Exception):
    pass


class Jbig2ParseError(Jbig2Error):
    pass


class Jbig2UnsupportedError(Jbig2Error):
    pass


GENERIC_TEMPLATE_0_DEFAULT_AT = ((3, -1), (-3, -1), (2, -2), (-2, -2))


class JBIG2Segment(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    number: int
    flags: int
    retention_flags: int
    page_association: int
    data: bytes

    __hash__ = None  # type: ignore[assignment]

    @property
    def segment_type(self) -> int:
        return self.flags & 0x3F


class JBIG2PageInfo(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    width: int
    height: int
    x_resolution: int
    y_resolution: int
    flags: int

    __hash__ = None  # type: ignore[assignment]


class JBIG2Region(GeneratedRecord):
    width: int
    height: int
    x: int
    y: int
    flags: int
    raw: bytes


class JBIG2GenericRegionHeader(GeneratedRecord):
    region: JBIG2Region
    mmr: bool
    template: int
    prediction: bool
    adaptive_pixels: tuple[tuple[int, int], ...]
    bitmap_start: int


class JBIG2SegmentHeader(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    number: int
    flags: int
    retention_flags: int
    referred_to_count: int
    referred_to_segments: list[int]
    page_association: int
    data_length: int
    header_length: int

    __hash__ = None  # type: ignore[assignment]


class JBIG2Image(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    width: int
    height: int
    stride: int
    data: bytearray

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def create(cls, width: int, height: int) -> JBIG2Image:
        if width <= 0 or height <= 0:
            raise Jbig2ParseError("invalid JBIG2 image dimensions")
        stride = (width + 7) // 8
        return cls(width=width, height=height, stride=stride, data=bytearray(stride * height))

    def fill(self, value: int) -> None:
        fill_byte = 0xFF if value else 0x00
        self.data[:] = bytes([fill_byte]) * len(self.data)


def read_be_int32(data: bytes, pos: int, *, signed: bool = False) -> int:
    chunk = data[pos : pos + 4]
    if len(chunk) != 4:
        raise Jbig2ParseError("truncated JBIG2 data")
    return int.from_bytes(chunk, "big", signed=signed)


def read_be_i8(data: bytes, pos: int) -> int:
    value = data[pos]
    return value - 0x100 if value & 0x80 else value


def parse_page_info(data: bytes) -> JBIG2PageInfo:
    if len(data) < 19:
        raise Jbig2ParseError("truncated JBIG2 page info")
    width = read_be_int32(data, 0)
    height = read_be_int32(data, 4)
    x_resolution = read_be_int32(data, 8)
    y_resolution = read_be_int32(data, 12)
    flags = data[16]
    return JBIG2PageInfo(
        width=width,
        height=height,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        flags=flags,
    )


def parse_region(data: bytes, kind: str) -> JBIG2Region:
    if len(data) < 17:
        raise Jbig2ParseError(f"truncated JBIG2 {kind} region")
    return JBIG2Region(
        width=read_be_int32(data, 0),
        height=read_be_int32(data, 4),
        x=read_be_int32(data, 8, signed=True),
        y=read_be_int32(data, 12, signed=True),
        flags=data[16],
        raw=data,
    )


def parse_generic_region_header(region: JBIG2Region) -> JBIG2GenericRegionHeader:
    data = region.raw
    if len(data) < 18:
        raise Jbig2ParseError("truncated JBIG2 generic region")
    flags = data[17]
    mmr = bool(flags & 1)
    template = (flags >> 1) & 3
    prediction = bool(flags & 8)
    pos = 18
    at: list[tuple[int, int]] = []
    if not mmr:
        at_count = 4 if template == 0 else 1
        if len(data) < pos + at_count * 2:
            raise Jbig2ParseError("truncated JBIG2 generic region")
        for ignored in range(at_count):
            x = read_be_i8(data, pos)
            y = read_be_i8(data, pos + 1)
            at.append((x, y))
            pos += 2
    return JBIG2GenericRegionHeader(region, mmr, template, prediction, tuple(at), pos)


def read_uint(data: bytes, pos: int, width: int) -> tuple[int, int]:
    end = pos + width
    if end > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return int.from_bytes(data[pos:end], "big"), end


def read_segment_references(data: bytes, pos: int, count: int, width: int) -> tuple[list[int], int]:
    end = pos + count * width
    if end > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment references")
    return [
        int.from_bytes(data[index : index + width], "big") for index in range(pos, end, width)
    ], end


def parse_segment_header(data: bytes, pos: int) -> tuple[JBIG2SegmentHeader, int]:
    start = pos
    if pos + 11 > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment header")
    number, pos = read_uint(data, pos, 4)
    flags, pos = read_uint(data, pos, 1)
    retention_flags, pos = read_uint(data, pos, 1)
    referred_to_count = retention_flags >> 5
    if referred_to_count == 7:
        ref_count, pos = read_uint(data, pos, 3)
        referred_to_count = ((retention_flags & 0x1F) << 24) | ref_count
        bit_bytes = (referred_to_count + 1 + 7) // 8
        if pos + bit_bytes > len(data):
            raise Jbig2ParseError("truncated JBIG2 segment header")
        pos += bit_bytes
    elif referred_to_count in (5, 6):
        raise Jbig2ParseError("invalid JBIG2 referred-to segment count")
    reference_width = 1 if number <= 256 else 2 if number <= 65536 else 4
    referred_to_segments, pos = read_segment_references(
        data, pos, referred_to_count, reference_width
    )
    if pos + (4 if (flags & 0x40) else 1) + 4 > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment header")
    page_association, pos = read_uint(data, pos, 4 if flags & 0x40 else 1)
    data_length, pos = read_uint(data, pos, 4)
    return (
        JBIG2SegmentHeader(
            number=number,
            flags=flags,
            retention_flags=retention_flags,
            referred_to_count=referred_to_count,
            referred_to_segments=referred_to_segments,
            page_association=page_association,
            data_length=data_length,
            header_length=pos - start,
        ),
        pos,
    )


def parse_embedded_segments(data: bytes) -> list[JBIG2Segment]:
    pos = 0
    segments: list[JBIG2Segment] = []
    while pos + 11 <= len(data):
        header, pos = parse_segment_header(data, pos)
        if header.data_length == 0xFFFFFFFF or pos + header.data_length > len(data):
            payload = data[pos:]
            pos = len(data)
        else:
            payload = data[pos : pos + header.data_length]
            pos += header.data_length
        segments.append(
            JBIG2Segment(
                number=header.number,
                flags=header.flags,
                retention_flags=header.retention_flags,
                page_association=header.page_association,
                data=payload,
            )
        )
        if header.flags & 0x3F == JBIG2_END_OF_FILE:
            break
    return segments


class JBIG2PageDecoder:
    supported_segment_types: ClassVar[frozenset[int] | None] = None

    def __init__(self) -> None:
        self.page_info: JBIG2PageInfo | None = None
        self.image: JBIG2Image | None = None
        self.max_x = 0
        self.max_y = 0

    def ensure_image(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        image = self.image
        if image is None:
            self.image = JBIG2Image.create(width, height)
            return
        if width <= image.width and height <= image.height:
            return
        new_image = JBIG2Image.create(max(width, image.width), max(height, image.height))
        source = uint8_matrix_view(image.data, image.height, image.stride)
        destination = uint8_matrix_view(new_image.data, new_image.height, new_image.stride)
        destination[: image.height, : image.stride] = source
        self.image = new_image

    def include_region(self, region: JBIG2Region) -> None:
        self.max_x = max(self.max_x, region.x + region.width)
        self.max_y = max(self.max_y, region.y + region.height)
        self.ensure_image(self.max_x, self.max_y)

    def decode_segment(self, segment: JBIG2Segment) -> None:
        kind = segment.segment_type
        supported = self.supported_segment_types
        if supported is not None and kind not in supported:
            return
        if kind == JBIG2_PAGE_INFO:
            self.page_info = parse_page_info(segment.data)
            self.image = JBIG2Image.create(self.page_info.width, self.page_info.height)
            self.image.fill(jbig2_page_default_pixel(self.page_info))
        elif kind in (
            JBIG2_INTERMEDIATE_TEXT_REGION,
            JBIG2_IMMEDIATE_TEXT_REGION,
            JBIG2_IMMEDIATE_LOSSLESS_TEXT_REGION,
        ):
            region = parse_region(segment.data, "text")
            self.include_region(region)
            self.decode_text_region(region)
        elif kind in (JBIG2_IMMEDIATE_GENERIC_REGION, JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION):
            region = parse_region(segment.data, "generic")
            self.include_region(region)
            if self.image is not None:
                self.decode_generic_region(parse_generic_region_header(region))
        elif kind in (49, JBIG2_END_OF_FILE):
            if segment.data:
                raise Jbig2ParseError("JBIG2 end marker has segment data")
        elif kind == 52:
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 profiles segment")
            count = read_be_int32(segment.data, 0)
            if len(segment.data) != 4 + count * 4:
                raise Jbig2ParseError("invalid JBIG2 profiles segment length")
        elif kind == 62:
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 extension segment")
            extension = read_be_int32(segment.data, 0)
            if extension & (1 << 31):
                if not extension & (1 << 29):
                    raise Jbig2ParseError("necessary JBIG2 extension requires reserved bit 29")
                raise Jbig2UnsupportedError("unsupported necessary JBIG2 extension")
        elif kind in (0, 16, 20, 22, 23, 36, 40, 42, 43, 50, 53):
            raise Jbig2UnsupportedError(f"unsupported JBIG2 segment type {kind}")
        else:
            raise Jbig2ParseError(f"reserved JBIG2 segment type {kind}")

    def decode_text_region(self, region: JBIG2Region) -> None:
        if len(region.raw) < 20:
            raise Jbig2ParseError("truncated JBIG2 text region")
        raise Jbig2UnsupportedError("JBIG2 text region decoding is not implemented")

    def decode_generic_region(self, header: JBIG2GenericRegionHeader) -> None:
        if header.mmr:
            raise Jbig2UnsupportedError("JBIG2 MMR region decoding is not implemented")
        raise Jbig2UnsupportedError("JBIG2 arithmetic region decoding is not implemented")

    def finish(self) -> bytes:
        if self.image is None:
            raise Jbig2UnsupportedError("JBIG2Decode produced no image")
        return bytes(self.image.data)


def jbig2_page_default_pixel(page_info: JBIG2PageInfo) -> int:
    return (page_info.flags >> 2) & 1


def jbig2_page_combination_operator(page_info: JBIG2PageInfo | None) -> int:
    if page_info is None:
        return 0
    return (page_info.flags >> 3) & 3


def jbig2_page_allows_region_operator(page_info: JBIG2PageInfo | None) -> bool:
    return page_info is not None and bool(page_info.flags & 64)


def compose_packed_bitmap_region(
    region: JBIG2Region,
    packed_bitmap: bytes | bytearray,
    image: JBIG2Image,
    page_info: JBIG2PageInfo | None,
) -> None:
    operator = region_operator(region, page_info)
    if operator not in (0, 2):
        raise Jbig2UnsupportedError(f"unsupported JBIG2 combination operator {operator}")
    row_bytes = max(1, (region.width + 7) // 8)
    compose_packed_bitmap_data(
        packed_bitmap,
        min(region.height, len(packed_bitmap) // row_bytes),
        region.width,
        region.x,
        region.y,
        image.width,
        image.height,
        image.stride,
        image.data,
        operator,
    )


def region_operator(region: JBIG2Region, page_info: JBIG2PageInfo | None) -> int:
    if jbig2_page_allows_region_operator(page_info):
        return region.flags & 7
    return jbig2_page_combination_operator(page_info)


__all__ = (
    "JBIG2Region",
    "JBIG2GenericRegionHeader",
    "JBIG2PageDecoder",
    "parse_region",
    "parse_embedded_segments",
    "JBIG2_PAGE_INFO",
    "JBIG2_END_OF_FILE",
    "JBIG2_INTERMEDIATE_TEXT_REGION",
    "JBIG2_IMMEDIATE_TEXT_REGION",
    "JBIG2_IMMEDIATE_LOSSLESS_TEXT_REGION",
    "JBIG2_IMMEDIATE_GENERIC_REGION",
    "JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION",
    "Jbig2Error",
    "Jbig2ParseError",
    "Jbig2UnsupportedError",
    "GENERIC_TEMPLATE_0_DEFAULT_AT",
    "JBIG2Segment",
    "JBIG2PageInfo",
    "JBIG2SegmentHeader",
    "JBIG2Image",
    "parse_page_info",
    "parse_generic_region_header",
    "parse_segment_header",
    "compose_packed_bitmap_region",
)
