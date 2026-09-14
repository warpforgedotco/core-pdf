# SPDX-License-Identifier: AGPL-3.0-only
"""JBIG2 parsing and decoding codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.jbig2.bitmap_kernels import (
    compose_packed_bitmap_data,
    internal_uint8_view,
    uint8_matrix_view,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import coerce_to_bytes, is_pdf_null

JBIG2_PAGE_INFO = 48
JBIG2_END_OF_FILE = 51
JBIG2_IMMEDIATE_GENERIC_REGION = 38
JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION = 39


class Jbig2Error(Exception):
    """Base error for JBIG2 codec failures."""


class Jbig2ParseError(Jbig2Error):
    """Raised when JBIG2 bytes are malformed."""


class Jbig2UnsupportedError(Jbig2Error):
    """Raised when valid JBIG2 data uses unsupported features."""


GENERIC_TEMPLATE_0_DEFAULT_AT = ((3, -1), (-3, -1), (2, -2), (-2, -2))

# The MQ-coder probability estimation table, ITU-T T.88 Table E.1. One row per
# state index: the Qe probability, the next index after a more-probable-symbol
# renormalization, the next index after a less-probable one, and whether that
# LPS path swaps the sense of the MPS.
#
internal_MQ_STATES: tuple[tuple[int, int, int, int], ...] = (
    (0x5601, 1, 1, 1),
    (0x3401, 2, 6, 0),
    (0x1801, 3, 9, 0),
    (0x0AC1, 4, 12, 0),
    (0x0521, 5, 29, 0),
    (0x0221, 38, 33, 0),
    (0x5601, 7, 6, 1),
    (0x5401, 8, 14, 0),
    (0x4801, 9, 14, 0),
    (0x3801, 10, 14, 0),
    (0x3001, 11, 17, 0),
    (0x2401, 12, 18, 0),
    (0x1C01, 13, 20, 0),
    (0x1601, 29, 21, 0),
    (0x5601, 15, 14, 1),
    (0x5401, 16, 14, 0),
    (0x5101, 17, 15, 0),
    (0x4801, 18, 16, 0),
    (0x3801, 19, 17, 0),
    (0x3401, 20, 18, 0),
    (0x3001, 21, 19, 0),
    (0x2801, 22, 19, 0),
    (0x2401, 23, 20, 0),
    (0x2201, 24, 21, 0),
    (0x1C01, 25, 22, 0),
    (0x1801, 26, 23, 0),
    (0x1601, 27, 24, 0),
    (0x1401, 28, 25, 0),
    (0x1201, 29, 26, 0),
    (0x1101, 30, 27, 0),
    (0x0AC1, 31, 28, 0),
    (0x09C1, 32, 29, 0),
    (0x08A1, 33, 30, 0),
    (0x0521, 34, 31, 0),
    (0x0441, 35, 32, 0),
    (0x02A1, 36, 33, 0),
    (0x0221, 37, 34, 0),
    (0x0141, 38, 35, 0),
    (0x0111, 39, 36, 0),
    (0x0085, 40, 37, 0),
    (0x0049, 41, 38, 0),
    (0x0025, 42, 39, 0),
    (0x0015, 43, 40, 0),
    (0x0009, 44, 41, 0),
    (0x0005, 45, 42, 0),
    (0x0001, 45, 43, 0),
    (0x5601, 46, 46, 0),
)

# The decoder indexes these per pixel, so the columns stay flat tuples of ints.
MQ_QE = tuple(state[0] for state in internal_MQ_STATES)
MQ_NMPS = tuple(state[1] for state in internal_MQ_STATES)
MQ_NLPS = tuple(state[2] for state in internal_MQ_STATES)
MQ_SWITCH = tuple(state[3] for state in internal_MQ_STATES)


@dataclass(slots=True)
class JBIG2Segment:
    number: int
    flags: int
    retention_flags: int
    page_association: int
    data: bytes

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


@dataclass(frozen=True, slots=True)
class JBIG2Region:
    width: int
    height: int
    x: int
    y: int
    flags: int
    raw: bytes


@dataclass(frozen=True, slots=True)
class JBIG2GenericRegionHeader:
    region: JBIG2Region
    mmr: bool
    template: int
    prediction: bool
    adaptive_pixels: tuple[tuple[int, int], ...]
    bitmap_start: int


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
    def create(cls, width: int, height: int) -> "JBIG2Image":
        if width <= 0 or height <= 0:
            raise Jbig2ParseError("invalid JBIG2 image dimensions")
        stride = (width + 7) // 8
        return cls(width=width, height=height, stride=stride, data=bytearray(stride * height))

    def fill(self, value: int) -> None:
        fill_byte = 0xFF if value else 0x00
        self.data[:] = bytes([fill_byte]) * len(self.data)


class JBIG2MQDecoder:
    __slots__ = ("data", "bp", "data_end", "a", "chigh", "clow", "ct")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bp = 0
        self.data_end = len(data)
        self.chigh = data[0] if data else 0xFF
        self.clow = 0
        self.ct = 0
        self.byte_in()
        self.chigh = ((self.chigh << 7) & 0xFFFF) | ((self.clow >> 9) & 0x7F)
        self.clow = (self.clow << 7) & 0xFFFF
        self.ct -= 7
        self.a = 0x8000

    def byte_in(self) -> None:
        data = self.data
        bp = self.bp
        current = data[bp] if bp < self.data_end else 0xFF
        following = data[bp + 1] if bp + 1 < self.data_end else 0xFF
        if current == 0xFF:
            if following > 0x8F:
                self.clow += 0xFF00
                self.ct = 8
            else:
                bp += 1
                value = data[bp] if bp < self.data_end else 0xFF
                self.clow += value << 9
                self.ct = 7
                self.bp = bp
        else:
            bp += 1
            value = data[bp] if bp < self.data_end else 0xFF
            self.clow += value << 8
            self.ct = 8
            self.bp = bp
        if self.clow > 0xFFFF:
            self.chigh += self.clow >> 16
            self.clow &= 0xFFFF


def read_be_u32(data: bytes, pos: int) -> int:
    return (data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3]


def read_be_i32(data: bytes, pos: int) -> int:
    value = read_be_u32(data, pos)
    return value - 0x100000000 if value & 0x80000000 else value


def read_be_i8(data: bytes, pos: int) -> int:
    value = data[pos]
    return value - 0x100 if value & 0x80 else value


def parse_page_info(data: bytes) -> JBIG2PageInfo:
    if len(data) < 19:
        raise Jbig2ParseError("truncated JBIG2 page info")
    width = read_be_u32(data, 0)
    height = read_be_u32(data, 4)
    x_resolution = read_be_u32(data, 8)
    y_resolution = read_be_u32(data, 12)
    flags = data[16]
    return JBIG2PageInfo(
        width=width,
        height=height,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        flags=flags,
    )


def parse_region(data: bytes, kind: str) -> JBIG2Region:
    """Read the information field shared by every JBIG2 region type."""
    if len(data) < 17:
        raise Jbig2ParseError(f"truncated JBIG2 {kind} region")
    return JBIG2Region(
        width=read_be_u32(data, 0),
        height=read_be_u32(data, 4),
        x=read_be_i32(data, 8),
        y=read_be_i32(data, 12),
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


def read_u8(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 1 > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return data[pos], pos + 1


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return read_be_u32(data, pos), pos + 4


def read_u24(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 3 > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return (data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2], pos + 3


def parse_page_association(data: bytes, pos: int, long_form: bool) -> tuple[int, int]:
    if long_form:
        return read_u32(data, pos)
    return read_u8(data, pos)


def parse_referred_to_segments(
    data: bytes, pos: int, count: int, long_form: bool
) -> tuple[list[int], int]:
    return internal_read_segment_references(data, pos, count, 4 if long_form else 1)


def internal_read_segment_references(
    data: bytes, pos: int, count: int, width: int
) -> tuple[list[int], int]:
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
    number, pos = read_u32(data, pos)
    flags, pos = read_u8(data, pos)
    retention_flags, pos = read_u8(data, pos)
    referred_to_count = retention_flags >> 5
    # T.88 7.2.4: the long form uses all 29 count bits, then retention bytes.
    if referred_to_count == 7:
        ref_count, pos = read_u24(data, pos)
        referred_to_count = ((retention_flags & 0x1F) << 24) | ref_count
        bit_bytes = (referred_to_count + 1 + 7) // 8
        if pos + bit_bytes > len(data):
            raise Jbig2ParseError("truncated JBIG2 segment header")
        pos += bit_bytes
    elif referred_to_count in (5, 6):
        raise Jbig2ParseError("invalid JBIG2 referred-to segment count")
    # T.88 7.2.5: reference width depends on this segment's number, in both forms.
    reference_width = 1 if number <= 256 else 2 if number <= 65536 else 4
    referred_to_segments, pos = internal_read_segment_references(
        data, pos, referred_to_count, reference_width
    )
    if pos + (4 if (flags & 0x40) else 1) + 4 > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment header")
    page_association, pos = parse_page_association(data, pos, bool(flags & 0x40))
    data_length, pos = read_u32(data, pos)
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
    """Read the headerless segment organization used by PDF (T.88 Annex D.3)."""
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
    """Own one page canvas and the supported T.88 segment decoding procedures.

    Reader implementations may override ``decode_segment``, ``decode_text_region``
    and ``decode_generic_region`` to own recovery. Region methods receive metadata
    parsed once, after the canvas includes the region's extent.
    """

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
        if kind == JBIG2_PAGE_INFO:
            self.page_info = parse_page_info(segment.data)
            self.image = JBIG2Image.create(self.page_info.width, self.page_info.height)
            self.image.fill(jbig2_page_default_pixel(self.page_info))
        elif kind in (4, 6, 7):
            region = parse_region(segment.data, "text")
            self.include_region(region)
            self.decode_text_region(region)
        elif kind in (JBIG2_IMMEDIATE_GENERIC_REGION, JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION):
            region = parse_region(segment.data, "generic")
            self.include_region(region)
            if self.image is not None:
                self.decode_generic_region(parse_generic_region_header(region))
        elif kind in (49, JBIG2_END_OF_FILE):
            # T.88, 7.4.9 and 7.4.11: these markers have no associated data.
            if segment.data:
                raise Jbig2ParseError("JBIG2 end marker has segment data")
        elif kind == 52:
            # T.88, 7.4.12: profile declarations do not contribute image pixels.
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 profiles segment")
            count = read_be_u32(segment.data, 0)
            if len(segment.data) != 4 + count * 4:
                raise Jbig2ParseError("invalid JBIG2 profiles segment length")
        elif kind == 62:
            # T.88, 7.4.14: an unknown necessary extension prevents decoding.
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 extension segment")
            extension = read_be_u32(segment.data, 0)
            if extension & (1 << 31):
                if not extension & (1 << 29):
                    raise Jbig2ParseError("necessary JBIG2 extension requires reserved bit 29")
                raise Jbig2UnsupportedError("unsupported necessary JBIG2 extension")
        elif kind in (0, 16, 20, 22, 23, 36, 40, 42, 43, 50, 53):
            raise Jbig2UnsupportedError(f"unsupported JBIG2 segment type {kind}")
        else:
            raise Jbig2ParseError(f"reserved JBIG2 segment type {kind}")

    def decode_text_region(self, region: JBIG2Region) -> None:
        # T.88, 6.4: text regions require symbol-instance decoding.
        if len(region.raw) < 20:
            raise Jbig2ParseError("truncated JBIG2 text region")
        raise Jbig2UnsupportedError("JBIG2 text region decoding is not implemented")

    def decode_generic_region(self, header: JBIG2GenericRegionHeader) -> None:
        # T.88, 6.2.6 requires T.6 decoding for an MMR region, not raw pixels.
        if header.mmr:
            raise Jbig2UnsupportedError("JBIG2 MMR region decoding is not implemented")
        region = header.region
        if region.width <= 0 or region.height <= 0:
            return
        bitmap = decode_arithmetic_generic_bitmap(
            region.raw[header.bitmap_start :],
            region.width,
            region.height,
            header.template,
            header.prediction,
            header.adaptive_pixels,
        )
        if self.image is not None:
            compose_packed_bitmap_region(region, bitmap, self.image, self.page_info)

    def finish(self) -> bytes:
        if self.image is None:
            raise Jbig2UnsupportedError("JBIG2Decode produced no image")
        return jbig2_bitmap_to_pdf_image(bytes(self.image.data))


def jbig2_page_default_pixel(page_info: JBIG2PageInfo) -> int:
    return (page_info.flags >> 2) & 1


def jbig2_page_combination_operator(page_info: JBIG2PageInfo | None) -> int:
    if page_info is None:
        return 0
    return (page_info.flags >> 3) & 3


def jbig2_page_allows_region_operator(page_info: JBIG2PageInfo | None) -> bool:
    return page_info is not None and bool(page_info.flags & 64)


def jbig2_bitmap_to_pdf_image(data: bytes | bytearray) -> bytes:
    if isinstance(data, bytearray):
        image = internal_uint8_view(data)
        numpy.bitwise_xor(image, 0xFF, out=image)
        return bytes(data)
    if len(data) < 4096:
        return bytes(byte ^ 0xFF for byte in data)
    return numpy.bitwise_xor(internal_uint8_view(data), 0xFF).tobytes()


def decode_arithmetic_generic_bitmap(
    data: bytes,
    width: int,
    height: int,
    template: int,
    prediction: bool,
    at: tuple[tuple[int, int], ...],
) -> bytes | bytearray:
    if (
        template != 0
        or prediction
        or at != GENERIC_TEMPLATE_0_DEFAULT_AT
        or width <= 0
        or height <= 0
    ):
        raise Jbig2UnsupportedError("unsupported JBIG2 generic bitmap template")
    return decode_arithmetic_generic_template0(data, width, height)


def decode_arithmetic_generic_template0(data: bytes, width: int, height: int) -> bytearray:
    decoder = JBIG2MQDecoder(data)
    contexts = [0] * 65536
    row_byte_length = (width + 7) // 8
    bitmap = bytearray(row_byte_length * height)
    previous_row = bytearray(width + 4)
    previous_previous_row = bytearray(width + 4)
    old_pixel_mask = 0x7BF7
    a = decoder.a
    chigh = decoder.chigh
    clow = decoder.clow
    ct = decoder.ct
    bp = decoder.bp
    data_end = decoder.data_end
    for row_index in range(height):
        # Four sentinel bytes eliminate bounds checks for the look-ahead
        # samples at col + 3 and col + 4 in the template-0 context.
        row = bytearray(width + 4)
        row1 = row if row_index < 1 else previous_row
        row2 = row if row_index < 2 else previous_previous_row
        context = (
            (row2[0] << 13)
            | (row2[1] << 12)
            | (row2[2] << 11)
            | (row1[0] << 7)
            | (row1[1] << 6)
            | (row1[2] << 5)
            | (row1[3] << 4)
        )
        for col in range(width):
            packed = contexts[context]
            idx = packed >> 1
            mps = packed & 1
            qe = MQ_QE[idx]
            next_a = a - qe
            if chigh < qe:
                if next_a < qe:
                    next_a = qe
                    pixel = mps
                    idx = MQ_NMPS[idx]
                else:
                    next_a = qe
                    pixel = 1 ^ mps
                    if MQ_SWITCH[idx]:
                        mps = pixel
                    idx = MQ_NLPS[idx]
            else:
                chigh -= qe
                if next_a & 0x8000:
                    a = next_a
                    contexts[context] = (idx << 1) | mps
                    pixel = mps
                    row[col] = pixel
                    if pixel:
                        bitmap[row_index * row_byte_length + (col >> 3)] |= 0x80 >> (col & 7)
                    context = (
                        ((context & old_pixel_mask) << 1)
                        | (row2[col + 3] << 11)
                        | (row1[col + 4] << 4)
                        | pixel
                    )
                    continue
                if next_a < qe:
                    pixel = 1 ^ mps
                    if MQ_SWITCH[idx]:
                        mps = pixel
                    idx = MQ_NLPS[idx]
                else:
                    pixel = mps
                    idx = MQ_NMPS[idx]
            while not (next_a & 0x8000):
                if ct == 0:
                    current = data[bp] if bp < data_end else 0xFF
                    following = data[bp + 1] if bp + 1 < data_end else 0xFF
                    if current == 0xFF:
                        if following > 0x8F:
                            clow += 0xFF00
                            ct = 8
                        else:
                            bp += 1
                            value = data[bp] if bp < data_end else 0xFF
                            clow += value << 9
                            ct = 7
                    else:
                        bp += 1
                        value = data[bp] if bp < data_end else 0xFF
                        clow += value << 8
                        ct = 8
                    if clow > 0xFFFF:
                        chigh += clow >> 16
                        clow &= 0xFFFF
                next_a <<= 1
                chigh = ((chigh << 1) & 0xFFFF) | ((clow >> 15) & 1)
                clow = (clow << 1) & 0xFFFF
                ct -= 1
            a = next_a
            contexts[context] = (idx << 1) | mps
            row[col] = pixel
            if pixel:
                bitmap[row_index * row_byte_length + (col >> 3)] |= 0x80 >> (col & 7)
            context = (
                ((context & old_pixel_mask) << 1)
                | (row2[col + 3] << 11)
                | (row1[col + 4] << 4)
                | pixel
            )
        previous_previous_row = previous_row
        previous_row = row
    return bitmap


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
        return decoder.finish()
    except Jbig2UnsupportedError as exc:
        raise FilterUnsupportedError(str(exc)) from exc
    except Jbig2ParseError as exc:
        raise FilterParseError(str(exc)) from exc


__all__ = (
    "JBIG2Region",
    "JBIG2GenericRegionHeader",
    "JBIG2PageDecoder",
    "parse_region",
    "parse_embedded_segments",
    "decode_jbig2",
    "JBIG2_PAGE_INFO",
    "JBIG2_END_OF_FILE",
    "JBIG2_IMMEDIATE_GENERIC_REGION",
    "JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION",
    "Jbig2Error",
    "Jbig2ParseError",
    "Jbig2UnsupportedError",
    "GENERIC_TEMPLATE_0_DEFAULT_AT",
    "MQ_QE",
    "MQ_NMPS",
    "MQ_NLPS",
    "MQ_SWITCH",
    "JBIG2Segment",
    "JBIG2PageInfo",
    "JBIG2SegmentHeader",
    "JBIG2Image",
    "JBIG2MQDecoder",
    "read_be_u32",
    "read_be_i32",
    "read_be_i8",
    "parse_page_info",
    "parse_generic_region_header",
    "read_u8",
    "read_u32",
    "read_u24",
    "parse_page_association",
    "parse_referred_to_segments",
    "parse_segment_header",
    "jbig2_page_default_pixel",
    "jbig2_page_combination_operator",
    "jbig2_page_allows_region_operator",
    "jbig2_bitmap_to_pdf_image",
    "decode_arithmetic_generic_bitmap",
    "decode_arithmetic_generic_template0",
    "compose_packed_bitmap_region",
    "region_operator",
)
