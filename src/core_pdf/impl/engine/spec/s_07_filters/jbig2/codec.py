"""JBIG2 parsing and decoding codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from core_pdf.impl.engine.spec.s_07_filters.jbig2.bitmap_kernels import compose_packed_bitmap_data
from core_pdf.impl.engine.spec.s_07_filters.jbig2.buffer_views import uint8_matrix_view, uint8_view

JBIG2_FILE_HEADER = b"\x97JB2\r\n\x1a\n"
JBIG2_PAGE_INFO = 48
JBIG2_END_OF_PAGE = 49
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

MQ_QE = (
    0x5601,
    0x3401,
    0x1801,
    0x0AC1,
    0x0521,
    0x0221,
    0x5601,
    0x5401,
    0x4801,
    0x3801,
    0x3001,
    0x2401,
    0x1C01,
    0x1601,
    0x5601,
    0x5401,
    0x5101,
    0x4801,
    0x3801,
    0x3401,
    0x3001,
    0x2801,
    0x2401,
    0x2201,
    0x1C01,
    0x1801,
    0x1601,
    0x1401,
    0x1201,
    0x1101,
    0x0AC1,
    0x09C1,
    0x08A1,
    0x0521,
    0x0441,
    0x02A1,
    0x0221,
    0x0141,
    0x0111,
    0x0085,
    0x0049,
    0x0025,
    0x0015,
    0x0009,
    0x0005,
    0x0001,
    0x5601,
)
MQ_NMPS = (
    1,
    2,
    3,
    4,
    5,
    38,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    29,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    45,
    46,
)
MQ_NLPS = (
    1,
    6,
    9,
    12,
    29,
    33,
    6,
    14,
    14,
    14,
    17,
    18,
    20,
    21,
    14,
    14,
    15,
    16,
    17,
    18,
    19,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
)
MQ_SWITCH = (
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)


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


@dataclass(slots=True)
class JBIG2SymbolDictionary:
    sbatx: int
    sbridge: int
    referred_to_count: int
    referred_to_segments: list[int]
    flags: int
    raw: bytes


@dataclass(slots=True)
class JBIG2TextRegion:
    width: int
    height: int
    x: int
    y: int
    flags: int
    raw: bytes


@dataclass(slots=True)
class JBIG2GenericRegion:
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
    def create(cls, width: int, height: int) -> "JBIG2Image":
        if width <= 0 or height <= 0:
            raise Jbig2ParseError("invalid JBIG2 image dimensions")
        stride = (width + 7) // 8
        return cls(width=width, height=height, stride=stride, data=bytearray(stride * height))

    def fill(self, value: int) -> None:
        fill_byte = 0xFF if value else 0x00
        self.data[:] = bytes([fill_byte]) * len(self.data)


class JBIG2MQDecoder:
    __slots__ = ("data", "bp", "data_end", "a", "chigh", "clow", "ct", "ctx")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bp = 0
        self.data_end = len(data)
        self.chigh = data[0] if data else 0xFF
        self.clow = 0
        self.ct = 0
        self.ctx: dict[int, int] = {}
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


def read_be_i16(data: bytes, pos: int) -> int:
    value = (data[pos] << 8) | data[pos + 1]
    return value - 0x10000 if value & 0x8000 else value


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


def parse_symbol_dictionary(data: bytes) -> JBIG2SymbolDictionary:
    if len(data) < 8:
        raise Jbig2ParseError("truncated JBIG2 symbol dictionary")
    sbatx = read_be_i16(data, 0)
    sbridge = read_be_i16(data, 2)
    referred_to_count = data[4]
    if len(data) < 6 + referred_to_count:
        raise Jbig2ParseError("truncated JBIG2 symbol dictionary")
    referred_to_segments = list(data[5 : 5 + referred_to_count])
    flags = data[5 + referred_to_count] if 5 + referred_to_count < len(data) else 0
    if 5 + referred_to_count >= len(data):
        raise Jbig2ParseError("truncated JBIG2 symbol dictionary")
    return JBIG2SymbolDictionary(
        sbatx=sbatx,
        sbridge=sbridge,
        referred_to_count=referred_to_count,
        referred_to_segments=referred_to_segments,
        flags=flags,
        raw=data,
    )


def parse_text_region(data: bytes) -> JBIG2TextRegion:
    if len(data) < 17:
        raise Jbig2ParseError("truncated JBIG2 text region")
    width = read_be_u32(data, 0)
    height = read_be_u32(data, 4)
    x = read_be_i32(data, 8)
    y = read_be_i32(data, 12)
    flags = data[16]
    return JBIG2TextRegion(width=width, height=height, x=x, y=y, flags=flags, raw=data)


def parse_generic_region(data: bytes) -> JBIG2GenericRegion:
    if len(data) < 17:
        raise Jbig2ParseError("truncated JBIG2 generic region")
    width = read_be_u32(data, 0)
    height = read_be_u32(data, 4)
    x = read_be_i32(data, 8)
    y = read_be_i32(data, 12)
    flags = data[16]
    return JBIG2GenericRegion(width=width, height=height, x=x, y=y, flags=flags, raw=data)


def parse_generic_region_header(
    data: bytes,
) -> tuple[JBIG2GenericRegion, bool, int, bool, tuple[tuple[int, int], ...], int]:
    region = parse_generic_region(data)
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
    return region, mmr, template, prediction, tuple(at), pos


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
    segments: list[int] = []
    for ignored in range(count):
        value, pos = read_u32(data, pos) if long_form else read_u8(data, pos)
        segments.append(value)
    return segments, pos


def parse_segment_header(data: bytes, pos: int) -> tuple[JBIG2SegmentHeader, int]:
    start = pos
    if pos + 10 > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment header")
    number, pos = read_u32(data, pos)
    flags, pos = read_u8(data, pos)
    retention_flags, pos = read_u8(data, pos)
    referred_to_count = retention_flags >> 5
    referred_to_segments: list[int] = []
    if referred_to_count == 7:
        if pos + 3 > len(data):
            raise Jbig2ParseError("truncated JBIG2 segment header")
        ref_count, pos = read_u24(data, pos)
        referred_to_count = ref_count & 0x1FFFFFFF
        bit_bytes = (referred_to_count + 1 + 7) // 8
        if pos + bit_bytes > len(data):
            raise Jbig2ParseError("truncated JBIG2 segment header")
        pos += bit_bytes
    else:
        referred_to_segments, pos = parse_referred_to_segments(
            data, pos, referred_to_count, number > 65536
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


def parse_jbig2_file(data: bytes) -> list[JBIG2Segment]:
    if data.startswith(JBIG2_FILE_HEADER):
        pos = len(JBIG2_FILE_HEADER)
        if pos >= len(data):
            return []
        pos += 1
        ignored, pos = read_u32(data, pos)
    else:
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


def decode_jbig2_segments(segments: list[JBIG2Segment]) -> bytes:
    page_info: JBIG2PageInfo | None = None
    image: JBIG2Image | None = None
    symbol_dictionaries: dict[int, JBIG2SymbolDictionary] = {}
    max_x = 0
    max_y = 0
    inferred_width = 0
    inferred_height = 0

    def ensure_image(width: int, height: int) -> None:
        nonlocal image, inferred_width, inferred_height
        if width <= 0 or height <= 0:
            return
        if image is None:
            image = JBIG2Image.create(width, height)
            inferred_width = width
            inferred_height = height
            return
        if width <= inferred_width and height <= inferred_height:
            return
        new_width = max(width, inferred_width)
        new_height = max(height, inferred_height)
        new_image = JBIG2Image.create(new_width, new_height)
        source = uint8_matrix_view(image.data, image.height, image.stride)
        destination = uint8_matrix_view(new_image.data, new_image.height, new_image.stride)
        destination[: image.height, : image.stride] = source
        image = new_image
        inferred_width = new_width
        inferred_height = new_height

    for segment in segments:
        if segment.segment_type == JBIG2_PAGE_INFO:
            page_info = parse_page_info(segment.data)
            image = JBIG2Image.create(page_info.width, page_info.height)
            image.fill(jbig2_page_default_pixel(page_info))
            inferred_width = page_info.width
            inferred_height = page_info.height
        elif segment.segment_type == 0:
            symbol_dictionaries[segment.number] = parse_symbol_dictionary(segment.data)
        elif segment.segment_type == 6:
            text_region = parse_text_region(segment.data)
            max_x = max(max_x, text_region.x + text_region.width)
            max_y = max(max_y, text_region.y + text_region.height)
            ensure_image(max_x, max_y)
            if image is not None:
                decode_text_region(segment.data, image, symbol_dictionaries)
        elif segment.segment_type in {
            JBIG2_IMMEDIATE_GENERIC_REGION,
            JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION,
        }:
            generic_region = parse_generic_region(segment.data)
            max_x = max(max_x, generic_region.x + generic_region.width)
            max_y = max(max_y, generic_region.y + generic_region.height)
            ensure_image(max_x, max_y)
            if image is not None:
                decode_generic_region(segment.data, image, page_info)
    if image is None:
        raise Jbig2UnsupportedError("JBIG2Decode produced no image")
    return jbig2_bitmap_to_pdf_image(image.data)


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
        image = uint8_view(data)
        numpy.bitwise_xor(image, 0xFF, out=image)
        return bytes(data)
    if len(data) < 4096:
        return bytes(byte ^ 0xFF for byte in data)
    return numpy.bitwise_xor(uint8_view(data), 0xFF).tobytes()


def decode_text_region(
    data: bytes, image: JBIG2Image, symbols: dict[int, JBIG2SymbolDictionary]
) -> None:

    region = parse_text_region(data)
    if len(region.raw) < 20:
        raise Jbig2ParseError("truncated JBIG2 text region")
    x = region.x
    y = region.y
    width = region.width
    height = region.height
    bitmap = region.raw[20:]
    row_bytes = max(1, (width + 7) // 8)
    compose_packed_bitmap_data(
        bitmap,
        min(height, len(bitmap) // row_bytes),
        width,
        x,
        y,
        image.width,
        image.height,
        image.stride,
        image.data,
        0,
    )


def decode_generic_region(
    data: bytes, image: JBIG2Image, page_info: JBIG2PageInfo | None = None
) -> None:
    region, mmr, template, prediction, at, bitmap_start = parse_generic_region_header(data)
    if region.width <= 0 or region.height <= 0:
        return
    if mmr:
        packed_bitmap = region.raw[bitmap_start:]
        compose_packed_bitmap_region(region, packed_bitmap, image, page_info)
        return
    packed_bitmap = decode_arithmetic_generic_bitmap(
        region.raw[bitmap_start:], region.width, region.height, template, prediction, at
    )
    compose_packed_bitmap_region(region, packed_bitmap, image, page_info)


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
    decoder.a = a
    decoder.chigh = chigh
    decoder.clow = clow
    decoder.ct = ct
    decoder.bp = bp
    return bitmap


def compose_packed_bitmap_region(
    region: JBIG2GenericRegion,
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


def region_operator(region: JBIG2GenericRegion, page_info: JBIG2PageInfo | None) -> int:
    if jbig2_page_allows_region_operator(page_info):
        return region.flags & 7
    return jbig2_page_combination_operator(page_info)


def assemble_embedded_jbig2(globals_data: bytes, page_data: bytes) -> bytes:
    parts = [JBIG2_FILE_HEADER, b"\x01", (1).to_bytes(4, "big")]
    if globals_data:
        parts.append(globals_data)
    parts.append(page_data)
    return b"".join(parts)


def decode_embedded_jbig2(data: bytes) -> bytes:
    return decode_jbig2_segments(parse_jbig2_file(data))


__all__ = (
    "JBIG2Image",
    "JBIG2Segment",
    "JBIG2_FILE_HEADER",
    "Jbig2Error",
    "Jbig2ParseError",
    "Jbig2UnsupportedError",
    "assemble_embedded_jbig2",
    "decode_embedded_jbig2",
    "parse_jbig2_file",
)
