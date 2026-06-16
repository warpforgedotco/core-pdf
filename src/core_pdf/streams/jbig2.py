from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_pdf.syntax.errors import PdfParseError, PdfUnsupportedError

JBIG2_FILE_HEADER = b"\x97JB2\r\n\x1a\n"
JBIG2_PAGE_INFO = 48
JBIG2_END_OF_PAGE = 49
JBIG2_END_OF_FILE = 51

_MQ_QE = (
    0x5601, 0x3401, 0x1801, 0x0AC1, 0x0521, 0x0221, 0x5601, 0x5401,
    0x4801, 0x3801, 0x3001, 0x2401, 0x1C01, 0x1601, 0x5601, 0x5401,
    0x5101, 0x4801, 0x3801, 0x3401, 0x3001, 0x2801, 0x2401, 0x2201,
    0x1C01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101, 0x0AC1, 0x09C1,
    0x08A1, 0x0521, 0x0441, 0x02A1, 0x0221, 0x0141, 0x0111, 0x0085,
    0x0049, 0x0025, 0x0015, 0x0009, 0x0005, 0x0001, 0x5601,
)
_MQ_NMPS = (
    1, 2, 3, 4, 5, 38, 7, 8, 9, 10, 11, 12, 13, 29, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 45, 46,
)
_MQ_NLPS = (
    1, 6, 9, 12, 29, 33, 6, 14, 14, 14, 17, 18, 20, 21, 14, 14,
    15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46,
)
_MQ_SWITCH = (
    1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
            raise PdfParseError("invalid JBIG2 image dimensions")
        stride = (width + 7) // 8
        return cls(width=width, height=height, stride=stride, data=bytearray(stride * height))

    def set_pixel(self, x: int, y: int, value: int) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        idx = y * self.stride + (x >> 3)
        mask = 0x80 >> (x & 7)
        if value:
            self.data[idx] |= mask
        else:
            self.data[idx] &= ~mask


@dataclass(slots=True)
class JBIG2BitReader:
    data: bytes
    bit_pos: int = 0

    @property
    def remaining_bits(self) -> int:
        return max(0, len(self.data) * 8 - self.bit_pos)

    def peek(self, width: int) -> int:
        if width <= 0:
            raise ValueError("invalid bit width")
        if self.bit_pos + width > len(self.data) * 8:
            raise EOFError("insufficient JBIG2 data")
        byte_pos = self.bit_pos >> 3
        bit_offset = self.bit_pos & 7
        needed = (width + bit_offset + 7) >> 3
        word = 0
        for i in range(needed):
            word = (word << 8) | self.data[byte_pos + i]
        shift = needed * 8 - width - bit_offset
        return (word >> shift) & ((1 << width) - 1)

    def advance(self, width: int) -> None:
        if width < 0 or self.bit_pos + width > len(self.data) * 8:
            raise EOFError("insufficient JBIG2 data")
        self.bit_pos += width


class JBIG2MQDecoder:
    __slots__ = ("data", "pos", "_a", "_c", "ct", "ctx")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self._a = 0x8000
        self._c = 0
        self.ct = 0
        self.ctx = [(0, 0)] * 19
        self.init_bytes()

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            return 0xFF
        b = self.data[self.pos]
        self.pos += 1
        return b

    def init_bytes(self) -> None:
        b = self.read_byte()
        if b == 0xFF:
            b = self.read_byte()
        self._c = b << 8
        b = self.read_byte()
        if b == 0xFF:
            b = self.read_byte()
        self._c |= b
        self.ct = 0

    def byte_in(self) -> None:
        b = self.read_byte()
        if b == 0xFF:
            _ = self.read_byte()
        self._c |= b << self.ct
        self.ct = 8

    def renorm(self) -> None:
        while self._a < 0x8000:
            if self.ct == 0:
                self.byte_in()
            self._c <<= 1
            self._a <<= 1
            self.ct -= 1

    def decode_bit(self, cx: int) -> int:
        idx, mps = self.ctx[cx]
        qe = _MQ_QE[idx]
        self._a -= qe
        if (self._c >> 16) < qe:
            if self._a < qe:
                self._a = qe
            else:
                mps ^= 1
            self.ctx[cx] = (_MQ_NLPS[idx], mps ^ _MQ_SWITCH[idx])
            self.renorm()
            return mps ^ 1
        self._c -= qe << 16
        if self._a < 0x8000:
            self.ctx[cx] = (_MQ_NMPS[idx], mps)
            self.renorm()
        return mps


class JBIG2ContextCache:
    __slots__ = ("cache",)

    def __init__(self) -> None:
        self.cache: dict[Any, dict[int, tuple[int, int]]] = {}

    def get_contexts(self, key: Any) -> dict[int, tuple[int, int]]:
        contexts = self.cache.get(key)
        if contexts is None:
            contexts = {}
            self.cache[key] = contexts
        return contexts


class JBIG2DecodingContext:
    __slots__ = ("data", "start", "end", "_decoder", "cache")

    def __init__(self, data: bytes, start: int = 0, end: int | None = None) -> None:
        self.data = data
        self.start = start
        self.end = len(data) if end is None else end
        self._decoder: JBIG2MQDecoder | None = None
        self.cache: JBIG2ContextCache | None = None

    @property
    def decoder(self) -> JBIG2MQDecoder:
        if self._decoder is None:
            self._decoder = JBIG2MQDecoder(self.data[self.start : self.end])
        return self._decoder

    @property
    def context_cache(self) -> JBIG2ContextCache:
        if self.cache is None:
            self.cache = JBIG2ContextCache()
        return self.cache


def decode_integer(context_cache: JBIG2ContextCache, procedure: Any, decoder: JBIG2MQDecoder) -> int | None:
    prev = 1

    def read_bits(length: int) -> int:
        nonlocal prev
        value = 0
        for _ in range(length):
            bit = decoder.decode_bit(prev)
            prev = prev < 256 and ((prev << 1) | bit) or ((((prev << 1) | bit) & 0x1FF) | 0x100)
            value = (value << 1) | bit
        return value

    sign = read_bits(1)
    if read_bits(1):
        if read_bits(1):
            if read_bits(1):
                if read_bits(1):
                    if read_bits(1):
                        value = read_bits(32) + 4436
                    else:
                        value = read_bits(12) + 340
                else:
                    value = read_bits(8) + 84
            else:
                value = read_bits(6) + 20
        else:
            value = read_bits(4) + 4
    else:
        value = read_bits(2)
    signed_value = value if sign == 0 else -value if value > 0 else 0
    return signed_value


def decode_iaid(context_cache: JBIG2ContextCache, decoder: JBIG2MQDecoder, code_length: int) -> int:
    prev = 1
    value = 0
    for _ in range(code_length):
        bit = decoder.decode_bit(prev)
        prev = (prev << 1) | bit
        value = (value << 1) | bit
    if code_length < 31:
        return prev & ((1 << code_length) - 1)
    return prev & 0x7FFFFFFF


def read_u16(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 2 > len(data):
        raise PdfParseError("truncated JBIG2 data")
    return int.from_bytes(data[pos : pos + 2], "big"), pos + 2


def parse_page_info(data: bytes) -> JBIG2PageInfo:
    if len(data) < 19:
        raise PdfParseError("truncated JBIG2 page info")
    width = int.from_bytes(data[0:4], "big")
    height = int.from_bytes(data[4:8], "big")
    x_resolution = int.from_bytes(data[8:12], "big")
    y_resolution = int.from_bytes(data[12:16], "big")
    flags = int.from_bytes(data[16:17], "big")
    return JBIG2PageInfo(width=width, height=height, x_resolution=x_resolution, y_resolution=y_resolution, flags=flags)


def parse_symbol_dictionary(data: bytes) -> JBIG2SymbolDictionary:
    if len(data) < 8:
        raise PdfParseError("truncated JBIG2 symbol dictionary")
    sbatx = int.from_bytes(data[0:2], "big", signed=True)
    sbridge = int.from_bytes(data[2:4], "big", signed=True)
    referred_to_count = data[4]
    if len(data) < 6 + referred_to_count:
        raise PdfParseError("truncated JBIG2 symbol dictionary")
    referred_to_segments = list(data[5 : 5 + referred_to_count])
    flags = data[5 + referred_to_count] if 5 + referred_to_count < len(data) else 0
    if 5 + referred_to_count >= len(data):
        raise PdfParseError("truncated JBIG2 symbol dictionary")
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
        raise PdfParseError("truncated JBIG2 text region")
    width = int.from_bytes(data[0:4], "big")
    height = int.from_bytes(data[4:8], "big")
    x = int.from_bytes(data[8:12], "big", signed=True)
    y = int.from_bytes(data[12:16], "big", signed=True)
    flags = data[16]
    return JBIG2TextRegion(width=width, height=height, x=x, y=y, flags=flags, raw=data)


def parse_generic_region(data: bytes) -> JBIG2GenericRegion:
    if len(data) < 17:
        raise PdfParseError("truncated JBIG2 generic region")
    width = int.from_bytes(data[0:4], "big")
    height = int.from_bytes(data[4:8], "big")
    x = int.from_bytes(data[8:12], "big", signed=True)
    y = int.from_bytes(data[12:16], "big", signed=True)
    flags = data[16]
    return JBIG2GenericRegion(width=width, height=height, x=x, y=y, flags=flags, raw=data)


def read_u8(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 1 > len(data):
        raise PdfParseError("truncated JBIG2 data")
    return data[pos], pos + 1


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(data):
        raise PdfParseError("truncated JBIG2 data")
    return int.from_bytes(data[pos : pos + 4], "big"), pos + 4


def read_u24(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 3 > len(data):
        raise PdfParseError("truncated JBIG2 data")
    return int.from_bytes(data[pos : pos + 3], "big"), pos + 3


def parse_page_association(data: bytes, pos: int, long_form: bool) -> tuple[int, int]:
    if long_form:
        return read_u32(data, pos)
    return read_u8(data, pos)


def parse_referred_to_segments(
    data: bytes, pos: int, count: int, long_form: bool
) -> tuple[list[int], int]:
    segments: list[int] = []
    for _ in range(count):
        value, pos = read_u32(data, pos) if long_form else read_u8(data, pos)
        segments.append(value)
    return segments, pos


def parse_segment_header(data: bytes, pos: int) -> tuple[JBIG2SegmentHeader, int]:
    start = pos
    if pos + 10 > len(data):
        raise PdfParseError("truncated JBIG2 segment header")
    number, pos = read_u32(data, pos)
    flags, pos = read_u8(data, pos)
    retention_flags, pos = read_u8(data, pos)
    referred_to_count = retention_flags >> 5
    referred_to_segments: list[int] = []
    if referred_to_count == 7:
        if pos + 3 > len(data):
            raise PdfParseError("truncated JBIG2 segment header")
        ref_count, pos = read_u24(data, pos)
        referred_to_count = ref_count & 0x1FFFFFFF
        bit_bytes = (referred_to_count + 1 + 7) // 8
        if pos + bit_bytes > len(data):
            raise PdfParseError("truncated JBIG2 segment header")
        pos += bit_bytes
    else:
        referred_to_segments, pos = parse_referred_to_segments(data, pos, referred_to_count, number > 65536)
    if pos + (4 if (flags & 0x40) else 1) + 4 > len(data):
        raise PdfParseError("truncated JBIG2 segment header")
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
        pos += 1  # file header flags
        _, pos = read_u32(data, pos)  # number of pages
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
        for row in range(image.height):
            src = row * image.stride
            dst = row * new_image.stride
            new_image.data[dst : dst + image.stride] = image.data[src : src + image.stride]
        image = new_image
        inferred_width = new_width
        inferred_height = new_height

    for segment in segments:
        if segment.segment_type == JBIG2_PAGE_INFO:
            page_info = parse_page_info(segment.data)
            image = JBIG2Image.create(page_info.width, page_info.height)
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
        elif segment.segment_type == 50:
            generic_region = parse_generic_region(segment.data)
            max_x = max(max_x, generic_region.x + generic_region.width)
            max_y = max(max_y, generic_region.y + generic_region.height)
            ensure_image(max_x, max_y)
            if image is not None:
                decode_generic_region(segment.data, image)
    if image is None:
        raise PdfUnsupportedError("JBIG2Decode produced no image")
    return bytes(image.data)


def decode_text_region(
    data: bytes, image: JBIG2Image, symbols: dict[int, JBIG2SymbolDictionary]
) -> None:
    # Minimal placeholder decoder for the common case: a single bitmap per text region.
    # This keeps the implementation pure Python and opens the door to real symbol decoding.
    region = parse_text_region(data)
    if len(region.raw) < 20:
        raise PdfParseError("truncated JBIG2 text region")
    x = region.x
    y = region.y
    width = region.width
    height = region.height
    bitmap = region.raw[20:]
    row_bytes = max(1, (width + 7) // 8)
    for row in range(min(height, len(bitmap) // row_bytes)):
        src = bitmap[row * row_bytes : (row + 1) * row_bytes]
        for col in range(width):
            if src[col >> 3] & (0x80 >> (col & 7)):
                image.set_pixel(x + col, y + row, 1)


def decode_generic_region(data: bytes, image: JBIG2Image) -> None:
    region = parse_generic_region(data)
    if len(region.raw) < 20:
        raise PdfParseError("truncated JBIG2 generic region")
    if region.width <= 0 or region.height <= 0:
        return
    bitmap = region.raw[20:]
    row_bytes = max(1, (region.width + 7) // 8)
    for row in range(min(region.height, len(bitmap) // row_bytes)):
        src = bitmap[row * row_bytes : (row + 1) * row_bytes]
        for col in range(region.width):
            if src[col >> 3] & (0x80 >> (col & 7)):
                image.set_pixel(region.x + col, region.y + row, 1)


def assemble_embedded_jbig2(globals_data: bytes, page_data: bytes) -> bytes:
    parts = [JBIG2_FILE_HEADER, b"\x01", (1).to_bytes(4, "big")]
    if globals_data:
        parts.append(globals_data)
    parts.append(page_data)
    return b"".join(parts)


def decode_embedded_jbig2(data: bytes) -> bytes:
    return decode_jbig2_segments(parse_jbig2_file(data))
