"""JBIG2 parsing and decoding codec."""

from __future__ import annotations

import numpy

from core_pdf.impl.spec.s_07_filters.jbig2.arithmetic import decode_arithmetic_generic_bitmap
from core_pdf.impl.spec.s_07_filters.jbig2.bitmap_kernels import (
    compose_packed_bitmap_data,
    uint8_matrix_view,
    uint8_view,
)
from core_pdf.impl.spec.s_07_filters.jbig2.structures import (
    MAX_REFERENCES,
    MAX_SYMBOLS,
    Jbig2Error,
    JBIG2Image,
    JBIG2PageInfo,
    Jbig2ParseError,
    JBIG2Region,
    JBIG2Segment,
    JBIG2SegmentHeader,
    Jbig2UnsupportedError,
    read_be_i8,
    read_be_i32,
    read_be_u32,
)
from core_pdf.impl.spec.s_07_filters.jbig2.symbols import (
    decode_symbol_dictionary,
    decode_text_bitmap,
)

JBIG2_FILE_HEADER = b"\x97JB2\r\n\x1a\n"
JBIG2_PAGE_INFO = 48
JBIG2_END_OF_FILE = 51
JBIG2_IMMEDIATE_GENERIC_REGION = 38
JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION = 39


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


def internal_region_fields(data: bytes, kind: str) -> tuple[int, int, int, int, int]:
    """Read the region segment information field shared by every region type."""
    if len(data) < 17:
        raise Jbig2ParseError(f"truncated JBIG2 {kind} region")
    return (
        read_be_u32(data, 0),
        read_be_u32(data, 4),
        read_be_i32(data, 8),
        read_be_i32(data, 12),
        data[16],
    )


def parse_region(data: bytes) -> JBIG2Region:
    width, height, x, y, flags = internal_region_fields(data, "generic")
    return JBIG2Region(width=width, height=height, x=x, y=y, flags=flags, raw=data)


def parse_generic_region_header(
    data: bytes,
) -> tuple[JBIG2Region, bool, int, bool, tuple[tuple[int, int], ...], int]:
    region = parse_region(data)
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
    data: bytes, pos: int, count: int, number: int
) -> tuple[list[int], int]:
    width = 1 if number <= 256 else 2 if number <= 65536 else 4
    if count > (len(data) - pos) // width:
        raise Jbig2ParseError("truncated JBIG2 segment references")
    if count > MAX_REFERENCES:
        raise Jbig2UnsupportedError("JBIG2 segment exceeds supported reference limit")
    segments: list[int] = []
    for ignored in range(count):
        value = int.from_bytes(data[pos : pos + width], "big")
        pos += width
        if value >= number:
            raise Jbig2ParseError("JBIG2 segment reference must precede its referring segment")
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
        referred_to_count = ((retention_flags & 31) << 24) | ref_count
        bit_bytes = (referred_to_count + 1 + 7) // 8
        if pos + bit_bytes > len(data):
            raise Jbig2ParseError("truncated JBIG2 segment header")
        pos += bit_bytes
    elif referred_to_count > 4:
        raise Jbig2ParseError("invalid JBIG2 segment reference count")
    referred_to_segments, pos = parse_referred_to_segments(data, pos, referred_to_count, number)
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
        flags, pos = read_u8(data, pos)
        if not flags & 1:
            raise Jbig2UnsupportedError("JBIG2 random-access file organization is unsupported")
        if flags & 0xFC:
            raise Jbig2ParseError("invalid JBIG2 file header flags")
        if not flags & 2:
            ignored, pos = read_u32(data, pos)
    else:
        pos = 0

    segments: list[JBIG2Segment] = []
    while pos < len(data):
        header, pos = parse_segment_header(data, pos)
        if header.data_length == 0xFFFFFFFF:
            raise Jbig2UnsupportedError("JBIG2 regions with unknown data length are unsupported")
        if header.data_length > len(data) - pos:
            raise Jbig2ParseError("truncated JBIG2 segment data")
        payload = data[pos : pos + header.data_length]
        pos += header.data_length
        segments.append(
            JBIG2Segment(
                number=header.number,
                flags=header.flags,
                retention_flags=header.retention_flags,
                page_association=header.page_association,
                data=payload,
                referred_to_segments=tuple(header.referred_to_segments),
            )
        )
        if header.flags & 0x3F == JBIG2_END_OF_FILE:
            break
    return segments


def decode_jbig2_segments(segments: list[JBIG2Segment]) -> bytes:
    page_info: JBIG2PageInfo | None = None
    image: JBIG2Image | None = None
    max_x = 0
    max_y = 0
    inferred_width = 0
    inferred_height = 0
    decoded_segments: dict[int, JBIG2Segment] = {}
    dictionaries: dict[int, list[JBIG2Image]] = {}

    def ensure_image(width: int, height: int) -> None:
        nonlocal image, inferred_width, inferred_height
        if width <= 0 or height <= 0:
            return
        if page_info is not None:
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
        if segment.number in decoded_segments:
            raise Jbig2ParseError("duplicate JBIG2 segment number")
        imported: list[JBIG2Image] = []
        for reference_id in segment.referred_to_segments:
            reference = decoded_segments.get(reference_id)
            if reference is None or reference_id >= segment.number:
                raise Jbig2ParseError("missing or invalid JBIG2 segment reference")
            if reference.page_association not in (0, segment.page_association):
                raise Jbig2ParseError("JBIG2 segment refers to a different page")
            if segment.segment_type == 62:
                continue
            if reference_id in dictionaries:
                if len(imported) + len(dictionaries[reference_id]) > MAX_SYMBOLS:
                    raise Jbig2UnsupportedError("JBIG2 segment exceeds supported symbol limit")
                imported.extend(dictionaries[reference_id])
            else:
                raise Jbig2UnsupportedError("unsupported JBIG2 non-dictionary segment reference")
        decoded_segments[segment.number] = segment
        if segment.segment_type == JBIG2_PAGE_INFO:
            if page_info is not None:
                raise Jbig2UnsupportedError("multiple JBIG2 pages in one image are unsupported")
            page_info = parse_page_info(segment.data)
            image = JBIG2Image.create(page_info.width, page_info.height)
            image.fill(jbig2_page_default_pixel(page_info))
            inferred_width = page_info.width
            inferred_height = page_info.height
        elif segment.segment_type == 0:
            dictionaries[segment.number] = decode_symbol_dictionary(segment.data, imported)
        elif segment.segment_type in {6, 7}:
            text_region = parse_region(segment.data)
            max_x = max(max_x, text_region.x + text_region.width)
            max_y = max(max_y, text_region.y + text_region.height)
            ensure_image(max_x, max_y)
            bitmap = decode_text_bitmap(segment.data, imported)
            if image is not None:
                compose_packed_bitmap_region(text_region, bitmap.data, image, page_info)
        elif segment.segment_type in {
            JBIG2_IMMEDIATE_GENERIC_REGION,
            JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION,
        }:
            generic_region = parse_region(segment.data)
            max_x = max(max_x, generic_region.x + generic_region.width)
            max_y = max(max_y, generic_region.y + generic_region.height)
            ensure_image(max_x, max_y)
            if image is not None:
                decode_generic_region(segment.data, image, page_info)
        elif segment.segment_type == 62:
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 extension segment")
            if read_be_u32(segment.data, 0) & 0x80000000:
                raise Jbig2UnsupportedError("unsupported mandatory JBIG2 extension")
        elif segment.segment_type not in {49, 50, 51, 52}:
            raise Jbig2UnsupportedError(f"unsupported JBIG2 segment type {segment.segment_type}")
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


def decode_generic_region(
    data: bytes, image: JBIG2Image, page_info: JBIG2PageInfo | None = None
) -> None:
    region, mmr, template, prediction, at, bitmap_start = parse_generic_region_header(data)
    if region.width <= 0 or region.height <= 0:
        return
    if mmr:
        raise Jbig2UnsupportedError("JBIG2 MMR generic regions are unsupported")
    packed_bitmap = decode_arithmetic_generic_bitmap(
        region.raw[bitmap_start:], region.width, region.height, template, prediction, at
    )
    compose_packed_bitmap_region(region, packed_bitmap, image, page_info)


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
