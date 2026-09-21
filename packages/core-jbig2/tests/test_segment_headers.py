# SPDX-License-Identifier: AGPL-3.0-only

import struct

import pytest

from core_jbig2.codec import (
    Jbig2ParseError,
    parse_embedded_segments,
    parse_referred_to_segments,
    parse_segment_header,
)


def header(number: int, width: int, count: int, page: int, long_page: bool) -> bytes:
    if count <= 4:
        retention = bytes([count << 5])
    else:
        retention = struct.pack(">I", 0xE0000000 | count) + bytes((count + 8) // 8)
    refs = b"".join((number - index - 1).to_bytes(width, "big") for index in range(count))
    return (
        struct.pack(">IB", number, 52 | (0x40 if long_page else 0))
        + retention
        + refs
        + page.to_bytes(4 if long_page else 1, "big")
        + struct.pack(">I", 4)
    )


@pytest.mark.parametrize(("number", "width"), [(256, 1), (257, 2), (65536, 2), (65537, 4)])
@pytest.mark.parametrize("count", [0, 1, 4, 5, 7, 8, 9])
@pytest.mark.parametrize("long_page", [False, True])
def test_reference_width_count_and_page_size_preserve_payload_boundary(
    number, width, count, long_page
):
    page = 0x12345678 if long_page else 17
    encoded = header(number, width, count, page, long_page)
    prefix = b"prefix"
    parsed, pos = parse_segment_header(prefix + encoded + b"data", len(prefix))
    assert parsed.number == number
    assert parsed.referred_to_count == count
    assert parsed.referred_to_segments == [number - index - 1 for index in range(count)]
    assert parsed.page_association == page
    assert parsed.data_length == 4
    assert parsed.header_length == len(encoded)
    assert pos == len(prefix) + len(encoded)
    segments = parse_embedded_segments(encoded + b"data")
    assert len(segments) == 1
    assert segments[0].data == b"data"
    assert segments[0].page_association == page


@pytest.mark.parametrize("count_tag", [5, 6])
def test_reserved_short_count_tags_are_rejected(count_tag):
    data = struct.pack(">IBB", 100, 52, count_tag << 5) + bytes(40)
    with pytest.raises(Jbig2ParseError, match="referred-to segment count"):
        parse_segment_header(data, 0)


@pytest.mark.parametrize("count", [1, 9])
def test_every_truncated_header_prefix_fails_before_returning_partial_fields(count):
    data = header(65537, 4, count, 0x12345678, True)
    for end in range(len(data)):
        with pytest.raises(Jbig2ParseError):
            parse_segment_header(data[:end], 0)
    assert parse_segment_header(data, 0)[1] == len(data)


def test_extended_count_keeps_high_five_bits_when_checking_required_storage():
    data = struct.pack(">IBI", 0x2000000, 52, 0xE1000000) + bytes(10)
    with pytest.raises(Jbig2ParseError):
        parse_segment_header(data, 0)


@pytest.mark.parametrize(
    ("long_form", "data", "expected"),
    [(False, b"\x01\xfe", [1, 254]), (True, bytes.fromhex("00000100 01000000"), [256, 16777216])],
)
def test_exported_reference_reader_preserves_legacy_byte_width_contract(long_form, data, expected):
    assert parse_referred_to_segments(b"x" + data + b"z", 1, 2, long_form) == (
        expected,
        1 + len(data),
    )
    with pytest.raises(Jbig2ParseError):
        parse_referred_to_segments(data[:-1], 0, 2, long_form)
