# SPDX-License-Identifier: AGPL-3.0-only

import struct

import pytest

from core_jbig2.codec import (
    JBIG2PageDecoder,
    Jbig2ParseError,
    Jbig2UnsupportedError,
    parse_embedded_segments,
    parse_generic_region_header,
    parse_region,
)
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_filters.jbig2 import decode_jbig2


def segment(number: int, kind: int, data: bytes, page: int = 1) -> bytes:
    return struct.pack(">IBBBI", number, kind, 0, page, len(data)) + data


def page_info(flags: int = 0) -> bytes:
    return segment(1, 48, struct.pack(">IIIIBH", 8, 1, 0, 0, flags, 0))


def generic_region(mmr: bool, payload: bytes) -> bytes:
    header = struct.pack(">IIiiBB", 8, 1, 0, 0, 0, int(mmr))
    if not mmr:
        header += bytes.fromhex("03 ff fd ff 02 fe fe fe")
    return segment(2, 38, header + payload)


@pytest.mark.parametrize(("flags", "expected"), [(0, b"\xff"), (4, b"\x00")])
def test_default_page_pixels_and_harmless_segments(flags: int, expected: bytes) -> None:
    encoded = (
        segment(0, 52, struct.pack(">II", 1, 1), page=0)
        + page_info(flags)
        + segment(2, 62, struct.pack(">I", 1) + b"optional metadata")
        + segment(3, 49, b"")
        + segment(4, 51, b"", page=0)
    )
    assert decode_jbig2(encoded, None) == expected


def test_supported_arithmetic_template_zero_decodes_black_row() -> None:
    encoded = page_info() + generic_region(False, bytes.fromhex("ff ac")) + segment(3, 49, b"")
    assert decode_jbig2(encoded, None) == b"\x00"


def test_mmr_compressed_bits_are_never_returned_as_pixels() -> None:
    encoded = page_info() + generic_region(True, bytes.fromhex("80 08 00 80"))
    with pytest.raises(FilterUnsupportedError, match="MMR region") as error:
        decode_jbig2(encoded, None)
    assert isinstance(error.value.__cause__, Jbig2UnsupportedError)


@pytest.mark.parametrize("kind", [4, 6, 7])
def test_text_regions_are_explicitly_unsupported(kind: int) -> None:
    data = struct.pack(">IIiiB", 8, 1, 0, 0, 0) + b"\x00" * 7
    with pytest.raises(FilterUnsupportedError, match="text region"):
        decode_jbig2(page_info() + segment(2, kind, data), None)


@pytest.mark.parametrize("kind", [0, 16, 20, 22, 23, 36, 40, 42, 43, 50, 53])
def test_unsupported_segment_types_do_not_silently_return_partial_pages(kind: int) -> None:
    with pytest.raises(FilterUnsupportedError, match=f"segment type {kind}"):
        decode_jbig2(page_info() + segment(2, kind, b""), None)


@pytest.mark.parametrize("kind", [1, 5, 8, 47, 54, 63])
def test_reserved_segment_types_are_malformed(kind: int) -> None:
    with pytest.raises(FilterParseError, match=f"reserved JBIG2 segment type {kind}"):
        decode_jbig2(page_info() + segment(2, kind, b""), None)


@pytest.mark.parametrize(
    ("kind", "data", "message"),
    [
        (49, b"x", "end marker"),
        (51, b"x", "end marker"),
        (52, b"\x00" * 3, "truncated JBIG2 profiles"),
        (52, struct.pack(">I", 1), "profiles segment length"),
        (52, struct.pack(">II", 0, 1), "profiles segment length"),
        (62, b"\x00" * 3, "truncated JBIG2 extension"),
        (62, struct.pack(">I", 0x80000000), "reserved bit 29"),
        (6, b"\x00" * 16, "truncated JBIG2 text region"),
        (38, struct.pack(">IIiiB", 8, 1, 0, 0, 0), "truncated JBIG2 generic region"),
    ],
)
def test_malformed_segment_metadata_keeps_parse_error_mapping(
    kind: int, data: bytes, message: str
) -> None:
    with pytest.raises(FilterParseError, match=message) as error:
        decode_jbig2(page_info() + segment(2, kind, data), None)
    assert isinstance(error.value.__cause__, Jbig2ParseError)


def test_necessary_unknown_extension_is_unsupported() -> None:
    data = page_info() + segment(2, 62, struct.pack(">I", 0xA0000000))
    with pytest.raises(FilterUnsupportedError, match="necessary JBIG2 extension"):
        decode_jbig2(data, None)


def test_globals_stream_is_read_once_and_precedes_page_segments() -> None:
    class Globals:
        reads = 0

        @property
        def data(self) -> bytes:
            self.reads += 1
            return segment(0, 52, struct.pack(">I", 0), page=0)

    globals_stream = Globals()
    assert decode_jbig2(page_info(), {"JBIG2Globals": globals_stream}) == b"\xff"
    assert globals_stream.reads == 1


@pytest.mark.parametrize("value", [b"", bytearray(), memoryview(b"")])
def test_globals_accept_byte_buffers(value: object) -> None:
    assert decode_jbig2(page_info(), {"JBIG2Globals": value}) == b"\xff"


def test_invalid_globals_and_absent_images_have_specific_errors() -> None:
    with pytest.raises(FilterParseError, match="invalid JBIG2 globals"):
        decode_jbig2(page_info(), {"JBIG2Globals": 42})
    with pytest.raises(FilterUnsupportedError, match="produced no image"):
        decode_jbig2(b"", None)


def test_generic_header_keeps_common_metadata_and_signed_offsets() -> None:
    data = struct.pack(">IIiiBB", 8, 1, -4, 2, 0, 0) + bytes.fromhex("03 ff fd ff 02 fe fe fe")
    region = parse_region(data, "generic")
    header = parse_generic_region_header(region)
    assert header.region is region
    assert (region.x, region.y) == (-4, 2)
    assert header.adaptive_pixels == ((3, -1), (-3, -1), (2, -2), (-2, -2))
    assert header.bitmap_start == 26


def test_decode_jbig2_inverts_t88_polarity_for_pdf() -> None:
    decoder = JBIG2PageDecoder()
    for item in parse_embedded_segments(page_info(4)):
        decoder.decode_segment(item)
    assert decoder.finish() == b"\xff"
    assert decode_jbig2(page_info(4), None) == b"\x00"
