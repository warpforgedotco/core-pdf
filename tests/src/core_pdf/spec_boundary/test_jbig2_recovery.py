# SPDX-License-Identifier: AGPL-3.0-only
"""JBIG2 reader guesses remain outside strict decoding."""

import struct

import pytest

from core_pdf.impl._impl.graphics.codec_dispatch import decode_jbig2
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError


def segment(number: int, kind: int, data: bytes) -> bytes:
    return struct.pack(">IBBBI", number, kind, 0, 1, len(data)) + data


def page_info() -> bytes:
    return segment(1, 48, struct.pack(">IIIIBH", 8, 1, 0, 0, 0, 0))


def test_reader_retains_raw_mmr_approximation() -> None:
    header = struct.pack(">IIiiBB", 8, 1, 0, 0, 0, 1)
    data = page_info() + segment(2, 38, header + bytes.fromhex("80 08 00 80"))
    assert decode_jbig2(data, None) == b"\x7f"


def test_reader_retains_raw_text_offset_and_or_composition() -> None:
    header = struct.pack(">IIiiB", 8, 1, 0, 0, 2) + b"\x00" * 3
    data = page_info() + segment(2, 6, header + b"\x80") + segment(3, 6, header + b"\x80")
    assert decode_jbig2(data, None) == b"\x7f"


@pytest.mark.parametrize(
    "kind", [0, 1, 4, 7, 16, 20, 22, 23, 36, 40, 42, 43, 49, 50, 52, 53, 62, 63]
)
def test_reader_retains_ignored_segment_types_even_with_invalid_payloads(kind: int) -> None:
    assert decode_jbig2(page_info() + segment(2, kind, b"x"), None) == b"\xff"


def test_reader_does_not_guess_unsupported_arithmetic_templates() -> None:
    header = struct.pack(">IIiiBB", 8, 1, 0, 0, 0, 2) + b"\x03\xff"
    with pytest.raises(FilterUnsupportedError, match="generic bitmap template"):
        decode_jbig2(page_info() + segment(2, 38, header + b"\xff\xac"), None)


def test_reader_retains_parameter_coercion_and_supported_arithmetic() -> None:
    header = struct.pack(">IIiiBB", 8, 1, 0, 0, 0, 0) + bytes.fromhex("03 ff fd ff 02 fe fe fe")
    data = page_info() + segment(2, 38, header + b"\xff\xac")
    assert decode_jbig2(data, {"Columns": "8"}) == b"\x00"


def test_reader_retains_truncated_text_errors() -> None:
    with pytest.raises(FilterParseError, match="truncated JBIG2 text region"):
        decode_jbig2(page_info() + segment(2, 6, struct.pack(">IIiiB", 8, 1, 0, 0, 0)), None)
