# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.tokens import DELIMITERS, WHITESPACE
from core_pdf.impl.objects import PdfStream


@pytest.mark.parametrize("separator", WHITESPACE + DELIMITERS)
def test_find_separator_recognizes_pdf_separator_bytes(separator: int) -> None:
    lexer = PdfLexer(b"ordinary" + bytes((separator,)) + b"suffix")

    assert lexer.find_separator(0) == len(b"ordinary")


@pytest.mark.parametrize(
    "data",
    [
        b"ordinary-token",
        bytearray(b"ordinary-token"),
        memoryview(b"ordinary-token"),
    ],
)
def test_find_separator_returns_data_length_without_separator(
    data: bytes | bytearray | memoryview,
) -> None:
    lexer = PdfLexer(data)

    assert lexer.find_separator(0) == len(data)
    assert lexer.find_separator(8) == len(data)


@pytest.mark.parametrize(
    "data",
    [
        b"[1\v2]",
        memoryview(b"prefix[1\v2]suffix")[len(b"prefix") : -len(b"suffix")],
    ],
)
def test_numeric_array_fast_path_uses_pdf_whitespace(data: bytes | memoryview) -> None:
    assert PdfLexer(data).parse_object() == ["1\v2"]


@pytest.mark.parametrize(
    "data",
    [
        b"[(plain) (escaped\\)value) (nested(inner)) (line\n\rending)]",
        memoryview(b"prefix[(plain) (escaped\\)value) (nested(inner)) (line\n\rending)]suffix")[
            len(b"prefix") : -len(b"suffix")
        ],
    ],
)
def test_simple_tj_array_uses_literal_string_semantics(data: bytes | memoryview) -> None:
    lexer = PdfLexer(data)

    assert lexer.parse_simple_tj_array() == [
        b"plain",
        b"escaped)value",
        b"nested(inner)",
        b"line\nending",
    ]


def test_find_stream_end_prefers_delimited_keyword_over_payload_bytes() -> None:
    data = b"binary-endstream-data\nendstream\nendobj"
    lexer = PdfLexer(data)

    assert lexer.find_stream_end(0) == data.index(b"endstream", len(b"binary-endstream"))


def test_find_stream_end_prefers_candidate_after_expected_position() -> None:
    data = b"payload\nendstream\nmore payload\nendstream\nendobj"
    lexer = PdfLexer(data)
    second_marker = data.rindex(b"endstream")

    assert lexer.find_stream_end(0, preferred=second_marker - 2) == second_marker


def test_find_stream_end_uses_nearest_delimited_candidate() -> None:
    data = b"payload\nendstream\nshort gap plus a much longer payload\nendstream\nendobj"
    first_marker = data.index(b"endstream")
    lexer = PdfLexer(data)

    assert lexer.find_stream_end(0, preferred=first_marker + 12) == first_marker


def test_find_stream_end_uses_compatible_fallback_for_malformed_keyword() -> None:
    data = b"payload-endstream-endobj"
    lexer = PdfLexer(data)

    assert lexer.find_stream_end(0) == data.index(b"endstream")


def test_find_object_end_prefers_delimited_keyword() -> None:
    data = b"(embedded-endobj-value)\nendobj\n"
    lexer = PdfLexer(data)

    assert lexer.find_object_end(0) == data.rindex(b"endobj")


def test_keyword_recovery_supports_sliced_memoryview() -> None:
    source = b"prefixpayload\nendstream\nendobj-suffix"
    data = memoryview(source)[len(b"prefix") : -len(b"-suffix")]
    lexer = PdfLexer(data)

    assert lexer.find_stream_end(0) == bytes(data).index(b"endstream")
    assert lexer.find_object_end(0) == bytes(data).index(b"endobj")


@pytest.mark.parametrize("declared_length", [-1, 7, 999])
def test_parse_stream_recovery_rejects_embedded_keyword_bytes(declared_length: int) -> None:
    payload = b"payload-endstream-data\n"
    data = f"<< /Length {declared_length} >>\nstream\n".encode() + payload + b"endstream\nendobj"

    stream = PdfLexer(data).parse_object()

    assert isinstance(stream, PdfStream)
    assert bytes(stream.raw_data) == payload


def test_parse_stream_trusts_exact_length_with_embedded_keyword() -> None:
    payload = b"payload\nendstream\ninside"
    data = f"<< /Length {len(payload)} >>\nstream\n".encode() + payload + b"\nendstream\nendobj"

    stream = PdfLexer(data).parse_object()

    assert isinstance(stream, PdfStream)
    assert bytes(stream.raw_data) == payload


@pytest.mark.parametrize(
    "data",
    [
        b"(first\n\rsecond)",
        memoryview(b"prefix(first\n\rsecond)suffix")[len(b"prefix") : -len(b"suffix")],
    ],
)
def test_read_string_normalizes_lfcr_as_one_line_ending(
    data: bytes | memoryview,
) -> None:
    lexer = PdfLexer(data)

    assert lexer.read_string() == b"first\nsecond"


@pytest.mark.parametrize("line_ending", [b"\n", b"\r", b"\r\n", b"\n\r"])
def test_read_string_normalizes_all_supported_line_endings(line_ending: bytes) -> None:
    lexer = PdfLexer(b"(first" + line_ending + b"second)")

    assert lexer.read_string() == b"first\nsecond"
