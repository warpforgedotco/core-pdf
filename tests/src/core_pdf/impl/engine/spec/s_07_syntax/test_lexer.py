# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.tokens import DELIMITERS, WHITESPACE
from core_pdf.impl.objects import PdfStream, PdfString


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


def test_reversed_memoryview_does_not_reuse_backing_buffer_for_native_search() -> None:
    data = memoryview(b")abc(")[::-1]
    lexer = PdfLexer(data)

    assert lexer.source_buffer is None
    assert lexer.parse_object() == PdfString(b"cba")


def test_find_separator_supports_reversed_memoryview() -> None:
    data = b"ordinary token"
    lexer = PdfLexer(memoryview(data[::-1])[::-1])

    assert lexer.find_separator(0) == len(b"ordinary")


@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"48656c6c6f", b"Hello"),
        (b"48 65\n6c6c6f", b"Hello"),
        (b"486", b"H\x60"),
        (b"4g86", b"H\x60"),
        (b"zz", b""),
    ],
)
def test_hex_string_decode_supports_all_byte_views(
    view_kind: str,
    encoded: bytes,
    expected: bytes,
) -> None:
    content = b"<" + encoded + b"> trailing data"
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content

    assert PdfLexer(data).parse_object() == PdfString(expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (memoryview(bytearray(b"(\xff)")).cast("b"), b"\xff"),
        (memoryview(b"(AB)").cast("c"), b"AB"),
        (memoryview(b"(AB)").cast("H"), b"AB"),
        (memoryview(b"(AB)").cast("B", shape=(2, 2)), b"AB"),
    ],
)
def test_lexer_normalizes_non_byte_or_multidimensional_views(
    data: memoryview,
    expected: bytes,
) -> None:
    lexer = PdfLexer(data)

    assert lexer.raw_data.format == "B"
    assert lexer.raw_data.ndim == 1
    assert lexer.parse_object() == PdfString(expected)


@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"plain", b"plain"),
        (b"escaped\\)value", b"escaped)value"),
        (b"nested(inner)", b"nested(inner)"),
        (b"line\r\nending", b"line\nending"),
        (b"line\n\rending", b"line\nending"),
    ],
)
def test_literal_string_special_scan_supports_all_byte_views(
    view_kind: str,
    encoded: bytes,
    expected: bytes,
) -> None:
    content = b"(" + encoded + b") trailing data"
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content

    assert PdfLexer(data).parse_object() == PdfString(expected)


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


@pytest.mark.parametrize("view_kind", ["sliced", "reversed"])
def test_find_stream_end_reuses_logical_view_for_bidirectional_search(view_kind: str) -> None:
    content = b"payload\nendstream\nshort gap\nendstream\nendobj"
    if view_kind == "sliced":
        data = memoryview(b"prefix" + content + b"suffix")[6:-6]
    else:
        data = memoryview(content[::-1])[::-1]
    lexer = PdfLexer(data)
    first_marker = content.index(b"endstream")

    assert lexer.find_stream_end(0, preferred=first_marker + 9) == first_marker


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
