# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfString


@pytest.mark.parametrize(
    ("view_kind", "encoded", "expected"),
    [
        ("bytes", b"48656c6c6f", b"Hello"),
        ("bytes", b"48 65\n6c6c6f", b"Hello"),
        ("bytes", b"486", b"H\x60"),
        ("bytes", b"4g86", b"H\x60"),
        ("bytes", b"zz", b""),
        ("sliced", b"48 65\n6c6c6f", b"Hello"),
        ("reversed", b"486", b"H\x60"),
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


def test_strict_object_parsing_rejects_recovered_hex_string_bytes() -> None:
    with pytest.raises(PdfParseError, match="invalid hex string"):
        PdfLexer(b"<4g86>", recover_malformed_objects=False).parse_object()


@pytest.mark.parametrize(
    ("view_kind", "encoded", "expected"),
    [
        ("bytes", b"plain", b"plain"),
        ("bytes", b"escaped\\)value", b"escaped)value"),
        ("bytes", b"nested(inner)", b"nested(inner)"),
        ("bytes", b"line\r\nending", b"line\nending"),
        ("bytes", b"line\n\rending", b"line\nending"),
        ("sliced", b"escaped\\)value", b"escaped)value"),
        ("reversed", b"line\n\rending", b"line\nending"),
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


def test_read_string_normalizes_lfcr_in_sliced_memoryview() -> None:
    data = memoryview(b"prefix(first\n\rsecond)suffix")[len(b"prefix") : -len(b"suffix")]
    lexer = PdfLexer(data)

    assert lexer.read_string() == b"first\nsecond"


def test_read_string_preserves_legacy_lfcr_continuation() -> None:
    assert PdfLexer(b"(first\\\n\rsecond)").read_string() == b"firstsecond"


@pytest.mark.parametrize(("number", "expected"), [(b"1_0", 10), (b"1.0e2", 100.0)])
@pytest.mark.parametrize("separator", [b" ", b" % comment\n"])
def test_numeric_array_fast_paths_preserve_reader_numeric_acceptance(
    number: bytes, expected: int | float, separator: bytes
) -> None:
    assert PdfLexer(b"[0" + separator + number + b"]").parse_object() == [0, expected]


@pytest.mark.parametrize("line_ending", [b"\n", b"\r", b"\r\n", b"\n\r"])
def test_read_string_normalizes_all_supported_line_endings(line_ending: bytes) -> None:
    lexer = PdfLexer(b"(first" + line_ending + b"second)")

    assert lexer.read_string() == b"first\nsecond"


def test_read_string_can_project_pdfminer_unknown_escape_behavior() -> None:
    source = b"(27\\ mm glandsizes\\/torque)"

    native = PdfLexer(source)
    compatibility = PdfLexer(source)

    assert native.read_string() == b"27 mm glandsizes/torque"
    assert compatibility.read_string(drop_unknown_escapes=True) == b"27mm glandsizestorque"
