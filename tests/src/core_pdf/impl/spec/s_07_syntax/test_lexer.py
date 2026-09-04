# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError
from core_pdf.impl.primitives import PdfReference, PdfString
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


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
    ("content", "expected"),
    [
        (b"[]", []),
        (b"[1 -2 3.5]", [1, -2, 3.5]),
        (b"[1 % comment\n2]", [1, 2]),
        (b"[1 [2] 3]", [1, [2], 3]),
        (b"[1 0 R]", [PdfReference(1, 0)]),
        (b"[1.2.3]", ["1.2.3"]),
    ],
)
def test_numeric_array_slice_fast_path_preserves_general_array_semantics(
    content: bytes,
    expected: list[object],
) -> None:
    data = memoryview(b"prefix" + content + b"suffix")[6:-6]

    assert PdfLexer(data).parse_object() == expected


@pytest.mark.parametrize(
    ("view_kind", "ignored", "expected"),
    [
        ("bytes", b"\x00\t\n\x0c\r ", 6),
        ("bytes", b"% comment\r\n", 11),
        ("bytes", b"% comment\n\r \t", 13),
        ("bytes", b" % first\n% second\r", 18),
        ("bytes", b" " * 8, 8),
        ("bytes", b" " * 9, 9),
        ("bytes", b" " * 7 + b"% comment\n", 17),
        ("bytes", b" " * 9 + b"% comment\r", 19),
        ("bytes", b"\v", 0),
        ("sliced", b"% comment\r\n", 11),
        ("reversed", b" % first\n% second\r", 18),
    ],
)
def test_skip_ignored_preserves_pdf_comment_and_whitespace_semantics(
    view_kind: str,
    ignored: bytes,
    expected: int,
) -> None:
    content = ignored + b"/Name"
    if view_kind == "sliced":
        data: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
    elif view_kind == "reversed":
        data = memoryview(content[::-1])[::-1]
    else:
        data = content

    assert PdfLexer(data).skip_ignored_at(0) == expected


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


def test_parse_stream_preserves_carriage_return_after_lf_delimiter() -> None:
    payload = b"\rbinary stream data"
    data = f"<< /Length {len(payload)} >>\nstream\n".encode() + payload + b"\nendstream\nendobj"

    stream = PdfLexer(data).parse_object()

    assert isinstance(stream, PdfStream)
    assert bytes(stream.raw_data) == payload


def test_read_string_normalizes_lfcr_in_sliced_memoryview() -> None:
    data = memoryview(b"prefix(first\n\rsecond)suffix")[len(b"prefix") : -len(b"suffix")]
    lexer = PdfLexer(data)

    assert lexer.read_string() == b"first\nsecond"


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


@pytest.mark.parametrize(
    "signature_entries",
    [
        b"/Contents <30820100> /Type /Sig",
        b"/Type /DocTimeStamp /Contents <30820100>",
        b"/Contents <30820100> /ByteRange [0 10 20 30]",
    ],
)
def test_signature_hex_contents_are_not_deciphered(signature_entries: bytes) -> None:
    def decipher(
        _object_number: int,
        _generation_number: int,
        _data: bytes,
        _dictionary: PdfDict | None,
    ) -> bytes:
        raise AssertionError("signature contents must remain unencrypted")

    lexer = PdfLexer(b"7 0 obj << " + signature_entries + b" >> endobj", decipher=decipher)

    dictionary = lexer.parse_indirect_object()

    assert isinstance(dictionary, dict)
    assert dictionary["Contents"] == PdfString(bytes.fromhex("30820100"))


def test_non_signature_hex_contents_are_deciphered_after_dictionary_parse() -> None:
    calls: list[tuple[int, int, bytes]] = []

    def decipher(
        object_number: int,
        generation_number: int,
        data: bytes,
        _dictionary: PdfDict | None,
    ) -> bytes:
        calls.append((object_number, generation_number, data))
        return b"deciphered"

    lexer = PdfLexer(
        b"7 2 obj << /Contents <0102> /Type /Annot >> endobj",
        decipher=decipher,
    )

    dictionary = lexer.parse_indirect_object()

    assert isinstance(dictionary, dict)
    assert dictionary["Contents"] == PdfString(b"deciphered")
    assert calls == [(7, 2, b"\x01\x02")]


def test_non_signature_hex_contents_propagate_decryption_failure() -> None:
    def decipher(
        _object_number: int,
        _generation_number: int,
        _data: bytes,
        _dictionary: PdfDict | None,
    ) -> bytes:
        raise PdfDecryptionError("Invalid encrypted object ciphertext")

    lexer = PdfLexer(b"7 0 obj << /Contents <0102> /Type /Annot >> endobj", decipher=decipher)

    with pytest.raises(PdfDecryptionError, match="Invalid encrypted object ciphertext"):
        lexer.parse_indirect_object()
