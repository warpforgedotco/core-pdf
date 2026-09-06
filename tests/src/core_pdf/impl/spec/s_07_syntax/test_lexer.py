# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import PdfReference, PdfString


def test_literal_strings_keep_lfcr_as_two_pdf_line_endings() -> None:
    assert PdfLexer(b"[(first\n\rsecond) (first\\\n\rsecond)]").parse_object() == [
        PdfString(b"first\n\nsecond"),
        PdfString(b"first\nsecond"),
    ]


@pytest.mark.parametrize("number", [b"1_0", b"1.0e2", b"1.0e+2", b"1_.0"])
@pytest.mark.parametrize("separator", [b" ", b" % comment\n"])
def test_numeric_array_fast_paths_use_the_scalar_pdf_number_grammar(
    number: bytes, separator: bytes
) -> None:
    expected = PdfLexer(number).parse_object()
    assert PdfLexer(b"[0" + separator + number + b"]").parse_object() == [0, expected]


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


@pytest.mark.parametrize("content", [b"(plain", b"(nested(inner)", b"(trailing\\", b"("])
def test_unterminated_literal_string_leaves_cursor_at_end(content: bytes) -> None:
    lexer = PdfLexer(b"42 " + content)
    assert lexer.parse_object() == 42

    with pytest.raises(PdfParseError, match="^unterminated string$"):
        lexer.parse_object()

    assert lexer.pos == lexer.data_len


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
