# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from array import array

import pytest

from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import decode_pdf_literal_string


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"()", b""),
        (b"(plain) trailing data", b"plain"),
        (b"(nested(inner))", b"nested(inner)"),
        (rb"(\(escaped\)\\)", b"(escaped)\\"),
        (rb"(\n\r\t\b\f)", b"\n\r\t\b\f"),
        (rb"(\7\77\377\777\1234)", b"\x07?\xff\xffS4"),
        (rb"(unknown\q\8\9)", b"unknownq89"),
        (b"(a\r\nb\n\rc\rd\ne)", b"a\nb\nc\nd\ne"),
        (b"(a\\\r\nb\\\n\rc\\\rd\\\ne)", b"abcde"),
    ],
)
@pytest.mark.parametrize("view_kind", ["bytes", "bytearray", "sliced", "strided"])
def test_cmap_literal_string_decoding(encoded: bytes, expected: bytes, view_kind: str) -> None:
    data: bytes | bytearray | memoryview
    if view_kind == "bytearray":
        data = bytearray(encoded)
    elif view_kind == "sliced":
        data = memoryview(b"prefix" + encoded + b"suffix")[6:-6]
    elif view_kind == "strided":
        data = memoryview(encoded[::-1])[::-1]
    else:
        data = encoded

    assert decode_pdf_literal_string(data) == expected


@pytest.mark.parametrize("encoded", [b"", b"(", b"plain", b"<41>"])
def test_cmap_literal_string_rejects_invalid_delimiters(encoded: bytes) -> None:
    with pytest.raises(ValueError, match="^invalid PDF literal string$"):
        decode_pdf_literal_string(encoded)


@pytest.mark.parametrize("encoded", [b"(plain", b"(nested(inner)", b"(trailing\\"])
def test_cmap_literal_string_rejects_unterminated_input(encoded: bytes) -> None:
    with pytest.raises(ValueError, match="^unterminated PDF literal string$"):
        decode_pdf_literal_string(encoded)


@pytest.mark.parametrize(("typecode", "invalid"), [("H", 256), ("I", 256), ("b", -1)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"()", b""),
        (b"(A)", b"A"),
        (b"(nested(inner))", b"nested(inner)"),
        (rb"(\123\7\n)", b"S\7\n"),
    ],
)
def test_cmap_literal_string_decodes_numeric_elements_and_ignores_trailing_values(
    typecode: str, invalid: int, strided: bool, encoded: bytes, expected: bytes
) -> None:
    values = [*encoded, invalid]
    data = (
        memoryview(array(typecode, (part for value in values for part in (value, 0))))[::2]
        if strided
        else memoryview(array(typecode, values))
    )

    assert decode_pdf_literal_string(data) == expected


@pytest.mark.parametrize(("typecode", "invalid"), [("H", 256), ("I", 256), ("b", -1)])
@pytest.mark.parametrize("strided", [False, True])
def test_cmap_literal_string_rejects_out_of_byte_range_numeric_elements(
    typecode: str, invalid: int, strided: bool
) -> None:
    data = (
        memoryview(array(typecode, (40, 0, invalid, 0, 41, 0)))[::2]
        if strided
        else memoryview(array(typecode, (40, invalid, 41)))
    )

    with pytest.raises(ValueError, match="^byte must be in range\\(0, 256\\)$"):
        decode_pdf_literal_string(data)
