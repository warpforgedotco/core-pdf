# SPDX-License-Identifier: AGPL-3.0-only
"""PostScript literal strings decode escapes, nesting, and line ends per PLRM 3.3.2."""

from __future__ import annotations

import array

import pytest

from core_adobe_fonts.cmap.lexical import (
    DELIMITERS,
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
    read_literal_string,
)
from core_adobe_fonts.cmap.tokenizer import decode_literal_string


def read(data: bytes | memoryview) -> tuple[bytes | None, int]:
    return read_literal_string(data, 0, len(data))


def test_tables_agree_with_the_byte_sets() -> None:
    assert all(WS_TABLE[byte] for byte in WHITESPACE)
    assert all(SEPARATOR_TABLE[byte] for byte in WHITESPACE + DELIMITERS)
    assert not SEPARATOR_TABLE[ord("{")]
    assert not SEPARATOR_TABLE[ord("a")]


def test_plain_string_uses_the_fast_path_and_reports_the_end() -> None:
    assert read(b"(abc) rest") == (b"abc", 5)
    assert read(b"()") == (b"", 2)


def test_balanced_parentheses_nest_without_escapes() -> None:
    assert read(b"(a(b(c))d)") == (b"a(b(c))d", 10)


@pytest.mark.parametrize(
    ("escaped", "expected"),
    [
        (b"\\n", b"\n"),
        (b"\\r", b"\r"),
        (b"\\t", b"\t"),
        (b"\\b", b"\b"),
        (b"\\f", b"\x0c"),
        (b"\\(", b"("),
        (b"\\)", b")"),
        (b"\\\\", b"\\"),
        (b"\\7", b"\x07"),
        (b"\\53", b"+"),
        (b"\\053", b"+"),
        (b"\\0053", b"\x053"),
        (b"\\400", b"\x00"),
        (b"\\q", b"q"),
    ],
)
def test_escape_sequences_follow_the_table(escaped: bytes, expected: bytes) -> None:
    assert read(b"(" + escaped + b")") == (expected, len(escaped) + 2)


@pytest.mark.parametrize("line_end", [b"\r", b"\n", b"\r\n"])
def test_escaped_line_end_is_a_continuation(line_end: bytes) -> None:
    assert read(b"(a\\" + line_end + b"b)") == (b"ab", len(line_end) + 5)


@pytest.mark.parametrize("line_end", [b"\r", b"\n", b"\r\n"])
def test_unescaped_line_ends_normalise_to_a_newline(line_end: bytes) -> None:
    assert read(b"(a" + line_end + b"b)") == (b"a\nb", len(line_end) + 4)


def test_trailing_backslash_at_end_of_data_is_ignored() -> None:
    assert read(b"(ab\\") == (None, 4)


def test_unterminated_string_returns_none_and_the_exhausted_position() -> None:
    assert read(b"(abc") == (None, 4)
    assert read(b"(a(b)") == (None, 5)


def test_non_contiguous_memoryview_takes_the_slow_scan() -> None:
    data = memoryview(b"(xaxbxcx)x")[::2]
    assert bytes(data) == b"(abc)"
    assert read(data) == (b"abc", 5)
    escaped = memoryview(b"(xax\\xnx)x")[::2]
    assert bytes(escaped) == b"(a\\n)"
    assert read(escaped) == (b"a\n", 5)


def test_wide_memoryview_elements_are_decoded_as_values() -> None:
    codes = array.array("i", [40, 65, 92, 110, 41, 66])
    view = memoryview(codes)
    assert view.format == "i"
    assert read_literal_string(view, 0, len(view)) == (b"A\n", 5)


def test_tokenizer_literal_helper_requires_a_complete_string() -> None:
    assert decode_literal_string(b"(a\\)b)") == b"a)b"
    with pytest.raises(ValueError, match="invalid literal string"):
        decode_literal_string(b"abc")
    with pytest.raises(ValueError, match="unterminated literal string"):
        decode_literal_string(b"(abc")
