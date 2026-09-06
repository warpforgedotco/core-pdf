# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from itertools import product

import pytest

from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    read_literal_string,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
    skip_name,
)

PDF_SEPARATORS = b"\x00\t\n\x0c\r ()<>[]{}/%"


def reference_skip_literal_string(data: bytes) -> int:
    pos = 1
    depth = 1
    while pos < len(data) and depth:
        byte = data[pos]
        if byte == 92:
            pos = min(pos + 2, len(data))
            continue
        if byte == 40:
            depth += 1
        elif byte == 41:
            depth -= 1
        pos += 1
    return pos


def test_skip_literal_string_matches_reference_exhaustively() -> None:
    alphabet = b"()\\ab"
    for length in range(7):
        for suffix in product(alphabet, repeat=length):
            data = b"(" + bytes(suffix)
            expected = reference_skip_literal_string(data)

            assert skip_literal_string(data, 0, len(data)) == expected
            assert skip_literal_string(memoryview(data), 0, len(data)) == expected


def test_skip_name_uses_canonical_pdf_separators() -> None:
    for separator in PDF_SEPARATORS:
        data = b"/Name" + bytes((separator,)) + b"suffix"

        assert skip_name(data, 0, len(data)) == len(b"/Name")
        assert skip_name(memoryview(data), 0, len(data)) == len(b"/Name")

    data = b"/Name\vsuffix"
    assert skip_name(data, 0, len(data)) == len(data)


def test_skip_literal_string_respects_explicit_scan_boundary() -> None:
    visible = b"(unterminated"
    data = visible + b") outside (\\"

    assert skip_literal_string(data, 0, len(visible)) == len(visible)


def test_read_literal_string_respects_position_and_explicit_boundary() -> None:
    data = memoryview(b"prefix (a\\nb) outside")
    assert read_literal_string(data, 7, len(data)) == (b"a\nb", 13)
    assert read_literal_string(data, 7, 12) == (None, 12)


@pytest.mark.parametrize(
    ("ending", "plain", "escaped"),
    [
        (b"\r", b"\n", b""),
        (b"\n", b"\n", b""),
        (b"\r\n", b"\n", b""),
        (b"\n\r", b"\n\n", b"\n"),
    ],
)
@pytest.mark.parametrize("view_kind", ["bytes", "sliced", "reversed"])
def test_read_literal_string_uses_pdf_eol_sequences(
    ending: bytes, plain: bytes, escaped: bytes, view_kind: str
) -> None:
    # ISO 32000-1:2008 7.2.3 defines CR, LF, and CRLF, so LFCR has two markers.
    for prefix, expected in ((b"", plain), (b"\\", escaped)):
        content = b"(first" + prefix + ending + b"second)"
        if view_kind == "sliced":
            source: bytes | memoryview = memoryview(b"prefix" + content + b"suffix")[6:-6]
        elif view_kind == "reversed":
            source = memoryview(content[::-1])[::-1]
        else:
            source = content
        assert read_literal_string(source, 0, len(source)) == (
            b"first" + expected + b"second",
            len(source),
        )


def test_read_literal_string_delegates_only_unknown_escapes() -> None:
    seen: list[int] = []

    def unknown_escape(byte: int) -> bytes:
        seen.append(byte)
        return b"?"

    content = b"(a\\q\\n\\050\\\\\\/z)"
    assert read_literal_string(content, 0, len(content), unknown_escape=unknown_escape) == (
        b"a?\n(\\?z",
        len(content),
    )
    assert seen == [ord("q"), ord("/")]


def test_simple_scanners_respect_explicit_scan_boundary() -> None:
    comment = b"% unterminated" + b"\n outside"
    hex_string = b"<unterminated" + b"> outside"
    comment_end = len(b"% unterminated")
    hex_end = len(b"<unterminated")

    assert skip_comment(comment, 0, comment_end) == comment_end
    assert skip_comment(memoryview(comment), 0, comment_end) == comment_end
    assert skip_hex_string(hex_string, 0, hex_end) == hex_end
    assert skip_hex_string(memoryview(hex_string), 0, hex_end) == hex_end
