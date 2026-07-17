# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from itertools import product

from core_pdf.impl.engine.spec.s_07_syntax.scanning import skip_literal_string, skip_name
from core_pdf.impl.engine.spec.s_07_syntax.tokens import DELIMITERS, WHITESPACE


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
    for separator in WHITESPACE + DELIMITERS:
        data = b"/Name" + bytes((separator,)) + b"suffix"

        assert skip_name(data, 0, len(data)) == len(b"/Name")
        assert skip_name(memoryview(data), 0, len(data)) == len(b"/Name")

    data = b"/Name\vsuffix"
    assert skip_name(data, 0, len(data)) == len(data)


def test_skip_literal_string_respects_explicit_scan_boundary() -> None:
    visible = b"(unterminated"
    data = visible + b") outside (\\"

    assert skip_literal_string(data, 0, len(visible)) == len(visible)
