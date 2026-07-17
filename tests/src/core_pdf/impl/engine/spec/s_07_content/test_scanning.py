# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from itertools import product

from core_pdf.impl.engine.spec.s_07_content.scanning import skip_literal_string


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
