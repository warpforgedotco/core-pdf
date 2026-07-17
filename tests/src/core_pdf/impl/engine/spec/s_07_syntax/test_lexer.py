# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.tokens import DELIMITERS, WHITESPACE


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
