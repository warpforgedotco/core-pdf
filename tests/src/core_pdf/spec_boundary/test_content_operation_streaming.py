# SPDX-License-Identifier: AGPL-3.0-only
"""Reader operation iteration preserves recovery while yielding incrementally."""

import pytest

from core_pdf.impl._impl.capture.recovery import iter_content_operations
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.inline_images import InlineImage


def test_reader_iterator_retains_operand_limit_structural_skips_and_tolerant_eof() -> None:
    prefix = b"] " + b" ".join(str(value).encode() for value in range(18)) + b" extension"
    lexer = PdfLexer(prefix + b" 3 R 4 w 5")
    operations = iter_content_operations(lexer)
    assert lexer.pos == 0
    assert next(operations) == ("extension", tuple(range(16)))
    assert lexer.pos == len(prefix)
    assert list(operations) == [("w", (4,))]


def test_reader_iterator_keeps_damaged_inline_image_and_continues() -> None:
    lexer = PdfLexer(b"1 w BI /W 3 /H 1 /BPC 8 /CS /G ID a EI 2 w")
    operations = iter_content_operations(lexer)
    assert next(operations) == ("w", (1,))
    name, operands = next(operations)
    assert name == "BI"
    image = operands[0]
    assert isinstance(image, InlineImage)
    assert image.data == b"a"
    assert list(operations) == [("w", (2,))]


def test_reader_iterator_uses_explicit_recovery_operator_vocabulary() -> None:
    content = b"BI broken EI extension 3 w"
    with pytest.raises(PdfParseError, match="inline image keys"):
        list(iter_content_operations(PdfLexer(content)))
    operations = iter_content_operations(
        PdfLexer(content), is_operator=lambda word: word == b"extension"
    )
    assert list(operations) == [("extension", ()), ("w", (3,))]
