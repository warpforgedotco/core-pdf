from __future__ import annotations

import mmap
from typing import Any

from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.types import PdfSource

BBox = tuple[float, float, float, float]
PdfInput = PdfSource


class ClosingMixin:
    def close(self) -> None:
        return None

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def parse_indirect_object_at(
    data: bytes | bytearray | memoryview | mmap.mmap, offset: int, **lexer_options: Any
) -> Any:
    """The indirect object whose header starts at offset, parsed by a fresh lexer.

    Whatever the parse raises reaches the caller, and the lexer is closed
    either way.
    """
    lexer = PdfLexer(data, **lexer_options)
    try:
        lexer.rewind(offset)
        return lexer.parse_indirect_object()
    finally:
        lexer.close()
