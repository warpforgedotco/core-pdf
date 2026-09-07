# SPDX-License-Identifier: AGPL-3.0-only
"""Established tolerant content recovery selected by application composition."""

from collections.abc import Callable, Iterator

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.operations import (
    ContentOperation,
)
from core_pdf.impl.spec.s_07_content.operations import (
    iter_content_operations as strict_operations,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import WHITESPACE
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


def recover_inline_image_position(
    lexer: PdfLexer,
    position: int,
    is_valid_operator: Callable[[bytes], bool] | None = None,
) -> int | None:
    data = lexer.raw_data
    data_len = lexer.data_len
    source_buffer = lexer.source_buffer
    source_bytes: bytes | None = source_buffer if type(source_buffer) is bytes else None
    search_data = source_bytes if source_bytes is not None else data.tobytes()
    pos = position
    while pos < data_len:
        marker = search_data.find(b"EI", pos)
        if marker < 0:
            return None
        after = marker + 2
        if (
            (marker == 0 or data[marker - 1] in WHITESPACE)
            and after < data_len
            and data[after] in WHITESPACE
        ):
            next_pos = lexer.skip_ignored_at(after)
            word = lexer.scan_word_at(next_pos, skip_ignored=False)
            if word is None:
                return after
            token, ignored = word
            if (
                is_valid_operator(bytes(token))
                if is_valid_operator is not None
                else token in (b"BT", b"ET", b"q", b"Q", b"cm", b"Do", b"BI")
            ):
                return next_pos
        pos = marker + 1
    return None


class CaptureRecovery:
    max_stream_depth: int | None = 10
    skip_form_errors = True
    max_operands: int | None = 16

    def invalid_resources(self, value: object) -> PdfDict | None:
        return None

    def handle_error(self, error: Exception, context: str) -> None:
        pass

    def matrix_operand(self, value: object, context: str) -> Matrix:
        if context == "form" and isinstance(value, (list, tuple)) and len(value) > 6:
            return Matrix.from_operand(value[:6])
        if context == "pattern":
            return IDENTITY_MATRIX
        return Matrix.from_operand(value)

    def resume(
        self,
        lexer: PdfLexer,
        error: PdfParseError,
        kind: str,
        start: int,
        is_operator: Callable[[bytes], bool] | None = None,
    ) -> int | None:
        message = str(error)
        if kind == "inline-image":
            if message not in {
                "unterminated inline image",
                "unterminated inline image data",
                "inline image keys must be names",
                "expected inline image data separator",
            }:
                return None
            position = recover_inline_image_position(lexer, start, is_operator)
            if position is None and message == "unterminated inline image data":
                return lexer.data_len
            return position
        if kind == "array":
            if message == "unterminated array" and lexer.pos >= lexer.data_len:
                return lexer.data_len
        elif kind == "dictionary" and (
            lexer.pos >= lexer.data_len or message == "unexpected end of PDF input"
        ):
            return lexer.data_len
        return lexer.pos if lexer.pos > start else None


def iter_content_operations(lexer: PdfLexer) -> Iterator[ContentOperation]:
    yield from strict_operations(lexer, recovery=CaptureRecovery())
