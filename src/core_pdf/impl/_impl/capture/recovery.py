# SPDX-License-Identifier: AGPL-3.0-only
"""Established tolerant content recovery selected by application composition."""

from collections.abc import Callable, Iterator
from typing import cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf_spec.s_07_content.inline_images import (
    InlineImage,
    InlineImageDataLengthError,
    scan_inline_image_data,
)
from core_pdf_spec.s_07_content.operations import (
    ContentOperand,
    ContentOperands,
    ContentOperation,
    ContentToken,
    OperationHandler,
    parse_content_token,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.tokens import WHITESPACE
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


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


def recover_inline_image_data(lexer: PdfLexer, error: InlineImageDataLengthError) -> InlineImage:
    """Preserve the reader's permissive known-length and delimiter fallback."""
    data_end = error.data_start + error.expected_length
    marker = data_end
    while marker < lexer.data_len and lexer.raw_data[marker] in WHITESPACE:
        marker += 1
    if data_end <= lexer.data_len and lexer.raw_data[marker : marker + 2] == b"EI":
        lexer.pos = marker + 2
        return InlineImage(error.dictionary, bytes(lexer.raw_data[error.data_start : data_end]))
    return scan_inline_image_data(lexer, error.dictionary, error.data_start)


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
    results: list[ContentOperation] = []

    def get_handler(op_name: str) -> OperationHandler:
        def collect(operands: ContentOperands, depth: int) -> None:
            results.append((op_name, operands))

        return collect

    dispatch_operations(lexer, get_handler, 0, handlers_reject_unknown=False)
    yield from results


def dispatch_operations(
    lexer: PdfLexer,
    get_handler: Callable[[str], OperationHandler | None],
    depth: int,
    *,
    recovery: CaptureRecovery | None = None,
    handlers_reject_unknown: bool = True,
) -> None:
    """Apply reader operand limits and resume after malformed content tokens."""
    operands: list[ContentOperand] = []
    recovery = recovery if recovery is not None else CaptureRecovery()
    while True:
        lexer.skip_ignored()
        start = lexer.pos
        try:
            try:
                token = parse_content_token(lexer)
            except InlineImageDataLengthError as error:
                token = ContentToken(start, recover_inline_image_data(lexer, error))
        except PdfParseError as error:
            prefix = bytes(lexer.raw_data[start : start + 2])
            if str(error) == "unexpected delimiter in content stream":
                # The reader historically skips all standalone delimiters,
                # including stray closing strings and PostScript braces.
                lexer.pos = start + (2 if prefix == b">>" else 1)
                continue
            kind = (
                "inline-image"
                if prefix == b"BI"
                else "dictionary"
                if prefix == b"<<"
                else "array"
                if prefix.startswith(b"[")
                else "token"
            )
            resumed = recovery.resume(
                lexer,
                error,
                kind,
                start,
                (lambda word: get_handler(word.decode("latin-1")) is not None)
                if handlers_reject_unknown
                else None,
            )
            if resumed is None:
                raise
            lexer.pos = resumed
            if kind == "inline-image":
                operands.clear()
            continue
        if token is None:
            return
        if isinstance(token.value, InlineImage):
            if len(operands) < 16:
                operands.append(token.value)
            op_name = "BI"
        elif token.is_operator:
            op_name = cast(str, token.value)
        else:
            if len(operands) < 16:
                operands.append(token.value)
            continue
        if op_name not in {"R", "obj", "endobj", "stream", "endstream"}:
            handler = get_handler(op_name)
            if handler is not None:
                handler(tuple(operands), depth)
        operands.clear()
