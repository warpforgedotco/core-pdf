# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable, Iterator

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.types import PdfName
from core_pdf_cythonized import ContentScanner
from core_pdf_spec.s_07_content.inline_images import (
    InlineImage,
    InlineImageDataLengthError,
    scan_inline_image_data,
)
from core_pdf_spec.s_07_content.operations import (
    ContentOperand,
    ContentOperation,
    ContentToken,
    parse_content_token,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax_primitives.tokens import WHITESPACE


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
    data_end = error.data_start + error.expected_length
    marker = data_end
    while marker < lexer.data_len and lexer.raw_data[marker] in WHITESPACE:
        marker += 1
    if data_end <= lexer.data_len and lexer.raw_data[marker : marker + 2] == b"EI":
        lexer.pos = marker + 2
        return InlineImage(error.dictionary, bytes(lexer.raw_data[error.data_start : data_end]))
    return scan_inline_image_data(lexer, error.dictionary, error.data_start)


class CaptureRecovery:
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


KEYWORD_TOKENS = frozenset((b"BI", b"true", b"false", b"null"))
OBJECT_KEYWORDS = frozenset(("R", "obj", "endobj", "stream", "endstream"))


def iter_content_operations(
    lexer: PdfLexer,
    *,
    recovery: CaptureRecovery | None = None,
    is_operator: Callable[[bytes], bool] | None = None,
) -> Iterator[ContentOperation]:
    recovery = recovery if recovery is not None else CaptureRecovery()
    # The kernel scans whitespace, comments, numbers, names and operators,
    # appending operands as it goes and handing back each operation whole, so
    # the loop below runs once per operation rather than once per token. It
    # hands back a byte offset for anything it does not own -- strings,
    # arrays, dictionaries, inline images, the BI/true/false/null keywords,
    # and anything malformed -- and every one of those, with all the recovery
    # around them, is still parsed here onto the kernel's operand list.
    scanner = ContentScanner(lexer.raw_data, KEYWORD_TOKENS, OBJECT_KEYWORDS, PdfName.of)
    operands: list[ContentOperand] = scanner.operands
    scan = scanner.next_operation
    while True:
        # The slow path below owns lexer.pos and moves it in ways the scanner
        # cannot see, so the two cursors meet here once per operation.
        scanner.pos = lexer.pos
        result = scan()
        if isinstance(result, tuple):
            lexer.pos = scanner.pos
            yield result
            continue
        lexer.pos = cursor = result
        try:
            try:
                token = parse_content_token(lexer)
            except InlineImageDataLengthError as error:
                start = lexer.skip_ignored_at(cursor)
                token = ContentToken(start, recover_inline_image_data(lexer, error))
        except PdfParseError as error:
            start = lexer.skip_ignored_at(cursor)
            prefix = bytes(lexer.raw_data[start : start + 2])
            if str(error) == "unexpected delimiter in content stream":
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
                is_operator,
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
        elif token.is_operator and isinstance(token.value, str):
            op_name = token.value
        else:
            if len(operands) < 16:
                operands.append(token.value)
            continue
        operation = (op_name, tuple(operands))
        operands.clear()
        if op_name not in OBJECT_KEYWORDS:
            yield operation
