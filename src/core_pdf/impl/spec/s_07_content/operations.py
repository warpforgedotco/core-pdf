# SPDX-License-Identifier: AGPL-3.0-only
"""Tokenize and dispatch content-stream operators."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import TypeAlias, cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.inline_images import (
    InlineImage,
    parse_inline_image,
    recover_inline_image_position,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.types import CachedPdfObject
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    is_number_word_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
    skip_name,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import SEPARATOR_TABLE
from core_pdf.impl.types import PdfName, PdfString

PdfName_of = PdfName.of

ContentOperand: TypeAlias = CachedPdfObject | InlineImage
ContentOperands: TypeAlias = tuple[ContentOperand, ...]
ContentOperation: TypeAlias = tuple[str, ContentOperands]


internal_INLINE_IMAGE_MARKER_RE = re.compile(rb"[%(/<>\[\]]|BI")


def internal_next_inline_image(
    raw_bytes: bytes,
    pos: int,
    data_len: int,
) -> int | None:
    """Find a top-level BI token, ignoring names, strings and containers."""
    container_depth = 0
    while match := internal_INLINE_IMAGE_MARKER_RE.search(raw_bytes, pos):
        marker = match.start()
        token = match.group()
        if token == b"%":
            pos = skip_comment(raw_bytes, marker, data_len)
            continue
        if token == b"(":
            pos = skip_literal_string(raw_bytes, marker, data_len)
            continue
        if token == b"<":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 60:
                container_depth += 1
                pos = marker + 2
            else:
                pos = skip_hex_string(raw_bytes, marker, data_len)
            continue
        if token == b">":
            if marker + 1 < data_len and raw_bytes[marker + 1] == 62:
                container_depth = max(0, container_depth - 1)
                pos = marker + 2
            else:
                pos = marker + 1
            continue
        if token == b"[":
            container_depth += 1
            pos = marker + 1
            continue
        if token == b"]":
            container_depth = max(0, container_depth - 1)
            pos = marker + 1
            continue
        if token == b"/":
            pos = skip_name(raw_bytes, marker, data_len)
            continue
        after = match.end()
        delimited = bool(
            (marker == 0 or SEPARATOR_TABLE[raw_bytes[marker - 1]])
            and (after == data_len or SEPARATOR_TABLE[raw_bytes[after]])
        )
        if not container_depth and delimited:
            return after
        pos = after
    return None


OperationHandler: TypeAlias = Callable[[ContentOperands, int], None]


def dispatch_operations(
    lexer: PdfLexer,
    get_handler: Callable[[str], OperationHandler | None],
    depth: int,
    *,
    handlers_reject_unknown: bool = True,
) -> None:
    """Tokenize `lexer` and drive each operator through `get_handler`.

    `handlers_reject_unknown` states whether `get_handler` returns None for a
    word that is not a real operator. Inline-image recovery uses it to tell an
    operator boundary from arbitrary image bytes; callers that record every
    word indiscriminately must pass False.
    """
    operands: list[ContentOperand] = []

    def append_operand(value: ContentOperand) -> None:
        if len(operands) < 16:
            operands.append(value)

    raw_data = lexer.raw_data
    data_len = lexer.data_len
    raw_bytes: bytes | memoryview
    source_bytes = full_source_bytes(raw_data)
    raw_bytes = source_bytes if source_bytes is not None else raw_data

    should_decipher = lexer.decipher is not None and lexer.current_obj_num is not None

    pos = lexer.pos
    while pos < data_len:
        pos = lexer.skip_ignored_at(pos)
        if pos >= data_len:
            break
        byte = raw_bytes[pos]

        if not SEPARATOR_TABLE[byte]:
            scanned = lexer.scan_word_at(pos, skip_ignored=False)
            if scanned is None:
                break
            raw_key, pos = scanned

            if raw_key:
                if is_number_word_bytes(raw_key):
                    append_operand(float(raw_key) if b"." in raw_key else int(raw_key))
                    continue

                if raw_key == b"BI":
                    lexer.pos = pos
                    try:
                        image = parse_inline_image(lexer)
                    except PdfParseError as exc:
                        message = str(exc)
                        if message in (
                            "unterminated inline image",
                            "unterminated inline image data",
                            "inline image keys must be names",
                            "expected inline image data separator",
                        ):
                            recovered_pos = recover_inline_image_position(
                                lexer,
                                pos,
                                (lambda token: get_handler(token.decode("latin-1")) is not None)
                                if handlers_reject_unknown
                                else None,
                            )
                            if recovered_pos is None:
                                if message == "unterminated inline image data":
                                    pos = data_len
                                    break
                                raise
                            pos = recovered_pos
                            operands.clear()
                            continue
                        raise
                    pos = lexer.pos
                    append_operand(image)
                    handler = get_handler("BI")
                    if handler is not None:
                        lexer.pos = pos
                        handler(tuple(operands), depth)
                    operands.clear()
                    continue

                if raw_key in (
                    b"R",
                    b"obj",
                    b"endobj",
                    b"stream",
                    b"endstream",
                ):
                    operands.clear()
                    continue
                op_name = cast(str, lexer.parse_keyword(raw_key))
                handler = get_handler(op_name)

                if handler is not None:
                    lexer.pos = pos
                    handler(tuple(operands), depth)
                operands.clear()
                continue

        lexer.pos = pos
        if byte == 91:
            operand_start = pos
            try:
                append_operand(cast(ContentOperand, lexer.parse_array()))
            except PdfParseError as exc:
                if str(exc) == "unterminated array" and lexer.pos >= data_len:
                    pos = data_len
                    break
                if lexer.pos > operand_start:
                    pos = lexer.pos
                    continue
                raise
            pos = lexer.pos
            continue
        if byte == 60:
            if pos + 1 < data_len and raw_bytes[pos + 1] == 60:
                operand_start = pos
                try:
                    append_operand(cast(ContentOperand, lexer.parse_dictionary_or_stream()))
                except PdfParseError as exc:
                    if lexer.pos >= data_len or str(exc) == "unexpected end of PDF input":
                        pos = data_len
                        break
                    if lexer.pos > operand_start:
                        pos = lexer.pos
                        continue
                    raise
            else:
                raw_string = lexer.read_hex_string()
                if should_decipher:
                    raw_string = lexer.apply_decipher(raw_string)
                value = PdfString(raw_string, is_literal=False)
                append_operand(value)
            pos = lexer.pos
            continue
        if byte == 40:
            raw_string = lexer.read_string()
            if should_decipher:
                raw_string = lexer.apply_decipher(raw_string)
            append_operand(PdfString(raw_string, is_literal=True))
            pos = lexer.pos
            continue
        if byte == 47:
            append_operand(PdfName_of(lexer.read_name()))
            pos = lexer.pos
            continue
        if byte == 62:
            pos = pos + 2 if pos + 1 < data_len and raw_bytes[pos + 1] == 62 else pos + 1
            continue
        if byte == 93:
            pos += 1
            continue

        pos += 1

    lexer.pos = pos


def iter_content_operations(lexer: PdfLexer) -> Iterator[ContentOperation]:
    results: list[ContentOperation] = []

    def get_handler(op_name: str) -> OperationHandler:
        def collect(operands: ContentOperands, _depth: int) -> None:
            results.append((op_name, operands))

        return collect

    dispatch_operations(lexer, get_handler, 0, handlers_reject_unknown=False)
    yield from results


def validate_inline_images(data: bytes | memoryview) -> None:
    """Validate inline-image boundaries without disabling normal stream recovery."""
    raw_bytes = full_source_bytes(data)
    if raw_bytes is None:
        raw_bytes = bytes(data)
    data_len = len(raw_bytes)
    pos = 0
    lexer = PdfLexer(raw_bytes)
    while (after := internal_next_inline_image(raw_bytes, pos, data_len)) is not None:
        lexer.pos = after
        parse_inline_image(lexer)
        pos = lexer.pos


__all__ = (
    "ContentOperand",
    "ContentOperands",
    "ContentOperation",
    "dispatch_operations",
    "iter_content_operations",
    "validate_inline_images",
)
