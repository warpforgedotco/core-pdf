# SPDX-License-Identifier: AGPL-3.0-only
"""Strict tokenization and dispatch of PDF content-stream operations.

Token parsing advances the lexer only through the token being read. On failure
its cursor identifies the failure location; readers may recover outside this
module and call the primitive again at a selected boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeAlias, cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.inline_images import InlineImage, parse_inline_image
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.types import CachedPdfObject
from core_pdf_spec.s_07_syntax_primitives.content_operators import (
    CONTENT_OPERATOR_HANDLERS,
    CONTENT_OPERATOR_SIGNATURES,
)
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    is_number_word_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
    skip_name,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import SEPARATOR_TABLE
from core_pdf_spec.types import PdfName, PdfString

ContentOperand: TypeAlias = CachedPdfObject | InlineImage
ContentOperands: TypeAlias = tuple[ContentOperand, ...]
ContentOperation: TypeAlias = tuple[str, ContentOperands]
OperationHandler: TypeAlias = Callable[[ContentOperands, int], None]


@dataclass(frozen=True, slots=True)
class ContentToken:
    """An operand or operator, with its exact byte start position."""

    start: int
    value: ContentOperand
    is_operator: bool = False


def parse_content_token(lexer: PdfLexer) -> ContentToken | None:
    """Consume one token, or return None at EOF; malformed input raises."""
    lexer.skip_ignored()
    start = lexer.pos
    if start >= lexer.data_len:
        return None
    byte = lexer.raw_data[start]
    if not SEPARATOR_TABLE[byte]:
        scanned = lexer.scan_word_at(start, skip_ignored=False)
        assert scanned is not None
        word, lexer.pos = scanned
        if is_number_word_bytes(word):
            return ContentToken(start, float(word) if b"." in word else int(word))
        if word == b"BI":
            return ContentToken(start, parse_inline_image(lexer))
        if word in (b"true", b"false", b"null"):
            return ContentToken(start, cast(ContentOperand, lexer.parse_keyword(word)))
        return ContentToken(start, word.decode("latin-1"), is_operator=True)
    if byte == 91:
        return ContentToken(start, cast(ContentOperand, lexer.parse_array()))
    if byte == 60 and lexer.raw_data[start : start + 2] == b"<<":
        return ContentToken(start, cast(ContentOperand, lexer.parse_dictionary_or_stream()))
    if byte in (40, 60):
        data = lexer.read_string() if byte == 40 else lexer.read_hex_string()
        if lexer.decipher is not None and lexer.current_obj_num is not None:
            data = lexer.apply_decipher(data)
        return ContentToken(start, PdfString(data, is_literal=byte == 40))
    if byte == 47:
        return ContentToken(start, PdfName.of(lexer.read_name()))
    raise PdfParseError("unexpected delimiter in content stream")


def validate_content_operands(operator: str, operands: ContentOperands) -> None:
    """Check the fixed PDF operator signature before any state transition."""
    signature = CONTENT_OPERATOR_SIGNATURES.get(operator)
    if signature is None:
        return
    if len(operands) != len(signature):
        raise PdfParseError(f"{operator} requires {len(signature)} operands")
    for category, operand in zip(signature, operands, strict=True):
        valid = False
        if category == "n":
            valid = type(operand) in (int, float) and math.isfinite(cast(float, operand))
        elif category == "i":
            valid = type(operand) is int
        elif category == "/":
            valid = isinstance(operand, PdfName)
        elif category == "s":
            valid = isinstance(operand, PdfString)
        elif category == "a":
            valid = isinstance(operand, (list, tuple))
        elif category == "p":
            valid = isinstance(operand, (PdfName, dict))
        elif category == "I":
            valid = isinstance(operand, InlineImage)
        if not valid:
            raise PdfParseError(f"invalid {operator} operand")
    if operator in {"TJ", "d"}:
        array = cast(list[ContentOperand], operands[0])
        for value in array:
            if type(value) in (int, float) and math.isfinite(cast(float, value)):
                if operator == "d" and cast(float, value) < 0:
                    raise PdfParseError("negative dash length")
                continue
            if operator == "TJ" and isinstance(value, PdfString):
                continue
            raise PdfParseError(f"invalid {operator} array entry")
        if operator == "d" and array and not any(array):
            raise PdfParseError("dash array cannot contain only zero lengths")
    if operator in {"J", "j", "Tr"}:
        upper = 7 if operator == "Tr" else 2
        if not 0 <= cast(int, operands[0]) <= upper:
            raise PdfParseError(f"invalid {operator} value")
    if operator == "w" and cast(float, operands[0]) < 0:
        raise PdfParseError("line width must not be negative")
    if operator == "M" and cast(float, operands[0]) < 1:
        raise PdfParseError("miter limit must be at least one")


@dataclass(slots=True)
class ContentOperationState:
    """Compatibility nesting retained when execution suspends for a child stream."""

    compatibility_depth: int = 0


def dispatch_operations(
    lexer: PdfLexer,
    get_handler: Callable[[str], OperationHandler | None],
    depth: int,
    *,
    operation_state: ContentOperationState | None = None,
) -> None:
    """Execute complete operations; unknown operators require a BX/EX scope.

    The lexer is positioned after the complete operation before calling its
    handler, so a suspended nested stream can resume without replaying it.
    """
    operands: list[ContentOperand] = []
    state = operation_state if operation_state is not None else ContentOperationState()
    while (token := parse_content_token(lexer)) is not None:
        if isinstance(token.value, InlineImage):
            operands.append(token.value)
            op_name = "BI"
        elif not token.is_operator:
            operands.append(token.value)
            continue
        else:
            op_name = cast(str, token.value)
        if op_name not in CONTENT_OPERATOR_HANDLERS:
            if not state.compatibility_depth:
                raise PdfParseError(f"unknown content operator: {op_name}")
            operands.clear()
            continue
        if op_name == "BX":
            state.compatibility_depth += 1
        elif op_name == "EX":
            if not state.compatibility_depth:
                raise PdfParseError("unmatched EX operator")
            state.compatibility_depth -= 1
        handler = get_handler(op_name)
        if handler is None:
            raise PdfParseError(f"unsupported content operator: {op_name}")
        validate_content_operands(op_name, tuple(operands))
        handler(tuple(operands), depth)
        operands.clear()
    if operands:
        raise PdfParseError("content stream ends with operands")
    if state.compatibility_depth:
        raise PdfParseError("unterminated compatibility section")


def iter_content_operations(lexer: PdfLexer) -> Iterator[ContentOperation]:
    results: list[ContentOperation] = []

    def get_handler(op_name: str) -> OperationHandler:
        def collect(operands: ContentOperands, depth: int) -> None:
            results.append((op_name, operands))

        return collect

    dispatch_operations(lexer, get_handler, 0)
    yield from results


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


def validate_inline_images(data: bytes | memoryview) -> None:
    """Validate inline-image boundaries without executing content operators."""
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
    "ContentToken",
    "ContentOperationState",
    "OperationHandler",
    "parse_content_token",
    "validate_content_operands",
    "dispatch_operations",
    "iter_content_operations",
    "validate_inline_images",
)
