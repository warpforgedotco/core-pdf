# SPDX-License-Identifier: AGPL-3.0-only
"""Content-stream tokenization, operation parsing, and operand validation.

Token parsing advances the lexer only through the token being read. On failure
its cursor identifies the failure location; readers may recover outside this
module and call the primitive again at a selected boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.inline_images import InlineImage, parse_inline_image
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.types import CachedPdfObject
from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_integer, require_pdf_number
from core_pdf_spec.s_07_syntax_primitives.content_operators import (
    CONTENT_OPERATOR_SIGNATURES,
)
from core_pdf_spec.s_07_syntax_primitives.numbers import is_number_token
from core_pdf_spec.types import PdfName, PdfString

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.streams import ContentStreamFrame

ContentOperand: TypeAlias = CachedPdfObject | InlineImage
ContentOperands: TypeAlias = tuple[ContentOperand, ...]
ContentOperation: TypeAlias = tuple[str, ContentOperands]
OperationHandler: TypeAlias = Callable[[ContentOperands, int], "ContentStreamFrame | None"]


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
    if not lexer.lexical_rules.separator_table[byte]:
        scanned = lexer.scan_word_at(start, skip_ignored=False)
        assert scanned is not None
        word, lexer.pos = scanned
        if is_number_token(word):
            return ContentToken(
                start,
                lexer.parse_real_token(word) if b"." in word else lexer.parse_integer_token(word),
            )
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
        if category in {"n", "i"}:
            try:
                if category == "n":
                    require_pdf_number(operand, f"invalid {operator} operand")
                else:
                    require_pdf_integer(operand, f"invalid {operator} operand")
            except ValueError as error:
                raise PdfParseError(str(error)) from error
            valid = True
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
            if operator == "TJ" and isinstance(value, PdfString):
                continue
            try:
                number = require_pdf_number(value, f"invalid {operator} array entry")
            except ValueError as error:
                raise PdfParseError(str(error)) from error
            if operator == "d" and number < 0:
                raise PdfParseError("negative dash length")
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


def iter_content_operations(lexer: PdfLexer) -> Iterator[ContentOperation]:
    """Yield complete operations without executing or validating their semantics.

    The lexer advances past each operation before yielding. Interpreter execution
    may then suspend for a child stream and resume at the next parent operation.
    Operator signatures and state constraints belong to execute_operation.
    """
    operands: list[ContentOperand] = []
    while (token := parse_content_token(lexer)) is not None:
        if isinstance(token.value, InlineImage):
            operands.append(token.value)
            op_name = "BI"
        elif not token.is_operator:
            operands.append(token.value)
            continue
        else:
            op_name = cast(str, token.value)
        operation = (op_name, tuple(operands))
        operands.clear()
        yield operation
    if operands:
        raise PdfParseError("content stream ends with operands")


__all__ = (
    "ContentOperand",
    "ContentOperands",
    "ContentOperation",
    "ContentToken",
    "OperationHandler",
    "parse_content_token",
    "validate_content_operands",
    "iter_content_operations",
)
