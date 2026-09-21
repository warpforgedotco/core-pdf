from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import ContentSink
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
    iter_content_operations,
    validate_content_operands,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax_primitives.content_operators import (
    CONTENT_OPERATOR_HANDLERS,
    CONTENT_OPERATOR_SIGNATURES,
    INLINE_IMAGE_DATA_OPERATORS,
    PDF_CONTENT_OPERATOR_BYTES,
)
from core_pdf_spec.types import PdfString


def test_operator_vocabulary_separates_fixed_color_and_inline_data_operators() -> None:
    assert CONTENT_OPERATOR_HANDLERS.keys() - CONTENT_OPERATOR_SIGNATURES.keys() == {
        "SC",
        "SCN",
        "sc",
        "scn",
    }
    assert not CONTENT_OPERATOR_SIGNATURES.keys() - CONTENT_OPERATOR_HANDLERS.keys()
    assert {b"ID", b"EI"} == INLINE_IMAGE_DATA_OPERATORS
    assert {name.encode("latin-1") for name in CONTENT_OPERATOR_HANDLERS} | {
        b"ID",
        b"EI",
    } == PDF_CONTENT_OPERATOR_BYTES
    assert "ID" not in CONTENT_OPERATOR_HANDLERS
    assert "EI" not in CONTENT_OPERATOR_HANDLERS
    assert all(
        callable(getattr(ContentInterpreter, name)) for name in CONTENT_OPERATOR_HANDLERS.values()
    )


@pytest.mark.parametrize(
    "content",
    [b"1 F", b"(text) 2 '", b'1 (text) 2 "', b"/Tag 1 DP", b"1 0 0 cm"],
)
def test_catalog_fixed_signatures_reject_wrong_operands(content: bytes) -> None:
    with pytest.raises(PdfParseError):
        for name, operands in iter_content_operations(PdfLexer(content)):
            validate_content_operands(name, operands)


def test_operator_aliases_keep_overridden_handler_bindings() -> None:
    calls: list[tuple[str, ContentOperands, int]] = []

    class State(ContentInterpreter):
        def op_paint_fill(self, operands: ContentOperands, depth: int) -> None:
            calls.append(("fill", operands, depth))

        def op_quote(self, operands: ContentOperands, depth: int) -> None:
            calls.append(("quote", operands, depth))

        def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
            calls.append(("double_quote", operands, depth))

    resolver = ObjectResolver(b"", {})
    state = State(resolver, cast(ContentSink, object()), cast(Any, None))
    try:
        for name, operands in iter_content_operations(PdfLexer(b"f F (one) ' 1 2 (two) \"")):
            assert state.execute_operation(name, operands, 3) is None
    finally:
        resolver.close()
    assert calls == [
        ("fill", (), 3),
        ("fill", (), 3),
        ("quote", (PdfString(b"one"),), 3),
        ("double_quote", (1, 2, PdfString(b"two")), 3),
    ]
