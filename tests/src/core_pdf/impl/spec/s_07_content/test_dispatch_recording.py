# SPDX-License-Identifier: AGPL-3.0-only
"""Operand handling of ``dispatch_operations`` observed through a recording ``m`` handler."""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_07_content.operations import OperandWindow, dispatch_operations
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer


def dispatch_m_operands(source: bytes) -> list[object]:
    """Run ``source`` through the dispatcher and return what the ``m`` operator received."""
    received: list[object] = []

    def move_to(operands: OperandWindow, internal_depth: int) -> None:
        received.extend(operands)

    dispatch_operations(PdfLexer(source), {"m": move_to}.get, None, 0)
    return received


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (b"0", 0),
        (b"12", 12),
        (b"-5", -5),
        (b"+5", 5),
        (b"123", 123),
        (b"-12", -12),
        (b"1234", 1234),
        (b"-123", -123),
        (b"12345", 12345),
        (b".5", 0.5),
        (b"+.5", 0.5),
        (b"-.5", -0.5),
        (b"1.", 1.0),
        (b"0.123", 0.123),
        (b"-0.123", -0.123),
        (b"3.728", 3.728),
        (b"7.6993", 7.6993),
        (b"-12.345", -12.345),
        (b"123.45", 123.45),
    ],
)
def test_number_is_passed_to_operator(token: bytes, expected: int | float) -> None:
    assert dispatch_m_operands(token + b" 1 m") == [expected, 1]


def test_unsupported_operator_does_not_leak_operands() -> None:
    assert dispatch_m_operands(b"99 UNKNOWN 1 2 m") == [1, 2]


@pytest.mark.parametrize("invalid_operator", [b"12foo", b".x", b"+bad", b".", b"+", b"-"])
def test_number_shaped_unsupported_operator_does_not_leak_operands(
    invalid_operator: bytes,
) -> None:
    assert dispatch_m_operands(b"99 " + invalid_operator + b" 1 2 m") == [1, 2]
