# SPDX-License-Identifier: AGPL-3.0-only
"""Operand handling of ``dispatch_operations`` observed through a recording ``m`` handler."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from core_pdf.impl.spec.s_07_content.operations import dispatch_operations
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer


def dispatch_m_operands(source: bytes) -> list[object]:
    """Run ``source`` through the dispatcher and return what the ``m`` operator received."""
    received: list[object] = []

    def move_to(operands: Sequence[object], internal_depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    cast(Any, dispatch_operations)(
        PdfLexer(source), {"m": move_to}, None, fast_handlers, {}, None, 0
    )
    return received


def test_leading_dot_number_is_passed_to_operator() -> None:
    assert dispatch_m_operands(b".5 1 m") == [0.5, 1]


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (b"0.123", 0.123),
        (b"-0.123", -0.123),
        (b"3.728", 3.728),
        (b"7.6993", 7.6993),
        (b"-12.345", -12.345),
        (b"123.45", 123.45),
    ],
)
def test_three_decimal_number_is_passed_to_operator(token: bytes, expected: float) -> None:
    assert dispatch_m_operands(token + b" 1 m") == [expected, 1]


def test_unsupported_operator_does_not_leak_operands() -> None:
    assert dispatch_m_operands(b"99 UNKNOWN 1 2 m") == [1, 2]


@pytest.mark.parametrize("invalid_operator", [b"12foo", b".x", b"+bad", b".", b"+", b"-"])
def test_number_shaped_unsupported_operator_does_not_leak_operands(
    invalid_operator: bytes,
) -> None:
    assert dispatch_m_operands(b"99 " + invalid_operator + b" 1 2 m") == [1, 2]
