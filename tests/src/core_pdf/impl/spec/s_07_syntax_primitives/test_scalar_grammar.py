# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl._impl.runtime import scalars
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float, parse_int


@pytest.mark.parametrize("token", [b"1_000", b" 12", b"12 ", "１２"])
def test_pdf_integer_tokens_do_not_use_host_integer_extensions(token: object) -> None:
    assert parse_int(token) is None
    assert scalars.parse_int(token) is not None


@pytest.mark.parametrize("token", [b"1e2", b"NaN", b"Infinity", b"1_000.0"])
def test_pdf_real_tokens_do_not_use_host_float_extensions(token: object) -> None:
    assert parse_float(token, default=None) is None
    assert scalars.parse_float(token, default=None) is not None


@pytest.mark.parametrize(("token", "expected"), [(b"+.5", 0.5), (b"-2.", -2.0), (b"0.25", 0.25)])
def test_pdf_real_tokens_follow_the_decimal_grammar(token: bytes, expected: float) -> None:
    assert parse_float(token) == expected
