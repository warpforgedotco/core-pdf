# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_number_pairs
from core_postscript.calculator import compile_calculator


def bounds(value: object, kind: str) -> tuple[tuple[float, float], ...]:
    bounds = require_pdf_number_pairs(value, f"invalid calculator {kind}")
    if any(lower > upper for lower, upper in bounds):
        raise ValueError(f"invalid calculator {kind}")
    return bounds


def compile_calculator_function(function: PdfStream) -> Callable[..., tuple[float, ...]]:
    domains = bounds(function.dictionary.get("Domain"), "domain")
    ranges = bounds(function.dictionary.get("Range"), "range")
    return compile_calculator(function.data, domains, ranges)


__all__ = ()
