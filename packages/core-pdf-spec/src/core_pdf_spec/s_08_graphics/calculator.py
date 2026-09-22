# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_number_array
from core_postscript.calculator import compile_calculator


def bounds(value: object, kind: str) -> tuple[tuple[float, float], ...]:
    values = require_pdf_number_array(value, f"invalid calculator {kind}")
    if not values or len(values) % 2:
        raise ValueError(f"invalid calculator {kind}")
    bounds = tuple(zip(values[::2], values[1::2], strict=True))
    if any(lower > upper for lower, upper in bounds):
        raise ValueError(f"invalid calculator {kind}")
    return bounds


def compile_calculator_function(function: PdfStream) -> Callable[..., tuple[float, ...]]:
    domains = bounds(function.dictionary.get("Domain"), "domain")
    ranges = bounds(function.dictionary.get("Range"), "range")
    return compile_calculator(function.data, domains, ranges)


__all__ = ()
