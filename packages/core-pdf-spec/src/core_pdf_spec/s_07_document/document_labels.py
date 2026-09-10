# SPDX-License-Identifier: AGPL-3.0-only
"""PDF page-label numbering, ISO 32000-2, 12.4.2."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import normalize_pdf_name, parse_text_string

ResolveFn = Callable[[object], object]


class PageLabelStyle(StrEnum):
    """PDF page-label numbering styles."""

    LOWER_ROMAN = "r"
    UPPER_ROMAN = "R"
    LOWER_ALPHA = "a"
    UPPER_ALPHA = "A"
    DECIMAL = "D"


def format_page_label(
    spec: PdfDict,
    page_offset: int,
    resolve: ResolveFn,
) -> str:
    style = normalize_page_label_style(resolve(spec.get("S")))
    raw_prefix = resolve(spec.get("P"))
    prefix = parse_text_string(raw_prefix)
    if raw_prefix is not None and prefix is None:
        raise ValueError("invalid page label prefix")
    prefix = prefix or ""
    start = resolve(spec.get("St"))
    if start is not None and (type(start) is not int or start <= 0):
        raise ValueError("invalid page label start")
    number = (start if type(start) is int else 1) + page_offset

    match style:
        case PageLabelStyle.LOWER_ROMAN:
            return prefix + format_roman(number).lower()
        case PageLabelStyle.UPPER_ROMAN:
            return prefix + format_roman(number).upper()
        case PageLabelStyle.LOWER_ALPHA:
            return prefix + format_alpha(number).lower()
        case PageLabelStyle.UPPER_ALPHA:
            return prefix + format_alpha(number).upper()
        case PageLabelStyle.DECIMAL:
            return prefix + str(number)
        case _:
            return prefix


def normalize_page_label_style(value: object) -> PageLabelStyle | None:
    style = normalize_pdf_name(value)
    try:
        return PageLabelStyle(style) if style is not None else None
    except ValueError as error:
        raise ValueError("invalid page label style") from error


numerals = (
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
)


def format_roman(value: int) -> str:
    if value <= 0:
        return ""

    result: list[str] = []
    for amount, numeral in numerals:
        while value >= amount:
            result.append(numeral)
            value -= amount
    return "".join(result)


def format_alpha(value: int) -> str:
    if value <= 0:
        return ""
    repeat_count, remainder = divmod(value - 1, 26)
    return chr(97 + remainder) * (repeat_count + 1)


__all__ = (
    "PageLabelStyle",
    "ResolveFn",
    "format_alpha",
    "format_page_label",
    "format_roman",
    "normalize_page_label_style",
)
