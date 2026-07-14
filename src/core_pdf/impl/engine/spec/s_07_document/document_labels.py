# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Callable

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.types import PdfDict

ResolveFn = Callable[[object], object]


def format_page_label(
    spec: PdfDict,
    page_offset: int,
    resolve: ResolveFn,
) -> str:
    style = normalize_page_label_style(resolve(lookup_dict_key(spec, "S")))
    prefix = decode_page_label_prefix(resolve(lookup_dict_key(spec, "P")))
    start = resolve(lookup_dict_key(spec, "St"))
    number = (start if type(start) is int and start > 0 else 1) + page_offset

    if style == "r":
        return prefix + format_roman(number).lower()
    if style == "R":
        return prefix + format_roman(number).upper()
    if style == "a":
        return prefix + format_alpha(number).lower()
    if style == "A":
        return prefix + format_alpha(number).upper()
    if style == "D" or style is None:
        return prefix + str(number)
    return prefix


def normalize_page_label_style(value: object) -> str | None:
    return normalize_pdf_name(value)


def decode_page_label_prefix(value: object) -> str:
    if value is None:
        return ""
    data = getattr(value, "data", None)
    if isinstance(data, bytes):
        return decode_pdf_text_string(data)
    if isinstance(value, bytes):
        return decode_pdf_text_string(value)
    if isinstance(value, str):
        return value
    return ""


def infer_page_tree_node_type(node: PdfDict) -> str | None:
    if lookup_dict_key(node, "Kids") is not None:
        return "Pages"
    if lookup_dict_key(node, "Count") is not None:
        return "Pages"
    for key in ("Contents", "MediaBox", "Resources", "Parent", "Annots"):
        if lookup_dict_key(node, key) is not None:
            return "Page"
    return None


def format_roman(value: int) -> str:
    if value <= 0:
        return ""
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
    result: list[str] = []
    for amount, numeral in numerals:
        while value >= amount:
            result.append(numeral)
            value -= amount
    return "".join(result)


def format_alpha(value: int) -> str:
    if value <= 0:
        return ""
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(97 + remainder))
    chars.reverse()
    return "".join(chars)


__all__ = (
    "decode_page_label_prefix",
    "format_alpha",
    "format_page_label",
    "format_roman",
    "infer_page_tree_node_type",
    "normalize_page_label_style",
)
