# SPDX-License-Identifier: AGPL-3.0-only
"""Native page-label and page-tree classification helpers."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from core_pdf.impl.engine.spec.s_07_syntax.resolver_values import PdfValueResolver
from core_pdf.impl.engine.spec.s_07_syntax.text_string import decode_pdf_text_string
from core_pdf.impl.engine.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.engine.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key

ResolveFn = Callable[[object], object]
MAX_PAGE_TREE_DEPTH = 100


class PageLabelStyle(StrEnum):
    """PDF page-label numbering styles."""

    LOWER_ROMAN = "r"
    UPPER_ROMAN = "R"
    LOWER_ALPHA = "a"
    UPPER_ALPHA = "A"
    DECIMAL = "D"


def resolve_page_tree_node_type(resolver: PdfValueResolver, node: PdfDict) -> str | None:
    """Resolve a page-tree node type, falling back to structural inference."""
    node_type = resolver.resolve_name(lookup_dict_key(node, "Type"))
    if node_type is not None:
        return node_type
    inferred = infer_page_tree_node_type(node, include_page_properties=False)
    if inferred is not None:
        return inferred
    # A parent pointer is the defining structural signal for an untyped leaf.
    # Content/media/resource keys alone also occur in Form XObjects and other
    # dictionaries and must not promote those objects into the page tree.
    if lookup_dict_key(node, "Parent") is not None:
        return "Page"
    return None


def format_page_label(
    spec: PdfDict,
    page_offset: int,
    resolve: ResolveFn,
) -> str:
    style = normalize_page_label_style(resolve(lookup_dict_key(spec, "S")))
    prefix = decode_page_label_prefix(resolve(lookup_dict_key(spec, "P")))
    start = resolve(lookup_dict_key(spec, "St"))
    number = (start if type(start) is int and start > 0 else 1) + page_offset

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
    except ValueError:
        return None


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


def infer_page_tree_node_type(
    node: PdfDict,
    *,
    include_page_properties: bool = True,
) -> str | None:
    if lookup_dict_key(node, "Kids") is not None:
        return "Pages"
    if lookup_dict_key(node, "Count") is not None:
        return "Pages"
    if not include_page_properties:
        return None
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
    repeat_count, remainder = divmod(value - 1, 26)
    return chr(97 + remainder) * (repeat_count + 1)


__all__ = (
    "decode_page_label_prefix",
    "format_alpha",
    "format_page_label",
    "format_roman",
    "infer_page_tree_node_type",
    "normalize_page_label_style",
)
