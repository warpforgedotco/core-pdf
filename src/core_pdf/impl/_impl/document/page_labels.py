# SPDX-License-Identifier: AGPL-3.0-only
"""Reader defaults for malformed page-label dictionaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf.impl._impl.document.recovery.text_strings import parse_text_string
from core_pdf.impl.spec.s_07_document.document_labels import PageLabelStyle
from core_pdf.impl.spec.s_07_document.document_labels import (
    format_page_label as format_spec_page_label,
)
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.types import PdfName


def format_page_label(
    spec: PdfDict,
    page_offset: int,
    resolve: Callable[[object], object],
    *,
    decode_prefix: Callable[[object], str | None] = parse_text_string,
) -> str:
    style = normalize_pdf_name(resolve(spec.get("S")))
    prefix = decode_prefix(resolve(spec.get("P"))) or ""
    start = resolve(spec.get("St"))
    normalized: PdfDict = {
        "P": cast(PdfObject, prefix),
        "St": start if type(start) is int and start > 0 else 1,
    }
    if style is not None and style in PageLabelStyle:
        normalized["S"] = PdfName.of(style)
    return format_spec_page_label(normalized, page_offset, lambda value: value)
