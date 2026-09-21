# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf.impl._impl.document.recovery.text_strings import parse_text_string
from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfName
from core_pdf_spec.s_07_document.document_labels import PageLabelStyle
from core_pdf_spec.s_07_document.document_labels import (
    format_page_label as format_spec_page_label,
)
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject


def format_page_label(spec: PdfDict, page_offset: int, resolve: Callable[[object], object]) -> str:
    style = recover_pdf_name(resolve(spec.get("S")))
    prefix = parse_text_string(resolve(spec.get("P"))) or ""
    start = resolve(spec.get("St"))
    normalized: PdfDict = {
        "P": cast(PdfObject, prefix),
        "St": start if type(start) is int and start > 0 else 1,
    }
    if style is not None and style in PageLabelStyle:
        normalized["S"] = PdfName.of(style)
    return format_spec_page_label(normalized, page_offset, lambda value: value)
