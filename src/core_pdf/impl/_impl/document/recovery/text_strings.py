# SPDX-License-Identifier: AGPL-3.0-only
"""Reader text-string recovery beyond the PDF-defined encodings."""

from __future__ import annotations

from core_pdf.impl.types import PdfString
from core_pdf_spec.s_07_syntax_primitives.text_string import (
    decode_pdf_text_string as decode_spec_text_string,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext


def decode_pdf_text_string(
    data: bytes | memoryview, *, context: SemanticContext | None = None
) -> str:
    if type(data) is memoryview:
        data = data.tobytes()
    if data.startswith(b"\xff\xfe"):
        try:
            return data[2:].decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16LE data") from exc
    # Readers have historically accepted under-declared Unicode text strings.
    # Keep that policy explicit while strict spec callers use the selected version.
    if context is not None and (
        context.version is None
        or not context.version.recognized
        or (context.version < PdfVersion(2, 0) and data.startswith(b"\xef\xbb\xbf"))
        or (context.version < PdfVersion(1, 2) and data.startswith(b"\xfe\xff"))
    ):
        context = None
    return decode_spec_text_string(data, context=context)


def parse_text_string(value: object) -> str | None:
    if isinstance(value, PdfString):
        return decode_pdf_text_string(value.data)
    if isinstance(value, bytes):
        return decode_pdf_text_string(value)
    return value if isinstance(value, str) else None
