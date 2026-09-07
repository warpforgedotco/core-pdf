# SPDX-License-Identifier: AGPL-3.0-only
"""Reader text-string recovery beyond the PDF-defined encodings."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_syntax_primitives.text_string import (
    decode_pdf_text_string as decode_spec_text_string,
)
from core_pdf.impl.types import PdfString


def decode_pdf_text_string(data: bytes | memoryview) -> str:
    if type(data) is memoryview:
        data = data.tobytes()
    if data.startswith(b"\xff\xfe"):
        try:
            return data[2:].decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16LE data") from exc
    return decode_spec_text_string(data)


def parse_text_string(value: object) -> str | None:
    if isinstance(value, PdfString):
        return decode_pdf_text_string(value.data)
    if isinstance(value, bytes):
        return decode_pdf_text_string(value)
    return value if isinstance(value, str) else None
