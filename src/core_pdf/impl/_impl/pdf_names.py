# SPDX-License-Identifier: AGPL-3.0-only
"""Reader recovery for names supplied as slash-prefixed text."""

from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf_spec.types import PdfName


def recover_pdf_name(value: object, default: str | None = None) -> str | None:
    name = decoded_name(value, default)
    if not isinstance(value, PdfName) and name is not None and name.startswith("/"):
        return name[1:]
    return name
