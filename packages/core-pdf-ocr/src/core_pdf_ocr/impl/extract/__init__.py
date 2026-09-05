# SPDX-License-Identifier: AGPL-3.0-only
"""OCR-aware extraction entry points."""

from core_pdf_ocr.impl.extract.pipeline import extract_page
from core_pdf_ocr.impl.extract.selection import extract_document

__all__ = ("extract_document", "extract_page")
