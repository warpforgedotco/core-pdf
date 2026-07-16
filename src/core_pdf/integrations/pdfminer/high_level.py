# SPDX-License-Identifier: AGPL-3.0-only
"""Public pdfminer.six-compatible high-level extraction functions."""

from __future__ import annotations

from core_pdf.impl.integrations.pdfminer.high_level import extract_pages, extract_text

__all__ = ("extract_pages", "extract_text")
