# SPDX-License-Identifier: AGPL-3.0-only
"""pdfminer.six-compatible APIs backed by core-pdf."""

from __future__ import annotations

from core_pdf.impl.integrations.pdfminer.high_level import extract_pages, extract_text

__version__ = "20260107"

__all__ = ("__version__", "extract_pages", "extract_text")
