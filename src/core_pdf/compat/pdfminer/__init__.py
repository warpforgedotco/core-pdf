# SPDX-License-Identifier: AGPL-3.0-only
"""pdfminer.six-compatible public adapters."""

from __future__ import annotations

from core_pdf.compat.pdfminer.high_level import extract_text
from core_pdf.compat.pdfminer.unstructured import iter_unstructured_region_layouts

__all__ = ["extract_text", "iter_unstructured_region_layouts"]
