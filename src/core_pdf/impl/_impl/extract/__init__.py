# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical page and document extraction entry points."""

from core_pdf.impl._impl.extract.pipeline import extract_page
from core_pdf.impl._impl.extract.selection import extract_document

__all__ = (
    "extract_document",
    "extract_page",
)
