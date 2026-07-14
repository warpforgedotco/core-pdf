# SPDX-License-Identifier: AGPL-3.0-only
"""Core PDF exception hierarchy."""

from __future__ import annotations


class PdfError(Exception):
    """Base class for all core_pdf errors."""


class PdfSourceError(PdfError):
    """Raised when a PDF source cannot be opened or normalized."""


class PdfParseError(PdfError):
    """Raised when the input does not conform to the PDF grammar."""


class PdfUnsupportedError(PdfError):
    """Raised for valid PDF features that are not implemented yet."""
