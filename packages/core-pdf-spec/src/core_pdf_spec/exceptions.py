# SPDX-License-Identifier: AGPL-3.0-only
"""Errors shared by strict PDF algorithms and their consumers."""

from __future__ import annotations


class PdfError(Exception):
    pass


class PdfParseError(PdfError):
    pass


class PdfDecryptionError(PdfError):
    """Raised when encrypted PDF data fails format-mandated decryption validation."""


class PdfUnsupportedError(PdfError):
    pass


__all__ = (
    "PdfDecryptionError",
    "PdfError",
    "PdfParseError",
    "PdfUnsupportedError",
)
