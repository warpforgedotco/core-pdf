# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations


class PdfError(Exception):
    pass


class PdfParseError(PdfError):
    pass


class PdfDecryptionError(PdfError):
    pass


class PdfUnsupportedError(PdfError):
    pass


__all__ = (
    "PdfDecryptionError",
    "PdfError",
    "PdfParseError",
    "PdfUnsupportedError",
)
