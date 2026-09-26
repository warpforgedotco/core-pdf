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


class PdfPasswordError(PdfUnsupportedError):
    """The password opens neither as the user nor as the owner password."""


__all__ = (
    "PdfDecryptionError",
    "PdfError",
    "PdfParseError",
    "PdfPasswordError",
    "PdfUnsupportedError",
)
