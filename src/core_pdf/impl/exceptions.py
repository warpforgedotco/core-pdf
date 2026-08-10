# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class PdfError(Exception):
    pass


class PdfSourceError(PdfError):
    pass


class PdfParseError(PdfError):
    pass


class PdfUnsupportedError(PdfError):
    pass


class PdfContractError(PdfError, TypeError):
    """Raised when an internal typed extraction contract is violated."""


class PdfRasterTooLargeError(PdfError, ValueError):
    pass


class PdfDocumentClosedError(PdfError, ValueError):
    """Raised when an operation is attempted on a closed document."""
