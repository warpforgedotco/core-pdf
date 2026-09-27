# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable

from core_pdf_spec.exceptions import (
    PdfDecryptionError,
    PdfError,
    PdfParseError,
    PdfPasswordError,
    PdfUnsupportedError,
)


class ExtractionCancelled(RuntimeError):
    def __init__(self) -> None:
        super().__init__("PDF extraction was cancelled")


class ExtractionScope:
    __slots__ = ("_cancelled",)

    def __init__(
        self,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._cancelled = cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise ExtractionCancelled()


class PdfSourceError(PdfError):
    pass


class PdfEmptySourceError(PdfSourceError): ...


class PdfContractError(PdfError, TypeError):
    pass


class PdfRasterTooLargeError(PdfError, ValueError):
    pass


class PdfDocumentClosedError(PdfError, ValueError):
    pass


__all__ = (
    "ExtractionCancelled",
    "ExtractionScope",
    "PdfContractError",
    "PdfDecryptionError",
    "PdfDocumentClosedError",
    "PdfEmptySourceError",
    "PdfError",
    "PdfParseError",
    "PdfPasswordError",
    "PdfRasterTooLargeError",
    "PdfSourceError",
    "PdfUnsupportedError",
)
