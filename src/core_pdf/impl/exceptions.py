# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable

from core_pdf_spec.exceptions import PdfDecryptionError as PdfDecryptionError
from core_pdf_spec.exceptions import PdfError as PdfError
from core_pdf_spec.exceptions import PdfParseError as PdfParseError
from core_pdf_spec.exceptions import PdfPasswordError as PdfPasswordError
from core_pdf_spec.exceptions import PdfUnsupportedError as PdfUnsupportedError


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
