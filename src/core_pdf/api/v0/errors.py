"""Stable errors raised by the versioned core-pdf API."""

from __future__ import annotations

from core_pdf.impl.exceptions import PdfDocumentClosedError


class ApiError(Exception):
    """Base class for public API errors."""


class InvalidRequest(ApiError, ValueError):
    """A request or operation option is invalid."""


class OperationCancelled(ApiError):
    """A local operation was cancelled."""


class DocumentClosed(ApiError, PdfDocumentClosedError):
    """An operation was attempted after its document was closed.

    Subclasses the engine's ``PdfDocumentClosedError`` so callers can catch
    closed-document failures from either layer with one exception type.
    """


__all__ = (
    "ApiError",
    "DocumentClosed",
    "InvalidRequest",
    "OperationCancelled",
)
