"""Public local capability API for core-pdf."""

from __future__ import annotations

from .document import PdfDocument, PdfPage
from .editor import PdfEditor
from .errors import ApiError, DocumentClosed, InvalidRequest, OperationCancelled
from .execution import AnalysisCache, LocalCancellationToken, LocalExecutionContext
from .models import CoordinateOrigin, CoordinateSpace, Point, Rect, Severity, SourceRef
from .types import (
    CancellationToken,
    ExecutionContext,
    PageSelection,
    PdfInput,
    ReadableSource,
    SignatureProvider,
)

__all__ = (
    "AnalysisCache",
    "ApiError",
    "CancellationToken",
    "CoordinateOrigin",
    "CoordinateSpace",
    "DocumentClosed",
    "ExecutionContext",
    "InvalidRequest",
    "LocalCancellationToken",
    "LocalExecutionContext",
    "OperationCancelled",
    "PageSelection",
    "PdfDocument",
    "PdfEditor",
    "PdfInput",
    "PdfPage",
    "Point",
    "ReadableSource",
    "Rect",
    "Severity",
    "SignatureProvider",
    "SourceRef",
)
