"""Version 0 of the public local core-pdf capability API.

Specialist records and operations live in :mod:`core_pdf.api.v0.models` and
:mod:`core_pdf.api.v0.operations`; the package root contains only entry points
and cross-domain primitives.
"""

from .document import PdfDocument, PdfEditor, PdfPage
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
