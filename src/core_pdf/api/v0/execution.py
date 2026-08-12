"""Small local execution primitives used by capability operations."""

from __future__ import annotations

from collections.abc import Iterable
from threading import Event
from typing import TYPE_CHECKING

from .errors import OperationCancelled
from .models import AnalysisReport, IncrementalAnalysisPlan
from .types import ExecutionContext

if TYPE_CHECKING:
    from .document import PdfDocument
    from .operations.base import AnalysisOperation


class AnalysisCache:
    """Process-local deterministic cache for analysis reports."""

    __slots__ = ("_hits", "_misses", "_reports")

    def __init__(self) -> None:
        self._reports: dict[
            tuple[str, str, str], tuple[AnalysisReport, frozenset[int], frozenset[tuple[int, int]]]
        ] = {}
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def get(self, operation: AnalysisOperation, document: PdfDocument) -> AnalysisReport | None:
        fingerprint = document.fingerprint().document_sha256
        entry = self._reports.get((operation.operation_id, operation.version, fingerprint))
        return entry[0] if entry is not None else None

    def run(
        self,
        operation: AnalysisOperation,
        document: PdfDocument,
        context: ExecutionContext,
        options: dict[str, object] | None = None,
        *,
        dependency_pages: Iterable[int] = (),
        dependency_objects: Iterable[tuple[int, int]] = (),
    ) -> AnalysisReport:
        cached = self.get(operation, document)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        report = operation.run(document, context, options)
        if not dependency_pages:
            dependency_pages = tuple(
                sorted(
                    {
                        page
                        for finding in report.findings
                        for page in (finding.page_number,)
                        if page is not None
                    }
                )
            )
        if not dependency_objects:
            dependency_objects = tuple(
                sorted(
                    {
                        (source.object_number, source.generation or 0)
                        for finding in report.findings
                        for evidence in finding.evidence
                        for source in (evidence.source,)
                        if source.object_number is not None
                    }
                )
            )
        fingerprint = document.fingerprint().document_sha256
        self._reports[(operation.operation_id, operation.version, fingerprint)] = (
            report,
            frozenset(dependency_pages),
            frozenset(dependency_objects),
        )
        return report

    def invalidate(self, plan: IncrementalAnalysisPlan) -> int:
        if plan.full_document_scan:
            count = len(self._reports)
            self._reports.clear()
            return count
        affected_pages = set(plan.affected_pages)
        affected_objects = set(plan.affected_objects) | set(plan.revision_objects)
        evicted = 0
        for key, (_, pages, objects) in tuple(self._reports.items()):
            if not pages and not objects or pages & affected_pages or objects & affected_objects:
                del self._reports[key]
                evicted += 1
        return evicted


class LocalCancellationToken:
    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled


class LocalExecutionContext:
    __slots__ = ("cancellation",)

    def __init__(self, *, cancellation: LocalCancellationToken | None = None) -> None:
        self.cancellation = cancellation or LocalCancellationToken()


__all__ = ("AnalysisCache", "LocalCancellationToken", "LocalExecutionContext")
