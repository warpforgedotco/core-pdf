"""Template-method framework shared by the v0 analysis operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, cast

from ...errors import InvalidRequest
from ...execution import LocalExecutionContext
from ...models import AnalysisFinding, AnalysisReport, EvidenceRecord, Rect, Severity
from ...types import ExecutionContext, PageSelection

if TYPE_CHECKING:
    from ...document import PdfDocument, PdfPage


class OperationOptions:
    """Typed accessor over the raw options mapping.

    Raises :class:`InvalidRequest` when an option value cannot be coerced to
    the requested type.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Mapping[str, object] | None) -> None:
        self._raw: dict[str, object] = dict(raw) if raw else {}

    @property
    def raw(self) -> Mapping[str, object]:
        """The raw options mapping, for forwarding to composed operations."""
        return self._raw

    def get_float(self, key: str, default: float) -> float:
        value = self._raw.get(key, default)
        if not isinstance(value, bool) and isinstance(value, (int, float, str)):
            try:
                return float(value)
            except ValueError:
                pass
        raise InvalidRequest(f"option {key!r} must be a number, got {value!r}")

    def get_bool(self, key: str, default: bool) -> bool:
        value = self._raw.get(key, default)
        if isinstance(value, bool):
            return value
        raise InvalidRequest(f"option {key!r} must be a boolean, got {value!r}")

    def get_str(self, key: str, default: str) -> str:
        value = self._raw.get(key, default)
        if isinstance(value, str):
            return value
        raise InvalidRequest(f"option {key!r} must be a string, got {value!r}")

    @property
    def pages(self) -> PageSelection | None:
        return cast("PageSelection | None", self._raw.get("pages"))


class FindingCollector:
    """Accumulates findings and metrics for one analysis operation run."""

    __slots__ = ("_metrics", "findings")

    def __init__(self) -> None:
        self.findings: list[AnalysisFinding] = []
        self._metrics: dict[str, object] = {}

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        page: int | None = None,
        bbox: Rect | None = None,
        evidence: tuple[EvidenceRecord, ...] = (),
        confidence: float | None = None,
        remediation: str | None = None,
    ) -> None:
        self.findings.append(
            AnalysisFinding(
                code=code,
                severity=severity,
                message=message,
                evidence=evidence,
                page_number=page,
                bbox=bbox,
                confidence=confidence,
                remediation=remediation,
            )
        )

    def count(self, key: str, increment: int = 1) -> None:
        """Increment an integer metric counter, creating it at zero."""
        current = self._metrics.get(key, 0)
        self._metrics[key] = (current if isinstance(current, int) else 0) + increment

    def set_metric(self, key: str, value: object) -> None:
        self._metrics[key] = value

    def metrics(self) -> dict[str, object]:
        return dict(self._metrics)

    def metrics_with_finding_count(self) -> dict[str, object]:
        return {**self._metrics, "finding_count": len(self.findings)}


class AnalysisOperation(ABC):
    """Template-method base class for local analysis operations.

    Subclasses implement :meth:`_analyze`; the base class owns context
    defaulting, option validation, and report assembly.  Operations that
    historically published a ``finding_count`` metric set
    ``emit_finding_count = True``.
    """

    operation_id: ClassVar[str]
    version: ClassVar[str] = "1.0"
    emit_finding_count: ClassVar[bool] = False

    def run(
        self,
        document: PdfDocument,
        context: ExecutionContext | None = None,
        options: Mapping[str, object] | None = None,
    ) -> AnalysisReport:
        ctx = context if context is not None else cast(ExecutionContext, LocalExecutionContext())
        out = FindingCollector()
        self._analyze(document, ctx, OperationOptions(options), out)
        metrics = out.metrics_with_finding_count() if self.emit_finding_count else out.metrics()
        return AnalysisReport(self.operation_id, self.version, tuple(out.findings), metrics)

    @abstractmethod
    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None: ...

    @staticmethod
    def _pages(
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
    ) -> Iterator[PdfPage]:
        """Iterate the selected pages, honouring cancellation between pages."""
        for page in document.pages(options.pages):
            context.cancellation.raise_if_cancelled()
            yield page


__all__ = ("AnalysisOperation", "FindingCollector", "OperationOptions")
