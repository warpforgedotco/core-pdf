"""Composite compliance and quality preflight operations."""

from __future__ import annotations

from ..models import CompliancePreflightSummary, Severity
from ..protocols import ExecutionContext, PdfDocumentProtocol
from .base import AnalysisOperation, FindingCollector, OperationOptions
from .validation import (
    AccessibilityValidationOperation,
    AnnotationValidationOperation,
    AttachmentValidationOperation,
    DocumentIntegrityOperation,
    FontValidationOperation,
    FormValidationOperation,
    GeometryValidationOperation,
    ImageValidationOperation,
    LinkValidationOperation,
)


class CompliancePreflightOperation(AnalysisOperation):
    """Run conservative local PDF/A and PDF/UA-style preflight checks.

    Document-title and figure alternate-text coverage are owned by
    :class:`AccessibilityValidationOperation` and are deliberately not
    duplicated here.
    """

    operation_id = "analysis.compliance-preflight"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        profile = options.get_str("profile", "pdf/a-2u").casefold()
        inventory = document.inventory()

        if "pdf/a" in profile and inventory.encrypted:
            out.add(
                "pdfa.encryption",
                Severity.ERROR,
                "Encryption is present in a PDF/A-style preflight profile.",
                remediation="Remove encryption or use a profile that explicitly permits it.",
            )
        if "pdf/a" in profile and inventory.has_javascript:
            out.add(
                "pdfa.javascript",
                Severity.ERROR,
                "JavaScript is present in a PDF/A-style preflight profile.",
                remediation="Remove executable actions before archival.",
            )
        if "pdf/a" in profile:
            for diagnostic in document.resource_diagnostics():
                out.add(
                    f"pdfa.resource.{diagnostic.code}",
                    Severity.ERROR if diagnostic.severity == Severity.ERROR else Severity.WARNING,
                    diagnostic.message,
                    remediation="Repair or replace the affected resource before archival.",
                )
        if "pdf/ua" in profile or "pdfua" in profile:
            structure = tuple(document.structure_elements())
            if not structure:
                out.add(
                    "pdfua.structure-missing",
                    Severity.ERROR,
                    "No tagged logical structure was found for a PDF/UA-style profile.",
                    remediation="Add tagged structure and validate its reading order and roles.",
                )
            headings = sum(1 for element in structure if element.role.startswith("H"))
            outline_count = sum(1 for _ in document.outlines)
            if headings and outline_count < headings:
                out.add(
                    "pdfua.navigation-heading-mismatch",
                    Severity.WARNING,
                    "The tagged heading structure has more headings than the document outline.",
                    remediation="Add outline entries for headings so users can navigate "
                    "the document.",
                )
        findings = out.findings
        out.set_metric("profile", profile)
        out.set_metric(
            "summary",
            CompliancePreflightSummary(
                profile=profile,
                passed=not any(finding.severity == Severity.ERROR for finding in findings),
                error_count=sum(finding.severity == Severity.ERROR for finding in findings),
                warning_count=sum(finding.severity == Severity.WARNING for finding in findings),
                finding_codes=tuple(finding.code for finding in findings),
                has_resource_errors=any(
                    finding.code.startswith("pdfa.resource.") and finding.severity == Severity.ERROR
                    for finding in findings
                ),
                has_font_errors=any("font" in finding.code for finding in findings),
                has_color_warnings=any("color" in finding.code for finding in findings),
                has_transparency_warnings=any(
                    "transparency" in finding.code for finding in findings
                ),
            ),
        )
        out.set_metric("encrypted", inventory.encrypted)
        out.set_metric("tagged_structure_count", len(tuple(document.structure_elements())))


class QualityPreflightOperation(AnalysisOperation):
    """Run the standard local document-quality validators as one report."""

    operation_id = "analysis.quality-preflight"

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        operations: tuple[AnalysisOperation, ...] = (
            CompliancePreflightOperation(),
            AccessibilityValidationOperation(),
            FormValidationOperation(),
            LinkValidationOperation(),
            AttachmentValidationOperation(),
            FontValidationOperation(),
            ImageValidationOperation(),
            GeometryValidationOperation(),
            AnnotationValidationOperation(),
            DocumentIntegrityOperation(),
        )
        reports = tuple(operation.run(document, context, options.raw) for operation in operations)
        context.cancellation.raise_if_cancelled()
        for report in reports:
            out.findings.extend(report.findings)
        out.set_metric("operation_count", len(reports))
        out.set_metric("finding_count", sum(len(report.findings) for report in reports))
        out.set_metric(
            "error_count",
            sum(
                sum(finding.severity == Severity.ERROR for finding in report.findings)
                for report in reports
            ),
        )
        out.set_metric(
            "warning_count",
            sum(
                sum(finding.severity == Severity.WARNING for finding in report.findings)
                for report in reports
            ),
        )
        out.set_metric("reports", {report.analyzer_id: report.metrics for report in reports})


__all__ = ("CompliancePreflightOperation", "QualityPreflightOperation")
