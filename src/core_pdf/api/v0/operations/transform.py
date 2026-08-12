"""Local transform helpers producing editor-ready values from evidence."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf.impl.text import collapse_ws

from ...models import AnalysisReport, RemediationAction, SourceRef

_METADATA_KEYS = {
    "title": "Title",
    "author": "Author",
    "subject": "Subject",
    "keywords": "Keywords",
    "creator": "Creator",
    "producer": "Producer",
    "creationdate": "CreationDate",
    "moddate": "ModDate",
    "language": "Language",
}

_REMEDIATION_STAGE = "accessibility-remediation"


def normalize_metadata(values: Mapping[str, object]) -> dict[str, object]:
    """Return canonical common metadata while preserving unknown fields."""
    normalized: dict[str, object] = {}
    for key, value in values.items():
        canonical_key = _METADATA_KEYS.get(str(key).lstrip("/").casefold(), str(key))
        if isinstance(value, str):
            normalized[canonical_key] = collapse_ws(value)
        else:
            normalized[canonical_key] = value
    return normalized


def plan_accessibility_remediation(report: AnalysisReport) -> tuple[RemediationAction, ...]:
    """Derive explicit remediation actions from an accessibility report.

    Accepts an :class:`AccessibilityValidationOperation` report and maps its
    findings to concrete review or transformation steps without rescanning
    the source document.
    """
    actions: list[RemediationAction] = []
    for finding in report.findings:
        if finding.code == "ua.document-title":
            actions.append(
                RemediationAction(
                    code="metadata.title",
                    priority="high",
                    message="Set a descriptive document title.",
                    source=SourceRef(stage=_REMEDIATION_STAGE),
                )
            )
        elif finding.code == "ua.tagged-structure":
            actions.append(
                RemediationAction(
                    code="structure.tagged-content",
                    priority="high",
                    message="Add tagged structure and validate reading order.",
                    source=SourceRef(stage=_REMEDIATION_STAGE),
                )
            )
        elif finding.code == "ua.figure-alternate-text":
            actions.append(
                RemediationAction(
                    code="figure.alternate-text",
                    priority="high",
                    message="Provide alternate text or mark this figure decorative.",
                    page_number=finding.page_number,
                    source=SourceRef(
                        page_number=finding.page_number,
                        stage=_REMEDIATION_STAGE,
                    ),
                )
            )
    return tuple(actions)


__all__ = ("normalize_metadata", "plan_accessibility_remediation")
