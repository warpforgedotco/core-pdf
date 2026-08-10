"""Native local operations built on the v0 protocols."""

from .base import AnalysisOperation, FindingCollector, OperationOptions
from .content import (
    CitationAnalysisOperation,
    FigureCaptionAnalysisOperation,
    IdentifierAnalysisOperation,
    LayoutAnalysisOperation,
    ReferenceEntryAnalysisOperation,
    SectionHierarchyAnalysisOperation,
    StructureAnalysisOperation,
)
from .forensics import (
    BadRedactionOperation,
    ForensicAnalysisOperation,
    LayerConsistencyOperation,
)
from .preflight import CompliancePreflightOperation, QualityPreflightOperation
from .transform import normalize_metadata, plan_accessibility_remediation
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

__all__ = (
    "AccessibilityValidationOperation",
    "AnalysisOperation",
    "AnnotationValidationOperation",
    "AttachmentValidationOperation",
    "BadRedactionOperation",
    "CitationAnalysisOperation",
    "CompliancePreflightOperation",
    "DocumentIntegrityOperation",
    "FigureCaptionAnalysisOperation",
    "FindingCollector",
    "FontValidationOperation",
    "ForensicAnalysisOperation",
    "FormValidationOperation",
    "GeometryValidationOperation",
    "IdentifierAnalysisOperation",
    "ImageValidationOperation",
    "LayerConsistencyOperation",
    "LayoutAnalysisOperation",
    "LinkValidationOperation",
    "OperationOptions",
    "QualityPreflightOperation",
    "ReferenceEntryAnalysisOperation",
    "SectionHierarchyAnalysisOperation",
    "StructureAnalysisOperation",
    "normalize_metadata",
    "plan_accessibility_remediation",
)
