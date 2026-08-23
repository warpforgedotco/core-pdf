# SPDX-License-Identifier: AGPL-3.0-only
"""Capture, recognize, and emit PDF content.

The package exports the pipeline entry points and the shared stage models;
stage internals live in (and are imported from) the owning submodule.
"""

from __future__ import annotations

from core_pdf.impl.engine.parse.model import (
    CapturedPage,
    FusionPolicy,
    GlyphEvidence,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    PageEvidence,
    PagePlanReason,
    PageRoute,
    ParsedBlock,
    ParsedLine,
    ParsedPage,
    ParseReport,
    ReadingOrderEvidence,
    RecognitionReport,
    RecognitionResult,
    TextQualityStats,
    WorkPlan,
)
from core_pdf.impl.engine.parse.ocr import prewarm_runtime
from core_pdf.impl.engine.parse.pipeline import (
    extract_page,
    page_extraction,
    parse_document,
    parse_page,
)

__all__ = (
    "CapturedPage",
    "FusionPolicy",
    "GlyphEvidence",
    "ObservationBatch",
    "ObservationSource",
    "OcrPass",
    "OcrPassScope",
    "PageEvidence",
    "PagePlanReason",
    "PageRoute",
    "ParseReport",
    "ParsedBlock",
    "ParsedLine",
    "ParsedPage",
    "ReadingOrderEvidence",
    "RecognitionReport",
    "RecognitionResult",
    "TextQualityStats",
    "WorkPlan",
    "extract_page",
    "page_extraction",
    "parse_document",
    "parse_page",
    "prewarm_runtime",
)
