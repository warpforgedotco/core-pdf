# SPDX-License-Identifier: AGPL-3.0-only
"""Capture, recognize, and emit PDF content.

The package exports the pipeline entry points and the shared stage models;
stage internals live in (and are imported from) the owning submodule.
"""

from __future__ import annotations

from core_pdf.impl.engine.parse.capture import capture_page, preflight_page
from core_pdf.impl.engine.parse.emit import assemble_page
from core_pdf.impl.engine.parse.fusion import fuse_observations, maximum_candidate_coverage
from core_pdf.impl.engine.parse.layout import layout_blocks, layout_blocks_with_evidence
from core_pdf.impl.engine.parse.model import (
    CapturedPage,
    GlyphEvidence,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    PageEvidence,
    PagePreflightClass,
    PageRoute,
    ParsedBlock,
    ParsedLine,
    ParsedPage,
    ReadingOrderEvidence,
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
from core_pdf.impl.engine.parse.route import plan_page
from core_pdf.impl.engine.parse.tables import extract_tables

__all__ = (
    "CapturedPage",
    "GlyphEvidence",
    "ObservationBatch",
    "ObservationSource",
    "OcrPass",
    "OcrPassScope",
    "PageEvidence",
    "PagePreflightClass",
    "PageRoute",
    "ParsedBlock",
    "ParsedLine",
    "ParsedPage",
    "ReadingOrderEvidence",
    "TextQualityStats",
    "WorkPlan",
    "assemble_page",
    "capture_page",
    "extract_page",
    "extract_tables",
    "fuse_observations",
    "layout_blocks",
    "layout_blocks_with_evidence",
    "maximum_candidate_coverage",
    "page_extraction",
    "parse_document",
    "parse_page",
    "plan_page",
    "preflight_page",
    "prewarm_runtime",
)
