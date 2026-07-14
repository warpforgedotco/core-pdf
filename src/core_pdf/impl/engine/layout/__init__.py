# SPDX-License-Identifier: AGPL-3.0-only
"""Shared page-space geometry and layout models."""

from core_pdf.impl.engine.layout.geometry import BBox, RectBox
from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometryIssue,
    LayoutGeometryIssueRecord,
    LayoutGeometrySummary,
    LayoutGeometrySummaryRecord,
    layout_geometry_issue_record,
    layout_geometry_should_trigger_ocr,
    layout_geometry_summary_record,
    layout_geometry_summary_from_record,
    page_layout_geometry_issue_records,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issue_records,
    text_run_geometry_issues,
    text_run_has_repairable_glyph_geometry_issue,
)
from core_pdf.impl.engine.layout.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl.engine.layout.models import (
    LayoutBox,
    LayoutLine,
    LayoutWord,
    TableGrid,
    TextRun,
)

__all__ = (
    "BBox",
    "GlyphCluster",
    "GlyphObservation",
    "LayoutBox",
    "LayoutGeometryIssue",
    "LayoutGeometryIssueRecord",
    "LayoutGeometrySummary",
    "LayoutGeometrySummaryRecord",
    "LayoutLine",
    "LayoutWord",
    "RectBox",
    "TableGrid",
    "TextRun",
    "layout_geometry_issue_record",
    "layout_geometry_should_trigger_ocr",
    "layout_geometry_summary_record",
    "layout_geometry_summary_from_record",
    "page_layout_geometry_issue_records",
    "page_layout_geometry_issues",
    "page_layout_geometry_summary",
    "text_run_geometry_issue_records",
    "text_run_geometry_issues",
    "text_run_has_repairable_glyph_geometry_issue",
)
