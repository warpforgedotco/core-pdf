"""Shared geometry, text-layout, and layout-quality components."""

from core_pdf.impl.engine.layout.geometry import BBox, RectBox
from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometryIssue,
    LayoutGeometryIssueRecord,
    LayoutGeometrySummary,
    LayoutGeometrySummaryRecord,
    layout_geometry_issue_record,
    layout_geometry_should_trigger_ocr,
    layout_geometry_summary_from_record,
    layout_geometry_summary_record,
    page_layout_geometry_issue_records,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issue_records,
    text_run_geometry_issues,
    text_run_has_repairable_glyph_geometry_issue,
)
from core_pdf.impl.engine.layout.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl.engine.layout.models import LayoutBox, LayoutLine, LayoutWord, TableGrid, TextRun
from core_pdf.impl.engine.layout.spatial import (
    SpatialHit,
    SpatialIndex,
    bbox_area,
    bbox_intersection_area,
    bbox_overlap_ratio,
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
    "SpatialHit",
    "SpatialIndex",
    "TableGrid",
    "TextRun",
    "bbox_area",
    "bbox_intersection_area",
    "bbox_overlap_ratio",
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
