"""Shared geometry, text-layout, and layout-quality components."""

from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.layout.geometry_quality import (
    LayoutGeometryIssue,
    LayoutGeometrySummary,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issues,
)
from core_pdf.impl.engine.layout.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl.engine.layout.models import LayoutLine, TextRun
from core_pdf.impl.engine.layout.spatial import (
    SpatialHit,
    SpatialIndex,
    bbox_area,
    bbox_intersection_area,
)

__all__ = (
    "GlyphCluster",
    "GlyphObservation",
    "LayoutGeometryIssue",
    "LayoutGeometrySummary",
    "LayoutLine",
    "RectBox",
    "SpatialHit",
    "SpatialIndex",
    "TextRun",
    "bbox_area",
    "bbox_intersection_area",
    "page_layout_geometry_issues",
    "page_layout_geometry_summary",
    "text_run_geometry_issues",
)
