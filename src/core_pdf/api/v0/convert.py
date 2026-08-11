"""The single conversion point from engine IR records to api/v0 models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from .models import (
    ChunkRecord,
    CoordinateOrigin,
    CoordinateSpace,
    GeometryIssue,
    GeometrySummary,
    Rect,
    Severity,
    SourceRef,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout import LayoutGeometryIssue, LayoutGeometrySummary
    from core_pdf.impl.engine.structured import ChunkRecord as EngineChunkRecord


def page_space(page: Any) -> CoordinateSpace:
    return CoordinateSpace(
        name="pdf-page",
        origin=CoordinateOrigin.BOTTOM_LEFT,
        width=float(page.width),
        height=float(page.height),
    )


def to_rect(value: object, space: CoordinateSpace) -> Rect | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    raw_value = cast(Any, value)
    x0, y0, x1, y1 = (float(item) for item in raw_value)
    return Rect(x0, y0, x1, y1, space)


def item_bbox(value: object, space: CoordinateSpace) -> Rect | None:
    if isinstance(value, (tuple, list)) and len(value) == 4:
        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in value):
            raw_value = cast(Any, value)
            points = [(float(item[0]), float(item[1])) for item in raw_value]
            return Rect(
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
                space,
            )
        try:
            return to_rect(value, space)
        except (TypeError, ValueError):
            return None
    candidate = value
    if all(hasattr(candidate, name) for name in ("x0", "y0", "x1", "y1")):
        candidate = cast(Any, candidate)
        return Rect(
            float(candidate.x0),
            float(candidate.y0),
            float(candidate.x1),
            float(candidate.y1),
            space,
        )
    return None


def source_ref(page: Any, sequence: int | None = None, stage: str | None = None) -> SourceRef:
    page_number = getattr(page, "page_number", page.structured_view.page_number)
    page_label = getattr(page, "label", page.structured_view.page_label)
    return SourceRef(
        page_index=page_number - 1,
        page_number=page_number,
        page_label=page_label,
        sequence=sequence,
        stage=stage,
    )


def to_geometry_issue(issue: LayoutGeometryIssue, space: CoordinateSpace) -> GeometryIssue:
    return GeometryIssue(
        code=issue.code,
        severity=Severity(issue.severity),
        subject=issue.subject,
        bbox=to_rect(issue.bbox, space),
        message=issue.message,
        details=dict(issue.details),
        repairable=issue.repairable,
    )


def to_geometry_summary(summary: LayoutGeometrySummary) -> GeometrySummary:
    return GeometrySummary(
        issue_count=summary.issue_count,
        error_count=summary.error_count,
        warning_count=summary.warning_count,
        repairable_count=summary.repairable_count,
        text_run_count=summary.text_run_count,
        line_count=summary.line_count,
        issue_codes=summary.issue_codes,
        suspicion_score=summary.suspicion_score,
    )


def to_chunk_record(chunk: EngineChunkRecord, spaces: Mapping[int, CoordinateSpace]) -> ChunkRecord:
    element_bboxes = tuple(
        Rect(
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
            spaces[page_number],
        )
        for page_number, bbox in chunk.element_geometry
        if page_number in spaces
    )
    return ChunkRecord(
        text=chunk.text,
        page_numbers=chunk.page_numbers,
        element_ids=chunk.element_ids,
        element_types=chunk.element_types,
        section_path=chunk.section_path,
        metadata={
            "element_types": chunk.element_types,
            "element_geometry": chunk.element_geometry,
        },
        sources=tuple(
            SourceRef(page_number=page_number, stage="retrieval-chunk")
            for page_number in chunk.page_numbers
        ),
        element_bboxes=element_bboxes,
    )


__all__ = (
    "item_bbox",
    "page_space",
    "source_ref",
    "to_chunk_record",
    "to_geometry_issue",
    "to_geometry_summary",
    "to_rect",
)
