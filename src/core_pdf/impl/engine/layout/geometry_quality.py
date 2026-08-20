# SPDX-License-Identifier: AGPL-3.0-only
"""Detect geometry problems in text runs and lines, and summarize them per page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.engine.layout.geometry import (
    bbox_union,
    finite_rect,
    overlap_ratio_of,
)
from core_pdf.impl.engine.layout.glyphs import (
    glyph_text_has_unsupported_codepoint,
)
from core_pdf.impl.engine.layout.models import internal_track_text_run

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.models import LayoutLine, TextRun

BBoxTuple = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class LayoutGeometryIssue:
    code: str
    severity: str
    subject: str
    bbox: BBoxTuple | None = None
    message: str = ""
    details: tuple[tuple[str, object], ...] = ()
    repairable: bool = False


@dataclass(frozen=True, slots=True)
class LayoutGeometrySummary:
    issue_count: int
    error_count: int
    warning_count: int
    repairable_count: int
    text_run_count: int
    line_count: int
    issue_codes: tuple[tuple[str, int], ...]
    suspicion_score: float

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0

    @property
    def has_repairable_issues(self) -> bool:
        return self.repairable_count > 0


def text_run_geometry_issues(run: TextRun) -> tuple[LayoutGeometryIssue, ...]:
    key = (
        run.internal_revision,
        tuple(run.coords),
        run.advance_bbox,
        run.ink_bbox,
        id(run.glyph_clusters),
    )
    cache = run.internal_geometry_issues_cache
    if cache is not None and cache[0] == key:
        return cast(tuple[LayoutGeometryIssue, ...], cache[1])
    issues = internal_compute_text_run_geometry_issues(run)
    internal_track_text_run(run)
    object.__setattr__(run, "internal_geometry_issues_cache", (key, issues))
    return issues


def internal_compute_text_run_geometry_issues(run: TextRun) -> tuple[LayoutGeometryIssue, ...]:
    issues: list[LayoutGeometryIssue] = []
    run_bbox = (run.x0, run.y0, run.x1, run.y1)
    visible_text = run.visible and bool(run.text.strip())

    if visible_text and not bbox_is_positive(run_bbox):
        issues.append(
            LayoutGeometryIssue(
                code="run_nonpositive_bbox",
                severity="error",
                subject="text_run",
                bbox=run_bbox,
                message="Visible text run has no positive page-space area.",
            )
        )

    if visible_text and glyph_text_has_unsupported_codepoint(run.text):
        issues.append(
            LayoutGeometryIssue(
                code="unsupported_text_run",
                severity="warning",
                subject="text_run",
                bbox=run_bbox,
                message="Text run contains unsupported/private-use glyph text.",
                repairable=True,
            )
        )

    confidence = run.confidence
    if visible_text and confidence is not None and confidence <= 0.35:
        issues.append(
            LayoutGeometryIssue(
                code="low_confidence_text_run",
                severity="warning",
                subject="text_run",
                bbox=run_bbox,
                message="Text run came from low-confidence glyph decoding.",
                details=(("confidence", confidence),),
                repairable=True,
            )
        )

    advance_bbox = numeric_bbox(run.advance_bbox)
    ink_bbox = numeric_bbox(run.ink_bbox)
    if visible_text and (advance_bbox is None or not bbox_is_positive(advance_bbox)):
        issues.append(
            LayoutGeometryIssue(
                code="run_nonpositive_advance_bbox",
                severity="error",
                subject="text_run.advance_bbox",
                bbox=advance_bbox,
                message="Visible text run has no positive advance bbox.",
            )
        )
    if visible_text and (ink_bbox is None or not bbox_is_positive(ink_bbox)):
        issues.append(
            LayoutGeometryIssue(
                code="run_nonpositive_ink_bbox",
                severity="warning",
                subject="text_run.ink_bbox",
                bbox=ink_bbox,
                message="Visible text run has no positive ink bbox.",
            )
        )

    clusters = tuple(run.glyph_clusters or ())
    if not clusters:
        return tuple(issues)

    cluster_text = "".join(cluster.text for cluster in clusters)
    if cluster_text != run.text:
        issues.append(
            LayoutGeometryIssue(
                code="glyph_cluster_text_mismatch",
                severity="error",
                subject="text_run.glyph_clusters",
                bbox=run_bbox,
                message="Glyph cluster text does not reconstruct the text run.",
                details=(
                    ("run_text", run.text),
                    ("cluster_text", cluster_text),
                    ("cluster_count", len(clusters)),
                ),
                repairable=True,
            )
        )

    cluster_bboxes: list[BBoxTuple] = []
    for cluster_index, cluster in enumerate(clusters):
        cluster_bbox = numeric_bbox(cluster.advance_bbox)
        if cluster_bbox is None or not bbox_is_positive(cluster_bbox):
            issues.append(
                LayoutGeometryIssue(
                    code="glyph_cluster_nonpositive_bbox",
                    severity="error",
                    subject=f"glyph_cluster[{cluster_index}]",
                    bbox=cluster_bbox,
                    message="Glyph cluster has no positive advance bbox.",
                    details=(("cluster_id", cluster.cluster_id),),
                    repairable=True,
                )
            )
        else:
            cluster_bboxes.append(cluster_bbox)

        cluster_text_value = cluster.text
        if not cluster_text_value.strip():
            continue
        cluster_confidence = cluster.confidence
        unsupported = glyph_text_has_unsupported_codepoint(cluster_text_value)
        low_confidence = cluster_confidence is not None and cluster_confidence <= 0.35
        suspicious_low_confidence = (
            cluster_confidence is not None
            and cluster_confidence <= 0.62
            and not cluster_text_value.isalnum()
        )
        if unsupported or low_confidence or suspicious_low_confidence:
            issues.append(
                LayoutGeometryIssue(
                    code=(
                        "unsupported_glyph_cluster_text"
                        if unsupported
                        else "low_confidence_repairable_glyph"
                    ),
                    severity="warning",
                    subject=f"glyph_cluster[{cluster_index}]",
                    bbox=cluster_bbox,
                    message="Glyph cluster is a repairable low-confidence text observation.",
                    details=(
                        ("cluster_id", cluster.cluster_id),
                        ("text", cluster_text_value),
                        ("confidence", cluster_confidence),
                    ),
                    repairable=True,
                )
            )

    cluster_union = bbox_union(tuple(cluster_bboxes))
    if cluster_union is None or advance_bbox is None or not bbox_is_positive(advance_bbox):
        return tuple(issues)

    # Glyph clusters are derived from glyph advance geometry, which is also the
    # canonical page-space extent used by layout.  The legacy run coordinates
    # can describe the text origin and nominal font box instead, notably for
    # vertical writing where those boxes are intentionally offset.
    cluster_inside_advance = overlap_ratio_of(cluster_union, advance_bbox)
    if cluster_inside_advance < 0.80:
        issues.append(
            LayoutGeometryIssue(
                code="glyph_clusters_outside_advance_bbox",
                severity="error",
                subject="text_run.glyph_clusters",
                bbox=cluster_union,
                message="Glyph cluster geometry does not overlap the owning advance bbox.",
                details=(
                    ("overlap_ratio", round(cluster_inside_advance, 4)),
                    ("reference_region", "advance_bbox"),
                ),
            )
        )

    axis = "y" if run.rotation_angle in (90, 270) else "x"
    run_span = bbox_height(run_bbox) if axis == "y" else bbox_width(run_bbox)
    cluster_span = bbox_height(cluster_union) if axis == "y" else bbox_width(cluster_union)
    min_reference_span = max(20.0, run.font_size * 2.0)
    if cluster_span > 0.0 and run_span > min_reference_span and run_span > cluster_span * 2.5:
        issues.append(
            LayoutGeometryIssue(
                code="run_bbox_oversized_for_glyph_clusters",
                severity="warning",
                subject="text_run",
                bbox=run_bbox,
                message="Run bbox is much larger than its glyph cluster geometry.",
                details=(
                    ("axis", axis),
                    ("run_span", round(run_span, 4)),
                    ("cluster_span", round(cluster_span, 4)),
                ),
            )
        )

    return tuple(issues)


def layout_line_geometry_issues(line: LayoutLine) -> tuple[LayoutGeometryIssue, ...]:
    issues: list[LayoutGeometryIssue] = []
    line_bbox = (line.x0, line.y0, line.x1, line.y1)
    has_visible_text = any(run.visible and run.text.strip() for run in line.runs)
    if has_visible_text and not bbox_is_positive(line_bbox):
        issues.append(
            LayoutGeometryIssue(
                code="line_nonpositive_bbox",
                severity="error",
                subject="layout_line",
                bbox=line_bbox,
                message="Line with visible text has no positive page-space area.",
            )
        )

    if not bbox_is_positive(line_bbox):
        return tuple(issues)

    for run_index, run in enumerate(line.runs):
        if not run.visible or not run.text.strip():
            continue
        run_bbox = (run.x0, run.y0, run.x1, run.y1)
        if not bbox_is_positive(run_bbox):
            continue
        overlap = overlap_ratio_of(run_bbox, line_bbox)
        if overlap < 0.80:
            issues.append(
                LayoutGeometryIssue(
                    code="run_outside_line_bbox",
                    severity="error",
                    subject=f"layout_line.runs[{run_index}]",
                    bbox=run_bbox,
                    message="Run geometry falls outside the owning line bbox.",
                    details=(("overlap_ratio", round(overlap, 4)),),
                )
            )

    words = line.cached_text_and_words()[1]
    word_bboxes = tuple(word.bbox for word in words if bbox_is_positive(word.bbox))
    if not word_bboxes:
        return tuple(issues)
    word_union = bbox_union(word_bboxes)
    if word_union is None:
        return tuple(issues)
    axis = "y" if line.rotation_angle in (90, 270) else "x"
    line_span = bbox_height(line_bbox) if axis == "y" else bbox_width(line_bbox)
    word_span = bbox_height(word_union) if axis == "y" else bbox_width(word_union)
    if word_span > 0.0 and line_span > max(20.0, word_span * 2.5):
        issues.append(
            LayoutGeometryIssue(
                code="line_bbox_oversized_for_words",
                severity="warning",
                subject="layout_line",
                bbox=line_bbox,
                message="Line bbox is much larger than reconstructed word geometry.",
                details=(
                    ("axis", axis),
                    ("line_span", round(line_span, 4)),
                    ("word_span", round(word_span, 4)),
                ),
            )
        )

    return tuple(issues)


def page_layout_geometry_issues(
    lines: list[LayoutLine],
) -> tuple[LayoutGeometryIssue, ...]:
    issues: list[LayoutGeometryIssue] = []
    for line_index, line in enumerate(lines):
        for issue in layout_line_geometry_issues(line):
            issues.append(with_issue_detail(issue, "line_index", line_index))
        for run_index, run in enumerate(line.runs):
            for issue in text_run_geometry_issues(run):
                issues.append(
                    with_issue_detail(
                        with_issue_detail(issue, "run_index", run_index),
                        "line_index",
                        line_index,
                    )
                )
    return tuple(issues)


def page_layout_geometry_summary(lines: list[LayoutLine]) -> LayoutGeometrySummary:
    issues = page_layout_geometry_issues(lines)
    error_count = 0
    warning_count = 0
    repairable_count = 0
    text_run_count = 0
    counts: dict[str, int] = {}
    for issue in issues:
        if issue.severity == "error":
            error_count += 1
        elif issue.severity == "warning":
            warning_count += 1
        if issue.repairable:
            repairable_count += 1
        counts[issue.code] = counts.get(issue.code, 0) + 1
    for line in lines:
        text_run_count += len(line.runs)
    return LayoutGeometrySummary(
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        repairable_count=repairable_count,
        text_run_count=text_run_count,
        line_count=len(lines),
        issue_codes=tuple(sorted(counts.items())),
        suspicion_score=layout_geometry_suspicion_score(issues),
    )


def layout_geometry_suspicion_score(
    issues: tuple[LayoutGeometryIssue, ...],
) -> float:
    score = 0.0
    for issue in issues:
        if issue.severity == "error":
            score += 2.5
        elif issue.severity == "warning":
            score += 1.0
        else:
            score += 0.5
        if issue.repairable:
            score += 1.5
        if issue.code in {
            "unsupported_text_run",
            "glyph_cluster_text_mismatch",
            "unsupported_glyph_cluster_text",
            "low_confidence_text_run",
        }:
            score += 1.5
        elif issue.code in {
            "glyph_clusters_outside_run_bbox",
            "run_bbox_oversized_for_glyph_clusters",
            "line_bbox_oversized_for_words",
        }:
            score += 0.75
    return round(score, 4)


def with_issue_detail(
    issue: LayoutGeometryIssue,
    key: str,
    value: Any,
) -> LayoutGeometryIssue:
    return LayoutGeometryIssue(
        code=issue.code,
        severity=issue.severity,
        subject=issue.subject,
        bbox=issue.bbox,
        message=issue.message,
        details=(*issue.details, (key, value)),
        repairable=issue.repairable,
    )


def numeric_bbox(value: Any) -> BBoxTuple | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return finite_rect(value, require_positive=False)


def bbox_is_positive(bbox: BBoxTuple | None) -> bool:
    return bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def bbox_width(bbox: BBoxTuple) -> float:
    return bbox[2] - bbox[0]


def bbox_height(bbox: BBoxTuple) -> float:
    return bbox[3] - bbox[1]


__all__ = (
    "LayoutGeometryIssue",
    "LayoutGeometrySummary",
    "layout_line_geometry_issues",
    "page_layout_geometry_issues",
    "page_layout_geometry_summary",
    "text_run_geometry_issues",
)
