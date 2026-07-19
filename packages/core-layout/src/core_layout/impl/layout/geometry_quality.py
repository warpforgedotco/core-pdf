# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, TypedDict, cast

from core_layout.impl.layout.glyphs import glyph_text_has_unsupported_codepoint

if TYPE_CHECKING:
    from core_layout.impl.layout.models import LayoutLine, TextRun

BBoxTuple = tuple[float, float, float, float]


class LayoutGeometryIssueRecord(TypedDict, total=False):
    code: str
    severity: str
    subject: str
    repairable: bool
    bbox: BBoxTuple
    message: str
    details: tuple[tuple[str, object], ...]


class LayoutGeometrySummaryRecord(TypedDict):
    issue_count: int
    error_count: int
    warning_count: int
    repairable_count: int
    text_run_count: int
    line_count: int
    issue_codes: tuple[tuple[str, int], ...]
    suspicion_score: float


def int_record_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"expected int-compatible value, got {type(value).__name__}")


def float_record_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


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
    key = (run._revision, tuple(run.coords))
    cache = run._geometry_issues_cache
    if cache is not None and cache[0] == key:
        return cast(tuple[LayoutGeometryIssue, ...], cache[1])
    issues = _compute_text_run_geometry_issues(run)
    object.__setattr__(run, "_cache_tracking", True)
    object.__setattr__(run, "_geometry_issues_cache", (key, issues))
    return issues


def _compute_text_run_geometry_issues(run: TextRun) -> tuple[LayoutGeometryIssue, ...]:
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

    cluster_union = union_bboxes(tuple(cluster_bboxes))
    if cluster_union is None or not bbox_is_positive(run_bbox):
        return tuple(issues)

    cluster_inside_run = bbox_overlap_ratio(cluster_union, run_bbox)
    if cluster_inside_run < 0.80:
        issues.append(
            LayoutGeometryIssue(
                code="glyph_clusters_outside_run_bbox",
                severity="error",
                subject="text_run.glyph_clusters",
                bbox=cluster_union,
                message="Glyph cluster geometry does not overlap the owning run bbox.",
                details=(("overlap_ratio", round(cluster_inside_run, 4)),),
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
        overlap = bbox_overlap_ratio(run_bbox, line_bbox)
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

    ignored_text, words = line.cached_text_and_words()
    word_bboxes = tuple(word.bbox for word in words if bbox_is_positive(word.bbox))
    if not word_bboxes:
        return tuple(issues)
    word_union = union_bboxes(word_bboxes)
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
    counts = Counter(issue.code for issue in issues)
    return LayoutGeometrySummary(
        issue_count=len(issues),
        error_count=sum(1 for issue in issues if issue.severity == "error"),
        warning_count=sum(1 for issue in issues if issue.severity == "warning"),
        repairable_count=sum(1 for issue in issues if issue.repairable),
        text_run_count=sum(len(line.runs) for line in lines),
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


def text_run_has_repairable_glyph_geometry_issue(run: TextRun) -> bool:
    return any(issue.repairable for issue in text_run_geometry_issues(run))


def layout_geometry_issue_record(
    issue: LayoutGeometryIssue,
) -> LayoutGeometryIssueRecord:
    record: LayoutGeometryIssueRecord = {
        "code": issue.code,
        "severity": issue.severity,
        "subject": issue.subject,
        "repairable": issue.repairable,
    }
    if issue.bbox is not None:
        record["bbox"] = issue.bbox
    if issue.message:
        record["message"] = issue.message
    if issue.details:
        record["details"] = issue.details
    return record


def text_run_geometry_issue_records(run: TextRun) -> list[LayoutGeometryIssueRecord]:
    return [layout_geometry_issue_record(issue) for issue in text_run_geometry_issues(run)]


def page_layout_geometry_issue_records(
    lines: list[LayoutLine],
) -> list[LayoutGeometryIssueRecord]:
    return [layout_geometry_issue_record(issue) for issue in page_layout_geometry_issues(lines)]


def layout_geometry_summary_record(
    summary: LayoutGeometrySummary,
) -> LayoutGeometrySummaryRecord:
    return {
        "issue_count": summary.issue_count,
        "error_count": summary.error_count,
        "warning_count": summary.warning_count,
        "repairable_count": summary.repairable_count,
        "text_run_count": summary.text_run_count,
        "line_count": summary.line_count,
        "issue_codes": summary.issue_codes,
        "suspicion_score": summary.suspicion_score,
    }


def layout_geometry_summary_from_record(
    record: Mapping[str, object],
) -> LayoutGeometrySummary | None:
    try:
        issue_codes_value = record.get("issue_codes", ())
        if not isinstance(issue_codes_value, tuple | list):
            return None
        issue_codes_list: list[tuple[str, int]] = []
        for item in issue_codes_value:
            if not isinstance(item, tuple | list) or len(item) != 2:
                return None
            code, count = item
            issue_codes_list.append((str(code), int_record_value(count)))
        issue_codes = tuple(issue_codes_list)
        return LayoutGeometrySummary(
            issue_count=int_record_value(record.get("issue_count", 0)),
            error_count=int_record_value(record.get("error_count", 0)),
            warning_count=int_record_value(record.get("warning_count", 0)),
            repairable_count=int_record_value(record.get("repairable_count", 0)),
            text_run_count=int_record_value(record.get("text_run_count", 0)),
            line_count=int_record_value(record.get("line_count", 0)),
            issue_codes=issue_codes,
            suspicion_score=float_record_value(record.get("suspicion_score", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def layout_geometry_should_trigger_ocr(
    summary: LayoutGeometrySummary | None,
    *,
    text_tokens: int,
) -> bool:
    if summary is None or not summary.has_issues:
        return False
    if text_tokens <= 20:
        return summary.error_count > 0 or summary.has_repairable_issues
    if (
        text_tokens >= 250
        and summary.text_run_count >= 100
        and summary.issue_count / summary.text_run_count <= 0.08
    ):
        # A few suspect formula or symbol glyphs do not make an otherwise
        # substantial native layer unreliable. Full-page OCR tends to turn
        # those isolated constructs into high-confidence mirrored gibberish.
        return False
    if summary.has_repairable_issues and summary.suspicion_score >= 5.0:
        return True
    if summary.error_count >= 2 and summary.suspicion_score >= 7.0:
        return True
    return summary.suspicion_score >= 12.0 and text_tokens <= 1000


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
    try:
        bbox = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(coord) for coord in bbox):
        return None
    return bbox


def bbox_is_positive(bbox: BBoxTuple | None) -> bool:
    return bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def bbox_width(bbox: BBoxTuple) -> float:
    return bbox[2] - bbox[0]


def bbox_height(bbox: BBoxTuple) -> float:
    return bbox[3] - bbox[1]


def bbox_area(bbox: BBoxTuple) -> float:
    return max(0.0, bbox_width(bbox)) * max(0.0, bbox_height(bbox))


def bbox_intersection_area(a: BBoxTuple, b: BBoxTuple) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def bbox_overlap_ratio(subject: BBoxTuple, container: BBoxTuple) -> float:
    subject_area = bbox_area(subject)
    if subject_area <= 0.0:
        return 0.0
    return bbox_intersection_area(subject, container) / subject_area


def union_bboxes(boxes: tuple[BBoxTuple, ...]) -> BBoxTuple | None:
    if not boxes:
        return None
    x0, y0, x1, y1 = boxes[0]
    for bx0, by0, bx1, by1 in boxes[1:]:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


__all__ = (
    "LayoutGeometryIssue",
    "LayoutGeometryIssueRecord",
    "LayoutGeometrySummary",
    "LayoutGeometrySummaryRecord",
    "layout_geometry_issue_record",
    "layout_geometry_summary_record",
    "layout_geometry_summary_from_record",
    "layout_geometry_should_trigger_ocr",
    "layout_line_geometry_issues",
    "page_layout_geometry_issue_records",
    "page_layout_geometry_issues",
    "page_layout_geometry_summary",
    "text_run_geometry_issue_records",
    "text_run_geometry_issues",
    "text_run_has_repairable_glyph_geometry_issue",
)
