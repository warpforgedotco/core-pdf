# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import replace
from itertools import islice
from typing import TYPE_CHECKING, Any, ClassVar, Self

from core_pdf.impl.model.geometry import bbox_union, finite_rect, overlap_ratio_of
from core_pdf.impl.model.glyphs import glyph_text_has_unsupported_codepoint
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.records import Record
from core_pdf.impl.types import Rectangle, TextWord

if TYPE_CHECKING:
    from core_pdf.impl.model.runs import (
        LayoutLineText,
        LayoutLineTextSegment,
    )

frozen_setattr = object.__setattr__


class LayoutLine:
    __slots__ = (
        "runs",
        "x0",
        "y0",
        "x1",
        "y1",
        "is_vertical",
        "rotation_angle",
        "height",
        "is_all_caps_text",
    )

    runs: list[TextRun]
    x0: float
    y0: float
    x1: float
    y1: float
    is_vertical: bool
    rotation_angle: int
    is_all_caps_text: bool

    def __init__(self, runs: list[TextRun] | None = None) -> None:
        self.runs = run_list = runs if runs is not None else []
        if not run_list:
            self.x0 = self.y0 = self.x1 = self.y1 = 0.0
            self.is_vertical = False
            self.rotation_angle = 0
            self.height = 0.0
            self.is_all_caps_text = True
            return

        first_run = run_list[0]
        x0 = first_run.x0
        y0 = first_run.y0
        x1 = first_run.x1
        y1 = first_run.y1
        is_all_caps_text = not first_run.has_text or first_run.text_is_upper

        for run in islice(run_list, 1, None):
            run_x0 = run.x0
            run_y0 = run.y0
            run_x1 = run.x1
            run_y1 = run.y1
            if run_x0 < x0:
                x0 = run_x0
            if run_y0 < y0:
                y0 = run_y0
            if run_x1 > x1:
                x1 = run_x1
            if run_y1 > y1:
                y1 = run_y1
            if is_all_caps_text and run.has_text and not run.text_is_upper:
                is_all_caps_text = False

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.is_vertical = first_run.is_vertical
        self.rotation_angle = first_run.rotation_angle
        self.height = y1 - y0
        self.is_all_caps_text = is_all_caps_text

    def reconstructed_text(self) -> LayoutLineText:
        from core_pdf.impl.layout.reconstruction import reconstruct_layout_line_text

        return reconstruct_layout_line_text(self.runs, is_all_caps_text=self.is_all_caps_text)

    def text_and_words(
        self, reconstructed: LayoutLineText | None = None
    ) -> tuple[str, tuple[TextWord, ...]]:
        if reconstructed is None:
            reconstructed = self.reconstructed_text()
        parts: list[str] = []
        words: list[TextWord] = []
        word = ""
        word_x0 = word_y0 = word_x1 = word_y1 = 0.0
        append_part = parts.append
        append_word = words.append

        def flush_word() -> None:
            nonlocal word, word_x0, word_y0, word_x1, word_y1
            if not word:
                return
            append_word(TextWord(word, (word_x0, word_y0, word_x1, word_y1)))
            word = ""

        def extend_word(char: str, bbox: tuple[float, float, float, float]) -> None:
            nonlocal word, word_x0, word_y0, word_x1, word_y1
            if not word:
                word_x0, word_y0, word_x1, word_y1 = bbox
            else:
                word_x0 = min(word_x0, bbox[0])
                word_y0 = min(word_y0, bbox[1])
                word_x1 = max(word_x1, bbox[2])
                word_y1 = max(word_y1, bbox[3])
            word += char

        def append_space() -> None:
            if parts and parts[-1] == " ":
                return
            flush_word()
            append_part(" ")

        for segment in reconstructed.segments:
            if segment.separator_before:
                append_space()
            text = segment.text
            text_length = len(text)
            for index, char in enumerate(text):
                bbox = layout_line_segment_char_bbox(segment, index, text_length)
                if char.isspace():
                    append_space()
                    continue
                extend_word(char, bbox)
                append_part(char)

        flush_word()
        return "".join(parts).rstrip(), tuple(words)


def layout_line_segment_char_bbox(
    segment: LayoutLineTextSegment,
    index: int,
    text_length: int,
) -> tuple[float, float, float, float]:
    if text_length <= 1:
        return segment.advance_bbox
    x0, y0, x1, y1 = segment.advance_bbox
    if segment.rotation_angle in (90, 270):
        step = (y1 - y0) / text_length
        char_y0 = y0 + step * index
        return (x0, char_y0, x1, char_y0 + step)
    step = (x1 - x0) / text_length
    char_x0 = x0 + step * index
    return (char_x0, y0, char_x0 + step, y1)


class LayoutGeometryIssue(Record):
    __slots__ = ("code", "severity", "subject", "bbox", "message", "details", "repairable")

    code: str
    severity: str
    subject: str
    bbox: Rectangle | None
    message: str
    details: tuple[tuple[str, object], ...]
    repairable: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "code",
        "severity",
        "subject",
        "bbox",
        "message",
        "details",
        "repairable",
    )
    __match_args__ = ("code", "severity", "subject", "bbox", "message", "details", "repairable")

    def __init__(
        self,
        code: str,
        severity: str,
        subject: str,
        bbox: Rectangle | None = None,
        message: str = "",
        details: tuple[tuple[str, object], ...] = (),
        repairable: bool = False,
    ) -> None:
        frozen_setattr(self, "code", code)
        frozen_setattr(self, "severity", severity)
        frozen_setattr(self, "subject", subject)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "message", message)
        frozen_setattr(self, "details", details)
        frozen_setattr(self, "repairable", repairable)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"code={self.code!r}, "
            f"severity={self.severity!r}, "
            f"subject={self.subject!r}, "
            f"bbox={self.bbox!r}, "
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"repairable={self.repairable!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.code == other.code
            and self.severity == other.severity
            and self.subject == other.subject
            and self.bbox == other.bbox
            and self.message == other.message
            and self.details == other.details
            and self.repairable == other.repairable
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.code,
                self.severity,
                self.subject,
                self.bbox,
                self.message,
                self.details,
                self.repairable,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        code = changes.pop("code", self.code)
        severity = changes.pop("severity", self.severity)
        subject = changes.pop("subject", self.subject)
        bbox = changes.pop("bbox", self.bbox)
        message = changes.pop("message", self.message)
        details = changes.pop("details", self.details)
        repairable = changes.pop("repairable", self.repairable)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(code, severity, subject, bbox, message, details, repairable)


class LayoutGeometrySummary(Record):
    __slots__ = (
        "issue_count",
        "error_count",
        "warning_count",
        "repairable_count",
        "text_run_count",
        "line_count",
    )

    issue_count: int
    error_count: int
    warning_count: int
    repairable_count: int
    text_run_count: int
    line_count: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "issue_count",
        "error_count",
        "warning_count",
        "repairable_count",
        "text_run_count",
        "line_count",
    )
    __match_args__ = (
        "issue_count",
        "error_count",
        "warning_count",
        "repairable_count",
        "text_run_count",
        "line_count",
    )

    def __init__(
        self,
        issue_count: int,
        error_count: int,
        warning_count: int,
        repairable_count: int,
        text_run_count: int,
        line_count: int,
    ) -> None:
        frozen_setattr(self, "issue_count", issue_count)
        frozen_setattr(self, "error_count", error_count)
        frozen_setattr(self, "warning_count", warning_count)
        frozen_setattr(self, "repairable_count", repairable_count)
        frozen_setattr(self, "text_run_count", text_run_count)
        frozen_setattr(self, "line_count", line_count)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"issue_count={self.issue_count!r}, "
            f"error_count={self.error_count!r}, "
            f"warning_count={self.warning_count!r}, "
            f"repairable_count={self.repairable_count!r}, "
            f"text_run_count={self.text_run_count!r}, "
            f"line_count={self.line_count!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.issue_count == other.issue_count
            and self.error_count == other.error_count
            and self.warning_count == other.warning_count
            and self.repairable_count == other.repairable_count
            and self.text_run_count == other.text_run_count
            and self.line_count == other.line_count
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.issue_count,
                self.error_count,
                self.warning_count,
                self.repairable_count,
                self.text_run_count,
                self.line_count,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        issue_count = changes.pop("issue_count", self.issue_count)
        error_count = changes.pop("error_count", self.error_count)
        warning_count = changes.pop("warning_count", self.warning_count)
        repairable_count = changes.pop("repairable_count", self.repairable_count)
        text_run_count = changes.pop("text_run_count", self.text_run_count)
        line_count = changes.pop("line_count", self.line_count)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            issue_count,
            error_count,
            warning_count,
            repairable_count,
            text_run_count,
            line_count,
        )


def text_run_geometry_issues(run: TextRun) -> tuple[LayoutGeometryIssue, ...]:
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

    advance_bbox = finite_rect(run.advance_bbox, require_positive=False)
    ink_bbox = finite_rect(run.ink_bbox, require_positive=False)
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

    cluster_bboxes: list[Rectangle] = []
    for cluster_index, cluster in enumerate(clusters):
        cluster_bbox = finite_rect(cluster.advance_bbox, require_positive=False)
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

    words = line.text_and_words()[1]
    word_bboxes = tuple(
        bbox for word in words if (bbox := word.bbox) is not None and bbox_is_positive(bbox)
    )
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
        issues.extend(
            replace(issue, details=(*issue.details, ("line_index", line_index)))
            for issue in layout_line_geometry_issues(line)
        )
        for run_index, run in enumerate(line.runs):
            issues.extend(
                replace(
                    issue,
                    details=(
                        *issue.details,
                        ("run_index", run_index),
                        ("line_index", line_index),
                    ),
                )
                for issue in text_run_geometry_issues(run)
            )
    return tuple(issues)


def page_layout_geometry_summary(lines: list[LayoutLine]) -> LayoutGeometrySummary:
    issues = page_layout_geometry_issues(lines)
    error_count = 0
    warning_count = 0
    repairable_count = 0
    text_run_count = 0
    for issue in issues:
        if issue.severity == "error":
            error_count += 1
        elif issue.severity == "warning":
            warning_count += 1
        if issue.repairable:
            repairable_count += 1
    for line in lines:
        text_run_count += len(line.runs)
    return LayoutGeometrySummary(
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        repairable_count=repairable_count,
        text_run_count=text_run_count,
        line_count=len(lines),
    )


def bbox_is_positive(bbox: Rectangle | None) -> bool:
    return bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def bbox_width(bbox: Rectangle) -> float:
    return bbox[2] - bbox[0]


def bbox_height(bbox: Rectangle) -> float:
    return bbox[3] - bbox[1]
