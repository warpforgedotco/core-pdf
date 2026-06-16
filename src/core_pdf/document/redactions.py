from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from core_pdf.layout.geometry import RectBox, bbox_intersection_lengths, bbox_tuple, merge_bbox

RedactionClass = Literal[
    "bad_redaction",
    "date",
    "filing_stamp",
    "placeholder",
    "numeric_fragment",
    "visible_prose",
    "unknown",
]

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE_RE = re.compile(r"\b[0-3]?\d[/\-][0-3]?\d[/\-]\d{2,4}\b")
_FILING_RE = re.compile(r"(?i)filed.*\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{1,4})?")


@dataclass(frozen=True, slots=True)
class RedactionFeatures:
    hidden_text: str
    text_length: int
    token_count: int
    digit_count: int
    alpha_count: int
    whitespace_count: int
    repeated_char_ratio: float
    has_year: bool
    has_date_pattern: bool
    has_filing_stamp_pattern: bool
    is_all_caps: bool
    is_numeric_only: bool
    is_visible_prose: bool
    paint_group_size: int
    occlusion_ratio: float
    fill_opacity: float | None
    paint_order_delta: int


@dataclass(frozen=True, slots=True)
class RedactionTextSpan:
    text: str
    bbox: tuple[float, float, float, float]
    seqno: int
    fill: tuple[float, ...] | None
    visible: bool


@dataclass(frozen=True, slots=True)
class RedactionPaintSpan:
    kind: Literal["rect", "path", "image", "stroke", "fill"]
    bbox: tuple[float, float, float, float] | None
    seqno: int
    fill: tuple[float, ...] | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None = None
    stroke_opacity: float | None = None


@dataclass(frozen=True, slots=True)
class RedactionCandidate:
    bbox: tuple[float, float, float, float]
    paint: RedactionPaintSpan
    paint_spans: tuple[RedactionPaintSpan, ...]
    hidden_text: str
    visible_text: str
    occlusion_ratio: float
    reasons: tuple[str, ...]
    confidence: float
    features: RedactionFeatures
    class_name: RedactionClass
    score: float


@dataclass(frozen=True, slots=True)
class RedactionAnalysis:
    candidates: tuple[RedactionCandidate, ...]
    text_spans: tuple[RedactionTextSpan, ...]
    paint_spans: tuple[RedactionPaintSpan, ...]
    paint_groups: tuple[tuple[RedactionPaintSpan, ...], ...]


@dataclass(frozen=True, slots=True)
class Glyph:
    text: str
    bbox: tuple[float, float, float, float]
    seqno: int
    fill: tuple[float, ...] | None


class RedactionAnalyzer:
    """High-level analysis logic for detecting and classifying redactions."""

    def analyze(self, page: Any) -> RedactionAnalysis:
        """Perform redaction analysis on the page."""
        text_spans = self.iter_text_spans(page)
        glyphs = self.iter_glyphs(page)
        paint_spans = self.iter_paint_spans(page.get_drawings())
        paint_groups = self.cluster_paint_spans(paint_spans)

        candidates: list[RedactionCandidate] = []
        for paint_group in paint_groups:
            candidate = self.build_candidate(paint_group, glyphs)
            if candidate:
                candidates.append(candidate)

        return RedactionAnalysis(
            candidates=tuple(candidates),
            text_spans=tuple(text_spans),
            paint_spans=tuple(paint_spans),
            paint_groups=tuple(paint_groups),
        )

    def iter_text_spans(self, page: Any) -> list[RedactionTextSpan]:
        runs = page.chars
        if runs:
            return [build_text_span(r) for r in runs if r.text]
        return [build_text_span(s) for s in page.get_texttrace()]

    def iter_glyphs(self, page: Any) -> list[Glyph]:
        glyphs: list[Glyph] = []
        for span in page.get_texttrace():
            seqno = span["seqno"]
            fill = span["color"]
            for code, _, _, rect in span["chars"]:
                glyphs.append(
                    Glyph(
                        text=chr(code),
                        bbox=bbox_tuple(rect),
                        seqno=seqno,
                        fill=fill,
                    )
                )
        return glyphs

    def iter_paint_spans(self, drawings: list[dict[str, Any]]) -> list[RedactionPaintSpan]:
        spans: list[RedactionPaintSpan] = []
        for drawing in drawings:
            items = drawing.get("items", ())
            bbox = drawing.get("rect")
            if not items:
                continue
            fill = drawing.get("fill")
            fill_opacity = drawing.get("fill_opacity")
            stroke_color = drawing.get("stroke_color")
            stroke_opacity = drawing.get("stroke_opacity")
            has_fill = fill_opacity == 1 and is_dark_color(fill)
            has_stroke = stroke_opacity == 1 and is_dark_color(stroke_color)
            if not has_fill and not has_stroke:
                continue
            rect = bbox
            if rect is None:
                item_rects = [item_rect for _, item_rect in items if isinstance(item_rect, RectBox)]
                if item_rects:
                    rect = item_rects[0]
                    for item_rect in item_rects[1:]:
                        rect = merge_bbox(bbox_tuple(rect), bbox_tuple(item_rect))
            if rect is None:
                continue
            if has_stroke and not has_fill and len(items) < 4:
                continue
            span_kind: Literal["rect", "path", "image", "stroke", "fill"] = "fill"
            if drawing.get("kind") == "stroke" or has_stroke and not has_fill:
                span_kind = "stroke"
            elif drawing.get("kind") == "fill" and len(items) > 1:
                span_kind = "path"
            spans.append(
                RedactionPaintSpan(
                    kind=span_kind,
                    bbox=bbox_tuple(rect) if rect is not None else None,
                    seqno=drawing["seqno"],
                    fill=fill,
                    fill_opacity=fill_opacity,
                    stroke_color=stroke_color,
                    stroke_opacity=stroke_opacity,
                )
            )
        return spans

    def cluster_paint_spans(self, spans: list[RedactionPaintSpan]) -> list[tuple[RedactionPaintSpan, ...]]:
        if not spans:
            return []

        groups_by_key: dict[tuple[Any, ...], list[tuple[int, RedactionPaintSpan]]] = {}
        for idx, span in enumerate(spans):
            key = paint_cluster_key(span)
            groups_by_key.setdefault(key, []).append((idx, span))

        parents = list(range(len(spans)))

        def find(i: int) -> int:
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parents[rb] = ra

        for group in groups_by_key.values():
            g_len = len(group)
            for i in range(g_len):
                ai, span_a = group[i]
                for j in range(i + 1, g_len):
                    bi, span_b = group[j]
                    if self.are_paint_spans_adjacent(span_a, span_b):
                        union(ai, bi)

        result_groups: dict[int, list[RedactionPaintSpan]] = {}
        for idx, span in enumerate(spans):
            root = find(idx)
            result_groups.setdefault(root, []).append(span)

        result: list[tuple[RedactionPaintSpan, ...]] = []
        for group in result_groups.values():
            group.sort(key=lambda s: s.seqno)
            result.append(tuple(group))
        result.sort(key=lambda g: g[0].seqno if g else -1)
        return result

    def are_paint_spans_adjacent(self, left: RedactionPaintSpan, right: RedactionPaintSpan) -> bool:
        if left.bbox is None or right.bbox is None:
            return False
        if paint_cluster_key(left) != paint_cluster_key(right):
            return False
        if left.seqno > right.seqno and left.seqno - right.seqno > 128:
            return False
        if right.seqno > left.seqno and right.seqno - left.seqno > 128:
            return False
        overlap_x, overlap_y, overlap_area = bbox_intersection_lengths(left.bbox, right.bbox)
        if overlap_area <= 0.0:
            return False
        if overlap_x < 2.0 or overlap_y < 2.0:
            return False
        return True

    def build_candidate(
        self, paint_group: tuple[RedactionPaintSpan, ...], glyphs: list[Glyph]
    ) -> RedactionCandidate | None:
        paint = paint_group[0]
        first_bbox = paint.bbox
        assert first_bbox is not None
        group_bbox: tuple[float, float, float, float] = first_bbox
        min_seqno = paint.seqno
        max_seqno = paint.seqno
        for member in paint_group[1:]:
            m_bbox = member.bbox
            assert m_bbox is not None
            group_bbox = merge_bbox(group_bbox, m_bbox)
            if member.seqno < min_seqno:
                min_seqno = member.seqno
            if member.seqno > max_seqno:
                max_seqno = member.seqno

        group_rect = RectBox.from_bbox(
            group_bbox, seqno=min_seqno, fill=paint.fill, fill_opacity=paint.fill_opacity
        )
        hidden_glyphs: list[Glyph] = []
        visible_glyphs: list[Glyph] = []
        hidden_text_area = 0.0
        overlapped_text_area = 0.0
        reasons = ["opaque_dark_fill"]
        if len(paint_group) > 1:
            reasons.append("grouped_paint_traces")

        paint_order_deltas: list[int] = []
        for glyph in glyphs:
            text_rect = RectBox.from_bbox(glyph.bbox, seqno=glyph.seqno, fill=glyph.fill)
            intersection = text_rect.get_intersection_area(group_rect)
            if intersection <= 0:
                continue
            text_area = abs(text_rect.get_area())
            overlapped_text_area += text_area
            if glyph.seqno < min_seqno:
                hidden_glyphs.append(glyph)
                hidden_text_area += intersection
                reasons.append("text_hidden_by_paint")
                paint_order_deltas.append(min_seqno - glyph.seqno)
            elif glyph.seqno > max_seqno:
                visible_glyphs.append(glyph)

        if not hidden_glyphs:
            return None

        occlusion_ratio = (hidden_text_area / overlapped_text_area) if overlapped_text_area > 0 else 0.0
        paint_order_delta = min(paint_order_deltas) if paint_order_deltas else 0
        hidden_text = self.text_from_glyphs(hidden_glyphs)
        visible_text = self.text_from_glyphs(visible_glyphs)

        features = self.extract_redaction_features(
            hidden_text, paint_group, occlusion_ratio, paint_order_delta
        )

        class_name, score, classify_reasons = self.classify_redaction(features)
        confidence = min(1.0, 0.5 + occlusion_ratio * 0.5)

        return RedactionCandidate(
            bbox=group_bbox,
            paint=paint,
            paint_spans=paint_group,
            hidden_text=hidden_text,
            visible_text=visible_text,
            occlusion_ratio=occlusion_ratio,
            reasons=tuple(dict.fromkeys([*reasons, *classify_reasons])),
            confidence=confidence,
            features=features,
            class_name=class_name,
            score=score,
        )

    def text_from_glyphs(self, glyphs: list[Glyph]) -> str:
        if not glyphs:
            return ""
        lines: list[dict[str, Any]] = []
        for glyph in sorted(glyphs, key=lambda g: (-(g.bbox[1] + g.bbox[3]) * 0.5, g.bbox[0], g.seqno)):
            cy = (glyph.bbox[1] + glyph.bbox[3]) * 0.5
            height = glyph.bbox[3] - glyph.bbox[1]
            placed = False
            for line in lines:
                if abs(line["cy"] - cy) <= line["tol"]:
                    line["glyphs"].append(glyph)
                    line["cy"] = (line["cy"] * line["n"] + cy) / (line["n"] + 1)
                    line["n"] += 1
                    line["tol"] = max(line["tol"], max(1.0, height * 0.6))
                    placed = True
                    break
            if not placed:
                lines.append({"cy": cy, "tol": max(1.0, height * 0.6), "glyphs": [glyph], "n": 1})

        text_parts: list[str] = []
        for line in sorted(lines, key=lambda line: -line["cy"]):
            line_glyphs = sorted(line["glyphs"], key=lambda glyph: (glyph.bbox[0], glyph.seqno))
            previous: Glyph | None = None
            for glyph in line_glyphs:
                if previous is not None:
                    gap = glyph.bbox[0] - previous.bbox[2]
                    previous_width = previous.bbox[2] - previous.bbox[0]
                    if gap > max(1.5, previous_width * 0.4):
                        text_parts.append(" ")
                text_parts.append(glyph.text)
                previous = glyph
            text_parts.append(" ")
        return normalize_text("".join(text_parts))

    def extract_redaction_features(
        self,
        hidden_text: str,
        paint_group: tuple[RedactionPaintSpan, ...],
        occlusion_ratio: float,
        paint_order_delta: int,
    ) -> RedactionFeatures:
        text_length = len(hidden_text)
        token_count = len(hidden_text.split())
        digit_count = sum(1 for ch in hidden_text if ch.isdigit())
        alpha_count = sum(1 for ch in hidden_text if ch.isalpha())
        whitespace_count = sum(1 for ch in hidden_text if ch.isspace())

        is_visible_prose = bool(
            (token_count >= 4 and alpha_count >= 15)
            or (text_length >= 60 and token_count >= 10)
            or (text_length >= 20 and token_count >= 2 and alpha_count >= max(10, text_length // 2))
        )

        return RedactionFeatures(
            hidden_text=hidden_text,
            text_length=text_length,
            token_count=token_count,
            digit_count=digit_count,
            alpha_count=alpha_count,
            whitespace_count=whitespace_count,
            repeated_char_ratio=repeated_char_ratio(hidden_text),
            has_year=bool(_YEAR_RE.search(hidden_text)),
            has_date_pattern=bool(_DATE_RE.search(hidden_text)),
            has_filing_stamp_pattern=bool(_FILING_RE.search(hidden_text)),
            is_all_caps=bool(hidden_text) and all(not ch.islower() for ch in hidden_text if ch.isalpha()),
            is_numeric_only=bool(hidden_text) and not any(ch.isalpha() for ch in hidden_text),
            is_visible_prose=is_visible_prose,
            paint_group_size=len(paint_group),
            occlusion_ratio=occlusion_ratio,
            fill_opacity=paint_group[0].fill_opacity,
            paint_order_delta=paint_order_delta,
        )

    def classify_redaction(
        self, features: RedactionFeatures
    ) -> tuple[RedactionClass, float, tuple[str, ...]]:
        reasons: list[str] = []
        if self.is_redaction_date(features):
            reasons.append("date_pattern")
            return "date", 0.97, tuple(reasons)
        if features.has_filing_stamp_pattern:
            reasons.append("filing_stamp_pattern")
            return "filing_stamp", 0.95, tuple(reasons)
        if self.is_redaction_placeholder(features):
            reasons.append("placeholder_like")
            return "placeholder", 0.9, tuple(reasons)
        if self.is_redaction_numeric(features):
            reasons.append("numeric_fragment")
            return "numeric_fragment", 0.88, tuple(reasons)
        if (
            features.digit_count > 0
            and features.token_count >= 3
            and features.text_length <= 24
            and features.alpha_count <= 10
        ):
            reasons.append("short_mixed_prose")
            return "visible_prose", 0.83, tuple(reasons)
        if (
            features.paint_group_size > 1
            and not features.has_date_pattern
            and not features.is_numeric_only
            and features.text_length <= 60
            and features.alpha_count >= 20
            and features.token_count >= 4
            and features.occlusion_ratio >= 0.2
        ):
            reasons.append("grouped_redaction")
            return "bad_redaction", 0.89, tuple(reasons)
        if features.is_visible_prose:
            reasons.append("visible_prose")
            return "visible_prose", 0.85, tuple(reasons)

        score = (
            0.3 * min(1.0, features.text_length / 20.0)
            + 0.25 * min(1.0, features.alpha_count / 12.0)
            + 0.25 * min(1.0, features.paint_order_delta / 40.0)
            + 0.15 * min(1.0, features.occlusion_ratio * 2.0)
            + 0.05 * (1.0 if features.token_count >= 2 else 0.0)
            + 0.05 * (1.0 if features.fill_opacity == 1 else 0.0)
        )
        if score >= 0.65:
            if (
                features.text_length >= 8
                and features.alpha_count >= 6
                and (features.token_count >= 2 or features.alpha_count >= 12)
            ):
                reasons.append("occluded_with_opaque_paint")
                return "bad_redaction", score, tuple(reasons)
            if features.digit_count > 0 and features.alpha_count == 0:
                reasons.append("numeric_fragment")
                return "numeric_fragment", 0.9, tuple(reasons)
        return "unknown", score, tuple(reasons)

    def is_redaction_date(self, features: RedactionFeatures) -> bool:
        return features.has_date_pattern or (features.has_year and features.is_numeric_only)

    def is_redaction_placeholder(self, features: RedactionFeatures) -> bool:
        if features.is_all_caps and features.text_length <= 3:
            return True
        if features.repeated_char_ratio >= 0.8:
            return True
        if features.text_length <= 4 and features.alpha_count <= 1:
            return True
        return False

    def is_redaction_numeric(self, features: RedactionFeatures) -> bool:
        if features.is_numeric_only and features.text_length > 0:
            return True
        if (
            features.text_length >= 20
            and features.digit_count >= 20
            and features.digit_count >= features.text_length // 2
            and features.alpha_count <= 12
        ):
            return True
        return False


def is_dark_color(color: tuple[float, ...] | None) -> bool:
    if color is None:
        return False
    return sum(color[:3]) <= 0.6


def span_text(chars: list[tuple[int, int, int, RectBox]]) -> str:
    return "".join(chr(char[0]) for char in chars)


def build_text_span(span: Any) -> RedactionTextSpan:
    if isinstance(span, dict):
        text = span.get("text") or span_text(span["chars"])
        return RedactionTextSpan(
            text=text,
            bbox=bbox_tuple(span["bbox"]),
            seqno=int(span["seqno"]),
            fill=span.get("color"),
            visible=span.get("visible", True),
        )
    text = span.text or span_text(getattr(span, "chars", []))
    bbox_val = getattr(span, "bbox", None)
    if bbox_val is None:
        bbox_val = (span.x0, span.y0, span.x1, span.y1)
    return RedactionTextSpan(
        text=text,
        bbox=bbox_tuple(bbox_val),
        seqno=int(getattr(span, "seqno", -1)),
        fill=getattr(span, "fill_color", None),
        visible=bool(getattr(span, "visible", True)),
    )


def quantize_color(color: tuple[float, ...] | None) -> tuple[float, ...] | None:
    if color is None:
        return None
    return tuple(round(c, 3) for c in color[:3])


def paint_cluster_key(span: RedactionPaintSpan) -> tuple[Any, ...]:
    return (
        quantize_color(span.fill),
        span.fill_opacity,
        quantize_color(span.stroke_color),
        span.stroke_opacity,
    )


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def repeated_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    max_run = 1
    current_run = 1
    for idx in range(1, len(text)):
        if text[idx] == text[idx - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run / len(text)
