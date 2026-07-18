# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from core_ocr.impl import candidates as ocr_candidates
from core_ocr.impl import iterator_layout as ocr_iterator_layout
from core_ocr.impl import selection as ocr_selection
from core_ocr.impl.candidates import OcrCandidate
from core_ocr.impl.page_analysis import (
    TextGeometryLine,
    text_geometry_line_from_bbox,
)
from core_ocr.impl.text_analysis import (
    extracted_text_token_count,
    normalized_text_tokens,
    text_ocr_quality_score,
    token_alnum_count,
    vector_text_supports_schematic_tiled_ocr,
)
from core_ocr.impl.types import (
    TESSERACT_RIL_TEXTLINE,
    TESSERACT_RIL_WORD,
    OcrIteratorLayout,
    OcrObservation,
    OcrTextResult,
)
from core_ocr.impl.vector_text import VectorStrokeOcrResult

from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.common.ordering import LayoutAnalyzer
from core_pdf.impl.engine.rendering.models import RenderedPage

if TYPE_CHECKING:
    from core_layout.impl.layout.models import TextRun

OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TOKENS = 80
OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS = 12
OCR_SCHEMATIC_ROW_SUPPLEMENT_MIN_CONFIDENCE = 80
OCR_SCHEMATIC_ROW_SUPPLEMENT_VALUE_MIN_CONFIDENCE = 40
OCR_SCHEMATIC_ROW_SUPPLEMENT_VALUE_LINE_MIN_CONFIDENCE = 55
OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_TOKENS = 280
OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_PER_TOKEN = 20
OCR_SCHEMATIC_ROW_SUPPLEMENT_PIN_MAX_PER_TOKEN = 40
OCR_SCHEMATIC_ROW_SUPPLEMENT_RAIL_MAX_PER_TOKEN = 40
OCR_SCHEMATIC_ROW_SUPPLEMENT_VALUE_MAX_PER_TOKEN = 16
OCR_SCHEMATIC_ROW_SUPPLEMENT_REFERENCE_MAX_PER_TOKEN = 12
OCR_SCHEMATIC_OBSERVATION_GRAPH_MIN_CLUSTERS = 8
VECTOR_TABLE_SYMBOL_MIN_MARKS = 6
VECTOR_TABLE_SYMBOL_MIN_COLUMN_MARKS = 4
VECTOR_TABLE_SYMBOL_COLUMN_TOLERANCE = 8.0
VECTOR_TABLE_SYMBOL_ROW_TOLERANCE = 7.0
NONSPACE_TOKEN_RE = re.compile(r"\S+")
SCHEMATIC_EDGE_CHARS = "‘’“”«»`~_=|¦¬^°•·.,;:!?()[]{}<>€"
SCHEMATIC_REPAIR_WORDS = frozenset({"gnd", "vcc", "vdd", "vee", "vref"})
SCHEMATIC_CONFUSABLE_DIGITS = frozenset("@oOiIlLsS|")
SCHEMATIC_STANDALONE_VALUE_SIGNS = frozenset({"+", "-", "–", "—"})
SCHEMATIC_ARTIFACT_TOKEN_CHARS = frozenset('|=<>[]{}()\\/“”"`~^°•·¦¬!;:?')
SCHEMATIC_NET_LABEL_SEPARATORS = frozenset("_/–—-")


@dataclass(frozen=True)
class VectorTableSymbolMark:
    token: str
    bbox: tuple[float, float, float, float]
    x_center: float
    y_center: float


@dataclass(frozen=True)
class SchematicSupplementEntry:
    token: str
    key: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: int | None = None
    token_type: str | None = None
    source: str = ""
    evidence_type: str = ""
    evidence_count: int = 1
    source_count: int = 1


@dataclass(frozen=True)
class SchematicSupplementCluster:
    entries: tuple[SchematicSupplementEntry, ...]
    key: str
    token: str
    bbox: tuple[float, float, float, float] | None
    confidence: int | None
    token_type: str | None
    score: float


@dataclass(frozen=True)
class SchematicSupplementRow:
    clusters: tuple[SchematicSupplementCluster, ...]
    bbox: tuple[float, float, float, float] | None
    score: float


@dataclass(frozen=True)
class SchematicRenderedSupplementRow:
    text: str
    entries: tuple[SchematicSupplementEntry, ...]
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class PageRegionClassification:
    kind: str
    confidence: float
    signals: Mapping[str, Any]


@dataclass(frozen=True)
class TokenConsensusGraph:
    kind: str
    rows: tuple[SchematicSupplementRow, ...]
    clusters: tuple[SchematicSupplementCluster, ...]
    source_candidates: int
    evidence_count: int
    support_token_count: int


@dataclass(frozen=True)
class SchematicOcrRepairContext:
    enabled: bool
    support_tokens: frozenset[str]
    support_nonspace_tokens: frozenset[str]
    support_token_display: dict[str, str]


def region_classification_supports_schematic_consensus(
    classification: PageRegionClassification,
) -> bool:
    return classification.kind == "schematic" and classification.confidence >= 0.55


def schematic_candidate_layout_signal_count(
    candidates: tuple[OcrCandidate, ...],
) -> int:
    best = 0
    for candidate in candidates:
        if not candidate.result.line_rows and not candidate.result.word_rows:
            continue
        layout = OcrIteratorLayout(
            list(candidate.result.line_rows),
            list(candidate.result.word_rows),
            [],
        )
        best = max(best, schematic_layout_signal_count(layout))
    return best


def should_try_vector_table_symbol_supplement(
    text: str,
    chars: list[TextRun],
    media_box: tuple[float, float, float, float] | None,
) -> bool:
    if not text.strip():
        return False
    tokens = extracted_text_token_count(text)
    if not (40 <= tokens <= 2500):
        return False
    text_lines = [line for line in text.splitlines() if line.strip()]
    if not (10 <= len(text_lines) <= 220):
        return False
    lines = vector_table_symbol_text_lines(chars)
    return vector_table_symbol_text_lines_look_table_like(lines, media_box)


def append_vector_table_symbol_supplement(
    text: str,
    chars: list[TextRun],
    rendered: RenderedPage,
    media_box: tuple[float, float, float, float] | None,
) -> str:
    lines = vector_table_symbol_text_lines(chars)
    if not vector_table_symbol_text_lines_look_table_like(lines, media_box):
        return text
    marks = vector_table_symbol_marks(rendered)
    if len(marks) < VECTOR_TABLE_SYMBOL_MIN_MARKS:
        return text
    line_index_map = vector_table_symbol_text_line_index_map(text, lines)
    if not line_index_map:
        return text
    additions: defaultdict[int, list[tuple[float, str]]] = defaultdict(list)
    for row in vector_table_symbol_mark_rows(marks):
        target_index = vector_table_symbol_target_line_index(row, lines, media_box)
        if target_index is None:
            continue
        text_index = line_index_map.get(target_index)
        if text_index is None:
            continue
        for mark in sorted(row, key=lambda item: item.x_center):
            additions[text_index].append((mark.x_center, mark.token))
    if not additions:
        return text

    text_lines = text.splitlines()
    added = 0
    for index, row_additions in additions.items():
        if index >= len(text_lines):
            continue
        tokens = [token for ignored_x, token in sorted(row_additions)]
        if not tokens:
            continue
        suffix = " ".join(tokens)
        line = text_lines[index].rstrip()
        if line.endswith(f" {suffix}") or line == suffix:
            continue
        text_lines[index] = f"{line} {suffix}" if line else suffix
        added += len(tokens)
    if added < VECTOR_TABLE_SYMBOL_MIN_MARKS:
        return text
    return "\n".join(text_lines)


def vector_table_symbol_text_lines(chars: list[TextRun]) -> list[Any]:
    return [line for line in LayoutAnalyzer.cluster_into_lines(chars) if line.text().strip()]


def vector_table_symbol_text_lines_look_table_like(
    lines: list[Any],
    media_box: tuple[float, float, float, float] | None,
) -> bool:
    if len(lines) < 10:
        return False
    box = page_geometry.rect_box_tuple(media_box)
    if box is None:
        return False
    page_x0, _page_y0, page_x1, _page_y1 = box
    page_width = page_x1 - page_x0
    if page_width <= 0:
        return False
    left_rows = 0
    left_top: float | None = None
    right_header_lines = 0
    right_body_lines = 0
    for line in lines:
        text = line.text().strip()
        tokens = normalized_text_tokens(text)
        if not tokens:
            continue
        if (
            line.x0 <= page_x0 + page_width * 0.45
            and line.x1 <= page_x0 + page_width * 0.82
            and len(tokens) <= 28
        ):
            left_rows += 1
            left_top = line.y0 if left_top is None else max(left_top, line.y0)
    if left_rows < 8 or left_top is None:
        return False
    for line in lines:
        text = line.text().strip()
        tokens = normalized_text_tokens(text)
        if not tokens:
            continue
        if not (
            line.x0 >= page_x0 + page_width * 0.45
            and line.x1 <= page_x1 + page_width * 0.04
            and len(tokens) <= 12
        ):
            continue
        if line.y0 < left_top - 12.0:
            right_body_lines += 1
        else:
            right_header_lines += 1
    return 1 <= right_header_lines <= 6 and right_body_lines <= 3


def vector_table_symbol_marks(rendered: RenderedPage) -> list[VectorTableSymbolMark]:
    raw_marks: list[VectorTableSymbolMark] = []
    for item in getattr(rendered.display_list, "items", ()):
        mark = vector_table_symbol_mark_from_display_item(item)
        if mark is not None:
            raw_marks.append(mark)
    if len(raw_marks) < VECTOR_TABLE_SYMBOL_MIN_MARKS:
        return []
    return vector_table_symbol_filter_aligned_columns(raw_marks)


def vector_table_symbol_marks_from_drawings(
    drawings: Iterable[Any],
) -> list[VectorTableSymbolMark]:
    raw_marks: list[VectorTableSymbolMark] = []
    for drawing in drawings:
        mark = vector_table_symbol_mark(
            getattr(drawing, "kind", None),
            getattr(drawing, "bbox", None),
            getattr(drawing, "path", None),
        )
        if mark is not None:
            raw_marks.append(mark)
    if len(raw_marks) < VECTOR_TABLE_SYMBOL_MIN_MARKS:
        return []
    return vector_table_symbol_filter_aligned_columns(raw_marks)


def vector_table_symbol_mark_from_display_item(
    item: Any,
) -> VectorTableSymbolMark | None:
    data = getattr(item, "data", None)
    if not isinstance(data, dict):
        return None
    return vector_table_symbol_mark(
        getattr(item, "kind", None),
        data.get("bbox"),
        data.get("path"),
    )


def vector_table_symbol_mark(
    kind: Any,
    bbox_value: Any,
    path: Any,
) -> VectorTableSymbolMark | None:
    if kind != "fill":
        return None
    bbox = page_geometry.rect_box_tuple(bbox_value)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if not (5.0 <= width <= 28.0 and 5.0 <= height <= 24.0):
        return None
    token = vector_table_symbol_token_from_path(path, width, height)
    if token is None:
        return None
    return VectorTableSymbolMark(
        token,
        bbox,
        (x0 + x1) * 0.5,
        (y0 + y1) * 0.5,
    )


def vector_table_symbol_token_from_path(
    path: Any,
    width: float,
    height: float,
) -> str | None:
    if height <= 0:
        return None
    subpaths = list(getattr(path, "subpaths", ()) or ())
    if len(subpaths) != 1:
        return None
    subpath = subpaths[0]
    if getattr(subpath, "closed", False) is not True:
        return None
    points = list(getattr(subpath, "points", ()) or ())
    point_count = len(points)
    ratio = width / height
    if point_count == 6 and ratio >= 1.05:
        return "✓"
    if 8 <= point_count <= 14 and 0.65 <= ratio <= 1.45:
        return "x"
    return None


def vector_table_symbol_filter_aligned_columns(
    marks: list[VectorTableSymbolMark],
) -> list[VectorTableSymbolMark]:
    columns: list[list[VectorTableSymbolMark]] = []
    for mark in sorted(marks, key=lambda item: item.x_center):
        for column in columns:
            center = sum(item.x_center for item in column) / len(column)
            if abs(mark.x_center - center) <= VECTOR_TABLE_SYMBOL_COLUMN_TOLERANCE:
                column.append(mark)
                break
        else:
            columns.append([mark])
    aligned = [
        mark
        for column in columns
        if len(column) >= VECTOR_TABLE_SYMBOL_MIN_COLUMN_MARKS
        for mark in column
    ]
    if len(aligned) < VECTOR_TABLE_SYMBOL_MIN_MARKS:
        return []
    return aligned


def vector_table_symbol_mark_rows(
    marks: list[VectorTableSymbolMark],
) -> list[list[VectorTableSymbolMark]]:
    rows: list[list[VectorTableSymbolMark]] = []
    for mark in sorted(marks, key=lambda item: -item.y_center):
        for row in rows:
            center = sum(item.y_center for item in row) / len(row)
            if abs(mark.y_center - center) <= VECTOR_TABLE_SYMBOL_ROW_TOLERANCE:
                row.append(mark)
                break
        else:
            rows.append([mark])
    return rows


def vector_table_symbol_target_line_index(
    row: list[VectorTableSymbolMark],
    lines: list[Any],
    media_box: tuple[float, float, float, float] | None,
) -> int | None:
    box = page_geometry.rect_box_tuple(media_box)
    if box is None or not row:
        return None
    page_x0, _page_y0, page_x1, _page_y1 = box
    page_width = page_x1 - page_x0
    if page_width <= 0:
        return None
    row_y = sum(mark.y_center for mark in row) / len(row)
    row_min_x = min(mark.bbox[0] for mark in row)
    row_height = max(mark.bbox[3] - mark.bbox[1] for mark in row)
    best_index: int | None = None
    best_distance = 1_000_000.0
    for index, line in enumerate(lines):
        text = line.text().strip()
        if not text:
            continue
        tokens = normalized_text_tokens(text)
        if not tokens or len(tokens) > 32:
            continue
        if line.x0 > page_x0 + page_width * 0.48:
            continue
        if row_min_x <= line.x0 + page_width * 0.20:
            continue
        if row_min_x <= line.x1 + max(6.0, page_width * 0.012):
            continue
        line_height = max(1.0, line.y1 - line.y0)
        tolerance = max(
            VECTOR_TABLE_SYMBOL_ROW_TOLERANCE,
            line_height * 0.85,
            row_height * 1.15,
        )
        line_y = (line.y0 + line.y1) * 0.5
        distance = abs(row_y - line_y)
        if distance <= tolerance and distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def vector_table_symbol_text_line_index_map(
    text: str,
    lines: list[Any],
) -> dict[int, int]:
    text_lines = text.splitlines()
    nonempty_indexes = [index for index, line in enumerate(text_lines) if line.strip()]
    if len(nonempty_indexes) < min(8, len(lines)):
        return {}
    mapping: dict[int, int] = {}
    if abs(len(nonempty_indexes) - len(lines)) <= max(3, int(len(lines) * 0.18)):
        for geo_index, text_index in enumerate(nonempty_indexes[: len(lines)]):
            if vector_table_symbol_lines_share_content(
                lines[geo_index].text(),
                text_lines[text_index],
            ):
                mapping[geo_index] = text_index
        return mapping

    search_from = 0
    for geo_index, line in enumerate(lines):
        for offset, text_index in enumerate(nonempty_indexes[search_from:]):
            if not vector_table_symbol_lines_share_content(
                line.text(),
                text_lines[text_index],
            ):
                continue
            mapping[geo_index] = text_index
            search_from += offset + 1
            break
    return mapping


def vector_table_symbol_lines_share_content(left: str, right: str) -> bool:
    left_tokens = set(normalized_text_tokens(left))
    right_tokens = set(normalized_text_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    common = left_tokens.intersection(right_tokens)
    if common:
        return True
    left_compact = "".join(left_tokens)
    right_compact = "".join(right_tokens)
    return bool(
        left_compact
        and right_compact
        and (left_compact in right_compact or right_compact in left_compact)
    )


def repair_schematic_tile_layout(
    layout: OcrIteratorLayout,
    support_text: str,
) -> tuple[OcrIteratorLayout, int]:
    context = schematic_ocr_repair_context(layout, support_text)
    if not context.enabled:
        return layout, 0
    line_rows, repaired_lines = repair_schematic_iterator_rows(
        layout.textline_rows,
        context,
    )
    word_rows, repaired_words = repair_schematic_iterator_rows(
        layout.word_rows,
        context,
    )
    repaired = repaired_lines + repaired_words
    if repaired == 0:
        return layout, 0
    return OcrIteratorLayout(line_rows, word_rows, layout.symbol_rows), repaired


def repair_schematic_ocr_text_with_support(text: str, support_text: str) -> str:
    if not text or not support_text:
        return text
    layout = OcrIteratorLayout(
        [{"text": line} for line in text.splitlines() if line.strip()],
        [],
        [],
    )
    context = schematic_ocr_repair_context(layout, support_text)
    if not context.enabled:
        return text
    repaired = repair_schematic_ocr_row_text(text, context)
    return remove_schematic_ocr_artifact_tokens(repaired)


def schematic_supplement_coverage_lines(
    page: Any,
    vector_result: VectorStrokeOcrResult,
) -> tuple[TextGeometryLine, ...]:
    lines: list[TextGeometryLine] = []
    try:
        for line in page.get_text_lines():
            text = line.text().strip()
            if not text:
                continue
            lines.append(
                text_geometry_line_from_bbox(
                    text,
                    (float(line.x0), float(line.y0), float(line.x1), float(line.y1)),
                    source="native",
                    kind="schematic_coverage_line",
                )
            )
    except Exception:
        pass
    for line in vector_result.lines:
        text = line.text.strip()
        if not text:
            continue
        bbox = page_geometry.rect_box_tuple(line.bbox)
        if bbox is None:
            continue
        lines.append(
            text_geometry_line_from_bbox(
                text,
                bbox,
                line.confidence,
                source="vector_stroke",
                kind="schematic_coverage_line",
            )
        )
    return tuple(lines)


def schematic_ocr_repair_context(
    layout: OcrIteratorLayout,
    support_text: str,
) -> SchematicOcrRepairContext:
    support_tokens = frozenset(normalized_text_tokens(support_text))
    support_nonspace_tokens, support_token_display = schematic_support_repair_tokens(support_text)
    enabled = bool(support_nonspace_tokens) and (
        len(support_tokens) >= OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TOKENS
        or schematic_layout_signal_count(layout) >= 12
    )
    return SchematicOcrRepairContext(
        enabled,
        support_tokens,
        support_nonspace_tokens,
        support_token_display,
    )


def schematic_support_repair_tokens(
    support_text: str,
) -> tuple[frozenset[str], dict[str, str]]:
    tokens: set[str] = set()
    display: dict[str, str] = {}

    def add_token(raw_token: str) -> None:
        core = schematic_token_core(raw_token)
        if not core:
            return
        normalized = core.casefold()
        intrinsic = intrinsic_schematic_token_repair(normalized)
        if intrinsic is not None:
            normalized = schematic_token_core(intrinsic).casefold()
        if not schematic_support_token_is_repair_target(normalized):
            return
        tokens.add(normalized)
        display.setdefault(normalized, canonical_schematic_display_token(normalized))

    for match in NONSPACE_TOKEN_RE.finditer(support_text):
        add_token(match.group(0))
    for token in normalized_text_tokens(support_text):
        add_token(token)
    return frozenset(tokens), display


def schematic_layout_signal_count(layout: OcrIteratorLayout) -> int:
    count = 0
    for row in [*layout.textline_rows, *layout.word_rows]:
        for match in NONSPACE_TOKEN_RE.finditer(str(row.get("text", ""))):
            if schematic_token_looks_repairable(schematic_token_core(match.group(0))):
                count += 1
    return count


def repair_schematic_iterator_rows(
    rows: list[dict[str, Any]],
    context: SchematicOcrRepairContext,
) -> tuple[list[dict[str, Any]], int]:
    repaired_rows: list[dict[str, Any]] = []
    repaired_count = 0
    for row in rows:
        text = str(row.get("text", ""))
        repaired_text = repair_schematic_ocr_row_text(text, context)
        if repaired_text == text:
            repaired_rows.append(row)
            continue
        repaired_row = dict(row)
        repaired_row["text"] = repaired_text
        repaired_rows.append(repaired_row)
        repaired_count += 1
    return repaired_rows, repaired_count


def repair_schematic_ocr_row_text(
    text: str,
    context: SchematicOcrRepairContext,
) -> str:
    if not text:
        return text
    return NONSPACE_TOKEN_RE.sub(
        lambda match: repair_schematic_nonspace_token(match.group(0), context),
        text,
    )


def remove_schematic_ocr_artifact_tokens(text: str) -> str:
    if not text:
        return text
    cleaned_lines: list[str] = []
    removed = 0
    for line in text.splitlines():
        cleaned_tokens: list[str] = []
        for token in line.strip().split():
            if should_remove_schematic_ocr_artifact_token(token):
                removed += 1
                continue
            cleaned_tokens.append(token)
        if cleaned_tokens:
            cleaned_lines.append(" ".join(cleaned_tokens))
    if removed == 0 or not cleaned_lines:
        return text
    cleaned = "\n".join(cleaned_lines)
    original_tokens = extracted_text_token_count(text)
    cleaned_token_count = extracted_text_token_count(cleaned)
    if cleaned_token_count < int(original_tokens * 0.92):
        return text
    return cleaned


def should_remove_schematic_ocr_artifact_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped or stripped in SCHEMATIC_STANDALONE_VALUE_SIGNS:
        return False
    alnum = token_alnum_count(stripped)
    if alnum == 0:
        return any(ch in SCHEMATIC_ARTIFACT_TOKEN_CHARS for ch in stripped)
    if alnum == 1 and len(stripped) >= 3:
        return any(ch in SCHEMATIC_ARTIFACT_TOKEN_CHARS for ch in stripped)
    return False


def supplement_schematic_ocr_text_from_candidate(
    text: str,
    candidate: OcrCandidate | None,
    support_text: str,
    *,
    coverage_lines: tuple[TextGeometryLine, ...] = (),
) -> str:
    return schematic_ocr_text_candidate_supplement(
        text,
        candidate,
        support_text,
        coverage_lines=coverage_lines,
    )


def schematic_ocr_text_candidate_supplement(
    text: str,
    candidate: OcrCandidate | None,
    support_text: str,
    *,
    coverage_lines: tuple[TextGeometryLine, ...] = (),
) -> str:
    return schematic_ocr_text_candidates_supplement(
        text,
        (candidate,) if candidate is not None else (),
        support_text,
        coverage_lines=coverage_lines,
    )


def schematic_ocr_text_candidates_supplement(
    text: str,
    candidates: tuple[OcrCandidate, ...],
    support_text: str,
    *,
    coverage_lines: tuple[TextGeometryLine, ...] = (),
) -> str:
    if not text or not candidates or not support_text:
        return text
    if not vector_text_supports_schematic_tiled_ocr(support_text):
        return text
    ordered_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.name.startswith("rendered_page_") and candidate.name.endswith("_tiled")
        ),
        key=lambda candidate: ocr_selection.ocr_candidate_score(
            candidate, support_text=support_text
        ),
        reverse=True,
    )
    if not ordered_candidates:
        return text
    additions = schematic_fused_supplement_entries(
        text,
        tuple(ordered_candidates),
        support_text,
        coverage_lines=coverage_lines,
    )
    if not additions:
        return text
    supplement_text = render_positioned_schematic_supplement(additions)
    if not supplement_text:
        return text
    strict_additions = [
        entry for entry in additions if "line" not in entry.evidence_type.split("+")
    ]
    recall_additions = [entry for entry in additions if "line" in entry.evidence_type.split("+")]
    supplemented_text = apply_schematic_supplement_geometry(
        text,
        strict_additions,
        coverage_lines,
    )
    if recall_additions:
        recall_text = render_positioned_schematic_supplement(recall_additions)
        if recall_text:
            supplemented_text = supplemented_text.rstrip() + "\n" + recall_text
    return supplemented_text


def schematic_token_consensus_graph_candidate(
    candidates: Iterable[OcrCandidate],
    support_text: str,
    classification: PageRegionClassification,
) -> OcrCandidate | None:
    graph = schematic_token_consensus_graph(candidates, support_text, classification)
    if graph is None:
        return None
    text = emit_final_text_from_token_consensus_graph(graph)
    if not text:
        return None
    confidence = schematic_average_cluster_confidence(list(graph.clusters))
    line_rows = tuple(schematic_observation_graph_line_rows(list(graph.rows)))
    word_rows = tuple(schematic_observation_graph_word_rows(list(graph.clusters)))
    result = OcrTextResult(
        text,
        confidence,
        line_rows=line_rows,
        word_rows=word_rows,
        observations=tuple(schematic_observation_graph_observations(list(graph.clusters))),
    )
    return ocr_candidates.OcrCandidate(
        "schematic_token_consensus_graph",
        result,
        region_count=len(graph.clusters),
    )


def schematic_token_consensus_graph(
    candidates: Iterable[OcrCandidate],
    support_text: str,
    classification: PageRegionClassification,
) -> TokenConsensusGraph | None:
    candidate_tuple = tuple(candidates)
    if not region_classification_supports_schematic_consensus(classification):
        return None
    tiled_candidates = tuple(
        candidate
        for candidate in candidate_tuple
        if candidate.name.startswith("rendered_page_") and candidate.name.endswith("_tiled")
    )
    if not tiled_candidates:
        return None
    evidence_entries: list[SchematicSupplementEntry] = []
    for candidate in tiled_candidates:
        evidence_entries.extend(
            schematic_candidate_supplement_evidence_entries(
                candidate,
                support_text,
            )
        )
    clusters = schematic_accepted_supplement_clusters(
        schematic_supplement_evidence_clusters(evidence_entries)
    )
    if len(clusters) < OCR_SCHEMATIC_OBSERVATION_GRAPH_MIN_CLUSTERS:
        return None
    rows = schematic_supplement_rows(clusters)
    if not rows:
        return None
    support_targets, _ = schematic_support_repair_tokens(support_text)
    return TokenConsensusGraph(
        "schematic",
        tuple(rows),
        tuple(clusters),
        len(tiled_candidates),
        len(evidence_entries),
        len(support_targets),
    )


def emit_final_text_from_token_consensus_graph(graph: TokenConsensusGraph) -> str:
    return render_schematic_observation_graph_text(list(graph.rows))


def render_schematic_observation_graph_text(
    rows: list[SchematicSupplementRow],
) -> str:
    lines: list[str] = []
    for row in rows:
        tokens = [cluster.token for cluster in row.clusters if cluster.token.strip()]
        if tokens:
            lines.append(" ".join(tokens))
    return "\n".join(lines)


def schematic_average_cluster_confidence(
    clusters: list[SchematicSupplementCluster],
) -> int | None:
    confidences = [cluster.confidence for cluster in clusters if cluster.confidence is not None]
    if not confidences:
        return None
    return int(round(sum(confidences) / len(confidences)))


def schematic_observation_graph_line_rows(
    rows: list[SchematicSupplementRow],
) -> list[dict[str, Any]]:
    line_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        text = " ".join(cluster.token for cluster in row.clusters if cluster.token)
        if not text.strip() or row.bbox is None:
            continue
        line_rows.append(
            schematic_observation_row(
                text,
                row.bbox,
                int(round(row.score)),
                line_num=index,
                level=TESSERACT_RIL_TEXTLINE,
            )
        )
    return line_rows


def schematic_observation_graph_word_rows(
    clusters: list[SchematicSupplementCluster],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cluster in enumerate(
        sorted(clusters, key=schematic_cluster_order_key),
        start=1,
    ):
        if cluster.bbox is None:
            continue
        rows.append(
            schematic_observation_row(
                cluster.token,
                cluster.bbox,
                cluster.confidence,
                line_num=index,
                level=TESSERACT_RIL_WORD,
                token_type=cluster.token_type,
            )
        )
    return rows


def schematic_observation_row(
    text: str,
    bbox: tuple[float, float, float, float],
    confidence: int | None,
    *,
    line_num: int,
    level: int,
    token_type: str | None = None,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    row: dict[str, Any] = {
        "text": text,
        "conf": max(0, min(100, confidence)) if confidence is not None else None,
        "level": level,
        "left": int(round(x0)),
        "top": int(round(y0)),
        "width": max(1, int(round(x1 - x0))),
        "height": max(1, int(round(y1 - y0))),
        "page_bbox": bbox,
        "page_num": 1,
        "block_num": 1,
        "par_num": 1,
        "line_num": line_num,
        "word_num": line_num if level == TESSERACT_RIL_WORD else 0,
        "symbol_num": 0,
    }
    if token_type is not None:
        row["token_type"] = token_type
    return row


def schematic_observation_graph_observations(
    clusters: list[SchematicSupplementCluster],
) -> list[OcrObservation]:
    observations: list[OcrObservation] = []
    for index, cluster in enumerate(
        sorted(clusters, key=schematic_cluster_order_key),
        start=1,
    ):
        if cluster.bbox is None:
            continue
        x0, y0, x1, y1 = cluster.bbox
        observations.append(
            OcrObservation(
                text=cluster.token,
                level=TESSERACT_RIL_WORD,
                confidence=cluster.confidence,
                bbox=(
                    int(round(x0)),
                    int(round(y0)),
                    int(round(x1)),
                    int(round(y1)),
                ),
                page_bbox=cluster.bbox,
                source="schematic_token_consensus_graph",
                page_num=1,
                block_num=1,
                par_num=1,
                line_num=index,
                word_num=index,
                token_type=cluster.token_type,
            )
        )
    return observations


def apply_schematic_supplement_geometry(
    text: str,
    additions: list[SchematicSupplementEntry],
    coverage_lines: tuple[TextGeometryLine, ...],
) -> str:
    rows = positioned_schematic_supplement_rows(additions)
    supplement_lines = [row.text for row in rows if row.text]
    if not supplement_lines:
        return text
    base_lines = text.rstrip().splitlines()
    replacements: dict[int, int] = {}
    used_rows: set[int] = set()
    for line in coverage_lines:
        line_index = matching_schematic_text_line_index(
            base_lines,
            line.text,
            replacements.keys(),
        )
        if line_index is None:
            continue
        row_index = replacement_schematic_row_index(rows, line, used_rows)
        if row_index is None:
            continue
        replacements[line_index] = row_index
        used_rows.add(row_index)
    if not replacements:
        return text.rstrip() + "\n" + "\n".join(supplement_lines)

    merged_lines: list[str] = []
    for index, base_line in enumerate(base_lines):
        if index in replacements:
            merged_lines.append(rows[replacements[index]].text)
        else:
            merged_lines.append(base_line)
    merged_lines.extend(
        row.text for index, row in enumerate(rows) if index not in used_rows and row.text
    )
    return "\n".join(line for line in merged_lines if line.strip())


def matching_schematic_text_line_index(
    text_lines: list[str],
    coverage_text: str,
    replaced_indices: Iterable[int],
) -> int | None:
    normalized_coverage = coverage_text.strip()
    if not normalized_coverage:
        return None
    unavailable = set(replaced_indices)
    for index, line in enumerate(text_lines):
        if index in unavailable:
            continue
        if line.strip() == normalized_coverage:
            return index
    return None


def replacement_schematic_row_index(
    rows: list[SchematicRenderedSupplementRow],
    line: TextGeometryLine,
    used_rows: set[int],
) -> int | None:
    best_index: int | None = None
    best_score = 0.0
    for index, row in enumerate(rows):
        if index in used_rows:
            continue
        if not schematic_row_should_replace_coverage_line(row, line):
            continue
        score = schematic_row_line_overlap_score(row, line)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def schematic_row_should_replace_coverage_line(
    row: SchematicRenderedSupplementRow,
    line: TextGeometryLine,
) -> bool:
    if row.bbox is None or not row.text.strip():
        return False
    if row.text.strip() == line.text.strip():
        return False
    if len(row.entries) < 2:
        return False
    if schematic_row_line_overlap_score(row, line) < 0.20:
        return False
    confidence = line.confidence
    if confidence is not None:
        return confidence < 80
    return text_ocr_quality_score(line.text) >= 0.20


def schematic_row_line_overlap_score(
    row: SchematicRenderedSupplementRow,
    line: TextGeometryLine,
) -> float:
    row_observation = schematic_rendered_supplement_row_observation(row)
    line_observation = page_geometry.page_observation_from_text_line(
        line,
        source="coverage_text",
        kind="coverage_line",
    )
    if row_observation is None or line_observation is None:
        return 0.0
    overlap_ratio = page_geometry.observation_overlap_ratio(
        row_observation,
        line_observation,
        denominator="smaller",
    )
    if overlap_ratio > 0.0:
        return overlap_ratio
    row_center = page_geometry.observation_center(row_observation)
    line_center = page_geometry.observation_center(line_observation)
    if row_center is None or line_center is None:
        return 0.0
    row_mid_y = row_center[1]
    line_mid_y = line_center[1]
    row_height = max(1.0, page_geometry.observation_height(row_observation))
    line_height = max(1.0, page_geometry.observation_height(line_observation))
    if abs(row_mid_y - line_mid_y) > max(row_height, line_height) * 0.75:
        return 0.0
    row_x0, _row_y0, row_x1, _row_y1 = row_observation.bbox or (0.0, 0.0, 0.0, 0.0)
    line_x0, _line_y0, line_x1, _line_y1 = line_observation.bbox or (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    x_overlap = max(0.0, min(row_x1, line_x1) - max(row_x0, line_x0))
    return x_overlap / max(
        1.0,
        min(
            page_geometry.observation_width(row_observation),
            page_geometry.observation_width(line_observation),
        ),
    )


def schematic_supplement_cluster_observation(
    cluster: SchematicSupplementCluster,
) -> page_geometry.PageObservation | None:
    return page_geometry.page_observation_from_bbox(
        cluster.bbox,
        source="schematic",
        kind="schematic_cluster",
        text=cluster.token,
        confidence=page_geometry.numeric_confidence(cluster.confidence),
        provenance={
            "key": cluster.key,
            "token_type": cluster.token_type,
            "score": cluster.score,
            "entry_count": len(cluster.entries),
        },
    )


def schematic_supplement_entry_observation(
    entry: SchematicSupplementEntry,
) -> page_geometry.PageObservation | None:
    return page_geometry.page_observation_from_bbox(
        entry.bbox,
        source=entry.source or "schematic",
        kind="schematic_entry",
        text=entry.token,
        confidence=page_geometry.numeric_confidence(entry.confidence),
        provenance={
            "key": entry.key,
            "token_type": entry.token_type,
            "evidence_type": entry.evidence_type,
            "evidence_count": entry.evidence_count,
            "source_count": entry.source_count,
        },
    )


def schematic_rendered_supplement_row_observation(
    row: SchematicRenderedSupplementRow,
) -> page_geometry.PageObservation | None:
    return page_geometry.page_observation_from_bbox(
        row.bbox,
        source="schematic",
        kind="schematic_row",
        text=row.text,
        provenance={"entry_count": len(row.entries)},
    )


def schematic_fused_supplement_entries(
    text: str,
    candidates: tuple[OcrCandidate, ...],
    support_text: str,
    *,
    coverage_lines: tuple[TextGeometryLine, ...] = (),
) -> list[SchematicSupplementEntry]:
    if not text.strip():
        return []
    evidence_entries: list[SchematicSupplementEntry] = []
    for candidate in candidates:
        entries = schematic_candidate_supplement_evidence_entries(
            candidate,
            support_text,
        )
        if not entries:
            continue
        evidence_entries.extend(entries)
    clusters = schematic_supplement_evidence_clusters(evidence_entries)
    accepted_clusters = schematic_accepted_supplement_clusters(clusters)
    uncovered_clusters = [
        cluster
        for cluster in accepted_clusters
        if not schematic_cluster_is_covered_by_text_geometry(cluster, coverage_lines)
    ]
    additions = schematic_select_key_supplement_entries(uncovered_clusters)
    room = OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_TOKENS - len(additions)
    if room > 0:
        recall_clusters = schematic_low_confidence_value_supplement_clusters(
            candidates,
            support_text,
            coverage_lines,
            uncovered_clusters,
        )
        if recall_clusters:
            recall_additions = schematic_select_key_supplement_entries(
                recall_clusters,
                max_tokens=room,
            )
            additions = sorted(
                [*additions, *recall_additions],
                key=schematic_supplement_entry_order_key,
            )
    return additions


def schematic_low_confidence_value_supplement_clusters(
    candidates: tuple[OcrCandidate, ...],
    support_text: str,
    coverage_lines: tuple[TextGeometryLine, ...],
    existing_clusters: list[SchematicSupplementCluster],
) -> list[SchematicSupplementCluster]:
    entries: list[SchematicSupplementEntry] = []
    for candidate in candidates:
        candidate_entries = schematic_candidate_supplement_evidence_entries(
            candidate,
            support_text,
            include_low_confidence_value_lines=True,
        )
        entries.extend(
            entry
            for entry in candidate_entries
            if schematic_entry_is_low_confidence_value_recall(entry)
        )
    clusters = schematic_accepted_supplement_clusters(
        schematic_supplement_evidence_clusters(entries)
    )
    return [
        cluster
        for cluster in clusters
        if cluster.token_type == "value"
        and not schematic_cluster_is_covered_by_text_geometry(cluster, coverage_lines)
        and not schematic_cluster_matches_existing_clusters(cluster, existing_clusters)
    ]


def schematic_entry_is_low_confidence_value_recall(
    entry: SchematicSupplementEntry,
) -> bool:
    if entry.token_type != "value":
        return False
    if entry.evidence_type == "line":
        return True
    confidence = entry.confidence if entry.confidence is not None else 0
    return confidence < OCR_SCHEMATIC_ROW_SUPPLEMENT_MIN_CONFIDENCE


def schematic_cluster_matches_existing_clusters(
    cluster: SchematicSupplementCluster,
    existing_clusters: list[SchematicSupplementCluster],
) -> bool:
    if cluster.bbox is None:
        return False
    return any(
        existing.key == cluster.key
        and existing.bbox is not None
        and schematic_entry_boxes_match(cluster.bbox, existing.bbox)
        for existing in existing_clusters
    )


def schematic_select_key_supplement_entries(
    accepted_clusters: list[SchematicSupplementCluster],
    *,
    max_tokens: int = OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_TOKENS,
) -> list[SchematicSupplementEntry]:
    additions: list[SchematicSupplementEntry] = []
    clusters_by_key: defaultdict[str, list[SchematicSupplementCluster]] = defaultdict(list)
    for cluster in sorted(
        accepted_clusters,
        key=lambda cluster: (-cluster.score, schematic_cluster_order_key(cluster)),
    ):
        clusters_by_key[cluster.key].append(cluster)
    for key, key_clusters in sorted(
        clusters_by_key.items(),
        key=lambda item: (
            schematic_supplement_key_is_untyped(item[1]),
            -sum(schematic_supplement_cluster_occurrence_count(cluster) for cluster in item[1]),
            schematic_cluster_order_key(item[1][0]),
        ),
    ):
        remaining_for_key = min(
            sum(schematic_supplement_cluster_occurrence_count(cluster) for cluster in key_clusters),
            schematic_supplement_cluster_max_per_token(key_clusters),
        )
        if remaining_for_key <= 0:
            continue
        for cluster in sorted(
            key_clusters,
            key=lambda cluster: (-cluster.score, schematic_cluster_order_key(cluster)),
        ):
            if remaining_for_key <= 0:
                break
            if len(additions) >= max_tokens:
                return sorted(additions, key=schematic_supplement_entry_order_key)
            entry = schematic_entry_from_cluster(cluster)
            room = max_tokens - len(additions)
            repeat_count = min(
                schematic_supplement_cluster_occurrence_count(cluster),
                remaining_for_key,
                room,
            )
            additions.extend([entry] * repeat_count)
            remaining_for_key -= repeat_count
    return sorted(additions, key=schematic_supplement_entry_order_key)


def schematic_accepted_supplement_clusters(
    clusters: list[SchematicSupplementCluster],
) -> list[SchematicSupplementCluster]:
    base_clusters = [
        cluster for cluster in clusters if schematic_base_supplement_cluster_is_accepted(cluster)
    ]
    context_clusters = [
        cluster
        for cluster in clusters
        if cluster not in base_clusters
        and schematic_symbol_only_cluster_is_accepted(cluster, base_clusters)
    ]
    return base_clusters + context_clusters


def schematic_candidate_supplement_evidence_entries(
    candidate: OcrCandidate,
    support_text: str,
    *,
    include_low_confidence_value_lines: bool = False,
) -> list[SchematicSupplementEntry]:
    context = schematic_ocr_repair_context(
        OcrIteratorLayout(
            list(candidate.result.line_rows),
            list(candidate.result.word_rows),
            [],
        ),
        support_text,
    )
    if not context.enabled:
        return []
    support_targets, support_display = schematic_support_repair_tokens(support_text)
    entries: list[SchematicSupplementEntry] = []
    for values in schematic_row_supplement_entry_map(
        candidate.result.word_rows,
        context,
        support_targets,
        support_display,
        source=candidate.name,
        evidence_type="word",
        value_min_confidence=(
            OCR_SCHEMATIC_ROW_SUPPLEMENT_VALUE_MIN_CONFIDENCE
            if include_low_confidence_value_lines
            else None
        ),
    ).values():
        entries.extend(values)
    for values in schematic_row_supplement_entry_map(
        schematic_symbol_word_rows(candidate.result.symbol_rows),
        context,
        support_targets,
        support_display,
        source=candidate.name,
        evidence_type="symbol",
    ).values():
        entries.extend(values)
    if include_low_confidence_value_lines:
        for values in schematic_line_value_supplement_entry_map(
            candidate.result.line_rows,
            context,
            support_targets,
            support_display,
            source=candidate.name,
        ).values():
            entries.extend(values)
    return entries


def schematic_supplement_evidence_clusters(
    entries: list[SchematicSupplementEntry],
) -> list[SchematicSupplementCluster]:
    raw_clusters: list[list[SchematicSupplementEntry]] = []
    for entry in sorted(entries, key=schematic_supplement_entry_order_key):
        target_cluster: list[SchematicSupplementEntry] | None = None
        for cluster in raw_clusters:
            if schematic_entry_belongs_to_cluster(entry, cluster):
                target_cluster = cluster
                break
        if target_cluster is None:
            raw_clusters.append([entry])
        else:
            target_cluster.append(entry)
    clusters: list[SchematicSupplementCluster] = []
    for raw_cluster in raw_clusters:
        supplement_cluster = schematic_supplement_cluster(raw_cluster)
        if supplement_cluster is not None:
            clusters.append(supplement_cluster)
    return clusters


def schematic_entry_belongs_to_cluster(
    entry: SchematicSupplementEntry,
    cluster: list[SchematicSupplementEntry],
) -> bool:
    if entry.bbox is None:
        return any(other.key == entry.key and other.bbox is None for other in cluster)
    return any(
        other.key == entry.key
        and other.bbox is not None
        and schematic_entry_boxes_match(entry.bbox, other.bbox)
        for other in cluster
    )


def schematic_entry_boxes_match(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_observation = page_geometry.page_observation_from_bbox(
        left,
        source="schematic",
        kind="schematic_entry",
    )
    right_observation = page_geometry.page_observation_from_bbox(
        right,
        source="schematic",
        kind="schematic_entry",
    )
    if left_observation is None or right_observation is None:
        return False
    if (
        page_geometry.observation_overlap_ratio(
            left_observation,
            right_observation,
            denominator="smaller",
        )
        >= 0.25
    ):
        return True
    left_center = page_geometry.observation_center(left_observation)
    right_center = page_geometry.observation_center(right_observation)
    if left_center is None or right_center is None:
        return False
    left_width = max(1.0, page_geometry.observation_width(left_observation))
    left_height = max(1.0, page_geometry.observation_height(left_observation))
    right_width = max(1.0, page_geometry.observation_width(right_observation))
    right_height = max(1.0, page_geometry.observation_height(right_observation))
    return (
        abs(left_center[0] - right_center[0]) <= max(left_width, right_width) * 0.65
        and abs(left_center[1] - right_center[1]) <= max(left_height, right_height) * 0.75
    )


def schematic_supplement_cluster(
    entries: list[SchematicSupplementEntry],
) -> SchematicSupplementCluster | None:
    if not entries:
        return None
    key_scores: dict[str, float] = {}
    for key in {entry.key for entry in entries}:
        key_entries = [entry for entry in entries if entry.key == key]
        source_count = len({entry.source for entry in key_entries if entry.source})
        evidence_count = len(key_entries)
        average_confidence = schematic_average_confidence(key_entries) or 0.0
        type_bonus = 3.0 if any(entry.token_type for entry in key_entries) else 0.0
        key_scores[key] = (
            evidence_count * 10.0 + source_count * 5.0 + average_confidence / 12.0 + type_bonus
        )
    key = max(key_scores, key=lambda item: key_scores[item])
    key_entries = [entry for entry in entries if entry.key == key]
    representative = max(
        key_entries,
        key=lambda entry: (
            entry.confidence if entry.confidence is not None else -1,
            len(entry.token),
        ),
    )
    source_count = len({entry.source for entry in key_entries if entry.source})
    confidence = schematic_average_confidence(key_entries)
    bbox = union_schematic_entry_bboxes(key_entries)
    return SchematicSupplementCluster(
        entries=tuple(key_entries),
        key=key,
        token=representative.token,
        bbox=bbox,
        confidence=confidence,
        token_type=representative.token_type,
        score=key_scores[key] + schematic_cluster_geometry_bonus(key_entries),
    )


def schematic_base_supplement_cluster_is_accepted(
    cluster: SchematicSupplementCluster,
) -> bool:
    evidence_types = {entry.evidence_type for entry in cluster.entries}
    source_count = len({entry.source for entry in cluster.entries if entry.source})
    if {"word", "symbol"}.issubset(evidence_types):
        return True
    if (
        cluster.token_type == "value"
        and "line" in evidence_types
        and ("word" in evidence_types or "symbol" in evidence_types)
    ):
        return True
    return bool(source_count >= 2 and len(cluster.entries) >= 2)


def schematic_symbol_only_cluster_is_accepted(
    cluster: SchematicSupplementCluster,
    context_clusters: list[SchematicSupplementCluster],
) -> bool:
    evidence_types = {entry.evidence_type for entry in cluster.entries}
    if evidence_types != {"symbol"} or len(cluster.entries) != 1:
        return False
    if cluster.bbox is None:
        return False
    confidence = cluster.confidence if cluster.confidence is not None else 0
    core = schematic_token_core(cluster.token).casefold()
    has_context = schematic_cluster_has_row_context(cluster, context_clusters)
    if cluster.token_type == "opamp_label":
        return confidence >= 84 or (has_context and confidence >= 80)
    if cluster.token_type == "pin":
        return (
            len(core) == 1
            and core.isdigit()
            and (confidence >= 92 or (has_context and confidence >= 88))
        )
    if cluster.token_type == "value":
        return confidence >= 94 or (has_context and confidence >= 90)
    if cluster.token_type == "reference":
        return confidence >= 95 or (has_context and confidence >= 92)
    return False


def schematic_cluster_has_row_context(
    cluster: SchematicSupplementCluster,
    context_clusters: list[SchematicSupplementCluster],
) -> bool:
    if cluster.bbox is None:
        return False
    for other in context_clusters:
        if other is cluster or other.bbox is None:
            continue
        if not schematic_clusters_share_row(cluster, other):
            continue
        if not schematic_token_types_are_contextual(
            cluster.token_type,
            other.token_type,
        ):
            continue
        return True
    return False


def schematic_token_types_are_contextual(
    token_type: str | None,
    other_type: str | None,
) -> bool:
    if token_type == "opamp_label":
        return other_type in {"pin", "reference", "opamp_label"}
    if token_type == "pin":
        return other_type in {"pin", "reference", "value", "rail", "opamp_label"}
    if token_type == "value":
        return other_type in {"pin", "reference", "value"}
    if token_type == "reference":
        return other_type in {"pin", "value", "rail", "opamp_label"}
    return False


def schematic_cluster_is_covered_by_text_geometry(
    cluster: SchematicSupplementCluster,
    coverage_lines: tuple[TextGeometryLine, ...],
) -> bool:
    if cluster.bbox is None:
        return False
    if cluster.token_type in {"pin", "opamp_label"}:
        return False
    for line in coverage_lines:
        if not schematic_coverage_line_reliably_matches_cluster(line, cluster):
            continue
        if schematic_cluster_is_covered_by_line(cluster, line):
            return True
    return False


def schematic_coverage_line_reliably_matches_cluster(
    line: TextGeometryLine,
    cluster: SchematicSupplementCluster,
) -> bool:
    counts = schematic_supplement_text_counts(line.text)
    if cluster.key not in counts:
        return False
    token_count = sum(counts.values())
    if line.confidence is not None and line.confidence < 90:
        return False
    if line.confidence is None and token_count > 2:
        return False
    return token_count <= 4


def schematic_cluster_is_covered_by_line(
    cluster: SchematicSupplementCluster,
    line: TextGeometryLine,
) -> bool:
    cluster_observation = schematic_supplement_cluster_observation(cluster)
    line_observation = page_geometry.page_observation_from_text_line(
        line,
        source="coverage_text",
        kind="coverage_line",
    )
    if cluster_observation is None or line_observation is None:
        return False
    if (
        page_geometry.observation_overlap_ratio(
            cluster_observation,
            line_observation,
            denominator="left",
        )
        >= 0.55
    ):
        return True
    cluster_center = page_geometry.observation_center(cluster_observation)
    if cluster_center is None or line_observation.bbox is None:
        return False
    x0, y0, x1, y1 = line_observation.bbox
    mid_x, mid_y = cluster_center
    height = max(1.0, page_geometry.observation_height(line_observation))
    return x0 - height <= mid_x <= x1 + height and y0 - height * 0.5 <= mid_y <= y1 + height * 0.5


def schematic_supplement_rows(
    clusters: list[SchematicSupplementCluster],
) -> list[SchematicSupplementRow]:
    positioned = [cluster for cluster in clusters if cluster.bbox is not None]
    unpositioned = [cluster for cluster in clusters if cluster.bbox is None]
    raw_rows: list[list[SchematicSupplementCluster]] = []
    for cluster in sorted(positioned, key=schematic_cluster_order_key):
        target_row: list[SchematicSupplementCluster] | None = None
        for row in raw_rows:
            if schematic_clusters_share_row(cluster, row[0]):
                target_row = row
                break
        if target_row is None:
            target_row = []
            raw_rows.append(target_row)
        target_row.append(cluster)
    rows = [schematic_supplement_row(row) for row in raw_rows if row]
    if unpositioned:
        rows.append(schematic_supplement_row(unpositioned))
    return rows


def schematic_supplement_row(
    clusters: list[SchematicSupplementCluster],
) -> SchematicSupplementRow:
    ordered = tuple(sorted(clusters, key=schematic_cluster_order_key))
    bbox = union_schematic_cluster_bboxes(list(ordered))
    return SchematicSupplementRow(
        ordered,
        bbox,
        schematic_supplement_row_score(ordered),
    )


def schematic_supplement_row_score(
    clusters: tuple[SchematicSupplementCluster, ...],
) -> float:
    token_types = {cluster.token_type for cluster in clusters if cluster.token_type}
    score = sum(cluster.score for cluster in clusters)
    score += min(len(clusters), 8) * 4.0
    if {"value", "pin"}.issubset(token_types):
        score += 18.0
    if {"reference", "pin"}.issubset(token_types):
        score += 16.0
    if {"opamp_label", "pin"}.issubset(token_types):
        score += 14.0
    if {"rail", "pin"}.issubset(token_types):
        score += 10.0
    if len(token_types) >= 3:
        score += 8.0
    return score


def schematic_clusters_share_row(
    left: SchematicSupplementCluster,
    right: SchematicSupplementCluster,
) -> bool:
    left_observation = schematic_supplement_cluster_observation(left)
    right_observation = schematic_supplement_cluster_observation(right)
    if left_observation is None or right_observation is None:
        return False
    left_center = page_geometry.observation_center(left_observation)
    right_center = page_geometry.observation_center(right_observation)
    if left_center is None or right_center is None:
        return False
    left_height = max(1.0, page_geometry.observation_height(left_observation))
    right_height = max(1.0, page_geometry.observation_height(right_observation))
    left_mid_y = left_center[1]
    right_mid_y = right_center[1]
    if abs(left_mid_y - right_mid_y) > max(left_height, right_height) * 0.75:
        return False
    left_x0, _left_y0, left_x1, _left_y1 = left_observation.bbox or (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    right_x0, _right_y0, right_x1, _right_y1 = right_observation.bbox or (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    horizontal_gap = max(0.0, max(left_x0, right_x0) - min(left_x1, right_x1))
    return horizontal_gap <= max(left_height, right_height) * 14.0


def schematic_supplement_cluster_occurrence_count(
    cluster: SchematicSupplementCluster,
) -> int:
    return 1


def schematic_supplement_cluster_max_per_token(
    clusters: list[SchematicSupplementCluster],
) -> int:
    token_type = schematic_supplement_cluster_token_type(clusters)
    return schematic_supplement_token_max_per_token(token_type)


def schematic_supplement_key_is_untyped(
    clusters: list[SchematicSupplementCluster],
) -> bool:
    return schematic_supplement_cluster_token_type(clusters) is None


def schematic_supplement_cluster_token_type(
    clusters: list[SchematicSupplementCluster],
) -> str | None:
    token_types = Counter(
        cluster.token_type for cluster in clusters if cluster.token_type is not None
    )
    return token_types.most_common(1)[0][0] if token_types else None


def schematic_supplement_token_max_per_token(token_type: str | None) -> int:
    if token_type == "pin":
        return OCR_SCHEMATIC_ROW_SUPPLEMENT_PIN_MAX_PER_TOKEN
    if token_type == "rail":
        return OCR_SCHEMATIC_ROW_SUPPLEMENT_RAIL_MAX_PER_TOKEN
    if token_type == "value":
        return OCR_SCHEMATIC_ROW_SUPPLEMENT_VALUE_MAX_PER_TOKEN
    if token_type == "reference":
        return OCR_SCHEMATIC_ROW_SUPPLEMENT_REFERENCE_MAX_PER_TOKEN
    return OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_PER_TOKEN


def schematic_entry_from_cluster(
    cluster: SchematicSupplementCluster,
) -> SchematicSupplementEntry:
    sources = sorted({entry.source for entry in cluster.entries if entry.source})
    evidence_types = sorted(
        {entry.evidence_type for entry in cluster.entries if entry.evidence_type}
    )
    return SchematicSupplementEntry(
        token=cluster.token,
        key=cluster.key,
        bbox=cluster.bbox,
        confidence=cluster.confidence,
        token_type=cluster.token_type,
        source="+".join(sources),
        evidence_type="+".join(evidence_types),
        evidence_count=len(cluster.entries),
        source_count=len(sources),
    )


def schematic_average_confidence(
    entries: list[SchematicSupplementEntry],
) -> int | None:
    confidences = [entry.confidence for entry in entries if entry.confidence is not None]
    if not confidences:
        return None
    return int(round(sum(confidences) / len(confidences)))


def union_schematic_entry_bboxes(
    entries: list[SchematicSupplementEntry],
) -> tuple[float, float, float, float] | None:
    boxes = [entry.bbox for entry in entries if entry.bbox is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def union_schematic_cluster_bboxes(
    clusters: list[SchematicSupplementCluster],
) -> tuple[float, float, float, float] | None:
    boxes = [cluster.bbox for cluster in clusters if cluster.bbox is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def schematic_cluster_geometry_bonus(
    entries: list[SchematicSupplementEntry],
) -> float:
    return 4.0 if any(entry.bbox is not None for entry in entries) else 0.0


def schematic_cluster_order_key(
    cluster: SchematicSupplementCluster,
) -> tuple[float, float, str]:
    if cluster.bbox is None:
        return (float("inf"), float("inf"), cluster.token)
    x0, y0, x1, y1 = cluster.bbox
    return (-(y0 + y1) * 0.5, x0, cluster.token)


def schematic_row_supplement_tokens(
    text: str,
    result: OcrTextResult,
    support_text: str,
    *,
    max_tokens: int | None = None,
    max_per_token: int | None = None,
) -> list[str]:
    return [
        entry.token
        for entry in schematic_row_supplement_entries(
            text,
            result,
            support_text,
            max_entries=max_tokens,
            max_per_token=max_per_token,
        )
    ]


def schematic_supplement_text_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in NONSPACE_TOKEN_RE.finditer(text):
        key = schematic_supplement_token_key(match.group(0))
        if key is not None:
            counts[key] += 1
    return counts


def schematic_supplement_token_key(token: str) -> str | None:
    core = schematic_token_core(token).casefold()
    if not core:
        return None
    if core in SCHEMATIC_STANDALONE_VALUE_SIGNS:
        return core
    if any(ch.isalnum() for ch in core):
        return core
    return None


def schematic_row_supplement_entries(
    text: str,
    result: OcrTextResult,
    support_text: str,
    *,
    max_entries: int | None = None,
    max_per_token: int | None = None,
) -> list[SchematicSupplementEntry]:
    token_limit = (
        OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_TOKENS if max_entries is None else max(0, max_entries)
    )
    if token_limit <= 0:
        return []
    per_token_limit = (
        OCR_SCHEMATIC_ROW_SUPPLEMENT_MAX_PER_TOKEN
        if max_per_token is None
        else max(0, max_per_token)
    )
    if per_token_limit <= 0:
        return []
    context = schematic_ocr_repair_context(
        OcrIteratorLayout(list(result.line_rows), list(result.word_rows), []),
        support_text,
    )
    if not context.enabled:
        return []
    support_targets, support_display = schematic_support_repair_tokens(support_text)
    word_entries = schematic_row_supplement_entry_map(
        result.word_rows,
        context,
        support_targets,
        support_display,
    )
    symbol_entries = schematic_row_supplement_entry_map(
        schematic_symbol_word_rows(result.symbol_rows),
        context,
        support_targets,
        support_display,
    )
    word_counts = Counter({token: len(entries) for token, entries in word_entries.items()})
    symbol_counts = Counter({token: len(entries) for token, entries in symbol_entries.items()})
    current_counts = schematic_supplement_text_counts(text)
    agreed_counts = word_counts + symbol_counts
    additions: list[SchematicSupplementEntry] = []
    for token, observed in agreed_counts.most_common():
        if token not in word_counts or token not in symbol_counts:
            continue
        missing = observed - current_counts[token]
        if missing <= 0:
            continue
        token_room = token_limit - len(additions)
        if token_room <= 0:
            return additions
        token_entries = [
            *word_entries.get(token, ()),
            *symbol_entries.get(token, ()),
        ]
        if not token_entries:
            continue
        token_entry_limit = min(
            per_token_limit,
            schematic_supplement_token_max_per_token(token_entries[0].token_type),
        )
        additions.extend(token_entries[: min(missing, token_entry_limit, token_room)])
        if len(additions) >= token_limit:
            return additions[:token_limit]
    return additions


def schematic_row_supplement_entry_map(
    rows: Any,
    context: SchematicOcrRepairContext,
    support_targets: frozenset[str],
    support_display: dict[str, str],
    *,
    source: str = "",
    evidence_type: str = "",
    value_min_confidence: int | None = None,
) -> dict[str, list[SchematicSupplementEntry]]:
    entries: dict[str, list[SchematicSupplementEntry]] = defaultdict(list)
    for row in rows:
        raw_text = str(row.get("text", ""))
        confidence = ocr_iterator_layout.iterator_row_confidence(row)
        if confidence is None:
            continue
        token = schematic_row_supplement_display_token(
            raw_text,
            context,
            support_targets,
            support_display,
        )
        if token is None:
            continue
        key = schematic_supplement_token_key(token)
        if key is None:
            continue
        token_type = classify_schematic_token_type(token)
        min_confidence = schematic_supplement_entry_min_confidence(
            token_type,
            value_min_confidence=value_min_confidence,
        )
        if confidence < min_confidence:
            continue
        entries[key].append(
            SchematicSupplementEntry(
                token=token,
                key=key,
                bbox=ocr_iterator_layout.iterator_row_page_bbox(row),
                confidence=confidence,
                token_type=token_type,
                source=source or str(row.get("source", "")),
                evidence_type=evidence_type,
            )
        )
    for key, values in list(entries.items()):
        entries[key] = sorted(values, key=schematic_supplement_entry_order_key)
    return entries


def schematic_supplement_entry_min_confidence(
    token_type: str | None,
    *,
    value_min_confidence: int | None = None,
) -> int:
    if token_type == "value" and value_min_confidence is not None:
        return value_min_confidence
    return OCR_SCHEMATIC_ROW_SUPPLEMENT_MIN_CONFIDENCE


def schematic_line_value_supplement_entry_map(
    rows: Any,
    context: SchematicOcrRepairContext,
    support_targets: frozenset[str],
    support_display: dict[str, str],
    *,
    source: str = "",
) -> dict[str, list[SchematicSupplementEntry]]:
    entries: dict[str, list[SchematicSupplementEntry]] = defaultdict(list)
    for row in rows:
        raw_text = str(row.get("text", ""))
        confidence = ocr_iterator_layout.iterator_row_confidence(row)
        if (
            confidence is None
            or confidence < OCR_SCHEMATIC_ROW_SUPPLEMENT_VALUE_LINE_MIN_CONFIDENCE
        ):
            continue
        row_bbox = ocr_iterator_layout.iterator_row_page_bbox(row)
        if row_bbox is None:
            continue
        for match in NONSPACE_TOKEN_RE.finditer(raw_text):
            token = schematic_row_supplement_display_token(
                match.group(0),
                context,
                support_targets,
                support_display,
            )
            if token is None or classify_schematic_token_type(token) != "value":
                continue
            key = schematic_supplement_token_key(token)
            if key is None:
                continue
            entries[key].append(
                SchematicSupplementEntry(
                    token=token,
                    key=key,
                    bbox=schematic_line_token_bbox(raw_text, match, row_bbox),
                    confidence=confidence,
                    token_type="value",
                    source=source or str(row.get("source", "")),
                    evidence_type="line",
                )
            )
    for key, values in list(entries.items()):
        entries[key] = sorted(values, key=schematic_supplement_entry_order_key)
    return entries


def schematic_line_token_bbox(
    text: str,
    match: re.Match[str],
    row_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = row_bbox
    width = max(1.0, x1 - x0)
    text_len = max(1, len(text))
    token_x0 = x0 + width * (match.start() / text_len)
    token_x1 = x0 + width * (match.end() / text_len)
    return (token_x0, y0, max(token_x0 + 1.0, token_x1), y1)


def render_positioned_schematic_supplement(
    entries: list[SchematicSupplementEntry],
) -> str:
    return "\n".join(row.text for row in positioned_schematic_supplement_rows(entries) if row.text)


def positioned_schematic_supplement_rows(
    entries: list[SchematicSupplementEntry],
) -> list[SchematicRenderedSupplementRow]:
    if not entries:
        return []
    positioned = [entry for entry in entries if entry.bbox is not None]
    if not positioned:
        text = " ".join(entry.token for entry in entries)
        return [SchematicRenderedSupplementRow(text, tuple(entries), None)] if text else []
    lines: list[list[SchematicSupplementEntry]] = []
    for entry in sorted(positioned, key=schematic_supplement_entry_order_key):
        entry_observation = schematic_supplement_entry_observation(entry)
        entry_center = (
            page_geometry.observation_center(entry_observation)
            if entry_observation is not None
            else None
        )
        if entry_observation is None or entry_center is None:
            continue
        mid_y = entry_center[1]
        height = max(1.0, page_geometry.observation_height(entry_observation))
        target_line: list[SchematicSupplementEntry] | None = None
        for line in lines:
            line_observation = schematic_supplement_entry_observation(line[0])
            line_center = (
                page_geometry.observation_center(line_observation)
                if line_observation is not None
                else None
            )
            if line_observation is None or line_center is None:
                continue
            line_mid_y = line_center[1]
            line_height = max(1.0, page_geometry.observation_height(line_observation))
            if abs(mid_y - line_mid_y) <= max(height, line_height) * 0.65:
                target_line = line
                break
        if target_line is None:
            target_line = []
            lines.append(target_line)
        target_line.append(entry)
    rows: list[SchematicRenderedSupplementRow] = []
    for line in lines:
        ordered = sorted(line, key=lambda entry: (entry.bbox or (0.0,))[0])
        text = " ".join(entry.token for entry in ordered)
        if text:
            rows.append(
                SchematicRenderedSupplementRow(
                    text,
                    tuple(ordered),
                    union_schematic_entry_bboxes(list(ordered)),
                )
            )
    unpositioned_entries = [entry for entry in entries if entry.bbox is None]
    unpositioned = [entry.token for entry in unpositioned_entries]
    if unpositioned:
        rows.append(
            SchematicRenderedSupplementRow(
                " ".join(unpositioned),
                tuple(unpositioned_entries),
                None,
            )
        )
    return rows


def schematic_supplement_entry_order_key(
    entry: SchematicSupplementEntry,
) -> tuple[float, float, str]:
    if entry.bbox is None:
        return (float("inf"), float("inf"), entry.token)
    x0, y0, x1, y1 = entry.bbox
    return (-(y0 + y1) * 0.5, x0, entry.token)


def schematic_row_supplement_counts(
    rows: Any,
    context: SchematicOcrRepairContext,
    support_targets: frozenset[str],
    support_display: dict[str, str],
) -> tuple[Counter[str], dict[str, str]]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for raw_text, confidence in rows:
        if confidence is None or confidence < OCR_SCHEMATIC_ROW_SUPPLEMENT_MIN_CONFIDENCE:
            continue
        token = schematic_row_supplement_display_token(
            raw_text,
            context,
            support_targets,
            support_display,
        )
        if token is None:
            continue
        key = schematic_supplement_token_key(token)
        if key is None:
            continue
        counts[key] += 1
        display.setdefault(key, token)
    return counts, display


def schematic_symbol_word_texts(
    rows: tuple[dict[str, Any], ...],
) -> list[tuple[str, int | None]]:
    return [
        (str(row.get("text", "")), ocr_iterator_layout.iterator_row_confidence(row))
        for row in schematic_symbol_word_rows(rows)
    ]


def schematic_symbol_word_rows(
    rows: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=ocr_iterator_layout.iterator_row_page_order_key):
        key = (
            int(row.get("page_num", 1)),
            int(row.get("block_num", 1)),
            int(row.get("par_num", 1)),
            int(row.get("line_num", 1)),
            int(row.get("word_num", 0)),
        )
        grouped[key].append(row)
    words: list[dict[str, Any]] = []
    for symbol_rows in grouped.values():
        ordered_symbols = sorted(
            symbol_rows,
            key=lambda row: (
                int(row.get("symbol_num", 0)),
                int(row.get("left", 0)),
            ),
        )
        text = "".join(str(row.get("text", "")).strip() for row in ordered_symbols).strip()
        if not text:
            continue
        confidences = [
            confidence
            for row in symbol_rows
            if (confidence := ocr_iterator_layout.iterator_row_confidence(row)) is not None
        ]
        word_row: dict[str, Any] = {
            "text": text,
            "conf": min(confidences) if confidences else None,
            "level": TESSERACT_RIL_WORD,
            "page_num": int(ordered_symbols[0].get("page_num", 1)),
            "block_num": int(ordered_symbols[0].get("block_num", 1)),
            "par_num": int(ordered_symbols[0].get("par_num", 1)),
            "line_num": int(ordered_symbols[0].get("line_num", 1)),
            "word_num": int(ordered_symbols[0].get("word_num", 0)),
            "symbol_num": 0,
        }
        page_bbox = ocr_iterator_layout.union_iterator_row_page_bboxes(ordered_symbols)
        if page_bbox is not None:
            word_row["page_bbox"] = page_bbox
        words.append(word_row)
    return words


def schematic_row_supplement_display_token(
    raw_text: str,
    context: SchematicOcrRepairContext,
    support_targets: frozenset[str],
    support_display: dict[str, str],
) -> str | None:
    repaired = remove_schematic_ocr_artifact_tokens(
        repair_schematic_ocr_row_text(raw_text, context)
    ).strip()
    if not repaired or any(ch.isspace() for ch in repaired):
        return None
    core = schematic_token_core(repaired).casefold()
    if not core:
        return None
    if core in support_targets:
        return support_display.get(core, canonical_schematic_display_token(core))
    intrinsic = intrinsic_schematic_token_repair(core)
    if intrinsic is not None:
        return intrinsic
    if schematic_row_supplement_core_is_allowed(core):
        return canonical_schematic_display_token(core)
    return None


def schematic_row_supplement_core_is_allowed(core: str) -> bool:
    if core in SCHEMATIC_STANDALONE_VALUE_SIGNS:
        return True
    if core.isdigit():
        return 1 <= int(core) <= 15
    return schematic_support_token_is_repair_target(core)


def repair_schematic_nonspace_token(
    token: str,
    context: SchematicOcrRepairContext,
) -> str:
    if token == "—":
        return "–"
    prefix, core, suffix = split_schematic_token_edges(token)
    if not schematic_token_looks_repairable(core):
        return token
    target = schematic_token_repair_target(core, context)
    if target is None:
        return token
    return f"{prefix}{target}{suffix}"


def split_schematic_token_edges(token: str) -> tuple[str, str, str]:
    start = 0
    end = len(token)
    while start < end and token[start] in SCHEMATIC_EDGE_CHARS:
        start += 1
    while end > start and token[end - 1] in SCHEMATIC_EDGE_CHARS:
        end -= 1
    return token[:start], token[start:end], token[end:]


def schematic_token_core(token: str) -> str:
    return split_schematic_token_edges(token)[1]


def schematic_token_repair_target(
    token: str,
    context: SchematicOcrRepairContext,
) -> str | None:
    token_normalized = token.casefold()
    intrinsic = intrinsic_schematic_token_repair(token_normalized)
    if intrinsic is not None:
        return intrinsic
    if token_normalized in context.support_nonspace_tokens:
        return context.support_token_display.get(
            token_normalized,
            canonical_schematic_display_token(token_normalized),
        )
    best_target: str | None = None
    best_distance = float("inf")
    for target in context.support_nonspace_tokens:
        if target == token_normalized:
            continue
        if not schematic_tokens_are_comparable(token_normalized, target):
            continue
        distance = schematic_token_distance(token_normalized, target)
        if distance < best_distance:
            best_distance = distance
            best_target = target
    if best_target is None:
        return intrinsic_schematic_token_repair(token_normalized)
    max_length = max(len(token_normalized), len(best_target), 1)
    if schematic_consensus_distance_is_too_large(
        token_normalized,
        best_target,
        best_distance,
        max_length,
    ):
        return intrinsic_schematic_token_repair(token_normalized)
    return context.support_token_display.get(
        best_target,
        canonical_schematic_display_token(best_target),
    )


def schematic_consensus_distance_is_too_large(
    token: str,
    target: str,
    distance: float,
    max_length: int,
) -> bool:
    ratio = distance / max(max_length, 1)
    if target in SCHEMATIC_REPAIR_WORDS:
        return distance > 1.1 or ratio > 0.34
    if schematic_token_is_rail_like(target):
        return distance > 1.1 or ratio > 0.28
    if schematic_token_is_net_label(target):
        return distance > 2.25 or ratio > 0.20
    if schematic_repaired_bus_label_body(target) is not None:
        return distance > 1.25 or ratio > 0.28
    return distance > 0.55 or ratio > 0.22


def intrinsic_schematic_token_repair(token: str) -> str | None:
    voltage = intrinsic_schematic_voltage_repair(token)
    if voltage is not None:
        return voltage
    rail = intrinsic_schematic_rail_repair(token)
    if rail is not None:
        return rail
    bus_label = intrinsic_schematic_bus_label_repair(token)
    if bus_label is not None:
        return bus_label
    value = intrinsic_schematic_value_repair(token)
    if value is not None:
        return value
    return intrinsic_schematic_refdes_repair(token)


def intrinsic_schematic_voltage_repair(token: str) -> str | None:
    if len(token) == 3 and token[0] in {"+", "-"} and token[1:] == "sv":
        return f"{token[0]}5V"
    return None


def intrinsic_schematic_rail_repair(token: str) -> str | None:
    sign = token[:1] if token[:1] in {"+", "-"} else ""
    body = token[1:] if sign else token
    voltage_net = schematic_voltage_net_label_display(body)
    if voltage_net is not None:
        display = f"{sign}{voltage_net}"
        return display if display != token else None
    if body in SCHEMATIC_REPAIR_WORDS:
        display = canonical_schematic_display_token(token)
        return display if display != token else None
    if body.startswith("gnd") and (
        len(body) == 3 or body[3:].isdigit() or body[3:].startswith("_")
    ):
        display = f"{sign}{body.upper()}"
        return display if display != token else None
    for prefix in ("vcc", "vdd", "vss", "vee"):
        if not body.startswith(prefix):
            continue
        suffix = body[len(prefix) :]
        if not suffix or (len(suffix) == 1 and suffix.isalpha()):
            display = f"{sign}{body.upper()}"
            return display if display != token else None
    for prefix in ("vref", "vin", "vbat"):
        if not body.startswith(prefix):
            continue
        if body == prefix or (
            len(body) <= len(prefix) + 4
            and all(ch.isalnum() or ch == "_" for ch in body[len(prefix) :])
        ):
            display = f"{sign}{body.upper()}"
            return display if display != token else None
    return None


def intrinsic_schematic_bus_label_repair(token: str) -> str | None:
    sign = token[:1] if token[:1] in {"+", "-"} else ""
    body = token[1:] if sign else token
    repaired = schematic_repaired_bus_label_body(body)
    if repaired is None or repaired.casefold() == body.casefold():
        return None
    return f"{sign}{repaired}"


def schematic_repaired_bus_label_body(body: str) -> str | None:
    if not body:
        return None
    pieces: list[str] = []
    current: list[str] = []
    changed = False

    def flush_segment() -> None:
        nonlocal changed
        if not current:
            return
        segment = "".join(current)
        current.clear()
        repaired = schematic_repaired_net_label_segment(segment)
        if repaired is None:
            pieces.append(segment)
            return
        pieces.append(repaired)
        changed = True

    for char in body:
        if char in SCHEMATIC_NET_LABEL_SEPARATORS:
            flush_segment()
            pieces.append("–" if char == "—" else char)
        else:
            current.append(char)
    flush_segment()
    if not changed:
        return None
    return "".join(pieces).upper()


def schematic_repaired_net_label_segment(segment: str) -> str | None:
    i2c = schematic_repaired_i2c_segment(segment)
    if i2c is not None:
        return i2c
    return schematic_repaired_gpio_segment(segment)


def schematic_repaired_i2c_segment(segment: str) -> str | None:
    folded = segment.casefold()
    if len(folded) < 3:
        return None
    if folded[0] not in {"i", "1", "l", "|"}:
        return None
    if folded[1:3] != "2c":
        return None
    suffix = folded[3:]
    if suffix and not all(ch.isalnum() for ch in suffix):
        return None
    return f"I2C{suffix.upper()}"


def schematic_repaired_gpio_segment(segment: str) -> str | None:
    folded = segment.casefold()
    if len(folded) < 5 or not folded.startswith("gp"):
        return None
    if folded[2] not in {"i", "1", "l", "|"}:
        return None
    if folded[3] not in {"o", "0"}:
        return None
    suffix = folded[4:]
    if not suffix or not suffix[0].isdigit():
        return None
    if not all(ch.isalnum() or ch == "_" for ch in suffix):
        return None
    return f"GPIO{suffix.upper()}"


def intrinsic_schematic_value_repair(token: str) -> str | None:
    sign = token[:1] if token[:1] in {"+", "-"} else ""
    body = token[1:] if sign else token
    for unit in ("meg", "nf", "pf", "uf", "k", "v"):
        if not body.endswith(unit):
            continue
        prefix = body[: -len(unit)]
        repaired_prefix = schematic_digit_like_text(prefix)
        if repaired_prefix is None or repaired_prefix == prefix:
            return None
        return canonical_schematic_display_token(f"{sign}{repaired_prefix}{unit}")
    return None


def intrinsic_schematic_refdes_repair(token: str) -> str | None:
    sign = token[:1] if token[:1] in {"+", "-"} else ""
    body = token[1:] if sign else token
    if len(body) < 2 or body[0] not in {"c", "d", "j", "q", "r"}:
        return None
    suffix = body[1:]
    if not any(ch.isdigit() for ch in suffix):
        return None
    repaired_suffix = schematic_digit_like_text(suffix)
    if repaired_suffix is None or repaired_suffix == suffix:
        return None
    return canonical_schematic_display_token(f"{sign}{body[0]}{repaired_suffix}")


def schematic_digit_like_text(text: str) -> str | None:
    if not text:
        return None
    repaired: list[str] = []
    for ch in text:
        if ch == ".":
            repaired.append(ch)
            continue
        digit = schematic_digit_like_char(ch)
        if digit is None:
            return None
        repaired.append(digit)
    if not any(ch.isdigit() for ch in repaired):
        return None
    return "".join(repaired)


def schematic_digit_like_char(ch: str) -> str | None:
    folded = ch.casefold()
    if folded.isdigit():
        return folded
    if folded in {"@", "o"}:
        return "0"
    if folded in {"i", "l", "|"}:
        return "1"
    if folded == "s":
        return "5"
    return None


def schematic_tokens_are_comparable(token: str, target: str) -> bool:
    length_delta = abs(len(token) - len(target))
    if schematic_token_is_net_label(target):
        return (
            length_delta <= 2
            and schematic_token_has_net_label_shape(token)
            and schematic_net_label_digit_groups_are_compatible(token, target)
        )
    if length_delta > 1:
        return False
    target_has_digit = any(ch.isdigit() for ch in target)
    token_has_digit_like = any(ch.isdigit() or ch in SCHEMATIC_CONFUSABLE_DIGITS for ch in token)
    if target_has_digit:
        return token_has_digit_like
    if schematic_token_is_rail_like(target):
        return len(token) <= max(4, len(target) + 1)
    return target in SCHEMATIC_REPAIR_WORDS and len(token) <= 4


def schematic_token_looks_repairable(token: str) -> bool:
    if not token:
        return False
    if not (2 <= len(token) <= 40):
        return False
    if not any(ch.isalnum() for ch in token):
        return False
    folded = token.casefold()
    if schematic_token_has_net_label_shape(folded):
        return True
    if schematic_token_is_rail_like(folded):
        return True
    if schematic_repaired_bus_label_body(folded) is not None:
        return True
    if len(token) > 16:
        return False
    if any(ch.isdigit() or ch in SCHEMATIC_CONFUSABLE_DIGITS for ch in token):
        return True
    return len(token) <= 4 and token.upper() == token


def schematic_support_token_is_repair_target(token: str) -> bool:
    if token in SCHEMATIC_REPAIR_WORDS or schematic_token_is_rail_like(token):
        return True
    if schematic_token_is_net_label(token):
        return True
    if schematic_repaired_bus_label_body(token) is not None:
        return True
    body = token[1:] if token[:1] in {"+", "-"} else token
    if len(body) < 2:
        return False
    if body[0].isdigit():
        return any(ch.isalpha() for ch in body)
    for prefix in ("jp", "rv", "sw", "tp"):
        suffix = body[len(prefix) :] if body.startswith(prefix) else ""
        if suffix[:1].isdigit():
            return True
    return (
        body[0] in {"c", "d", "j", "l", "p", "q", "r", "s", "t", "u", "y"}
        and len(body) >= 2
        and body[1].isdigit()
    )


def schematic_token_is_rail_like(token: str) -> bool:
    body = token[1:] if token[:1] in {"+", "-", "–"} else token
    if schematic_voltage_net_label_display(body) is not None:
        return True
    if body in SCHEMATIC_REPAIR_WORDS:
        return True
    if body.startswith("gnd"):
        return len(body) == 3 or (len(body) <= 12 and (body[3].isdigit() or body[3] == "_"))
    for prefix in ("vcc", "vdd", "vss", "vee", "vref", "vin", "vbat"):
        if body.startswith(prefix) and len(body) <= 12:
            return any(ch.isalpha() for ch in body)
    return False


def schematic_voltage_net_label_display(body: str) -> str | None:
    folded = body.casefold()
    if "v" not in folded:
        return None
    prefix, _, suffix = folded.partition("v")
    if not prefix or not prefix.replace(".", "", 1).isdigit():
        return None
    if not suffix:
        return f"{prefix}V"
    if not all(ch.isdigit() or ch == "o" for ch in suffix):
        return None
    repaired_suffix = suffix.replace("o", "0")
    return f"{prefix}V{repaired_suffix}"


def schematic_token_is_net_label(token: str) -> bool:
    return schematic_token_has_net_label_shape(token) and any(ch.isdigit() for ch in token)


def schematic_net_label_digit_groups_are_compatible(
    token: str,
    target: str,
) -> bool:
    token_groups = schematic_net_label_digit_groups(token)
    target_groups = schematic_net_label_digit_groups(target)
    if not token_groups or not target_groups:
        return False
    return token_groups == target_groups


def schematic_net_label_digit_groups(token: str) -> tuple[str, ...]:
    body = token[1:] if token[:1] in {"+", "-", "–"} else token
    repaired = schematic_repaired_bus_label_body(body)
    normalized = (repaired if repaired is not None else body).casefold()
    groups = re.findall(r"\d+", normalized)
    return tuple(group.lstrip("0") or "0" for group in groups)


def schematic_token_has_net_label_shape(token: str) -> bool:
    body = token[1:] if token[:1] in {"+", "-", "–"} else token
    if not (4 <= len(body) <= 48):
        return False
    if not any(char in SCHEMATIC_NET_LABEL_SEPARATORS for char in body):
        return False
    alpha_count = sum(1 for char in body if char.isalpha())
    if alpha_count < 2:
        return False
    return all(
        char.isalnum()
        or char in SCHEMATIC_NET_LABEL_SEPARATORS
        or char in SCHEMATIC_CONFUSABLE_DIGITS
        or char in {"+", "."}
        for char in body
    )


def schematic_token_distance(left: str, right: str) -> float:
    previous = [float(index) for index in range(len(right) + 1)]
    for left_index, left_char in enumerate(left, 1):
        current = [float(left_index)]
        for right_index, right_char in enumerate(right, 1):
            substitution = previous[right_index - 1] + schematic_char_distance(
                left_char,
                right_char,
            )
            deletion = previous[right_index] + 1.0
            insertion = current[right_index - 1] + 1.0
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def schematic_char_distance(left: str, right: str) -> float:
    left = left.casefold()
    right = right.casefold()
    if left == right:
        return 0.0
    if left in {"@", "o", "0"} and right in {"@", "o", "0"}:
        return 0.05
    if left in {"i", "l", "|", "1"} and right in {"i", "l", "|", "1"}:
        return 0.05
    if left in {"s", "5"} and right in {"s", "5"}:
        return 0.05
    if left in {"u", "o", "0"} and right == "d":
        return 0.35
    if left in {"-", "–", "—"} and right in {"-", "–", "—"}:
        return 0.05
    if left in {"i", "l", "1"} and right == "d":
        return 0.45
    return 1.0


def canonical_schematic_display_token(token: str) -> str:
    token = token.casefold()
    if token in SCHEMATIC_STANDALONE_VALUE_SIGNS:
        return "+" if token == "+" else "–"
    if token == "gnd":
        return "GND"
    sign = token[:1] if token[:1] in {"+", "-"} else ""
    body = token[1:] if sign else token
    voltage_net = schematic_voltage_net_label_display(body)
    if voltage_net is not None:
        return f"{sign}{voltage_net}"
    bus_label = schematic_repaired_bus_label_body(body)
    if bus_label is not None:
        return f"{sign}{bus_label}"
    if schematic_token_is_rail_like(body) or schematic_token_is_net_label(body):
        return f"{sign}{body.upper()}"
    if body.endswith("nf") and any(ch.isdigit() for ch in body[:-2]):
        return f"{sign}{body[:-2]}nF"
    if body.endswith("meg") and any(ch.isdigit() for ch in body[:-3]):
        return f"{sign}{body[:-3]}MEG"
    if body.endswith("v") and any(ch.isdigit() for ch in body[:-1]):
        return f"{sign}{body[:-1]}V"
    if body.endswith("k") and any(ch.isdigit() for ch in body[:-1]):
        return f"{sign}{body[:-1]}k"
    if len(body) >= 2 and body[0] in {"c", "d", "j", "q", "r", "u"} and body[1].isdigit():
        return f"{sign}{body[0].upper()}{body[1:]}"
    return f"{sign}{body}"


def classify_schematic_token_type(token: str) -> str | None:
    core = schematic_token_core(token).casefold()
    if not core:
        return None
    if core in SCHEMATIC_STANDALONE_VALUE_SIGNS:
        return "opamp_label"
    if schematic_token_is_pin(core):
        return "pin"
    if schematic_token_is_rail(core):
        return "rail"
    if schematic_token_is_refdes(core):
        return "reference"
    if schematic_token_is_value(core):
        return "value"
    if core in {"a", "b", "c", "d", "v+", "v-"}:
        return "opamp_label"
    return None


def schematic_token_is_pin(token: str) -> bool:
    if not token.isdigit():
        return False
    value = int(token)
    return 1 <= value <= 99


def schematic_token_is_rail(token: str) -> bool:
    if schematic_token_is_rail_like(token):
        return True
    sign = token[:1] if token[:1] in {"+", "-", "–"} else ""
    body = token[1:] if sign else token
    if body in {"v+", "v-"}:
        return True
    if not body.endswith("v") or len(body) <= 1:
        return False
    return schematic_decimal_text_is_numeric(body[:-1])


def schematic_token_is_value(token: str) -> bool:
    sign = token[:1] if token[:1] in {"+", "-", "–"} else ""
    body = token[1:] if sign else token
    for unit in ("meg", "nf", "pf", "uf", "k", "p"):
        if body.endswith(unit) and len(body) > len(unit):
            return schematic_decimal_text_is_numeric(body[: -len(unit)])
    return False


def schematic_token_is_refdes(token: str) -> bool:
    body = token[1:] if token[:1] in {"+", "-", "–"} else token
    if len(body) < 2:
        return False
    prefix = body[0]
    if prefix not in {"c", "d", "j", "q", "r", "u"}:
        return False
    suffix = body[1:]
    return suffix.isdigit()


def schematic_decimal_text_is_numeric(text: str) -> bool:
    if not text:
        return False
    decimal_seen = False
    digit_seen = False
    for ch in text:
        if ch == "." and not decimal_seen:
            decimal_seen = True
            continue
        if not ch.isdigit():
            return False
        digit_seen = True
    return digit_seen
