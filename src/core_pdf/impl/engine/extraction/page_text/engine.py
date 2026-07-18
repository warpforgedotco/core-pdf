# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from re import compile as compile_regex
from statistics import median
from typing import TYPE_CHECKING, Any, Protocol

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.extraction.common import page_profile
from core_pdf.impl.exceptions import PdfParseError

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.metadata_types import MetadataRecord


@dataclass(frozen=True)
class ResolvedLineRecord:
    text: str
    break_before: int
    kind: str
    source: str
    bbox: tuple[float, float, float, float] | None
    advance_bbox: tuple[float, float, float, float] | None
    ink_bbox: tuple[float, float, float, float] | None
    confidence: float | None
    baseline: tuple[float, float, float, float] | None
    contributing_sources: tuple[str, ...]


@dataclass(frozen=True)
class TextBlock:
    order: int
    bbox: tuple[float, float, float, float] | None
    column_index: int | None
    rotation: int
    lines: tuple[ResolvedLineRecord, ...]
    kind: str = "paragraph"


@dataclass(frozen=True)
class PageExtractionResult:
    page_number: int
    page_label: str | None
    confidence: float
    page_class: str
    base_route: str
    resolved_lines: tuple[ResolvedLineRecord, ...]
    blocks: tuple[TextBlock, ...]


@dataclass(frozen=True)
class DocumentExtractionSummary:
    page_count: int
    low_confidence_page_count: int
    page_class_counts: dict[str, int]
    base_route_counts: dict[str, int]


@dataclass(frozen=True)
class DocumentExtractionResult:
    metadata: MetadataRecord
    pages: tuple[PageExtractionResult, ...]
    summary: DocumentExtractionSummary

    def to_markdown(self) -> str:
        return "\f".join(render_page_blocks(page.blocks) for page in self.pages) + "\f"


def document_summary(
    page_results: list[PageExtractionResult],
) -> DocumentExtractionSummary:
    page_class_counts = Counter(result.page_class for result in page_results)
    base_route_counts = Counter(result.base_route for result in page_results)
    low_confidence = sum(1 for result in page_results if result.confidence < 0.5)
    return DocumentExtractionSummary(
        page_count=len(page_results),
        low_confidence_page_count=low_confidence,
        page_class_counts=dict(page_class_counts),
        base_route_counts=dict(base_route_counts),
    )


class SupportsPageExtraction(Protocol):
    page_number: int
    extraction_cache: ExtractionCache | None

    @property
    def label(self) -> str | None: ...

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    def get_page_profile(self) -> page_profile.PageProfile: ...
    def extract_resolved_lines(self) -> list[dict[str, Any]]: ...


class SupportsDocumentExtraction(Protocol):
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def get_metadata(self) -> MetadataRecord: ...
    def iter_page_dicts(self) -> Any: ...

    @property
    def page_class(self) -> Any: ...


def build_page_extraction_result(
    page: SupportsPageExtraction,
) -> PageExtractionResult:
    cache = page.extraction_cache
    if cache is None:
        page.extraction_cache = cache = ExtractionCache()

    resolved_lines = tuple(
        resolved_line_record_from_content_record(record) for record in page.extract_resolved_lines()
    )
    text = "\n".join(line.text for line in resolved_lines)
    profile = page.get_page_profile()
    page_class, page_class_confidence = classify_page(profile, text)
    base_route = base_route_name(cache, profile, text)
    confidence = page_confidence(
        text,
        base_route,
        page_class,
        page_class_confidence,
    )
    result = PageExtractionResult(
        page_number=page.page_number,
        page_label=getattr(page, "label", None),
        confidence=confidence,
        page_class=page_class,
        base_route=base_route,
        resolved_lines=resolved_lines,
        blocks=build_text_blocks(
            resolved_lines,
            rotation=getattr(page, "rotation", 0),
            page_class=page_class,
        ),
    )
    return result


def resolved_line_record_from_content_record(
    record: dict[str, Any],
) -> ResolvedLineRecord:
    return ResolvedLineRecord(
        text=str(record.get("text") or ""),
        break_before=int(record.get("break_before") or 1),
        kind=str(record.get("observation_kind") or "text-line"),
        source=str(record.get("source") or "unknown"),
        bbox=rect_or_none(record.get("bbox")),
        advance_bbox=rect_or_none(record.get("advance_bbox")),
        ink_bbox=rect_or_none(record.get("ink_bbox")),
        confidence=coerce_float(record.get("confidence")),
        baseline=segment_or_none(record.get("baseline")),
        contributing_sources=tuple(
            str(source) for source in record.get("contributing_sources", ()) if source is not None
        ),
    )


def build_document_extraction_result(
    document: SupportsDocumentExtraction,
) -> DocumentExtractionResult:
    page_dicts = list(document.iter_page_dicts())
    can_skip_bad_page = (
        len(page_dicts) > 1
        or getattr(document, "xref_was_recovered", False)
        or getattr(document, "page_tree_was_recovered", False)
    )
    page_results: list[PageExtractionResult] = []
    for index, page_dict in enumerate(page_dicts):
        page = document.page_class(document, page_dict, index + 1)
        try:
            page_results.append(build_page_extraction_result(page))
        except PdfParseError:
            if not can_skip_bad_page:
                raise
            continue
        finally:
            page.state = None
            page.graphics = None
            page.grid_lines = None
            page.text_spans = None
            page.tables = {}
    return DocumentExtractionResult(
        metadata=document.get_metadata(),
        pages=tuple(page_results),
        summary=document_summary(page_results),
    )


def classify_page(
    profile: page_profile.PageProfile,
    text: str,
) -> tuple[str, float]:
    if not text.strip() and profile.can_skip_all_text:
        return "empty", 1.0
    if profile.likely_table_page:
        return "table", 0.62
    if profile.likely_image_page and profile.likely_text_page:
        return "mixed", 0.58
    if profile.likely_image_page:
        return "image", 0.58
    if profile.likely_text_page:
        return "native_text", 0.7
    if text.strip():
        return "mixed", 0.45
    return "empty", 0.4


def page_confidence(
    text: str,
    base_route: str,
    page_class: str,
    page_class_confidence: float,
) -> float:
    if base_route == "skip":
        return 1.0 if not text else 0.6
    text_len = len(text.strip())
    noise_penalty = text_noise_penalty(text)
    base = min(0.99, 0.30 + page_class_confidence * 0.35 + min(text_len, 800) / 2500)
    base += 0.12
    if page_class in {"table", "image", "mixed"}:
        base += 0.02
    confidence = max(0.0, min(0.99, base - noise_penalty))
    if not text.strip():
        return min(confidence, 0.25)
    return confidence


def base_route_name(
    cache: ExtractionCache,
    profile: page_profile.PageProfile,
    text: str,
) -> str:
    if not text.strip() and profile.can_skip_all_text:
        return "skip"
    if profile.recommended_strategy == "native_text" and not cache.get("resolved_output_lines"):
        return "native_fast"
    if profile.recommended_strategy == "text_table":
        return "native_layout"
    return "native_layout"


def text_noise_penalty(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.5
    punctuation = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    compact = sum(1 for ch in stripped if ch.isalnum())
    if compact == 0:
        return 0.35
    return min(0.35, punctuation / max(1, compact) * 0.5)


def coerce_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except TypeError:
        return None
    except ValueError:
        return None


def rect_or_none(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    x0 = coerce_float(value[0])
    y0 = coerce_float(value[1])
    x1 = coerce_float(value[2])
    y1 = coerce_float(value[3])
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    return (x0, y0, x1, y1)


def segment_or_none(value: Any) -> tuple[float, float, float, float] | None:
    return rect_or_none(value)


def build_text_blocks(
    lines: tuple[ResolvedLineRecord, ...],
    *,
    rotation: int,
    page_class: str | None = None,
) -> tuple[TextBlock, ...]:
    if not lines:
        return ()
    blocks: list[list[ResolvedLineRecord]] = []
    block_columns: list[int | None] = []
    column_anchors: list[float] = []
    for line in lines:
        bbox = line.bbox or line.advance_bbox or line.ink_bbox
        x0 = bbox[0] if bbox is not None else None
        column_index: int | None = None
        if x0 is not None:
            assert bbox is not None
            tolerance = column_tolerance(bbox)
            for index, anchor in enumerate(column_anchors):
                if abs(anchor - x0) <= tolerance:
                    column_index = index
                    break
            if column_index is None:
                column_index = len(column_anchors)
                column_anchors.append(x0)
        if not blocks or line.break_before > 1 or block_columns[-1] != column_index:
            blocks.append([line])
            block_columns.append(column_index)
        else:
            blocks[-1].append(line)
    line_heights = tuple(line_height(line) for line in lines)
    typical_line_height = (
        median(height for height in line_heights if height is not None)
        if any(height is not None for height in line_heights)
        else None
    )
    return tuple(
        TextBlock(
            order=index,
            bbox=block_bbox(block_lines),
            column_index=block_columns[index - 1],
            rotation=rotation % 360,
            lines=tuple(block_lines),
            kind=classify_block_kind(
                block_lines,
                typical_line_height,
                allow_semantics=page_class != "table",
            ),
        )
        for index, block_lines in enumerate(blocks, 1)
    )


def block_bbox(lines: list[ResolvedLineRecord]) -> tuple[float, float, float, float] | None:
    boxes = [line.bbox or line.advance_bbox or line.ink_bbox for line in lines]
    usable = [box for box in boxes if box is not None]
    if not usable:
        return None
    return (
        min(box[0] for box in usable),
        min(box[1] for box in usable),
        max(box[2] for box in usable),
        max(box[3] for box in usable),
    )


def column_tolerance(bbox: tuple[float, float, float, float]) -> float:
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return max(24.0, height * 2.0, min(width * 0.2, 96.0))


def line_height(line: ResolvedLineRecord) -> float | None:
    bbox = line.bbox or line.advance_bbox or line.ink_bbox
    if bbox is None:
        return None
    return max(0.0, bbox[3] - bbox[1])


_LIST_ITEM_PATTERN = compile_regex(r"^\s*(?:[-*•▪◦]|\d+[.)]|[A-Za-z][.)])\s+")


def classify_block_kind(
    lines: list[ResolvedLineRecord],
    typical_line_height: float | None,
    *,
    allow_semantics: bool,
) -> str:
    if not lines or not allow_semantics:
        return "paragraph"
    first_text = lines[0].text.strip()
    if _LIST_ITEM_PATTERN.match(first_text):
        return "list"
    if len(lines) != 1 or not is_heading_candidate(first_text):
        return "paragraph"
    height = line_height(lines[0])
    is_large = (
        height is not None
        and typical_line_height is not None
        and typical_line_height > 0
        and height >= typical_line_height * 1.35
    )
    letters = [character for character in first_text if character.isalpha()]
    is_uppercase = bool(letters) and all(character.isupper() for character in letters)
    if is_uppercase or is_large:
        return "heading"
    return "paragraph"


def is_heading_candidate(text: str) -> bool:
    words = text.split()
    if not 1 <= len(words) <= 16 or len(text) > 140:
        return False
    return text[-1:] not in ".,:;!?"


def render_page_blocks(blocks: tuple[TextBlock, ...]) -> str:
    rendered_blocks: list[str] = []
    for block in blocks:
        rendered_lines: list[str] = []
        for line in block.lines:
            if rendered_lines:
                rendered_lines.append("\n" * max(1, line.break_before))
            rendered_lines.append(line.text)
        rendered_blocks.append("".join(rendered_lines))
    return "\n\n".join(rendered_blocks)


__all__ = (
    "DocumentExtractionResult",
    "DocumentExtractionSummary",
    "PageExtractionResult",
    "ResolvedLineRecord",
    "TextBlock",
    "build_document_extraction_result",
    "build_page_extraction_result",
)
