# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.extraction.common import page_profile
from core_pdf.impl.exceptions import PdfParseError

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.metadata_types import MetadataRecord

PageRegionClassificationRecord = dict[str, Any]


@dataclass(frozen=True)
class ExtractionCandidate:
    route: str
    planner_score: float | None
    acceptance_score: float | None
    accepted: bool
    reason: str
    text_length: int
    confidence: float | None


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
class PageExtractionResult:
    page_number: int
    page_label: str | None
    text: str
    confidence: float
    page_class: str
    base_route: str
    supplements: tuple[str, ...]
    candidates: tuple[ExtractionCandidate, ...]
    resolved_lines: tuple[ResolvedLineRecord, ...]


@dataclass(frozen=True)
class DocumentExtractionSummary:
    page_count: int
    low_confidence_page_count: int
    ocr_assisted_page_count: int
    supplement_page_count: int
    replacement_page_count: int
    page_class_counts: dict[str, int]
    base_route_counts: dict[str, int]
    supplement_counts: dict[str, int]


@dataclass(frozen=True)
class DocumentExtractionResult:
    metadata: MetadataRecord
    text: str
    pages: tuple[PageExtractionResult, ...]
    summary: DocumentExtractionSummary

    def to_markdown(self) -> str:
        return "\f".join(page.text for page in self.pages) + "\f"


def document_summary(
    page_results: list[PageExtractionResult],
) -> DocumentExtractionSummary:
    page_class_counts = Counter(result.page_class for result in page_results)
    base_route_counts = Counter(result.base_route for result in page_results)
    supplement_counts = Counter(
        supplement for result in page_results for supplement in result.supplements
    )
    low_confidence = sum(1 for result in page_results if result.confidence < 0.5)
    ocr_assisted = sum(
        1
        for result in page_results
        if result.base_route not in {"skip", "native_fast", "native_layout"} or result.supplements
    )
    supplement_pages = sum(1 for result in page_results if result.supplements)
    replacement_pages = sum(
        1
        for result in page_results
        if result.base_route not in {"skip", "native_fast", "native_layout"}
    )
    return DocumentExtractionSummary(
        page_count=len(page_results),
        low_confidence_page_count=low_confidence,
        ocr_assisted_page_count=ocr_assisted,
        supplement_page_count=supplement_pages,
        replacement_page_count=replacement_pages,
        page_class_counts=dict(page_class_counts),
        base_route_counts=dict(base_route_counts),
        supplement_counts=dict(supplement_counts),
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
    def extract_text(self) -> str: ...
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
    *,
    page_index: int | None = None,
) -> PageExtractionResult:
    cache = page.extraction_cache
    if cache is None:
        page.extraction_cache = cache = ExtractionCache()

    text = page.extract_text()
    profile = page.get_page_profile()
    region = cache.get("page_region_classification")
    if region is None:
        from core_ocr.impl.policy import classify_page_region

        region = classify_page_region(
            text,
            page=page,
            native_runs=tuple(getattr(page, "chars", ()) or ()),
            media_box=getattr(page, "media_box", None),
            include_dominant_image=False,
        )
        cache["page_region_classification"] = region
    page_class, page_class_confidence = classify_page(profile, region, text)
    base_route = base_route_name(cache, profile, text)
    supplements = page_supplements(cache, base_route)
    candidates = tuple(
        route_candidates(cache, base_route, supplements, text, page_class_confidence)
    )
    confidence = page_confidence(
        text,
        base_route,
        page_class,
        page_class_confidence,
        cache,
        candidates,
        supplements,
    )
    result = PageExtractionResult(
        page_number=page.page_number,
        page_label=getattr(page, "label", None),
        text=text,
        confidence=confidence,
        page_class=page_class,
        base_route=base_route,
        supplements=supplements,
        candidates=candidates,
        resolved_lines=tuple(
            resolved_line_record_from_content_record(record)
            for record in page.extract_resolved_lines()
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
            page_results.append(build_page_extraction_result(page, page_index=index))
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
        text="\f".join(page.text for page in page_results) + "\f",
        pages=tuple(page_results),
        summary=document_summary(page_results),
    )


def classify_page(
    profile: page_profile.PageProfile,
    region: Any,
    text: str,
) -> tuple[str, float]:
    region_kind = string_or_none(getattr(region, "kind", None))
    region_confidence = coerce_float(getattr(region, "confidence", None)) or 0.0
    if region_kind == "schematic":
        return "schematic", region_confidence
    if region_kind == "dense_table":
        return "table", region_confidence
    if region_kind in {"form", "invoice"}:
        return "form", region_confidence
    if region_kind == "prose":
        return "native_text", max(region_confidence, 0.64)
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
    cache: ExtractionCache,
    candidates: tuple[ExtractionCandidate, ...],
    supplements: tuple[str, ...],
) -> float:
    if base_route == "skip":
        return 1.0 if not text else 0.6
    candidate_confidence = max(
        (candidate.confidence or 0.0) for candidate in candidates if candidate.accepted
    )
    text_len = len(text.strip())
    noise_penalty = text_noise_penalty(text)
    base = min(0.99, 0.30 + page_class_confidence * 0.35 + min(text_len, 800) / 2500)
    if base_route in {"native_fast", "native_layout"}:
        base += 0.12
    else:
        base += min(0.25, candidate_confidence * 0.25)
    if supplements:
        base += min(0.08, 0.02 * len(supplements))
    if page_class in {
        "schematic",
        "table",
        "figure",
        "image",
        "mixed",
    } and base_route in {
        "full_page_ocr",
        "figure_ocr",
        "embedded_image_ocr",
        "table_fusion",
        "vector_stroke",
    }:
        base += 0.08
    confidence = max(0.0, min(0.99, base - noise_penalty))
    if not text.strip():
        return min(confidence, 0.25)
    return confidence


def route_candidates(
    cache: ExtractionCache,
    base_route: str,
    supplements: tuple[str, ...],
    text: str,
    page_class_confidence: float,
) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    table_fusion = resolved_output_lines_have_source(cache, "table_fusion_text")
    if table_fusion:
        candidates.append(
            ExtractionCandidate(
                route="table_fusion",
                planner_score=page_class_confidence,
                acceptance_score=0.72,
                accepted=base_route == "table_fusion" or "table_fusion" in supplements,
                reason="table_fusion_output_lines",
                text_length=len(text),
                confidence=None,
            )
        )
    candidates.append(
        ExtractionCandidate(
            route=base_route,
            planner_score=page_class_confidence,
            acceptance_score=page_class_confidence,
            accepted=True,
            reason="selected",
            text_length=len(text),
            confidence=None,
        )
    )
    deduped: dict[str, ExtractionCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.route)
        if existing is None or candidate.accepted:
            deduped[candidate.route] = candidate
    return list(deduped.values())


def base_route_name(
    cache: ExtractionCache,
    profile: page_profile.PageProfile,
    text: str,
) -> str:
    reconciliation_input = cache_dict(cache.get("ocr_line_reconciliation_input"))
    text_source = string_or_none(reconciliation_input.get("text_source"))
    if not text.strip() and profile.can_skip_all_text:
        return "skip"
    if text_source and "figure_ocr" in text_source:
        return "figure_ocr"
    if text_source and "vector" in text_source:
        return "vector_stroke"
    if text_source and "ocr" in text_source:
        return "full_page_ocr"
    if profile.recommended_strategy == "native_text" and not cache.get("resolved_output_lines"):
        return "native_fast"
    if profile.recommended_strategy == "text_table":
        return "native_layout"
    return "native_layout"


def page_supplements(
    cache: ExtractionCache,
    base_route: str,
) -> tuple[str, ...]:
    supplements: list[str] = []
    if (
        resolved_output_lines_have_source(cache, "table_fusion_text")
        and base_route != "table_fusion"
    ):
        supplements.append("table_fusion")
    return tuple(dict.fromkeys(supplements))


def region_classification_record(region: Any) -> PageRegionClassificationRecord:
    if region is None:
        return {"kind": "unknown", "confidence": 0.0, "signals": {}}
    raw_signals = getattr(region, "signals", None)
    signals = (
        {str(key): value for key, value in raw_signals.items()}
        if isinstance(raw_signals, Mapping)
        else {}
    )
    return {
        "kind": string_or_none(getattr(region, "kind", None)) or "unknown",
        "confidence": coerce_float(getattr(region, "confidence", None)) or 0.0,
        "signals": signals,
    }


def resolved_output_lines_have_source(cache: ExtractionCache, source: str) -> bool:
    lines = cache.get("resolved_output_lines")
    if not isinstance(lines, tuple):
        return False
    return any(
        getattr(getattr(line, "observation", None), "source", None) == source for line in lines
    )


def text_noise_penalty(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.5
    punctuation = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    compact = sum(1 for ch in stripped if ch.isalnum())
    if compact == 0:
        return 0.35
    return min(0.35, punctuation / max(1, compact) * 0.5)


def cache_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except TypeError:
        return None
    except ValueError:
        return None


def string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


__all__ = (
    "build_document_extraction_result",
    "build_page_extraction_result",
)
