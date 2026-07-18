# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from core_layout.impl.layout.geometry_quality import LayoutGeometrySummary
from core_ocr.impl import geometry as ocr_geometry
from core_ocr.impl import layout as ocr_layout
from core_ocr.impl import page_analysis as ocr_page_analysis
from core_ocr.impl import text_analysis as ocr_text_analysis
from core_ocr.impl.text_analysis import (
    extracted_text_token_count,
    normalized_text_tokens,
    numeric_token_ratio,
    sparse_text_looks_noisy,
    text_has_many_digit_lines,
    text_ocr_quality_score,
)

from core_pdf.impl.engine.extraction.common.ordering import LayoutAnalyzer
from core_pdf.impl.engine.extraction.ocr import (
    schematic as ocr_schematic,
)
from core_pdf.impl.engine.extraction.ocr import (
    table_regions as ocr_table_regions,
)

if TYPE_CHECKING:
    from core_layout.impl.layout.models import LayoutLine, TextRun
    from core_ocr.impl.candidates import OcrCandidate, OcrPageTextResult


@dataclass(frozen=True, slots=True)
class PageTextGeometryProfile:
    page_width: float
    page_height: float
    native_run_count: int
    native_line_count: int
    wide_line_ratio: float
    short_line_ratio: float
    centered_line_ratio: float
    numeric_line_ratio: float
    left_anchor_count: int
    right_anchor_count: int
    estimated_column_count: int
    native_aligned_column_count: int
    candidate_aligned_column_count: int
    candidate_table_signals: int
    candidate_schematic_signals: int
    drawing_line_count: int
    horizontal_rule_count: int
    vertical_rule_count: int
    dominant_image: bool
    occupied_area_ratio: float


@dataclass(frozen=True, slots=True)
class OcrCandidateGeometryProfile:
    line_count: int
    word_count: int
    aligned_column_count: int
    occupied_area_ratio: float
    wide_line_ratio: float
    short_line_ratio: float
    line_coverage_score: float
    confidence: int
    text_quality: float
    artifact_score: float
    gibberish_score: float


def should_replace_text_with_ocr(
    page: Any,
    text: str,
    ocr_result: OcrPageTextResult,
    *,
    native_runs: Iterable[TextRun] = (),
    native_geometry: LayoutGeometrySummary | None = None,
    vector_text: str = "",
) -> bool:
    ocr_text = ocr_result.text
    ocr_tokens = extracted_text_token_count(ocr_text)
    if ocr_tokens == 0:
        return False
    text_tokens = extracted_text_token_count(text)
    if text_tokens == 0:
        return True
    candidate = ocr_result.candidate
    if candidate is None:
        return False
    if should_preserve_dense_numeric_native_text_against_ocr(
        text,
        ocr_text,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
    ):
        return False
    native_profile = page_text_geometry_profile(
        text,
        vector_text=vector_text,
        candidates=tuple(ocr_result.candidates),
        page=page,
        native_runs=tuple(native_runs),
        media_box=getattr(page, "media_box", None),
    )
    ocr_profile = ocr_candidate_geometry_profile(page, candidate)
    if should_preserve_native_text_against_rendered_page_ocr(
        page,
        text,
        ocr_text,
        candidate=candidate,
        native_profile=native_profile,
        ocr_profile=ocr_profile,
        native_geometry=native_geometry,
    ):
        return False
    if sparse_drawing_schematic_should_yield_to_ocr(
        text,
        ocr_text,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
        native_profile=native_profile,
        ocr_profile=ocr_profile,
    ):
        return True
    if text_tokens <= 24:
        if tiny_native_text_should_yield_to_ocr(
            text,
            ocr_text,
            text_tokens=text_tokens,
            ocr_tokens=ocr_tokens,
            native_profile=native_profile,
            ocr_profile=ocr_profile,
        ):
            return True
        return (
            ocr_profile.confidence >= 85
            and ocr_profile.text_quality + 0.08 < text_ocr_quality_score(text)
            and ocr_profile.line_count >= native_profile.native_line_count
            and ocr_profile.occupied_area_ratio >= native_profile.occupied_area_ratio * 0.9
        )
    if ocr_profile.confidence < 70 and text_tokens <= 120 and not native_profile.dominant_image:
        return False
    if geometry_supports_ocr_replacement(
        text,
        ocr_text,
        native_profile=native_profile,
        ocr_profile=ocr_profile,
        native_geometry=native_geometry,
    ):
        return True
    if native_geometry_supports_ocr_replacement(
        text,
        ocr_text,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
        native_geometry=native_geometry,
    ) and ocr_profile.line_count >= max(3, int(native_profile.native_line_count * 0.6)):
        return True
    if ocr_profile.line_count == 0:
        return False
    return (
        ocr_profile.text_quality <= text_ocr_quality_score(text) + 0.02
        and ocr_profile.line_count >= max(3, int(native_profile.native_line_count * 0.75))
        and ocr_profile.occupied_area_ratio >= native_profile.occupied_area_ratio * 0.75
        and ocr_text_is_cleaner_candidate(
            text,
            ocr_text,
            text_tokens=text_tokens,
            ocr_tokens=ocr_tokens,
        )
    )


def should_preserve_native_text_against_rendered_page_ocr(
    page: Any,
    text: str,
    ocr_text: str,
    *,
    candidate: OcrCandidate,
    native_profile: PageTextGeometryProfile,
    ocr_profile: OcrCandidateGeometryProfile,
    native_geometry: LayoutGeometrySummary | None,
) -> bool:
    candidate_name = str(getattr(candidate, "name", ""))
    if not candidate_name.startswith("rendered_page_"):
        return False
    try:
        profile = page.get_page_profile()
    except Exception:
        return False
    recommended_strategy = getattr(profile, "recommended_strategy", None)
    if recommended_strategy not in {"native_text", "text_table"}:
        return False
    text_tokens = extracted_text_token_count(text)
    ocr_tokens = extracted_text_token_count(ocr_text)
    if text_tokens < 120 or ocr_tokens < 120:
        return False
    text_quality = text_ocr_quality_score(text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    if native_source_should_win_low_quality_ocr(
        recommended_strategy,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
        ocr_quality=ocr_quality,
    ):
        return True
    confidence = candidate.result.confidence or 0
    if confidence >= 35:
        return False
    text_artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    ocr_artifact = ocr_text_analysis.scanned_ocr_artifact_score(ocr_text)
    if recommended_strategy == "native_text":
        if text_tokens < 140:
            return False
        if native_profile.native_line_count < 8:
            return False
        if native_profile.wide_line_ratio < 0.45:
            return False
        if ocr_profile.line_count < max(8, int(native_profile.native_line_count * 0.8)):
            return False
        if ocr_profile.occupied_area_ratio < native_profile.occupied_area_ratio * 0.75:
            return False
        if (
            native_geometry is not None
            and native_geometry.suspicion_score >= 12.0
            and not rendered_page_ocr_breaks_dense_native_layout(
                native_profile,
                ocr_profile,
                native_geometry=native_geometry,
            )
        ):
            return False
        if (
            native_profile.native_aligned_column_count >= 6
            and rendered_page_ocr_breaks_dense_native_layout(
                native_profile,
                ocr_profile,
                native_geometry=native_geometry,
            )
        ):
            return ocr_quality + 0.02 >= text_quality
        return ocr_quality + 0.02 >= text_quality and ocr_artifact + 0.005 >= text_artifact
    if recommended_strategy != "text_table":
        return False
    if text_tokens < 250 or native_profile.native_aligned_column_count < 6:
        return False
    if text_tokens < int(ocr_tokens * 1.15):
        return False
    if native_profile.occupied_area_ratio < 0.30:
        return False
    if not rendered_page_ocr_breaks_dense_native_layout(
        native_profile,
        ocr_profile,
        native_geometry=native_geometry,
    ):
        return False
    if text_artifact > ocr_artifact + 0.002:
        return False
    return text_quality <= ocr_quality + 0.015


def native_source_should_win_low_quality_ocr(
    recommended_strategy: str | None,
    *,
    text_tokens: int,
    ocr_tokens: int,
    ocr_quality: float,
) -> bool:
    """Prefer a substantial native layer over low-quality page OCR.

    This is intentionally source- and coverage-aware: it applies only to
    pages classified as native/table text, and only when native extraction
    retains at least half of the OCR token coverage.
    """
    return (
        recommended_strategy in {"native_text", "text_table"}
        and text_tokens >= 120
        and ocr_tokens >= 120
        and text_tokens >= int(ocr_tokens * 0.50)
        and ocr_quality <= 0.18
    )


def rendered_page_ocr_breaks_dense_native_layout(
    native_profile: PageTextGeometryProfile,
    ocr_profile: OcrCandidateGeometryProfile,
    *,
    native_geometry: LayoutGeometrySummary | None,
) -> bool:
    if (
        native_profile.native_aligned_column_count >= 6
        and text_table_like_layout_diverges_in_rendered_ocr(
            native_profile,
            ocr_profile,
        )
    ):
        return True
    return (native_geometry is None or native_geometry.suspicion_score < 24.0) and (
        native_profile.native_aligned_column_count >= 6
        and native_profile.wide_line_ratio >= 0.45
        and ocr_profile.line_count >= max(60, int(native_profile.native_line_count * 1.8))
        and ocr_profile.wide_line_ratio <= native_profile.wide_line_ratio * 0.35
        and ocr_profile.occupied_area_ratio + 0.04 < native_profile.occupied_area_ratio
    )


def text_table_like_layout_diverges_in_rendered_ocr(
    native_profile: PageTextGeometryProfile,
    ocr_profile: OcrCandidateGeometryProfile,
) -> bool:
    return (
        ocr_profile.wide_line_ratio >= max(0.45, native_profile.wide_line_ratio * 1.8)
        and ocr_profile.aligned_column_count >= native_profile.native_aligned_column_count * 2
        and ocr_profile.occupied_area_ratio >= native_profile.occupied_area_ratio * 1.15
    )


def sparse_drawing_schematic_should_yield_to_ocr(
    text: str,
    ocr_text: str,
    *,
    text_tokens: int,
    ocr_tokens: int,
    native_profile: PageTextGeometryProfile,
    ocr_profile: OcrCandidateGeometryProfile,
) -> bool:
    """Recover path-encoded labels from otherwise near-empty schematics."""
    if text_tokens == 0 or text_tokens > 12:
        return False
    if ocr_tokens < max(50, text_tokens * 10):
        return False
    if native_profile.drawing_line_count < 200:
        return False
    if native_profile.candidate_schematic_signals < 30:
        return False
    if ocr_profile.confidence < 55:
        return False
    ocr_quality = text_ocr_quality_score(ocr_text)
    if ocr_quality > 0.38:
        return False
    return ocr_quality + 0.10 < text_ocr_quality_score(text)


def tiny_native_text_should_yield_to_ocr(
    text: str,
    ocr_text: str,
    *,
    text_tokens: int,
    ocr_tokens: int,
    native_profile: PageTextGeometryProfile,
    ocr_profile: OcrCandidateGeometryProfile,
) -> bool:
    if text_tokens == 0 or text_tokens > 24 or ocr_tokens < max(24, text_tokens * 6):
        return False
    if ocr_profile.confidence < 50 or ocr_profile.line_count < max(
        3, native_profile.native_line_count + 1
    ):
        return False
    if ocr_profile.occupied_area_ratio < max(0.02, native_profile.occupied_area_ratio * 1.8):
        return False
    if ocr_profile.text_quality > text_ocr_quality_score(text) + 0.08:
        return False
    if native_profile.wide_line_ratio >= 0.45 and native_profile.native_line_count >= 3:
        return False
    return (
        native_profile.dominant_image
        or native_profile.horizontal_rule_count >= 4
        or native_profile.vertical_rule_count >= 3
        or native_profile.drawing_line_count >= 20
        or (native_profile.native_line_count <= 1 and native_profile.occupied_area_ratio <= 0.03)
    )


def geometry_supports_ocr_replacement(
    text: str,
    ocr_text: str,
    *,
    native_profile: PageTextGeometryProfile,
    ocr_profile: OcrCandidateGeometryProfile,
    native_geometry: LayoutGeometrySummary | None,
) -> bool:
    score = 0.0
    text_quality = text_ocr_quality_score(text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    if ocr_profile.line_count >= max(4, int(native_profile.native_line_count * 0.9)):
        score += 2.0
    elif ocr_profile.line_count >= max(3, int(native_profile.native_line_count * 0.7)):
        score += 1.0
    else:
        score -= 2.0
    if ocr_profile.occupied_area_ratio >= native_profile.occupied_area_ratio * 0.9:
        score += 2.0
    elif ocr_profile.occupied_area_ratio >= native_profile.occupied_area_ratio * 0.7:
        score += 1.0
    else:
        score -= 1.5
    if native_profile.native_aligned_column_count >= 3 and ocr_profile.aligned_column_count >= max(
        2, native_profile.native_aligned_column_count - 1
    ):
        score += 1.5
    if (
        native_profile.wide_line_ratio >= 0.4
        and native_profile.estimated_column_count <= 1
        and ocr_profile.wide_line_ratio < native_profile.wide_line_ratio * 0.5
    ):
        score -= 2.0
    if ocr_quality <= text_quality + 0.02:
        score += 1.0
    elif ocr_quality > text_quality + 0.08:
        score -= 1.5
    if ocr_profile.line_coverage_score >= 0.92:
        score += 1.0
    elif ocr_profile.line_coverage_score < 0.82:
        score -= 1.0
    if ocr_profile.artifact_score > 0.04 or ocr_profile.gibberish_score > 0.04:
        score -= 2.0
    if native_geometry is not None and native_geometry.suspicion_score >= 5.0:
        score += min(2.0, native_geometry.suspicion_score * 0.12)
    if native_profile.dominant_image and ocr_profile.occupied_area_ratio >= 0.18:
        score += 1.0
    return score >= 3.0


def native_geometry_supports_ocr_replacement(
    text: str,
    ocr_text: str,
    *,
    text_tokens: int,
    ocr_tokens: int,
    native_geometry: LayoutGeometrySummary | None,
) -> bool:
    if native_geometry is None or native_geometry.suspicion_score < 5.0:
        return False
    text_quality = text_ocr_quality_score(text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    token_ratio = ocr_tokens / max(1, text_tokens)
    if (
        native_geometry.error_count > 0
        and token_ratio >= 0.80
        and ocr_quality <= text_quality + 0.02
    ):
        return True
    if (
        native_geometry.has_repairable_issues
        and token_ratio >= 0.95
        and ocr_quality <= text_quality + 0.04
    ):
        return True
    return (
        native_geometry.suspicion_score >= 9.0
        and token_ratio >= 1.10
        and ocr_quality <= text_quality + 0.08
    )


def should_preserve_dense_numeric_native_text_against_ocr(
    text: str,
    ocr_text: str,
    *,
    text_tokens: int | None = None,
    ocr_tokens: int | None = None,
) -> bool:
    if text_tokens is None:
        text_tokens = extracted_text_token_count(text)
    if ocr_tokens is None:
        ocr_tokens = extracted_text_token_count(ocr_text)
    if text_tokens < 250:
        return False
    native_numeric_ratio = numeric_token_ratio(text)
    if native_numeric_ratio < 0.30 and not text_has_many_digit_lines(text):
        return False
    native_artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    ocr_artifact = ocr_text_analysis.scanned_ocr_artifact_score(ocr_text)
    if ocr_tokens < int(text_tokens * 1.15) and ocr_artifact + 0.05 >= native_artifact:
        return True
    if ocr_tokens < max(80, int(text_tokens * 0.82)):
        return True
    ocr_numeric_ratio = numeric_token_ratio(ocr_text)
    if ocr_numeric_ratio + 0.08 < native_numeric_ratio:
        return True
    native_digit_tokens = sum(
        1 for token in normalized_text_tokens(text) if any(ch.isdigit() for ch in token)
    )
    ocr_digit_tokens = sum(
        1 for token in normalized_text_tokens(ocr_text) if any(ch.isdigit() for ch in token)
    )
    return native_digit_tokens >= 100 and ocr_digit_tokens < int(native_digit_tokens * 0.78)


def should_replace_dominant_image_native_text_with_ocr(
    page: Any,
    text: str,
    ocr_text: str,
) -> bool:
    text_tokens = extracted_text_token_count(text)
    ocr_tokens = extracted_text_token_count(ocr_text)
    if should_preserve_dense_numeric_native_text_against_ocr(
        text,
        ocr_text,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
    ):
        return False
    try:
        dominant_image = ocr_page_analysis.has_dominant_page_image(page)
    except Exception:
        return False
    if text_tokens <= 20:
        if ocr_tokens < max(80, text_tokens * 6):
            return False
        if text_ocr_quality_score(ocr_text) > 0.55:
            return False
        if ocr_text_analysis.scanned_ocr_artifact_score(ocr_text) > 0.34:
            return False
        return dominant_image
    if text_tokens < 100 or ocr_tokens < 140:
        return False
    token_ratio = ocr_tokens / max(1, text_tokens)
    if token_ratio < 1.35:
        return False
    ocr_artifact = ocr_text_analysis.scanned_ocr_artifact_score(ocr_text)
    if ocr_artifact > 0.05:
        return False
    return dominant_image


def fragmented_invisible_text_layer_should_yield_to_ocr(
    text: str,
    ocr_text: str,
    confidence: int | None,
    *,
    native_layer_is_fragmented: bool,
) -> bool:
    """Replace a fragmented hidden OCR layer with a clean full-page OCR pass."""
    if not native_layer_is_fragmented or confidence is None or confidence < 80:
        return False
    text_tokens = extracted_text_token_count(text)
    ocr_tokens = extracted_text_token_count(ocr_text)
    if ocr_tokens < 60 or ocr_tokens < int(text_tokens * 0.75):
        return False
    ocr_quality = text_ocr_quality_score(ocr_text)
    if ocr_quality > max(0.16, text_ocr_quality_score(text) + 0.04):
        return False
    return ocr_text_analysis.scanned_ocr_artifact_score(ocr_text) <= 0.05


def should_preserve_substantial_text_table_native_text(
    page: Any,
    text: str,
    ocr_text: str,
) -> bool:
    """Keep a clean native table when OCR adds no material text coverage."""
    try:
        profile = page.get_page_profile()
    except Exception:
        return False
    if getattr(profile, "recommended_strategy", None) != "text_table":
        return False
    text_tokens = extracted_text_token_count(text)
    if text_tokens < 250:
        return False
    ocr_tokens = extracted_text_token_count(ocr_text)
    if ocr_tokens >= int(text_tokens * 1.20):
        return False
    if text_ocr_quality_score(text) > 0.32:
        return False
    return ocr_text_analysis.scanned_ocr_artifact_score(text) <= 0.10


def should_replace_symbol_encoded_text_with_ocr(
    page: Any,
    text: str,
    ocr_text: str,
) -> bool:
    if not ocr_page_analysis.symbol_encoded_native_text_layer_looks_weak(page, text):
        return False
    ocr_tokens = extracted_text_token_count(ocr_text)
    text_tokens = extracted_text_token_count(text)
    if should_preserve_dense_numeric_native_text_against_ocr(
        text,
        ocr_text,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
    ):
        return False
    if ocr_tokens < 120 or text_tokens < 240:
        return False
    token_ratio = ocr_tokens / max(1, text_tokens)
    if token_ratio < 0.35:
        return False
    text_quality = text_ocr_quality_score(text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    return ocr_quality + 0.06 < text_quality


def should_replace_noisy_native_text_with_compact_ocr(
    page: Any,
    text: str,
    ocr_text: str,
) -> bool:
    text_tokens = extracted_text_token_count(text)
    ocr_tokens = extracted_text_token_count(ocr_text)
    if should_preserve_dense_numeric_native_text_against_ocr(
        text,
        ocr_text,
        text_tokens=text_tokens,
        ocr_tokens=ocr_tokens,
    ):
        return False
    if text_tokens < 180 or ocr_tokens < 100:
        return False
    token_ratio = ocr_tokens / max(1, text_tokens)
    if not (0.35 <= token_ratio <= 0.75):
        return False
    text_quality = text_ocr_quality_score(text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    if text_quality < 0.20:
        return False
    if ocr_quality > 0.16 or ocr_quality + 0.07 >= text_quality:
        return False
    try:
        return ocr_page_analysis.has_dominant_page_image(page)
    except Exception:
        return False


def classify_page_region(
    text: str,
    *,
    vector_text: str = "",
    candidates: Iterable[OcrCandidate] = (),
    page: Any = None,
    native_runs: Iterable[TextRun] = (),
    media_box: tuple[float, float, float, float] | None = None,
    include_dominant_image: bool = True,
) -> ocr_schematic.PageRegionClassification:
    candidate_tuple = tuple(candidates)
    geometry = page_text_geometry_profile(
        text,
        vector_text=vector_text,
        candidates=candidate_tuple,
        page=page,
        native_runs=tuple(native_runs),
        media_box=media_box,
        include_dominant_image=include_dominant_image,
    )
    text_tokens = extracted_text_token_count(text)
    vector_tokens = extracted_text_token_count(vector_text)
    support_targets, _ = ocr_schematic.schematic_support_repair_tokens(vector_text)
    candidate_schematic_signals = geometry.candidate_schematic_signals
    candidate_table_signals = geometry.candidate_table_signals
    schematic_vector_signal = ocr_text_analysis.vector_text_supports_schematic_tiled_ocr(
        vector_text
    )
    dense_table_signal = (
        candidate_table_signals >= 8
        or geometry.native_aligned_column_count >= 4
        or (geometry.native_aligned_column_count >= 3 and geometry.numeric_line_ratio >= 0.22)
        or (geometry.candidate_aligned_column_count >= 3 and geometry.numeric_line_ratio >= 0.18)
        or (
            geometry.horizontal_rule_count >= 6
            and geometry.vertical_rule_count >= 2
            and geometry.numeric_line_ratio >= 0.18
        )
    )
    formula_signal = bool(text) and (
        ocr_text_analysis.ocr_text_has_dense_formula_notation(text) or formula_heavy_ocr_text(text)
    )
    form_signal = (
        geometry.horizontal_rule_count >= 8
        or (
            geometry.horizontal_rule_count >= 4
            and geometry.short_line_ratio >= 0.45
            and geometry.wide_line_ratio <= 0.24
            and geometry.right_anchor_count >= 2
        )
        or (
            geometry.vertical_rule_count >= 3
            and geometry.short_line_ratio >= 0.40
            and geometry.left_anchor_count >= 2
        )
    )
    invoice_signal = invoice_text_signal_count(text)
    invoice_geometry_signal = (
        form_signal and geometry.right_anchor_count >= 2 and geometry.numeric_line_ratio >= 0.12
    ) or (
        geometry.right_anchor_count >= 3
        and geometry.short_line_ratio >= 0.35
        and geometry.numeric_line_ratio >= 0.16
    )
    prose_signal = (
        geometry.native_line_count >= 8
        and geometry.wide_line_ratio >= 0.42
        and geometry.short_line_ratio <= 0.40
        and geometry.left_anchor_count <= 2
        and geometry.native_aligned_column_count <= 2
        and geometry.numeric_line_ratio < 0.18
        and geometry.horizontal_rule_count < 6
    )
    noisy_schematic_layout_only_signal = (
        not schematic_vector_signal
        and len(support_targets) < ocr_schematic.OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS
        and candidate_schematic_signals >= 12
        and dense_table_signal
        and geometry.dominant_image
    )
    signals: dict[str, Any] = {
        "text_tokens": text_tokens,
        "vector_tokens": vector_tokens,
        "schematic_support_targets": len(support_targets),
        "candidate_schematic_signals": candidate_schematic_signals,
        "candidate_table_signals": candidate_table_signals,
        "page_width": round(geometry.page_width, 2),
        "page_height": round(geometry.page_height, 2),
        "native_run_count": geometry.native_run_count,
        "native_line_count": geometry.native_line_count,
        "wide_line_ratio": round(geometry.wide_line_ratio, 4),
        "short_line_ratio": round(geometry.short_line_ratio, 4),
        "centered_line_ratio": round(geometry.centered_line_ratio, 4),
        "numeric_line_ratio": round(geometry.numeric_line_ratio, 4),
        "left_anchor_count": geometry.left_anchor_count,
        "right_anchor_count": geometry.right_anchor_count,
        "estimated_column_count": geometry.estimated_column_count,
        "native_aligned_column_count": geometry.native_aligned_column_count,
        "candidate_aligned_column_count": geometry.candidate_aligned_column_count,
        "drawing_line_count": geometry.drawing_line_count,
        "horizontal_rule_count": geometry.horizontal_rule_count,
        "vertical_rule_count": geometry.vertical_rule_count,
        "dominant_image": geometry.dominant_image,
        "schematic_vector_signal": schematic_vector_signal,
        "dense_table_signal": dense_table_signal,
        "formula_signal": formula_signal,
        "form_signal": form_signal,
        "invoice_signal": invoice_signal,
        "invoice_geometry_signal": invoice_geometry_signal,
        "prose_signal": prose_signal,
        "noisy_schematic_layout_only_signal": noisy_schematic_layout_only_signal,
        "ocr_candidates": len(candidate_tuple),
    }
    if (
        schematic_vector_signal
        or len(support_targets) >= ocr_schematic.OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS
        or (candidate_schematic_signals >= 12 and not noisy_schematic_layout_only_signal)
    ):
        confidence = 0.62
        if schematic_vector_signal:
            confidence += 0.2
        if len(support_targets) >= ocr_schematic.OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS:
            confidence += 0.1
        if candidate_schematic_signals >= 12:
            confidence += 0.08
        return ocr_schematic.PageRegionClassification(
            "schematic",
            min(confidence, 0.99),
            signals,
        )
    if formula_signal:
        return ocr_schematic.PageRegionClassification("patent_formula", 0.78, signals)
    if dense_table_signal:
        confidence = 0.72
        if geometry.native_aligned_column_count >= 4:
            confidence += 0.08
        if geometry.candidate_aligned_column_count >= 4:
            confidence += 0.06
        return ocr_schematic.PageRegionClassification(
            "dense_table",
            min(confidence, 0.99),
            signals,
        )
    if invoice_geometry_signal and invoice_signal >= 1:
        confidence = min(0.9, 0.62 + invoice_signal * 0.05)
        return ocr_schematic.PageRegionClassification("invoice", confidence, signals)
    if form_signal:
        confidence = 0.68
        if geometry.horizontal_rule_count >= 8:
            confidence += 0.08
        if geometry.right_anchor_count >= 2:
            confidence += 0.05
        return ocr_schematic.PageRegionClassification(
            "form",
            min(confidence, 0.92),
            signals,
        )
    if invoice_signal >= 3:
        confidence = min(0.86, 0.56 + invoice_signal * 0.05)
        return ocr_schematic.PageRegionClassification("invoice", confidence, signals)
    if prose_signal and not sparse_text_looks_noisy(text):
        return ocr_schematic.PageRegionClassification("prose", 0.64, signals)
    return ocr_schematic.PageRegionClassification("unknown", 0.0, signals)


def page_text_geometry_profile(
    text: str,
    *,
    vector_text: str = "",
    candidates: tuple[OcrCandidate, ...] = (),
    page: Any = None,
    native_runs: tuple[TextRun, ...] = (),
    media_box: tuple[float, float, float, float] | None = None,
    include_dominant_image: bool = True,
) -> PageTextGeometryProfile:
    del text, vector_text
    page_width, page_height = page_dimensions(page, media_box)
    lines = native_layout_lines(page, native_runs)
    left_anchor_count = anchor_cluster_count(
        [line.x0 for line in lines],
        bucket_width=max(6.0, page_width * 0.035),
    )
    right_anchor_count = anchor_cluster_count(
        [line.x1 for line in lines],
        bucket_width=max(6.0, page_width * 0.035),
    )
    drawing_lines = page_drawing_lines(page)
    horizontal_rule_count, vertical_rule_count = drawing_rule_counts(drawing_lines)
    line_widths = [max(0.0, line.x1 - line.x0) for line in lines if line.x1 > line.x0]
    wide_line_ratio = ratio(
        sum(1 for width in line_widths if page_width > 0 and width / page_width >= 0.55),
        len(line_widths),
    )
    short_line_ratio = ratio(
        sum(1 for width in line_widths if page_width > 0 and width / page_width <= 0.35),
        len(line_widths),
    )
    centered_line_ratio = ratio(
        sum(
            1
            for line in lines
            if page_width > 0
            and abs(((line.x0 + line.x1) * 0.5) - page_width * 0.5) <= page_width * 0.12
            and (line.x1 - line.x0) <= page_width * 0.70
        ),
        len(lines),
    )
    return PageTextGeometryProfile(
        page_width=page_width,
        page_height=page_height,
        native_run_count=len(native_runs) if native_runs else sum(len(line.runs) for line in lines),
        native_line_count=len(lines),
        wide_line_ratio=wide_line_ratio,
        short_line_ratio=short_line_ratio,
        centered_line_ratio=centered_line_ratio,
        numeric_line_ratio=ratio(
            sum(1 for line in lines if layout_line_has_numeric_signal(line)),
            len(lines),
        ),
        left_anchor_count=left_anchor_count,
        right_anchor_count=right_anchor_count,
        estimated_column_count=estimated_column_count(lines, page_width),
        native_aligned_column_count=native_aligned_column_count(lines),
        candidate_aligned_column_count=candidate_layout_aligned_column_count(candidates),
        candidate_table_signals=table_candidate_layout_signal_count(candidates),
        candidate_schematic_signals=ocr_schematic.schematic_candidate_layout_signal_count(
            candidates
        ),
        drawing_line_count=len(drawing_lines),
        horizontal_rule_count=horizontal_rule_count,
        vertical_rule_count=vertical_rule_count,
        dominant_image=page_has_dominant_image(page) if include_dominant_image else False,
        occupied_area_ratio=occupied_area_ratio_for_line_boxes(
            ((line.x0, line.y0, line.x1, line.y1) for line in lines),
            page_width=page_width,
            page_height=page_height,
        ),
    )


def ocr_candidate_geometry_profile(
    page: Any,
    candidate: OcrCandidate,
) -> OcrCandidateGeometryProfile:
    lines = ocr_geometry.ocr_candidate_textline_geometry_lines(page, candidate)
    page_width, page_height = page_dimensions(page, getattr(page, "media_box", None))
    line_boxes = [line.observation.bbox for line in lines if line.observation.bbox is not None]
    line_widths = [max(0.0, float(box[2]) - float(box[0])) for box in line_boxes]
    text = candidate.result.text
    return OcrCandidateGeometryProfile(
        line_count=len(lines),
        word_count=len(candidate.result.word_rows),
        aligned_column_count=candidate_layout_aligned_column_count((candidate,)),
        occupied_area_ratio=occupied_area_ratio_for_line_boxes(
            line_boxes,
            page_width=page_width,
            page_height=page_height,
        ),
        wide_line_ratio=ratio(
            sum(1 for width in line_widths if page_width > 0 and width / page_width >= 0.55),
            len(line_widths),
        ),
        short_line_ratio=ratio(
            sum(1 for width in line_widths if page_width > 0 and width / page_width <= 0.35),
            len(line_widths),
        ),
        line_coverage_score=ocr_text_analysis.rendered_ocr_line_coverage_score(
            text,
            line_rows=len(candidate.result.line_rows),
            word_rows=len(candidate.result.word_rows),
        ),
        confidence=int(candidate.result.confidence or 0),
        text_quality=text_ocr_quality_score(text),
        artifact_score=ocr_text_analysis.scanned_ocr_artifact_score(text),
        gibberish_score=ocr_text_analysis.alphabetic_gibberish_score(text),
    )


def ocr_candidate_is_complete_for_general_scan(
    page: Any,
    candidate: OcrCandidate,
    *,
    vector_text: str = "",
    strict: bool = False,
) -> bool:
    profile = ocr_candidate_geometry_profile(page, candidate)
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if tokens < (240 if strict else 180):
        return False
    if profile.line_count < (18 if strict else 12):
        return False
    if profile.confidence < (92 if strict else 84):
        return False
    if profile.text_quality > (0.08 if strict else 0.10):
        return False
    if profile.artifact_score > (0.03 if strict else 0.04):
        return False
    if profile.gibberish_score > (0.02 if strict else 0.04):
        return False
    if numeric_token_ratio(text) >= (0.18 if strict else 0.20):
        return False
    if text_has_many_digit_lines(text):
        return False
    if ocr_text_analysis.ocr_text_has_dense_formula_notation(text):
        return False
    if profile.line_coverage_score < (0.93 if strict else 0.90):
        return False
    if profile.occupied_area_ratio < (0.16 if strict else 0.12):
        return False
    classification = classify_page_region(
        text,
        vector_text=vector_text,
        candidates=(candidate,),
        page=page,
        media_box=getattr(page, "media_box", None),
    )
    return classification.kind not in {"dense_table", "form", "invoice", "schematic"}


def page_dimensions(
    page: Any,
    media_box: tuple[float, float, float, float] | None,
) -> tuple[float, float]:
    box = media_box
    if box is None and page is not None:
        box = getattr(page, "media_box", None)
    if box is None:
        return 0.0, 0.0
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def native_layout_lines(
    page: Any,
    native_runs: tuple[TextRun, ...],
) -> list[LayoutLine]:
    if native_runs:
        return LayoutAnalyzer.cluster_into_lines(
            [run for run in native_runs if getattr(run, "visible", True) and str(run.text).strip()]
        )
    if page is None:
        return []
    get_text_lines = getattr(page, "get_text_lines", None)
    if not callable(get_text_lines):
        return []
    try:
        return [
            line
            for line in get_text_lines()
            if any(
                getattr(run, "visible", True) and str(run.text).strip()
                for run in getattr(line, "runs", ())
            )
        ]
    except Exception:
        return []


def anchor_cluster_count(values: list[float], *, bucket_width: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    groups: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - groups[-1][-1]) <= bucket_width:
            groups[-1].append(value)
            continue
        groups.append([value])
    min_group_size = max(2, min(4, max(1, len(values) // 6)))
    return sum(1 for group in groups if len(group) >= min_group_size)


def estimated_column_count(lines: list[LayoutLine], page_width: float) -> int:
    if not lines:
        return 0
    if page_width <= 0 or len(lines) < 4:
        return 1
    centers = [((line.x0 + line.x1) * 0.5) for line in lines if line.x1 > line.x0]
    if not centers:
        return 1
    groups = anchor_cluster_count(centers, bucket_width=max(24.0, page_width * 0.18))
    if groups >= 2:
        narrow_ratio = ratio(
            sum(1 for line in lines if (line.x1 - line.x0) <= page_width * 0.42),
            len(lines),
        )
        if narrow_ratio >= 0.35:
            return 2
    return 1


def native_aligned_column_count(lines: list[LayoutLine]) -> int:
    word_lines = []
    for line in lines:
        if not text_may_have_native_word_table_signal(line.text()):
            continue
        text, words = line.cached_text_and_words()
        if not text.strip() or len(words) < 3 or not native_word_line_has_table_signal(words):
            continue
        word_lines.append(words)
    if len(word_lines) < 4:
        return 0
    heights = sorted(
        max(0.0, word.bbox[3] - word.bbox[1])
        for words in word_lines
        for word in words
        if max(0.0, word.bbox[3] - word.bbox[1]) > 0
    )
    bucket_width = max(6.0, (median(heights) * 0.55) if heights else 6.0)
    columns: Counter[int] = Counter()
    for words in word_lines:
        line_columns: set[int] = set()
        for word in words:
            if not any(ch.isalnum() for ch in word.text):
                continue
            line_columns.add(int(round(word.bbox[0] / bucket_width)))
        columns.update(line_columns)
    min_lines = max(3, min(5, len(word_lines) // 2))
    return sum(1 for count in columns.values() if count >= min_lines)


def text_may_have_native_word_table_signal(text: str) -> bool:
    alphanumeric_groups = 0
    previous_was_alphanumeric = False
    has_digit = False
    for char in text:
        is_alphanumeric = char.isalnum()
        if is_alphanumeric and not previous_was_alphanumeric:
            alphanumeric_groups += 1
        if char.isdigit():
            has_digit = True
        previous_was_alphanumeric = is_alphanumeric
    return has_digit and alphanumeric_groups >= 3


def candidate_layout_aligned_column_count(
    candidates: tuple[OcrCandidate, ...],
) -> int:
    best = 0
    for candidate in candidates:
        words = [
            word
            for row in candidate.result.word_rows
            if (word := ocr_layout.ocr_layout_word(row)) is not None
        ]
        if len(words) < 12:
            continue
        lines = [
            line
            for line in ocr_layout.ocr_words_to_lines(words)
            if table_ocr_geometry_line_has_table_signal(line)
        ]
        if len(lines) < 4:
            continue
        best = max(best, table_ocr_geometry_aligned_column_count(lines))
    return best


def page_drawing_lines(page: Any) -> list[Any]:
    if page is None:
        return []
    try:
        graphics = page.get_graphics()
    except Exception:
        return []
    return list(getattr(graphics, "lines", ()) or ())


def drawing_rule_counts(lines: list[Any]) -> tuple[int, int]:
    horizontal = 0
    vertical = 0
    for line in lines:
        try:
            x0, x1 = sorted((float(line.x0), float(line.x1)))
            y0, y1 = sorted((float(line.y0), float(line.y1)))
            line_width = float(getattr(line, "line_width", 1.0))
        except (TypeError, ValueError):
            continue
        width = x1 - x0
        height = y1 - y0
        if width >= 24.0 and height <= 1.25 and line_width <= 2.0:
            horizontal += 1
        elif height >= 10.0 and width <= 1.25 and line_width <= 2.0:
            vertical += 1
    return horizontal, vertical


def page_has_dominant_image(page: Any) -> bool:
    if page is None:
        return False
    try:
        return ocr_page_analysis.has_dominant_page_image(page)
    except Exception:
        return False


def layout_line_has_numeric_signal(line: LayoutLine) -> bool:
    text = line.text()
    if not text.strip():
        return False
    words = layout_text_word_strings(text)
    digit_words = 0
    alpha_words = 0
    token_count = 0
    for word in words:
        token = word.strip()
        if not token:
            continue
        token_count += 1
        if any(ch.isdigit() for ch in token):
            digit_words += 1
        elif any(ch.isalpha() for ch in token):
            alpha_words += 1
    return digit_words >= 2 or (digit_words >= 1 and token_count >= 4 and alpha_words <= 2)


def layout_text_word_strings(text: str) -> tuple[str, ...]:
    words: list[str] = []
    current = ""
    for char in text:
        if char.isspace():
            if current:
                words.append(current)
                current = ""
            continue
        if current and char.isalnum() != current[-1].isalnum():
            words.append(current)
            current = ""
        current += char
    if current:
        words.append(current)
    return tuple(words)


def native_word_line_has_table_signal(words: Sequence[Any]) -> bool:
    tokens = [word.text for word in words if any(ch.isalnum() for ch in word.text)]
    if len(tokens) < 3:
        return False
    digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    return digit_tokens >= 2 or (
        digit_tokens >= 1 and len(tokens) >= 4 and digit_tokens / len(tokens) >= 0.25
    )


def ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def occupied_area_ratio_for_line_boxes(
    boxes: Iterable[tuple[float, float, float, float]],
    *,
    page_width: float,
    page_height: float,
) -> float:
    page_area = max(1.0, page_width * page_height)
    total = 0.0
    for x0, y0, x1, y1 in boxes:
        width = max(0.0, float(x1) - float(x0))
        height = max(0.0, float(y1) - float(y0))
        total += width * height
    return min(1.0, total / page_area)


def table_candidate_layout_signal_count(
    candidates: tuple[OcrCandidate, ...],
) -> int:
    best = 0
    for candidate in candidates:
        words = [
            word
            for row in candidate.result.word_rows
            if (word := ocr_layout.ocr_layout_word(row)) is not None
        ]
        if len(words) < 12:
            continue
        lines = [
            line
            for line in ocr_layout.ocr_words_to_lines(words)
            if table_ocr_geometry_line_has_table_signal(line)
        ]
        if len(lines) < 4:
            continue
        aligned_columns = table_ocr_geometry_aligned_column_count(lines)
        score = len(lines) + min(8, aligned_columns)
        if candidate.name in ocr_table_regions.OCR_TABLE_CANDIDATE_NAMES:
            score += 2
        best = max(best, score)
    return best


def table_ocr_geometry_line_has_table_signal(
    words: list[ocr_layout.OcrLayoutWord],
) -> bool:
    tokens = [word.text for word in words if any(ch.isalnum() for ch in word.text)]
    if len(tokens) < 3:
        return False
    digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    return digit_tokens >= 2 or (
        digit_tokens >= 1 and len(tokens) >= 4 and digit_tokens / len(tokens) >= 0.25
    )


def table_ocr_geometry_aligned_column_count(
    lines: list[list[ocr_layout.OcrLayoutWord]],
) -> int:
    if len(lines) < 4:
        return 0
    columns: Counter[int] = Counter()
    heights = sorted(word.height for line in lines for word in line if word.height > 0)
    bucket_width = max(6.0, heights[len(heights) // 2] * 0.55) if heights else 6.0
    for line in lines:
        line_columns: set[int] = set()
        for word in line:
            if not any(ch.isalnum() for ch in word.text):
                continue
            line_columns.add(int(round(word.x0 / bucket_width)))
        columns.update(line_columns)
    min_lines = max(3, min(5, len(lines) // 2))
    return sum(1 for count in columns.values() if count >= min_lines)


def invoice_text_signal_count(text: str) -> int:
    if not text:
        return 0
    tokens = set(normalized_text_tokens(text))
    invoice_terms = {
        "amount",
        "balance",
        "bill",
        "billing",
        "due",
        "invoice",
        "rechnung",
        "subtotal",
        "tax",
        "total",
        "ust",
        "vat",
    }
    return len(tokens.intersection(invoice_terms))


def formula_heavy_ocr_text(text: str) -> bool:
    formula_lines = 0
    formula_chars = 0
    for line in text.splitlines():
        count = sum(1 for ch in line if ch in ocr_text_analysis.OCR_FORMULA_CHARS)
        if count >= 2:
            formula_lines += 1
        formula_chars += count
    return formula_lines >= 4 and formula_chars >= 20


def ocr_text_is_cleaner_candidate(
    text: str,
    ocr_text: str,
    *,
    text_tokens: int | None = None,
    ocr_tokens: int | None = None,
) -> bool:
    if text_tokens is None:
        text_tokens = extracted_text_token_count(text)
    if ocr_tokens is None:
        ocr_tokens = extracted_text_token_count(ocr_text)
    if text_tokens < 80 or ocr_tokens < 80:
        return False
    token_ratio = ocr_tokens / text_tokens
    if not (0.75 <= token_ratio <= 1.45):
        return False
    text_quality = text_ocr_quality_score(text)
    ocr_quality = text_ocr_quality_score(ocr_text)
    if (
        token_ratio <= 1.35
        and native_text_has_glued_uppercase_ocr_tokens(text)
        and ocr_quality <= text_quality + 0.01
    ):
        return True
    margin = 0.05 if sparse_text_looks_noisy(text) else 0.015
    return ocr_quality + margin < text_quality


def native_text_has_glued_uppercase_ocr_tokens(text: str) -> bool:
    glued_tokens = 0
    for token in normalized_text_tokens(text):
        if not token.isupper() or len(token) < 18:
            continue
        if any(ch.isdigit() or ch == "_" for ch in token):
            continue
        glued_tokens += 1
        if glued_tokens >= 1:
            return True
    return False


__all__ = (
    "OcrCandidateGeometryProfile",
    "PageTextGeometryProfile",
    "classify_page_region",
    "ocr_candidate_geometry_profile",
    "ocr_candidate_is_complete_for_general_scan",
    "page_text_geometry_profile",
    "should_preserve_dense_numeric_native_text_against_ocr",
    "should_preserve_native_text_against_rendered_page_ocr",
    "should_replace_dominant_image_native_text_with_ocr",
    "should_replace_noisy_native_text_with_compact_ocr",
    "should_replace_symbol_encoded_text_with_ocr",
    "should_replace_text_with_ocr",
)
