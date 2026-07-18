# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING

from core_ocr.impl import text_analysis as ocr_text_analysis

from core_pdf.impl.engine.extraction.ocr import rendering as ocr_rendering

if TYPE_CHECKING:
    from core_ocr.impl.candidates import OcrCandidate

    from core_pdf.impl.engine.extraction.page_text.mixin import (
        PageExtractionHost,
    )


def high_density_full_page_ocr_candidate_is_usable_without_region_retry(
    candidate: OcrCandidate,
) -> bool:
    if candidate.name != "high_density_full_page_image":
        return False
    text = candidate.result.text
    tokens = ocr_text_analysis.extracted_text_token_count(text)
    if tokens < 35 or tokens > 90:
        return False
    confidence = candidate.result.confidence or 0
    if confidence < 94:
        return False
    if ocr_text_analysis.text_ocr_quality_score(text) > 0.14:
        return False
    if len(candidate.result.line_rows) < 20 and len(candidate.result.word_rows) < 100:
        return False
    return ocr_candidate_score(candidate) >= 98.0


def rendered_sparse_ocr_candidate_is_usable_without_region_retry(
    candidate: OcrCandidate,
) -> bool:
    if not candidate.name.startswith("rendered_page_"):
        return False
    text = candidate.result.text
    tokens = ocr_text_analysis.extracted_text_token_count(text)
    if tokens < 8 or tokens > 80:
        return False
    confidence = candidate.result.confidence or 0
    if confidence < 35:
        return False
    if ocr_text_analysis.text_ocr_quality_score(text) > 0.12:
        return False
    if ocr_text_analysis.sparse_text_looks_noisy(text):
        return False
    return ocr_candidate_score(candidate) >= 34.0


def select_ocr_candidate(
    candidates: list[OcrCandidate],
    *,
    support_text: str = "",
) -> OcrCandidate | None:
    best: OcrCandidate | None = None
    best_score = float("-inf")
    for candidate in candidates:
        if not candidate.result.text or candidate.name.startswith("verification_"):
            continue
        candidate_score = ocr_candidate_score(candidate, support_text=support_text)
        if best is None or prefer_high_resolution_sparse_dense_candidate(candidate, best):
            best = candidate
            best_score = candidate_score
        elif (
            prefer_high_resolution_sparse_dense_candidate(best, candidate)
            or prefer_more_complete_short_rendered_candidate(
                best,
                candidate,
                candidate_score=best_score,
                best_score=candidate_score,
            )
            or prefer_sparse_formula_rendered_candidate(
                best,
                candidate,
                candidate_score=best_score,
                best_score=candidate_score,
            )
            or prefer_more_complete_compact_label_rendered_candidate(
                best,
                candidate,
                candidate_score=best_score,
                best_score=candidate_score,
            )
            or prefer_word_layout_same_source_candidate(best, candidate)
            or prefer_word_refined_same_source_candidate(best, candidate)
            or prefer_rendered_page_qa_candidate(best, candidate)
            or prefer_lexically_cleaner_rendered_candidate(best, candidate)
            or prefer_cleaner_same_render_dpi_candidate(best, candidate)
            or prefer_cleaner_line_art_candidate(best, candidate)
            or prefer_plain_rendered_page_candidate(
                best,
                candidate,
            )
            or prefer_more_complete_page_candidate(best, candidate)
        ):
            continue
        elif (
            prefer_more_complete_page_candidate(candidate, best)
            or candidate_score > best_score
            or prefer_more_complete_short_rendered_candidate(
                candidate,
                best,
                candidate_score=candidate_score,
                best_score=best_score,
            )
            or prefer_sparse_formula_rendered_candidate(
                candidate,
                best,
                candidate_score=candidate_score,
                best_score=best_score,
            )
            or prefer_more_complete_compact_label_rendered_candidate(
                candidate,
                best,
                candidate_score=candidate_score,
                best_score=best_score,
            )
            or prefer_word_layout_same_source_candidate(candidate, best)
            or prefer_word_refined_same_source_candidate(candidate, best)
            or prefer_rendered_page_qa_candidate(candidate, best)
            or prefer_lexically_cleaner_rendered_candidate(candidate, best)
            or prefer_cleaner_same_render_dpi_candidate(candidate, best)
            or prefer_cleaner_line_art_candidate(candidate, best)
            or prefer_plain_rendered_page_candidate(candidate, best)
        ):
            best = candidate
            best_score = candidate_score
    return best


def prefer_high_resolution_sparse_dense_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if candidate.name != "full_page_high_resolution_sparse":
        return False
    if best.name != "full_page_simple":
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < max(250, int(best_tokens * 0.88)):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence < best_confidence + 8:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    if candidate_quality > best_quality + 0.04:
        return False
    candidate_artifact = ocr_text_analysis.scanned_ocr_artifact_score(candidate_text)
    best_artifact = ocr_text_analysis.scanned_ocr_artifact_score(best_text)
    if candidate_artifact > best_artifact + 0.03:
        return False
    if ocr_text_analysis.numeric_token_ratio(candidate_text) < 0.22:
        return False
    return ocr_text_analysis.text_has_many_digit_lines(candidate_text)


def prefer_more_complete_page_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if not broad_page_candidate_name(candidate.name):
        return False
    if not compact_region_candidate_name(best.name):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < max(80, best_tokens * 2):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 12 < best_confidence:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    if candidate_quality > 0.42:
        return False
    candidate_score = ocr_candidate_score(candidate)
    best_score = ocr_candidate_score(best)
    allowed_gap = 14.0
    if best_tokens < 60 and candidate_tokens >= best_tokens * 4:
        allowed_gap = 28.0
    return best_score - candidate_score <= allowed_gap


def prefer_word_layout_same_source_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if not candidate.name.endswith("_word_layout"):
        return False
    if candidate.name.removesuffix("_word_layout") != best.name:
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < int(best_tokens * 0.90):
        return False
    if candidate_tokens > max(best_tokens + 40, int(best_tokens * 1.10)):
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    candidate_artifact = ocr_text_analysis.scanned_ocr_artifact_score(candidate_text)
    best_artifact = ocr_text_analysis.scanned_ocr_artifact_score(best_text)
    if candidate_quality <= best_quality + 0.02 and candidate_artifact + 0.01 < best_artifact:
        return True
    return candidate_quality + 0.01 < best_quality and candidate_artifact <= best_artifact + 0.02


def prefer_word_refined_same_source_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if not candidate.name.endswith("_word_refined"):
        return False
    if ocr_variant_source_name(candidate.name) != ocr_variant_source_name(best.name):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    if candidate_text == best_text:
        return False
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < int(best_tokens * 0.96):
        return False
    if candidate_tokens > max(best_tokens + 20, int(best_tokens * 1.04)):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 2 < best_confidence:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    if candidate_quality > best_quality + 0.015:
        return False
    candidate_artifact = ocr_text_analysis.scanned_ocr_artifact_score(candidate_text)
    best_artifact = ocr_text_analysis.scanned_ocr_artifact_score(best_text)
    if candidate_artifact > best_artifact + 0.015:
        return False
    return ocr_candidate_score(candidate) >= ocr_candidate_score(best) - 0.75


def ocr_variant_source_name(name: str) -> str:
    for suffix in (
        "_word_refined",
        "_reconciled_layout",
        "_word_layout",
    ):
        name = name.removesuffix(suffix)
    return name


def broad_page_candidate_name(name: str) -> bool:
    return (
        name == "full_page_image"
        or name == "high_density_full_page_image"
        or name.startswith("rendered_page_")
    )


def compact_region_candidate_name(name: str) -> bool:
    return name in {"rectangle_regions", "rendered_regions"} or name.endswith("_regions")


def prefer_more_complete_short_rendered_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
    *,
    candidate_score: float,
    best_score: float,
) -> bool:
    if best_score - candidate_score > 2.0:
        return False
    candidate_dpi = plain_rendered_page_candidate_dpi(candidate.name)
    best_dpi = plain_rendered_page_candidate_dpi(best.name)
    if candidate_dpi is None or best_dpi is None or candidate_dpi >= best_dpi:
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if not (40 <= candidate_tokens <= 140 and 40 <= best_tokens <= 140):
        return False
    if candidate_tokens < best_tokens + 2:
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 2 < best_confidence:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    if candidate_quality > 0.16 or best_quality > 0.18:
        return False
    return candidate_quality <= best_quality + 0.01


def prefer_sparse_formula_rendered_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
    *,
    candidate_score: float,
    best_score: float,
) -> bool:
    if not candidate.name.startswith("rendered_page_"):
        return False
    if not candidate.name.endswith("_sparse"):
        return False
    if not best.name.startswith("rendered_page_two_columns"):
        return False
    if best_score - candidate_score > 6.0:
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < 300 or best_tokens < 300:
        return False
    if candidate_tokens < int(best_tokens * 0.80):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence < best_confidence:
        return False
    return ocr_text_analysis.ocr_text_has_dense_formula_notation(candidate_text) and (
        ocr_text_analysis.ocr_text_has_dense_formula_notation(best_text)
    )


def prefer_more_complete_compact_label_rendered_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
    *,
    candidate_score: float,
    best_score: float,
) -> bool:
    candidate_dpi = rendered_candidate_base_dpi(candidate.name)
    best_dpi = rendered_candidate_base_dpi(best.name)
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    candidate_artifact = ocr_text_analysis.scanned_ocr_artifact_score(candidate_text)
    best_artifact = ocr_text_analysis.scanned_ocr_artifact_score(best_text)
    max_allowed_gap = 2.0
    min_candidate_tokens = max(best_tokens + 32, int(best_tokens * 3.0))
    return (
        best_score - candidate_score <= max_allowed_gap
        and candidate_dpi is not None
        and best_dpi is not None
        and candidate_dpi == best_dpi
        and candidate.name.endswith("_psm11")
        and best.name.endswith("_psm4")
        and 30 <= best_tokens <= 90
        and 80 <= candidate_tokens <= 220
        and candidate_tokens >= min_candidate_tokens
        and candidate_quality <= best_quality + 0.07
        and candidate_artifact <= best_artifact + 0.08
    )


def prefer_plain_rendered_page_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if candidate.region_count != 0 or best.region_count < 2:
        return False
    if not candidate.name.startswith("rendered_page_"):
        return False
    if not best.name.startswith("rendered_page_"):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < 200 or best_tokens <= candidate_tokens:
        return False
    if best_tokens >= max(int(candidate_tokens * 1.18), candidate_tokens + 48):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    return best_confidence <= candidate_confidence and best_quality >= candidate_quality - 0.005


def prefer_cleaner_same_render_dpi_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    candidate_dpi = rendered_candidate_base_dpi(candidate.name)
    best_dpi = rendered_candidate_base_dpi(best.name)
    if candidate_dpi is None or best_dpi is None or candidate_dpi != best_dpi:
        return False
    if candidate.name.endswith("_sparse"):
        return False
    if not best.name.endswith("_sparse"):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    max_tokens = max(candidate_tokens, best_tokens)
    if abs(candidate_tokens - best_tokens) > max(12, int(max_tokens * 0.03)):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 1 < best_confidence:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    return candidate_quality + 0.008 < best_quality


def prefer_lexically_cleaner_rendered_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if not candidate.name.startswith("rendered_page_"):
        return False
    if not best.name.startswith("rendered_page_"):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    max_tokens = max(candidate_tokens, best_tokens)
    if max_tokens < 120:
        return False
    if abs(candidate_tokens - best_tokens) > max(24, int(max_tokens * 0.08)):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 2 < best_confidence:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    if candidate_quality > best_quality + 0.02:
        return False
    candidate_unknown_ratio = ocr_text_analysis.alpha_unknown_word_ratio(candidate_text)
    best_unknown_ratio = ocr_text_analysis.alpha_unknown_word_ratio(best_text)
    if candidate_unknown_ratio is None or best_unknown_ratio is None:
        return False
    return candidate_unknown_ratio + 0.025 < best_unknown_ratio


def prefer_rendered_page_qa_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if not candidate.name.startswith("rendered_page_"):
        return False
    if not best.name.startswith("rendered_page_"):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    max_tokens = max(candidate_tokens, best_tokens)
    if max_tokens < 30:
        return False
    if abs(candidate_tokens - best_tokens) > max(18, int(max_tokens * 0.16)):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 4 < best_confidence:
        return False
    candidate_fragmentation = ocr_text_analysis.rendered_ocr_fragmentation_score(candidate_text)
    best_fragmentation = ocr_text_analysis.rendered_ocr_fragmentation_score(best_text)
    candidate_artifact = ocr_text_analysis.scanned_ocr_artifact_score(candidate_text)
    best_artifact = ocr_text_analysis.scanned_ocr_artifact_score(best_text)
    candidate_coverage = ocr_text_analysis.rendered_ocr_line_coverage_score(
        candidate_text,
        line_rows=len(candidate.result.line_rows),
        word_rows=len(candidate.result.word_rows),
    )
    best_coverage = ocr_text_analysis.rendered_ocr_line_coverage_score(
        best_text,
        line_rows=len(best.result.line_rows),
        word_rows=len(best.result.word_rows),
    )
    candidate_score = ocr_candidate_score(candidate)
    best_score = ocr_candidate_score(best)
    if candidate_score + 0.5 < best_score and candidate_artifact > best_artifact + 0.01:
        return False
    if candidate_fragmentation + 0.025 < best_fragmentation:
        return candidate_coverage + 0.04 >= best_coverage
    if candidate_artifact + 0.003 < best_artifact:
        return (
            candidate_fragmentation <= best_fragmentation + 0.015
            and candidate_coverage + 0.04 >= best_coverage
        )
    if candidate_coverage >= best_coverage + 0.10:
        return candidate_fragmentation <= best_fragmentation + 0.015
    return False


def plain_rendered_page_candidate_dpi(name: str) -> int | None:
    match = re.fullmatch(r"rendered_page_(\d+)dpi", name)
    if match is None:
        return None
    return int(match.group(1))


def rendered_candidate_base_dpi(name: str) -> int | None:
    match = re.fullmatch(r"rendered_page_(\d+)dpi(?:_psm\d+)?(?:_sparse)?", name)
    if match is None:
        return None
    return int(match.group(1))


def prefer_cleaner_line_art_candidate(
    candidate: OcrCandidate,
    best: OcrCandidate,
) -> bool:
    if not candidate.name.startswith("line_art_text_mask_"):
        return False
    if not broad_page_candidate_name(best.name):
        return False
    candidate_text = candidate.result.text
    best_text = best.result.text
    candidate_tokens = ocr_text_analysis.extracted_text_token_count(candidate_text)
    best_tokens = ocr_text_analysis.extracted_text_token_count(best_text)
    if candidate_tokens < int(best_tokens * 0.94):
        return False
    if candidate_tokens > max(best_tokens + 16, int(best_tokens * 1.08)):
        return False
    candidate_confidence = candidate.result.confidence or 0
    best_confidence = best.result.confidence or 0
    if candidate_confidence + 4 < best_confidence:
        return False
    candidate_quality = ocr_text_analysis.text_ocr_quality_score(candidate_text)
    best_quality = ocr_text_analysis.text_ocr_quality_score(best_text)
    if candidate_quality + 0.04 >= best_quality:
        return False
    candidate_artifact = ocr_text_analysis.scanned_ocr_artifact_score(candidate_text)
    best_artifact = ocr_text_analysis.scanned_ocr_artifact_score(best_text)
    return candidate_artifact <= best_artifact + 0.02


def ocr_candidate_score(candidate: OcrCandidate, *, support_text: str = "") -> float:
    return _ocr_candidate_score_from_signature(
        candidate.name,
        candidate.result.text,
        candidate.result.confidence,
        candidate.bbox,
        candidate.region_count,
        candidate.image_width,
        candidate.image_height,
        len(candidate.result.line_rows),
        len(candidate.result.word_rows),
        ocr_text_analysis.overlapping_ocr_word_penalty(candidate.result.word_rows),
        support_text,
    )


@lru_cache(maxsize=16384)
def _ocr_candidate_score_from_signature(
    name: str,
    text: str,
    confidence: int | None,
    bbox: tuple[int, int, int, int] | None,
    region_count: int,
    image_width: int | None,
    image_height: int | None,
    line_rows: int,
    word_rows: int,
    duplicate_word_ratio: float,
    support_text: str,
) -> float:
    if not text:
        return float("-inf")
    score = float(confidence if confidence is not None else 50)
    tokens = ocr_text_analysis.extracted_text_token_count(text)
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    if tokens <= 2:
        score -= 80.0
    elif tokens < 8:
        score -= 35.0
    score += min(tokens, 800) * 0.16
    if ocr_variant_source_name(name) == "full_page_image" and 35 <= tokens <= 100:
        score += min(3.0, (tokens - 30) * 0.10)
        score -= min(
            14.0,
            ocr_text_analysis.full_page_diagram_mixed_noise_score(text) * 18.0,
        )
    if 20 <= tokens <= 100 and (confidence or 0) >= 85 and quality <= 0.30:
        score += tokens * 0.08
    score -= quality * 20.0
    if tokens >= 300 and quality >= 0.45:
        score -= min(140.0, 80.0 + (tokens - 300) * 0.12 + (quality - 0.45) * 80.0)
    if tokens >= 300 and quality >= 0.35 and (confidence or 0) < 55:
        score -= min(70.0, 35.0 + (quality - 0.35) * 120.0)
    if bbox is not None and region_count == 1 and tokens >= 20:
        score += 2.0
    if name.startswith("rendered_page_two_columns") and region_count >= 2:
        score += min(120.0, region_count * 60.0)
    score += ocr_text_analysis.table_like_ocr_coverage_bonus(
        text,
        tokens,
        confidence,
        quality,
    )
    if name.startswith("rendered_page_"):
        fragmentation = ocr_text_analysis.rendered_ocr_fragmentation_score(text)
        line_coverage = ocr_text_analysis.rendered_ocr_line_coverage_score(
            text,
            line_rows=line_rows,
            word_rows=word_rows,
        )
        gibberish = ocr_text_analysis.alphabetic_gibberish_score(text)
        score -= min(10.0, fragmentation * 28.0)
        score += min(3.0, line_coverage * 3.0)
        score -= min(14.0, ocr_text_analysis.scanned_ocr_artifact_score(text) * 100.0)
        score -= min(60.0, gibberish * max(35.0, min(tokens, 800) * 0.80))
        if name.startswith(
            "rendered_page_two_columns"
        ) and ocr_text_analysis.ocr_text_has_dense_formula_notation(text):
            score += 2.0
    if broad_page_candidate_name(name):
        support_overlap = ocr_text_analysis.support_text_overlap_score(text, support_text)
        if support_overlap is not None:
            # Native text is useful evidence, but not ground truth. Keep the
            # adjustment small enough that clean OCR can still win on damaged
            # or incomplete native layers.
            score += (support_overlap - 0.50) * 10.0
        score -= min(18.0, duplicate_word_ratio * 36.0)
    if name == "table_cell_consensus":
        score += min(36.0, region_count * 0.35)
        if ocr_text_analysis.numeric_token_ratio(text) >= 0.22:
            score += 10.0
        if ocr_text_analysis.text_has_many_digit_lines(text):
            score += 12.0
        score -= min(18.0, ocr_text_analysis.scanned_ocr_artifact_score(text) * 80.0)
    if name.startswith("line_art_text_mask_"):
        score -= 12.0
    if ocr_text_analysis.sparse_text_looks_noisy(text):
        score -= 20.0
    noise = ocr_text_analysis.uninterpretable_char_count(text)
    if noise:
        score -= min(30.0, noise * 2.0)
    if bbox is not None and image_width is not None and image_height is not None:
        score -= ocr_text_analysis.fragmentary_region_candidate_penalty(
            SimpleNamespace(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            ),
            tokens,
        )
    return score


def ocr_render_dpi_candidates_for_page(
    page: PageExtractionHost,
    vector_text: str,
) -> tuple[int, ...]:
    candidates = ocr_rendering.ocr_render_dpi_candidates_for_vector_text(vector_text)
    dimensions = ocr_rendering.ocr_page_dimensions_points(page)
    if dimensions is None:
        return candidates
    width_points, height_points = dimensions
    safe_candidates: list[int] = []
    seen: set[int] = set()
    for dpi in candidates:
        if should_prefer_tiled_rendered_ocr(
            width_points,
            height_points,
            dpi,
            vector_text,
        ):
            continue
        safe_dpi = ocr_rendering.safe_ocr_render_dpi(width_points, height_points, dpi)
        if safe_dpi is None or safe_dpi in seen:
            continue
        seen.add(safe_dpi)
        safe_candidates.append(safe_dpi)
    return tuple(safe_candidates)


def ocr_tiled_render_dpi_candidates_for_page(
    page: PageExtractionHost,
    vector_text: str,
) -> tuple[int, ...]:
    candidates = ocr_rendering.ocr_render_dpi_candidates_for_vector_text(vector_text)
    dimensions = ocr_rendering.ocr_page_dimensions_points(page)
    if dimensions is None:
        return ()
    width_points, height_points = dimensions
    return tuple(
        dpi
        for dpi in candidates
        if not ocr_rendering.ocr_render_pixel_size_is_supported(width_points, height_points, dpi)
        or should_prefer_tiled_rendered_ocr(
            width_points,
            height_points,
            dpi,
            vector_text,
        )
    )


def should_prefer_tiled_rendered_ocr(
    width_points: float,
    height_points: float,
    dpi: int,
    vector_text: str,
) -> bool:
    if not vector_text_supports_tiled_rendered_ocr(vector_text):
        return False
    width, height = ocr_rendering.ocr_render_pixel_dimensions(width_points, height_points, dpi)
    return width * height >= ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MIN_PIXELS


def vector_text_supports_tiled_rendered_ocr(vector_text: str) -> bool:
    vector_tokens = ocr_text_analysis.extracted_text_token_count(vector_text)
    if vector_tokens >= ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MIN_TOKENS:
        return True
    return ocr_text_analysis.vector_text_supports_schematic_tiled_ocr(vector_text)


def ocr_render_tile_max_side_pixels_for_page(
    page: PageExtractionHost,
    dpi: int,
    vector_text: str,
) -> int:
    dimensions = ocr_rendering.ocr_page_dimensions_points(page)
    if dimensions is None:
        return ocr_rendering.OCR_RENDER_TILE_MAX_SIDE_PIXELS
    width_points, height_points = dimensions
    if should_prefer_tiled_rendered_ocr(width_points, height_points, dpi, vector_text):
        return ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MAX_SIDE_PIXELS
    if not ocr_text_analysis.vector_text_supports_schematic_tiled_ocr(vector_text):
        return ocr_rendering.OCR_RENDER_TILE_MAX_SIDE_PIXELS
    vector_tokens = ocr_text_analysis.extracted_text_token_count(vector_text)
    if vector_tokens < 120:
        return ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MAX_SIDE_PIXELS
    return ocr_rendering.OCR_RENDER_TILE_MAX_SIDE_PIXELS
