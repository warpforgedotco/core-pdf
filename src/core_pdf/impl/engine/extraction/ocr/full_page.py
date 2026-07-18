# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_ocr.impl.text_analysis import (
    extracted_text_token_count,
    normalized_text_tokens,
    numeric_token_ratio,
    scanned_ocr_artifact_score,
    sparse_text_looks_noisy,
    text_has_many_digit_lines,
    text_ocr_quality_score,
)
from core_ocr.impl.types import (
    OcrImage,
    OcrTextResult,
    leptonica_pix_size_is_supported,
)

from core_pdf.impl.engine.extraction.ocr import (
    execution as ocr_execution,
)
from core_pdf.impl.engine.extraction.ocr import (
    iterator_layout as ocr_iterator_layout,
)
from core_pdf.impl.engine.extraction.ocr.backend import TesseractCtypesBackend

OCR_FALLBACK_DPI = ocr_execution.OCR_DEFAULT_DPI
OCR_FALLBACK_PAGE_SEGMENTATION_MODE = ocr_execution.OCR_DEFAULT_PAGE_SEGMENTATION_MODE
OCR_FALLBACK_ALTERNATE_PAGE_SEGMENTATION_MODE = 4
OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE = 11
OCR_FALLBACK_SPARSE_OSD_PAGE_SEGMENTATION_MODE = 12
OCR_FALLBACK_ALTERNATE_CONFIDENCE_MARGIN = 10
OCR_TARGETED_THRESHOLDING_PROFILES = (
    ("threshold_leptonica_otsu", {"thresholding_method": "1"}),
    (
        "threshold_sauvola",
        {
            "thresholding_method": "2",
            "thresholding_window_size": "0.33",
            "thresholding_kfactor": "0.34",
        },
    ),
)


def ocr_image_to_text_with_timeout(image: OcrImage, timeout: float | None) -> str:
    return ocr_image_to_text_result_with_timeout(image, timeout).text


def ocr_image_to_text_result_with_timeout(image: OcrImage, timeout: float | None) -> OcrTextResult:
    del timeout
    try:
        return ocr_image_worker(image)
    except BaseException:
        return OcrTextResult("", None)


def ocr_image_worker(image: OcrImage) -> OcrTextResult:
    resolution = image.resolution or OCR_FALLBACK_DPI
    backend = TesseractCtypesBackend.from_system()
    if backend is None:
        return OcrTextResult("", None)
    try:
        return backend.image_to_text_result(
            image,
            psm=OCR_FALLBACK_PAGE_SEGMENTATION_MODE,
            resolution=resolution,
        )
    except BaseException:
        return OcrTextResult("", None)


def should_try_targeted_thresholding_ocr(
    image: OcrImage,
    result: OcrTextResult,
) -> bool:
    if image.encoded is not None or image.bytes_per_pixel not in {1, 3, 4}:
        return False
    if not image.data or image.width <= 0 or image.height <= 0:
        return False
    if not leptonica_pix_size_is_supported(image.width, image.height):
        return False
    if not result.text:
        return True
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence if result.confidence is not None else 0
    if tokens < 20:
        return True
    if confidence >= 55:
        return False
    quality = text_ocr_quality_score(result.text)
    artifact_score = scanned_ocr_artifact_score(result.text)
    return quality >= 0.28 and artifact_score >= 0.04 and tokens <= 360


def should_try_tesseract_table_profile_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return True
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence
    if tokens < 90:
        return True
    if confidence is not None and confidence < 62 and tokens < 500:
        return True
    if tokens < 900 and text_has_many_digit_lines(result.text):
        return text_ocr_quality_score(result.text) >= 0.10
    return False


def should_try_alternate_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return True
    tokens = extracted_text_token_count(result.text)
    if tokens < 80:
        return True
    confidence = result.confidence
    return confidence is None or confidence < 70


def select_targeted_thresholding_ocr_result(
    primary: OcrTextResult,
    thresholded: OcrTextResult,
) -> OcrTextResult:
    if not thresholded.text:
        return primary
    if not primary.text:
        return thresholded
    primary_tokens = extracted_text_token_count(primary.text)
    thresholded_tokens = extracted_text_token_count(thresholded.text)
    if primary_tokens >= 40 and thresholded_tokens < int(primary_tokens * 0.70):
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    thresholded_confidence = thresholded.confidence if thresholded.confidence is not None else 50
    primary_quality = text_ocr_quality_score(primary.text)
    thresholded_quality = text_ocr_quality_score(thresholded.text)
    if (
        thresholded_confidence >= primary_confidence + 6
        and thresholded_quality <= primary_quality + 0.03
    ):
        return thresholded
    if (
        thresholded_confidence >= primary_confidence - 3
        and thresholded_tokens >= int(primary_tokens * 0.92)
        and thresholded_quality + 0.04 < primary_quality
    ):
        return thresholded
    return primary


def should_try_iterator_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return True
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence
    if tokens < 80:
        return True
    if confidence is not None and confidence < 58 and tokens < 260:
        return True
    quality = text_ocr_quality_score(result.text)
    if 150 <= tokens <= 280 and confidence is not None and confidence >= 60 and (quality >= 0.20):
        return True
    if (
        180 <= tokens <= 320
        and confidence is not None
        and confidence >= 70
        and not sparse_text_looks_noisy(result.text)
        and quality >= 0.08
    ):
        return True
    return tokens < 180 and sparse_text_looks_noisy(result.text)


def should_try_high_confidence_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (40 <= tokens <= 220):
        return False
    if numeric_token_ratio(result.text) >= 0.45:
        return False
    confidence = result.confidence if result.confidence is not None else 50
    return confidence < 80 or tokens < 80


def should_try_auto_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (300 <= tokens <= 540):
        return False
    confidence = result.confidence if result.confidence is not None else 50
    return confidence >= 80 and text_ocr_quality_score(result.text) <= 0.12


def should_try_confidence_filtered_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (280 <= tokens <= 650):
        return False
    if tokens > 430 and not text_has_many_digit_lines(result.text):
        return False
    confidence = result.confidence
    if confidence is None or not (70 <= confidence <= 85):
        return False
    if numeric_token_ratio(result.text) >= 0.45:
        return False
    return text_ocr_quality_score(result.text) >= 0.08


def should_try_medium_sparse_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (220 <= tokens <= 520):
        return False
    if numeric_token_ratio(result.text) >= 0.35:
        return False
    quality = text_ocr_quality_score(result.text)
    if quality < 0.12:
        return False
    return (
        sparse_text_looks_noisy(result.text)
        or quality >= 0.16
        or scanned_ocr_artifact_score(result.text) >= 0.03
    )


def should_try_large_auto_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (900 <= tokens <= 1800):
        return False
    confidence = result.confidence if result.confidence is not None else 50
    return confidence < 70 and text_ocr_quality_score(result.text) <= 0.12


def should_try_large_sparse_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (650 <= tokens <= 1_600):
        return False
    if numeric_token_ratio(result.text) >= 0.45:
        return False
    confidence = result.confidence if result.confidence is not None else 50
    if confidence >= 75:
        return False
    return text_ocr_quality_score(result.text) >= 0.14


def should_try_rendered_sparse_layout_ocr(image: OcrImage, result: OcrTextResult) -> bool:
    if not image.source.startswith("rendered_page_"):
        return False
    if "_tile_" in image.source:
        return False
    if not result.text:
        return True
    tokens = extracted_text_token_count(result.text)
    if not (25 <= tokens <= 540):
        return False
    confidence = result.confidence if result.confidence is not None else 50
    quality = text_ocr_quality_score(result.text)
    if confidence >= 90 and quality <= 0.08:
        return False
    if confidence < 72:
        return True
    return quality >= 0.18 or tokens < 180


def should_try_sparse_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return True
    tokens = extracted_text_token_count(result.text)
    confidence = result.confidence
    if tokens < 150:
        return True
    quality = text_ocr_quality_score(result.text)
    if (
        180 <= tokens <= 340
        and confidence is not None
        and confidence >= 60
        and (0.08 <= quality <= 0.118)
    ):
        return True
    return confidence is not None and confidence < 58 and tokens < 600


def iterator_ocr_result(
    backend: TesseractCtypesBackend,
    image: OcrImage,
    *,
    psm: int,
    min_confidence: int = 0,
) -> OcrTextResult:
    return ocr_iterator_layout.iterator_layout_text_result(
        backend.image_to_iterator_layout(
            image,
            psm=psm,
            resolution=image.resolution or OCR_FALLBACK_DPI,
        ),
        min_confidence=min_confidence,
    )


def select_ocr_result(primary: OcrTextResult, alternate: OcrTextResult) -> OcrTextResult:
    primary_text = primary.text
    alternate_text = alternate.text
    if not primary_text:
        return alternate
    if not alternate_text:
        return primary
    primary_confidence = primary.confidence
    alternate_confidence = alternate.confidence
    if (
        primary_confidence is not None
        and alternate_confidence is not None
        and alternate_confidence >= primary_confidence + OCR_FALLBACK_ALTERNATE_CONFIDENCE_MARGIN
    ):
        return alternate
    return primary


def select_sparse_ocr_result(primary: OcrTextResult, sparse: OcrTextResult) -> OcrTextResult:
    if not sparse.text:
        return primary
    if not primary.text:
        return sparse
    primary_tokens = extracted_text_token_count(primary.text)
    sparse_tokens = extracted_text_token_count(sparse.text)
    if sparse_text_looks_noisy(sparse.text):
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    sparse_quality = text_ocr_quality_score(sparse.text)
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    sparse_confidence = sparse.confidence if sparse.confidence is not None else 50
    if sparse_tokens < max(20, int(primary_tokens * 1.20)):
        if (
            sparse_confidence >= primary_confidence
            and sparse_tokens >= int(primary_tokens * 1.06)
            and sparse_quality <= primary_quality + 0.04
        ):
            return sparse
        return primary
    if sparse_quality + 0.02 < primary_quality:
        return primary
    if (
        sparse_confidence >= primary_confidence
        and sparse_tokens >= int(primary_tokens * 1.06)
        and sparse_quality <= primary_quality + 0.04
    ):
        return sparse
    if primary_tokens < 150 and sparse_tokens >= int(primary_tokens * 1.35):
        return sparse
    if (
        primary_confidence < 60
        and sparse_confidence >= primary_confidence - 15
        and sparse_tokens >= int(primary_tokens * 1.30)
    ):
        return sparse
    return primary


def select_tesseract_table_profile_result(
    primary: OcrTextResult, table_profile: OcrTextResult
) -> OcrTextResult:
    if not table_profile.text:
        return primary
    if not primary.text:
        return table_profile
    primary_tokens = extracted_text_token_count(primary.text)
    table_tokens = extracted_text_token_count(table_profile.text)
    if primary_tokens >= 50 and table_tokens < int(primary_tokens * 0.70):
        return primary
    if table_tokens > max(primary_tokens + 180, int(primary_tokens * 1.75)):
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    table_quality = text_ocr_quality_score(table_profile.text)
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    table_confidence = table_profile.confidence if table_profile.confidence is not None else 50
    if table_quality > primary_quality + 0.08:
        return primary
    if table_confidence >= primary_confidence + 8 and table_tokens >= int(primary_tokens * 0.85):
        return table_profile
    if (
        table_tokens >= primary_tokens + 12
        and table_confidence >= primary_confidence - 4
        and table_quality <= primary_quality + 0.03
    ):
        return table_profile
    if table_quality <= primary_quality - 0.06 and table_tokens >= int(primary_tokens * 0.90):
        return table_profile
    return primary


def select_iterator_layout_result(
    primary: OcrTextResult, alternate: OcrTextResult
) -> OcrTextResult:
    if not alternate.text:
        return primary
    if not primary.text:
        return alternate
    primary_tokens = extracted_text_token_count(primary.text)
    alternate_tokens = extracted_text_token_count(alternate.text)
    if primary_tokens < 100 and alternate_tokens < primary_tokens:
        return primary
    primary_confidence = primary.confidence
    alternate_confidence = alternate.confidence
    if alternate_tokens < max(20, int(primary_tokens * 0.85)):
        if (
            primary_confidence is not None
            and alternate_confidence is not None
            and alternate_confidence >= primary_confidence + 12
            and alternate_tokens >= int(primary_tokens * 0.65)
            and text_ocr_quality_score(alternate.text)
            <= text_ocr_quality_score(primary.text) + 0.01
        ):
            return alternate
        return primary
    if primary_confidence is None or alternate_confidence is None:
        return select_ocr_result(primary, alternate)
    if alternate_confidence < primary_confidence + 5:
        return primary
    if text_ocr_quality_score(alternate.text) > text_ocr_quality_score(primary.text) + 0.03:
        return primary
    return alternate


def high_confidence_layout_result(
    backend: TesseractCtypesBackend, image: OcrImage, primary: OcrTextResult
) -> OcrTextResult:
    primary_layout = ocr_iterator_layout.iterator_text_result_from_existing_result(
        primary,
        min_confidence=80,
    )
    sparse_layout = backend.image_to_iterator_layout(
        image,
        psm=OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
        resolution=image.resolution or OCR_FALLBACK_DPI,
    )
    candidates = [
        primary_layout,
        ocr_iterator_layout.iterator_layout_text_result(sparse_layout, min_confidence=50),
        ocr_iterator_layout.iterator_layout_text_result(sparse_layout, min_confidence=80),
    ]
    if not primary_layout.text:
        page_layout = backend.image_to_iterator_layout(
            image,
            psm=OCR_FALLBACK_PAGE_SEGMENTATION_MODE,
            resolution=image.resolution or OCR_FALLBACK_DPI,
        )
        candidates.append(
            ocr_iterator_layout.iterator_layout_text_result(page_layout, min_confidence=80)
        )
    best = OcrTextResult("", None)
    best_score = float("-inf")
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    primary_tokens = extracted_text_token_count(primary.text)
    if primary_confidence >= 75 and primary_tokens < 80:
        confidence_weight = 2.0
        token_weight = 0.05
        quality_weight = 10.0
    else:
        confidence_weight = 1.0
        token_weight = 0.45 if primary_confidence < 65 else 0.12
        quality_weight = 35.0
    for candidate in candidates:
        if not candidate.text:
            continue
        confidence = candidate.confidence if candidate.confidence is not None else 50
        tokens = extracted_text_token_count(candidate.text)
        quality = text_ocr_quality_score(candidate.text)
        score = (
            confidence * confidence_weight
            + min(tokens, 120) * token_weight
            - quality * quality_weight
        )
        if score > best_score:
            best = candidate
            best_score = score
    return best


def select_high_confidence_layout_result(
    primary: OcrTextResult, high_confidence_layout: OcrTextResult
) -> OcrTextResult:
    if not high_confidence_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(high_confidence_layout.text)
    if layout_tokens < max(25, int(primary_tokens * 0.45)):
        return primary
    if layout_tokens > int(primary_tokens * 1.15):
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = (
        high_confidence_layout.confidence if high_confidence_layout.confidence is not None else 50
    )
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(high_confidence_layout.text)
    if layout_confidence >= primary_confidence + 10 and layout_quality <= primary_quality + 0.02:
        return high_confidence_layout
    if layout_confidence >= primary_confidence and layout_quality <= primary_quality - 0.04:
        return high_confidence_layout
    return primary


def select_auto_layout_result(primary: OcrTextResult, auto_layout: OcrTextResult) -> OcrTextResult:
    if not auto_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(auto_layout.text)
    if not (int(primary_tokens * 0.85) <= layout_tokens <= int(primary_tokens * 1.08)):
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = auto_layout.confidence if auto_layout.confidence is not None else 50
    if layout_confidence < primary_confidence:
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(auto_layout.text)
    if layout_quality > primary_quality + 0.005:
        return primary
    return auto_layout


def select_confidence_filtered_layout_result(
    primary: OcrTextResult, filtered_layout: OcrTextResult
) -> OcrTextResult:
    if not filtered_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(filtered_layout.text)
    if not (int(primary_tokens * 0.70) <= layout_tokens <= int(primary_tokens * 0.98)):
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = filtered_layout.confidence if filtered_layout.confidence is not None else 50
    if layout_confidence < primary_confidence + 10:
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(filtered_layout.text)
    if layout_quality > primary_quality + 0.03:
        return primary
    return filtered_layout


def medium_sparse_layout_result(
    backend: TesseractCtypesBackend, image: OcrImage, primary: OcrTextResult
) -> OcrTextResult:
    layout = backend.image_to_iterator_layout(
        image,
        psm=OCR_FALLBACK_SPARSE_PAGE_SEGMENTATION_MODE,
        resolution=image.resolution or OCR_FALLBACK_DPI,
    )
    candidates = [
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=50),
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=70),
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=80),
    ]
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    best = OcrTextResult("", None)
    best_score = float("-inf")
    for candidate in candidates:
        if not candidate.text:
            continue
        confidence = candidate.confidence if candidate.confidence is not None else 50
        tokens = extracted_text_token_count(candidate.text)
        quality = text_ocr_quality_score(candidate.text)
        score = (
            confidence * 1.5
            + min(tokens, 260) * 0.10
            - quality * 80.0
            + min(20, max(0, confidence - primary_confidence)) * 0.35
        )
        if score > best_score:
            best = candidate
            best_score = score
    return best


def select_medium_sparse_layout_result(
    primary: OcrTextResult, medium_sparse_layout: OcrTextResult
) -> OcrTextResult:
    if not medium_sparse_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(medium_sparse_layout.text)
    if layout_tokens < max(150, int(primary_tokens * 0.55)):
        return primary
    if layout_tokens > int(primary_tokens * 1.05):
        return primary
    if sparse_text_looks_noisy(medium_sparse_layout.text):
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(medium_sparse_layout.text)
    if layout_quality > primary_quality - 0.06:
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = (
        medium_sparse_layout.confidence if medium_sparse_layout.confidence is not None else 50
    )
    if layout_confidence < 88:
        return primary
    if layout_confidence < primary_confidence + 8 and layout_quality > primary_quality - 0.10:
        return primary
    return medium_sparse_layout


def large_sparse_layout_result(
    backend: TesseractCtypesBackend, image: OcrImage, primary: OcrTextResult
) -> OcrTextResult:
    layout = backend.image_to_iterator_layout(
        image,
        psm=OCR_FALLBACK_SPARSE_OSD_PAGE_SEGMENTATION_MODE,
        resolution=image.resolution or OCR_FALLBACK_DPI,
    )
    candidates = [
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=50),
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=60),
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=70),
        ocr_iterator_layout.iterator_layout_text_result(layout, min_confidence=80),
    ]
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    primary_tokens = extracted_text_token_count(primary.text)
    target_tokens = primary_tokens * 0.60
    best = OcrTextResult("", None)
    best_score = float("-inf")
    for candidate in candidates:
        if not candidate.text:
            continue
        confidence = candidate.confidence if candidate.confidence is not None else 50
        tokens = extracted_text_token_count(candidate.text)
        quality = text_ocr_quality_score(candidate.text)
        score = (
            confidence * 1.2
            + min(tokens, 700) * 0.12
            - quality * 160.0
            - abs(tokens - target_tokens) * 0.03
            + min(25, max(0, confidence - primary_confidence)) * 0.35
        )
        if score > best_score:
            best = candidate
            best_score = score
    return best


def select_large_sparse_layout_result(
    primary: OcrTextResult, large_sparse_layout: OcrTextResult
) -> OcrTextResult:
    if not large_sparse_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(large_sparse_layout.text)
    if layout_tokens < max(250, int(primary_tokens * 0.35)):
        return primary
    if layout_tokens > int(primary_tokens * 0.90):
        return primary
    if sparse_text_looks_noisy(large_sparse_layout.text):
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(large_sparse_layout.text)
    if layout_quality > primary_quality - 0.04:
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = (
        large_sparse_layout.confidence if large_sparse_layout.confidence is not None else 50
    )
    if layout_confidence < 75:
        return primary
    if layout_confidence < primary_confidence + 12 and layout_quality > primary_quality - 0.08:
        return primary
    return large_sparse_layout


def select_large_auto_layout_result(
    primary: OcrTextResult, auto_layout: OcrTextResult
) -> OcrTextResult:
    if not auto_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(auto_layout.text)
    if not (int(primary_tokens * 0.75) <= layout_tokens <= int(primary_tokens * 1.20)):
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = auto_layout.confidence if auto_layout.confidence is not None else 50
    if layout_confidence < primary_confidence + 8:
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(auto_layout.text)
    if layout_quality > primary_quality + 0.005:
        return primary
    return auto_layout


def rendered_sparse_layout_result(
    backend: TesseractCtypesBackend, image: OcrImage, primary: OcrTextResult
) -> OcrTextResult:
    candidates: list[OcrTextResult] = []
    iterator_layout = backend.image_to_iterator_layout(
        image,
        psm=OCR_FALLBACK_SPARSE_OSD_PAGE_SEGMENTATION_MODE,
        resolution=image.resolution or OCR_FALLBACK_DPI,
    )
    line_result = ocr_iterator_layout.iterator_rows_text_result(iterator_layout.textline_rows)
    if line_result.text:
        candidates.append(line_result)
    symbol_result = ocr_iterator_layout.iterator_symbol_rows_text_result(
        iterator_layout.symbol_rows
    )
    symbol_candidate = iterator_symbol_supplement_candidate(
        primary,
        line_result,
        symbol_result,
    )
    if symbol_candidate.text:
        candidates.append(symbol_candidate)
    candidates.extend(
        ocr_iterator_layout.iterator_layout_text_result(
            iterator_layout, min_confidence=min_confidence
        )
        for min_confidence in (20, 30, 40, 50)
    )
    best = OcrTextResult("", None)
    best_score = float("-inf")
    primary_tokens = extracted_text_token_count(primary.text)
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    target_tokens = max(60.0, min(float(primary_tokens), primary_tokens * 0.85))
    for candidate in candidates:
        if not candidate.text:
            continue
        confidence = candidate.confidence if candidate.confidence is not None else 50
        tokens = extracted_text_token_count(candidate.text)
        quality = text_ocr_quality_score(candidate.text)
        score = (
            confidence * 1.25
            + min(tokens, 420) * 0.16
            - quality * 85.0
            - abs(tokens - target_tokens) * 0.025
            + min(25, max(0, confidence - primary_confidence)) * 0.30
        )
        if sparse_text_looks_noisy(candidate.text):
            score -= 15.0
        if score > best_score:
            best = candidate
            best_score = score
    return best


def iterator_symbol_supplement_candidate(
    primary: OcrTextResult,
    line_result: OcrTextResult,
    symbol_result: OcrTextResult,
) -> OcrTextResult:
    if not symbol_result.text:
        return OcrTextResult("", None)
    symbol_tokens = normalized_text_tokens(symbol_result.text)
    if len(symbol_tokens) < 8:
        return OcrTextResult("", None)
    reference_tokens = set(
        normalized_text_tokens(primary.text) + normalized_text_tokens(line_result.text)
    )
    if reference_tokens:
        new_tokens = sum(1 for token in symbol_tokens if token not in reference_tokens)
        if new_tokens < max(2, int(len(symbol_tokens) * 0.08)):
            return OcrTextResult("", None)
    symbol_quality = text_ocr_quality_score(symbol_result.text)
    reference_text = line_result.text or primary.text
    if reference_text:
        reference_quality = text_ocr_quality_score(reference_text)
        if symbol_quality > reference_quality + 0.08:
            return OcrTextResult("", None)
    confidence = symbol_result.confidence
    if confidence is not None:
        comparable_confidences = [
            result.confidence for result in (line_result, primary) if result.confidence is not None
        ]
        if comparable_confidences:
            confidence = min(confidence, max(comparable_confidences) + 3)
    return OcrTextResult(
        symbol_result.text,
        confidence,
        deskew_info=symbol_result.deskew_info,
    )


def select_rendered_sparse_layout_result(
    primary: OcrTextResult, sparse_layout: OcrTextResult
) -> OcrTextResult:
    if not sparse_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(sparse_layout.text)
    if primary_tokens and layout_tokens < max(40, int(primary_tokens * 0.25)):
        return primary
    if primary_tokens and layout_tokens > max(primary_tokens + 120, int(primary_tokens * 1.35)):
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = sparse_layout.confidence if sparse_layout.confidence is not None else 50
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(sparse_layout.text)
    if sparse_text_looks_noisy(sparse_layout.text) and layout_quality >= primary_quality:
        return primary
    coverage_gain = layout_tokens - primary_tokens
    if (
        coverage_gain >= max(25, int(primary_tokens * 0.45))
        and layout_confidence >= primary_confidence - 2
        and layout_quality <= primary_quality + 0.03
    ):
        return sparse_layout
    if layout_confidence >= primary_confidence + 12:
        return sparse_layout
    if layout_confidence >= primary_confidence + 6 and layout_quality <= primary_quality - 0.04:
        return sparse_layout
    if layout_confidence >= 80 and primary_quality >= 0.25 and layout_quality <= primary_quality:
        return sparse_layout
    return primary


def should_try_confident_sparse_layout_ocr(result: OcrTextResult) -> bool:
    if not result.text:
        return False
    tokens = extracted_text_token_count(result.text)
    if not (55 <= tokens <= 180):
        return False
    if numeric_token_ratio(result.text) >= 0.28:
        return False
    confidence = result.confidence
    if confidence is not None and confidence >= 70:
        return False
    return text_ocr_quality_score(result.text) >= 0.18


def select_confident_sparse_layout_result(
    primary: OcrTextResult, confident_layout: OcrTextResult
) -> OcrTextResult:
    if not confident_layout.text:
        return primary
    primary_tokens = extracted_text_token_count(primary.text)
    layout_tokens = extracted_text_token_count(confident_layout.text)
    if layout_tokens < max(25, int(primary_tokens * 0.38)):
        return primary
    primary_quality = text_ocr_quality_score(primary.text)
    layout_quality = text_ocr_quality_score(confident_layout.text)
    if layout_quality > primary_quality - 0.04:
        return primary
    primary_confidence = primary.confidence if primary.confidence is not None else 50
    layout_confidence = (
        confident_layout.confidence if confident_layout.confidence is not None else 50
    )
    if layout_confidence < primary_confidence + 8:
        return primary
    return confident_layout


def select_ocr_text(primary: Any, alternate: Any) -> str:
    return select_ocr_result(primary, alternate).text
