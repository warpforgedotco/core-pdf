# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Mapping, Protocol

from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.ocr import (
    execution as ocr_execution,
)
from core_pdf.impl.engine.extraction.ocr import (
    iterator_layout as ocr_iterator_layout,
)
from core_pdf.impl.engine.extraction.ocr import (
    layout as ocr_layout,
)
from core_pdf.impl.engine.extraction.ocr import (
    selection as ocr_selection,
)
from core_pdf.impl.engine.extraction.ocr.candidates import OcrCandidate
from core_pdf.impl.engine.extraction.ocr.text_analysis import (
    extracted_text_token_count,
    numeric_token_ratio,
    scanned_ocr_artifact_score,
    text_ocr_quality_score,
)
from core_pdf.impl.engine.extraction.ocr.types import (
    OcrImage,
    OcrTextResult,
    leptonica_pix_size_is_supported,
    ocr_int_value,
    ocr_observations_from_rows,
)
from core_pdf.impl.engine.layout.text_lines import is_decorative_leader

OCR_HIGH_DENSITY_IMAGE_DPI = 200
OCR_HIGH_DENSITY_IMAGE_SCALE = 4
OCR_WORD_LAYOUT_SEPARATOR_WORDS = frozenset({"|", "!", "[", "]", "{", "}", "¢"})
OCR_LOW_CONFIDENCE_WORD_REFINEMENT_THRESHOLD = 55
OCR_LOW_CONFIDENCE_WORD_REFINEMENT_MAX_WORDS = 8
OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PADS = (4, 8, 12)
OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PSMS = (8, 13, 7)
OCR_LOW_CONFIDENCE_WORD_SYMBOL_CONFIDENCE = 95
OCR_LINE_ART_TEXT_MASK_MIN_COMPONENTS = 40
OCR_LINE_ART_TEXT_MASK_MAX_COMPONENTS = 4_000
OCR_LINE_ART_TEXT_MASK_MAX_RUN_WIDTH = 160
OCR_LINE_ART_TEXT_MASK_COMPONENT_MAX_WIDTH = 110
OCR_LINE_ART_TEXT_MASK_COMPONENT_MAX_HEIGHT = 120
OCR_LINE_ART_TEXT_MASK_COMPONENT_MAX_AREA = 2_600
OCR_LINE_ART_TEXT_MASK_PSMS = (11, 6)
OCR_LINE_ART_SKIP_COMPACT_LABELS = 10


@dataclass(frozen=True)
class WordRefinement:
    text: str
    confidence: int


class TokenTypeClassifier(Protocol):
    def __call__(self, text: str, /) -> str | None: ...


class OcrTextResultFunction(Protocol):
    def __call__(self, image: OcrImage, timeout: float | None) -> OcrTextResult: ...


class OcrPsmTextResultFunction(Protocol):
    def __call__(
        self,
        image: OcrImage,
        *,
        psm: int,
        timeout: float | None,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> OcrTextResult: ...


class OcrRegionsTextResultsFunction(Protocol):
    def __call__(
        self,
        image: OcrImage,
        requests: list[ocr_execution.RectangleOcrRequest],
        timeout: float | None,
    ) -> list[OcrTextResult]: ...


def line_art_text_mask_ocr_candidates(
    base_candidate: OcrCandidate | None,
    image: OcrImage | None,
    timeout: float | None,
    *,
    ocr_image_to_text_result_with_psm: OcrPsmTextResultFunction,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> list[OcrCandidate]:
    if not should_try_line_art_text_mask_ocr_candidate(base_candidate, image):
        return []
    assert image is not None
    mask = line_art_text_mask_image_for_ocr(image)
    if mask is None:
        return []
    candidates: list[OcrCandidate] = []
    for psm in OCR_LINE_ART_TEXT_MASK_PSMS:
        result = ocr_image_to_text_result_with_psm(
            mask,
            psm=psm,
            timeout=timeout,
            variables={"preserve_interword_spaces": "1"},
        )
        candidate = ocr_candidate_from_image(
            f"line_art_text_mask_psm{psm}",
            result,
            mask,
            token_type_classifier=token_type_classifier,
        )
        if should_keep_line_art_text_mask_ocr_candidate(candidate):
            candidates.append(candidate)
    return candidates


def should_try_line_art_text_mask_ocr_candidate(
    base_candidate: OcrCandidate | None,
    image: OcrImage | None,
) -> bool:
    if base_candidate is None or image is None:
        return False
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return False
    if image.width <= 0 or image.height <= 0:
        return False
    if image.width * image.height < 100_000:
        return False
    if not str(image.source).startswith("full_page_"):
        return False
    text = base_candidate.result.text
    tokens = extracted_text_token_count(text)
    if not (35 <= tokens <= 260):
        return False
    confidence = base_candidate.result.confidence
    if confidence is not None and confidence >= 88 and text_ocr_quality_score(text) <= 0.12:
        return False
    if (
        ocr_selection.ocr_candidate_score(base_candidate) >= 78.0
        and (confidence or 0) >= 55
        and text_ocr_quality_score(text) <= 0.18
        and compact_uppercase_label_count(text) >= OCR_LINE_ART_SKIP_COMPACT_LABELS
    ):
        return False
    return True


def should_keep_line_art_text_mask_ocr_candidate(candidate: OcrCandidate) -> bool:
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if not (8 <= tokens <= 320):
        return False
    confidence = candidate.result.confidence if candidate.result.confidence is not None else 0
    if confidence < 35:
        return False
    return text_ocr_quality_score(text) <= 0.38


def line_art_text_mask_image_for_ocr(image: OcrImage) -> OcrImage | None:
    components, runs, parent = line_art_text_mask_components(image)
    if not (
        OCR_LINE_ART_TEXT_MASK_MIN_COMPONENTS
        <= len(components)
        <= OCR_LINE_ART_TEXT_MASK_MAX_COMPONENTS
    ):
        return None
    keep_roots = line_art_text_mask_keep_roots(components)
    if len(keep_roots) < OCR_LINE_ART_TEXT_MASK_MIN_COMPONENTS:
        return None
    output = bytearray(b"\xff") * (image.width * image.height)
    for index, (y, x0, x1) in enumerate(runs):
        if line_art_text_mask_find(parent, index) not in keep_roots:
            continue
        start = y * image.width + x0
        output[start : start + (x1 - x0 + 1)] = b"\0" * (x1 - x0 + 1)
    return replace(
        image,
        data=bytes(output),
        bytes_per_pixel=1,
        bytes_per_line=image.width,
        encoded=None,
        source=f"{image.source}_line_art_text_mask",
        resolution=image.resolution or 300,
    )


def line_art_text_mask_components(
    image: OcrImage,
) -> tuple[dict[int, list[int]], list[tuple[int, int, int]], list[int]]:
    parent: list[int] = []
    rank: list[int] = []
    runs: list[tuple[int, int, int]] = []
    previous_row: list[int] = []

    def add_run(y: int, x0: int, x1: int) -> int:
        index = len(runs)
        parent.append(index)
        rank.append(0)
        runs.append((y, x0, x1))
        return index

    for y in range(image.height):
        row_offset = y * image.bytes_per_line
        current_row: list[int] = []
        x = 0
        while x < image.width:
            offset = row_offset + x * image.bytes_per_pixel
            if not line_art_text_mask_pixel_is_foreground(image, offset):
                x += 1
                continue
            x0 = x
            x += 1
            while x < image.width:
                offset = row_offset + x * image.bytes_per_pixel
                if not line_art_text_mask_pixel_is_foreground(image, offset):
                    break
                x += 1
            x1 = x - 1
            if x1 - x0 + 1 > OCR_LINE_ART_TEXT_MASK_MAX_RUN_WIDTH:
                continue
            index = add_run(y, x0, x1)
            current_row.append(index)
            for previous in previous_row:
                _py, px0, px1 = runs[previous]
                if px1 < x0 - 1:
                    continue
                if px0 > x1 + 1:
                    break
                line_art_text_mask_union(parent, rank, index, previous)
        previous_row = current_row

    components: dict[int, list[int]] = {}
    for index, (y, x0, x1) in enumerate(runs):
        root = line_art_text_mask_find(parent, index)
        component = components.get(root)
        area = x1 - x0 + 1
        if component is None:
            components[root] = [x0, y, x1, y, area]
        else:
            component[0] = min(component[0], x0)
            component[1] = min(component[1], y)
            component[2] = max(component[2], x1)
            component[3] = max(component[3], y)
            component[4] += area
    return components, runs, parent


def line_art_text_mask_find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def line_art_text_mask_union(
    parent: list[int],
    rank: list[int],
    left: int,
    right: int,
) -> None:
    left_root = line_art_text_mask_find(parent, left)
    right_root = line_art_text_mask_find(parent, right)
    if left_root == right_root:
        return
    if rank[left_root] < rank[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    if rank[left_root] == rank[right_root]:
        rank[left_root] += 1


def line_art_text_mask_pixel_is_foreground(image: OcrImage, offset: int) -> bool:
    if image.bytes_per_pixel == 1:
        return image.data[offset] < 190
    if image.bytes_per_pixel == 4 and image.data[offset + 3] <= 16:
        return False
    return (
        image.data[offset] * 30 + image.data[offset + 1] * 59 + image.data[offset + 2] * 11
    ) < 19_000


def line_art_text_mask_keep_roots(
    components: dict[int, list[int]],
) -> set[int]:
    keep: set[int] = set()
    for root, (x0, y0, x1, y1, area) in components.items():
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        if area < 4:
            continue
        if width < 2 or height < 5:
            continue
        if (
            width > OCR_LINE_ART_TEXT_MASK_COMPONENT_MAX_WIDTH
            or height > OCR_LINE_ART_TEXT_MASK_COMPONENT_MAX_HEIGHT
        ):
            continue
        if area > OCR_LINE_ART_TEXT_MASK_COMPONENT_MAX_AREA:
            continue
        if width > 80 and height < 12:
            continue
        if height > 95 and width < 8:
            continue
        keep.add(root)
    return keep


def compact_uppercase_label_count(text: str) -> int:
    count = 0
    for raw in text.split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if not (2 <= len(token) <= 8):
            continue
        if not token.isalpha():
            continue
        if token.upper() != token:
            continue
        count += 1
    return count


def rendered_downsampled_page_ocr_candidate(
    page: Any,
    dpi: int,
    base_candidate: OcrCandidate,
    base_image: OcrImage,
    timeout: float | None,
    *,
    image_dominant: bool = False,
    ocr_image_to_text_result: OcrTextResultFunction | None = None,
    ocr_image_to_text_result_with_psm: OcrPsmTextResultFunction | None = None,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrCandidate | None:
    if not should_try_rendered_downsampled_page_ocr_candidate(
        base_candidate,
        dpi,
        image_dominant=image_dominant,
    ):
        return None
    source = f"rendered_page_{dpi}dpi_downsample_dark"
    target_resolution = max(1, (base_image.resolution or dpi) // 2)
    image = downsample_ocr_image_dark_2x(
        base_image,
        source=source,
        resolution=target_resolution,
    )
    if image is None:
        return None
    if image_dominant and ocr_image_to_text_result_with_psm is not None:
        result = ocr_image_to_text_result_with_psm(
            image,
            psm=3,
            timeout=timeout,
        )
    elif ocr_image_to_text_result is not None:
        result = ocr_image_to_text_result(image, timeout)
    else:
        return None
    candidate = ocr_candidate_from_image(
        source,
        result,
        image,
        token_type_classifier=token_type_classifier,
    )
    if not should_keep_rendered_downsampled_page_ocr_candidate(
        base_candidate,
        candidate,
    ):
        return None
    return candidate


def should_try_rendered_downsampled_page_ocr_candidate(
    candidate: OcrCandidate,
    dpi: int,
    *,
    image_dominant: bool = False,
) -> bool:
    if dpi not in {300, 400}:
        return False
    if not candidate.name.startswith("rendered_page_"):
        return False
    if candidate.name.endswith("_tiled"):
        return False
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if not (240 <= tokens <= 900):
        return False
    if numeric_token_ratio(text) >= 0.25:
        return False
    confidence = candidate.result.confidence if candidate.result.confidence is not None else 50
    if confidence < 68:
        return False
    quality = text_ocr_quality_score(text)
    if quality > 0.14:
        return False
    if dpi == 300 and not image_dominant:
        return False
    if image_dominant:
        return confidence >= 80 and tokens >= 240 and quality <= 0.10
    return True


def should_keep_rendered_downsampled_page_ocr_candidate(
    base: OcrCandidate,
    candidate: OcrCandidate,
) -> bool:
    if not candidate.result.text:
        return False
    base_tokens = extracted_text_token_count(base.result.text)
    tokens = extracted_text_token_count(candidate.result.text)
    if base_tokens >= 80 and tokens < int(base_tokens * 0.90):
        return False
    if tokens > max(base_tokens + 80, int(base_tokens * 1.15)):
        return False
    base_confidence = base.result.confidence if base.result.confidence is not None else 50
    confidence = candidate.result.confidence if candidate.result.confidence is not None else 50
    if confidence + 3 < base_confidence:
        return False
    return True


def rendered_sparse_page_ocr_candidate(
    base_candidate: OcrCandidate,
    image: OcrImage,
    timeout: float | None,
    *,
    psm: int,
    ocr_image_to_text_result_with_psm: OcrPsmTextResultFunction,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrCandidate | None:
    if not should_try_rendered_sparse_page_ocr_candidate(base_candidate, image):
        return None
    source = f"{base_candidate.name}_sparse"
    sparse_image = replace(image, source=source)
    result = ocr_image_to_text_result_with_psm(
        sparse_image,
        psm=psm,
        timeout=timeout,
    )
    candidate = ocr_candidate_from_image(
        source,
        result,
        sparse_image,
        token_type_classifier=token_type_classifier,
    )
    if not should_keep_rendered_sparse_page_ocr_candidate(base_candidate, candidate):
        return None
    return candidate


def should_try_rendered_sparse_page_ocr_candidate(
    candidate: OcrCandidate,
    image: OcrImage,
) -> bool:
    if not candidate.name.startswith("rendered_page_"):
        return False
    if "_tile_" in candidate.name or candidate.name.endswith("_sparse"):
        return False
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return False
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if tokens < 180:
        return False
    if numeric_token_ratio(text) >= 0.45:
        return False
    confidence = candidate.result.confidence if candidate.result.confidence is not None else 50
    quality = text_ocr_quality_score(text)
    return confidence < 88 or quality >= 0.10 or tokens < 850


def should_keep_rendered_sparse_page_ocr_candidate(
    base_candidate: OcrCandidate,
    sparse_candidate: OcrCandidate,
) -> bool:
    sparse_text = sparse_candidate.result.text
    if not sparse_text:
        return False
    sparse_tokens = extracted_text_token_count(sparse_text)
    if sparse_tokens < 120:
        return False
    base_text = base_candidate.result.text
    base_tokens = extracted_text_token_count(base_text)
    if sparse_tokens < int(base_tokens * 0.72):
        return False
    sparse_confidence = sparse_candidate.result.confidence or 0
    base_confidence = base_candidate.result.confidence or 0
    sparse_quality = text_ocr_quality_score(sparse_text)
    base_quality = text_ocr_quality_score(base_text)
    if sparse_tokens >= int(base_tokens * 1.08) and sparse_quality <= base_quality + 0.12:
        return True
    if sparse_confidence >= base_confidence - 8 and sparse_quality + 0.04 < base_quality:
        return True
    return (
        ocr_selection.ocr_candidate_score(sparse_candidate)
        >= ocr_selection.ocr_candidate_score(base_candidate) - 4.0
    )


def word_layout_ocr_candidate(base_candidate: OcrCandidate) -> OcrCandidate | None:
    if not should_try_word_layout_ocr_candidate(base_candidate):
        return None
    text = rebuild_ocr_text_from_word_layout(base_candidate.result.word_rows)
    if not should_keep_word_layout_ocr_candidate(base_candidate, text):
        return None
    result = OcrTextResult(
        text,
        base_candidate.result.confidence,
        line_rows=base_candidate.result.line_rows,
        word_rows=base_candidate.result.word_rows,
        symbol_rows=base_candidate.result.symbol_rows,
    )
    return OcrCandidate(
        f"{base_candidate.name}_word_layout",
        result,
        bbox=base_candidate.bbox,
        region_count=base_candidate.region_count,
        image_width=base_candidate.image_width,
        image_height=base_candidate.image_height,
        image_resolution=base_candidate.image_resolution,
        page_width=base_candidate.page_width,
        page_height=base_candidate.page_height,
        page_bbox=base_candidate.page_bbox,
    )


def reconciled_layout_ocr_candidate(
    base_candidate: OcrCandidate,
) -> OcrCandidate | None:
    if not should_try_reconciled_layout_ocr_candidate(base_candidate):
        return None
    result = ocr_iterator_layout.reconciled_iterator_text_result_from_existing_result(
        base_candidate.result
    )
    if not should_keep_reconciled_layout_ocr_candidate(base_candidate, result):
        return None
    return OcrCandidate(
        f"{base_candidate.name}_reconciled_layout",
        result,
        bbox=base_candidate.bbox,
        region_count=base_candidate.region_count,
        image_width=base_candidate.image_width,
        image_height=base_candidate.image_height,
        image_resolution=base_candidate.image_resolution,
        page_width=base_candidate.page_width,
        page_height=base_candidate.page_height,
        page_bbox=base_candidate.page_bbox,
    )


def should_try_reconciled_layout_ocr_candidate(candidate: OcrCandidate) -> bool:
    result = candidate.result
    if len(result.line_rows) < 4 or len(result.word_rows) < 8:
        return False
    if len(result.symbol_rows) < 8:
        return False
    tokens = extracted_text_token_count(result.text)
    return 20 <= tokens <= 2_500


def should_keep_reconciled_layout_ocr_candidate(
    base_candidate: OcrCandidate,
    result: OcrTextResult,
) -> bool:
    text = result.text
    if not text or text == base_candidate.result.text:
        return False
    base_text = base_candidate.result.text
    base_tokens = extracted_text_token_count(base_text)
    tokens = extracted_text_token_count(text)
    if base_tokens and tokens < int(base_tokens * 0.85):
        return False
    if tokens > max(base_tokens + 80, int(base_tokens * 1.18)):
        return False
    base_quality = text_ocr_quality_score(base_text)
    quality = text_ocr_quality_score(text)
    base_artifact = scanned_ocr_artifact_score(base_text)
    artifact = scanned_ocr_artifact_score(text)
    if quality <= base_quality + 0.015 and artifact <= base_artifact + 0.015:
        return True
    return quality + 0.02 < base_quality


def low_confidence_word_refinement_ocr_candidate(
    base_candidate: OcrCandidate,
    image: OcrImage,
    timeout: float | None,
    *,
    ocr_image_to_text_result_with_psm: OcrPsmTextResultFunction,
    ocr_image_regions_to_text_results: OcrRegionsTextResultsFunction | None = None,
) -> OcrCandidate | None:
    if not should_try_low_confidence_word_refinement_ocr_candidate(
        base_candidate,
        image,
    ):
        return None
    line_rows = list(base_candidate.result.line_rows)
    word_rows = [dict(row) for row in base_candidate.result.word_rows]
    symbol_rows = tuple(base_candidate.result.symbol_rows)
    symbol_rows_by_line = ocr_iterator_layout.iterator_rows_by_line_key(symbol_rows)
    refinements = (
        batched_low_confidence_word_refinements(
            word_rows,
            symbol_rows_by_line,
            image,
            timeout,
            ocr_image_regions_to_text_results=ocr_image_regions_to_text_results,
        )
        if ocr_image_regions_to_text_results is not None
        else {}
    )
    changed_word_indexes: set[int] = set()
    refined_count = 0
    for word_index, row in enumerate(word_rows):
        if not low_confidence_word_row_should_be_refined(row):
            continue
        line_key = ocr_iterator_layout.iterator_line_key(row)
        line_symbol_rows = symbol_rows_by_line.get(line_key, ())
        word_symbol_rows = tuple(
            symbol_row
            for symbol_row in line_symbol_rows
            if ocr_int_value(symbol_row.get("word_num", 0)) == ocr_int_value(row.get("word_num", 0))
        )
        refinement = refinements.get(word_index)
        if refinement is None:
            refinement = refine_low_confidence_word_row(
                row,
                word_symbol_rows,
                image,
                timeout,
                ocr_image_to_text_result_with_psm=ocr_image_to_text_result_with_psm,
            )
        if refinement is None:
            continue
        row["text"] = refinement.text
        row["conf"] = refinement.confidence
        changed_word_indexes.add(word_index)
        refined_count += 1
        if refined_count >= OCR_LOW_CONFIDENCE_WORD_REFINEMENT_MAX_WORDS:
            break
    if not changed_word_indexes:
        return None
    changed_lines = {
        ocr_iterator_layout.iterator_line_key(word_rows[index]) for index in changed_word_indexes
    }
    word_rows_by_line = ocr_iterator_layout.iterator_rows_by_line_key(tuple(word_rows))
    refined_line_rows: list[dict[str, Any]] = []
    for line_row in line_rows:
        line_key = ocr_iterator_layout.iterator_line_key(line_row)
        if line_key not in changed_lines:
            refined_line_rows.append(line_row)
            continue
        line_word_rows = word_rows_by_line.get(line_key, ())
        text = rebuild_ocr_line_text_from_word_rows(line_word_rows)
        if not text:
            refined_line_rows.append(line_row)
            continue
        row = ocr_iterator_layout.iterator_line_row_with_text(
            line_row,
            text,
            line_word_rows,
        )
        confidence = ocr_iterator_layout.iterator_rows_confidence(line_word_rows)
        if confidence is not None:
            row["conf"] = confidence
        refined_line_rows.append(row)
    result = ocr_iterator_layout.iterator_rows_text_result(refined_line_rows)
    refined_result = ocr_layout.geometry_rendered_ocr_result(
        OcrTextResult(
            result.text,
            result.confidence,
            line_rows=tuple(refined_line_rows),
            word_rows=tuple(word_rows),
            symbol_rows=symbol_rows,
            observations=ocr_observations_from_rows([*refined_line_rows, *word_rows, *symbol_rows]),
        )
    )
    if not should_keep_low_confidence_word_refinement_ocr_candidate(
        base_candidate,
        refined_result,
    ):
        return None
    return OcrCandidate(
        f"{base_candidate.name}_word_refined",
        refined_result,
        bbox=base_candidate.bbox,
        region_count=base_candidate.region_count,
        image_width=base_candidate.image_width,
        image_height=base_candidate.image_height,
        image_resolution=base_candidate.image_resolution,
        page_width=base_candidate.page_width,
        page_height=base_candidate.page_height,
        page_bbox=base_candidate.page_bbox,
    )


def batched_low_confidence_word_refinements(
    word_rows: list[dict[str, Any]],
    symbol_rows_by_line: dict[tuple[int, int, int, int], tuple[dict[str, Any], ...]],
    image: OcrImage,
    timeout: float | None,
    *,
    ocr_image_regions_to_text_results: OcrRegionsTextResultsFunction,
) -> dict[int, WordRefinement]:
    candidates: list[
        tuple[
            int,
            dict[str, Any],
            tuple[dict[str, Any], ...],
            str,
            int,
            float,
        ]
    ] = []
    requests: list[ocr_execution.RectangleOcrRequest] = []
    request_metadata: list[tuple[int, str, tuple[dict[str, Any], ...]]] = []
    for word_index, row in enumerate(word_rows):
        if not low_confidence_word_row_should_be_refined(row):
            continue
        line_key = ocr_iterator_layout.iterator_line_key(row)
        line_symbol_rows = symbol_rows_by_line.get(line_key, ())
        word_symbol_rows = tuple(
            symbol_row
            for symbol_row in line_symbol_rows
            if int(symbol_row.get("word_num", 0)) == int(row.get("word_num", 0))
        )
        base_text = str(row.get("text", "")).strip()
        base_confidence = ocr_iterator_layout.iterator_row_confidence(row) or 0
        base_best = WordRefinement(base_text, base_confidence)
        candidates.append(
            (
                word_index,
                row,
                word_symbol_rows,
                base_text,
                base_confidence,
                word_refinement_score(base_best, base_text, word_symbol_rows),
            )
        )
        for pad in OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PADS:
            rectangle = word_refinement_rectangle(row, pad)
            if rectangle is None:
                continue
            requests.extend(
                ocr_execution.RectangleOcrRequest(
                    rectangle=rectangle,
                    psm=psm,
                    variables={},
                )
                for psm in OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PSMS
            )
            request_metadata.extend(
                (word_index, base_text, word_symbol_rows)
                for _psm in OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PSMS
            )
        if len(candidates) >= OCR_LOW_CONFIDENCE_WORD_REFINEMENT_MAX_WORDS:
            break
    if not requests:
        return {}
    results = ocr_image_regions_to_text_results(image, requests, timeout)
    best_by_index: dict[int, tuple[WordRefinement, float]] = {
        index: (WordRefinement(base_text, base_confidence), best_score)
        for index, _row, _symbols, base_text, base_confidence, best_score in candidates
    }
    for result, metadata in zip(results, request_metadata, strict=False):
        word_index, base_text, word_symbol_rows = metadata
        text = normalized_refined_word_text(result.text, base_text)
        if not text:
            continue
        refinement = WordRefinement(text, result.confidence or 0)
        score = word_refinement_score(refinement, base_text, word_symbol_rows)
        best_refinement, best_score = best_by_index[word_index]
        if score > best_score:
            best_by_index[word_index] = (refinement, score)
    accepted: dict[int, WordRefinement] = {}
    for (
        index,
        _row,
        word_symbol_rows,
        base_text,
        base_confidence,
        _best_score,
    ) in candidates:
        best_refinement, best_score = best_by_index[index]
        if best_refinement.text == base_text:
            continue
        base_score = word_refinement_score(
            WordRefinement(base_text, base_confidence),
            base_text,
            word_symbol_rows,
        )
        if best_score >= base_score + 8.0:
            accepted[index] = best_refinement
    return accepted


def should_try_low_confidence_word_refinement_ocr_candidate(
    candidate: OcrCandidate,
    image: OcrImage,
) -> bool:
    if candidate.name.endswith(("_word_refined", "_reconciled_layout", "_word_layout")):
        return False
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return False
    if candidate.image_width != image.width or candidate.image_height != image.height:
        return False
    result = candidate.result
    if not result.line_rows or not result.word_rows or not result.symbol_rows:
        return False
    tokens = extracted_text_token_count(result.text)
    if tokens < 20 or tokens > 2_500:
        return False
    return any(low_confidence_word_row_should_be_refined(row) for row in result.word_rows)


def low_confidence_word_row_should_be_refined(row: dict[str, Any]) -> bool:
    confidence = ocr_iterator_layout.iterator_row_confidence(row)
    if confidence is None:
        return False
    if confidence >= OCR_LOW_CONFIDENCE_WORD_REFINEMENT_THRESHOLD:
        return False
    text = str(row.get("text", "")).strip()
    if len(text) < 4:
        return False
    try:
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return False
    return width >= 18 and height >= 8


def refine_low_confidence_word_row(
    row: dict[str, Any],
    symbol_rows: tuple[dict[str, Any], ...],
    image: OcrImage,
    timeout: float | None,
    *,
    ocr_image_to_text_result_with_psm: OcrPsmTextResultFunction,
) -> WordRefinement | None:
    base_text = str(row.get("text", "")).strip()
    base_confidence = ocr_iterator_layout.iterator_row_confidence(row) or 0
    best = WordRefinement(base_text, base_confidence)
    best_score = word_refinement_score(best, base_text, symbol_rows)
    for pad in OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PADS:
        crop = crop_word_refinement_image(image, row, pad)
        if crop is None:
            continue
        for psm in OCR_LOW_CONFIDENCE_WORD_REFINEMENT_PSMS:
            result = ocr_image_to_text_result_with_psm(
                crop,
                psm=psm,
                timeout=timeout,
                variables={},
            )
            text = normalized_refined_word_text(result.text, base_text)
            if not text:
                continue
            refinement = WordRefinement(text, result.confidence or 0)
            score = word_refinement_score(refinement, base_text, symbol_rows)
            if score > best_score:
                best = refinement
                best_score = score
    if best.text == base_text:
        return None
    if (
        best_score
        < word_refinement_score(
            WordRefinement(base_text, base_confidence),
            base_text,
            symbol_rows,
        )
        + 8.0
    ):
        return None
    return best


def crop_word_refinement_image(
    image: OcrImage,
    row: dict[str, Any],
    pad: int,
) -> OcrImage | None:
    rectangle = word_refinement_rectangle(row, pad)
    if rectangle is None:
        return None
    return ocr_execution.crop_ocr_image_region(
        image,
        rectangle,
    )


def word_refinement_rectangle(
    row: dict[str, Any],
    pad: int,
) -> tuple[int, int, int, int] | None:
    try:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return (
        left - pad,
        top - pad,
        left + width + pad,
        top + height + pad,
    )


def normalized_refined_word_text(text: str, base_text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    parts = stripped.split()
    if len(parts) <= 1:
        return stripped
    if " " not in base_text:
        return "".join(parts)
    return " ".join(parts)


def word_refinement_score(
    refinement: WordRefinement,
    base_text: str,
    symbol_rows: tuple[dict[str, Any], ...],
) -> float:
    text = refinement.text
    score = float(refinement.confidence)
    score += high_confidence_symbol_agreement_score(text, symbol_rows)
    score += word_refinement_length_score(text, base_text)
    score -= text_ocr_quality_score(text) * 45.0
    score -= scanned_ocr_artifact_score(text) * 25.0
    return score


def high_confidence_symbol_agreement_score(
    text: str,
    symbol_rows: tuple[dict[str, Any], ...],
) -> float:
    if not symbol_rows:
        return 0.0
    score = 0.0
    for index, row in enumerate(
        sorted(symbol_rows, key=lambda item: int(item.get("symbol_num", 0)))
    ):
        confidence = ocr_iterator_layout.iterator_row_confidence(row) or 0
        if confidence < OCR_LOW_CONFIDENCE_WORD_SYMBOL_CONFIDENCE:
            continue
        expected = str(row.get("text", "")).strip()
        if len(expected) != 1:
            continue
        if index < len(text) and text[index].casefold() == expected.casefold():
            score += 1.0
        else:
            score -= 20.0
    return score


def word_refinement_length_score(text: str, base_text: str) -> float:
    if not base_text:
        return 0.0
    delta = abs(len(text) - len(base_text))
    if delta == 0:
        return 4.0
    return -8.0 * delta


def rebuild_ocr_line_text_from_word_rows(
    rows: tuple[dict[str, Any], ...],
) -> str:
    words = []
    for row_index, row in enumerate(rows):
        word = ocr_layout.ocr_layout_word(row, row_index=row_index)
        if word is not None:
            words.append(word)
    if not words:
        return ""
    return ocr_layout.render_ocr_word_line(
        sorted(words, key=lambda word: (word.x0, word.word_num, word.row_index))
    )


def should_keep_low_confidence_word_refinement_ocr_candidate(
    base_candidate: OcrCandidate,
    result: OcrTextResult,
) -> bool:
    text = result.text
    if not text or text == base_candidate.result.text:
        return False
    base_text = base_candidate.result.text
    base_tokens = extracted_text_token_count(base_text)
    tokens = extracted_text_token_count(text)
    if base_tokens and tokens < int(base_tokens * 0.92):
        return False
    if tokens > max(base_tokens + 50, int(base_tokens * 1.12)):
        return False
    base_quality = text_ocr_quality_score(base_text)
    quality = text_ocr_quality_score(text)
    base_artifact = scanned_ocr_artifact_score(base_text)
    artifact = scanned_ocr_artifact_score(text)
    if quality <= base_quality + 0.015 and artifact <= base_artifact + 0.015:
        return True
    return quality + 0.02 < base_quality and artifact <= base_artifact + 0.025


def should_try_word_layout_ocr_candidate(candidate: OcrCandidate) -> bool:
    if not should_generate_layout_variants(candidate):
        return False
    if candidate.name != "full_page_image" and candidate.name != "high_density_full_page_image":
        return False
    text = candidate.result.text
    tokens = extracted_text_token_count(text)
    if not (120 <= tokens <= 1_500):
        return False
    word_rows = candidate.result.word_rows
    if len(word_rows) < 40:
        return False
    separator_words = 0
    decorative_leaders = 0
    for row in word_rows:
        word = str(row.get("text") or "").strip()
        if not word:
            continue
        if word in OCR_WORD_LAYOUT_SEPARATOR_WORDS:
            separator_words += 1
        elif is_decorative_leader(word):
            decorative_leaders += 1
    return separator_words >= 2 or decorative_leaders >= 2


def rebuild_ocr_text_from_word_layout(
    word_rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> str:
    words = []
    heights: list[float] = []
    for row in word_rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if text in OCR_WORD_LAYOUT_SEPARATOR_WORDS or is_decorative_leader(text):
            continue
        try:
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        words.append((top, left, text, height))
        heights.append(height)
    if not words:
        return ""
    line_threshold = max(1.0, median(heights) * 0.55)
    words.sort(key=lambda item: (item[0], item[1], item[2]))
    grouped: list[list[tuple[float, float, str, float]]] = []
    current: list[tuple[float, float, str, float]] = []
    current_center = 0.0
    for word in words:
        center = word[0] + word[3] * 0.5
        if not current:
            current = [word]
            current_center = center
            continue
        if abs(center - current_center) <= line_threshold:
            current.append(word)
            current_center = (current_center * (len(current) - 1) + center) / len(current)
        else:
            grouped.append(current)
            current = [word]
            current_center = center
    if current:
        grouped.append(current)
    lines: list[str] = []
    for line_words in grouped:
        line_text = " ".join(
            text
            for ignored_top, ignored_left, text, ignored_height in sorted(
                line_words, key=lambda item: item[1]
            )
        ).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines)


def should_keep_word_layout_ocr_candidate(base_candidate: OcrCandidate, text: str) -> bool:
    if not text:
        return False
    base_text = base_candidate.result.text
    base_tokens = extracted_text_token_count(base_text)
    tokens = extracted_text_token_count(text)
    if tokens < int(base_tokens * 0.90):
        return False
    if tokens > max(base_tokens + 40, int(base_tokens * 1.10)):
        return False
    base_quality = text_ocr_quality_score(base_text)
    quality = text_ocr_quality_score(text)
    base_artifact = scanned_ocr_artifact_score(base_text)
    artifact = scanned_ocr_artifact_score(text)
    if quality <= base_quality + 0.02 and artifact + 0.01 < base_artifact:
        return True
    return quality + 0.01 < base_quality and artifact <= base_artifact + 0.02


def should_generate_layout_variants(candidate: OcrCandidate) -> bool:
    text = candidate.result.text
    if not text:
        return False
    tokens = extracted_text_token_count(text)
    if tokens < 20:
        return False
    confidence = candidate.result.confidence or 0
    quality = text_ocr_quality_score(text)
    artifact = scanned_ocr_artifact_score(text)
    if confidence >= 88 and tokens >= 80 and quality <= 0.10 and artifact <= 0.05:
        return False
    return True


def ocr_candidate_from_image(
    name: str,
    result: OcrTextResult,
    image: OcrImage,
    *,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrCandidate:
    geometry = page_geometry.ImageSpace.from_ocr_image(image, source=name)
    result = ocr_result_with_page_observations(
        result,
        source=name,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        page_width=geometry.page_width,
        page_height=geometry.page_height,
        page_bbox=geometry.page_bbox,
        clockwise_quarter_turns=geometry.clockwise_quarter_turns,
        token_type_classifier=token_type_classifier,
    )
    return OcrCandidate(
        name,
        result,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        page_width=geometry.page_width,
        page_height=geometry.page_height,
        page_bbox=geometry.page_bbox,
    )


def ocr_result_with_page_observations(
    result: OcrTextResult,
    *,
    source: str,
    image_width: int | None,
    image_height: int | None,
    image_resolution: int | None,
    page_width: float | None = None,
    page_height: float | None = None,
    page_bbox: tuple[float, float, float, float] | None = None,
    clockwise_quarter_turns: int = 0,
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrTextResult:
    geometry = page_geometry.ImageSpace.from_dimensions(
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
        page_width=page_width,
        page_height=page_height,
        page_bbox=page_bbox,
        clockwise_quarter_turns=clockwise_quarter_turns,
        source=source,
    )
    line_rows = tuple(
        page_geometry.normalize_ocr_row_to_page(
            row,
            geometry,
            token_type_classifier=token_type_classifier,
        )
        for row in result.line_rows
    )
    word_rows = tuple(
        page_geometry.normalize_ocr_row_to_page(
            row,
            geometry,
            token_type_classifier=token_type_classifier,
        )
        for row in result.word_rows
    )
    symbol_rows = tuple(
        page_geometry.normalize_ocr_row_to_page(
            row,
            geometry,
            token_type_classifier=token_type_classifier,
        )
        for row in result.symbol_rows
    )
    row_observations = ocr_observations_from_rows(
        (*line_rows, *word_rows, *symbol_rows),
        source=source,
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
    )
    if row_observations:
        observations = tuple(
            page_geometry.annotate_ocr_observation_page_geometry(
                observation,
                geometry,
                source=source,
            )
            for observation in row_observations
        )
    else:
        observations = tuple(
            page_geometry.normalize_ocr_observation_to_page(
                observation,
                geometry,
                source=source,
                token_type_classifier=token_type_classifier,
            )
            for observation in result.observations
        )
    normalized = OcrTextResult(
        result.text,
        result.confidence,
        line_rows=line_rows,
        word_rows=word_rows,
        symbol_rows=symbol_rows,
        component_boxes=result.component_boxes,
        observations=observations,
    )
    return ocr_layout.geometry_rendered_ocr_result(normalized)


def high_density_full_page_image_for_ocr(image: OcrImage) -> OcrImage | None:
    if image.width <= 0 or image.height <= 0:
        return None
    target_width = max(1, image.width * OCR_HIGH_DENSITY_IMAGE_SCALE)
    target_height = max(1, image.height * OCR_HIGH_DENSITY_IMAGE_SCALE)
    if not leptonica_pix_size_is_supported(target_width, target_height):
        return None
    current_target_width = image.target_width or image.width
    current_target_height = image.target_height or image.height
    if target_width <= current_target_width or target_height <= current_target_height:
        return None
    return OcrImage(
        data=image.data,
        width=image.width,
        height=image.height,
        bytes_per_pixel=image.bytes_per_pixel,
        bytes_per_line=image.bytes_per_line,
        encoded=image.encoded,
        source=f"{image.source}_high_density",
        target_width=target_width,
        target_height=target_height,
        resolution=OCR_HIGH_DENSITY_IMAGE_DPI,
        clockwise_quarter_turns=image.clockwise_quarter_turns,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
    )


def downsample_ocr_image_dark_2x(
    image: OcrImage,
    *,
    source: str,
    resolution: int,
) -> OcrImage | None:
    if image.bytes_per_pixel != 4 or not image.data:
        return None
    if image.width < 2 or image.height < 2 or image.bytes_per_line < image.width * 4:
        return None
    target_width = image.width // 2
    target_height = image.height // 2
    required_size = (target_height * 2 - 1) * image.bytes_per_line + target_width * 2 * 4
    if target_width <= 0 or target_height <= 0 or len(image.data) < required_size:
        return None
    output = bytearray(target_width * target_height * 4)
    out = 0
    for y in range(target_height):
        source_y = y * 2
        for x in range(target_width):
            source_x = x * 2
            min_red = 255
            min_green = 255
            min_blue = 255
            for dy in (0, 1):
                row = (source_y + dy) * image.bytes_per_line
                for dx in (0, 1):
                    offset = row + (source_x + dx) * 4
                    red = image.data[offset]
                    green = image.data[offset + 1]
                    blue = image.data[offset + 2]
                    alpha = image.data[offset + 3]
                    if alpha < 255:
                        red = (red * alpha + 255 * (255 - alpha)) // 255
                        green = (green * alpha + 255 * (255 - alpha)) // 255
                        blue = (blue * alpha + 255 * (255 - alpha)) // 255
                    min_red = min(min_red, red)
                    min_green = min(min_green, green)
                    min_blue = min(min_blue, blue)
            output[out] = min_red
            output[out + 1] = min_green
            output[out + 2] = min_blue
            output[out + 3] = 255
            out += 4
    return OcrImage(
        bytes(output),
        target_width,
        target_height,
        4,
        target_width * 4,
        source=source,
        resolution=resolution,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
    )


def darken_ocr_image_min_3x3(
    image: OcrImage,
    *,
    source: str,
    resolution: int,
) -> OcrImage | None:
    if image.bytes_per_pixel not in {3, 4} or not image.data:
        return None
    if image.width <= 0 or image.height <= 0:
        return None
    if image.bytes_per_line < image.width * image.bytes_per_pixel:
        return None
    required_size = (image.height - 1) * image.bytes_per_line + image.width * image.bytes_per_pixel
    if len(image.data) < required_size:
        return None
    data = image.data
    stride = image.bytes_per_line
    width = image.width
    height = image.height

    has_alpha = False
    if image.bytes_per_pixel == 4:
        for y in range(height):
            row = y * stride
            for x in range(width):
                if data[row + x * 4 + 3] < 255:
                    has_alpha = True
                    break
            if has_alpha:
                break

    source_data: bytes | bytearray
    if has_alpha:
        rgb = bytearray(width * height * 3)
        out = 0
        for y in range(height):
            row = y * stride
            for x in range(width):
                offset = row + x * 4
                red = data[offset]
                green = data[offset + 1]
                blue = data[offset + 2]
                alpha = data[offset + 3]
                if alpha < 255:
                    inverse_alpha = 255 - alpha
                    red = (red * alpha + 255 * inverse_alpha) // 255
                    green = (green * alpha + 255 * inverse_alpha) // 255
                    blue = (blue * alpha + 255 * inverse_alpha) // 255
                rgb[out] = red
                rgb[out + 1] = green
                rgb[out + 2] = blue
                out += 3
        source_data = rgb
        source_stride = width * 3
        source_bpp = 3
    else:
        source_data = data
        source_stride = stride
        source_bpp = image.bytes_per_pixel

    row_min = bytearray(width * height * 3)
    for y in range(height):
        row = y * source_stride
        out = y * width * 3
        for x in range(width):
            x0 = x - 1 if x > 0 else x
            x1 = x
            x2 = x + 1 if x + 1 < width else x
            o0 = row + x0 * source_bpp
            o1 = row + x1 * source_bpp
            o2 = row + x2 * source_bpp
            min_red = source_data[o0]
            red = source_data[o1]
            if red < min_red:
                min_red = red
            red = source_data[o2]
            if red < min_red:
                min_red = red
            min_green = source_data[o0 + 1]
            green = source_data[o1 + 1]
            if green < min_green:
                min_green = green
            green = source_data[o2 + 1]
            if green < min_green:
                min_green = green
            min_blue = source_data[o0 + 2]
            blue = source_data[o1 + 2]
            if blue < min_blue:
                min_blue = blue
            blue = source_data[o2 + 2]
            if blue < min_blue:
                min_blue = blue
            row_min[out] = min_red
            row_min[out + 1] = min_green
            row_min[out + 2] = min_blue
            out += 3

    output = bytearray(width * height * 4)
    out = 0
    row_min_stride = width * 3
    for y in range(height):
        y0 = y - 1 if y > 0 else y
        y1 = y
        y2 = y + 1 if y + 1 < height else y
        row0 = y0 * row_min_stride
        row1 = y1 * row_min_stride
        row2 = y2 * row_min_stride
        for x in range(width):
            column = x * 3
            o0 = row0 + column
            o1 = row1 + column
            o2 = row2 + column
            min_red = row_min[o0]
            red = row_min[o1]
            if red < min_red:
                min_red = red
            red = row_min[o2]
            if red < min_red:
                min_red = red
            min_green = row_min[o0 + 1]
            green = row_min[o1 + 1]
            if green < min_green:
                min_green = green
            green = row_min[o2 + 1]
            if green < min_green:
                min_green = green
            min_blue = row_min[o0 + 2]
            blue = row_min[o1 + 2]
            if blue < min_blue:
                min_blue = blue
            blue = row_min[o2 + 2]
            if blue < min_blue:
                min_blue = blue
            output[out] = min_red
            output[out + 1] = min_green
            output[out + 2] = min_blue
            output[out + 3] = 255
            out += 4
    return OcrImage(
        bytes(output),
        width,
        height,
        4,
        width * 4,
        source=source,
        resolution=resolution,
        page_bbox=image.page_bbox,
        page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
    )


def should_try_high_density_full_page_image(
    image: OcrImage | None,
    candidate: OcrCandidate | None,
) -> bool:
    if image is None:
        return False
    if candidate is None or not candidate.result.text:
        return True
    if stable_encoded_full_page_image_ocr_candidate(image, candidate):
        return False
    if image.encoded is not None:
        return True
    tokens = extracted_text_token_count(candidate.result.text)
    confidence = candidate.result.confidence or 0
    quality = text_ocr_quality_score(candidate.result.text)
    if "_rotated_" in image.source:
        if (
            image.width * image.height >= 6_000_000
            and tokens >= 35
            and confidence >= 90
            and quality <= 0.18
        ):
            return False
        return True
    if (
        image.width * image.height >= 6_000_000
        and tokens < 80
        and confidence >= 88
        and quality >= 0.20
    ):
        return False
    return True


def should_skip_rendered_ocr_after_full_page_image(
    image: OcrImage | None,
    candidate: OcrCandidate | None,
) -> bool:
    if image is None or candidate is None or not candidate.result.text:
        return False
    if not image.source.startswith("full_page_"):
        return False
    tokens = extracted_text_token_count(candidate.result.text)
    confidence = candidate.result.confidence
    if confidence is None:
        return False
    quality = text_ocr_quality_score(candidate.result.text)
    if stable_encoded_full_page_image_ocr_candidate(
        image,
        candidate,
        tokens=tokens,
        confidence=confidence,
        quality=quality,
    ):
        return True
    if compact_confident_full_page_image_ocr_candidate(
        image,
        candidate,
        tokens=tokens,
        confidence=confidence,
        quality=quality,
    ):
        return True
    if confidence >= 65 and tokens >= 1000 and quality <= 0.65:
        return True
    if image.source.endswith("_high_density"):
        if confidence >= 84 and tokens >= 120 and quality <= 0.20:
            return True
        return confidence >= 88 and tokens >= 240 and quality <= 0.22
    if (
        image.encoded is not None
        and image.width * image.height >= 8_000_000
        and confidence >= 65
        and tokens >= 240
        and quality <= 0.24
    ):
        return True
    if (
        image.encoded is not None
        and image.width * image.height >= 24_000_000
        and confidence >= 68
        and tokens >= 350
        and quality <= 0.25
    ):
        return True
    if confidence >= 82 and tokens >= 100 and quality <= 0.16:
        return True
    if confidence >= 76 and tokens >= 180 and quality <= 0.18:
        return True
    return False


def stable_encoded_full_page_image_ocr_candidate(
    image: OcrImage | None,
    candidate: OcrCandidate | None,
    *,
    tokens: int | None = None,
    confidence: int | None = None,
    quality: float | None = None,
) -> bool:
    if image is None or candidate is None or not candidate.result.text:
        return False
    if not image.source.startswith("full_page_") or image.encoded is None:
        return False
    if image.width * image.height < 8_000_000:
        return False
    if tokens is None:
        tokens = extracted_text_token_count(candidate.result.text)
    if tokens < 250:
        return False
    if confidence is None:
        confidence = candidate.result.confidence
    if confidence is None or confidence < 82:
        return False
    if quality is None:
        quality = text_ocr_quality_score(candidate.result.text)
    if quality > 0.30:
        return False
    return ocr_selection.ocr_candidate_score(candidate) >= 95.0


def compact_confident_full_page_image_ocr_candidate(
    image: OcrImage | None,
    candidate: OcrCandidate | None,
    *,
    tokens: int | None = None,
    confidence: int | None = None,
    quality: float | None = None,
) -> bool:
    if image is None or candidate is None or not candidate.result.text:
        return False
    if not image.source.startswith("full_page_"):
        return False
    if image.width * image.height < 4_000_000:
        return False
    if tokens is None:
        tokens = extracted_text_token_count(candidate.result.text)
    if not (40 <= tokens <= 95):
        return False
    if confidence is None:
        confidence = candidate.result.confidence
    if confidence is None or confidence < 90:
        return False
    if quality is None:
        quality = text_ocr_quality_score(candidate.result.text)
    if quality > 0.30:
        return False
    return ocr_selection.ocr_candidate_score(candidate) >= 95.0
