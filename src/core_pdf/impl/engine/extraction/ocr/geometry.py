# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.engine.extraction.ocr import page_analysis as ocr_page_analysis
from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.ocr.text_analysis import normalized_text_tokens
from core_pdf.impl.engine.extraction.ocr.types import OcrRow, ocr_int_value

if TYPE_CHECKING:
    from core_pdf.impl.engine.extraction.ocr.candidates import OcrCandidate
    from core_pdf.impl.engine.extraction.ocr.page_analysis import TextGeometryLine
    from core_pdf.impl.engine.extraction.page_text.mixin import (
        PageExtractionHost,
    )


def ocr_candidate_geometry_lines(
    page: PageExtractionHost,
    candidate: OcrCandidate | None,
    *,
    selected_tokens: set[str] | None = None,
) -> list[TextGeometryLine]:
    if candidate is None:
        return []
    rows = candidate.result.line_rows + selected_iterator_geometry_rows(
        candidate,
        selected_tokens,
    )
    if not rows:
        return []
    lines: list[ocr_page_analysis.TextGeometryLine] = []
    for row in rows:
        line = ocr_row_geometry_line(page, candidate, row)
        if line is not None:
            lines.append(line)
    return lines


def ocr_candidate_textline_geometry_lines(
    page: PageExtractionHost,
    candidate: OcrCandidate | None,
) -> list[TextGeometryLine]:
    if candidate is None:
        return []
    lines: list[ocr_page_analysis.TextGeometryLine] = []
    for row in candidate.result.line_rows:
        line = ocr_row_geometry_line(page, candidate, row)
        if line is not None:
            lines.append(line)
    return lines


def selected_iterator_geometry_rows(
    candidate: OcrCandidate,
    selected_tokens: set[str] | None,
) -> tuple[OcrRow, ...]:
    word_rows = candidate.result.word_rows
    symbol_rows = candidate.result.symbol_rows
    if selected_tokens is None:
        return (*word_rows, *symbol_rows)
    selected_word_rows = tuple(
        row
        for row in word_rows
        if selected_tokens.intersection(
            normalized_text_tokens(str(row.get("text", "")))
        )
    )
    selected_symbol_rows = tuple(
        row
        for row in symbol_rows
        if iterator_symbol_row_matches_selected_token(row, selected_tokens)
    )
    return (*selected_word_rows, *selected_symbol_rows)


def iterator_symbol_row_matches_selected_token(
    row: OcrRow,
    selected_tokens: set[str],
) -> bool:
    row_tokens = normalized_text_tokens(str(row.get("text", "")))
    if not row_tokens:
        return False
    for row_token in row_tokens:
        if row_token in selected_tokens:
            return True
        if len(row_token) == 1 and any(row_token in token for token in selected_tokens):
            return True
    return False


def ocr_row_geometry_line(
    page: PageExtractionHost,
    candidate: OcrCandidate,
    row: OcrRow,
) -> TextGeometryLine | None:
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    bbox = page_geometry.normalize_rect(row.get("page_bbox"))
    if not page_geometry.valid_rect(bbox):
        bbox = None
    try:
        left = ocr_int_value(row["left"])
        top = ocr_int_value(row["top"])
        right = left + ocr_int_value(row["width"])
        bottom = top + ocr_int_value(row["height"])
    except KeyError, TypeError, ValueError:
        if bbox is None:
            return None
    else:
        bbox = bbox or ocr_pixel_bbox_to_page_bbox(
            page,
            candidate,
            (left, top, right, bottom),
        )
    if bbox is None:
        return None
    confidence = row.get("conf")
    try:
        confidence_int = ocr_int_value(confidence) if confidence is not None else None
    except TypeError, ValueError:
        confidence_int = None
    return ocr_page_analysis.text_geometry_line_from_bbox(
        text,
        bbox,
        confidence_int,
        source=candidate.name,
        kind="ocr_textline",
    )


def ocr_pixel_bbox_to_page_bbox(
    page: PageExtractionHost,
    candidate: OcrCandidate,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float] | None:
    if candidate.bbox is not None:
        bbox = (
            bbox[0] + candidate.bbox[0],
            bbox[1] + candidate.bbox[1],
            bbox[2] + candidate.bbox[0],
            bbox[3] + candidate.bbox[1],
        )
    geometry = ocr_candidate_page_geometry(page, candidate)
    if geometry is None:
        return None
    return page_geometry.image_bbox_to_page_bbox(bbox, geometry)


def ocr_candidate_page_geometry(
    page: PageExtractionHost,
    candidate: OcrCandidate,
) -> page_geometry.ImageSpace | None:
    return page_geometry.image_space_from_ocr_candidate(
        page,
        candidate,
        fallback_resolution=ocr_page_analysis.OCR_FALLBACK_DPI,
    )
