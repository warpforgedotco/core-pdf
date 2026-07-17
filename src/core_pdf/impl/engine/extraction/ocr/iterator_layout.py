# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.extraction.ocr import (
    layout as ocr_layout,
)
from core_pdf.impl.engine.extraction.ocr import (
    rendering as ocr_rendering,
)
from core_pdf.impl.engine.extraction.ocr.text_analysis import normalized_text_tokens
from core_pdf.impl.engine.extraction.ocr.types import (
    OcrIteratorLayout,
    OcrRow,
    OcrTextResult,
    ocr_float_value,
    ocr_int_value,
    ocr_observations_from_rows,
)

OCR_TILE_UNSUPPORTED_LINE_MIN_CONFIDENCE = 40
OCR_RECONCILED_LINE_MIN_CONFIDENCE_GAIN = 4.0
OCR_RECONCILED_LINE_MIN_SCORE_GAIN = 7.0


@dataclass(frozen=True)
class OcrLineAlternative:
    text: str
    confidence: int | None
    source: str
    row: OcrRow
    rows: tuple[OcrRow, ...]


def deduplicate_tile_iterator_layout(layout: OcrIteratorLayout) -> OcrIteratorLayout:
    return OcrIteratorLayout(
        deduplicate_iterator_rows_by_geometry(layout.textline_rows),
        deduplicate_iterator_rows_by_geometry(layout.word_rows),
        deduplicate_iterator_rows_by_geometry(layout.symbol_rows),
    )


def deduplicate_iterator_rows_by_geometry(
    rows: list[OcrRow],
) -> list[OcrRow]:
    kept: list[OcrRow] = []
    for row in sorted(rows, key=iterator_row_page_order_key):
        duplicate_index = matching_iterator_row_index(kept, row)
        if duplicate_index is None:
            kept.append(row)
            continue
        if iterator_row_preferred(row, kept[duplicate_index]):
            kept[duplicate_index] = row
    return sorted(kept, key=iterator_row_page_order_key)


def matching_iterator_row_index(
    rows: list[OcrRow],
    candidate: OcrRow,
) -> int | None:
    candidate_tokens = normalized_text_tokens(str(candidate.get("text", "")))
    if not candidate_tokens:
        return None
    candidate_text = " ".join(candidate_tokens)
    candidate_bbox = iterator_row_bbox(candidate)
    if candidate_bbox is None:
        return None
    for index, row in enumerate(rows):
        row_tokens = normalized_text_tokens(str(row.get("text", "")))
        if not row_tokens or " ".join(row_tokens) != candidate_text:
            continue
        row_bbox = iterator_row_bbox(row)
        if row_bbox is None:
            continue
        if iterator_row_overlap_ratio(candidate_bbox, row_bbox) >= 0.65:
            return index
    return None


def iterator_row_preferred(
    candidate: OcrRow,
    current: OcrRow,
) -> bool:
    candidate_conf = iterator_row_confidence(candidate)
    current_conf = iterator_row_confidence(current)
    if (candidate_conf or -1) != (current_conf or -1):
        return (candidate_conf or -1) > (current_conf or -1)
    return iterator_row_area(candidate) > iterator_row_area(current)


def iterator_row_bbox(row: OcrRow) -> tuple[float, float, float, float] | None:
    try:
        left = ocr_float_value(row["left"])
        top = ocr_float_value(row["top"])
        width = ocr_float_value(row["width"])
        height = ocr_float_value(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return (left, top, left + width, top + height)


def iterator_row_area(row: OcrRow) -> float:
    bbox = iterator_row_bbox(row)
    return page_geometry.rect_area(bbox) if bbox is not None else 0.0


def iterator_row_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    overlap = page_geometry.rect_intersection_area(left, right)
    smaller = min(page_geometry.rect_area(left), page_geometry.rect_area(right))
    if smaller <= 0.0:
        return 0.0
    return overlap / smaller


def filter_unsupported_low_confidence_tile_lines(
    layout: OcrIteratorLayout,
    support_text: str,
) -> tuple[OcrIteratorLayout, int]:
    support_tokens = set(normalized_text_tokens(support_text))
    if len(support_tokens) < ocr_rendering.OCR_DENSE_VECTOR_RENDER_TILE_MIN_TOKENS // 2:
        return layout, 0
    kept: list[OcrRow] = []
    removed = 0
    for row in layout.textline_rows:
        if should_keep_tile_textline_row(row, support_tokens):
            kept.append(row)
        else:
            removed += 1
    if removed == 0:
        return layout, 0
    return OcrIteratorLayout(kept, layout.word_rows, layout.symbol_rows), removed


def should_keep_tile_textline_row(
    row: OcrRow,
    support_tokens: set[str],
) -> bool:
    confidence = iterator_row_confidence(row)
    if confidence is None or confidence >= OCR_TILE_UNSUPPORTED_LINE_MIN_CONFIDENCE:
        return True
    row_tokens = set(normalized_text_tokens(str(row.get("text", ""))))
    if not row_tokens:
        return False
    return bool(row_tokens.intersection(support_tokens))


def iterator_text_result_from_existing_result(
    result: OcrTextResult,
    *,
    min_confidence: int = 0,
) -> OcrTextResult:
    if not result.line_rows:
        return OcrTextResult("", None, deskew_info=result.deskew_info)
    converted = iterator_layout_text_result(
        OcrIteratorLayout(
            list(result.line_rows),
            list(result.word_rows),
            list(result.symbol_rows),
        ),
        min_confidence=min_confidence,
    )
    return replace(converted, deskew_info=result.deskew_info)


def reconciled_iterator_text_result_from_existing_result(
    result: OcrTextResult,
) -> OcrTextResult:
    reconciled = reconciled_iterator_layout_text_result(
        OcrIteratorLayout(
            list(result.line_rows),
            list(result.word_rows),
            list(result.symbol_rows),
            result.text,
            result.confidence,
        )
    )
    return replace(reconciled, deskew_info=result.deskew_info)


def iterator_layout_text_result(
    layout: OcrIteratorLayout,
    *,
    min_confidence: int = 0,
) -> OcrTextResult:
    if min_confidence <= 0 and layout.text.strip():
        return ocr_layout.geometry_rendered_ocr_result(
            OcrTextResult(
                layout.text,
                layout.confidence,
                line_rows=tuple(layout.textline_rows),
                word_rows=tuple(layout.word_rows),
                symbol_rows=tuple(layout.symbol_rows),
                observations=ocr_observations_from_rows(
                    [*layout.textline_rows, *layout.word_rows, *layout.symbol_rows]
                ),
            )
        )
    rows = iterator_rows_above_confidence(
        layout.textline_rows,
        min_confidence=min_confidence,
    )
    result = iterator_rows_text_result(rows)
    return ocr_layout.geometry_rendered_ocr_result(
        OcrTextResult(
            result.text,
            result.confidence,
            line_rows=tuple(rows),
            word_rows=tuple(layout.word_rows),
            symbol_rows=tuple(layout.symbol_rows),
            observations=ocr_observations_from_rows(
                [*rows, *layout.word_rows, *layout.symbol_rows]
            ),
        )
    )


def reconciled_iterator_layout_text_result(layout: OcrIteratorLayout) -> OcrTextResult:
    if not layout.textline_rows:
        return OcrTextResult("", None)
    word_rows_by_line = iterator_rows_by_line_key(layout.word_rows)
    symbol_rows_by_line = iterator_rows_by_line_key(layout.symbol_rows)
    word_row_indexes = {id(row): row_index for row_index, row in enumerate(layout.word_rows)}
    word_roles = iterator_word_row_roles(layout.word_rows)
    reconciled_rows: list[OcrRow] = []
    changed = False
    for line_row in sorted(layout.textline_rows, key=iterator_row_page_order_key):
        line_key = iterator_line_key(line_row)
        word_rows = word_rows_by_line.get(line_key, ())
        symbol_rows = symbol_rows_by_line.get(line_key, ())
        alternative = best_reconciled_line_alternative(
            line_row,
            word_rows,
            symbol_rows,
            word_row_indexes,
            word_roles,
        )
        if alternative is None:
            reconciled_rows.append(line_row)
            continue
        reconciled_rows.append(alternative.row)
        changed = (
            changed
            or str(alternative.row.get("text", "")).strip() != str(line_row.get("text", "")).strip()
        )
    if not changed:
        return OcrTextResult("", None)
    result = iterator_rows_text_result(reconciled_rows)
    return OcrTextResult(
        result.text,
        result.confidence,
        line_rows=tuple(reconciled_rows),
        word_rows=tuple(layout.word_rows),
        symbol_rows=tuple(layout.symbol_rows),
        observations=ocr_observations_from_rows(
            [*reconciled_rows, *layout.word_rows, *layout.symbol_rows]
        ),
    )


def iterator_rows_by_line_key(
    rows: list[OcrRow] | tuple[OcrRow, ...],
) -> dict[tuple[int, int, int, int], tuple[OcrRow, ...]]:
    grouped: dict[tuple[int, int, int, int], list[OcrRow]] = {}
    for row in rows:
        grouped.setdefault(iterator_line_key(row), []).append(row)
    return {
        key: tuple(sorted(value, key=iterator_row_page_order_key)) for key, value in grouped.items()
    }


def iterator_word_row_roles(
    rows: list[OcrRow] | tuple[OcrRow, ...],
) -> dict[int, ocr_layout.OcrLayoutWordRole]:
    words = [
        word
        for row_index, row in enumerate(rows)
        if (word := ocr_layout.ocr_layout_word(row, row_index=row_index)) is not None
    ]
    if not words:
        return {}
    return ocr_layout.ocr_rendered_layout(words).roles


def best_reconciled_line_alternative(
    line_row: OcrRow,
    word_rows: tuple[OcrRow, ...],
    symbol_rows: tuple[OcrRow, ...],
    word_row_indexes: dict[int, int] | None = None,
    word_roles: dict[int, ocr_layout.OcrLayoutWordRole] | None = None,
) -> OcrLineAlternative | None:
    alternatives = [line_alternative_from_textline(line_row)]
    word_alternative = line_alternative_from_word_rows(
        line_row,
        word_rows,
        word_row_indexes,
        word_roles,
    )
    if word_alternative is not None:
        alternatives.append(word_alternative)
    symbol_alternative = line_alternative_from_symbol_rows(
        line_row,
        symbol_rows,
        use_choices=False,
    )
    if symbol_alternative is not None:
        alternatives.append(symbol_alternative)
    choice_symbol_alternative = line_alternative_from_symbol_choices_with_word_rows(
        line_row,
        word_rows,
        symbol_rows,
        word_row_indexes,
        word_roles,
    )
    if choice_symbol_alternative is None:
        choice_symbol_alternative = line_alternative_from_symbol_rows(
            line_row,
            symbol_rows,
            use_choices=True,
        )
    if choice_symbol_alternative is not None:
        alternatives.append(choice_symbol_alternative)
    base = alternatives[0]
    best = max(alternatives, key=lambda item: line_alternative_score(item, alternatives))
    if best.source == base.source:
        return None
    if not line_alternative_should_replace(base, best, alternatives):
        return None
    return best


def line_alternative_from_textline(row: OcrRow) -> OcrLineAlternative:
    text = str(row.get("text", "")).strip()
    return OcrLineAlternative(text, iterator_row_confidence(row), "textline", row, (row,))


def line_alternative_from_word_rows(
    line_row: OcrRow,
    rows: tuple[OcrRow, ...],
    word_row_indexes: dict[int, int] | None = None,
    word_roles: dict[int, ocr_layout.OcrLayoutWordRole] | None = None,
) -> OcrLineAlternative | None:
    rows = tuple(row for row in rows if str(row.get("text", "")).strip())
    if not rows:
        return None
    words = []
    for local_index, row in enumerate(rows):
        row_index = (
            word_row_indexes.get(id(row), local_index)
            if word_row_indexes is not None
            else local_index
        )
        word = ocr_layout.ocr_layout_word(row, row_index=row_index)
        if word is not None:
            words.append(word)
    if not words:
        return None
    text = ocr_layout.render_ocr_word_line(words, word_roles)
    if not text:
        return None
    return OcrLineAlternative(
        text,
        iterator_rows_confidence(rows),
        "word",
        iterator_line_row_with_text(line_row, text, rows),
        rows,
    )


def line_alternative_from_symbol_rows(
    line_row: OcrRow,
    rows: tuple[OcrRow, ...],
    *,
    use_choices: bool,
) -> OcrLineAlternative | None:
    rows = tuple(row for row in rows if str(row.get("text", "")).strip())
    if not rows:
        return None
    text = iterator_symbol_rows_line_text(rows, use_choices=use_choices)
    if not text:
        return None
    median_width = iterator_median_row_width(rows)
    confidence = iterator_symbol_rows_confidence(
        rows,
        median_width,
        use_choices=use_choices,
    )
    row = iterator_line_row_with_text(line_row, text, rows)
    if confidence is not None:
        row["conf"] = confidence
    return OcrLineAlternative(
        text,
        confidence,
        "symbol_choices" if use_choices else "symbol",
        row,
        rows,
    )


def line_alternative_from_symbol_choices_with_word_rows(
    line_row: OcrRow,
    word_rows: tuple[OcrRow, ...],
    symbol_rows: tuple[OcrRow, ...],
    word_row_indexes: dict[int, int] | None = None,
    word_roles: dict[int, ocr_layout.OcrLayoutWordRole] | None = None,
) -> OcrLineAlternative | None:
    word_rows = tuple(row for row in word_rows if str(row.get("text", "")).strip())
    symbol_rows = tuple(row for row in symbol_rows if str(row.get("text", "")).strip())
    if not word_rows or not symbol_rows:
        return None
    symbol_rows_by_word = iterator_symbol_rows_by_word_num(symbol_rows)
    median_width = iterator_median_row_width(symbol_rows)
    words = []
    confidences: list[int] = []
    changed = False
    for local_index, word_row in enumerate(word_rows):
        row_index = (
            word_row_indexes.get(id(word_row), local_index)
            if word_row_indexes is not None
            else local_index
        )
        word_num = ocr_int_value(word_row.get("word_num", row_index + 1))
        rows = symbol_rows_by_word.get(word_num, ())
        text = str(word_row.get("text", "")).strip()
        confidence = iterator_row_confidence(word_row)
        if rows:
            symbol_text = iterator_symbol_rows_word_text(
                rows,
                median_width,
                use_choices=True,
            )
            if symbol_text:
                changed = changed or symbol_text != text
                text = symbol_text
                confidence = iterator_symbol_rows_confidence(
                    rows,
                    median_width,
                    use_choices=True,
                )
        if confidence is not None:
            confidences.append(confidence)
        word = ocr_layout.ocr_layout_word(
            {**word_row, "text": text, "conf": confidence},
            row_index=row_index,
        )
        if word is not None:
            words.append(word)
    if not changed or not words:
        return None
    text = ocr_layout.render_ocr_word_line(words, word_roles)
    if not text:
        return None
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    row = iterator_line_row_with_text(line_row, text, word_rows)
    if confidence is not None:
        row["conf"] = confidence
    return OcrLineAlternative(
        text,
        confidence,
        "symbol_choices",
        row,
        (*word_rows, *symbol_rows),
    )


def iterator_symbol_rows_by_word_num(
    rows: tuple[OcrRow, ...],
) -> dict[int, tuple[OcrRow, ...]]:
    grouped: dict[int, list[OcrRow]] = {}
    for row in rows:
        grouped.setdefault(ocr_int_value(row.get("word_num", 0)), []).append(row)
    return {
        key: tuple(sorted(value, key=lambda row: ocr_int_value(row.get("symbol_num", 0))))
        for key, value in grouped.items()
    }


def iterator_symbol_rows_word_text(
    rows: tuple[OcrRow, ...],
    median_width: float,
    *,
    use_choices: bool,
) -> str:
    symbols = [
        iterator_symbol_row_text(row, median_width, use_choices)
        for row in sorted(rows, key=lambda row: ocr_int_value(row.get("symbol_num", 0)))
    ]
    return "".join(symbols).strip()


def iterator_symbol_rows_line_text(
    rows: tuple[OcrRow, ...],
    *,
    use_choices: bool,
) -> str:
    median_width = iterator_median_row_width(rows)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            ocr_int_value(row.get("word_num", 0)),
            ocr_int_value(row.get("symbol_num", 0)),
            ocr_float_value(row.get("left", 0)),
        ),
    )
    words: list[tuple[str, tuple[float, float, float, float] | None]] = []
    current_word_num: int | None = None
    current_symbols: list[str] = []
    current_rows: list[OcrRow] = []

    def flush_word() -> None:
        if not current_symbols:
            return
        text = "".join(current_symbols).strip()
        if text:
            words.append((text, union_iterator_row_bboxes(current_rows)))
        current_symbols.clear()
        current_rows.clear()

    for row in sorted_rows:
        word_num = ocr_int_value(row.get("word_num", 0))
        if current_word_num is not None and word_num != current_word_num:
            flush_word()
        current_word_num = word_num
        current_symbols.append(iterator_symbol_row_text(row, median_width, use_choices))
        current_rows.append(row)
    flush_word()
    return iterator_join_symbol_words(words, median_width)


def iterator_symbol_row_text(
    row: OcrRow,
    median_width: float,
    use_choices: bool,
) -> str:
    text = str(row.get("text", "")).strip()
    if not use_choices:
        return text
    choice = iterator_preferred_symbol_choice(row, median_width)
    return choice[0] if choice is not None else text


def iterator_symbol_rows_confidence(
    rows: tuple[OcrRow, ...],
    median_width: float,
    *,
    use_choices: bool,
) -> int | None:
    confidences: list[int] = []
    for row in rows:
        confidence = iterator_row_confidence(row)
        if use_choices:
            choice = iterator_preferred_symbol_choice(row, median_width)
            if choice is not None:
                confidence = choice[1]
        if confidence is not None:
            confidences.append(confidence)
    if not confidences:
        return None
    return int(round(sum(confidences) / len(confidences)))


def iterator_preferred_symbol_choice(
    row: OcrRow,
    median_width: float,
) -> tuple[str, int] | None:
    choices = row.get("choices", ())
    if not isinstance(choices, (tuple, list)) or not choices:
        return None
    current = str(row.get("text", "")).strip()
    current_confidence = iterator_row_confidence(row) or 0
    row_width = iterator_row_width(row)
    best_text = current
    best_confidence = current_confidence
    best_score = iterator_symbol_choice_score(
        current,
        current_confidence,
        row_width,
        median_width,
    )
    for choice in choices:
        text = str(getattr(choice, "text", "")).strip()
        if not text:
            continue
        confidence = getattr(choice, "confidence", None)
        try:
            confidence_value = int(confidence) if confidence is not None else 0
        except (TypeError, ValueError):
            confidence_value = 0
        score = iterator_symbol_choice_score(
            text,
            confidence_value,
            row_width,
            median_width,
        )
        if score > best_score + 4.0:
            best_text = text
            best_confidence = confidence_value
            best_score = score
    if best_text == current:
        return None
    return best_text, best_confidence


def iterator_symbol_choice_score(
    text: str,
    confidence: int,
    width: float,
    median_width: float,
) -> float:
    if not text:
        return float("-inf")
    score = float(confidence)
    if len(text) > 1:
        score -= 8.0 * (len(text) - 1)
    if median_width > 0.0 and width > 0.0:
        width_ratio = width / median_width
        if iterator_text_is_narrow_symbol(text):
            if width_ratio <= 0.75:
                score += 6.0
            elif width_ratio >= 1.20:
                score -= 5.0
        elif text.isalnum():
            if width_ratio >= 0.65:
                score += 2.0
            else:
                score -= 5.0
    return score


def iterator_text_is_narrow_symbol(text: str) -> bool:
    return len(text) == 1 and not text.isalnum()


def iterator_join_symbol_words(
    words: list[tuple[str, tuple[float, float, float, float] | None]],
    median_width: float,
) -> str:
    parts: list[str] = []
    previous_box: tuple[float, float, float, float] | None = None
    previous_text = ""
    gap_threshold = max(2.0, median_width * 1.6)
    for text, box in words:
        if not text:
            continue
        if parts and iterator_symbol_words_need_space(
            previous_text,
            text,
            previous_box,
            box,
            gap_threshold,
        ):
            parts.append(" ")
        parts.append(text)
        previous_text = text
        previous_box = box
    return "".join(parts).strip()


def iterator_symbol_words_need_space(
    previous_text: str,
    text: str,
    previous_box: tuple[float, float, float, float] | None,
    box: tuple[float, float, float, float] | None,
    gap_threshold: float,
) -> bool:
    if not previous_text or not text:
        return False
    if text[0].isspace() or previous_text[-1].isspace():
        return False
    if box is None or previous_box is None:
        return True
    gap = box[0] - previous_box[2]
    return gap > gap_threshold


def line_alternative_should_replace(
    base: OcrLineAlternative,
    best: OcrLineAlternative,
    alternatives: list[OcrLineAlternative],
) -> bool:
    base_score = line_alternative_score(base, alternatives)
    best_score = line_alternative_score(best, alternatives)
    if best_score < base_score + OCR_RECONCILED_LINE_MIN_SCORE_GAIN:
        return False
    base_confidence = base.confidence if base.confidence is not None else 0
    best_confidence = best.confidence if best.confidence is not None else 0
    if best_confidence + OCR_RECONCILED_LINE_MIN_CONFIDENCE_GAIN < base_confidence:
        return False
    base_tokens = normalized_text_tokens(base.text)
    best_tokens = normalized_text_tokens(best.text)
    if iterator_compact_text_key(base.text) == iterator_compact_text_key(best.text) and len(
        best_tokens
    ) < len(base_tokens):
        return False
    if len(best_tokens) < max(1, int(len(base_tokens) * 0.75)):
        return False
    return not len(best_tokens) > max(len(base_tokens) + 5, int(len(base_tokens) * 1.25))


def line_alternative_score(
    alternative: OcrLineAlternative,
    alternatives: list[OcrLineAlternative],
) -> float:
    text = alternative.text.strip()
    if not text:
        return float("-inf")
    confidence = alternative.confidence if alternative.confidence is not None else 50
    score = float(confidence)
    score += min(15.0, len(normalized_text_tokens(text)) * 1.5)
    score -= iterator_line_fragmentation_penalty(text)
    score += iterator_line_agreement_score(alternative, alternatives)
    if alternative.source == "symbol":
        score -= 2.0
    return score


def iterator_line_agreement_score(
    alternative: OcrLineAlternative,
    alternatives: list[OcrLineAlternative],
) -> float:
    text_key = iterator_compact_text_key(alternative.text)
    token_set = set(normalized_text_tokens(alternative.text))
    score = 0.0
    for other in alternatives:
        if other is alternative:
            continue
        if text_key and text_key == iterator_compact_text_key(other.text):
            score += 10.0
            continue
        other_tokens = set(normalized_text_tokens(other.text))
        if token_set and other_tokens:
            overlap = len(token_set & other_tokens) / max(len(token_set), len(other_tokens))
            score += overlap * 8.0
    return min(score, 18.0)


def iterator_compact_text_key(text: str) -> str:
    return "".join(ch.casefold() for ch in text if not ch.isspace())


def iterator_line_fragmentation_penalty(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 30.0
    short_tokens = [token for token in text.split() if len(token) == 1]
    penalty = min(12.0, len(short_tokens) * 2.0)
    punctuation = sum(1 for ch in nonspace if not ch.isalnum())
    if punctuation / len(nonspace) > 0.35:
        penalty += 6.0
    return penalty


def iterator_line_row_with_text(
    base_row: OcrRow,
    text: str,
    support_rows: tuple[OcrRow, ...],
) -> OcrRow:
    row = dict(base_row)
    row["text"] = text
    confidence = iterator_rows_confidence(support_rows)
    if confidence is not None:
        row["conf"] = confidence
    bbox = union_iterator_row_bboxes(list(support_rows))
    if bbox is not None:
        row["left"] = int(round(bbox[0]))
        row["top"] = int(round(bbox[1]))
        row["width"] = max(1, int(round(bbox[2] - bbox[0])))
        row["height"] = max(1, int(round(bbox[3] - bbox[1])))
    page_bbox = union_iterator_row_page_bboxes(list(support_rows))
    if page_bbox is not None:
        row["page_bbox"] = page_bbox
    return row


def iterator_rows_confidence(rows: tuple[OcrRow, ...]) -> int | None:
    confidences = [
        confidence for row in rows if (confidence := iterator_row_confidence(row)) is not None
    ]
    if not confidences:
        return None
    return int(round(sum(confidences) / len(confidences)))


def iterator_row_width(row: OcrRow) -> float:
    try:
        return ocr_float_value(row["width"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def iterator_median_row_width(rows: tuple[OcrRow, ...]) -> float:
    widths = sorted(iterator_row_width(row) for row in rows if iterator_row_width(row) > 0)
    if not widths:
        return 0.0
    return widths[len(widths) // 2]


def union_iterator_row_bboxes(
    rows: list[OcrRow] | tuple[OcrRow, ...],
) -> tuple[float, float, float, float] | None:
    boxes = [box for row in rows if (box := iterator_row_bbox(row)) is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def iterator_line_key(row: OcrRow) -> tuple[int, int, int, int]:
    return (
        ocr_int_value(row.get("page_num", 1)),
        ocr_int_value(row.get("block_num", 1)),
        ocr_int_value(row.get("par_num", 1)),
        ocr_int_value(row.get("line_num", 1)),
    )


def iterator_tile_layout_text_result(
    layout: OcrIteratorLayout,
    *,
    min_confidence: int = 0,
) -> OcrTextResult:
    filtered = iterator_rows_above_confidence(
        layout.textline_rows,
        min_confidence=min_confidence,
    )
    ordered = sorted(filtered, key=iterator_row_page_order_key)
    result = iterator_rows_text_result(ordered)
    ordered_word_rows = tuple(sorted(layout.word_rows, key=iterator_row_page_order_key))
    ordered_symbol_rows = tuple(sorted(layout.symbol_rows, key=iterator_row_page_order_key))
    return ocr_layout.geometry_rendered_ocr_result(
        OcrTextResult(
            result.text,
            result.confidence,
            line_rows=tuple(ordered),
            word_rows=ordered_word_rows,
            symbol_rows=ordered_symbol_rows,
            observations=ocr_observations_from_rows(
                [
                    *ordered,
                    *ordered_word_rows,
                    *ordered_symbol_rows,
                ]
            ),
        )
    )


def iterator_rows_text_result(rows: list[OcrRow]) -> OcrTextResult:
    lines = [str(row["text"]).strip() for row in rows if str(row.get("text", "")).strip()]
    if not lines:
        return OcrTextResult("", None)
    confidences = [ocr_int_value(row["conf"]) for row in rows if "conf" in row]
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return OcrTextResult(
        "\n".join(lines),
        confidence,
        line_rows=tuple(rows),
        observations=ocr_observations_from_rows(rows),
    )


def iterator_symbol_rows_text_result(rows: list[OcrRow]) -> OcrTextResult:
    if not rows:
        return OcrTextResult("", None)
    lines: list[str] = []
    line_words: list[str] = []
    word_symbols: list[str] = []
    current_line: tuple[int, int, int, int] | None = None
    current_word: tuple[int, int, int, int, int] | None = None
    confidences: list[int] = []

    def flush_word() -> None:
        if word_symbols:
            word = "".join(word_symbols).strip()
            if word:
                line_words.append(word)
            word_symbols.clear()

    def flush_line() -> None:
        flush_word()
        if line_words:
            line = " ".join(line_words).strip()
            if line:
                lines.append(line)
            line_words.clear()

    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        line_key = (
            ocr_int_value(row.get("page_num", 1)),
            ocr_int_value(row.get("block_num", 1)),
            ocr_int_value(row.get("par_num", 1)),
            ocr_int_value(row.get("line_num", 1)),
        )
        word_key = (*line_key, ocr_int_value(row.get("word_num", 0)))
        if current_line is not None and line_key != current_line:
            flush_line()
            current_word = None
        elif current_word is not None and word_key != current_word:
            flush_word()
        current_line = line_key
        current_word = word_key
        word_symbols.append(text)
        if "conf" in row:
            confidences.append(ocr_int_value(row["conf"]))
    flush_line()
    if not lines:
        return OcrTextResult("", None)
    confidence = int(round(sum(confidences) / len(confidences))) if confidences else None
    return OcrTextResult(
        "\n".join(lines),
        confidence,
        symbol_rows=tuple(rows),
        observations=ocr_observations_from_rows(rows),
    )


def iterator_rows_above_confidence(
    rows: list[OcrRow],
    *,
    min_confidence: int,
) -> list[OcrRow]:
    if min_confidence <= 0:
        return rows
    filtered: list[OcrRow] = []
    for row in rows:
        confidence = iterator_row_confidence(row)
        if confidence is not None and confidence >= min_confidence:
            filtered.append(row)
    return filtered


def iterator_row_confidence(row: OcrRow) -> int | None:
    try:
        return int(round(ocr_float_value(row["conf"])))
    except (KeyError, TypeError, ValueError):
        return None


def iterator_row_page_order_key(row: OcrRow) -> tuple[int, int, int, int, int]:
    return (
        ocr_int_value(row.get("top", 0)),
        ocr_int_value(row.get("left", 0)),
        ocr_int_value(row.get("block_num", 0)),
        ocr_int_value(row.get("par_num", 0)),
        ocr_int_value(row.get("line_num", 0)),
    )


def iterator_row_page_bbox(
    row: OcrRow,
) -> tuple[float, float, float, float] | None:
    bbox = row.get("page_bbox")
    bbox_type = type(bbox)
    if bbox_type is not list and bbox_type is not tuple:
        return None
    bbox = cast(Sequence[object], bbox)
    if len(bbox) != 4:
        return None
    try:
        return (
            ocr_float_value(bbox[0]),
            ocr_float_value(bbox[1]),
            ocr_float_value(bbox[2]),
            ocr_float_value(bbox[3]),
        )
    except (TypeError, ValueError):
        return None


def union_iterator_row_page_bboxes(
    rows: list[OcrRow],
) -> tuple[float, float, float, float] | None:
    boxes = [box for row in rows if (box := iterator_row_page_bbox(row)) is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
