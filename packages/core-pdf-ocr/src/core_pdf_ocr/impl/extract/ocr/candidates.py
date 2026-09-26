# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict

import numpy

from core_pdf.impl.array_views import finite_median
from core_pdf.impl.extract_contracts import ObservationBatch, bbox_tuple
from core_pdf.impl.geometry import overlap_ratio_min
from core_pdf.impl.text import CONTENT_TOKEN_RE, search_key, text_tokens
from core_pdf_ocr.impl.extract.contracts import (
    HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS,
    HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP,
    HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP,
)
from core_pdf_ocr.impl.extract.observations import maximum_candidate_coverage
from core_pdf_ocr.impl.extract.quality import (
    Candidate,
    make_candidate,
    text_utility_stats,
)

OCR_TOKEN_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def normalized_ocr_token_key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(OCR_TOKEN_TRANSLATION).casefold()


def hidden_text_verification(
    hidden: ObservationBatch,
    preview: ObservationBatch,
) -> bool:
    hidden_by_token: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for text, raw_box in zip(hidden.text, hidden.bbox, strict=True):
        box = bbox_tuple(raw_box)
        for token in text_tokens(text):
            hidden_by_token[token].append(box)

    preview_entries = tuple(
        (token, bbox_tuple(raw_box))
        for text, raw_box in zip(preview.text, preview.bbox, strict=True)
        for token in text_tokens(text)
    )
    used: dict[str, set[int]] = defaultdict(set)
    matched = 0
    spatially_matched = 0
    for token, preview_box in preview_entries:
        candidates = hidden_by_token.get(token, ())
        available = (
            (index, box) for index, box in enumerate(candidates) if index not in used[token]
        )
        preview_center_x = (preview_box[0] + preview_box[2]) * 0.5
        preview_center_y = (preview_box[1] + preview_box[3]) * 0.5
        closest = min(
            available,
            key=lambda item: (
                ((item[1][0] + item[1][2]) * 0.5 - preview_center_x) ** 2
                + ((item[1][1] + item[1][3]) * 0.5 - preview_center_y) ** 2
            ),
            default=None,
        )
        if closest is None:
            continue
        index, hidden_box = closest
        used[token].add(index)
        matched += 1
        hidden_center_x = (hidden_box[0] + hidden_box[2]) * 0.5
        hidden_center_y = (hidden_box[1] + hidden_box[3]) * 0.5
        x_tolerance = max(
            12.0,
            (preview_box[2] - preview_box[0]) * 1.5,
            (hidden_box[2] - hidden_box[0]) * 1.5,
        )
        y_tolerance = max(
            6.0,
            (preview_box[3] - preview_box[1]) * 1.5,
            (hidden_box[3] - hidden_box[1]) * 1.5,
        )
        spatially_matched += int(
            abs(preview_center_x - hidden_center_x) <= x_tolerance
            and abs(preview_center_y - hidden_center_y) <= y_tolerance
        )

    preview_tokens = len(preview_entries)
    if matched < HIDDEN_TEXT_VERIFY_MIN_MATCHED_TOKENS:
        return False
    token_overlap = matched / max(1, preview_tokens)
    if token_overlap < HIDDEN_TEXT_VERIFY_MIN_TOKEN_OVERLAP:
        return False
    spatial_overlap = spatially_matched / max(1, preview_tokens)
    return spatial_overlap >= HIDDEN_TEXT_VERIFY_MIN_SPATIAL_OVERLAP


def candidate_text_containment(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter or sum(len(token) for token in shorter) < 4:
        return False
    width = len(shorter)
    return any(longer[index : index + width] == shorter for index in range(len(longer) - width + 1))


def merge_candidate_batches(
    candidates: tuple[Candidate, ...],
) -> Candidate:
    if not candidates:
        return make_candidate(-1, ObservationBatch.empty())
    if len(candidates) == 1:
        return candidates[0]
    modes = {candidate.mode for candidate in candidates}
    merged_by_mode: list[Candidate] = []
    for mode in sorted(modes):
        mode_candidates = tuple(candidate for candidate in candidates if candidate.mode == mode)
        combined = ObservationBatch.concatenate(
            *(candidate.observations for candidate in mode_candidates)
        )
        combined_symbols = ObservationBatch.concatenate(
            *(candidate.symbols for candidate in mode_candidates)
        )
        fuzzy_tile_deduplication = len(mode_candidates) > 1
        order = numpy.lexsort((combined.bbox[:, 0], -combined.bbox[:, 1]))
        descending_y = -combined.bbox[order, 1]
        normalized_text = tuple(search_key(text) for text in combined.text)
        normalized_tokens = tuple(
            tuple(
                key
                for match in CONTENT_TOKEN_RE.finditer(text)
                if (key := normalized_ocr_token_key(match.group(0)))
            )
            for text in combined.text
        )
        observation_utility = numpy.fromiter(
            (
                text_utility_stats(text, float(confidence)).utility
                for text, confidence in zip(
                    combined.text,
                    combined.confidence,
                    strict=True,
                )
            ),
            dtype=numpy.float32,
            count=len(combined),
        )
        deduplicated: list[int] = []
        accepted_positions: dict[int, int] = {}
        for order_position, raw_index in enumerate(order):
            index = int(raw_index)
            box = combined.bbox[index]
            start = int(numpy.searchsorted(descending_y, -box[3], side="right"))
            nearby_positions: list[int] = []
            for raw_other in order[start:order_position]:
                other = int(raw_other)
                accepted_position = accepted_positions.get(other)
                if accepted_position is None:
                    continue
                other_box = combined.bbox[other]
                if box[0] < other_box[2] and other_box[0] < box[2] and box[1] < other_box[3]:
                    nearby_positions.append(accepted_position)
            duplicate_index = next(
                (
                    accepted_position
                    for accepted_position in sorted(nearby_positions)
                    if (
                        overlap_ratio_min(
                            combined.bbox[index],
                            combined.bbox[deduplicated[accepted_position]],
                        )
                        >= (
                            0.35
                            if candidate_text_containment(
                                normalized_tokens[deduplicated[accepted_position]],
                                normalized_tokens[index],
                            )
                            else (
                                0.45
                                if normalized_text[deduplicated[accepted_position]]
                                == normalized_text[index]
                                else (0.70 if fuzzy_tile_deduplication else math.inf)
                            )
                        )
                    )
                ),
                None,
            )
            if duplicate_index is None:
                accepted_positions[index] = len(deduplicated)
                deduplicated.append(index)
                continue
            accepted_index = deduplicated[duplicate_index]
            containment = candidate_text_containment(
                normalized_tokens[accepted_index],
                normalized_tokens[index],
            )
            if containment and len(normalized_text[index]) != len(normalized_text[accepted_index]):
                if len(normalized_text[index]) > len(normalized_text[accepted_index]):
                    deduplicated[duplicate_index] = index
            elif observation_utility[index] > observation_utility[accepted_index]:
                deduplicated[duplicate_index] = index
            if deduplicated[duplicate_index] == index:
                del accepted_positions[accepted_index]
                accepted_positions[index] = duplicate_index
        heights = tuple(
            candidate.metrics.median_text_height
            for candidate in mode_candidates
            if candidate.metrics.median_text_height > 0.0
        )
        merged_by_mode.append(
            make_candidate(
                mode,
                combined.take(deduplicated),
                symbols=combined_symbols,
                median_text_height=(
                    finite_median(numpy.asarray(heights, dtype=numpy.float64)) if heights else 0.0
                ),
            )
        )
    return max(merged_by_mode, key=lambda candidate: candidate.metrics.utility)


def augment_candidate(
    primary: Candidate,
    supplement: Candidate,
    *,
    minimum_confidence: float,
) -> tuple[Candidate, int]:
    if not len(supplement.observations):
        return primary, 0
    observations = supplement.observations
    confidence = observations.confidence
    informative = numpy.fromiter(
        (sum(character.isalnum() for character in text) >= 1 for text in observations.text),
        dtype=numpy.bool_,
        count=len(observations),
    )
    useful = numpy.fromiter(
        (
            (
                text_utility_stats(text, float(value)).utility >= 2.0
                or (len(text.strip()) == 1 and text.strip().isalnum() and float(value) >= 85.0)
            )
            for text, value in zip(observations.text, confidence, strict=True)
        ),
        dtype=numpy.bool_,
        count=len(observations),
    )
    coverage = maximum_candidate_coverage(
        observations.bbox,
        primary.observations.bbox,
    )
    additions = (
        (confidence >= max(70.0, minimum_confidence)) & informative & useful & (coverage < 0.30)
    )
    added = int(numpy.count_nonzero(additions))
    if not added:
        return primary, 0
    combined = ObservationBatch.concatenate_selected(
        primary.observations,
        observations,
        additions,
    )
    return make_candidate(primary.mode, combined, symbols=primary.symbols), added
