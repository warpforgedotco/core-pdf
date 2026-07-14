# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from statistics import median
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from core_pdf.impl.engine.extraction.ocr.vector_text import (
    VectorStrokeOcrResult,
    page_has_vector_stroke_text_candidates,
)
from core_pdf.impl.engine.extraction.ocr import geometry as ocr_geometry
from core_pdf.impl.engine.extraction.ocr import selection as ocr_selection
from core_pdf.impl.engine.extraction.common import observation_resolver
from core_pdf.impl.engine.extraction.ocr import page_analysis as ocr_page_analysis
from core_pdf.impl.engine.extraction.ocr import text_analysis as ocr_text_analysis
from core_pdf.impl.engine.extraction.common import page_geometry
from core_pdf.impl.engine.layout.geometry_quality import (
    layout_geometry_should_trigger_ocr,
    layout_geometry_summary_from_record,
)
from core_pdf.impl.engine.layout.word_frequencies import word_rank

if TYPE_CHECKING:
    from core_pdf.impl.engine.extraction.ocr.candidates import OcrPageTextResult
    from core_pdf.impl.engine.extraction.page_text.mixin import (
        PageExtractionHost,
    )


OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS = 30_000
OCR_EMBEDDED_IMAGE_TEXT_MAX_PIXELS = 250_000
OCR_EMBEDDED_IMAGE_TEXT_MAX_AREA_RATIO = 0.025
OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXEL_DENSITY = 8.0
OCR_EMBEDDED_IMAGE_TEXT_MIN_ASPECT_RATIO = 0.55
OCR_EMBEDDED_IMAGE_TEXT_MAX_ASPECT_RATIO = 1.85
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_PIXELS = 750_000
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_AREA_RATIO = 0.085
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MIN_ASPECT_RATIO = 2.5
OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_ASPECT_RATIO = 8.0


@dataclass(frozen=True)
class TextGeometryRecord:
    text: str
    observation: page_geometry.PageObservation | None = None


OCR_ARTIFACT_PRUNABLE_SOURCE_PREFIXES = (
    "full_page_",
    "full_page_image",
    "high_density_full_page_image",
    "line_art_text_mask_",
    "rendered_page_",
)
OCR_ARTIFACT_EDGE_PUNCTUATION = "-‐‑‒–—−_"
DOCUMENT_LOCAL_TOKEN_EDGE_CHARS = "\"'“”‘’«»()[]{}<>.,;:!?"
_COMPOUND_TOKEN_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+")
_DOCUMENT_LOCAL_TOKEN_FRAGMENT_RE = re.compile(r"[A-Za-z0-9*_-]+")
OCR_EDGE_NOISE_PUNCTUATION = "\"'“”‘’`~_=|¦¬^°•·.,;:!?()[]{}<>/\\+-@%$"
OCR_ALPHA_JOINERS = frozenset({"'", "’", "-", "‐", "‑", "‒", "–", "—"})
PRIZE_AMOUNT_TOKEN_RE = re.compile(r"^(?P<whole>\d{1,4})[:,-](?P<cents>\d{2})$")
PRIZE_RANK_TOKEN_RE = re.compile(r"^(?P<rank>\d+)(?:\.\)|\)|\.)$")
MONTH_NAME_TOKENS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)


@dataclass(frozen=True)
class DocumentLocalTokenRepairContext:
    replacements: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OcrLineWordRow:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None


@dataclass(frozen=True)
class OcrLineRowWords:
    bbox: tuple[float, float, float, float]
    confidence: float | None
    words: tuple[OcrLineWordRow, ...]


@dataclass(frozen=True)
class OcrMatchedLineWords:
    index: int
    line: observation_resolver.ResolvedTextLine
    row: OcrLineRowWords


@dataclass(frozen=True)
class OcrSyntheticTextLine:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None


def ocr_is_enabled() -> bool:
    return os.environ.get("CORE_PDF_OCR", "").casefold() not in {
        "0",
        "false",
        "no",
        "off",
    }


def prune_weak_ocr_artifact_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    changed = False
    pruned: list[observation_resolver.ResolvedTextLine] = []
    for line in lines:
        replacement = prune_weak_ocr_artifact_line_text(line)
        if replacement is None:
            changed = True
            continue
        if replacement != line.text:
            changed = True
            pruned.append(resolved_line_with_pruned_text(line, replacement))
            continue
        pruned.append(line)
    return tuple(pruned) if changed else lines


def precision_clean_figure_region_label_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    broad_page_result: Any | None,
    figure_result: Any | None,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines or broad_page_result is None or figure_result is None:
        return lines
    figure_region_bbox = figure_result_region_bbox(figure_result)
    if figure_region_bbox is None:
        return lines
    support_rows = figure_region_label_support_rows(
        broad_page_result=broad_page_result,
        figure_result=figure_result,
    )
    if not support_rows:
        return lines
    cleaned: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        replacement = precision_clean_figure_region_label_line_text(
            line,
            figure_region_bbox=figure_region_bbox,
            support_rows=support_rows,
        )
        if replacement is None or replacement == line.text:
            cleaned.append(line)
            continue
        cleaned.append(resolved_line_with_pruned_text(line, replacement))
        changed = True
    return tuple(cleaned) if changed else lines


def prune_edge_noise_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    dominant_left_anchor = dominant_content_left_anchor(lines)
    page_width = output_lines_page_width(lines)
    if dominant_left_anchor is None or page_width is None:
        return lines
    changed = False
    pruned: list[observation_resolver.ResolvedTextLine] = []
    for line in lines:
        replacement = prune_edge_noise_line_text(
            line,
            dominant_left_anchor=dominant_left_anchor,
            page_width=page_width,
        )
        if replacement is None:
            changed = True
            continue
        if replacement != line.text:
            changed = True
            pruned.append(resolved_line_with_pruned_text(line, replacement))
            continue
        pruned.append(line)
    return tuple(pruned) if changed else lines


def prune_embedded_image_band_noise_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    page_width = output_lines_page_width(lines)
    if page_width is None:
        return lines
    embedded_bands = embedded_image_text_band_bboxes(lines)
    if not embedded_bands:
        return lines
    kept: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        if embedded_image_band_noise_line_should_drop(
            line,
            embedded_bands=embedded_bands,
            page_width=page_width,
        ):
            changed = True
            continue
        kept.append(line)
    return tuple(kept) if changed else lines


def prune_malformed_edge_url_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    page_width = output_lines_page_width(lines)
    page_bounds = output_lines_page_bounds(lines)
    if page_width is None or page_bounds is None:
        return lines
    min_y, max_y = page_bounds
    page_height = max(0.0, max_y - min_y)
    if page_height <= 0.0:
        return lines
    kept: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        if malformed_edge_url_line_should_drop(
            line,
            page_width=page_width,
            min_y=min_y,
            max_y=max_y,
            page_height=page_height,
        ):
            changed = True
            continue
        kept.append(line)
    return tuple(kept) if changed else lines


def repair_word_geometry_noise_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    candidate: Any,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    line_word_rows = candidate_line_word_rows(candidate)
    if not lines or not line_word_rows:
        return lines
    matches = matched_output_line_words(lines, line_word_rows)
    if not matches:
        return lines
    changed = False
    replacements: dict[int, str] = {}
    for band in matched_output_line_word_bands(matches):
        anchor = band_content_left_anchor(band)
        for match in band:
            replacement = reconstruct_row_text_from_gap_words(match.row)
            if replacement is None and anchor is not None:
                replacement = reconstruct_band_line_text(match.row, band_anchor=anchor)
            if replacement is None or replacement == match.line.text:
                continue
            replacements[match.index] = replacement
    repaired: list[observation_resolver.ResolvedTextLine] = []
    for index, line in enumerate(lines):
        replacement = replacements.get(index)
        if replacement is not None:
            repaired.append(resolved_line_with_pruned_text(line, replacement))
            changed = True
            continue
        repaired.append(line)
    return tuple(repaired) if changed else lines


def prune_shadowed_selected_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    kept: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    band_split_lines = [
        line for line in lines if ":band_split" in line.observation.source
    ]
    if not band_split_lines:
        return lines
    for line in lines:
        if not selected_output_line_is_shadowed_by_band_split(line, band_split_lines):
            kept.append(line)
            continue
        changed = True
    return tuple(kept) if changed else lines


def prune_shadowed_band_split_suffix_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    figure_lines = [
        line for line in lines if line.observation.source == "figure_ocr_regions"
    ]
    if not figure_lines:
        return lines
    kept: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        if not band_split_line_is_shadowed_suffix_fragment(line, figure_lines):
            kept.append(line)
            continue
        changed = True
    return tuple(kept) if changed else lines


def selected_output_line_is_shadowed_by_band_split(
    line: observation_resolver.ResolvedTextLine,
    band_split_lines: list[observation_resolver.ResolvedTextLine],
) -> bool:
    if ":selected_output" not in line.observation.source:
        return False
    bbox = line.observation.ink_bbox or line.observation.bbox
    if bbox is None:
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(line.text)
    if len(tokens) < 4 or not line_tokens_have_repeated_halves(tokens):
        return False
    token_set = set(tokens)
    if len(token_set) * 2 != len(tokens):
        return False
    center_y = (bbox[1] + bbox[3]) * 0.5
    height = max(1.0, bbox[3] - bbox[1])
    for candidate in band_split_lines:
        candidate_bbox = candidate.observation.ink_bbox or candidate.observation.bbox
        if candidate_bbox is None:
            continue
        candidate_center_y = (candidate_bbox[1] + candidate_bbox[3]) * 0.5
        candidate_height = max(1.0, candidate_bbox[3] - candidate_bbox[1])
        if abs(center_y - candidate_center_y) > max(height, candidate_height) * 1.6:
            continue
        candidate_tokens = ocr_text_analysis.normalized_text_tokens(candidate.text)
        if not candidate_tokens:
            continue
        overlap = len(token_set & set(candidate_tokens)) / max(1, len(token_set))
        if overlap >= 0.75:
            return True
    return False


def band_split_line_is_shadowed_suffix_fragment(
    line: observation_resolver.ResolvedTextLine,
    figure_lines: list[observation_resolver.ResolvedTextLine],
) -> bool:
    if ":band_split" not in line.observation.source:
        return False
    confidence = line.observation.confidence
    if confidence is None or confidence > 75.0:
        return False
    bbox = line.observation.ink_bbox or line.observation.bbox
    if bbox is None:
        return False
    if not line_text_has_symbol_corruption(line.text):
        return False
    first_token = leading_alpha_token(line.text)
    if first_token is None or len(first_token) < 4:
        return False
    center_y = (bbox[1] + bbox[3]) * 0.5
    height = max(1.0, bbox[3] - bbox[1])
    for figure_line in figure_lines:
        figure_bbox = figure_line.observation.ink_bbox or figure_line.observation.bbox
        if figure_bbox is None:
            continue
        figure_word = single_alpha_line_text(figure_line.text)
        if figure_word is None or len(figure_word) <= len(first_token):
            continue
        figure_center_y = (figure_bbox[1] + figure_bbox[3]) * 0.5
        figure_height = max(1.0, figure_bbox[3] - figure_bbox[1])
        if abs(center_y - figure_center_y) > max(height, figure_height) * 0.7:
            continue
        if not figure_word.casefold().endswith(first_token.casefold()):
            continue
        if (len(figure_word) - len(first_token)) > 3:
            continue
        return True
    return False


def line_tokens_have_repeated_halves(tokens: list[str]) -> bool:
    if len(tokens) < 4 or len(tokens) % 2 != 0:
        return False
    midpoint = len(tokens) // 2
    return tokens[:midpoint] == tokens[midpoint:]


def line_text_has_symbol_corruption(text: str) -> bool:
    return re.search(r"[^\w\s.,;:!?\"'()/-]", text) is not None


def leading_alpha_token(text: str) -> str | None:
    for raw in text.split():
        token = raw.strip(OCR_EDGE_NOISE_PUNCTUATION)
        if token.isalpha():
            return token
    return None


def single_alpha_line_text(text: str) -> str | None:
    tokens = [token.strip(OCR_EDGE_NOISE_PUNCTUATION) for token in text.split()]
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if len(alpha_tokens) != 1:
        return None
    return alpha_tokens[0]


def repair_document_local_identifier_output_lines(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    support_texts: Iterable[str] = (),
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not lines:
        return lines
    context = document_local_token_repair_context(
        [line.text for line in lines],
        support_texts=support_texts,
    )
    if not context.replacements:
        return lines
    repaired: list[observation_resolver.ResolvedTextLine] = []
    changed = False
    for line in lines:
        text = repair_document_local_identifier_line_text(line.text, context)
        if text != line.text:
            repaired.append(resolved_line_with_repaired_text(line, text))
            changed = True
            continue
        repaired.append(line)
    return tuple(repaired) if changed else lines


def repair_document_local_identifier_text(
    text: str,
    *,
    support_texts: Iterable[str] = (),
) -> str:
    context = document_local_token_repair_context((text,), support_texts=support_texts)
    return "\n".join(
        repair_document_local_identifier_line_text(line, context)
        for line in text.splitlines()
    )


def document_local_token_repair_context(
    current_texts: Iterable[str],
    *,
    support_texts: Iterable[str] = (),
) -> DocumentLocalTokenRepairContext:
    current_text = "\n".join(text for text in current_texts if text).casefold()
    if not current_text:
        return DocumentLocalTokenRepairContext(())
    support_counts: Counter[str] = Counter()
    support_display: dict[str, str] = {}
    for text in support_texts:
        if not text:
            continue
        for token in document_local_support_tokens(text):
            normalized = token.casefold()
            support_counts[normalized] += 1
            best = support_display.get(normalized)
            if best is None or len(token) > len(best):
                support_display[normalized] = token
    replacements: list[tuple[str, str, int]] = []
    for normalized, count in support_counts.items():
        display = support_display.get(normalized)
        if not display:
            continue
        spaced = spaced_compound_token_variant(display)
        if not spaced:
            continue
        if spaced.casefold() not in current_text:
            continue
        replacements.append((spaced, display, count))
    replacements.sort(
        key=lambda item: (-len(item[0]), -item[2], -len(item[1]), item[0].casefold())
    )
    return DocumentLocalTokenRepairContext(
        tuple((spaced, display) for spaced, display, _count in replacements)
    )


def document_local_support_tokens(text: str) -> Iterable[str]:
    for raw in text.split():
        for token in _DOCUMENT_LOCAL_TOKEN_FRAGMENT_RE.findall(raw):
            token = token.strip(DOCUMENT_LOCAL_TOKEN_EDGE_CHARS)
            if not token or len(token) < 4:
                continue
            if not any(ch.isalpha() for ch in token):
                continue
            if not spaced_compound_token_variant(token):
                continue
            yield token


def spaced_compound_token_variant(token: str) -> str | None:
    if len(token) < 4 or not any(ch.isalpha() for ch in token):
        return None
    parts = token.split("-")
    rebuilt: list[str] = []
    changed = False
    for part in parts:
        split = split_compound_token_part(part)
        if len(split) > 1:
            rebuilt.append(" ".join(split))
            changed = True
            continue
        rebuilt.append(part)
    if not changed:
        return None
    spaced = "-".join(rebuilt)
    return spaced if spaced.casefold() != token.casefold() else None


def split_compound_token_part(token: str) -> list[str]:
    if len(token) < 3:
        return [token]
    if not any(ch.islower() for ch in token):
        return [token]
    if not any(ch.isupper() for ch in token[1:]):
        return [token]
    parts = _COMPOUND_TOKEN_PART_RE.findall(token)
    if len(parts) < 2 or "".join(parts) != token:
        return [token]
    return parts


def repair_document_local_identifier_line_text(
    text: str,
    context: DocumentLocalTokenRepairContext,
) -> str:
    repaired = normalize_generic_ocr_line_text(text)
    repaired = intrinsic_identifier_spacing(repaired)
    for spaced, display in context.replacements:
        pattern = re.compile(
            rf"(?<!\w){re.escape(spaced).replace(r'\\ ', r'[ \\t]+')}(?!\w)",
            re.IGNORECASE,
        )
        repaired = pattern.sub(display, repaired)
    return intrinsic_identifier_compaction(repaired)


def normalize_generic_ocr_line_text(text: str) -> str:
    repaired = re.sub(r"(?<=[A-Za-z])\|(?=[A-Za-z])", "", text)
    repaired = re.sub(r"(?<!\S)\|(?!\S)", "", repaired)
    repaired = normalize_leading_l_alpha_confusions(repaired)
    repaired = normalize_rare_alpha_confusion_tokens(repaired)
    repaired = normalize_short_lowercase_prefix_titlecase_tokens(repaired)
    repaired = normalize_hyphenated_titlecase_compounds(repaired)
    repaired = re.sub(r"(?i)\b(model)([¹²³⁴⁵⁶⁷⁸⁹⁰]+)", r"\1 \2", repaired)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired).strip()
    repaired = normalize_precision_first_prize_line_text(repaired)
    return repaired


def normalize_rare_alpha_confusion_tokens(text: str) -> str:
    return re.sub(
        r"(?<![A-Za-z])([A-Za-z]{5,12})(?![A-Za-z])",
        replace_rare_alpha_confusion_token,
        text,
    )


def replace_rare_alpha_confusion_token(match: re.Match[str]) -> str:
    token = match.group(1)
    normalized = token.casefold()
    current_rank = word_rank(normalized)
    if current_rank is not None:
        return token
    candidate = best_ranked_alpha_confusion_candidate(token)
    return candidate if candidate is not None else token


def best_ranked_alpha_confusion_candidate(token: str) -> str | None:
    normalized = token.casefold()
    candidates: list[tuple[int, str]] = []
    for source, target in common_alpha_confusion_replacements():
        if source not in normalized:
            continue
        candidate = normalized.replace(source, target)
        if candidate == normalized:
            continue
        rank = word_rank(candidate)
        if rank is None or rank > 25_000:
            continue
        candidates.append((rank, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_rank, best_candidate = candidates[0]
    if len(candidates) > 1 and candidates[1][0] <= best_rank * 2:
        return None
    return token_with_preserved_case(token, best_candidate)


def common_alpha_confusion_replacements() -> tuple[tuple[str, str], ...]:
    return (("tt", "ff"),)


def token_with_preserved_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def repair_prose_line_break_artifacts_text(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    repaired: list[str | None] = list(lines)
    changed = False
    index = 0
    while index < len(repaired):
        line = repaired[index]
        if line is None:
            index += 1
            continue
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        next_nonempty = next_nonempty_line_index(repaired, index + 1)
        if next_nonempty is not None:
            right = repaired[next_nonempty]
            if right is None:
                index += 1
                continue
            merged = merge_blank_line_hyphenated_continuation(
                line,
                right,
            )
            if merged is not None:
                repaired[index] = merged
                for blank_index in range(index + 1, next_nonempty + 1):
                    repaired[blank_index] = None
                changed = True
        index += 1
    previous_nonempty: str | None = None
    for index, line in enumerate(repaired):
        if line is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if standalone_fragment_text_line_should_drop(
            line,
            previous_nonempty=previous_nonempty,
        ):
            repaired[index] = None
            changed = True
            continue
        pruned = prune_orphaned_trailing_hyphen_fragment(
            line,
            repaired,
            index,
        )
        if pruned != line:
            repaired[index] = pruned if pruned.strip() else None
            stripped = pruned.strip()
            changed = True
            if not stripped:
                continue
        if stripped:
            previous_nonempty = repaired[index]
    if not changed:
        return text
    joined = "\n".join(line for line in repaired if line is not None)
    return re.sub(r"\n{3,}", "\n\n", joined)


def repair_direct_hyphenated_line_continuations_text(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    repaired: list[str | None] = list(lines)
    changed = False
    index = 0
    while index < len(repaired) - 1:
        left = repaired[index]
        if left is None:
            index += 1
            continue
        prefix = trailing_hyphenated_alpha_fragment(left)
        if prefix is None:
            index += 1
            continue
        right = repaired[index + 1]
        if right is None:
            index += 1
            continue
        suffix = leading_alpha_fragment_text(right)
        if len(suffix) < 2 or not suffix[:1].islower():
            index += 1
            continue
        if not direct_hyphenated_line_join_is_plausible(prefix, suffix, right):
            index += 1
            continue
        suffix_end = leading_alpha_fragment_end(right)
        if suffix_end is None:
            index += 1
            continue
        left_trimmed = trim_trailing_hyphenated_alpha_fragment(left)
        repaired[index] = f"{left_trimmed}{prefix}{suffix}{right[suffix_end:]}"
        repaired[index + 1] = None
        changed = True
        index += 2
    if not changed:
        return text
    joined = "\n".join(line for line in repaired if line is not None)
    return re.sub(r"\n{3,}", "\n\n", joined)


def direct_hyphenated_line_join_is_plausible(
    prefix: str,
    suffix: str,
    right: str,
) -> bool:
    if joined_word_is_plausible(prefix, suffix):
        return True
    if len(prefix) < 4 or len(suffix) < 4:
        return False
    if not suffix[:1].islower():
        return False
    right_tokens = ocr_text_analysis.normalized_text_tokens(right)
    if len(right_tokens) < 2:
        return False
    return not ocr_text_analysis.alpha_token_looks_ocr_garbled(f"{prefix}{suffix}")


def compact_footnote_url_markers_text(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    repaired: list[str] = []
    index = 0
    changed = False
    while index < len(lines):
        current = lines[index]
        marker = standalone_footnote_url_marker(current)
        if marker is not None and index + 1 < len(lines):
            following = lines[index + 1].lstrip()
            if following.startswith(("http://", "https://")):
                repaired.append(f"{normalize_footnote_marker_text(marker)}{following}")
                changed = True
                index += 2
                continue
        compacted = re.sub(r"(?<!\S)([¹²³⁴⁵⁶⁷⁸⁹⁰]+)\s+(https?://)", r"\1\2", current)
        if compacted != current:
            changed = True
        repaired.append(compacted)
        index += 1
    if not changed:
        return text
    return "\n".join(repaired)


def standalone_footnote_url_marker(text: str) -> str | None:
    stripped = text.strip()
    if not stripped or len(stripped) > 4:
        return None
    return stripped if all(ch.isdigit() for ch in stripped) else None


def normalize_footnote_marker_text(text: str) -> str:
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return text.translate(superscripts)


def normalize_precision_first_prize_line_text(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 3:
        return text
    rank_count = sum(1 for token in tokens if prize_rank_token(token) is not None)
    amount_count = sum(
        1
        for token in tokens
        if prize_amount_token(token) is not None
        or token.startswith("$")
        or prize_numeric_amount_like_token(token)
    )
    if rank_count == 0 or (rank_count + amount_count) < 3:
        return text
    normalized: list[str] = []
    for index, token in enumerate(tokens):
        rank_one = prize_rank_one_confusion_token(token)
        if rank_one is not None:
            normalized.append(rank_one)
            continue
        if rare_short_alpha_prize_junk_token(tokens, index):
            continue
        if single_alpha_numeric_junk_token(tokens, index):
            continue
        if malformed_prize_digit_token_should_drop(tokens, index):
            continue
        normalized.append(token)
    return " ".join(normalized) if normalized else text


def next_nonempty_line_index(lines: list[str | None], start: int) -> int | None:
    for index in range(start, len(lines)):
        line = lines[index]
        if line is not None and line.strip():
            return index
    return None


def merge_blank_line_hyphenated_continuation(left: str, right: str) -> str | None:
    prefix = trailing_hyphenated_alpha_fragment(left)
    if prefix is None:
        return None
    suffix = leading_alpha_fragment_text(right)
    if len(suffix) < 2 or not suffix[:1].islower():
        return None
    if not joined_word_is_plausible(prefix, suffix):
        return None
    suffix_end = leading_alpha_fragment_end(right)
    if suffix_end is None:
        return None
    left_trimmed = trim_trailing_hyphenated_alpha_fragment(left)
    right_remainder = right[suffix_end:]
    return f"{left_trimmed}{prefix}{suffix}{right_remainder}"


def trailing_hyphenated_alpha_fragment(text: str) -> str | None:
    match = re.search(r"([A-Za-z]{2,})-\s*$", text)
    return match.group(1) if match is not None else None


def trim_trailing_hyphenated_alpha_fragment(text: str) -> str:
    return re.sub(r"([A-Za-z]{2,})-\s*$", "", text)


def leading_alpha_fragment_text(text: str) -> str:
    match = re.match(r"\s*([A-Za-z]+)", text)
    return match.group(1) if match is not None else ""


def leading_alpha_fragment_end(text: str) -> int | None:
    match = re.match(r"\s*[A-Za-z]+", text)
    return match.end() if match is not None else None


def joined_word_is_plausible(prefix: str, suffix: str) -> bool:
    joined = f"{prefix}{suffix}".casefold()
    joined_rank = word_rank(joined)
    if joined_rank is None:
        return False
    prefix_rank = word_rank(prefix.casefold())
    suffix_rank = word_rank(suffix.casefold())
    if joined_rank > 150_000:
        return False
    if prefix_rank is None or suffix_rank is None:
        return True
    return joined_rank < max(prefix_rank, suffix_rank) or (
        len(suffix) <= 4 and joined_rank <= 75_000
    )


def prune_orphaned_trailing_hyphen_fragment(
    line: str,
    lines: list[str | None],
    index: int,
) -> str:
    match = re.search(r"(?P<prefix>.*\b)(?P<fragment>[a-z]{1,6})-\s*$", line)
    if match is None:
        return line
    next_nonempty = next_nonempty_line_index(lines, index + 1)
    if next_nonempty is not None:
        next_line = lines[next_nonempty]
        if next_line is None:
            return match.group("prefix").rstrip()
        suffix = leading_alpha_fragment_text(next_line)
        if len(suffix) >= 2 and suffix[:1].islower():
            if joined_word_is_plausible(match.group("fragment"), suffix):
                return line
    return match.group("prefix").rstrip()


def standalone_fragment_text_line_should_drop(
    line: str,
    *,
    previous_nonempty: str | None,
) -> bool:
    stripped = line.strip()
    if not stripped or previous_nonempty is None:
        return False
    if isolated_numeric_symbol_junk_line(stripped):
        return True
    return False


def isolated_numeric_symbol_junk_line(text: str) -> bool:
    digit_count = sum(1 for ch in text if ch.isdigit())
    alpha_count = sum(1 for ch in text if ch.isalpha())
    punct_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    if digit_count < 2 or punct_count == 0:
        return False
    if alpha_count > 2:
        return False
    return punct_count >= 1


def intrinsic_identifier_spacing(text: str) -> str:
    return re.sub(
        r"(?<!\w)([A-Z][A-Za-z]{3,})(?!\w)",
        split_titlecase_compound_token_match,
        text,
    )


def normalize_hyphenated_titlecase_compounds(text: str) -> str:
    return re.sub(
        r"(?<!\w)([A-Z][A-Za-z]+(?:-[A-Z][A-Za-z]+)+)(?!\w)",
        split_hyphenated_titlecase_compound_token_match,
        text,
    )


def intrinsic_identifier_compaction(text: str) -> str:
    repaired = text
    repaired = re.sub(
        r"(?<!\w)([A-Z][a-z]{2,10})[ \t]+(Bench|Net|Res|Vis)(?=$|[ \t]+(?:[a-z]|\d|[-/:]))",
        compact_technical_suffix_pair,
        repaired,
    )
    repaired = re.sub(
        r"(?<!\w)([A-Z][a-z]{1,4})[ \t]+([A-Z]{2,}[A-Z0-9]*)(?!\w)",
        compact_short_titlecase_acronym_pair,
        repaired,
    )
    repaired = re.sub(
        r"(?<!\w)([A-Z][a-z]{1,4})[ \t]+([A-Z][A-Z0-9]*(?:-[A-Z0-9][A-Za-z0-9*]*)+)(?!\w)",
        compact_short_titlecase_acronym_pair,
        repaired,
    )
    repaired = re.sub(
        r"(?<!\w)([A-Z][a-z]{2,6})[ \t]+([A-Z])(?!\w)",
        compact_short_titlecase_acronym_pair,
        repaired,
    )
    if line_has_identifier_context(repaired):
        repaired = re.sub(
            r"(?<!\w)([A-Z][a-z]{2,3})[ \t]+([A-Z][a-z]{2,3})(?=[ \t]+[a-z])",
            compact_contextual_short_titlecase_pair,
            repaired,
        )
        repaired = re.sub(
            r"(?<!\w)([A-Z][a-z]{2,3})[ \t]+([A-Z][a-z]{2,3})(?=$|[ \t]+(?:\d|[-/:]))",
            compact_contextual_short_titlecase_pair,
            repaired,
        )
    repaired = re.sub(
        r"(?<!\w)([A-Z][a-z]{2,10})[ \t]+([A-Z][a-z]{2,10})(?!\w)",
        compact_titlecase_identifier_pair,
        repaired,
    )
    return repaired


def compact_titlecase_identifier_pair(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if min(len(left), len(right)) >= 4:
        return match.group(0)
    if short_titlecase_pair_should_stay_split(left, right):
        return match.group(0)
    if titlecase_phrase_lead_word(left) and titlecase_compound_common_word(right):
        return match.group(0)
    elif not (
        titlecase_token_fragment_is_uncommon(left)
        or titlecase_token_fragment_is_uncommon(right)
    ):
        return match.group(0)
    return f"{left}{right}"


def compact_contextual_short_titlecase_pair(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if titlecase_phrase_lead_word(left) and titlecase_compound_common_word(right):
        return match.group(0)
    if not (
        titlecase_token_fragment_is_uncommon(left)
        or titlecase_token_fragment_is_uncommon(right)
    ):
        return match.group(0)
    return f"{left}{right}"


def compact_technical_suffix_pair(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if left in {"A", "An", "The", "This", "That"}:
        return match.group(0)
    return f"{left}{right}"


def split_titlecase_compound_token_match(match: re.Match[str]) -> str:
    token = match.group(1)
    parts = split_compound_token_part(token)
    if len(parts) < 2:
        return token
    if not all(
        part.isalpha() and part[:1].isupper() and part[1:].islower() for part in parts
    ):
        return token
    if not titlecase_compound_parts_should_split(parts):
        return token
    return " ".join(parts)


def titlecase_compound_parts_should_split(parts: list[str]) -> bool:
    if any(len(part) < 3 for part in parts):
        return False
    if all(titlecase_compound_common_word(part) for part in parts):
        return True
    if all(len(part) >= 4 for part in parts):
        return True
    return False


def split_hyphenated_titlecase_compound_token_match(match: re.Match[str]) -> str:
    token = match.group(1)
    segments = token.split("-")
    rebuilt: list[str] = []
    changed = False
    for segment in segments:
        parts = split_compound_token_part(segment)
        if len(parts) >= 2 and titlecase_compound_parts_should_split(parts):
            rebuilt.append(" ".join(parts))
            changed = True
            continue
        rebuilt.append(segment)
    return "-".join(rebuilt) if changed else token


def compact_short_titlecase_acronym_pair(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if not short_titlecase_prefix_is_identifier_fragment(left):
        return match.group(0)
    return f"{left}{right}"


def short_titlecase_prefix_is_identifier_fragment(token: str) -> bool:
    if len(token) <= 2:
        return True
    rank = word_rank(token.casefold())
    return rank is None or rank > 20_000


def line_has_identifier_context(text: str) -> bool:
    return re.search(r"(?<!\w)[A-Z]{2,}(?:-[A-Z0-9*]+)*(?!\w)", text) is not None


def titlecase_token_fragment_is_uncommon(token: str) -> bool:
    rank = word_rank(token.casefold())
    if rank is None:
        if len(token) >= 4 and token[:1].isupper() and token[1:].islower():
            return False
        return True
    return rank > 3_000


def titlecase_compound_common_word(token: str) -> bool:
    rank = word_rank(token.casefold())
    return rank is not None and rank <= 50_000


def titlecase_phrase_lead_word(token: str) -> bool:
    return token in {"All"}


def short_titlecase_pair_should_stay_split(left: str, right: str) -> bool:
    left_rank = word_rank(left.casefold())
    right_rank = word_rank(right.casefold())
    if left_rank is None or right_rank is None:
        return False
    if left_rank > 50_000 or right_rank > 50_000:
        return False
    return max(left_rank, right_rank) > 3_000


def prize_rank_one_confusion_token(token: str) -> str | None:
    stripped = token.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return None
    letters = stripped.replace("1", "I").replace("!", "I").replace("|", "I")
    if letters not in {"IL", "LI", "II"}:
        return None
    return "1."


def single_alpha_numeric_junk_token(tokens: list[str], index: int) -> bool:
    stripped = tokens[index].strip(OCR_EDGE_NOISE_PUNCTUATION)
    if len(stripped) != 1 or not stripped.isalpha():
        return False
    prev_numeric = index > 0 and word_text_is_numeric_like(tokens[index - 1])
    next_numeric = index + 1 < len(tokens) and word_text_is_numeric_like(
        tokens[index + 1]
    )
    return prev_numeric or next_numeric


def malformed_prize_digit_token_should_drop(tokens: list[str], index: int) -> bool:
    token = tokens[index].strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not token.isdigit() or len(token) < 4:
        return False
    if prize_amount_token(token) is not None:
        return False
    if index > 0 and prize_rank_token(tokens[index - 1]) is None:
        return False
    if index + 1 < len(tokens) and prize_rank_token(tokens[index + 1]) is None:
        return False
    return True


def prize_numeric_amount_like_token(token: str) -> bool:
    stripped = token.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not word_text_is_numeric_like(stripped):
        return False
    digit_count = sum(1 for ch in stripped if ch.isdigit())
    return digit_count >= 3


def rare_short_alpha_prize_junk_token(tokens: list[str], index: int) -> bool:
    token = tokens[index].strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not token.isalpha() or len(token) > 4:
        return False
    prev_numeric = any(
        prize_numeric_amount_like_token(tokens[candidate_index])
        for candidate_index in range(max(0, index - 3), index)
    )
    next_prize = any(
        prize_numeric_amount_like_token(tokens[candidate_index])
        or prize_rank_token(tokens[candidate_index]) is not None
        for candidate_index in range(index + 1, min(len(tokens), index + 4))
    )
    if len(token) <= 2:
        return next_prize
    if not (prev_numeric and next_prize):
        return False
    rank = word_rank(token.casefold())
    return rank is None or rank > 10_000


def normalize_leading_l_alpha_confusions(text: str) -> str:
    return re.sub(r"(?<!\w)l[a-z]{3,7}(?!\w)", replace_leading_l_alpha_confusion, text)


def replace_leading_l_alpha_confusion(match: re.Match[str]) -> str:
    token = match.group(0)
    candidate = f"I{token[1:]}"
    token_rank = word_rank(token.casefold())
    candidate_rank = word_rank(candidate.casefold())
    if candidate_rank is None or candidate_rank > 20_000:
        return token
    if token_rank is not None and token_rank <= candidate_rank:
        return token
    return candidate


def normalize_short_lowercase_prefix_titlecase_tokens(text: str) -> str:
    return re.sub(
        r"(?<!\w)([a-z]{1,3})([A-Z][a-z]{4,})(?!\w)",
        replace_short_lowercase_prefix_titlecase_token,
        text,
    )


def replace_short_lowercase_prefix_titlecase_token(match: re.Match[str]) -> str:
    prefix = match.group(1)
    suffix = match.group(2)
    prefix_rank = word_rank(prefix.casefold())
    suffix_rank = word_rank(suffix.casefold())
    if suffix_rank is None or suffix_rank > 80_000:
        return match.group(0)
    if prefix_rank is not None and prefix_rank <= suffix_rank:
        return match.group(0)
    return suffix


def prune_weak_ocr_artifact_line_text(
    line: observation_resolver.ResolvedTextLine,
) -> str | None:
    if not ocr_artifact_prunable_line(line):
        return line.text
    if geometryless_table_fusion_noise_line_should_drop(line.text, line):
        return None
    geometryless_date = trim_geometryless_table_fusion_date_noise_text(line.text, line)
    if geometryless_date is not None:
        return geometryless_date
    token_matches = list(ocr_text_analysis.TEXT_TOKEN_RE.finditer(line.text))
    if not token_matches:
        return None
    raw_tokens = [match.group(0) for match in token_matches]
    normalized_tokens = [token.casefold() for token in raw_tokens]
    confidence = resolved_line_max_confidence(line)
    if broad_page_garbled_line_should_drop(raw_tokens, confidence, line):
        return None
    if weak_ocr_orphan_line_should_drop(
        raw_tokens,
        normalized_tokens,
        confidence,
        line,
    ):
        return None
    prune_indexes = {
        index
        for index, token in enumerate(raw_tokens)
        if weak_ocr_artifact_token_should_prune(
            token,
            index,
            raw_tokens,
            confidence,
            line,
        )
    }
    prune_indexes.update(
        weak_ocr_artifact_suffix_token_indexes(
            raw_tokens,
            confidence,
        )
    )
    if not prune_indexes:
        return line.text
    return text_without_token_indexes(line.text, token_matches, prune_indexes)


def broad_page_garbled_line_should_drop(
    raw_tokens: list[str],
    confidence: float | None,
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    """Drop confidently identified garbage from broad-page OCR output.

    This deliberately requires weak OCR confidence and a majority of unknown
    or malformed tokens. Short labels, numeric rows, and technical notation
    remain eligible for the existing specialized cleanup paths.
    """
    if not line.observation.source.startswith("rendered_page_"):
        return False
    if confidence is None or confidence > 60.0 or len(raw_tokens) < 5:
        return False
    if ocr_text_analysis.numeric_token_ratio(line.text) >= 0.55:
        return False
    if ocr_text_analysis.line_has_readable_technical_notation(line.text, raw_tokens):
        return False
    if ocr_text_analysis.ocr_text_has_dense_formula_notation(line.text):
        return False
    garbled = sum(1 for token in raw_tokens if ocr_line_token_looks_garbled(token))
    return garbled / len(raw_tokens) >= 0.60


def prune_edge_noise_line_text(
    line: observation_resolver.ResolvedTextLine,
    *,
    dominant_left_anchor: float,
    page_width: float,
) -> str | None:
    bbox = line.observation.ink_bbox or line.observation.bbox
    if bbox is None:
        return line.text
    x0, _y0, x1, _y1 = bbox
    width = max(0.0, x1 - x0)
    if width < page_width * 0.42:
        return line.text
    left_gap = dominant_left_anchor - x0
    if left_gap < max(18.0, page_width * 0.035):
        return line.text
    confidence = resolved_line_max_confidence(line)
    if confidence is not None and confidence >= 88.0:
        return line.text
    token_matches = list(ocr_text_analysis.NONSPACE_TOKEN_RE.finditer(line.text))
    if len(token_matches) < 4:
        return line.text
    raw_tokens = [match.group(0) for match in token_matches]
    keep_start = geometry_edge_noise_keep_start(
        raw_tokens,
    )
    if keep_start == 0:
        return line.text
    if len(raw_tokens) - keep_start < 2:
        return line.text
    replacement = text_with_only_token_indexes(
        line.text,
        token_matches,
        set(range(keep_start, len(raw_tokens))),
    )
    return replacement or None


def repair_word_geometry_noise_line_text(
    line: observation_resolver.ResolvedTextLine,
    line_word_rows: tuple[OcrLineRowWords, ...],
) -> str | None:
    if not line.observation.source.startswith("full_page_"):
        return None
    bbox = line.observation.ink_bbox or line.observation.bbox
    if bbox is None:
        return None
    matched = best_matching_line_word_row(bbox, line_word_rows)
    if matched is None or len(matched.words) < 4:
        return None
    return reconstruct_row_text_from_gap_words(matched)


def reconstruct_row_text_from_gap_words(
    row: OcrLineRowWords,
) -> str | None:
    if len(row.words) < 4:
        return None
    line_width = max(0.0, row.bbox[2] - row.bbox[0])
    if line_width <= 0.0:
        return None
    keep_start = leading_row_content_start(row, line_width=line_width)
    keep_end = trailing_row_content_end(row)
    if keep_start == 0 and keep_end == len(row.words):
        return None
    return reconstruct_row_text_from_word_span(
        row, keep_start=keep_start, keep_end=keep_end
    )


def leading_row_content_start(
    row: OcrLineRowWords,
    *,
    line_width: float,
) -> int:
    keep_start = 0
    strong_suffix_counts = strong_word_suffix_counts(row.words)
    for index, word in enumerate(row.words):
        if not word_is_geometry_prefix_noise(word):
            break
        if strong_suffix_counts[index + 1] < 2:
            break
        if index + 1 >= len(row.words):
            break
        gap = row.words[index + 1].bbox[0] - word.bbox[2]
        if gap < max(24.0, line_width * 0.06):
            continue
        keep_start = index + 1
    if keep_start == 0:
        for index in range(1, len(row.words)):
            prefix = row.words[:index]
            if not prefix:
                continue
            noisy_prefix = sum(
                1 for word in prefix if word_is_geometry_prefix_noise(word)
            )
            if noisy_prefix < max(2, len(prefix) - 1):
                continue
            if strong_suffix_counts[index] < 2:
                continue
            gap = row.words[index].bbox[0] - row.words[index - 1].bbox[2]
            if gap < max(16.0, line_width * 0.03) and len(prefix) < 3:
                continue
            keep_start = index
            break
    if (
        keep_start > 0
        and word_text_is_numeric_like(row.words[keep_start - 1].text)
        and word_text_is_numeric_like(row.words[keep_start].text)
    ):
        keep_start -= 1
    elif (
        keep_start > 0
        and word_text_is_numeric_like(row.words[keep_start].text)
        and word_is_geometry_prefix_noise(row.words[keep_start - 1])
    ):
        # OCR often emits a weak marker immediately before the first prize amount.
        keep_start = max(0, keep_start)
    return keep_start


def trailing_row_content_end(row: OcrLineRowWords) -> int:
    keep_end = len(row.words)
    strong_prefix_counts = strong_word_prefix_counts(row.words)
    while keep_end > 0:
        word = row.words[keep_end - 1]
        if not word_is_geometry_suffix_noise(word):
            break
        if strong_prefix_counts[keep_end - 1] < 2:
            break
        keep_end -= 1
    return keep_end


def reconstruct_row_text_from_word_span(
    row: OcrLineRowWords,
    *,
    keep_start: int,
    keep_end: int,
) -> str | None:
    if (
        keep_start > 0
        and word_text_is_numeric_like(row.words[keep_start - 1].text)
        and word_text_is_numeric_like(row.words[keep_start].text)
        and len(row.words[keep_start - 1].text.strip()) <= 3
    ):
        keep_start -= 1
    tokens: list[str] = []
    for index in range(keep_start, keep_end):
        word = row.words[index]
        if word_should_drop_inside_content(
            row, index, keep_start=keep_start, keep_end=keep_end
        ):
            continue
        tokens.append(word.text)
    replacement = " ".join(tokens).strip()
    return replacement or None


def matched_output_line_words(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    line_word_rows: tuple[OcrLineRowWords, ...],
) -> tuple[OcrMatchedLineWords, ...]:
    matched: list[OcrMatchedLineWords] = []
    for index, line in enumerate(lines):
        if not line.observation.source.startswith("full_page_"):
            continue
        if ":band_split" in line.observation.source:
            continue
        bbox = line.observation.ink_bbox or line.observation.bbox
        if bbox is None:
            continue
        row = best_matching_line_word_row(bbox, line_word_rows)
        if row is None or len(row.words) < 4:
            continue
        matched.append(OcrMatchedLineWords(index, line, row))
    return tuple(matched)


def matched_output_line_word_bands(
    matches: tuple[OcrMatchedLineWords, ...],
) -> tuple[tuple[OcrMatchedLineWords, ...], ...]:
    if not matches:
        return ()
    bands: list[list[OcrMatchedLineWords]] = [[matches[0]]]
    for match in matches[1:]:
        current = bands[-1]
        previous = current[-1]
        prev_box = previous.row.bbox
        box = match.row.bbox
        prev_height = max(1.0, prev_box[3] - prev_box[1])
        gap = prev_box[1] - box[3]
        tolerance = max(14.0, prev_height * 1.35)
        if gap <= tolerance:
            current.append(match)
            continue
        bands.append([match])
    return tuple(tuple(band) for band in bands)


def band_content_left_anchor(
    band: tuple[OcrMatchedLineWords, ...],
) -> float | None:
    x_values: list[float] = []
    for match in band:
        for word in match.row.words:
            if not word_is_band_anchor_candidate(word):
                continue
            x_values.append(word.bbox[0])
    if len(x_values) < 2:
        return None
    x_values.sort()
    return x_values[len(x_values) // 2]


def word_is_band_anchor_candidate(word: OcrLineWordRow) -> bool:
    confidence = word.confidence if word.confidence is not None else 0.0
    if confidence < 85.0:
        return False
    if word_text_is_numeric_like(word.text):
        return True
    return edge_token_is_strong(word.text)


def reconstruct_band_line_text(
    row: OcrLineRowWords,
    *,
    band_anchor: float,
) -> str | None:
    line_width = max(0.0, row.bbox[2] - row.bbox[0])
    if line_width <= 0.0:
        return None
    tolerance = max(12.0, line_width * 0.03)
    keep_start: int | None = None
    strong_suffix_counts = strong_word_suffix_counts(row.words)
    for index, word in enumerate(row.words):
        if strong_suffix_counts[index] < 2:
            continue
        if word.bbox[0] + tolerance < band_anchor and not word_text_is_numeric_like(
            word.text
        ):
            continue
        if keep_start is None:
            keep_start = index
            break
    if keep_start is None or keep_start == 0:
        return None
    prefix = row.words[:keep_start]
    noisy_prefix = sum(1 for word in prefix if word_is_geometry_prefix_noise(word))
    if not prefix or noisy_prefix < max(2, len(prefix) - 1):
        return None
    if (
        keep_start > 0
        and word_text_is_numeric_like(row.words[keep_start - 1].text)
        and word_text_is_numeric_like(row.words[keep_start].text)
    ):
        keep_start -= 1
    keep_end = trailing_row_content_end(row)
    return reconstruct_row_text_from_word_span(
        row, keep_start=keep_start, keep_end=keep_end
    )


def geometry_edge_noise_keep_start(
    raw_tokens: list[str],
) -> int:
    strong_suffix_counts = strong_edge_token_suffix_counts(raw_tokens)
    keep_start = 0
    for index, token in enumerate(raw_tokens):
        if not edge_token_is_noise(token):
            break
        if strong_suffix_counts[index + 1] < 2:
            break
        keep_start = index + 1
    return keep_start


def strong_edge_token_suffix_counts(raw_tokens: list[str]) -> list[int]:
    counts = [0] * (len(raw_tokens) + 1)
    running = 0
    for index in range(len(raw_tokens) - 1, -1, -1):
        if edge_token_is_strong(raw_tokens[index]):
            running += 1
        counts[index] = running
    counts[len(raw_tokens)] = 0
    return counts


def strong_word_suffix_counts(words: tuple[OcrLineWordRow, ...]) -> list[int]:
    counts = [0] * (len(words) + 1)
    running = 0
    for index in range(len(words) - 1, -1, -1):
        if word_is_strong(words[index]):
            running += 1
        counts[index] = running
    counts[len(words)] = 0
    return counts


def strong_word_prefix_counts(words: tuple[OcrLineWordRow, ...]) -> list[int]:
    counts = [0] * (len(words) + 1)
    running = 0
    for index, word in enumerate(words, start=1):
        if word_is_strong(word):
            running += 1
        counts[index] = running
    return counts


def strong_edge_token_prefix_count(raw_tokens: list[str], end: int) -> int:
    return sum(1 for token in raw_tokens[:end] if edge_token_is_strong(token))


def edge_token_is_noise(token: str) -> bool:
    stripped = token.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return True
    alnum = ocr_text_analysis.token_alnum_count(stripped)
    if alnum == 0:
        return True
    if any(ch.isdigit() for ch in stripped):
        if any(ch.isalpha() for ch in stripped) and not stripped.isalnum():
            return True
        if alnum <= 2:
            return True
        return False
    if not any(ch.isalpha() for ch in stripped):
        return True
    alpha_only = alpha_token_letters(stripped)
    if not alpha_only:
        return True
    if ocr_text_analysis.alpha_token_looks_ocr_garbled(alpha_only):
        return True
    rank = word_rank(alpha_only.casefold()) if alpha_only.isalpha() else None
    if len(alpha_only) <= 2:
        return rank is None or rank > 2_000
    if len(alpha_only) <= 3:
        return rank is None or rank > 12_000
    if stripped != alpha_only and not token_uses_only_alpha_joiners(stripped):
        return True
    return False


def edge_token_is_strong(token: str) -> bool:
    stripped = token.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return False
    if any(ch.isdigit() for ch in stripped):
        return ocr_text_analysis.token_alnum_count(stripped) >= 2
    alpha_only = alpha_token_letters(stripped)
    if not alpha_only or (
        stripped != alpha_only and not token_uses_only_alpha_joiners(stripped)
    ):
        return False
    rank = word_rank(alpha_only.casefold())
    if alpha_only.isupper() and len(alpha_only) >= 4:
        return True
    return rank is not None and rank <= 80_000 and len(alpha_only) >= 3


def alpha_token_letters(token: str) -> str:
    return "".join(ch for ch in token if ch.isalpha())


def token_uses_only_alpha_joiners(token: str) -> bool:
    return all(ch.isalpha() or ch in OCR_ALPHA_JOINERS for ch in token)


def word_is_geometry_noise(word: OcrLineWordRow) -> bool:
    confidence = word.confidence if word.confidence is not None else 0.0
    if confidence >= 60.0 and not edge_token_is_noise(word.text):
        return False
    return edge_token_is_noise(word.text) or confidence < 45.0


def word_is_geometry_prefix_noise(word: OcrLineWordRow) -> bool:
    confidence = word.confidence if word.confidence is not None else 0.0
    if edge_token_is_noise(word.text) or confidence < 45.0:
        return True
    stripped = word.text.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return True
    if word_text_is_numeric_like(stripped):
        return confidence < 35.0
    alpha_only = "".join(ch for ch in stripped if ch.isalpha())
    if len(alpha_only) <= 1:
        return confidence < 95.0
    if len(alpha_only) <= 3:
        return confidence < 90.0
    return False


def word_is_geometry_suffix_noise(word: OcrLineWordRow) -> bool:
    confidence = word.confidence if word.confidence is not None else 0.0
    if edge_token_is_noise(word.text) or confidence < 35.0:
        return True
    stripped = word.text.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return True
    if word_text_is_numeric_like(stripped):
        return False
    alpha_only = "".join(ch for ch in stripped if ch.isalpha())
    if not alpha_only:
        return True
    if len(alpha_only) <= 2:
        return confidence < 92.0
    return False


def word_is_strong(word: OcrLineWordRow) -> bool:
    confidence = word.confidence if word.confidence is not None else 0.0
    if word_text_is_numeric_like(word.text):
        return confidence >= 40.0
    if confidence < 60.0:
        return False
    return edge_token_is_strong(word.text)


def word_should_drop_inside_content(
    row: OcrLineRowWords,
    index: int,
    *,
    keep_start: int,
    keep_end: int,
) -> bool:
    word = row.words[index]
    if not word_is_geometry_noise(word):
        return False
    if word_text_is_numeric_like(word.text):
        return False
    if index <= keep_start or index >= keep_end - 1:
        return False
    previous = row.words[index - 1]
    following = row.words[index + 1]
    if not row_word_is_contentful(previous) or not row_word_is_contentful(following):
        return False
    gap_before = word.bbox[0] - previous.bbox[2]
    gap_after = following.bbox[0] - word.bbox[2]
    row_width = max(0.0, row.bbox[2] - row.bbox[0])
    gap_threshold = max(12.0, row_width * 0.025)
    if gap_before >= gap_threshold and gap_after >= gap_threshold:
        return True
    stripped = word.text.strip(OCR_EDGE_NOISE_PUNCTUATION)
    return len(stripped) <= 2


def row_word_is_contentful(word: OcrLineWordRow) -> bool:
    if word_text_is_numeric_like(word.text):
        return True
    if word_is_strong(word):
        return True
    confidence = word.confidence if word.confidence is not None else 0.0
    stripped = word.text.strip(OCR_EDGE_NOISE_PUNCTUATION)
    alpha_only = "".join(ch for ch in stripped if ch.isalpha())
    return (
        confidence >= 82.0
        and len(alpha_only) >= 2
        and not edge_token_is_noise(word.text)
    )


def word_text_is_numeric_like(text: str) -> bool:
    stripped = text.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return False
    digit_count = sum(1 for ch in stripped if ch.isdigit())
    if digit_count == 0:
        return False
    alpha_count = sum(1 for ch in stripped if ch.isalpha())
    return alpha_count == 0 and digit_count >= max(1, len(stripped) // 2)


def candidate_line_word_rows(candidate: Any) -> tuple[OcrLineRowWords, ...]:
    result = getattr(candidate, "result", None)
    line_rows = getattr(result, "line_rows", ()) if result is not None else ()
    word_rows = getattr(result, "word_rows", ()) if result is not None else ()
    if not line_rows or not word_rows:
        return ()
    words_by_key: dict[tuple[int, int, int], list[OcrLineWordRow]] = {}
    for row in word_rows:
        bbox = page_geometry.rect_box_tuple(row.get("page_bbox"))
        if bbox is None:
            continue
        key = (
            int(row.get("block_num", 0)),
            int(row.get("par_num", 0)),
            int(row.get("line_num", 0)),
        )
        words_by_key.setdefault(key, []).append(
            OcrLineWordRow(
                text=str(row.get("text", "")).strip(),
                bbox=bbox,
                confidence=page_geometry.numeric_confidence(row.get("conf")),
            )
        )
    output: list[OcrLineRowWords] = []
    for row in line_rows:
        bbox = page_geometry.rect_box_tuple(row.get("page_bbox"))
        if bbox is None:
            continue
        key = (
            int(row.get("block_num", 0)),
            int(row.get("par_num", 0)),
            int(row.get("line_num", 0)),
        )
        words = tuple(
            sorted(
                (word for word in words_by_key.get(key, []) if word.text),
                key=lambda word: (word.bbox[0], word.bbox[1]),
            )
        )
        if len(words) < 2:
            continue
        output.append(
            OcrLineRowWords(
                bbox=bbox,
                confidence=page_geometry.numeric_confidence(row.get("conf")),
                words=words,
            )
        )
    return tuple(output)


def best_matching_line_word_row(
    bbox: tuple[float, float, float, float],
    line_word_rows: tuple[OcrLineRowWords, ...],
) -> OcrLineRowWords | None:
    best: OcrLineRowWords | None = None
    best_score = 0.0
    target = page_geometry.PageObservation(
        kind="ocr_line",
        source="output_line",
        bbox=bbox,
        advance_bbox=bbox,
        ink_bbox=bbox,
    )
    for row in line_word_rows:
        observation = page_geometry.PageObservation(
            kind="ocr_line",
            source="candidate_line",
            bbox=row.bbox,
            advance_bbox=row.bbox,
            ink_bbox=row.bbox,
        )
        score = page_geometry.observation_geometry_match_score(target, observation)
        if score > best_score:
            best = row
            best_score = score
    if best_score < 0.82:
        return None
    return best


def dominant_content_left_anchor(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> float | None:
    buckets: Counter[int] = Counter()
    bucket_values: dict[int, list[float]] = {}
    page_width = output_lines_page_width(lines)
    if page_width is None:
        return None
    bucket_width = max(10.0, page_width * 0.02)
    for line in lines:
        bbox = line.observation.ink_bbox or line.observation.bbox
        if bbox is None:
            continue
        x0, _y0, x1, _y1 = bbox
        width = max(0.0, x1 - x0)
        if width < page_width * 0.22:
            continue
        readable = line_readable_token_count(line.text)
        confidence = resolved_line_max_confidence(line)
        if readable < 2 and (confidence is None or confidence < 70.0):
            continue
        bucket = int(round(x0 / bucket_width))
        buckets[bucket] += 1
        bucket_values.setdefault(bucket, []).append(x0)
    if not buckets:
        return None
    bucket = max(buckets, key=lambda key: (buckets[key], key))
    values = sorted(bucket_values.get(bucket, ()))
    if not values:
        return None
    return values[len(values) // 2]


def output_lines_page_width(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> float | None:
    boxes: list[tuple[float, float, float, float]] = []
    for line in lines:
        box = line.observation.ink_bbox or line.observation.bbox
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    max_x = max(box[2] for box in boxes)
    width = max(0.0, max_x - min_x)
    return width if width > 0.0 else None


def output_lines_page_bounds(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[float, float] | None:
    boxes: list[tuple[float, float, float, float]] = []
    for line in lines:
        box = line.observation.ink_bbox or line.observation.bbox
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return min(box[1] for box in boxes), max(box[3] for box in boxes)


def line_readable_token_count(text: str) -> int:
    count = 0
    for token in ocr_text_analysis.normalized_text_tokens(text):
        if any(ch.isdigit() for ch in token):
            count += 1
            continue
        if len(token) < 3 or not token.isalpha():
            continue
        if ocr_text_analysis.alpha_token_looks_ocr_garbled(token):
            continue
        rank = word_rank(token)
        if rank is not None and rank <= 120_000:
            count += 1
    return count


def malformed_edge_url_line_should_drop(
    line: observation_resolver.ResolvedTextLine,
    *,
    page_width: float,
    min_y: float,
    max_y: float,
    page_height: float,
) -> bool:
    observation = line.observation
    bbox = observation.ink_bbox or observation.bbox
    if bbox is None:
        return False
    if not observation.source.startswith("rendered_page_"):
        return False
    text = line.text.strip()
    if not text:
        return False
    lowered = text.casefold()
    if not malformed_url_text_signal(lowered):
        return False
    width = max(0.0, bbox[2] - bbox[0])
    if width > page_width * 0.42:
        return False
    edge_band = max(24.0, page_height * 0.10)
    if not (bbox[3] <= min_y + edge_band or bbox[1] >= max_y - edge_band):
        return False
    return True


def malformed_url_text_signal(text: str) -> bool:
    if "https:/" in text and "https://" not in text:
        return True
    if "http:/" in text and "http://" not in text:
        return True
    return "avww." in text


def ocr_artifact_prunable_line(
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    observations = (line.observation, *line.contributing_observations)
    return any(
        ocr_artifact_prunable_source(observation.source) for observation in observations
    )


def ocr_artifact_prunable_source(source: str) -> bool:
    if source == "table_fusion_text":
        return True
    return source.startswith(OCR_ARTIFACT_PRUNABLE_SOURCE_PREFIXES)


def resolved_line_max_confidence(
    line: observation_resolver.ResolvedTextLine,
) -> float | None:
    values = [
        confidence
        for confidence in (
            page_geometry.numeric_confidence(observation.confidence)
            for observation in (line.observation, *line.contributing_observations)
        )
        if confidence is not None
    ]
    return max(values) if values else None


def weak_ocr_orphan_line_should_drop(
    raw_tokens: list[str],
    normalized_tokens: list[str],
    confidence: float | None,
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    if low_confidence_garbled_ocr_line_should_drop(raw_tokens, confidence, line):
        return True
    if any(any(ch.isdigit() for ch in token) for token in raw_tokens):
        return False
    if repeated_short_alpha_noise_line_should_drop(
        raw_tokens,
        line,
    ):
        return True
    if short_alpha_fragment_noise_line_should_drop(raw_tokens, line):
        return True
    if not all(token.isalpha() for token in raw_tokens):
        return False
    if len(raw_tokens) == 1:
        token = raw_tokens[0]
        return (
            len(token) <= 4
            and not token.isupper()
            and confidence_is_weak(confidence, threshold=35.0)
            and ocr_line_geometry_is_tiny(line)
        )
    if len(raw_tokens) > 4 or max(len(token) for token in normalized_tokens) > 4:
        return False
    if not all(token == token.casefold() for token in raw_tokens):
        return False
    if confidence is None:
        return any(
            observation.source == "table_fusion_text"
            for observation in (line.observation, *line.contributing_observations)
        )
    return confidence <= 65.0


def repeated_short_alpha_noise_line_should_drop(
    raw_tokens: list[str],
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    normalized_tokens = [
        alpha_token_letters(token).casefold()
        for token in raw_tokens
        if alpha_token_letters(token)
    ]
    if len(normalized_tokens) != len(raw_tokens):
        return False
    if len(normalized_tokens) < 2 or len(normalized_tokens) > 3:
        return False
    if max(len(token) for token in normalized_tokens) > 2:
        return False
    if len(set(normalized_tokens)) != 1:
        return False
    return ocr_line_geometry_is_tiny(line)


def short_alpha_fragment_noise_line_should_drop(
    raw_tokens: list[str],
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    normalized_tokens = [
        alpha_token_letters(token).casefold()
        for token in raw_tokens
        if alpha_token_letters(token)
    ]
    if len(normalized_tokens) != len(raw_tokens) or not normalized_tokens:
        return False
    if max(len(token) for token in normalized_tokens) > 2:
        return False
    return ocr_line_geometry_is_tiny(line)


def low_confidence_garbled_ocr_line_should_drop(
    raw_tokens: list[str],
    confidence: float | None,
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    if not confidence_is_weak(confidence, threshold=45.0):
        return False
    if len(raw_tokens) < 3:
        return False
    if not ocr_line_geometry_is_tiny(line):
        return False
    weird_tokens = sum(1 for token in raw_tokens if ocr_line_token_looks_garbled(token))
    return weird_tokens >= max(2, len(raw_tokens) // 2)


def ocr_line_token_looks_garbled(token: str) -> bool:
    stripped = token.strip(OCR_EDGE_NOISE_PUNCTUATION)
    if not stripped:
        return True
    alpha_only = alpha_token_letters(stripped)
    if any(ch.isdigit() for ch in stripped) and any(ch.isalpha() for ch in stripped):
        return True
    if alpha_only and ocr_text_analysis.alpha_token_looks_ocr_garbled(alpha_only):
        return True
    if alpha_only:
        rank = word_rank(alpha_only.casefold())
        if rank is None and len(alpha_only) >= 4:
            return True
    return False


def weak_ocr_artifact_token_should_prune(
    token: str,
    index: int,
    raw_tokens: list[str],
    confidence: float | None,
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    if token and not any(ch.isalnum() for ch in token):
        return True
    if token == "_":
        return True
    if (
        token.isalpha()
        and token.islower()
        and len(token) == 1
        and index in {0, len(raw_tokens) - 1}
        and any(strong_neighbor_token(neighbor) for neighbor in raw_tokens)
    ):
        return True
    if not token.isalpha():
        return False
    if not token.islower() or len(token) > 3:
        return False
    if index not in {0, len(raw_tokens) - 1}:
        return False
    if not confidence_is_weak(confidence, threshold=65.0):
        return False
    if not any(strong_neighbor_token(neighbor) for neighbor in raw_tokens):
        return False
    return ocr_line_geometry_is_tiny(line) or len(raw_tokens) >= 3


def weak_ocr_artifact_suffix_token_indexes(
    raw_tokens: list[str],
    confidence: float | None,
) -> set[int]:
    if not confidence_is_weak(confidence, threshold=60.0):
        return set()
    if len(raw_tokens) < 4:
        return set()
    prefix = raw_tokens[:2]
    if not all(header_prefix_token(token) for token in prefix):
        return set()
    suffix = raw_tokens[2:]
    if len(suffix) > 3:
        return set()
    normalized_suffix = [alpha_token_letters(token).casefold() for token in suffix]
    if not all(normalized_suffix):
        return set()
    if len(normalized_suffix) < 2:
        return set()
    if any(len(token) > 4 for token in normalized_suffix):
        return set()
    if any(token.isupper() for token in suffix):
        return set()
    weak_count = sum(
        1
        for token in normalized_suffix
        if len(token) <= 2 or (rank := word_rank(token)) is None or rank > 150_000
    )
    if weak_count < 2:
        return set()
    return set(range(2, len(raw_tokens)))


def header_prefix_token(token: str) -> bool:
    if not token.isalpha():
        return False
    normalized = alpha_token_letters(token)
    if len(normalized) < 3:
        return False
    return normalized.isupper()


def confidence_is_weak(confidence: float | None, *, threshold: float) -> bool:
    return confidence is None or confidence <= threshold


def strong_neighbor_token(token: str) -> bool:
    if len(token) < 4:
        return False
    if any(ch.isdigit() for ch in token):
        return True
    if token.isupper():
        return True
    normalized = alpha_token_letters(token).casefold()
    if len(normalized) < 4:
        return False
    rank = word_rank(normalized)
    return rank is not None and rank <= 120_000


def ocr_line_geometry_is_tiny(
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    box = line.observation.ink_bbox or line.observation.bbox
    if box is None:
        return False
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return height <= 10.5 or width <= 42.0


def text_without_token_indexes(
    text: str,
    token_matches: list[re.Match[str]],
    prune_indexes: set[int],
) -> str:
    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(token_matches):
        if index not in prune_indexes:
            pieces.append(text[cursor : match.end()])
        else:
            pieces.append(text[cursor : match.start()])
        cursor = match.end()
    pieces.append(text[cursor:])
    pruned = "".join(pieces)
    pruned = " ".join(pruned.split())
    pruned = pruned.strip()
    edge_chars = OCR_ARTIFACT_EDGE_PUNCTUATION + OCR_EDGE_NOISE_PUNCTUATION
    return pruned.rstrip(edge_chars).strip()


def postprocess_ocr_row_page_bbox(
    row: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = row.get("page_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except TypeError, ValueError:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def figure_result_region_bbox(result: Any) -> tuple[float, float, float, float] | None:
    candidate = getattr(result, "candidate", None)
    if candidate is None:
        return None
    boxes: list[tuple[float, float, float, float]] = []
    for row in getattr(candidate.result, "line_rows", ()):
        box = postprocess_ocr_row_page_bbox(row)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def figure_region_label_support_rows(
    *,
    broad_page_result: Any,
    figure_result: Any,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    broad_candidates = getattr(broad_page_result, "candidates", ()) or ()
    for candidate in broad_candidates:
        source = str(getattr(candidate, "name", ""))
        for row in getattr(candidate.result, "line_rows", ()):
            bbox = postprocess_ocr_row_page_bbox(row)
            text = str(row.get("text", "")).strip()
            if bbox is None or not text:
                continue
            key = (text, bbox)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "source": source,
                    "tokens": ocr_text_analysis.normalized_text_tokens(text),
                }
            )
    figure_candidate = getattr(figure_result, "candidate", None)
    if figure_candidate is not None:
        for row in getattr(figure_candidate.result, "line_rows", ()):
            bbox = postprocess_ocr_row_page_bbox(row)
            text = str(row.get("text", "")).strip()
            if bbox is None or not text:
                continue
            key = (text, bbox)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "source": str(
                        getattr(figure_candidate, "name", "figure_ocr_regions")
                    ),
                    "tokens": ocr_text_analysis.normalized_text_tokens(text),
                }
            )
    return tuple(rows)


def precision_clean_figure_region_label_line_text(
    line: observation_resolver.ResolvedTextLine,
    *,
    figure_region_bbox: tuple[float, float, float, float],
    support_rows: tuple[dict[str, Any], ...],
) -> str | None:
    bbox = line.observation.ink_bbox or line.observation.bbox
    if bbox is None:
        return None
    source = line.observation.source
    if source == "figure_ocr_regions":
        return None
    if not (source.startswith("rendered_page_") or source == "table_fusion_text"):
        return None
    if (
        page_geometry.observation_overlap_ratio(
            page_geometry.PageObservation(
                kind="ocr_textline",
                source=source,
                bbox=bbox,
                advance_bbox=bbox,
                ink_bbox=bbox,
                text=line.text,
            ),
            page_geometry.PageObservation(
                kind="ocr_textline",
                source="figure_region",
                bbox=figure_region_bbox,
                advance_bbox=figure_region_bbox,
                ink_bbox=figure_region_bbox,
                text="",
            ),
            denominator="smaller",
        )
        < 0.18
    ):
        return None
    raw_tokens = [
        match.group(0) for match in ocr_text_analysis.TEXT_TOKEN_RE.finditer(line.text)
    ]
    if len(raw_tokens) < 3 or len(raw_tokens) > 6:
        return None
    alpha_tokens = [token for token in raw_tokens if token.isalpha()]
    if len(alpha_tokens) < 2:
        return None
    if not any(any(ch.isdigit() for ch in token) for token in raw_tokens) and all(
        token.isalpha() and len(token) >= 3 for token in raw_tokens
    ):
        return None
    support = overlapping_support_token_counts(
        bbox,
        support_rows=support_rows,
    )
    cleaned_tokens = figure_region_cleaned_label_tokens(
        raw_tokens,
        support=support,
    )
    if cleaned_tokens is None:
        return None
    replacement = " ".join(cleaned_tokens)
    if ocr_text_analysis.normalized_text_tokens(
        replacement
    ) == ocr_text_analysis.normalized_text_tokens(line.text):
        return None
    return replacement


def overlapping_support_token_counts(
    bbox: tuple[float, float, float, float],
    *,
    support_rows: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    fragment_numeric: Counter[str] = Counter()
    for row in support_rows:
        row_bbox = row["bbox"]
        if not support_row_overlaps_label_bbox(bbox, row_bbox):
            continue
        tokens = row["tokens"]
        counts.update(set(tokens))
        width = max(0.0, row_bbox[2] - row_bbox[0])
        row_token_count = len(tokens)
        if row_token_count <= 2 or width <= max(18.0, (bbox[2] - bbox[0]) * 0.18):
            for token in tokens:
                if token.isdigit():
                    fragment_numeric[token] += 1
    return {
        "counts": dict(counts),
        "fragment_numeric": dict(fragment_numeric),
    }


def support_row_overlaps_label_bbox(
    line_bbox: tuple[float, float, float, float],
    row_bbox: tuple[float, float, float, float],
) -> bool:
    vertical_overlap = min(line_bbox[3], row_bbox[3]) - max(line_bbox[1], row_bbox[1])
    if vertical_overlap <= 0.0:
        return False
    line_height = max(1.0, line_bbox[3] - line_bbox[1])
    row_height = max(1.0, row_bbox[3] - row_bbox[1])
    if vertical_overlap < min(line_height, row_height) * 0.40:
        return False
    horizontal_overlap = min(line_bbox[2], row_bbox[2]) - max(line_bbox[0], row_bbox[0])
    return horizontal_overlap > 0.0


def figure_region_cleaned_label_tokens(
    raw_tokens: list[str],
    *,
    support: dict[str, dict[str, int]],
) -> list[str] | None:
    alpha_kept = [
        token
        for token in raw_tokens
        if token.isalpha() and (len(token) >= 4 or token.isupper())
    ]
    if len(alpha_kept) < 2:
        return None
    counts = support["counts"]
    fragment_numeric = support["fragment_numeric"]
    numeric_tokens = sorted(
        (token for token, count in fragment_numeric.items() if count >= 1),
        key=lambda token: (-fragment_numeric[token], token),
    )
    cleaned: list[str] = []
    for token in raw_tokens:
        normalized = token.casefold()
        if token.isalpha():
            if token in alpha_kept:
                cleaned.append(token)
            elif counts.get(normalized, 0) >= 2 and len(token) >= 4:
                cleaned.append(token)
            continue
        if token.isdigit():
            if fragment_numeric.get(normalized, 0) >= 1:
                cleaned.append(token)
            continue
    if numeric_tokens and not any(token.isdigit() for token in cleaned):
        cleaned.append(numeric_tokens[0])
    if len(cleaned) < 2:
        return None
    return cleaned


def embedded_image_text_band_bboxes(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    boxes = [
        line.observation.bbox
        for line in lines
        if line.observation.source == "embedded_image_text"
        and line.observation.bbox is not None
    ]
    if not boxes:
        return ()
    sorted_boxes = sorted(boxes, key=lambda box: (-box[3], box[0]))
    bands: list[list[tuple[float, float, float, float]]] = []
    for box in sorted_boxes:
        if not bands or not boxes_share_vertical_band(bands[-1][-1], box):
            bands.append([box])
            continue
        bands[-1].append(box)
    return tuple(
        (
            min(box[0] for box in group),
            min(box[1] for box in group),
            max(box[2] for box in group),
            max(box[3] for box in group),
        )
        for group in bands
    )


def boxes_share_vertical_band(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    overlap = min(left[3], right[3]) - max(left[1], right[1])
    min_height = min(left[3] - left[1], right[3] - right[1])
    return overlap >= max(6.0, min_height * 0.30)


def embedded_image_band_noise_line_should_drop(
    line: observation_resolver.ResolvedTextLine,
    *,
    embedded_bands: tuple[tuple[float, float, float, float], ...],
    page_width: float,
) -> bool:
    observation = line.observation
    bbox = observation.bbox
    if bbox is None:
        return False
    if observation.kind != "table_ocr_line":
        return False
    if not observation.source.startswith("rendered_page_"):
        return False
    width = max(0.0, bbox[2] - bbox[0])
    if width > page_width * 0.24:
        return False
    if bbox[2] > page_width * 0.32:
        return False
    if line_readable_token_count(line.text) > 1:
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(line.text)
    if not tokens or not all(token.isalpha() for token in tokens):
        return False
    if short_token_ratio(tokens) < 0.50:
        return False
    return any(
        embedded_image_band_noise_line_overlaps_band(
            bbox,
            band_bbox=band_bbox,
            page_width=page_width,
        )
        for band_bbox in embedded_bands
    )


def embedded_image_band_noise_line_overlaps_band(
    bbox: tuple[float, float, float, float],
    *,
    band_bbox: tuple[float, float, float, float],
    page_width: float,
) -> bool:
    vertical_overlap = min(bbox[3], band_bbox[3]) - max(bbox[1], band_bbox[1])
    if vertical_overlap <= 0.0:
        return False
    line_height = max(1.0, bbox[3] - bbox[1])
    if vertical_overlap < line_height * 0.45:
        return False
    return bbox[2] <= band_bbox[2] + max(18.0, page_width * 0.04)


def short_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if len(token) <= 2) / len(tokens)


def trim_geometryless_table_fusion_date_noise_text(
    text: str,
    line: observation_resolver.ResolvedTextLine,
) -> str | None:
    if line.observation.source != "table_fusion_text":
        return None
    if line.observation.bbox is not None or line.observation.ink_bbox is not None:
        return None
    raw_tokens = [
        match.group(0) for match in ocr_text_analysis.TEXT_TOKEN_RE.finditer(text)
    ]
    if len(raw_tokens) < 6:
        return None
    month_index = next(
        (
            index
            for index, token in enumerate(raw_tokens)
            if token.casefold() in MONTH_NAME_TOKENS
        ),
        None,
    )
    if month_index is None or month_index < 3:
        return None
    prefix_tokens = raw_tokens[:month_index]
    if geometryless_table_fusion_prefix_noise_ratio(prefix_tokens) < 0.75:
        return None
    date_text = month_day_year_text(raw_tokens[month_index:])
    return date_text if date_text is not None else None


def geometryless_table_fusion_noise_line_should_drop(
    text: str,
    line: observation_resolver.ResolvedTextLine,
) -> bool:
    if line.observation.source != "table_fusion_text":
        return False
    if line.observation.bbox is not None or line.observation.ink_bbox is not None:
        return False
    raw_tokens = [
        match.group(0) for match in ocr_text_analysis.TEXT_TOKEN_RE.finditer(text)
    ]
    if len(raw_tokens) < 2 or len(raw_tokens) > 6:
        return False
    if any(any(ch.isdigit() for ch in token) for token in raw_tokens):
        return False
    alpha_tokens = [alpha_token_letters(token) for token in raw_tokens]
    if not all(alpha_tokens):
        return False
    readable_count = line_readable_token_count(text)
    if readable_count >= 2:
        return False
    total_alpha_chars = sum(len(token) for token in alpha_tokens)
    weak_tokens = sum(
        1
        for token in alpha_tokens
        if len(token) <= 3
        or ocr_text_analysis.alpha_token_looks_ocr_garbled(token)
        or word_rank(token.casefold()) is None
    )
    return weak_tokens >= len(alpha_tokens) - 1 and total_alpha_chars <= 12


def geometryless_table_fusion_prefix_noise_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        alpha_only = alpha_token_letters(token)
        if not alpha_only:
            noisy += 1
            continue
        normalized = alpha_only.casefold()
        if (
            ocr_text_analysis.alpha_token_looks_ocr_garbled(alpha_only)
            or len(normalized) <= 3
            or word_rank(normalized) is None
        ):
            noisy += 1
    return noisy / len(tokens)


def month_day_year_text(tokens: list[str]) -> str | None:
    if len(tokens) < 3:
        return None
    month = tokens[0]
    if month.casefold() not in MONTH_NAME_TOKENS:
        return None
    day_index = next(
        (
            index
            for index, token in enumerate(tokens[1:4], start=1)
            if token.isdigit() and 1 <= len(token) <= 2
        ),
        None,
    )
    if day_index is None:
        return None
    year_index = next(
        (
            index
            for index, token in enumerate(
                tokens[day_index + 1 : day_index + 4], start=day_index + 1
            )
            if token.isdigit() and len(token) == 4
        ),
        None,
    )
    if year_index is None:
        return None
    day = tokens[day_index]
    year = tokens[year_index]
    return f"{month} {day}, {year}"


def text_with_only_token_indexes(
    text: str,
    token_matches: list[re.Match[str]],
    keep_indexes: set[int],
) -> str:
    pieces: list[str] = []
    for index, match in enumerate(token_matches):
        if index not in keep_indexes:
            continue
        pieces.append(match.group(0))
    selected = " ".join(pieces).strip()
    return selected.rstrip(OCR_ARTIFACT_EDGE_PUNCTUATION).strip()


def resolved_line_with_pruned_text(
    line: observation_resolver.ResolvedTextLine,
    text: str,
) -> observation_resolver.ResolvedTextLine:
    observation = replace(
        line.observation,
        text=text,
        provenance=(
            *line.observation.provenance,
            *page_geometry.provenance_tuple(ocr_artifact_pruned=True),
        ),
    )
    return observation_resolver.ResolvedTextLine(
        text,
        observation,
        break_before=line.break_before,
        contributing_observations=line.contributing_observations,
        resolution=line.resolution,
    )


def resolved_line_with_repaired_text(
    line: observation_resolver.ResolvedTextLine,
    text: str,
) -> observation_resolver.ResolvedTextLine:
    observation = replace(
        line.observation,
        text=text,
        provenance=(
            *line.observation.provenance,
            *page_geometry.provenance_tuple(document_local_token_repaired=True),
        ),
    )
    return observation_resolver.ResolvedTextLine(
        text,
        observation,
        break_before=line.break_before,
        contributing_observations=line.contributing_observations,
        resolution=line.resolution,
    )


def should_ocr_fallback(page: PageExtractionHost, text: str) -> bool:
    if not text.strip():
        return True
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if native_text_layer_looks_reliable_enough(
        page,
        text,
        text_tokens=text_tokens,
    ):
        return False
    native_geometry_summary = native_layout_geometry_summary_from_page_cache(page)
    if text_tokens <= 20:
        if layout_geometry_should_trigger_ocr(
            native_geometry_summary,
            text_tokens=text_tokens,
        ):
            return True
        if ocr_text_analysis.sparse_text_looks_noisy(text):
            return True
        try:
            return ocr_page_analysis.has_dominant_page_image(
                page
            ) or ocr_page_analysis.has_uninterpretable_type3_fonts(page)
        except Exception:
            return False
    try:
        if ocr_page_analysis.scanned_table_native_text_layer_looks_weak(
            page,
            text,
            text_tokens=text_tokens,
        ):
            return True
    except Exception:
        pass
    if ocr_page_analysis.dense_numeric_native_text_layer_is_preferable(
        text, text_tokens=text_tokens
    ):
        return False
    if layout_geometry_should_trigger_ocr(
        native_geometry_summary,
        text_tokens=text_tokens,
    ):
        return True
    cache = getattr(page, "extraction_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("native_visible_row_band_filter_applied") is True
        and text_tokens >= 250
        and ocr_text_analysis.text_ocr_quality_score(text) >= 0.1
    ):
        return False
    dominant_image = False
    if (
        text_tokens <= ocr_page_analysis.OCR_FALLBACK_SPARSE_TEXT_TOKENS
        and ocr_text_analysis.sparse_text_looks_noisy(text)
    ):
        try:
            dominant_image = ocr_page_analysis.has_dominant_page_image(page)
            if dominant_image:
                return True
        except Exception:
            pass
    try:
        if ocr_page_analysis.dominant_image_text_layer_looks_weak(text) and (
            dominant_image or ocr_page_analysis.has_dominant_page_image(page)
        ):
            return True
    except Exception:
        pass
    try:
        if ocr_page_analysis.symbol_encoded_native_text_layer_looks_weak(page, text):
            return True
    except Exception:
        pass
    try:
        return ocr_page_analysis.has_uninterpretable_type3_fonts(page)
    except Exception:
        return False


def native_text_layer_looks_reliable_enough(
    page: PageExtractionHost,
    text: str,
    *,
    text_tokens: int | None = None,
) -> bool:
    if text_tokens is None:
        text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens < 100:
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    if quality > 0.12:
        return False
    artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    if artifact > 0.02:
        return False
    if text_tokens < 430:
        return True
    try:
        return ocr_page_analysis.native_text_layer_has_substantial_page_coverage(
            page,
            text_tokens,
        )
    except Exception:
        return False


def native_layout_geometry_summary_from_page_cache(
    page: PageExtractionHost,
) -> Any | None:
    cache = getattr(page, "extraction_cache", None)
    if not isinstance(cache, dict):
        return None
    record = cache.get("native_layout_geometry_summary")
    if not isinstance(record, dict):
        return None
    return layout_geometry_summary_from_record(record)


def should_try_ocr_supplement(page: PageExtractionHost, text: str) -> bool:
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if native_text_layer_looks_reliable_enough(
        page,
        text,
        text_tokens=text_tokens,
    ):
        return False
    if text_tokens < 80:
        return tiny_native_text_should_trigger_ocr_supplement(
            page,
            text,
            text_tokens=text_tokens,
        )
    cache = getattr(page, "extraction_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("native_visible_row_band_filter_applied") is True
        and text_tokens >= 250
        and ocr_text_analysis.text_ocr_quality_score(text) >= 0.1
    ):
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    if 80 <= text_tokens <= 150:
        if quality > 0.22:
            return False
        if not ocr_text_analysis.text_has_many_digit_lines(text):
            return False
    elif text_tokens <= 700:
        if quality > 0.16:
            return False
        if ocr_text_analysis.numeric_token_ratio(text) >= 0.10:
            return False
        if ocr_page_analysis.native_text_layer_has_substantial_page_coverage(
            page, text_tokens
        ):
            return False
        try:
            if ocr_page_analysis.page_has_large_embedded_image(
                page
            ) and ocr_page_analysis.native_text_layer_has_sparse_page_coverage(page):
                return True
        except Exception:
            pass
        return False
    else:
        return False
    try:
        if ocr_page_analysis.has_dominant_page_image(page):
            return True
        if ocr_page_analysis.page_has_many_non_image_drawings(
            page
        ) and ocr_text_analysis.sparse_text_looks_noisy(text):
            return True
    except Exception:
        pass
    return False


def tiny_native_text_should_trigger_ocr_supplement(
    page: PageExtractionHost,
    text: str,
    *,
    text_tokens: int,
) -> bool:
    if text_tokens == 0 or text_tokens > 24:
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    if quality > 0.75:
        return False
    try:
        if ocr_page_analysis.has_dominant_page_image(page):
            return True
        if ocr_page_analysis.page_has_large_embedded_image(
            page
        ) and ocr_page_analysis.native_text_layer_has_sparse_page_coverage(page):
            return True
        if ocr_page_analysis.page_has_many_non_image_drawings(page):
            return ocr_text_analysis.sparse_text_looks_noisy(text)
    except Exception:
        return False
    return False


def should_try_figure_ocr_supplement(
    page: PageExtractionHost,
    text: str,
) -> bool:
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens < 80 or text_tokens > 1400:
        return False
    if native_text_layer_looks_reliable_enough(
        page,
        text,
        text_tokens=text_tokens,
    ):
        return False
    try:
        return ocr_page_analysis.page_has_figure_ocr_region(page)
    except Exception:
        return False


def figure_ocr_region_is_full_page_image(region: Any) -> bool:
    signals = getattr(region, "signals", None)
    if not isinstance(signals, dict):
        return False
    if signals.get("image_only_full_page_region") is True:
        return True
    area_ratio = signals.get("area_ratio")
    if area_ratio is None:
        return False
    try:
        return float(area_ratio) >= 0.75
    except (TypeError, ValueError):
        return False


def should_try_embedded_image_text_ocr_supplement(
    page: PageExtractionHost,
    text: str,
) -> bool:
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens < 80 or text_tokens > 1400:
        return False
    extract_images = getattr(page, "extract_images", None)
    if not callable(extract_images):
        return False
    try:
        images = extract_images()
    except Exception:
        return False
    page_space = page_geometry.PageSpace.from_page(page)
    page_area = page_geometry.rect_area(page_space.bbox) if page_space else 1.0
    for image in images:
        bbox = page_geometry.normalize_rect(image.get("bbox"))
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 0.0 or height <= 0.0:
            continue
        pixels = int(image.get("pixels") or 0)
        area = width * height
        area_ratio = area / max(page_area, 1.0)
        pixel_density = pixels / max(area, 1.0)
        if embedded_image_text_supplement_image_shape_is_eligible(
            pixels,
            width=width,
            height=height,
            area_ratio=area_ratio,
            pixel_density=pixel_density,
        ):
            return True
    return False


def embedded_image_text_supplement_image_shape_is_eligible(
    pixels: int,
    *,
    width: float,
    height: float,
    area_ratio: float,
    pixel_density: float,
) -> bool:
    if width <= 0.0 or height <= 0.0:
        return False
    if pixel_density < OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXEL_DENSITY:
        return False
    aspect = width / height
    if (
        OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS
        <= pixels
        <= OCR_EMBEDDED_IMAGE_TEXT_MAX_PIXELS
        and area_ratio <= OCR_EMBEDDED_IMAGE_TEXT_MAX_AREA_RATIO
        and OCR_EMBEDDED_IMAGE_TEXT_MIN_ASPECT_RATIO
        <= aspect
        <= OCR_EMBEDDED_IMAGE_TEXT_MAX_ASPECT_RATIO
    ):
        return True
    return (
        OCR_EMBEDDED_IMAGE_TEXT_MIN_PIXELS
        <= pixels
        <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_PIXELS
        and area_ratio <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_AREA_RATIO
        and OCR_EMBEDDED_IMAGE_TEXT_WIDE_MIN_ASPECT_RATIO
        <= aspect
        <= OCR_EMBEDDED_IMAGE_TEXT_WIDE_MAX_ASPECT_RATIO
    )


def embedded_image_text_ocr_supplemental_resolved_lines(
    page: PageExtractionHost,
    text: str,
    ocr_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if ocr_result.candidate is None or not ocr_result.text.strip():
        return ()
    ocr_lines = ocr_geometry.ocr_candidate_textline_geometry_lines(
        page,
        ocr_result.candidate,
    )
    if not ocr_lines:
        return embedded_image_text_resolved_lines_by_text(text, ocr_result.text)
    try:
        native_lines = ocr_page_analysis.native_text_geometry_lines(page)
    except Exception:
        native_lines = []
    accepted_observations = text_line_observations(
        native_lines,
        source="native_text",
        kind="native_line",
    )
    seen_tokens = set(ocr_text_analysis.normalized_text_tokens(text))
    supplemental_lines: list[observation_resolver.ResolvedTextLine] = []
    for line in ocr_lines:
        ocr_observation = page_geometry.page_observation_from_text_line(
            line,
            source=ocr_result.candidate.name,
            kind="ocr_textline",
        )
        if ocr_observation is None:
            continue
        resolution = observation_resolver.resolve_observation_append(
            ocr_observation,
            accepted_observations,
            existing_text=text,
        )
        if resolution.action != "append":
            continue
        stripped = str(getattr(line, "text", "")).strip()
        confidence = getattr(line, "confidence", None)
        if not embedded_image_text_line_should_append(
            stripped,
            seen_tokens,
            confidence=confidence,
        ):
            continue
        supplemental_lines.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                ocr_observation,
                break_before=1,
                contributing_observations=(ocr_observation,),
                resolution=resolution,
            )
        )
        accepted_observations.append(ocr_observation)
        seen_tokens.update(ocr_text_analysis.normalized_text_tokens(stripped))
    return tuple(supplemental_lines)


def embedded_image_text_resolved_lines_by_text(
    text: str,
    ocr_text: str,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    seen_tokens = set(ocr_text_analysis.normalized_text_tokens(text))
    supplemental_lines: list[observation_resolver.ResolvedTextLine] = []
    for line_index, line in enumerate(ocr_text.splitlines()):
        stripped = line.strip()
        if not embedded_image_text_line_should_append(stripped, seen_tokens):
            continue
        observation = page_geometry.PageObservation(
            kind="ocr_textline",
            source="embedded_image_text",
            text=stripped,
            provenance=page_geometry.provenance_tuple(line_index=line_index),
        )
        supplemental_lines.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
        seen_tokens.update(ocr_text_analysis.normalized_text_tokens(stripped))
    return tuple(supplemental_lines)


def embedded_image_text_line_should_append(
    line: str,
    seen_tokens: set[str],
    *,
    confidence: int | None = None,
) -> bool:
    tokens = ocr_text_analysis.normalized_text_tokens(line)
    confidence_value = confidence if confidence is not None else 0
    if len(tokens) == 1:
        return confidence_value >= 45 and embedded_image_text_single_token_label(line)
    if len(tokens) < 2:
        return False
    if ocr_text_analysis.supplemental_ocr_line_looks_fragmentary(tokens, line):
        return False
    if ocr_text_analysis.text_ocr_quality_score(line) > 0.34:
        return False
    new_tokens = [token for token in tokens if token not in seen_tokens]
    if new_tokens:
        return any(ocr_supplement_new_token_looks_useful(token) for token in new_tokens)
    if "&" in line:
        return True
    return confidence_value >= 35 and embedded_image_text_line_looks_like_label(
        line,
        tokens,
    )


def embedded_image_text_single_token_label(line: str) -> bool:
    compact = "".join(ch for ch in line if ch.isalpha())
    if not compact or compact != compact.upper():
        return False
    if not 2 <= len(compact) <= 6:
        return False
    unique = set(compact)
    if len(unique) <= 1:
        return False
    return max(compact.count(ch) for ch in unique) / len(compact) <= 0.55


def embedded_image_text_line_looks_like_label(line: str, tokens: list[str]) -> bool:
    alpha_chars = [ch for ch in line if ch.isalpha()]
    if not alpha_chars:
        return False
    uppercase = sum(1 for ch in alpha_chars if ch.isupper())
    if uppercase / len(alpha_chars) < 0.70:
        return False
    if len(tokens) > 9 or len(line) > 96:
        return False
    return any(len(token) >= 4 for token in tokens)


def should_replace_text_with_figure_ocr(
    page: PageExtractionHost,
    text: str,
    ocr_result: OcrPageTextResult,
) -> bool:
    if not figure_ocr_page_result_looks_useful(ocr_result):
        return False
    if getattr(page, "chars", ()):
        return False
    try:
        if not ocr_page_analysis.has_dominant_page_image(page):
            return False
    except Exception:
        return False
    figure_text = ocr_result.text
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    figure_tokens = ocr_text_analysis.extracted_text_token_count(figure_text)
    if figure_tokens < 40:
        return False
    current_quality = ocr_text_analysis.text_ocr_quality_score(text)
    figure_quality = ocr_text_analysis.text_ocr_quality_score(figure_text)
    current_artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    figure_artifact = ocr_text_analysis.scanned_ocr_artifact_score(figure_text)
    current_gibberish = ocr_text_analysis.alphabetic_gibberish_score(text)
    figure_gibberish = ocr_text_analysis.alphabetic_gibberish_score(figure_text)
    if figure_tokens < int(text_tokens * 0.65):
        if figure_tokens < int(text_tokens * 0.35):
            return False
        if not (
            figure_quality + 0.08 < current_quality
            or figure_artifact + 0.12 < current_artifact
            or figure_gibberish + 0.04 < current_gibberish
        ):
            return False
    if figure_quality + 0.01 < current_quality:
        return True
    return figure_artifact + 0.03 < current_artifact


def should_try_vector_stroke_ocr(page: PageExtractionHost, text: str) -> bool:
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    cache = getattr(page, "extraction_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("native_visible_row_band_filter_applied") is True
        and text_tokens >= 250
        and ocr_text_analysis.text_ocr_quality_score(text) >= 0.1
    ):
        return False
    if text_tokens > 250 and not ocr_text_analysis.sparse_text_looks_noisy(text):
        return False
    try:
        return page_has_vector_stroke_text_candidates(page)
    except Exception:
        return False


def should_try_full_ocr_after_vector_stroke(
    vector_result: VectorStrokeOcrResult,
) -> bool:
    vector_text = vector_result.text
    vector_tokens = ocr_text_analysis.extracted_text_token_count(vector_text)
    if vector_tokens < 8:
        return False
    if vector_stroke_text_looks_fragmented(vector_text):
        return ocr_text_analysis.vector_text_supports_schematic_tiled_ocr(vector_text)
    if (
        ocr_text_analysis.vector_text_supports_schematic_tiled_ocr(vector_text)
        and (vector_result.confidence or 0) < 75
    ):
        return True
    return not should_trust_vector_stroke_text_without_full_ocr(vector_result)


def should_trust_vector_stroke_text_without_full_ocr(
    vector_result: VectorStrokeOcrResult,
) -> bool:
    vector_text = vector_result.text
    if ocr_text_analysis.extracted_text_token_count(vector_text) < 400:
        return False
    if (vector_result.confidence or 0) < 75:
        return False
    return ocr_text_analysis.vector_text_supports_schematic_tiled_ocr(vector_text)


def dense_numeric_ocr_supplement_would_reduce_recall(text: str, ocr_text: str) -> bool:
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens < 250:
        return False
    native_numeric_ratio = ocr_text_analysis.numeric_token_ratio(text)
    if native_numeric_ratio < 0.30 and not ocr_text_analysis.text_has_many_digit_lines(
        text
    ):
        return False
    ocr_tokens = ocr_text_analysis.extracted_text_token_count(ocr_text)
    native_artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    ocr_artifact = ocr_text_analysis.scanned_ocr_artifact_score(ocr_text)
    if ocr_tokens < int(text_tokens * 1.15) and ocr_artifact + 0.05 >= native_artifact:
        return True
    if ocr_tokens < max(80, int(text_tokens * 0.82)):
        return True
    return ocr_text_analysis.numeric_token_ratio(ocr_text) + 0.08 < native_numeric_ratio


def ocr_page_result_resolved_lines(
    page: PageExtractionHost,
    ocr_result: OcrPageTextResult,
    *,
    source: str | None = None,
    kind: str = "ocr_textline",
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    candidate = ocr_result.candidate
    candidate_source = source or (candidate.name if candidate is not None else "ocr")
    split_band_lines = candidate_multi_column_band_split_lines(candidate)
    output_lines = getattr(ocr_result, "output_lines", ())
    if split_band_lines and (
        not output_lines or len(split_band_lines) >= len(output_lines) + 8
    ):
        return resolved_text_lines_from_geometry_lines(
            split_band_lines,
            source=f"{candidate_source}:band_split",
            kind=kind,
        )
    if output_lines:
        return tuple(output_lines)
    geometry_lines = ocr_geometry.ocr_candidate_geometry_lines(page, candidate)
    if geometry_lines:
        return resolved_text_lines_from_geometry_lines(
            geometry_lines,
            source=candidate_source,
            kind=kind,
        )
    return resolved_text_lines_from_strings(
        ocr_result.text.splitlines(),
        source=candidate_source,
        kind=kind,
    )


def resolved_text_lines_from_geometry_lines(
    lines: list[Any] | tuple[Any, ...],
    *,
    source: str,
    kind: str,
    resolve: bool = True,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    output_lines: list[observation_resolver.ResolvedTextLine] = []
    for line_index, line in enumerate(lines):
        stripped = str(getattr(line, "text", "")).strip()
        if not stripped:
            continue
        observation = page_geometry.page_observation_from_text_line(
            line,
            source=source,
            kind=kind,
            line_index=line_index,
        )
        if observation is None:
            observation = page_geometry.PageObservation(
                kind=kind,
                source=source,
                text=stripped,
                confidence=page_geometry.numeric_confidence(
                    getattr(line, "confidence", None)
                ),
                provenance=page_geometry.provenance_tuple(line_index=line_index),
            )
        output_lines.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
    if not resolve:
        return tuple(output_lines)
    return observation_resolver.resolve_text_lines(output_lines)


def candidate_multi_column_band_split_lines(
    candidate: Any,
) -> tuple[OcrSyntheticTextLine, ...]:
    if candidate is None or getattr(candidate, "region_count", 0) != 0:
        return ()
    candidate_name = str(getattr(candidate, "name", ""))
    if candidate_name != "full_page_simple":
        return ()
    line_word_rows = candidate_line_word_rows(candidate)
    if len(line_word_rows) < 8:
        return ()
    page_bbox = candidate_page_bbox(line_word_rows, candidate)
    if page_bbox is None:
        return ()
    page_width = max(0.0, page_bbox[2] - page_bbox[0])
    page_height = max(0.0, page_bbox[3] - page_bbox[1])
    if page_width <= 0.0 or page_height <= 0.0:
        return ()
    ordered_rows = ordered_candidate_line_word_rows(line_word_rows)
    groups = vertical_line_word_groups(ordered_rows, page_height=page_height)
    if not groups:
        return ()
    split_any = False
    synthetic_lines: list[OcrSyntheticTextLine] = []
    for group in groups:
        segment = dominant_multi_column_band_segment(
            group,
            page_width=page_width,
            page_height=page_height,
        )
        if segment is None:
            synthetic_lines.extend(row_to_synthetic_lines(group))
            continue
        start_index, end_index, split = segment
        if start_index > 0:
            synthetic_lines.extend(row_to_synthetic_lines(group[:start_index]))
        band_lines, split_count = split_multi_column_band_group(
            group[start_index:end_index],
            split_x=split,
            page_width=page_width,
        )
        if split_count < 3:
            synthetic_lines.extend(row_to_synthetic_lines(group))
            continue
        split_any = True
        synthetic_lines.extend(band_lines)
        if end_index < len(group):
            synthetic_lines.extend(row_to_synthetic_lines(group[end_index:]))
    return tuple(synthetic_lines) if split_any else ()


def resolved_text_lines_from_strings(
    lines: list[str] | tuple[str, ...],
    *,
    source: str,
    kind: str,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    output_lines: list[observation_resolver.ResolvedTextLine] = []
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        observation = page_geometry.PageObservation(
            kind=kind,
            source=source,
            text=stripped,
            provenance=page_geometry.provenance_tuple(line_index=line_index),
        )
        output_lines.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                observation,
                break_before=1,
                contributing_observations=(observation,),
            )
        )
    return observation_resolver.resolve_text_lines(output_lines)


def ordered_candidate_line_word_rows(
    rows: tuple[OcrLineRowWords, ...],
) -> tuple[OcrLineRowWords, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -(row.bbox[1] + row.bbox[3]) * 0.5,
                row.bbox[0],
                row.bbox[2],
            ),
        )
    )


def candidate_page_bbox(
    rows: tuple[OcrLineRowWords, ...],
    candidate: Any,
) -> tuple[float, float, float, float] | None:
    candidate_bbox = page_geometry.rect_box_tuple(getattr(candidate, "page_bbox", None))
    if candidate_bbox is not None:
        return candidate_bbox
    if not rows:
        return None
    return (
        min(row.bbox[0] for row in rows),
        min(row.bbox[1] for row in rows),
        max(row.bbox[2] for row in rows),
        max(row.bbox[3] for row in rows),
    )


def vertical_line_word_groups(
    rows: tuple[OcrLineRowWords, ...],
    *,
    page_height: float,
) -> tuple[tuple[OcrLineRowWords, ...], ...]:
    if not rows:
        return ()
    heights = [row.bbox[3] - row.bbox[1] for row in rows if row.bbox[3] > row.bbox[1]]
    median_height = median(heights) if heights else 12.0
    gap_threshold = max(20.0, median_height * 2.8, page_height * 0.014)
    groups: list[list[OcrLineRowWords]] = []
    current: list[OcrLineRowWords] = []
    previous_center_y: float | None = None
    for row in rows:
        center_y = (row.bbox[1] + row.bbox[3]) * 0.5
        if (
            previous_center_y is not None
            and previous_center_y - center_y > gap_threshold
        ):
            if current:
                groups.append(current)
            current = [row]
        else:
            current.append(row)
        previous_center_y = center_y
    if current:
        groups.append(current)
    return tuple(tuple(group) for group in groups)


def dominant_multi_column_band_segment(
    group: tuple[OcrLineRowWords, ...],
    *,
    page_width: float,
    page_height: float,
) -> tuple[int, int, float] | None:
    if len(group) < 8:
        return None
    band_bbox = rows_union_bbox(group)
    if band_bbox is None:
        return None
    band_width = band_bbox[2] - band_bbox[0]
    if band_width < page_width * 0.45:
        return None
    row_splits = [
        row_column_split_midpoint(row, band_width=band_width) for row in group
    ]
    split_candidates = [split for split in row_splits if split is not None]
    if len(split_candidates) < 6:
        return None
    bucket_width = max(18.0, band_width * 0.04)
    buckets: Counter[int] = Counter(
        int(round(split / bucket_width)) for split in split_candidates
    )
    band_center_x = (band_bbox[0] + band_bbox[2]) * 0.5
    bucket = max(
        buckets,
        key=lambda key: (
            sum(
                count
                for other_key, count in buckets.items()
                if abs(other_key - key) <= 1
            ),
            -abs((key * bucket_width) - band_center_x),
        ),
    )
    support_indexes = [
        index
        for index, split in enumerate(row_splits)
        if split is not None and abs(int(round(split / bucket_width)) - bucket) <= 1
    ]
    runs = contiguous_index_runs(support_indexes, max_gap=1)
    if not runs:
        return None
    start_index, end_index = max(runs, key=lambda run: run[1] - run[0])
    segment = group[start_index : end_index + 1]
    if len(segment) < 8:
        return None
    segment_bbox = rows_union_bbox(segment)
    if segment_bbox is None:
        return None
    segment_height = segment_bbox[3] - segment_bbox[1]
    if segment_height < max(40.0, page_height * 0.06):
        return None
    if segment_height / max(1.0, page_height) > 0.24:
        return None
    clustered = [
        split for split in row_splits[start_index : end_index + 1] if split is not None
    ]
    split_x = median(clustered)
    gutter_half = max(12.0, band_width * 0.025)
    left = 0
    right = 0
    crossing = 0
    for row in segment:
        if row.bbox[2] <= split_x - gutter_half:
            left += 1
        elif row.bbox[0] >= split_x + gutter_half:
            right += 1
        elif row_has_words_on_both_sides(row, split_x=split_x, gutter_half=gutter_half):
            left += 1
            right += 1
        else:
            crossing += 1
    if left < 6 or right < 6:
        return None
    if crossing > max(3, int(len(segment) * 0.20)):
        return None
    return start_index, end_index + 1, split_x


def contiguous_index_runs(
    indexes: list[int],
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    if not indexes:
        return []
    runs: list[tuple[int, int]] = []
    start = indexes[0]
    previous = indexes[0]
    for index in indexes[1:]:
        if index - previous <= max_gap + 1:
            previous = index
            continue
        runs.append((start, previous))
        start = index
        previous = index
    runs.append((start, previous))
    return runs


def row_column_split_midpoint(
    row: OcrLineRowWords,
    *,
    band_width: float,
) -> float | None:
    if len(row.words) < 4:
        return None
    best_gap = 0.0
    best_midpoint: float | None = None
    min_gap = max(22.0, band_width * 0.05)
    for index in range(len(row.words) - 1):
        left_words = row.words[: index + 1]
        right_words = row.words[index + 1 :]
        if not row_words_support_column_text(left_words):
            continue
        if not row_words_support_column_text(right_words):
            continue
        gap = row.words[index + 1].bbox[0] - row.words[index].bbox[2]
        if gap < min_gap or gap <= best_gap:
            continue
        best_gap = gap
        best_midpoint = (row.words[index].bbox[2] + row.words[index + 1].bbox[0]) * 0.5
    return best_midpoint


def row_words_support_column_text(words: tuple[OcrLineWordRow, ...]) -> bool:
    if not words:
        return False
    contentful = [word for word in words if row_word_is_contentful(word)]
    if len(contentful) >= 2:
        return True
    if len(contentful) == 1:
        token = contentful[0].text.strip(OCR_EDGE_NOISE_PUNCTUATION)
        alpha_only = alpha_token_letters(token)
        if len(alpha_only) >= 5:
            return True
    numeric_like = [word for word in words if word_text_is_numeric_like(word.text)]
    return len(numeric_like) >= 2


def row_has_words_on_both_sides(
    row: OcrLineRowWords,
    *,
    split_x: float,
    gutter_half: float,
) -> bool:
    left_words = tuple(
        word for word in row.words if word_center_x(word) <= split_x - gutter_half
    )
    right_words = tuple(
        word for word in row.words if word_center_x(word) >= split_x + gutter_half
    )
    return row_words_support_column_text(left_words) and row_words_support_column_text(
        right_words
    )


def split_multi_column_band_group(
    group: tuple[OcrLineRowWords, ...],
    *,
    split_x: float,
    page_width: float,
) -> tuple[list[OcrSyntheticTextLine], int]:
    split_count = 0
    output: list[OcrSyntheticTextLine] = []
    gutter_half = max(12.0, page_width * 0.02)
    for row in group:
        split_lines = split_row_into_column_synthetic_lines(
            row,
            split_x=split_x,
            gutter_half=gutter_half,
        )
        if len(split_lines) >= 2:
            split_count += 1
        output.extend(split_lines)
    return deduplicate_band_twin_lines(output, split_x=split_x), split_count


def split_row_into_column_synthetic_lines(
    row: OcrLineRowWords,
    *,
    split_x: float,
    gutter_half: float,
) -> tuple[OcrSyntheticTextLine, ...]:
    left_words = tuple(
        word for word in row.words if word_center_x(word) <= split_x - gutter_half
    )
    right_words = tuple(
        word for word in row.words if word_center_x(word) >= split_x + gutter_half
    )
    if not row_words_support_column_text(
        left_words
    ) or not row_words_support_column_text(right_words):
        return row_to_synthetic_lines((row,))
    left_line = synthetic_text_line_from_words(left_words, side_local=True)
    right_line = synthetic_text_line_from_words(right_words, side_local=True)
    if left_line is None or right_line is None:
        return row_to_synthetic_lines((row,))
    left_usable = synthetic_split_line_is_usable(left_words, left_line)
    right_usable = synthetic_split_line_is_usable(right_words, right_line)
    if left_usable and right_usable:
        if split_line_pair_is_usable(
            row,
            left_words=left_words,
            left_line=left_line,
            right_words=right_words,
            right_line=right_line,
        ):
            return (left_line, right_line)
    elif left_usable or right_usable:
        if synthetic_split_line_has_minimal_value(
            left_words,
            left_line,
        ) and synthetic_split_line_has_minimal_value(right_words, right_line):
            return (left_line, right_line)
        if left_usable:
            fallback = fallback_synthetic_text_line_from_words(right_words)
            if fallback is not None:
                return (left_line, fallback)
        if right_usable:
            fallback = fallback_synthetic_text_line_from_words(left_words)
            if fallback is not None:
                return (fallback, right_line)
    return row_to_synthetic_lines((row,))


def deduplicate_band_twin_lines(
    lines: list[OcrSyntheticTextLine],
    *,
    split_x: float,
) -> list[OcrSyntheticTextLine]:
    deduped: list[OcrSyntheticTextLine] = []
    for line in lines:
        if deduped and band_twin_lines_match(deduped[-1], line, split_x=split_x):
            previous = deduped[-1]
            previous_confidence = previous.confidence or 0.0
            line_confidence = line.confidence or 0.0
            previous_width = max(0.0, previous.bbox[2] - previous.bbox[0])
            line_width = max(0.0, line.bbox[2] - line.bbox[0])
            if (line_confidence, line_width) > (previous_confidence, previous_width):
                deduped[-1] = line
            continue
        deduped.append(line)
    return deduped


def band_twin_lines_match(
    left: OcrSyntheticTextLine,
    right: OcrSyntheticTextLine,
    *,
    split_x: float,
) -> bool:
    if left.text != right.text:
        return False
    left_tokens = ocr_text_analysis.normalized_text_tokens(left.text)
    right_tokens = ocr_text_analysis.normalized_text_tokens(right.text)
    if len(left_tokens) < 4 or left_tokens != right_tokens:
        return False
    left_center_y = (left.bbox[1] + left.bbox[3]) * 0.5
    right_center_y = (right.bbox[1] + right.bbox[3]) * 0.5
    left_height = max(1.0, left.bbox[3] - left.bbox[1])
    right_height = max(1.0, right.bbox[3] - right.bbox[1])
    if abs(left_center_y - right_center_y) > max(left_height, right_height) * 1.1:
        return False
    left_center_x = (left.bbox[0] + left.bbox[2]) * 0.5
    right_center_x = (right.bbox[0] + right.bbox[2]) * 0.5
    return (left_center_x < split_x < right_center_x) or (
        right_center_x < split_x < left_center_x
    )


def row_to_synthetic_lines(
    rows: tuple[OcrLineRowWords, ...],
) -> tuple[OcrSyntheticTextLine, ...]:
    output: list[OcrSyntheticTextLine] = []
    for row in rows:
        line = synthetic_text_line_from_words(row.words)
        if line is not None:
            output.append(line)
    return tuple(output)


def synthetic_text_line_from_words(
    words: tuple[OcrLineWordRow, ...],
    *,
    side_local: bool = False,
) -> OcrSyntheticTextLine | None:
    if not words:
        return None
    row = OcrLineRowWords(
        bbox=words_union_bbox(words),
        confidence=median_word_confidence(words),
        words=words,
    )
    text = reconstruct_side_local_text_from_words(words) if side_local else None
    if text is None:
        text = reconstruct_row_text_from_gap_words(row)
    if text is None:
        keep_start = leading_row_content_start(
            row,
            line_width=max(0.0, row.bbox[2] - row.bbox[0]),
        )
        keep_end = trailing_row_content_end(row)
        text = reconstruct_row_text_from_word_span(
            row,
            keep_start=keep_start,
            keep_end=keep_end,
        )
    text = (text or " ".join(word.text for word in words)).strip()
    if not text:
        return None
    text = normalize_prize_ladder_line_text(text)
    return OcrSyntheticTextLine(text=text, bbox=row.bbox, confidence=row.confidence)


def reconstruct_side_local_text_from_words(
    words: tuple[OcrLineWordRow, ...],
) -> str | None:
    if not words:
        return None
    row = OcrLineRowWords(
        bbox=words_union_bbox(words),
        confidence=median_word_confidence(words),
        words=words,
    )
    keep_start = leading_row_content_start(
        row,
        line_width=max(0.0, row.bbox[2] - row.bbox[0]),
    )
    keep_end = trailing_row_content_end(row)
    text = reconstruct_row_text_from_word_span(
        row,
        keep_start=keep_start,
        keep_end=keep_end,
    )
    if text:
        tokens = ocr_text_analysis.normalized_text_tokens(text)
        noisy_tokens = sum(1 for token in tokens if edge_token_is_noise(token))
        numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
        if numeric_tokens >= 2:
            return text
        if sum(
            1 for word in words if row_word_is_contentful(word)
        ) >= 2 and noisy_tokens <= max(1, len(tokens) // 4):
            return text
    clusters = split_words_into_local_clusters(words)
    if not clusters:
        return best_side_local_suffix_text(words)
    combined_text = combined_side_local_cluster_text(clusters)
    if combined_text is not None:
        return combined_text
    suffix_text = best_side_local_suffix_text(words)
    if suffix_text is not None:
        return suffix_text
    best_text: str | None = None
    best_score: tuple[int, int, float, int] | None = None
    for cluster in clusters:
        text = reconstruct_local_cluster_text(cluster)
        if not text:
            continue
        contentful = sum(1 for word in cluster if row_word_is_contentful(word))
        numeric = sum(1 for word in cluster if word_text_is_numeric_like(word.text))
        confidence = median_word_confidence(cluster) or 0.0
        score = (contentful, numeric, int(round(confidence)), len(text))
        if best_score is None or score > best_score:
            best_text = text
            best_score = score
    return best_text


def reconstruct_local_cluster_text(
    words: tuple[OcrLineWordRow, ...],
) -> str:
    cluster_row = OcrLineRowWords(
        bbox=words_union_bbox(words),
        confidence=median_word_confidence(words),
        words=words,
    )
    text = reconstruct_row_text_from_gap_words(cluster_row)
    if text is None:
        keep_end = trailing_row_content_end(cluster_row)
        keep_start = leading_row_content_start(
            cluster_row,
            line_width=max(0.0, cluster_row.bbox[2] - cluster_row.bbox[0]),
        )
        text = reconstruct_row_text_from_word_span(
            cluster_row,
            keep_start=keep_start,
            keep_end=keep_end,
        )
    return (text or " ".join(word.text for word in words)).strip()


def combined_side_local_cluster_text(
    clusters: tuple[tuple[OcrLineWordRow, ...], ...],
) -> str | None:
    if len(clusters) < 2 or len(clusters) > 3:
        return None
    cluster_texts: list[str] = []
    rank_amount_clusters = 0
    age_like_clusters = 0
    for cluster in clusters:
        text = reconstruct_local_cluster_text(cluster)
        if not text:
            return None
        cluster_texts.append(text)
        tokens = ocr_text_analysis.normalized_text_tokens(text)
        if (
            sum(1 for token in tokens if prize_rank_token(token) is not None) >= 1
            and sum(
                1
                for token in tokens
                if prize_amount_token(token) is not None or token.startswith("$")
            )
            >= 1
        ):
            rank_amount_clusters += 1
        if any(token.isdigit() for token in tokens) and any(
            row_word_is_contentful(word) for word in cluster
        ):
            age_like_clusters += 1
    if rank_amount_clusters >= 2 or age_like_clusters >= 2:
        return " ".join(cluster_texts)
    return None


def best_side_local_suffix_text(
    words: tuple[OcrLineWordRow, ...],
) -> str | None:
    if not words:
        return None
    row = OcrLineRowWords(
        bbox=words_union_bbox(words),
        confidence=median_word_confidence(words),
        words=words,
    )
    keep_end = trailing_row_content_end(row)
    best_text: str | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    for index, word in enumerate(words):
        if not row_word_is_contentful(word) and not word_text_is_numeric_like(
            word.text
        ):
            continue
        text = reconstruct_row_text_from_word_span(
            row,
            keep_start=index,
            keep_end=keep_end,
        )
        text = (text or "").strip()
        if not text:
            continue
        cluster_words = words[index:keep_end]
        tokens = ocr_text_analysis.normalized_text_tokens(text)
        noisy_tokens = sum(1 for token in tokens if edge_token_is_noise(token))
        contentful = sum(
            1 for candidate in cluster_words if row_word_is_contentful(candidate)
        )
        numeric = sum(
            1
            for candidate in cluster_words
            if word_text_is_numeric_like(candidate.text)
        )
        confidence = int(round(median_word_confidence(cluster_words) or 0.0))
        score = (contentful, numeric, -noisy_tokens, confidence, len(text))
        if best_score is None or score > best_score:
            best_text = text
            best_score = score
    return best_text


def normalize_prize_ladder_line_text(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 2:
        return text
    rank_count = sum(1 for token in tokens if prize_rank_token(token) is not None)
    amount_like_count = sum(
        1
        for token in tokens
        if prize_amount_token(token) is not None or token.startswith("$")
    )
    if rank_count == 0 or amount_like_count == 0:
        return text
    normalized: list[str] = []
    changed = False
    for token in tokens:
        rank = prize_rank_token(token)
        if rank is not None:
            normalized.append(rank)
            changed = changed or rank != token
            continue
        amount = prize_amount_token(token)
        if amount is not None:
            normalized.append(amount)
            changed = changed or amount != token
            continue
        normalized.append(token)
    return " ".join(normalized) if changed else text


def prize_rank_token(token: str) -> str | None:
    match = PRIZE_RANK_TOKEN_RE.match(token)
    if match is None:
        return None
    return f"{match.group('rank')}."


def prize_amount_token(token: str) -> str | None:
    prefix = "$" if token.startswith("$") else ""
    core = token[1:] if prefix else token
    if core.count(".") == 1:
        return None
    match = PRIZE_AMOUNT_TOKEN_RE.match(core)
    if match is None:
        return None
    return f"{prefix}{match.group('whole')}.{match.group('cents')}"


def split_words_into_local_clusters(
    words: tuple[OcrLineWordRow, ...],
) -> tuple[tuple[OcrLineWordRow, ...], ...]:
    if not words:
        return ()
    bbox = words_union_bbox(words)
    width = max(0.0, bbox[2] - bbox[0])
    gap_threshold = max(18.0, width * 0.16)
    clusters: list[list[OcrLineWordRow]] = [[words[0]]]
    for word in words[1:]:
        previous = clusters[-1][-1]
        gap = word.bbox[0] - previous.bbox[2]
        if gap >= gap_threshold:
            clusters.append([word])
            continue
        clusters[-1].append(word)
    return tuple(
        tuple(cluster)
        for cluster in clusters
        if cluster_supports_local_text(tuple(cluster))
    )


def cluster_supports_local_text(
    words: tuple[OcrLineWordRow, ...],
) -> bool:
    if not words:
        return False
    contentful = [word for word in words if row_word_is_contentful(word)]
    if len(contentful) >= 2:
        return True
    if len(contentful) == 1:
        token = alpha_token_letters(contentful[0].text)
        if len(token) >= 4:
            return True
    if (
        len(contentful) == 1
        and len(
            numeric_like := [
                word for word in words if word_text_is_numeric_like(word.text)
            ]
        )
        >= 1
    ):
        return True
    numeric_like = [word for word in words if word_text_is_numeric_like(word.text)]
    return len(numeric_like) >= 2


def first_contentful_word_index(words: tuple[OcrLineWordRow, ...]) -> int:
    for index, word in enumerate(words):
        if row_word_is_contentful(word) or word_text_is_numeric_like(word.text):
            return index
    return 0


def split_line_pair_is_usable(
    row: OcrLineRowWords,
    *,
    left_words: tuple[OcrLineWordRow, ...],
    left_line: OcrSyntheticTextLine,
    right_words: tuple[OcrLineWordRow, ...],
    right_line: OcrSyntheticTextLine,
) -> bool:
    if not synthetic_split_line_is_usable(left_words, left_line):
        return False
    if not synthetic_split_line_is_usable(right_words, right_line):
        return False
    original_text = reconstruct_row_text_from_gap_words(row) or " ".join(
        word.text for word in row.words
    )
    original_tokens = ocr_text_analysis.normalized_text_tokens(original_text)
    split_tokens = ocr_text_analysis.normalized_text_tokens(
        left_line.text
    ) + ocr_text_analysis.normalized_text_tokens(right_line.text)
    if split_tokens and len(split_tokens) > max(6, len(original_tokens) + 2):
        return False
    return True


def synthetic_split_line_is_usable(
    words: tuple[OcrLineWordRow, ...],
    line: OcrSyntheticTextLine,
) -> bool:
    width = max(0.0, line.bbox[2] - line.bbox[0])
    if width < 42.0:
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(line.text)
    if not tokens:
        return False
    contentful_words = tuple(word for word in words if row_word_is_contentful(word))
    numeric_words = tuple(
        word for word in words if word_text_is_numeric_like(word.text)
    )
    if len(contentful_words) < 2 and len(numeric_words) < 2:
        return False
    prefix_noise = 0
    for word in words:
        if row_word_is_contentful(word):
            break
        prefix_noise += 1
    if prefix_noise >= max(2, len(words) - 1):
        return False
    numeric_confidence = (
        median_word_confidence(numeric_words) if numeric_words else None
    )
    if len(numeric_words) >= 2 and (numeric_confidence or 0.0) >= 55.0:
        return True
    if len(numeric_words) < 3 and split_words_have_multiple_clusters(words):
        return False
    if len(tokens) <= 2:
        return True
    noisy_tokens = sum(1 for token in tokens if edge_token_is_noise(token))
    return noisy_tokens <= max(1, len(tokens) // 4)


def synthetic_split_line_has_minimal_value(
    words: tuple[OcrLineWordRow, ...],
    line: OcrSyntheticTextLine,
) -> bool:
    width = max(0.0, line.bbox[2] - line.bbox[0])
    if width < 30.0:
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(line.text)
    if not tokens:
        return False
    contentful_words = tuple(word for word in words if row_word_is_contentful(word))
    numeric_words = tuple(
        word for word in words if word_text_is_numeric_like(word.text)
    )
    if len(contentful_words) < 1 and len(numeric_words) < 2:
        return False
    numeric_confidence = (
        median_word_confidence(numeric_words) if numeric_words else None
    )
    if len(numeric_words) >= 2 and (numeric_confidence or 0.0) >= 50.0:
        return True
    noisy_tokens = sum(1 for token in tokens if edge_token_is_noise(token))
    return noisy_tokens <= max(1, len(tokens) // 2)


def fallback_synthetic_text_line_from_words(
    words: tuple[OcrLineWordRow, ...],
) -> OcrSyntheticTextLine | None:
    line = synthetic_text_line_from_words(words, side_local=True)
    if line is not None and synthetic_split_line_has_minimal_value(words, line):
        return line
    line = synthetic_text_line_from_words(words)
    if line is not None and synthetic_split_line_has_minimal_value(words, line):
        return line
    return None


def split_words_have_multiple_clusters(
    words: tuple[OcrLineWordRow, ...],
) -> bool:
    if len(words) < 4:
        return False
    bbox = words_union_bbox(words)
    width = max(0.0, bbox[2] - bbox[0])
    gap_threshold = max(26.0, width * 0.22)
    cluster_breaks = 0
    for index in range(len(words) - 1):
        left = words[index]
        right = words[index + 1]
        if not row_word_is_contentful(left) or not row_word_is_contentful(right):
            continue
        left_contentful = sum(
            1 for word in words[: index + 1] if row_word_is_contentful(word)
        )
        right_contentful = sum(
            1 for word in words[index + 1 :] if row_word_is_contentful(word)
        )
        if left_contentful < 2 or right_contentful < 2:
            continue
        gap = right.bbox[0] - left.bbox[2]
        if gap >= gap_threshold:
            cluster_breaks += 1
    return cluster_breaks > 0


def rows_union_bbox(
    rows: tuple[OcrLineRowWords, ...],
) -> tuple[float, float, float, float] | None:
    if not rows:
        return None
    return (
        min(row.bbox[0] for row in rows),
        min(row.bbox[1] for row in rows),
        max(row.bbox[2] for row in rows),
        max(row.bbox[3] for row in rows),
    )


def words_union_bbox(
    words: tuple[OcrLineWordRow, ...],
) -> tuple[float, float, float, float]:
    return (
        min(word.bbox[0] for word in words),
        min(word.bbox[1] for word in words),
        max(word.bbox[2] for word in words),
        max(word.bbox[3] for word in words),
    )


def median_word_confidence(words: tuple[OcrLineWordRow, ...]) -> float | None:
    values = [word.confidence for word in words if word.confidence is not None]
    if not values:
        return None
    return float(median(values))


def word_center_x(word: OcrLineWordRow) -> float:
    return (word.bbox[0] + word.bbox[2]) * 0.5


def ocr_page_result_supplemental_resolved_lines(
    page: PageExtractionHost,
    text: str,
    ocr_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    candidate = ocr_result.candidate
    if candidate is None:
        return ()
    if candidate.name.endswith("_word_layout"):
        return ()
    if dense_numeric_ocr_supplement_would_reduce_recall(text, ocr_result.text):
        return ()
    ocr_lines = ocr_geometry.ocr_candidate_geometry_lines(page, ocr_result.candidate)
    if not ocr_lines:
        return ()
    native_lines = ocr_page_analysis.native_text_geometry_lines(page)
    if not native_lines:
        return ()
    return supplemental_resolved_text_lines_for_uncovered_geometry(
        text,
        native_lines,
        ocr_lines,
    )


def append_line_art_ocr_candidate_supplement(
    text: str,
    candidates: tuple[Any, ...],
) -> str:
    additions = line_art_ocr_candidate_supplement_tokens(text, candidates)
    if not additions:
        return text
    return text.rstrip() + "\n" + " ".join(additions)


def line_art_ocr_candidate_supplement_tokens(
    text: str,
    candidates: tuple[Any, ...],
    *,
    max_tokens: int = 32,
) -> list[str]:
    seen = Counter(ocr_text_analysis.normalized_text_tokens(text))
    original_seen = Counter(seen)
    observed: Counter[str] = Counter()
    observed_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    additions: list[str] = []
    rows = sorted(
        (
            (candidate, row)
            for candidate in candidates
            if str(getattr(candidate, "name", "")).startswith("line_art_text_mask_")
            for row in getattr(getattr(candidate, "result", None), "word_rows", ())
            if line_art_ocr_word_row_is_body(row, candidate)
        ),
        key=lambda item: line_art_ocr_word_row_order_key(item[1]),
    )
    for _candidate, row in rows:
        token = str(row.get("text", "")).strip()
        key = line_art_ocr_supplement_token_key(token)
        if key is None:
            continue
        core = "".join(ch for ch in token if ch.isalnum())
        if core.isalpha() and line_art_alpha_key_is_compound_of_seen(
            key, original_seen
        ):
            continue
        if line_art_ocr_supplement_row_was_observed(key, row, observed_boxes):
            continue
        observed[key] += 1
        if observed[key] <= seen[key]:
            continue
        confidence = line_art_ocr_word_row_confidence(row)
        if not line_art_ocr_supplement_token_is_useful(
            token,
            confidence,
            repeated=original_seen[key] > 0,
        ):
            continue
        additions.append(line_art_ocr_supplement_display_token(token, key))
        seen[key] += 1
        if len(additions) >= max_tokens:
            break
    return additions


def line_art_alpha_key_is_compound_of_seen(key: str, seen: Counter[str]) -> bool:
    if len(key) < 6:
        return False
    for split_at in range(2, len(key) - 1):
        if seen[key[:split_at]] > 0 and seen[key[split_at:]] > 0:
            return True
    return False


def line_art_ocr_word_row_is_body(row: dict[str, Any], candidate: Any) -> bool:
    image_width = getattr(candidate, "image_width", None)
    image_height = getattr(candidate, "image_height", None)
    if not isinstance(image_width, int) or not isinstance(image_height, int):
        return True
    if image_width <= 0 or image_height <= 0:
        return True
    try:
        left = float(row.get("left", 0))
        top = float(row.get("top", 0))
        width = float(row.get("width", 0))
        height = float(row.get("height", 0))
    except TypeError, ValueError:
        return True
    if top + height < image_height * 0.11:
        return False
    if top > image_height * 0.92:
        return False
    if left + width < image_width * 0.035:
        return False
    if left > image_width * 0.935:
        return False
    return True


def line_art_ocr_word_row_order_key(row: dict[str, Any]) -> tuple[float, float]:
    page_bbox = row.get("page_bbox")
    if isinstance(page_bbox, (list, tuple)) and len(page_bbox) == 4:
        try:
            return (-float(page_bbox[3]), float(page_bbox[0]))
        except TypeError, ValueError:
            pass
    try:
        return (float(row.get("top", 0)), float(row.get("left", 0)))
    except TypeError, ValueError:
        return (0.0, 0.0)


def line_art_ocr_word_row_confidence(row: dict[str, Any]) -> int:
    try:
        return int(round(float(row.get("conf", 0))))
    except TypeError, ValueError:
        return 0


def line_art_ocr_supplement_row_was_observed(
    key: str,
    row: dict[str, Any],
    observed_boxes: dict[str, list[tuple[float, float, float, float]]],
) -> bool:
    bbox = page_geometry.rect_box_tuple(row.get("page_bbox"))
    if bbox is None:
        return False
    boxes = observed_boxes.setdefault(key, [])
    if any(line_art_ocr_supplement_boxes_match(bbox, existing) for existing in boxes):
        return True
    boxes.append(bbox)
    return False


def line_art_ocr_supplement_boxes_match(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_area = page_geometry.rect_area(left)
    right_area = page_geometry.rect_area(right)
    if min(left_area, right_area) <= 0.0:
        return False
    overlap = page_geometry.rect_intersection_area(left, right)
    if overlap / min(left_area, right_area) >= 0.62:
        return True
    left_cx = (left[0] + left[2]) * 0.5
    left_cy = (left[1] + left[3]) * 0.5
    right_cx = (right[0] + right[2]) * 0.5
    right_cy = (right[1] + right[3]) * 0.5
    max_width = max(left[2] - left[0], right[2] - right[0], 1.0)
    max_height = max(left[3] - left[1], right[3] - right[1], 1.0)
    return abs(left_cx - right_cx) <= max(2.0, max_width * 0.35) and abs(
        left_cy - right_cy
    ) <= max(2.0, max_height * 0.35)


def line_art_ocr_supplement_token_key(token: str) -> str | None:
    core = "".join(ch for ch in token if ch.isalnum())
    if not core:
        return None
    return core.casefold()


def line_art_ocr_supplement_display_token(token: str, key: str) -> str:
    core = "".join(ch for ch in token if ch.isalnum())
    if core and core.isalpha() and core.upper() == core:
        return core
    if core and core.isdigit():
        return core
    return token.strip()


def line_art_ocr_supplement_token_is_useful(
    token: str,
    confidence: int,
    *,
    repeated: bool = False,
) -> bool:
    core = "".join(ch for ch in token if ch.isalnum())
    if not core:
        return False
    if core.isdigit():
        if len(core) == 1:
            return False
        min_confidence = 80 if repeated else 85
        return (
            1 <= len(core) <= 3
            and token.strip() == core
            and confidence >= min_confidence
        )
    if not core.isalpha():
        return False
    if not (2 <= len(core) <= 8):
        return False
    if core.upper() != core:
        return False
    min_confidence = 95 if repeated else 90
    return confidence >= min_confidence


def figure_ocr_page_result_supplemental_resolved_lines(
    page: PageExtractionHost,
    text: str,
    ocr_result: OcrPageTextResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not figure_ocr_page_result_looks_useful(ocr_result):
        return ()
    ocr_lines = ocr_geometry.ocr_candidate_textline_geometry_lines(
        page,
        ocr_result.candidate,
    )
    if not ocr_lines:
        return ()
    native_lines = ocr_page_analysis.native_text_geometry_lines(page)
    return figure_supplemental_resolved_text_lines(text, native_lines, ocr_lines)


def figure_ocr_page_result_looks_useful(ocr_result: OcrPageTextResult) -> bool:
    candidate = ocr_result.candidate
    if candidate is None or not ocr_result.text.strip():
        return False
    confidence = (
        candidate.result.confidence if candidate.result.confidence is not None else 0
    )
    if confidence < 58:
        return False
    tokens = ocr_text_analysis.extracted_text_token_count(ocr_result.text)
    if tokens < 4:
        return False
    gibberish = ocr_text_analysis.alphabetic_gibberish_score(ocr_result.text)
    if tokens >= 80 and gibberish >= 0.28:
        return False
    if tokens >= 200 and gibberish >= 0.18:
        return False
    return len([line for line in ocr_result.text.splitlines() if line.strip()]) >= 2


def figure_supplemental_resolved_text_lines(
    text: str,
    native_lines: list[Any],
    ocr_lines: list[Any],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    seen_tokens = set(ocr_text_analysis.normalized_text_tokens(text))
    accepted_observations = text_line_observations(
        native_lines,
        source="native_text",
        kind="native_line",
    )
    supplemental: list[observation_resolver.ResolvedTextLine] = []
    for line_index, line in enumerate(ocr_lines):
        ocr_observation = page_geometry.page_observation_from_text_line(
            line,
            source=str(getattr(line, "source", "figure_ocr")),
            kind="figure_text_line",
            line_index=line_index,
        )
        if ocr_observation is None:
            continue
        resolution = observation_resolver.resolve_observation_append(
            ocr_observation,
            accepted_observations,
            existing_text=text,
        )
        if resolution.action != "append":
            continue
        stripped = str(getattr(line, "text", "")).strip()
        if not figure_ocr_line_should_append(stripped, seen_tokens):
            continue
        supplemental.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                ocr_observation,
                break_before=1,
                contributing_observations=(ocr_observation,),
                resolution=resolution,
            )
        )
        accepted_observations.append(ocr_observation)
        seen_tokens.update(ocr_text_analysis.normalized_text_tokens(stripped))
    return tuple(supplemental)


def figure_ocr_line_should_append(
    stripped: str,
    seen_tokens: set[str],
) -> bool:
    if not stripped or not any(ch.isalnum() for ch in stripped):
        return False
    if not compact_alnum_text(stripped):
        return False
    if figure_ocr_line_noise_ratio(stripped) > 0.50:
        return False
    if ocr_text_analysis.alphabetic_gibberish_line_score(stripped) >= 0.55:
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(stripped)
    if not tokens:
        return False
    new_tokens = [token for token in tokens if token not in seen_tokens]
    useful_new_tokens = [
        token for token in new_tokens if ocr_supplement_new_token_looks_useful(token)
    ]
    has_diagram_signal = figure_ocr_line_has_diagram_signal(stripped, tokens)
    if len(tokens) <= 2:
        return bool(useful_new_tokens)
    if new_tokens and not useful_new_tokens:
        return False
    if has_diagram_signal and useful_new_tokens:
        return True
    if has_diagram_signal and figure_ocr_line_is_compact_diagram_label(
        stripped,
        tokens,
    ):
        return True
    return len(useful_new_tokens) >= max(1, int(len(tokens) * 0.25))


def ocr_supplement_new_token_looks_useful(token: str) -> bool:
    if not token:
        return False
    if token.isalpha():
        return len(token) >= 3
    if token.isdigit():
        return True
    if len(token) >= 4 and any(ch.isalpha() for ch in token):
        return True
    return any(ch in token for ch in "_/+-")


def figure_ocr_line_has_diagram_signal(text: str, tokens: list[str]) -> bool:
    if "=" in text or ":" in text:
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if any(ch.isupper() for ch in text):
        return True
    return any(len(token) >= 4 for token in tokens)


def figure_ocr_line_is_compact_diagram_label(text: str, tokens: list[str]) -> bool:
    if len(tokens) > 5 or len(text) > 48:
        return False
    if any(ch.isdigit() for ch in text):
        return True
    return any(ch in text for ch in "+_=/")


def figure_ocr_line_noise_ratio(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 1.0
    allowed_punctuation = frozenset("+-._/=():,[]{}<>")
    noisy = 0
    for ch in nonspace:
        if ch.isalnum() or ch in allowed_punctuation:
            continue
        noisy += 1
    return noisy / len(nonspace)


def insert_figure_supplemental_lines_near_caption(
    page: PageExtractionHost,
    text: str,
    supplemental_lines: list[str],
) -> str:
    text_lines = text.splitlines()
    insert_index = figure_caption_insert_index(page, text_lines)
    if insert_index is None:
        return text.rstrip() + "\n" + "\n".join(supplemental_lines)
    output = list(text_lines)
    output[insert_index:insert_index] = supplemental_lines
    return "\n".join(output).rstrip()


def figure_caption_insert_index(
    page: PageExtractionHost,
    text_lines: list[str],
) -> int | None:
    captions = [
        region.caption_text
        for region in ocr_page_analysis.figure_ocr_regions(page)
        if region.caption_text
    ]
    for caption in captions:
        caption_tokens = ocr_text_analysis.normalized_text_tokens(caption)
        if not caption_tokens:
            continue
        prefix_size = min(4, len(caption_tokens))
        caption_prefix = caption_tokens[:prefix_size]
        for index, line in enumerate(text_lines):
            line_tokens = ocr_text_analysis.normalized_text_tokens(line)
            if line_tokens[:prefix_size] == caption_prefix:
                return index
    return None


def reconcile_native_ocr_text_by_geometry(
    page: PageExtractionHost,
    text: str,
    ocr_result: OcrPageTextResult,
) -> str:
    return reconcile_native_ocr_lines_by_geometry(page, text, ocr_result)[0]


def reconcile_native_ocr_lines_by_geometry(
    page: PageExtractionHost,
    text: str,
    ocr_result: OcrPageTextResult,
) -> tuple[str, tuple[observation_resolver.ResolvedTextLine, ...]]:
    if not text.strip() or ocr_result.candidate is None:
        return text, ()
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    try:
        if ocr_page_analysis.has_dominant_page_image(
            page
        ) and not ocr_page_analysis.native_text_layer_has_substantial_page_coverage(
            page,
            text_tokens,
        ):
            return text, ()
    except Exception:
        pass
    native_lines = ocr_page_analysis.native_text_geometry_lines(
        page, include_hidden=True
    )
    if not native_lines:
        return text, ()
    ocr_lines = ocr_geometry.ocr_candidate_textline_geometry_lines(
        page,
        ocr_result.candidate,
    )
    if not ocr_lines:
        return text, ()
    return reconcile_text_geometry_lines(text, native_lines, ocr_lines)


def reconcile_text_geometry_lines(
    text: str,
    native_lines: list[Any],
    ocr_lines: list[Any],
) -> tuple[str, tuple[observation_resolver.ResolvedTextLine, ...]]:
    text_lines = text.splitlines()
    if not text_lines or not native_lines or not ocr_lines:
        return text, ()
    records = text_geometry_records_for_lines(text_lines, native_lines)
    output_lines = list(text_lines)
    used_ocr_indexes: set[int] = set()
    replacements = 0
    for index, record in enumerate(records):
        if record.observation is None or not record.text.strip():
            continue
        ocr_index, geometry_score = best_overlapping_text_geometry_line(
            record,
            ocr_lines,
            used_ocr_indexes,
        )
        if ocr_index is None:
            continue
        ocr_line = ocr_lines[ocr_index]
        if not should_replace_native_line_with_ocr_line(
            record.text,
            str(ocr_line.text),
            getattr(ocr_line, "confidence", None),
            geometry_score=geometry_score,
        ):
            continue
        output_lines[index] = str(ocr_line.text).strip()
        records[index] = TextGeometryRecord(
            output_lines[index],
            page_geometry.PageObservation(
                kind=record.observation.kind,
                source=record.observation.source,
                bbox=record.observation.bbox,
                advance_bbox=record.observation.advance_bbox,
                ink_bbox=record.observation.ink_bbox,
                confidence=getattr(ocr_line, "confidence", None),
                text=output_lines[index],
                baseline=record.observation.baseline,
                provenance=record.observation.provenance,
            ),
        )
        used_ocr_indexes.add(ocr_index)
        replacements += 1
    if replacements == 0:
        return text, ()
    resolved_lines = resolved_text_lines_from_geometry_records(records)
    return "\n".join(output_lines).rstrip(), resolved_lines


def resolved_text_lines_from_geometry_records(
    records: list[TextGeometryRecord],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    output_lines: list[observation_resolver.ResolvedTextLine] = []
    for record in records:
        text = record.text.strip()
        if not text or record.observation is None:
            continue
        observation = record.observation
        if observation.text != text:
            observation = page_geometry.PageObservation(
                kind=observation.kind,
                source=observation.source,
                bbox=observation.bbox,
                advance_bbox=observation.advance_bbox,
                ink_bbox=observation.ink_bbox,
                confidence=observation.confidence,
                text=text,
                baseline=observation.baseline,
                provenance=observation.provenance,
            )
        output_lines.append(
            observation_resolver.ResolvedTextLine(
                text,
                observation,
                contributing_observations=(observation,),
            )
        )
    return tuple(output_lines)


def should_use_reconciled_native_ocr_text(
    current: str,
    ocr_text: str,
    reconciled: str,
) -> bool:
    if reconciled == current:
        return False
    current_tokens = ocr_text_analysis.extracted_text_token_count(current)
    reconciled_tokens = ocr_text_analysis.extracted_text_token_count(reconciled)
    ocr_tokens = ocr_text_analysis.extracted_text_token_count(ocr_text)
    if current_tokens == 0:
        return reconciled_tokens > 0
    if dense_numeric_ocr_supplement_would_reduce_recall(current, ocr_text):
        return False
    if reconciled_tokens < current_tokens:
        return False
    if reconciled_tokens > max(current_tokens + ocr_tokens, int(current_tokens * 2.2)):
        return False
    current_quality = ocr_text_analysis.text_ocr_quality_score(current)
    reconciled_quality = ocr_text_analysis.text_ocr_quality_score(reconciled)
    if reconciled_quality > max(0.48, current_quality + 0.22):
        return False
    return True


def should_reject_full_page_ocr_result(
    current: str,
    ocr_text: str,
) -> bool:
    current_tokens = ocr_text_analysis.extracted_text_token_count(current)
    ocr_tokens = ocr_text_analysis.extracted_text_token_count(ocr_text)
    if current_tokens < 40 or ocr_tokens < 40:
        return False
    current_garbled_ratio = garbled_alpha_token_ratio(current)
    ocr_garbled_ratio = garbled_alpha_token_ratio(ocr_text)
    if ocr_garbled_ratio < 0.07:
        return False
    current_quality = ocr_text_analysis.text_ocr_quality_score(current)
    ocr_quality = ocr_text_analysis.text_ocr_quality_score(ocr_text)
    if ocr_quality + 0.02 < current_quality:
        return False
    return ocr_garbled_ratio > current_garbled_ratio + 0.04


def garbled_alpha_token_ratio(text: str) -> float:
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    if not alpha_tokens:
        return 0.0
    garbled = sum(
        1
        for token in alpha_tokens
        if ocr_text_analysis.alpha_token_looks_ocr_garbled(token)
    )
    return garbled / len(alpha_tokens)


def text_geometry_records_for_lines(
    text_lines: list[str],
    geometry_lines: list[Any],
) -> list[TextGeometryRecord]:
    records = [TextGeometryRecord(line) for line in text_lines]
    nonempty_indexes = [index for index, line in enumerate(text_lines) if line.strip()]
    if len(nonempty_indexes) == len(geometry_lines):
        for text_index, geometry_line in zip(
            nonempty_indexes,
            geometry_lines,
            strict=False,
        ):
            records[text_index] = TextGeometryRecord(
                text_lines[text_index],
                page_geometry.page_observation_from_text_line(
                    geometry_line,
                    source="native_text",
                    kind="native_line",
                ),
            )
        return records

    used_geometry: set[int] = set()
    for text_index in nonempty_indexes:
        tokens = ocr_text_analysis.normalized_text_tokens(text_lines[text_index])
        if not tokens:
            continue
        for geometry_index, geometry_line in enumerate(geometry_lines):
            if geometry_index in used_geometry:
                continue
            if (
                ocr_text_analysis.normalized_text_tokens(str(geometry_line.text))
                != tokens
            ):
                continue
            used_geometry.add(geometry_index)
            records[text_index] = TextGeometryRecord(
                text_lines[text_index],
                page_geometry.page_observation_from_text_line(
                    geometry_line,
                    source="native_text",
                    kind="native_line",
                ),
            )
            break
    return records


def best_overlapping_text_geometry_line(
    native_line: TextGeometryRecord,
    ocr_lines: list[Any],
    used_ocr_indexes: set[int],
) -> tuple[int | None, float]:
    if native_line.observation is None:
        return None, 0.0
    best_index: int | None = None
    best_score = 0.0
    for index, ocr_line in enumerate(ocr_lines):
        if index in used_ocr_indexes:
            continue
        ocr_observation = page_geometry.page_observation_from_text_line(
            ocr_line,
            source="ocr",
            kind="ocr_textline",
        )
        if ocr_observation is None:
            continue
        score = page_geometry.observation_geometry_match_score(
            native_line.observation,
            ocr_observation,
        )
        if score > best_score:
            best_index = index
            best_score = score
    if best_score < 0.58:
        return None, best_score
    return best_index, best_score


def should_replace_native_line_with_ocr_line(
    native_text: str,
    ocr_text: str,
    confidence: int | None,
    *,
    geometry_score: float,
) -> bool:
    native = native_text.strip()
    ocr = ocr_text.strip()
    if not native or not ocr or native == ocr:
        return False
    if geometry_score < 0.58:
        return False
    confidence_value = confidence if confidence is not None else 55
    if confidence_value < 45:
        return False
    native_tokens = ocr_text_analysis.normalized_text_tokens(native)
    ocr_tokens = ocr_text_analysis.normalized_text_tokens(ocr)
    if len(ocr_tokens) < max(2, int(len(native_tokens) * 0.60)):
        return False
    if len(ocr_tokens) > max(len(native_tokens) + 18, int(len(native_tokens) * 2.8)):
        return False
    native_compact = compact_alnum_text(native)
    ocr_compact = compact_alnum_text(ocr)
    if len(native_compact) < 8 or len(ocr_compact) < 8:
        return False
    compact_similarity = SequenceMatcher(None, native_compact, ocr_compact).ratio()
    native_quality = ocr_text_analysis.text_ocr_quality_score(native)
    ocr_quality = ocr_text_analysis.text_ocr_quality_score(ocr)
    if compact_similarity >= 0.72 and len(ocr_tokens) > len(native_tokens):
        return ocr_quality <= max(0.42, native_quality + 0.18)
    if native_quality >= 0.22 and ocr_quality + 0.08 < native_quality:
        return True
    return False


def compact_alnum_text(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def vector_stroke_page_result_supplemental_resolved_lines(
    page: PageExtractionHost,
    text: str,
    vector_result: VectorStrokeOcrResult,
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if not text.strip() or not vector_result.text.strip() or not vector_result.lines:
        return ()
    if vector_stroke_text_looks_fragmented(vector_result.text):
        return ()
    if vector_stroke_text_looks_noisy(vector_result.text, vector_result.confidence):
        return ()
    native_lines = ocr_page_analysis.native_text_geometry_lines(page)
    if not native_lines:
        return ()
    return supplemental_resolved_text_lines_for_uncovered_geometry(
        text,
        native_lines,
        vector_result.lines,
        candidate_filter=vector_stroke_line_is_spatial_supplement,
        allow_short_lines=True,
        candidate_source="vector_stroke",
        candidate_kind="vector_text_line",
    )


def merge_uncovered_text_geometry_lines(
    text: str,
    base_lines: list[Any],
    candidate_lines: list[Any] | tuple[Any, ...],
    *,
    candidate_filter: Callable[[Any], bool] | None = None,
    covered_by_base: Callable[[Any, list[Any]], bool] | None = None,
    allow_short_lines: bool = False,
) -> str:
    supplemental_lines = supplemental_resolved_text_lines_for_uncovered_geometry(
        text,
        base_lines,
        candidate_lines,
        candidate_filter=candidate_filter,
        covered_by_base=covered_by_base,
        allow_short_lines=allow_short_lines,
    )
    if not supplemental_lines:
        return text
    return text.rstrip() + "\n" + "\n".join(line.text for line in supplemental_lines)


def supplemental_resolved_text_lines_for_uncovered_geometry(
    text: str,
    base_lines: list[Any],
    candidate_lines: list[Any] | tuple[Any, ...],
    *,
    candidate_filter: Callable[[Any], bool] | None = None,
    covered_by_base: Callable[[Any, list[Any]], bool] | None = None,
    allow_short_lines: bool = False,
    candidate_source: str = "ocr",
    candidate_kind: str = "ocr_textline",
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    coverage = covered_by_base or ocr_line_is_covered_by_native_text
    accepted_observations = text_line_observations(
        base_lines,
        source="native_text",
        kind="native_line",
    )
    seen_tokens = set(ocr_text_analysis.normalized_text_tokens(text))
    table_like = (
        ocr_text_analysis.text_has_many_digit_lines(text)
        and ocr_text_analysis.numeric_token_ratio(text) >= 0.18
    )
    supplemental: list[observation_resolver.ResolvedTextLine] = []
    for line in candidate_lines:
        if candidate_filter is not None and not candidate_filter(line):
            continue
        if coverage(line, base_lines):
            continue
        candidate_observation = page_geometry.page_observation_from_text_line(
            line,
            source=candidate_source,
            kind=candidate_kind,
        )
        if candidate_observation is None:
            continue
        resolution = observation_resolver.resolve_observation_append(
            candidate_observation,
            accepted_observations,
            existing_text=text,
        )
        if resolution.action != "append":
            continue
        stripped = str(getattr(line, "text", "")).strip()
        if not stripped:
            continue
        line_tokens = ocr_text_analysis.normalized_text_tokens(stripped)
        if not supplemental_text_line_should_append(
            stripped,
            line_tokens,
            seen_tokens,
            table_like=table_like,
            allow_short_lines=allow_short_lines,
        ):
            continue
        supplemental.append(
            observation_resolver.ResolvedTextLine(
                stripped,
                candidate_observation,
                break_before=1,
                contributing_observations=(candidate_observation,),
                resolution=resolution,
            )
        )
        accepted_observations.append(candidate_observation)
        seen_tokens.update(line_tokens)
    return tuple(supplemental)


def text_line_observations(
    lines: list[Any] | tuple[Any, ...],
    *,
    source: str,
    kind: str,
) -> list[page_geometry.PageObservation]:
    observations: list[page_geometry.PageObservation] = []
    for line in lines:
        observation = page_geometry.page_observation_from_text_line(
            line,
            source=source,
            kind=kind,
        )
        if observation is not None:
            observations.append(observation)
    return observations


def ocr_line_is_covered_by_native_text(
    ocr_line: Any,
    native_lines: list[Any],
) -> bool:
    ocr_observation = page_geometry.page_observation_from_text_line(
        ocr_line,
        source="ocr",
        kind="ocr_textline",
    )
    if ocr_observation is None:
        return True
    native_observations = tuple(
        observation
        for line in native_lines
        if (
            observation := page_geometry.page_observation_from_text_line(
                line,
                source="native_text",
                kind="native_line",
            )
        )
        is not None
    )
    return page_geometry.observation_is_covered_by(
        ocr_observation,
        native_observations,
        single_overlap_ratio=0.35,
        cumulative_overlap_ratio=0.45,
    )


def supplemental_text_line_should_append(
    stripped: str,
    line_tokens: list[str],
    seen_tokens: set[str],
    *,
    table_like: bool = False,
    allow_short_lines: bool = False,
) -> bool:
    if not stripped or not line_tokens:
        return False
    if len(line_tokens) < 3:
        if allow_short_lines:
            return any(token not in seen_tokens for token in line_tokens)
        return (
            table_like
            and ocr_text_analysis.supplemental_ocr_short_line_looks_tabular(
                line_tokens,
                seen_tokens,
            )
        )
    if ocr_text_analysis.supplemental_ocr_line_looks_fragmentary(
        line_tokens,
        stripped,
    ):
        return False
    new_tokens = sum(1 for token in line_tokens if token not in seen_tokens)
    return new_tokens >= max(2, int(len(line_tokens) * 0.35))


def should_replace_text_with_vector_stroke_ocr(
    text: str,
    vector_text: str,
    confidence: int | None = None,
) -> bool:
    if vector_stroke_text_looks_fragmented(vector_text):
        return False
    if vector_stroke_text_looks_noisy(vector_text, confidence):
        return False
    vector_tokens = ocr_text_analysis.extracted_text_token_count(vector_text)
    if vector_tokens < 8:
        return False
    text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens == 0:
        return True
    if text_tokens < 80 and vector_tokens >= text_tokens + 20:
        return True
    return vector_tokens >= max(20, int(text_tokens * 1.20))


def vector_stroke_text_looks_noisy(text: str, confidence: int | None = None) -> bool:
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if len(tokens) < 8:
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    lower_alpha = sum(1 for token in tokens if token.isalpha() and token.islower())
    digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    short_tokens = sum(1 for token in tokens if len(token) <= 2)
    lower_ratio = lower_alpha / len(tokens)
    digit_ratio = digit_tokens / len(tokens)
    short_ratio = short_tokens / len(tokens)
    if confidence is not None and confidence < 65 and lower_ratio >= 0.20:
        return True
    if confidence is not None and confidence < 65 and quality >= 0.25:
        return True
    if quality >= 0.30 and lower_ratio >= 0.25:
        return True
    if quality >= 0.34 and short_ratio >= 0.55 and digit_ratio >= 0.25:
        return True
    return False


def vector_stroke_line_is_spatial_supplement(line: Any) -> bool:
    return should_keep_spatial_vector_stroke_line(
        line.text.strip(),
        line.confidence,
        line.bbox,
    )


def vector_stroke_text_looks_fragmented(text: str) -> bool:
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if len(tokens) < 8:
        return False
    short_tokens = sum(1 for token in tokens if len(token) <= 2)
    return short_tokens / len(tokens) >= 0.75


def merge_ocr_with_vector_stroke_geometry(
    page: PageExtractionHost,
    ocr_result: OcrPageTextResult,
    vector_result: VectorStrokeOcrResult,
) -> str:
    ocr_text = ocr_result.text
    vector_text = vector_result.text
    if not ocr_text.strip() or not vector_text.strip():
        return ocr_text
    seen_tokens = set(ocr_text_analysis.normalized_text_tokens(ocr_text))
    ocr_lines = ocr_geometry.ocr_candidate_geometry_lines(
        page,
        ocr_result.candidate,
        selected_tokens=seen_tokens,
    )
    if not ocr_lines or not vector_result.lines:
        return ocr_text
    return merge_uncovered_text_geometry_lines(
        ocr_text,
        ocr_lines,
        vector_result.lines,
        candidate_filter=vector_stroke_line_is_spatial_supplement,
        covered_by_base=vector_line_is_covered_by_ocr,
        allow_short_lines=True,
    )


def should_keep_spatial_vector_stroke_line(
    text: str,
    confidence: int | None,
    bbox: tuple[float, float, float, float],
) -> bool:
    if not text or not ocr_page_analysis.VECTOR_SPATIAL_TEXT_RE.search(text):
        return False
    confidence_value = confidence if confidence is not None else 35
    for ch in text:
        if (
            ch.isalnum()
            or ch.isspace()
            or ch in ocr_page_analysis.VECTOR_SPATIAL_ALLOWED_PUNCTUATION
        ):
            continue
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if not tokens:
        return False
    raw_tokens = [
        match.group(0) for match in ocr_text_analysis.TEXT_TOKEN_RE.finditer(text)
    ]
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if width <= 0.0 or height <= 0.0:
        return False
    compact_width = max(28.0, len(text) * 7.0)
    if len(tokens) >= 2:
        if len(tokens) > 3:
            return False
        if not any(ch.isalpha() for ch in text):
            return False
        if width > max(72.0, compact_width) or height > 24.0:
            return False
        if any(token.isalpha() and token.islower() for token in raw_tokens):
            return False
        return confidence_value >= 65
    token = tokens[0]
    raw_token = raw_tokens[0] if raw_tokens else token
    if width > compact_width or height > 24.0:
        return False
    has_digit = any(ch.isdigit() for ch in token)
    has_upper = any(ch.isupper() for ch in text)
    if raw_token.isalpha() and raw_token.islower():
        return False
    if has_digit and confidence_value < 70:
        return False
    if has_upper and confidence_value < 50:
        return False
    if len(token) >= 3:
        return has_digit or has_upper
    if has_digit or has_upper:
        return confidence_value >= 50
    return False


def vector_line_is_covered_by_ocr(vector_line: Any, ocr_lines: list[Any]) -> bool:
    vector_observation = page_geometry.page_observation_from_text_line(
        vector_line,
        source="vector_stroke",
        kind="vector_text_line",
    )
    if vector_observation is None:
        return True
    vector_tokens = set(ocr_text_analysis.normalized_text_tokens(vector_line.text))
    overlapping_texts: list[str] = []
    for ocr_line in ocr_lines:
        ocr_observation = page_geometry.page_observation_from_text_line(
            ocr_line,
            source="ocr",
            kind="ocr_textline",
        )
        if ocr_observation is None:
            continue
        overlap_ratio = page_geometry.observation_overlap_ratio(
            vector_observation,
            ocr_observation,
            denominator="left",
        )
        if overlap_ratio >= 0.20 and vector_tokens.intersection(
            ocr_text_analysis.normalized_text_tokens(ocr_line.text)
        ):
            return True
        if overlap_ratio >= 0.20:
            overlapping_texts.append(ocr_line.text)
    if overlapping_texts:
        spaced_tokens = set(
            ocr_text_analysis.normalized_text_tokens(" ".join(overlapping_texts))
        )
        joined_tokens = set(
            ocr_text_analysis.normalized_text_tokens("".join(overlapping_texts))
        )
        if vector_tokens.intersection(spaced_tokens | joined_tokens):
            return True
        compact_text = "".join(
            ocr_text_analysis.TEXT_TOKEN_RE.findall("".join(overlapping_texts))
        )
        compact_text = compact_text.casefold()
        if compact_text and any(
            len(token) >= 2 and token in compact_text for token in vector_tokens
        ):
            return True
    return False


def should_use_merged_vector_stroke_ocr(
    current: str, ocr_text: str, merged_text: str
) -> bool:
    merged_tokens = ocr_text_analysis.extracted_text_token_count(merged_text)
    ocr_tokens = ocr_text_analysis.extracted_text_token_count(ocr_text)
    current_tokens = ocr_text_analysis.extracted_text_token_count(current)
    if dense_numeric_ocr_supplement_would_reduce_recall(current, ocr_text):
        return False
    if merged_tokens < max(current_tokens, ocr_tokens) + 5:
        return False
    ocr_quality = ocr_text_analysis.text_ocr_quality_score(ocr_text)
    merged_quality = ocr_text_analysis.text_ocr_quality_score(merged_text)
    if (
        ocr_tokens >= 300
        and current_tokens >= int(ocr_tokens * 1.35)
        and merged_tokens > int(ocr_tokens * 1.60)
        and ocr_quality <= 0.24
    ):
        return False
    if (
        ocr_tokens >= 80
        and merged_tokens > int(ocr_tokens * 1.6)
        and merged_quality > ocr_quality + 0.015
    ):
        return False
    if merged_quality > max(0.65, ocr_quality + 0.25):
        return False
    return True


def should_replace_vector_stroke_text_with_ocr(
    current: str, ocr_text: str, vector_text: str
) -> bool:
    if not vector_text.strip() or not ocr_text.strip():
        return False
    current_tokens = ocr_text_analysis.extracted_text_token_count(current)
    ocr_tokens = ocr_text_analysis.extracted_text_token_count(ocr_text)
    vector_tokens = ocr_text_analysis.extracted_text_token_count(vector_text)
    if dense_numeric_ocr_supplement_would_reduce_recall(current, ocr_text):
        return False
    if ocr_tokens < 80 or current_tokens < 80 or vector_tokens < 80:
        return False
    if current_tokens < int(ocr_tokens * 1.20):
        return False
    if ocr_tokens < int(current_tokens * 0.35):
        return False
    ocr_quality = ocr_text_analysis.text_ocr_quality_score(ocr_text)
    if ocr_quality > 0.36:
        return False
    current_quality = ocr_text_analysis.text_ocr_quality_score(current)
    vector_quality = ocr_text_analysis.text_ocr_quality_score(vector_text)
    if (
        ocr_tokens >= 300
        and current_tokens >= int(ocr_tokens * 1.35)
        and ocr_quality <= 0.24
        and current_quality <= 0.24
    ):
        return True
    return ocr_quality + 0.03 < current_quality or ocr_quality + 0.03 < vector_quality
