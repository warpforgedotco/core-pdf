# SPDX-License-Identifier: AGPL-3.0-only
"""Document-selection enrichment and extraction orchestration."""

from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

from core_pdf.impl.extract.capture import (
    LearnedUnicodeMap,
    internal_capture_from_program,
)
from core_pdf.impl.extract.contracts import (
    CapturedPage,
    ObservationBatch,
    OcrPassScope,
    ParsedPage,
    RecognitionReport,
    RecognitionResult,
    internal_bbox_tuple,
    internal_candidate,
)
from core_pdf.impl.extract.ocr.strokes import (
    GlyphSignature,
    StrokedTextDecode,
    decode_stroked_text_profile_with_alphabet,
)
from core_pdf.impl.extract.pipeline import internal_PageExtraction, page_extraction
from core_pdf.impl.model.glyphs import GlyphUnicodeSemantics, glyph_unicode_semantics
from core_pdf.impl.output import SCHEMA_VERSION, Diagnostic, Document
from core_pdf.impl.runtime.execution import TaskScope, WorkStage

DOCUMENT_FONT_SEED_LIMIT = 4
DOCUMENT_FONT_SEEDS_PER_DECODER = 2
DOCUMENT_STROKED_MIN_DECODED_RUNS = 20
DOCUMENT_STROKED_MIN_RUN_COVERAGE = 0.70
DOCUMENT_STROKED_MIN_GLYPH_COVERAGE = 0.70


@dataclass(frozen=True, slots=True)
class internal_FontEnrichment:
    """Immutable learned Unicode overlay for one exact document selection."""

    seed_indexes: tuple[int, ...] = ()
    learned_unicode: LearnedUnicodeMap = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class internal_StrokedEnrichment:
    """Selection-local recognition replacements learned across compatible pages."""

    seed_count: int = 0
    reused_pages: int = 0
    recognition_by_index: Mapping[int, RecognitionResult] = field(
        default_factory=lambda: MappingProxyType({})
    )


def internal_unknown_decoder_counts(capture: CapturedPage) -> Counter[object]:
    counts: Counter[object] = Counter()
    quality = capture.evidence.text_quality
    corrupt = (
        capture.evidence.visible_native_characters >= 24
        and quality.noise_score >= 0.20
        and quality.wordlike_ratio < 0.20
    )
    glyph_evidence = capture.evidence.glyphs
    if not corrupt and not glyph_evidence.unknown_glyphs and not glyph_evidence.unsupported_glyphs:
        return counts
    for glyph in capture.program.products.glyphs:
        decoder = glyph.font_decoder
        if (
            decoder is None
            or not glyph.visible
            or not glyph.text
            or glyph.text.isspace()
            or not glyph.code_bytes
            or (
                not corrupt
                and glyph_unicode_semantics(glyph.text, glyph.unicode_source)
                not in {
                    GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER,
                    GlyphUnicodeSemantics.UNSUPPORTED,
                }
            )
        ):
            continue
        counts[decoder] += 1
    return counts


def internal_document_font_seed_indexes(captures: Sequence[CapturedPage]) -> tuple[int, ...]:
    pages_by_decoder: dict[object, list[tuple[int, int]]] = defaultdict(list)
    for page_index, capture in enumerate(captures):
        for decoder, count in internal_unknown_decoder_counts(capture).items():
            if count >= 8:
                pages_by_decoder[decoder].append((page_index, count))
    page_scores: Counter[int] = Counter()
    for entries in pages_by_decoder.values():
        if len(entries) < 2 or sum(count for _, count in entries) < 32:
            continue
        for page_index, count in sorted(entries, key=lambda item: -item[1])[
            :DOCUMENT_FONT_SEEDS_PER_DECODER
        ]:
            page_scores[page_index] += count
    return tuple(
        page_index
        for page_index, ignored_score in page_scores.most_common(DOCUMENT_FONT_SEED_LIMIT)
    )


def internal_font_mapping_votes(
    capture: CapturedPage,
    ocr: ObservationBatch,
) -> dict[object, dict[bytes, Counter[str]]]:
    votes: dict[object, dict[bytes, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    glyphs = tuple(
        glyph
        for glyph in capture.program.products.glyphs
        if glyph.visible
        and glyph.code_bytes
        and len(glyph.text) == 1
        and not glyph.text.isspace()
        and int(glyph.rotation_angle) % 360 == 0
    )
    if not glyphs:
        return votes
    # Each OCR word only inspects glyphs in its own vertical band, so bucket the
    # page's glyphs by ink center once instead of rescanning them all per word.
    glyphs_by_y = sorted(glyphs, key=lambda glyph: (glyph.ink_bbox[1] + glyph.ink_bbox[3]) * 0.5)
    y_centers = [(glyph.ink_bbox[1] + glyph.ink_bbox[3]) * 0.5 for glyph in glyphs_by_y]
    for text, bbox, confidence in zip(ocr.text, ocr.bbox, ocr.confidence, strict=True):
        if not math.isfinite(float(confidence)) or float(confidence) < 90.0:
            continue
        characters = tuple(character for character in text if not character.isspace())
        if len(characters) < 3:
            continue
        x0, y0, x1, y1 = internal_bbox_tuple(bbox)
        tolerance = max(1.0, (y1 - y0) * 0.10)
        y_start = bisect_left(y_centers, y0 - tolerance)
        y_stop = bisect_right(y_centers, y1 + tolerance)
        aligned = tuple(
            sorted(
                (
                    glyph
                    for glyph in glyphs_by_y[y_start:y_stop]
                    if x0 - tolerance
                    <= (glyph.ink_bbox[0] + glyph.ink_bbox[2]) * 0.5
                    <= x1 + tolerance
                ),
                key=lambda glyph: (glyph.ink_bbox[1], glyph.ink_bbox[0], glyph.seqno),
            )
        )
        if len(aligned) != len(characters):
            continue
        known_pairs = tuple(
            (glyph.text.casefold(), character.casefold())
            for glyph, character in zip(aligned, characters, strict=True)
            if glyph_unicode_semantics(glyph.text, glyph.unicode_source)
            in {GlyphUnicodeSemantics.AUTHORITATIVE, GlyphUnicodeSemantics.HEURISTIC}
        )
        if (
            known_pairs
            and sum(left == right for left, right in known_pairs) / len(known_pairs) < 0.8
        ):
            continue
        for glyph, character in zip(aligned, characters, strict=True):
            decoder = glyph.font_decoder
            if (
                decoder is None
                or glyph_unicode_semantics(glyph.text, glyph.unicode_source)
                not in {
                    GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER,
                    GlyphUnicodeSemantics.UNSUPPORTED,
                }
                or not character.isprintable()
            ):
                continue
            votes[decoder][glyph.code_bytes][character] += 1
    return votes


def internal_merge_font_mapping_votes(
    destination: dict[object, dict[bytes, Counter[str]]],
    source: dict[object, dict[bytes, Counter[str]]],
) -> None:
    for decoder, by_code in source.items():
        destination_codes = destination.setdefault(decoder, {})
        for code_bytes, counts in by_code.items():
            destination_codes.setdefault(code_bytes, Counter()).update(counts)


def internal_resolve_document_font_mappings(
    votes: dict[object, dict[bytes, Counter[str]]],
) -> LearnedUnicodeMap:
    resolved: dict[object, Mapping[bytes, str]] = {}
    for decoder, by_code in votes.items():
        mapping: dict[bytes, str] = {}
        for code_bytes, counts in by_code.items():
            if not counts:
                continue
            character, count = counts.most_common(1)[0]
            total = counts.total()
            if count >= 2 and count / total >= 0.90:
                mapping[code_bytes] = character
        if mapping:
            resolved[decoder] = MappingProxyType(mapping)
    return MappingProxyType(resolved)


def internal_prepare_document_font_mappings(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
) -> internal_FontEnrichment:
    seed_indexes = internal_document_font_seed_indexes(captures)
    if not seed_indexes:
        return internal_FontEnrichment()
    ocr_by_index: dict[int, ObservationBatch] = {}
    for completed in context.map_completed(
        lambda page_index: page_extraction(pages[page_index]).recognition(context).observations,
        seed_indexes,
        stage=WorkStage.PAGE,
    ):
        ocr_by_index[seed_indexes[completed.index]] = completed.value
    votes: dict[object, dict[bytes, Counter[str]]] = {}
    for page_index, ocr in ocr_by_index.items():
        internal_merge_font_mapping_votes(
            votes,
            internal_font_mapping_votes(captures[page_index], ocr),
        )
    return internal_FontEnrichment(
        seed_indexes=seed_indexes,
        learned_unicode=internal_resolve_document_font_mappings(votes),
    )


def internal_capture_uses_learned_unicode(
    capture: CapturedPage,
    learned_unicode: LearnedUnicodeMap,
) -> bool:
    return bool(learned_unicode) and any(
        glyph.font_decoder in learned_unicode for glyph in capture.program.products.glyphs
    )


def internal_apply_font_enrichment(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    font: internal_FontEnrichment,
) -> tuple[internal_PageExtraction, ...]:
    """Create local pipelines only for non-seed pages changed by the overlay."""
    seed_indexes = frozenset(font.seed_indexes)
    extractions: list[internal_PageExtraction] = []
    for index, (page, capture) in enumerate(zip(pages, captures, strict=True)):
        base = page_extraction(page)
        if index in seed_indexes or not internal_capture_uses_learned_unicode(
            capture, font.learned_unicode
        ):
            extractions.append(base)
            continue
        started = time.perf_counter()
        enriched_capture = internal_capture_from_program(
            page,
            capture.program,
            learned_unicode=font.learned_unicode,
        )
        extractions.append(
            internal_PageExtraction(
                page,
                capture=enriched_capture,
                capture_seconds=time.perf_counter() - started,
            )
        )
    return tuple(extractions)


def internal_merge_document_stroked_alphabet(
    destination: dict[GlyphSignature, str],
    ambiguous: set[GlyphSignature],
    source: Iterable[tuple[GlyphSignature, str]],
) -> None:
    """Merge exact glyph mappings and permanently exclude cross-page conflicts."""
    for signature, character in source:
        if signature in ambiguous:
            continue
        if signature not in destination:
            destination[signature] = character
        elif destination[signature] != character:
            destination.pop(signature)
            ambiguous.add(signature)


def internal_document_stroked_decode_is_sufficient(decoded: StrokedTextDecode) -> bool:
    return bool(
        len(decoded.observations) >= DOCUMENT_STROKED_MIN_DECODED_RUNS
        and decoded.decoded_candidate_runs >= DOCUMENT_STROKED_MIN_DECODED_RUNS
        and decoded.candidate_run_coverage >= DOCUMENT_STROKED_MIN_RUN_COVERAGE
        and decoded.candidate_glyph_coverage >= DOCUMENT_STROKED_MIN_GLYPH_COVERAGE
    )


def internal_document_stroked_recognition(
    capture: CapturedPage,
    decoded: StrokedTextDecode,
    *,
    seconds: float,
    seed_pages: tuple[int, ...],
    alphabet_size: int,
) -> RecognitionResult:
    """Build a selection-local zero-raster recognition result."""
    from core_pdf.impl.extract.ocr.vector import internal_stroked_vector_decoded_batch

    observations = internal_stroked_vector_decoded_batch(decoded.observations)
    candidate = internal_candidate(-1, observations)
    bbox = capture.evidence.stroked_vector_text.bbox
    pass_report: dict[str, object] = {
        "name": "document-stroked-glyphs",
        "scope": OcrPassScope.STROKED_VECTOR_TEXT.value,
        "scale": 0.0,
        "modes": (),
        "recognize_words": False,
        "character_confidence_threshold": None,
        "task_count": 0,
        "raster_pixels": 0,
        "skipped_raster_pixels": 0,
        "image_text_preflight": (),
        "region_stage": "document-glyph-alphabet",
        "region_boxes": (bbox,) if bbox is not None else (),
        "skipped_region_boxes": (),
        "full_page_fallback": False,
        "elapsed_seconds": seconds,
        "render_timings": {},
        "recognition_seconds": 0.0,
        "setup_seconds": 0.0,
        "api_seconds": 0.0,
        "iterator_seconds": 0.0,
        "cleanup_seconds": 0.0,
        "candidate_seconds": 0.0,
        "recognition_statuses": (),
        "accepted_additions": len(observations),
        "adaptive_retry_scale": None,
        "adaptive_preflight": None,
        "adaptive_rescue_decision": None,
        "adaptive_rescue": None,
        "pixel_budget": 0,
        "rectangles": (),
        "selected": True,
        **candidate.metrics.as_record(),
    }
    stroked_vector_decode = {
        "seconds": seconds,
        "eligible_seeds": 0,
        "aligned_seeds": 0,
        "accepted_seeds": 0,
        "initial_signatures": decoded.initial_signatures,
        "learned_signatures": decoded.learned_signatures,
        "approximate_signatures": decoded.approximate_signatures,
        "candidate_runs": decoded.candidate_runs,
        "decoded_candidate_runs": decoded.decoded_candidate_runs,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "additions": len(decoded.observations),
        "corrections": 0,
        "document_reuse": True,
    }
    stroked_vector_packed = {
        "accepted": True,
        "cells": 0,
        "raster_pixels": 0,
        "unmapped_observations": 0,
        "fallback_used": False,
        "document_reuse": True,
    }
    document_stroked_glyphs = {
        "role": "reuse",
        "seed_pages": seed_pages,
        "alphabet_size": alphabet_size,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "seconds": seconds,
    }
    return RecognitionResult(
        observations,
        RecognitionReport(
            passes=(pass_report,),
            candidates=(
                {
                    "name": "document-stroked-glyphs",
                    "mode": -1,
                    "selected": True,
                    **candidate.metrics.as_record(),
                },
            ),
            stroked_vector_decode=stroked_vector_decode,
            stroked_vector_packed=stroked_vector_packed,
            document_stroked_glyphs=document_stroked_glyphs,
            stroked_vector_alphabet=decoded.alphabet,
        ),
    )


def internal_prepare_document_stroked_mappings(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
    extractions: tuple[internal_PageExtraction, ...] | None = None,
) -> internal_StrokedEnrichment:
    """OCR the richest flattened-font page, then decode compatible pages structurally."""
    indexes = tuple(
        index
        for index, capture in enumerate(captures)
        if capture.evidence.stroked_vector_text.trusted
    )
    if len(indexes) < 2:
        return internal_StrokedEnrichment()
    from core_pdf.impl.extract.ocr.vector import internal_stroked_text_profile

    ordered = tuple(
        sorted(
            indexes,
            key=lambda index: (
                -captures[index].evidence.stroked_vector_text.candidate_paths,
                index,
            ),
        )
    )
    alphabet: dict[GlyphSignature, str] = {}
    ambiguous: set[GlyphSignature] = set()
    seed_indexes: list[int] = []
    reused_pages = 0
    recognition_by_index: dict[int, RecognitionResult] = {}
    if extractions is None:
        extractions = tuple(page_extraction(page) for page in pages)
    for page_index in ordered:
        page = pages[page_index]
        extraction = extractions[page_index]
        capture = extraction.capture()
        with page.internal_page_lock:
            recognition = extraction.internal_recognition
        if recognition is None and alphabet:
            extraction.plan()
            started = time.perf_counter()
            decoded = decode_stroked_text_profile_with_alphabet(
                internal_stroked_text_profile(capture),
                alphabet,
            )
            seconds = time.perf_counter() - started
            if internal_document_stroked_decode_is_sufficient(decoded):
                recognition_by_index[page_index] = internal_document_stroked_recognition(
                    capture,
                    decoded,
                    seconds=seconds,
                    seed_pages=tuple(int(pages[index].page_number) for index in seed_indexes),
                    alphabet_size=len(alphabet),
                )
                reused_pages += 1
                continue

        if recognition is None:
            recognition = extraction.recognition(context)
        learned = recognition.report.stroked_vector_alphabet
        if learned:
            internal_merge_document_stroked_alphabet(
                alphabet,
                ambiguous,
                cast(tuple[tuple[GlyphSignature, str], ...], learned),
            )
        seed_indexes.append(page_index)
        seed_report = {
            "role": "seed",
            "seed_pages": tuple(int(pages[index].page_number) for index in seed_indexes),
            "alphabet_size": len(alphabet),
            "ambiguous_signatures": len(ambiguous),
        }
        recognition_by_index[page_index] = replace(
            recognition,
            report=replace(recognition.report, document_stroked_glyphs=seed_report),
        )
    return internal_StrokedEnrichment(
        seed_count=len(seed_indexes),
        reused_pages=reused_pages,
        recognition_by_index=MappingProxyType(recognition_by_index),
    )


def internal_apply_stroked_enrichment(
    extractions: tuple[internal_PageExtraction, ...],
    stroked: internal_StrokedEnrichment,
) -> tuple[internal_PageExtraction, ...]:
    if not stroked.recognition_by_index:
        return extractions
    enriched = list(extractions)
    for index, recognition in stroked.recognition_by_index.items():
        base = extractions[index]
        enriched[index] = internal_PageExtraction(
            base.page,
            capture=base.capture(),
            plan=base.plan(),
            recognition=recognition,
            capture_seconds=base.internal_capture_seconds,
            planning_seconds=base.internal_planning_seconds,
            ocr_seconds=base.internal_ocr_seconds,
        )
    return tuple(enriched)


def internal_page_chunks(
    pages: tuple[Any, ...],
    worker_count: int,
) -> tuple[tuple[int, tuple[Any, ...]], ...]:
    """Bound scheduler overhead while retaining enough chunks for load balancing."""
    chunk_size = max(1, min(32, math.ceil(len(pages) / max(1, worker_count * 4))))
    return tuple(
        (start, pages[start : start + chunk_size]) for start in range(0, len(pages), chunk_size)
    )


def internal_capture_document_pages(
    pages: tuple[Any, ...],
    context: TaskScope,
) -> tuple[CapturedPage, ...]:
    captures_by_index: list[CapturedPage | None] = [None] * len(pages)

    def capture_chunk(
        indexed_pages: tuple[int, tuple[Any, ...]],
    ) -> tuple[int, tuple[CapturedPage, ...]]:
        start, chunk = indexed_pages
        captures: list[CapturedPage] = []
        for page in chunk:
            context.raise_if_cancelled()
            captures.append(page_extraction(page).capture())
        return start, tuple(captures)

    chunks = internal_page_chunks(pages, context.runtime.max_workers)
    for completed in context.map_completed(capture_chunk, chunks, stage=WorkStage.PAGE):
        start, captures = completed.value
        captures_by_index[start : start + len(captures)] = captures
    return tuple(capture for capture in captures_by_index if capture is not None)


def internal_parse_document_pages(
    extractions: tuple[internal_PageExtraction, ...],
    context: TaskScope,
) -> tuple[ParsedPage, ...]:
    parsed_by_index: list[ParsedPage | None] = [None] * len(extractions)
    futures: dict[int, Future[ParsedPage]] = {}
    direct_indexes: list[int] = []
    for index, extraction in enumerate(extractions):
        plan = extraction.plan()
        requires_ocr = extraction.internal_recognition is None and (
            bool(plan.ocr_passes) or plan.verify_hidden_text
        )
        if requires_ocr:
            futures[index] = context.submit(
                extraction.parsed_page,
                context,
                stage=WorkStage.PAGE,
            )
        else:
            direct_indexes.append(index)
    try:
        for index in direct_indexes:
            context.raise_if_cancelled()
            parsed_by_index[index] = extractions[index].parsed_page(context)
        for index, future in futures.items():
            parsed_by_index[index] = future.result()
    finally:
        for future in futures.values():
            future.cancel()
    return tuple(page for page in parsed_by_index if page is not None)


def internal_prepare_selection_state(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
) -> tuple[internal_PageExtraction, ...]:
    """Page pipelines enriched with everything learned across one exact selection."""
    font = internal_prepare_document_font_mappings(pages, captures, context)
    extractions = internal_apply_font_enrichment(pages, captures, font)
    stroked = internal_prepare_document_stroked_mappings(
        pages,
        tuple(extraction.capture() for extraction in extractions),
        context,
        extractions,
    )
    return internal_apply_stroked_enrichment(extractions, stroked)


def extract_document(
    document: Any,
    context: TaskScope,
    pages: Sequence[Any],
) -> Document:
    pages = tuple(pages)
    parsed_pages: tuple[ParsedPage, ...]
    extractions: tuple[internal_PageExtraction, ...]
    if len(pages) == 1:
        extractions = (page_extraction(pages[0]),)
        parsed_pages = (extractions[0].parsed_page(context),)
    else:
        captures = internal_capture_document_pages(pages, context)
        if len(captures) == len(pages):
            extractions = internal_prepare_selection_state(pages, captures, context)
        else:
            extractions = tuple(page_extraction(page) for page in pages)
        parsed_pages = internal_parse_document_pages(extractions, context)
    diagnostics = tuple(
        Diagnostic("parse", message, page_number=page.page_number)
        for page in parsed_pages
        for message in page.diagnostics
    )
    metadata = document.get_metadata()
    return Document(
        pages=tuple(extraction.assembled_page(context) for extraction in extractions),
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version=SCHEMA_VERSION,
    )
