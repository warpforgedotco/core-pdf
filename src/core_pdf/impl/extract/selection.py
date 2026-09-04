# SPDX-License-Identifier: AGPL-3.0-only
"""Document-selection enrichment and extraction orchestration."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from core_pdf.impl.extract.capture import (
    LearnedUnicodeMap,
    internal_capture_from_program,
)
from core_pdf.impl.extract.contracts import (
    ObservationBatch,
    PageAnalysis,
    RecognitionResult,
    internal_bbox_tuple,
)
from core_pdf.impl.extract.ocr.strokes import (
    GlyphSignature,
    StrokedTextDecode,
    decode_stroked_text_profile_with_alphabet,
)
from core_pdf.impl.extract.pipeline import internal_PageExtraction
from core_pdf.impl.model.glyphs import GlyphUnicodeSemantics, glyph_unicode_semantics
from core_pdf.impl.output import SCHEMA_VERSION, Document, Page
from core_pdf.impl.runtime.execution import ExtractionScope

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
    recognition_by_index: Mapping[int, RecognitionResult] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class internal_StrokedEnrichment:
    """Selection-local recognition replacements learned across compatible pages."""

    recognition_by_index: Mapping[int, RecognitionResult] = field(
        default_factory=lambda: MappingProxyType({})
    )


def internal_unknown_decoder_counts(capture: PageAnalysis) -> Counter[object]:
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
    for glyph in capture.program.glyphs:
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


def internal_document_font_seed_indexes(captures: Sequence[PageAnalysis]) -> tuple[int, ...]:
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
    capture: PageAnalysis,
    ocr: ObservationBatch,
) -> dict[object, dict[bytes, Counter[str]]]:
    votes: dict[object, dict[bytes, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    glyphs = tuple(
        glyph
        for glyph in capture.program.glyphs
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
    extractions: tuple[internal_PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    context: ExtractionScope,
) -> internal_FontEnrichment:
    seed_indexes = internal_document_font_seed_indexes(captures)
    if not seed_indexes:
        return internal_FontEnrichment()
    recognition_by_index: dict[int, RecognitionResult] = {}
    for page_index in seed_indexes:
        context.raise_if_cancelled()
        recognition_by_index[page_index] = extractions[page_index].recognition(context)
    votes: dict[object, dict[bytes, Counter[str]]] = {}
    for page_index, recognition in recognition_by_index.items():
        internal_merge_font_mapping_votes(
            votes,
            internal_font_mapping_votes(captures[page_index], recognition.observations),
        )
    return internal_FontEnrichment(
        seed_indexes=seed_indexes,
        learned_unicode=internal_resolve_document_font_mappings(votes),
        recognition_by_index=MappingProxyType(recognition_by_index),
    )


def internal_capture_uses_learned_unicode(
    capture: PageAnalysis,
    learned_unicode: LearnedUnicodeMap,
) -> bool:
    return bool(learned_unicode) and any(
        glyph.font_decoder in learned_unicode for glyph in capture.program.glyphs
    )


def internal_apply_font_enrichment(
    extractions: tuple[internal_PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    font: internal_FontEnrichment,
) -> tuple[internal_PageExtraction, ...]:
    """Create local pipelines only for non-seed pages changed by the overlay."""
    seed_indexes = frozenset(font.seed_indexes)
    enriched: list[internal_PageExtraction] = []
    for index, (base, capture) in enumerate(zip(extractions, captures, strict=True)):
        recognition = font.recognition_by_index.get(index)
        if index in seed_indexes:
            enriched.append(
                internal_PageExtraction(
                    base.page,
                    capture=base.capture(),
                    plan=base.plan(),
                    recognition=recognition,
                    fields=base.capture().fields,
                    structure=base.internal_structure,
                    hidden_layers=base.internal_hidden_layers,
                )
            )
            continue
        if not internal_capture_uses_learned_unicode(capture, font.learned_unicode):
            enriched.append(base)
            continue
        enriched_capture = internal_capture_from_program(
            base.page,
            capture.program,
            learned_unicode=font.learned_unicode,
            structure=base.internal_structure,
            fields=capture.fields,
            annotations=capture.annotations,
        )
        enriched.append(
            internal_PageExtraction(
                base.page,
                capture=enriched_capture,
                fields=enriched_capture.fields,
                structure=base.internal_structure,
                hidden_layers=base.internal_hidden_layers,
            )
        )
    return tuple(enriched)


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
    decoded: StrokedTextDecode,
) -> RecognitionResult:
    """Build a selection-local zero-raster recognition result."""
    from core_pdf.impl.extract.ocr.vector import internal_stroked_vector_decoded_batch

    observations = internal_stroked_vector_decoded_batch(decoded.observations)
    return RecognitionResult(observations, stroked_vector_alphabet=decoded.alphabet)


def internal_prepare_document_stroked_mappings(
    extractions: tuple[internal_PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    context: ExtractionScope,
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
    recognition_by_index: dict[int, RecognitionResult] = {}
    for page_index in ordered:
        extraction = extractions[page_index]
        capture = extraction.capture()
        recognition = extraction.internal_recognition
        if recognition is None and alphabet:
            extraction.plan()
            decoded = decode_stroked_text_profile_with_alphabet(
                internal_stroked_text_profile(capture),
                alphabet,
            )
            if internal_document_stroked_decode_is_sufficient(decoded):
                recognition_by_index[page_index] = internal_document_stroked_recognition(decoded)
                continue

        if recognition is None:
            recognition = extraction.recognition(context)
        learned = recognition.stroked_vector_alphabet
        if learned:
            internal_merge_document_stroked_alphabet(
                alphabet,
                ambiguous,
                cast(tuple[tuple[GlyphSignature, str], ...], learned),
            )
        recognition_by_index[page_index] = recognition
    return internal_StrokedEnrichment(
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
            fields=base.capture().fields,
            structure=base.internal_structure,
            hidden_layers=base.internal_hidden_layers,
        )
    return tuple(enriched)


def internal_capture_document_pages(
    extractions: tuple[internal_PageExtraction, ...],
    context: ExtractionScope,
) -> tuple[PageAnalysis, ...]:
    captures: list[PageAnalysis] = []
    for extraction in extractions:
        context.raise_if_cancelled()
        captures.append(extraction.capture())
    return tuple(captures)


def internal_assemble_document_pages(
    extractions: tuple[internal_PageExtraction, ...],
    context: ExtractionScope,
) -> tuple[Page, ...]:
    pages: list[Page] = []
    for extraction in extractions:
        context.raise_if_cancelled()
        pages.append(extraction.assembled_page(context))
    return tuple(pages)


def internal_prepare_selection_state(
    extractions: tuple[internal_PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    context: ExtractionScope,
) -> tuple[internal_PageExtraction, ...]:
    """Page pipelines enriched with everything learned across one exact selection."""
    font = internal_prepare_document_font_mappings(extractions, captures, context)
    extractions = internal_apply_font_enrichment(extractions, captures, font)
    stroked = internal_prepare_document_stroked_mappings(
        extractions,
        tuple(extraction.capture() for extraction in extractions),
        context,
    )
    return internal_apply_stroked_enrichment(extractions, stroked)


def extract_document(
    document: Any,
    context: ExtractionScope,
    pages: Sequence[Any],
) -> Document:
    pages = tuple(pages)
    hidden_layers = document.oc_hidden_layers() if pages else frozenset()
    structure_tree = None
    with suppress(IndexError, TypeError, ValueError):
        structure_tree = document.structure

    def page_structure(page: Any) -> Any:
        if structure_tree is None:
            return None
        try:
            return structure_tree.page_structure(page)
        except (IndexError, TypeError, ValueError):
            return None

    fields_by_page: dict[int, list[Any]] = {}
    with suppress(TypeError, ValueError):
        fields_by_page = document.fields_by_page(pages)
    extractions = tuple(
        internal_PageExtraction(
            page,
            fields=fields_by_page.get(int(page.page_number) - 1, ()),
            structure=page_structure(page),
            hidden_layers=hidden_layers,
        )
        for page in pages
    )
    if len(extractions) > 1:
        captures = internal_capture_document_pages(extractions, context)
        if len(captures) == len(pages):
            extractions = internal_prepare_selection_state(extractions, captures, context)
    assembled_pages = internal_assemble_document_pages(extractions, context)
    diagnostics = tuple(diagnostic for page in assembled_pages for diagnostic in page.diagnostics)
    metadata = document.get_metadata()
    return Document(
        pages=assembled_pages,
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version=SCHEMA_VERSION,
    )
