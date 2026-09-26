# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract.contracts import ObservationBatch, bbox_tuple
from core_pdf.impl.extract.selection import (
    assemble_document,
    prepare_document_pages,
)
from core_pdf.impl.glyphs import GlyphUnicodeSemantics, glyph_unicode_semantics
from core_pdf.impl.output.model import Document
from core_pdf.impl.types import Record, frozen_setattr
from core_pdf_ocr.impl.extract.capture import (
    LearnedUnicodeMap,
    capture_from_program,
)
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, RecognitionResult
from core_pdf_ocr.impl.extract.ocr.strokes import (
    GlyphSignature,
    StrokedTextDecode,
    decode_stroked_text_profile_with_alphabet,
)
from core_pdf_ocr.impl.extract.pipeline import PageExtraction

if TYPE_CHECKING:
    from core_pdf.impl.document.document import PdfDocument
    from core_pdf.impl.document.page import PdfPage


DOCUMENT_FONT_SEED_LIMIT = 4
DOCUMENT_FONT_SEEDS_PER_DECODER = 2
DOCUMENT_STROKED_MIN_DECODED_RUNS = 20
DOCUMENT_STROKED_MIN_RUN_COVERAGE = 0.70
DOCUMENT_STROKED_MIN_GLYPH_COVERAGE = 0.70


class FontEnrichment(Record):
    __slots__ = ("learned_unicode", "recognition_by_index")

    learned_unicode: LearnedUnicodeMap
    recognition_by_index: Mapping[int, RecognitionResult]

    __fields__: ClassVar[tuple[str, ...]] = ("learned_unicode", "recognition_by_index")
    __match_args__ = ("learned_unicode", "recognition_by_index")

    def __init__(
        self,
        learned_unicode: LearnedUnicodeMap | None = None,
        recognition_by_index: Mapping[int, RecognitionResult] | None = None,
    ) -> None:
        frozen_setattr(
            self,
            "learned_unicode",
            (lambda: MappingProxyType({}))() if learned_unicode is None else learned_unicode,
        )
        frozen_setattr(
            self,
            "recognition_by_index",
            (lambda: MappingProxyType({}))()
            if recognition_by_index is None
            else recognition_by_index,
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.learned_unicode == other.learned_unicode
            and self.recognition_by_index == other.recognition_by_index
        )

    def __hash__(self) -> int:
        return hash((self.learned_unicode, self.recognition_by_index))


def unknown_decoder_counts(capture: PageAnalysis) -> Counter[object]:
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


def document_font_seed_indexes(captures: Sequence[PageAnalysis]) -> tuple[int, ...]:
    pages_by_decoder: dict[object, list[tuple[int, int]]] = defaultdict(list)
    for page_index, capture in enumerate(captures):
        for decoder, count in unknown_decoder_counts(capture).items():
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


def font_mapping_votes(
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
    glyphs_by_y = sorted(glyphs, key=lambda glyph: (glyph.ink_bbox[1] + glyph.ink_bbox[3]) * 0.5)
    y_centers = [(glyph.ink_bbox[1] + glyph.ink_bbox[3]) * 0.5 for glyph in glyphs_by_y]
    for text, bbox, confidence in zip(ocr.text, ocr.bbox, ocr.confidence, strict=True):
        if not math.isfinite(float(confidence)) or float(confidence) < 90.0:
            continue
        characters = tuple(character for character in text if not character.isspace())
        if len(characters) < 3:
            continue
        x0, y0, x1, y1 = bbox_tuple(bbox)
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
                key=lambda glyph: (glyph.ink_bbox[0], glyph.seqno),
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


def merge_font_mapping_votes(
    destination: dict[object, dict[bytes, Counter[str]]],
    source: dict[object, dict[bytes, Counter[str]]],
) -> None:
    for decoder, by_code in source.items():
        destination_codes = destination.setdefault(decoder, {})
        for code_bytes, counts in by_code.items():
            destination_codes.setdefault(code_bytes, Counter()).update(counts)


def resolve_document_font_mappings(
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


def prepare_document_font_mappings(
    extractions: tuple[PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    context: ExtractionScope,
) -> FontEnrichment:
    seed_indexes = document_font_seed_indexes(captures)
    if not seed_indexes:
        return FontEnrichment()
    recognition_by_index: dict[int, RecognitionResult] = {}
    for page_index in seed_indexes:
        context.raise_if_cancelled()
        recognition_by_index[page_index] = extractions[page_index].recognize(context)
    votes: dict[object, dict[bytes, Counter[str]]] = {}
    for page_index, recognition in recognition_by_index.items():
        merge_font_mapping_votes(
            votes,
            font_mapping_votes(captures[page_index], recognition.observations),
        )
    return FontEnrichment(
        learned_unicode=resolve_document_font_mappings(votes),
        recognition_by_index=MappingProxyType(recognition_by_index),
    )


def capture_uses_learned_unicode(
    capture: PageAnalysis,
    learned_unicode: LearnedUnicodeMap,
) -> bool:
    return bool(learned_unicode) and any(
        glyph.font_decoder in learned_unicode for glyph in capture.program.glyphs
    )


def apply_font_enrichment(
    extractions: tuple[PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    font: FontEnrichment,
) -> tuple[PageExtraction, ...]:
    enriched: list[PageExtraction] = []
    for index, (base, capture) in enumerate(zip(extractions, captures, strict=True)):
        recognition = font.recognition_by_index.get(index)
        if recognition is not None:
            enriched.append(
                PageExtraction(
                    base.page,
                    capture=base.capture,
                    plan=base.plan,
                    recognition=recognition,
                    fields=base.capture.fields,
                    structure=base.structure_value,
                    hidden_layers=base.hidden_layer_names,
                    stroked_profile=base.stroked_profile_of,
                )
            )
            continue
        if not capture_uses_learned_unicode(capture, font.learned_unicode):
            enriched.append(base)
            continue
        enriched_capture = capture_from_program(
            base.page,
            capture.program,
            learned_unicode=font.learned_unicode,
            structure=base.structure_value,
            fields=capture.fields,
            annotations=capture.annotations,
        )
        enriched.append(
            PageExtraction(
                base.page,
                capture=enriched_capture,
                fields=enriched_capture.fields,
                structure=base.structure_value,
                hidden_layers=base.hidden_layer_names,
            )
        )
    return tuple(enriched)


def merge_document_stroked_alphabet(
    destination: dict[GlyphSignature, str],
    ambiguous: set[GlyphSignature],
    source: Iterable[tuple[GlyphSignature, str]],
) -> None:
    for signature, character in source:
        if signature in ambiguous:
            continue
        if signature not in destination:
            destination[signature] = character
        elif destination[signature] != character:
            destination.pop(signature)
            ambiguous.add(signature)


def document_stroked_decode_is_sufficient(decoded: StrokedTextDecode) -> bool:
    return bool(
        len(decoded.observations) >= DOCUMENT_STROKED_MIN_DECODED_RUNS
        and decoded.decoded_candidate_runs >= DOCUMENT_STROKED_MIN_DECODED_RUNS
        and decoded.candidate_run_coverage >= DOCUMENT_STROKED_MIN_RUN_COVERAGE
        and decoded.candidate_glyph_coverage >= DOCUMENT_STROKED_MIN_GLYPH_COVERAGE
    )


def document_stroked_recognition(
    decoded: StrokedTextDecode,
) -> RecognitionResult:
    from core_pdf_ocr.impl.extract.ocr.vector import stroked_vector_decoded_batch

    observations = stroked_vector_decoded_batch(decoded.observations)
    return RecognitionResult(observations, stroked_vector_alphabet=decoded.alphabet)


def prepare_document_stroked_mappings(
    extractions: tuple[PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    context: ExtractionScope,
) -> Mapping[int, RecognitionResult]:
    indexes = tuple(
        index
        for index, capture in enumerate(captures)
        if capture.evidence.stroked_vector_text.trusted
    )
    if len(indexes) < 2:
        return MappingProxyType({})
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
        context.raise_if_cancelled()
        extraction = extractions[page_index]
        recognition = extraction.recognition_result
        if recognition is None and alphabet and (profile := extraction.stroked_profile) is not None:
            decoded = decode_stroked_text_profile_with_alphabet(
                profile,
                alphabet,
            )
            if document_stroked_decode_is_sufficient(decoded):
                recognition_by_index[page_index] = document_stroked_recognition(decoded)
                continue

        if recognition is None:
            recognition = extraction.recognize(context)
        learned = recognition.stroked_vector_alphabet
        if learned:
            merge_document_stroked_alphabet(
                alphabet,
                ambiguous,
                learned,
            )
        recognition_by_index[page_index] = recognition
    return MappingProxyType(recognition_by_index)


def apply_stroked_enrichment(
    extractions: tuple[PageExtraction, ...],
    recognition_by_index: Mapping[int, RecognitionResult],
) -> tuple[PageExtraction, ...]:
    if not recognition_by_index:
        return extractions
    enriched = list(extractions)
    for index, recognition in recognition_by_index.items():
        base = extractions[index]
        enriched[index] = PageExtraction(
            base.page,
            capture=base.capture,
            plan=base.plan,
            recognition=recognition,
            fields=base.capture.fields,
            structure=base.structure_value,
            hidden_layers=base.hidden_layer_names,
            stroked_profile=base.stroked_profile_of,
        )
    return tuple(enriched)


def capture_document_pages(
    extractions: tuple[PageExtraction, ...],
    context: ExtractionScope,
) -> tuple[PageAnalysis, ...]:
    captures: list[PageAnalysis] = []
    for extraction in extractions:
        context.raise_if_cancelled()
        captures.append(extraction.capture)
    return tuple(captures)


def prepare_selection_state(
    extractions: tuple[PageExtraction, ...],
    captures: tuple[PageAnalysis, ...],
    context: ExtractionScope,
) -> tuple[PageExtraction, ...]:
    font = prepare_document_font_mappings(extractions, captures, context)
    extractions = apply_font_enrichment(extractions, captures, font)
    stroked = prepare_document_stroked_mappings(
        extractions,
        tuple(extraction.capture for extraction in extractions),
        context,
    )
    return apply_stroked_enrichment(extractions, stroked)


def extract_document(
    document: PdfDocument,
    context: ExtractionScope,
    pages: Sequence[PdfPage],
) -> Document:
    pages = tuple(pages)
    extractions = prepare_document_pages(document, pages, PageExtraction)
    if len(extractions) > 1:
        captures = capture_document_pages(extractions, context)
        extractions = prepare_selection_state(extractions, captures, context)
    return assemble_document(document, extractions, context)
