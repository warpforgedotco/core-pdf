# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestration and per-page caching."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import replace
from typing import Any, cast

from core_pdf.impl.engine.model.glyphs import GlyphUnicodeSemantics, glyph_unicode_semantics
from core_pdf.impl.engine.parse.capture import (
    CAPTURED_PAGE_CACHE_KEY,
    capture_page,
    internal_capture_from_program,
    internal_learned_glyph_text,
)
from core_pdf.impl.engine.parse.emit import (
    assemble_page,
)
from core_pdf.impl.engine.parse.fusion import (
    fuse_observations,
)
from core_pdf.impl.engine.parse.layout import layout_blocks_with_evidence
from core_pdf.impl.engine.parse.model import (
    CapturedPage,
    ObservationBatch,
    OcrPassScope,
    ParsedBlock,
    ParsedPage,
    ParseReport,
    ReadingOrderEvidence,
    RecognitionReport,
    RecognitionResult,
    WorkPlan,
    internal_bbox_tuple,
    internal_candidate,
)
from core_pdf.impl.engine.parse.route import (
    plan_page,
)
from core_pdf.impl.engine.parse.stroked_text import (
    GlyphSignature,
    StrokedTextDecode,
    decode_stroked_text_profile_with_alphabet,
)
from core_pdf.impl.engine.parse.tables import (
    extract_tables,
)
from core_pdf.impl.engine.spec.s_07_document.page_links import resolve_destination_value
from core_pdf.impl.engine.structured import (
    SCHEMA_VERSION,
    Annotation,
    Diagnostic,
    Document,
    Figure,
    FormField,
    Link,
    Table,
)
from core_pdf.impl.runtime.execution import TaskScope, WorkStage

PARSED_PAGE_CACHE_KEY = "parsed_page_v5"
PARSE_REPORT_CACHE_KEY = "parse_report_v1"
ASSEMBLED_PAGE_CACHE_KEY = "assembled_page_v2"
PAGE_EXTRACTION_CACHE_KEY = "page_extraction_v3"


def internal_report_number(
    report: Mapping[str, object],
    key: str,
    default: int | float = 0,
) -> int | float:
    """Read one numeric diagnostic at the untyped OCR/capture report boundary."""
    value = report.get(key, default)
    return value if isinstance(value, (int, float)) else default


class internal_PageExtraction:
    """Lazily materialized extraction products for one page."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.started = time.perf_counter()
        self.internal_capture: CapturedPage | None = None
        self.internal_captured_at: float | None = None
        self.internal_plan: WorkPlan | None = None
        self.internal_planned_at: float | None = None
        self.internal_recognition: RecognitionResult | None = None
        self.internal_recognized_at: float | None = None
        self.internal_observations: ObservationBatch | None = None
        self.internal_fused_at: float | None = None
        self.internal_tables: tuple[Table, ...] | None = None
        self.internal_tabled_at: float | None = None
        self.internal_layout: tuple[tuple[ParsedBlock, ...], str, ReadingOrderEvidence] | None = (
            None
        )
        self.internal_layout_finished_at: float | None = None

    def capture(self) -> CapturedPage:
        if self.internal_capture is not None:
            return self.internal_capture
        self.internal_capture = capture_page(self.page)
        self.internal_captured_at = time.perf_counter()
        return self.internal_capture

    def plan(self) -> WorkPlan:
        if self.internal_plan is not None:
            return self.internal_plan
        self.internal_plan = plan_page(self.capture())
        self.internal_planned_at = time.perf_counter()
        return self.internal_plan

    def recognition(self, context: TaskScope) -> RecognitionResult:
        if self.internal_recognition is None:
            plan = self.plan()
            if plan.ocr_passes:
                from core_pdf.impl.engine.parse.ocr import recognize_page

                self.internal_recognition = recognize_page(self.capture(), plan, context)
            else:
                # recognize_page() returns exactly this for a plan with no OCR
                # passes. Short-circuiting keeps parse.ocr — and with it
                # tesserocr, PIL and the rasterizer — off the native-text path.
                self.internal_recognition = RecognitionResult(
                    ObservationBatch.empty(), RecognitionReport()
                )
            self.internal_recognized_at = time.perf_counter()
        return self.internal_recognition

    def ocr(self, context: TaskScope) -> ObservationBatch:
        return self.recognition(context).observations

    def observations(self, context: TaskScope) -> ObservationBatch:
        if self.internal_observations is None:
            capture = self.capture()
            self.internal_observations = fuse_observations(
                capture.observations,
                self.ocr(context),
                self.plan(),
            )
            self.internal_fused_at = time.perf_counter()
        return self.internal_observations

    def tables(self, context: TaskScope) -> tuple[Table, ...]:
        if self.internal_tables is None:
            self.internal_tables = extract_tables(self.capture(), self.observations(context))
            self.internal_tabled_at = time.perf_counter()
        return self.internal_tables

    def internal_image_obstacles(self) -> tuple[tuple[float, float, float, float], ...]:
        capture = self.capture()
        return tuple(
            box
            for box in capture.evidence.image_boxes
            if 0.01 <= ((box[2] - box[0]) * (box[3] - box[1])) / capture.evidence.page_area < 0.65
        )

    def layout(self, context: TaskScope) -> tuple[ParsedBlock, ...]:
        if self.internal_layout is not None:
            return self.internal_layout[0]
        observations = self.observations(context)
        capture = self.capture()
        table_obstacles = tuple(
            table.bbox for table in self.tables(context) if table.bbox is not None
        )
        use_xy_cut = not (
            capture.evidence.image_count >= 8 and 0.05 <= capture.evidence.image_area_ratio < 0.65
        )
        blocks, order_evidence = layout_blocks_with_evidence(
            observations,
            obstacles=(*table_obstacles, *self.internal_image_obstacles()),
            use_xy_cut=use_xy_cut,
            rotation=int(getattr(self.page, "rotation", 0) or 0),
            page_width=float(capture.page.width),
            page_height=float(capture.page.height),
        )
        self.internal_layout = (
            blocks,
            "xy-cut" if use_xy_cut else "row-order",
            order_evidence,
        )
        self.internal_layout_finished_at = time.perf_counter()
        return blocks

    def parsed_page(self, context: TaskScope) -> ParsedPage:
        cache = self.page.extraction_cache
        cached = cache.get_as(PARSED_PAGE_CACHE_KEY, ParsedPage)
        if cached is not None:
            return cached
        capture = self.capture()
        plan = self.plan()
        recognition = self.recognition(context)
        ocr = recognition.observations
        recognition_report = recognition.report
        observations = self.observations(context)
        tables = self.tables(context)
        blocks = self.layout(context)
        figures = (
            ()
            if capture.evidence.full_page_image
            else tuple(
                Figure(order=index, bbox=box, kind="image", metadata={"source": "capture"})
                for index, box in enumerate(capture.evidence.image_boxes)
            )
        )
        finished = self.internal_layout_finished_at or time.perf_counter()
        ocr_diagnostics = recognition_report.passes
        newstroke_diagnostics = capture.newstroke_report
        stroked_decode_diagnostics = recognition_report.stroked_vector_decode
        stroked_packed_diagnostics = recognition_report.stroked_vector_packed
        document_stroked_diagnostics = recognition_report.document_stroked_glyphs
        ocr_raster_pixels = sum(
            int(internal_report_number(diagnostic, "raster_pixels"))
            for diagnostic in ocr_diagnostics
        )
        ocr_full_page_fallback = int(
            any(
                bool(diagnostic.get("full_page_fallback"))
                for diagnostic in ocr_diagnostics
                if isinstance(diagnostic, dict)
            )
        )
        layout_strategy = self.internal_layout[1] if self.internal_layout is not None else "xy-cut"
        order_evidence = self.internal_layout[2] if self.internal_layout is not None else None
        image_cache = getattr(self.page.document, "image_cache", None)
        image_cache_stats = image_cache.stats() if image_cache is not None else None
        decoder_cache = getattr(self.page.document, "decoder_cache", {})
        decoders = decoder_cache.values() if isinstance(decoder_cache, dict) else ()
        # One pass over the decoders for all five Type3 counters.  This runs for every
        # page, so five separate generator passes over the same values were five times
        # the iteration and attribute lookups for the same result.
        type3_cache_hits = 0
        type3_cache_misses = 0
        type3_compiled_programs = 0
        type3_compiled_operations = 0
        type3_unsafe_fallbacks = 0
        for decoder in decoders:
            type3_cache_hits += int(getattr(decoder, "type3_charproc_cache_hits", 0))
            type3_cache_misses += int(getattr(decoder, "type3_charproc_cache_misses", 0))
            type3_compiled_programs += int(getattr(decoder, "type3_charproc_compiled_programs", 0))
            type3_compiled_operations += int(
                getattr(decoder, "type3_charproc_compiled_operations", 0)
            )
            type3_unsafe_fallbacks += int(getattr(decoder, "type3_charproc_unsafe_fallbacks", 0))
        metrics: dict[str, float | int | str | bool] = {
            "route": plan.route.value,
            "page_program_seconds": (self.internal_captured_at or self.started) - self.started,
            "content_stream_passes": 1,
            "capture_product_count": 1,
            "capture_seconds": (self.internal_captured_at or self.started) - self.started,
            "planning_seconds": (self.internal_planned_at or self.started)
            - (self.internal_captured_at or self.started),
            "ocr_seconds": (self.internal_recognized_at or self.started)
            - (self.internal_planned_at or self.started),
            "fusion_seconds": (self.internal_fused_at or self.started)
            - (self.internal_recognized_at or self.started),
            "table_seconds": (self.internal_tabled_at or self.started)
            - (self.internal_fused_at or self.started),
            "layout_seconds": finished - (self.internal_tabled_at or self.started),
            "native_observations": len(capture.observations),
            "ocr_observations": len(ocr),
            "ocr_raster_pixels": ocr_raster_pixels,
            "ocr_full_page_fallback": ocr_full_page_fallback,
            "image_cache_hits": image_cache_stats.hits if image_cache_stats else 0,
            "image_cache_misses": image_cache_stats.misses if image_cache_stats else 0,
            "image_cache_evictions": image_cache_stats.evictions if image_cache_stats else 0,
            "image_cache_bytes": image_cache_stats.bytes if image_cache_stats else 0,
            "image_cache_peak_bytes": image_cache_stats.peak_bytes if image_cache_stats else 0,
            "type3_charproc_cache_hits": type3_cache_hits,
            "type3_charproc_cache_misses": type3_cache_misses,
            "type3_charproc_compiled_programs": type3_compiled_programs,
            "type3_charproc_compiled_operations": type3_compiled_operations,
            "type3_charproc_unsafe_fallbacks": type3_unsafe_fallbacks,
            "fused_observations": len(observations),
            "layout_strategy": layout_strategy,
            "reading_order_strategy": (
                order_evidence.strategy if order_evidence is not None else "source-stable"
            ),
            "reading_order_repaired": int(
                order_evidence.repaired if order_evidence is not None else False
            ),
            "reading_order_ambiguous": int(
                order_evidence.ambiguous if order_evidence is not None else False
            ),
            "reading_order_confidence": (
                order_evidence.confidence if order_evidence is not None else 1.0
            ),
            "reading_order_source_inversions": (
                order_evidence.source_inversions if order_evidence is not None else 0
            ),
            "reading_order_source_inversion_ratio": (
                order_evidence.source_inversion_ratio if order_evidence is not None else 0.0
            ),
            "reading_order_columns": (
                order_evidence.column_count if order_evidence is not None else 0
            ),
            "reading_order_rotations": (
                order_evidence.rotation_count if order_evidence is not None else 0
            ),
            "text_coverage": capture.evidence.text_coverage,
            "painted_text_coverage": capture.evidence.painted_text_coverage or 0.0,
            "glyph_mapped_ratio": capture.evidence.glyphs.mapped_ratio,
            "glyph_unknown_ratio": capture.evidence.glyphs.unknown_ratio,
            "trusted_hidden_text": int(capture.evidence.trusted_hidden_text),
            "vector_text_characters": capture.evidence.vector_text_characters,
            "vector_text_candidate_segments": (capture.evidence.vector_text_candidate_segments),
            "vector_text_matched_segments": capture.evidence.vector_text_matched_segments,
            "vector_text_segment_coverage": (capture.evidence.vector_text_segment_coverage),
            "vector_text_sequences": capture.evidence.vector_text_sequences,
            "vector_text_maximum_error": capture.evidence.vector_text_maximum_error,
            "vector_text_seconds": float(
                internal_report_number(newstroke_diagnostics, "seconds", 0.0)
            ),
            "vector_text_trusted": int(capture.evidence.vector_text_trusted),
            "stroked_vector_text_trusted": int(capture.evidence.stroked_vector_text.trusted),
            "stroked_vector_candidate_paths": (
                capture.evidence.stroked_vector_text.candidate_paths
            ),
            "stroked_vector_packed_cells": int(
                internal_report_number(stroked_packed_diagnostics, "cells")
            ),
            "stroked_vector_packed_fallback": int(
                bool(stroked_packed_diagnostics.get("fallback_used", False))
            ),
            "stroked_vector_document_reuse": int(
                document_stroked_diagnostics.get("role") == "reuse"
            ),
            "stroked_vector_document_alphabet": int(
                internal_report_number(document_stroked_diagnostics, "alphabet_size")
            ),
            "stroked_vector_decode_seconds": float(
                internal_report_number(stroked_decode_diagnostics, "seconds", 0.0)
            ),
            "stroked_vector_decoded_runs": int(
                internal_report_number(stroked_decode_diagnostics, "decoded_runs")
            ),
            "stroked_vector_decode_additions": int(
                internal_report_number(stroked_decode_diagnostics, "additions")
            ),
            "stroked_vector_decode_corrections": int(
                internal_report_number(stroked_decode_diagnostics, "corrections")
            ),
            "stroked_vector_approximate_signatures": int(
                internal_report_number(stroked_decode_diagnostics, "approximate_signatures")
            ),
            "verified_hidden_text": int(
                bool(recognition_report.hidden_text_verification.get("accepted", False))
            ),
            "full_page_image": capture.evidence.full_page_image,
            "uncovered_vector_area": capture.evidence.uncovered_vector_area or 0.0,
        }
        report = ParseReport(plan=plan, recognition=recognition_report, metrics=metrics)
        parsed = ParsedPage(
            page_number=int(self.page.page_number),
            width=float(self.page.width),
            height=float(self.page.height),
            rotation=int(self.page.rotation),
            route=plan.route,
            blocks=blocks,
            tables=tables,
            figures=figures,
            diagnostics=(
                ("reading-order-ambiguous",)
                if order_evidence is not None and order_evidence.ambiguous
                else ()
            ),
            full_page_image=capture.evidence.full_page_image,
            report=report,
        )
        cache[PARSED_PAGE_CACHE_KEY] = parsed
        cache[PARSE_REPORT_CACHE_KEY] = report
        return parsed

    def assembled_page(self, context: TaskScope) -> Any:
        cache = self.page.extraction_cache
        assembled = cache.get(ASSEMBLED_PAGE_CACHE_KEY)
        if assembled is None:
            assembled = assemble_page(self.parsed_page(context), self.capture().drawings)
            try:
                resolver = self.page.document.resolver
                annotations = tuple(
                    Annotation(
                        subtype=record.subtype,
                        bbox=record.rect,
                        contents=record.contents,
                        destination=resolve_destination_value(
                            resolver, record.dest or record.action
                        ),
                    )
                    for record in self.page.get_annotations()
                )
                links = tuple(
                    Link(
                        bbox=record.bbox,
                        url=record.url,
                        link_type=record.link_type,
                        text="",
                    )
                    for record in self.page.get_links()
                )
                fields = tuple(
                    FormField(
                        name=record.name,
                        field_type=record.type,
                        value_text=record.value_text,
                        bbox=record.rect,
                        field_index=index,
                        required=record.is_required,
                        read_only=record.is_read_only,
                        no_export=record.no_export,
                        options=record.options,
                    )
                    for index, record in enumerate(self.page.get_fields())
                )
                cropbox = self.page.crop_box
                assembled = replace(
                    assembled,
                    annotations=annotations,
                    links=links,
                    form_fields=fields,
                    cropbox=cropbox,
                )
            except (TypeError, ValueError):
                # Malformed optional interactive objects must not block extraction.
                pass
            cache[ASSEMBLED_PAGE_CACHE_KEY] = assembled
        return assembled


def page_extraction(page: Any) -> internal_PageExtraction:
    cache = page.extraction_cache
    with page.internal_page_lock:
        extraction = cache.get(PAGE_EXTRACTION_CACHE_KEY)
        if not isinstance(extraction, internal_PageExtraction):
            extraction = internal_PageExtraction(page)
            cache[PAGE_EXTRACTION_CACHE_KEY] = extraction
        return extraction


def parse_page(page: Any, context: TaskScope) -> ParsedPage:
    with page.internal_page_lock:
        return page_extraction(page).parsed_page(context)


def extract_page(page: Any, context: TaskScope) -> Any:
    """Return the canonical emitted page, parsing and emitting at most once."""
    with page.internal_page_lock:
        return page_extraction(page).assembled_page(context)


DOCUMENT_FONT_SEED_LIMIT = 4
DOCUMENT_FONT_SEEDS_PER_DECODER = 2
DOCUMENT_STROKED_MIN_DECODED_RUNS = 20
DOCUMENT_STROKED_MIN_RUN_COVERAGE = 0.70
DOCUMENT_STROKED_MIN_GLYPH_COVERAGE = 0.70


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
            or internal_learned_glyph_text(glyph) is not None
            or not callable(getattr(decoder, "install_learned_unicode", None))
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
    for text, bbox, confidence in zip(ocr.text, ocr.bbox, ocr.confidence, strict=True):
        if not math.isfinite(float(confidence)) or float(confidence) < 90.0:
            continue
        characters = tuple(character for character in text if not character.isspace())
        if len(characters) < 3:
            continue
        x0, y0, x1, y1 = internal_bbox_tuple(bbox)
        tolerance = max(1.0, (y1 - y0) * 0.10)
        aligned = tuple(
            sorted(
                (
                    glyph
                    for glyph in glyphs
                    if x0 - tolerance
                    <= (glyph.ink_bbox[0] + glyph.ink_bbox[2]) * 0.5
                    <= x1 + tolerance
                    and y0 - tolerance
                    <= (glyph.ink_bbox[1] + glyph.ink_bbox[3]) * 0.5
                    <= y1 + tolerance
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


def internal_install_document_font_mappings(
    votes: dict[object, dict[bytes, Counter[str]]],
) -> tuple[frozenset[object], int]:
    installed_decoders: set[object] = set()
    installed_characters = 0
    for decoder, by_code in votes.items():
        mapping: dict[bytes, str] = {}
        for code_bytes, counts in by_code.items():
            if not counts:
                continue
            character, count = counts.most_common(1)[0]
            total = counts.total()
            if count >= 2 and count / total >= 0.90:
                mapping[code_bytes] = character
        installer = getattr(decoder, "install_learned_unicode", None)
        if not mapping or not callable(installer):
            continue
        additions = int(installer(mapping))
        if additions:
            installed_decoders.add(decoder)
            installed_characters += additions
    return frozenset(installed_decoders), installed_characters


def internal_refresh_learned_capture(page: Any, decoders: frozenset[object]) -> None:
    extraction = page_extraction(page)
    capture = extraction.internal_capture
    if capture is None or not any(
        glyph.font_decoder in decoders for glyph in capture.program.products.glyphs
    ):
        return
    cache = page.extraction_cache
    cache.pop(CAPTURED_PAGE_CACHE_KEY, None)
    extraction.internal_capture = internal_capture_from_program(page, capture.program)
    extraction.internal_captured_at = time.perf_counter()


def internal_prepare_document_font_mappings(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
) -> tuple[int, int]:
    seed_indexes = internal_document_font_seed_indexes(captures)
    if not seed_indexes:
        return 0, 0
    ocr_by_index: dict[int, ObservationBatch] = {}
    for completed in context.map_completed(
        lambda page_index: page_extraction(pages[page_index]).ocr(context),
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
    installed_decoders, installed_characters = internal_install_document_font_mappings(votes)
    if not installed_decoders:
        return len(seed_indexes), 0
    seed_set = frozenset(seed_indexes)
    for page_index, page in enumerate(pages):
        if page_index not in seed_set:
            internal_refresh_learned_capture(page, installed_decoders)
    return len(seed_indexes), installed_characters


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


def internal_install_document_stroked_decode(
    page: Any,
    decoded: StrokedTextDecode,
    *,
    seconds: float,
    seed_pages: tuple[int, ...],
    alphabet_size: int,
) -> bool:
    """Install deterministic cross-page vector text as this page's zero-raster OCR result."""
    with page.internal_page_lock:
        extraction = page_extraction(page)
        if extraction.internal_recognition is not None:
            return False
        internal_install_document_stroked_decode_locked(
            extraction,
            decoded,
            seconds=seconds,
            seed_pages=seed_pages,
            alphabet_size=alphabet_size,
        )
    return True


def internal_install_document_stroked_decode_locked(
    extraction: internal_PageExtraction,
    decoded: StrokedTextDecode,
    *,
    seconds: float,
    seed_pages: tuple[int, ...],
    alphabet_size: int,
) -> None:
    from core_pdf.impl.engine.parse.ocr_stroked_vector import internal_stroked_vector_decoded_batch

    observations = internal_stroked_vector_decoded_batch(decoded.observations)
    candidate = internal_candidate(-1, observations)
    bbox = extraction.capture().evidence.stroked_vector_text.bbox
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
    extraction.internal_recognition = RecognitionResult(
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
    extraction.internal_recognized_at = time.perf_counter()


def internal_prepare_document_stroked_mappings(
    pages: tuple[Any, ...],
    captures: tuple[CapturedPage, ...],
    context: TaskScope,
) -> tuple[int, int]:
    """OCR the richest flattened-font page, then decode compatible pages structurally."""
    indexes = tuple(
        index
        for index, capture in enumerate(captures)
        if capture.evidence.stroked_vector_text.trusted
    )
    if len(indexes) < 2:
        return 0, 0
    from core_pdf.impl.engine.parse.ocr_stroked_vector import internal_stroked_text_profile

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
    for page_index in ordered:
        page = pages[page_index]
        extraction = page_extraction(page)
        capture = extraction.capture()
        if extraction.internal_recognition is None and alphabet:
            with page.internal_page_lock:
                extraction.plan()
            started = time.perf_counter()
            decoded = decode_stroked_text_profile_with_alphabet(
                internal_stroked_text_profile(capture),
                alphabet,
            )
            seconds = time.perf_counter() - started
            if internal_document_stroked_decode_is_sufficient(
                decoded
            ) and internal_install_document_stroked_decode(
                page,
                decoded,
                seconds=seconds,
                seed_pages=tuple(int(pages[index].page_number) for index in seed_indexes),
                alphabet_size=len(alphabet),
            ):
                reused_pages += 1
                continue

        if extraction.internal_recognition is None:
            with page.internal_page_lock:
                if extraction.internal_recognition is None:
                    extraction.ocr(context)
        recognition = extraction.internal_recognition
        learned = recognition.report.stroked_vector_alphabet if recognition is not None else ()
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
        if recognition is not None:
            extraction.internal_recognition = replace(
                recognition,
                report=replace(recognition.report, document_stroked_glyphs=seed_report),
            )
    return len(seed_indexes), reused_pages


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
    pages: tuple[Any, ...],
    context: TaskScope,
) -> tuple[ParsedPage, ...]:
    parsed_by_index: list[ParsedPage | None] = [None] * len(pages)
    futures: dict[int, Future[ParsedPage]] = {}
    direct_indexes: list[int] = []
    for index, page in enumerate(pages):
        extraction = page_extraction(page)
        plan = extraction.plan()
        requires_ocr = extraction.internal_recognition is None and (
            bool(plan.ocr_passes) or plan.verify_hidden_text
        )
        if requires_ocr:
            futures[index] = context.submit(parse_page, page, context, stage=WorkStage.PAGE)
        else:
            direct_indexes.append(index)
    try:
        for index in direct_indexes:
            context.raise_if_cancelled()
            parsed_by_index[index] = parse_page(pages[index], context)
        for index, future in futures.items():
            parsed_by_index[index] = future.result()
    finally:
        for future in futures.values():
            future.cancel()
    return tuple(page for page in parsed_by_index if page is not None)


def parse_document(
    document: Any,
    context: TaskScope,
    pages: Sequence[Any],
) -> Document:
    pages = tuple(pages)
    parsed_pages: tuple[ParsedPage, ...]
    if len(pages) == 1:
        parsed_pages = (parse_page(pages[0], context),)
    else:
        captures = internal_capture_document_pages(pages, context)
        if len(captures) == len(pages):
            internal_prepare_document_font_mappings(pages, captures, context)
            internal_prepare_document_stroked_mappings(pages, captures, context)
        parsed_pages = internal_parse_document_pages(pages, context)
    diagnostics = tuple(
        Diagnostic("parse", message, page_number=page.page_number)
        for page in parsed_pages
        for message in page.diagnostics
    )
    metadata = document.get_metadata()
    return Document(
        pages=tuple(page_extraction(source_page).assembled_page(context) for source_page in pages),
        metadata=metadata,
        diagnostics=diagnostics,
        schema_version=SCHEMA_VERSION,
    )
