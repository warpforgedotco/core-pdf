# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestration and per-page caching."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any, TypeVar

from core_pdf.impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl.extract.capture import capture_page
from core_pdf.impl.extract.contracts import (
    CapturedPage,
    ObservationBatch,
    ParsedBlock,
    ParsedPage,
    ParseReport,
    ReadingOrderEvidence,
    RecognitionReport,
    RecognitionResult,
    WorkPlan,
)
from core_pdf.impl.extract.emit import (
    assemble_page,
)
from core_pdf.impl.extract.observations import (
    fuse_observations,
    plan_page,
)
from core_pdf.impl.extract.tables import (
    extract_tables,
)
from core_pdf.impl.output import (
    Annotation,
    Figure,
    FormField,
    Link,
    Table,
)
from core_pdf.impl.runtime.execution import TaskScope
from core_pdf.impl.spec.s_07_document.page_links import resolve_destination_value

PAGE_EXTRACTION_CACHE_KEY = "page_extraction_v3"

internal_T = TypeVar("internal_T")


def internal_collected_records(
    fetch: Callable[[], Iterable[Any]],
    build: Callable[[int, Any], internal_T],
) -> tuple[internal_T, ...]:
    """Fetch page records and build one product per record, skipping bad entries."""
    records: Iterable[Any]
    try:
        records = fetch()
    except (TypeError, ValueError):
        records = ()
    output: list[internal_T] = []
    for index, record in enumerate(records):
        try:
            output.append(build(index, record))
        except (TypeError, ValueError):
            continue
    return tuple(output)


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

    def __init__(
        self,
        page: Any,
        *,
        capture: CapturedPage | None = None,
        plan: WorkPlan | None = None,
        recognition: RecognitionResult | None = None,
        capture_seconds: float = 0.0,
        planning_seconds: float = 0.0,
        ocr_seconds: float = 0.0,
    ) -> None:
        self.page = page
        self.internal_capture = capture
        self.internal_plan = plan
        self.internal_recognition = recognition
        self.internal_observations: ObservationBatch | None = None
        self.internal_tables: tuple[Table, ...] | None = None
        self.internal_layout: tuple[tuple[ParsedBlock, ...], str, ReadingOrderEvidence] | None = (
            None
        )
        self.internal_parsed_page: ParsedPage | None = None
        self.internal_parse_report: ParseReport | None = None
        self.internal_assembled_page: Any | None = None
        self.internal_capture_seconds = capture_seconds
        self.internal_planning_seconds = planning_seconds
        self.internal_ocr_seconds = ocr_seconds
        self.internal_fusion_seconds = 0.0
        self.internal_table_seconds = 0.0
        self.internal_layout_seconds = 0.0

    def internal_invalidate_after_capture(self) -> None:
        """Drop every product downstream of the capture, in pipeline order."""
        self.internal_plan = None
        self.internal_planning_seconds = 0.0
        self.internal_recognition = None
        self.internal_ocr_seconds = 0.0
        self.internal_observations = None
        self.internal_fusion_seconds = 0.0
        self.internal_tables = None
        self.internal_table_seconds = 0.0
        self.internal_layout = None
        self.internal_layout_seconds = 0.0
        self.internal_parsed_page = None
        self.internal_parse_report = None
        self.internal_assembled_page = None

    def replace_capture(self, capture: CapturedPage, *, seconds: float = 0.0) -> None:
        """Install a capture and atomically invalidate every dependent product."""
        with self.page.internal_page_lock:
            self.internal_capture = capture
            self.internal_capture_seconds = seconds
            self.internal_invalidate_after_capture()

    @property
    def report(self) -> ParseReport | None:
        with self.page.internal_page_lock:
            return self.internal_parse_report

    def capture(self) -> CapturedPage:
        with self.page.internal_page_lock:
            if self.internal_capture is not None:
                return self.internal_capture
            started = time.perf_counter()
            capture = capture_page(self.page)
            self.internal_capture_seconds = time.perf_counter() - started
            self.internal_capture = capture
            return capture

    def plan(self) -> WorkPlan:
        with self.page.internal_page_lock:
            if self.internal_plan is not None:
                return self.internal_plan
            started = time.perf_counter()
            plan = plan_page(self.capture())
            self.internal_planning_seconds = time.perf_counter() - started
            self.internal_plan = plan
            return plan

    def recognition(self, context: TaskScope) -> RecognitionResult:
        with self.page.internal_page_lock:
            if self.internal_recognition is not None:
                return self.internal_recognition
            started = time.perf_counter()
            plan = self.plan()
            if plan.ocr_passes:
                from core_pdf.impl.extract.ocr.pipeline import recognize_page

                recognition = recognize_page(self.capture(), plan, context)
            else:
                # recognize_page() returns exactly this for a plan with no OCR
                # passes. Short-circuiting keeps extract.ocr — and with it
                # tesserocr, PIL and the rasterizer — off the native-text path.
                recognition = RecognitionResult(ObservationBatch.empty(), RecognitionReport())
            self.internal_ocr_seconds = time.perf_counter() - started
            self.internal_recognition = recognition
            return recognition

    def observations(self, context: TaskScope) -> ObservationBatch:
        with self.page.internal_page_lock:
            if self.internal_observations is not None:
                return self.internal_observations
            started = time.perf_counter()
            capture = self.capture()
            observations = fuse_observations(
                capture.observations,
                self.recognition(context).observations,
                self.plan(),
            )
            self.internal_fusion_seconds = time.perf_counter() - started
            self.internal_observations = observations
            return observations

    def tables(self, context: TaskScope) -> tuple[Table, ...]:
        with self.page.internal_page_lock:
            if self.internal_tables is not None:
                return self.internal_tables
            started = time.perf_counter()
            tables = extract_tables(self.capture(), self.observations(context))
            self.internal_table_seconds = time.perf_counter() - started
            self.internal_tables = tables
            return tables

    def internal_image_obstacles(self) -> tuple[tuple[float, float, float, float], ...]:
        with self.page.internal_page_lock:
            capture = self.capture()
            return tuple(
                box
                for box in capture.evidence.image_boxes
                if 0.01
                <= ((box[2] - box[0]) * (box[3] - box[1])) / capture.evidence.page_area
                < 0.65
            )

    def layout(self, context: TaskScope) -> tuple[ParsedBlock, ...]:
        return self.internal_layout_result(context)[0]

    def internal_layout_result(
        self, context: TaskScope
    ) -> tuple[tuple[ParsedBlock, ...], str, ReadingOrderEvidence]:
        with self.page.internal_page_lock:
            if self.internal_layout is not None:
                return self.internal_layout
            started = time.perf_counter()
            observations = self.observations(context)
            capture = self.capture()
            table_obstacles = tuple(
                table.bbox for table in self.tables(context) if table.bbox is not None
            )
            use_xy_cut = not (
                capture.evidence.image_count >= 8
                and 0.05 <= capture.evidence.image_area_ratio < 0.65
            )
            blocks, order_evidence = layout_blocks_with_evidence(
                observations,
                obstacles=(*table_obstacles, *self.internal_image_obstacles()),
                use_xy_cut=use_xy_cut,
                rotation=int(getattr(self.page, "rotation", 0) or 0),
                page_width=float(capture.page.width),
                page_height=float(capture.page.height),
            )
            self.internal_layout_seconds = time.perf_counter() - started
            result = (blocks, "xy-cut" if use_xy_cut else "row-order", order_evidence)
            self.internal_layout = result
            return result

    def parsed_page(self, context: TaskScope) -> ParsedPage:
        with self.page.internal_page_lock:
            if self.internal_parsed_page is not None:
                return self.internal_parsed_page
            return self.internal_build_parsed_page(context)

    def internal_build_parsed_page(self, context: TaskScope) -> ParsedPage:
        capture = self.capture()
        plan = self.plan()
        recognition = self.recognition(context)
        ocr = recognition.observations
        recognition_report = recognition.report
        observations = self.observations(context)
        tables = self.tables(context)
        blocks, layout_strategy, order_evidence = self.internal_layout_result(context)
        figures = (
            ()
            if capture.evidence.full_page_image
            else tuple(
                Figure(order=index, bbox=box, kind="image", metadata={"source": "capture"})
                for index, box in enumerate(capture.evidence.image_boxes)
            )
        )
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
            "page_program_seconds": self.internal_capture_seconds,
            "content_stream_passes": 1,
            "capture_product_count": 1,
            "capture_seconds": self.internal_capture_seconds,
            "planning_seconds": self.internal_planning_seconds,
            "ocr_seconds": self.internal_ocr_seconds,
            "fusion_seconds": self.internal_fusion_seconds,
            "table_seconds": self.internal_table_seconds,
            "layout_seconds": self.internal_layout_seconds,
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
            "reading_order_strategy": order_evidence.strategy,
            "reading_order_repaired": int(order_evidence.repaired),
            "reading_order_ambiguous": int(order_evidence.ambiguous),
            "reading_order_confidence": order_evidence.confidence,
            "reading_order_source_inversions": order_evidence.source_inversions,
            "reading_order_source_inversion_ratio": order_evidence.source_inversion_ratio,
            "reading_order_columns": order_evidence.column_count,
            "reading_order_rotations": order_evidence.rotation_count,
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
            diagnostics=(("reading-order-ambiguous",) if order_evidence.ambiguous else ()),
            full_page_image=capture.evidence.full_page_image,
            report=report,
        )
        self.internal_parse_report = report
        self.internal_parsed_page = parsed
        return parsed

    def assembled_page(self, context: TaskScope) -> Any:
        with self.page.internal_page_lock:
            if self.internal_assembled_page is not None:
                return self.internal_assembled_page
            assembled = assemble_page(self.parsed_page(context), self.capture().drawings)
            resolver = self.page.document.resolver
            annotations = internal_collected_records(
                self.page.get_annotations,
                lambda _index, record: Annotation(
                    subtype=record.subtype,
                    bbox=record.rect,
                    contents=record.contents,
                    destination=resolve_destination_value(resolver, record.dest or record.action),
                ),
            )
            links = internal_collected_records(
                self.page.get_links,
                lambda _index, record: Link(
                    bbox=record.bbox,
                    url=record.url,
                    link_type=record.link_type,
                    text="",
                ),
            )
            fields = internal_collected_records(
                self.page.get_fields,
                lambda index, record: FormField(
                    name=record.name,
                    field_type=record.type,
                    value_text=record.value_text,
                    bbox=record.rect,
                    field_index=index,
                    required=record.is_required,
                    read_only=record.is_read_only,
                    no_export=record.no_export,
                    options=record.options,
                ),
            )
            cropbox = assembled.cropbox
            with suppress(TypeError, ValueError):
                cropbox = self.page.crop_box
            assembled = replace(
                assembled,
                annotations=annotations,
                links=links,
                form_fields=fields,
                cropbox=cropbox,
            )
            self.internal_assembled_page = assembled
            return assembled


def page_extraction(page: Any) -> internal_PageExtraction:
    cache = page.extraction_cache
    with page.internal_page_lock:
        extraction = cache.get(PAGE_EXTRACTION_CACHE_KEY)
        if not isinstance(extraction, internal_PageExtraction):
            extraction = internal_PageExtraction(page)
            cache[PAGE_EXTRACTION_CACHE_KEY] = extraction
        return extraction


def extract_page(page: Any, context: TaskScope) -> Any:
    """Return the canonical emitted page, parsing and emitting at most once."""
    return page_extraction(page).assembled_page(context)
