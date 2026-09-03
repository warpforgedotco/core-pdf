# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestration and per-page caching."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import replace
from typing import Any, TypeVar

from core_pdf.impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl.extract.capture import capture_page
from core_pdf.impl.extract.contracts import (
    CapturedPage,
    ObservationBatch,
    ParsedBlock,
    ReadingOrderEvidence,
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


class internal_PageExtraction:
    """Lazily materialized extraction products for one page."""

    def __init__(
        self,
        page: Any,
        *,
        capture: CapturedPage | None = None,
        plan: WorkPlan | None = None,
        recognition: RecognitionResult | None = None,
    ) -> None:
        self.page = page
        self.internal_capture = capture
        self.internal_plan = plan
        self.internal_recognition = recognition
        self.internal_observations: ObservationBatch | None = None
        self.internal_tables: tuple[Table, ...] | None = None
        self.internal_layout: tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence] | None = None
        self.internal_assembled_page: Any | None = None

    def internal_invalidate_after_capture(self) -> None:
        """Drop every product downstream of the capture, in pipeline order."""
        self.internal_plan = None
        self.internal_recognition = None
        self.internal_observations = None
        self.internal_tables = None
        self.internal_layout = None
        self.internal_assembled_page = None

    def replace_capture(self, capture: CapturedPage) -> None:
        """Install a capture and atomically invalidate every dependent product."""
        with self.page.internal_page_lock:
            self.internal_capture = capture
            self.internal_invalidate_after_capture()

    def capture(self) -> CapturedPage:
        with self.page.internal_page_lock:
            if self.internal_capture is not None:
                return self.internal_capture
            capture = capture_page(self.page)
            self.internal_capture = capture
            return capture

    def plan(self) -> WorkPlan:
        with self.page.internal_page_lock:
            if self.internal_plan is not None:
                return self.internal_plan
            plan = plan_page(self.capture())
            self.internal_plan = plan
            return plan

    def recognition(self, context: TaskScope) -> RecognitionResult:
        with self.page.internal_page_lock:
            if self.internal_recognition is not None:
                return self.internal_recognition
            plan = self.plan()
            if plan.ocr_passes:
                from core_pdf.impl.extract.ocr.pipeline import recognize_page

                recognition = recognize_page(self.capture(), plan, context)
            else:
                # recognize_page() returns exactly this for a plan with no OCR
                # passes. Short-circuiting keeps extract.ocr — and with it
                # tesserocr, PIL and the rasterizer — off the native-text path.
                recognition = RecognitionResult(ObservationBatch.empty())
            self.internal_recognition = recognition
            return recognition

    def observations(self, context: TaskScope) -> ObservationBatch:
        with self.page.internal_page_lock:
            if self.internal_observations is not None:
                return self.internal_observations
            capture = self.capture()
            observations = fuse_observations(
                capture.observations,
                self.recognition(context).observations,
                self.plan(),
            )
            self.internal_observations = observations
            return observations

    def tables(self, context: TaskScope) -> tuple[Table, ...]:
        with self.page.internal_page_lock:
            if self.internal_tables is not None:
                return self.internal_tables
            tables = extract_tables(self.capture(), self.observations(context))
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
    ) -> tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence]:
        with self.page.internal_page_lock:
            if self.internal_layout is not None:
                return self.internal_layout
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
            result = (blocks, order_evidence)
            self.internal_layout = result
            return result

    def assembled_page(self, context: TaskScope) -> Any:
        with self.page.internal_page_lock:
            if self.internal_assembled_page is not None:
                return self.internal_assembled_page
            capture = self.capture()
            blocks, order_evidence = self.internal_layout_result(context)
            figures = (
                ()
                if capture.evidence.full_page_image
                else tuple(
                    Figure(order=index, bbox=box, kind="image", metadata={"source": "capture"})
                    for index, box in enumerate(capture.evidence.image_boxes)
                )
            )
            assembled = assemble_page(
                blocks,
                page_number=int(self.page.page_number),
                width=float(self.page.width),
                height=float(self.page.height),
                rotation=int(self.page.rotation),
                route=self.plan().route,
                tables=self.tables(context),
                figures=figures,
                diagnostics=(("reading-order-ambiguous",) if order_evidence.ambiguous else ()),
                full_page_image=capture.evidence.full_page_image,
                drawings=capture.drawings,
            )
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
