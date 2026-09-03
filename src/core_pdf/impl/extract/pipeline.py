# SPDX-License-Identifier: AGPL-3.0-only
"""Page extraction orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
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
from core_pdf.impl.extract.table_pipeline import (
    extract_tables,
)
from core_pdf.impl.output import (
    Annotation,
    Figure,
    FormField,
    Link,
    Table,
)
from core_pdf.impl.runtime.execution import ExtractionScope
from core_pdf.impl.spec.s_07_document.page_links import resolve_destination_value

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


@dataclass(frozen=True, slots=True)
class internal_PageProducts:
    observations: ObservationBatch
    tables: tuple[Table, ...]
    blocks: tuple[ParsedBlock, ...]
    order_evidence: ReadingOrderEvidence


class internal_PageExtraction:
    """Explicit inputs for deriving one page's extraction products."""

    def __init__(
        self,
        page: Any,
        *,
        capture: CapturedPage | None = None,
        plan: WorkPlan | None = None,
        recognition: RecognitionResult | None = None,
    ) -> None:
        self.page = page
        self.internal_capture = capture if capture is not None else capture_page(page)
        self.internal_plan = plan if plan is not None else plan_page(self.internal_capture)
        self.internal_recognition = recognition

    def capture(self) -> CapturedPage:
        return self.internal_capture

    def plan(self) -> WorkPlan:
        return self.internal_plan

    def recognition(self, context: ExtractionScope) -> RecognitionResult:
        if self.internal_recognition is not None:
            return self.internal_recognition
        plan = self.internal_plan
        if plan.ocr_passes or plan.verify_hidden_text:
            from core_pdf.impl.extract.ocr.pipeline import recognize_page

            return recognize_page(self.internal_capture, plan, context)
        return RecognitionResult(ObservationBatch.empty())

    def internal_products(self, context: ExtractionScope) -> internal_PageProducts:
        capture = self.internal_capture
        plan = self.internal_plan
        observations = fuse_observations(
            capture.observations,
            self.recognition(context).observations,
            plan,
        )
        tables = extract_tables(capture, observations)
        table_obstacles = tuple(table.bbox for table in tables if table.bbox is not None)
        image_obstacles = tuple(
            box
            for box in capture.evidence.image_boxes
            if 0.01 <= ((box[2] - box[0]) * (box[3] - box[1])) / capture.evidence.page_area < 0.65
        )
        use_xy_cut = not (
            capture.evidence.image_count >= 8 and 0.05 <= capture.evidence.image_area_ratio < 0.65
        )
        blocks, order_evidence = layout_blocks_with_evidence(
            observations,
            obstacles=(*table_obstacles, *image_obstacles),
            use_xy_cut=use_xy_cut,
            rotation=int(getattr(self.page, "rotation", 0) or 0),
            page_width=float(capture.page.width),
            page_height=float(capture.page.height),
        )
        return internal_PageProducts(observations, tables, blocks, order_evidence)

    def observations(self, context: ExtractionScope) -> ObservationBatch:
        return self.internal_products(context).observations

    def tables(self, context: ExtractionScope) -> tuple[Table, ...]:
        return self.internal_products(context).tables

    def layout(self, context: ExtractionScope) -> tuple[ParsedBlock, ...]:
        return self.internal_products(context).blocks

    def internal_layout_result(
        self, context: ExtractionScope
    ) -> tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence]:
        products = self.internal_products(context)
        return products.blocks, products.order_evidence

    def assembled_page(self, context: ExtractionScope) -> Any:
        capture = self.internal_capture
        products = self.internal_products(context)
        blocks = products.blocks
        order_evidence = products.order_evidence
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
            route=self.internal_plan.route,
            tables=products.tables,
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
        return assembled


def extract_page(page: Any, context: ExtractionScope) -> Any:
    """Extract and emit one page."""
    return internal_PageExtraction(page).assembled_page(context)
