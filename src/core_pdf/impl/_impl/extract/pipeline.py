# SPDX-License-Identifier: AGPL-3.0-only
"""Page extraction orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, TypeVar

from core_pdf.impl._impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl._impl.extract.capture import capture_page, internal_STRUCTURE_UNSET
from core_pdf.impl._impl.extract.contracts import (
    ObservationBatch,
    PageAnalysis,
    ParsedBlock,
    ReadingOrderEvidence,
)
from core_pdf.impl._impl.extract.emit import (
    assemble_page,
)
from core_pdf.impl._impl.extract.table_detection import extract_tables
from core_pdf.impl._impl.output.model import (
    Annotation,
    Figure,
    FormField,
    Link,
    Page,
    Table,
)
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf.impl.spec.s_07_document.page_links import resolve_destination_value

if TYPE_CHECKING:
    from core_pdf.impl._impl.extract.capture import internal_StructureUnset
    from core_pdf.impl.spec.s_07_document.page import PdfPage
    from core_pdf.impl.spec.s_07_document.records import RawAnnotation, RawFormField
    from core_pdf.impl.spec.s_14_structure.tree import PageStructure

internal_T = TypeVar("internal_T")
internal_Record = TypeVar("internal_Record")


class internal_Layout(Protocol):
    def __call__(
        self,
        observations: ObservationBatch,
        *,
        obstacles: tuple[tuple[float, float, float, float], ...],
        use_xy_cut: bool,
        rotation: int,
        page_width: float,
        page_height: float,
    ) -> tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence]: ...


def internal_collected_records(
    fetch: Callable[[], Iterable[internal_Record]],
    build: Callable[[int, internal_Record], internal_T],
) -> tuple[internal_T, ...]:
    """Fetch page records and build one product per record, skipping bad entries."""
    records: Iterable[internal_Record]
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
    tables: tuple[Table, ...]
    blocks: tuple[ParsedBlock, ...]
    order_evidence: ReadingOrderEvidence


class internal_PageExtraction:
    """Native extraction with reusable metadata and layout orchestration."""

    internal_capture_page = staticmethod(capture_page)
    internal_assemble_page = staticmethod(assemble_page)

    @property
    def internal_route(self) -> str:
        return "native"

    @property
    def capture(self) -> PageAnalysis:
        return self.internal_capture

    def __init__(
        self,
        page: PdfPage,
        *,
        capture: PageAnalysis | None = None,
        fields: Iterable[RawFormField] | None = None,
        structure: PageStructure | None | internal_StructureUnset = internal_STRUCTURE_UNSET,
        hidden_layers: frozenset[str] | None = None,
    ) -> None:
        self.page = page
        self.internal_structure = structure
        self.internal_hidden_layers = hidden_layers
        field_records = tuple(fields) if fields is not None else None
        if capture is not None:
            self.internal_capture = (
                replace(capture, fields=field_records) if field_records is not None else capture
            )
        else:
            annotation_records: tuple[RawAnnotation, ...] | None
            try:
                # An empty strict projection can still hide recoverable raw
                # annotations (for example a non-array /Annots in recovery mode).
                annotation_records = tuple(page.get_annotations()) or None
            except (AttributeError, TypeError, ValueError):
                # Failed strict metadata collection is not an explicit request
                # to suppress appearances. Let capture enumerate tolerant raw
                # annotation dictionaries; its output metadata remains empty.
                annotation_records = None
            self.internal_capture = self.internal_capture_page(
                page,
                structure=structure,
                hidden_layers=hidden_layers,
                fields=field_records,
                annotations=annotation_records,
            )

    def run(self, context: ExtractionScope) -> internal_PageProducts:
        """Derive tables and layout solely from embedded PDF text."""
        context.raise_if_cancelled()
        observations = self.capture.observations
        return self.internal_layout_products(
            observations,
            extract_tables(self.capture, observations),
        )

    def internal_layout_products(
        self,
        observations: ObservationBatch,
        tables: tuple[Table, ...],
        *,
        layout: internal_Layout = layout_blocks_with_evidence,
    ) -> internal_PageProducts:
        """Use shared page geometry to lay out an already chosen observation set."""
        capture = self.capture
        table_obstacles = tuple(table.bbox for table in tables if table.bbox is not None)
        image_obstacles = tuple(
            box
            for box in capture.evidence.image_boxes
            if 0.01 <= ((box[2] - box[0]) * (box[3] - box[1])) / capture.evidence.page_area < 0.65
        )
        use_xy_cut = not (
            capture.evidence.image_count >= 8 and 0.05 <= capture.evidence.image_area_ratio < 0.65
        )
        blocks, order_evidence = layout(
            observations,
            obstacles=(*table_obstacles, *image_obstacles),
            use_xy_cut=use_xy_cut,
            rotation=capture.rotation,
            page_width=capture.width,
            page_height=capture.height,
        )
        return internal_PageProducts(tables, blocks, order_evidence)

    def assembled_page(self, context: ExtractionScope) -> Page:
        capture = self.capture
        products = self.run(context)
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
        assembled = self.internal_assemble_page(
            blocks,
            page_number=int(self.page.page_number),
            width=capture.width,
            height=capture.height,
            rotation=capture.rotation,
            route=self.internal_route,
            tables=products.tables,
            figures=figures,
            diagnostics=(("reading-order-ambiguous",) if order_evidence.ambiguous else ()),
            full_page_image=capture.evidence.full_page_image,
            drawings=capture.program.drawings,
        )
        resolver = self.page.document.resolver
        raw_annotations = capture.annotations or ()
        resolved_annotation_dicts = tuple(record.dict for record in raw_annotations)
        annotations = internal_collected_records(
            lambda: raw_annotations,
            lambda _index, record: Annotation(
                subtype=record.subtype,
                bbox=record.rect,
                contents=record.contents,
                destination=resolve_destination_value(resolver, record.dest or record.action),
            ),
        )
        links = internal_collected_records(
            lambda: self.page.get_links(resolved_annotation_dicts),
            lambda _index, record: Link(
                bbox=record.bbox,
                url=record.url,
                link_type=record.link_type,
                text="",
            ),
        )
        source_fields = capture.fields
        fetch_fields = self.page.get_fields if source_fields is None else lambda: source_fields
        field_records = internal_collected_records(
            fetch_fields,
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
            form_fields=field_records,
            cropbox=cropbox,
        )
        return assembled


def extract_page(page: PdfPage, context: ExtractionScope) -> Page:
    """Extract and emit one page."""
    return internal_PageExtraction(page).assembled_page(context)
