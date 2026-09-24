# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from copy import replace
from typing import TYPE_CHECKING, ClassVar, Protocol

from core_pdf.impl.document.page_links import resolve_destination_value
from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract.block_layout import layout_blocks_with_evidence
from core_pdf.impl.extract.capture import STRUCTURE_UNSET, capture_page
from core_pdf.impl.extract.contracts import (
    ObservationBatch,
    PageAnalysis,
    ParsedBlock,
    ReadingOrderEvidence,
)
from core_pdf.impl.extract.emit import (
    assemble_page,
)
from core_pdf.impl.extract.table_detection import extract_tables
from core_pdf.impl.output.model import (
    Annotation,
    Figure,
    FormField,
    Link,
    Page,
    Table,
)
from core_pdf.impl.types import Record, frozen_setattr

if TYPE_CHECKING:
    from core_pdf.impl.document.page import PdfPage
    from core_pdf.impl.document.records import RawAnnotation, RawFormField
    from core_pdf.impl.document.structure import PageStructure
    from core_pdf.impl.extract.capture import StructureUnset


class Layout(Protocol):
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


def collected_records[Record, T](
    fetch: Callable[[], Iterable[Record]],
    build: Callable[[int, Record], T],
) -> tuple[T, ...]:
    records: Iterable[Record]
    try:
        records = fetch()
    except TypeError, ValueError:
        records = ()
    output: list[T] = []
    for index, record in enumerate(records):
        try:
            output.append(build(index, record))
        except TypeError, ValueError:
            continue
    return tuple(output)


class PageProducts(Record):
    __slots__ = ("tables", "blocks", "order_evidence")

    tables: tuple[Table, ...]
    blocks: tuple[ParsedBlock, ...]
    order_evidence: ReadingOrderEvidence

    __fields__: ClassVar[tuple[str, ...]] = ("tables", "blocks", "order_evidence")
    __match_args__ = ("tables", "blocks", "order_evidence")

    def __init__(
        self,
        tables: tuple[Table, ...],
        blocks: tuple[ParsedBlock, ...],
        order_evidence: ReadingOrderEvidence,
    ) -> None:
        frozen_setattr(self, "tables", tables)
        frozen_setattr(self, "blocks", blocks)
        frozen_setattr(self, "order_evidence", order_evidence)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.tables == other.tables
            and self.blocks == other.blocks
            and self.order_evidence == other.order_evidence
        )

    def __hash__(self) -> int:
        return hash((self.tables, self.blocks, self.order_evidence))


class PageExtraction:
    capture_page_fn = staticmethod(capture_page)

    @property
    def route_name(self) -> str:
        return "native"

    @property
    def capture(self) -> PageAnalysis:
        return self.page_capture

    def __init__(
        self,
        page: PdfPage,
        *,
        capture: PageAnalysis | None = None,
        fields: Iterable[RawFormField] | None = None,
        structure: PageStructure | None | StructureUnset = STRUCTURE_UNSET,
        hidden_layers: frozenset[str] | None = None,
    ) -> None:
        self.page = page
        self.structure_value = structure
        self.hidden_layer_names = hidden_layers
        field_records = tuple(fields) if fields is not None else None
        if capture is not None:
            self.page_capture = (
                replace(capture, fields=field_records) if field_records is not None else capture
            )
        else:
            annotation_records: tuple[RawAnnotation, ...] | None
            try:
                annotation_records = tuple(page.get_annotations()) or None
            except AttributeError, TypeError, ValueError:
                annotation_records = None
            self.page_capture = self.capture_page_fn(
                page,
                structure=structure,
                hidden_layers=hidden_layers,
                fields=field_records,
                annotations=annotation_records,
            )

    def run(self, context: ExtractionScope) -> PageProducts:
        context.raise_if_cancelled()
        observations = self.capture.observations
        return self.layout_products(
            observations,
            extract_tables(self.capture, observations),
        )

    def layout_products(
        self,
        observations: ObservationBatch,
        tables: tuple[Table, ...],
        *,
        layout: Layout = layout_blocks_with_evidence,
    ) -> PageProducts:
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
        return PageProducts(tables, blocks, order_evidence)

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
        assembled = assemble_page(
            blocks,
            page_number=int(self.page.page_number),
            width=capture.width,
            height=capture.height,
            rotation=capture.rotation,
            route=self.route_name,
            tables=products.tables,
            figures=figures,
            diagnostics=(("reading-order-ambiguous",) if order_evidence.ambiguous else ()),
            full_page_image=capture.evidence.full_page_image,
            drawings=capture.program.drawings,
        )
        resolver = self.page.document.resolver
        raw_annotations = capture.annotations or ()
        resolved_annotation_dicts = tuple(record.dict for record in raw_annotations)
        annotations = collected_records(
            lambda: raw_annotations,
            lambda _index, record: Annotation(
                subtype=record.subtype,
                bbox=record.rect,
                contents=record.contents,
                destination=resolve_destination_value(resolver, record.dest or record.action),
            ),
        )
        links = collected_records(
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
        field_records = collected_records(
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
        return replace(
            assembled,
            annotations=annotations,
            links=links,
            form_fields=field_records,
            cropbox=cropbox,
            user_unit=self.page.user_unit,
        )


def extract_page(page: PdfPage, context: ExtractionScope) -> Page:
    return PageExtraction(page).assembled_page(context)
