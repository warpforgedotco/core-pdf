# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import suppress
from copy import replace
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from core_pdf.impl.base.exceptions import ExtractionScope
from core_pdf.impl.base.types import GeneratedRecord, Record
from core_pdf.impl.base.values import plain_pdf_value, resolve_destination_references
from core_pdf.impl.extract.block_layout import (
    NATIVE_LAYOUT_HOOKS,
    LayoutHooks,
    layout_blocks_with_evidence,
)
from core_pdf.impl.extract.capture import STRUCTURE_UNSET, capture_page
from core_pdf.impl.extract.contracts import PageAnalysis, PageFrame, PageState, ReadingOrderPolicy
from core_pdf.impl.extract.emit import assemble_page
from core_pdf.impl.extract.table_detection import NATIVE_TABLES, TableDetector
from core_pdf.impl.output.model import Annotation, Figure, FormField, Link, Page
from core_pdf.impl.spec.exceptions import PdfError
lazy from core_pdf.impl.spec.s_07_document.fields import VARIABLE_TEXT_FIELD_TYPES, field_quadding
lazy from core_pdf.impl.spec.s_07_document.markup_annotations import popup_link, reply_relation

if TYPE_CHECKING:
    from core_pdf.impl.base.raw_records import RawAnnotation, RawFormField
    from core_pdf.impl.document.pdf_page import PdfPage
    from core_pdf.impl.document.structure import PageStructure
    from core_pdf.impl.extract.capture import StructureUnset


def collected_records[Record, T](
    fetch: Callable[[], Iterable[Record]], build: Callable[[int, Record], T]
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


def link_annotations(
    built: tuple[tuple[RawAnnotation, Annotation], ...], resolver: Any
) -> tuple[Annotation, ...]:
    """The annotations with their replies and pop-ups linked (ISO 32000-2 12.5.6.2,
    12.5.6.14), as indexes in the annotations built, so that one whose build failed and was
    dropped shifts none. A link to an annotation not among them, on another page or
    dropped, is None, and so is the reply type of a reply whose target is."""
    # A dictionary Annots lists more than once is linked to at its first position.
    indexes: dict[int, int] = {}
    for index, (record, _) in enumerate(built):
        indexes.setdefault(id(record.dict), index)
    linked: list[Annotation] = []
    for record, built_annotation in built:
        try:
            reply = reply_relation(record.dict, resolver.resolve, resolver.resolve_name)
            target = popup_link(record.dict, record.subtype, resolver.resolve)
        except TypeError, ValueError, PdfError:
            linked.append(built_annotation)
            continue
        reply_index = None if reply is None else indexes.get(id(reply[0]))
        target_index = None if target is None else indexes.get(id(target))
        # Never a link to the annotation itself, whichever copy of it Annots lists.
        if reply is not None and reply[0] is record.dict:
            reply_index = None
        if target is record.dict:
            target_index = None
        is_popup = record.subtype == "Popup"
        if reply_index is None and target_index is None:
            linked.append(built_annotation)
            continue
        linked.append(
            replace(
                built_annotation,
                in_reply_to=reply_index,
                reply_type=None if reply is None or reply_index is None else reply[1],
                popup=None if is_popup else target_index,
                popup_parent=target_index if is_popup else None,
            )
        )
    return tuple(linked)


class Stage(Protocol):
    def __call__(
        self, state: PageState, extraction: PageExtraction, context: ExtractionScope, /
    ) -> PageState: ...


StageAnchor = type | Stage


def stage_matches(stage: Stage, anchor: StageAnchor) -> bool:
    return isinstance(stage, anchor) if isinstance(anchor, type) else stage is anchor


class PagePipeline(GeneratedRecord):
    stages: tuple[Stage, ...]

    def __hash__(self) -> int:
        return hash(self.stages)

    def run(self, extraction: PageExtraction, context: ExtractionScope) -> PageState:
        context.raise_if_cancelled()
        state = PageState(extraction.capture.observations)
        for stage in self.stages:
            state = stage(state, extraction, context)
        return state

    def position(self, anchor: StageAnchor) -> int:
        for index, stage in enumerate(self.stages):
            if stage_matches(stage, anchor):
                return index
        raise ValueError(f"pipeline has no stage matching {anchor!r}")

    def replacing(self, anchor: StageAnchor, stage: Stage) -> PagePipeline:
        index = self.position(anchor)
        return PagePipeline((*self.stages[:index], stage, *self.stages[index + 1 :]))

    def inserting_before(self, anchor: StageAnchor, stage: Stage) -> PagePipeline:
        index = self.position(anchor)
        return PagePipeline((*self.stages[:index], stage, *self.stages[index:]))

    def inserting_after(self, anchor: StageAnchor, stage: Stage) -> PagePipeline:
        index = self.position(anchor) + 1
        return PagePipeline((*self.stages[:index], stage, *self.stages[index:]))


class DetectTables:
    __slots__ = ("detector",)

    def __init__(self, detector: TableDetector = NATIVE_TABLES) -> None:
        self.detector = detector

    def __call__(
        self, state: PageState, extraction: PageExtraction, _context: ExtractionScope, /
    ) -> PageState:
        return replace(state, tables=self.detector.extract(extraction.capture, state.observations))


class LayoutBlocks:
    __slots__ = ("hooks",)

    def __init__(self, hooks: LayoutHooks = NATIVE_LAYOUT_HOOKS) -> None:
        self.hooks = hooks

    def __call__(
        self, state: PageState, extraction: PageExtraction, _context: ExtractionScope, /
    ) -> PageState:
        policy = extraction.reading_order
        table_obstacles = tuple(table.bbox for table in state.tables if table.bbox is not None)
        blocks, order_ambiguous = layout_blocks_with_evidence(
            state.observations,
            frame=extraction.frame,
            obstacles=(*table_obstacles, *policy.image_obstacles),
            use_xy_cut=policy.use_xy_cut,
            hooks=self.hooks,
        )
        return replace(state, blocks=blocks, order_ambiguous=order_ambiguous)


NATIVE_PIPELINE = PagePipeline((DetectTables(), LayoutBlocks()))


class PageExtraction:
    capture_page_fn = staticmethod(capture_page)
    pipeline: ClassVar[PagePipeline] = NATIVE_PIPELINE

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

    def run(self, context: ExtractionScope) -> PageState:
        return self.pipeline.run(self, context)

    @cached_property
    def reading_order(self) -> ReadingOrderPolicy:
        return ReadingOrderPolicy.from_evidence(self.capture.evidence)

    @cached_property
    def frame(self) -> PageFrame:
        capture = self.capture
        return PageFrame(
            capture.width, capture.height, capture.rotation, int(self.page.page_number)
        )

    def assembled_page(self, context: ExtractionScope) -> Page:
        capture = self.capture
        full_page_image = capture.evidence.full_page_image
        state = self.run(context)
        figures = (
            ()
            if full_page_image
            else tuple(
                Figure(order=index, bbox=box, kind="image", metadata={"source": "capture"})
                for index, box in enumerate(capture.evidence.image_boxes)
            )
        )
        assembled = assemble_page(
            state,
            self.frame,
            self.route_name,
            figures=figures,
            full_page_image=full_page_image,
            drawings=capture.program.drawings,
        )
        resolver = self.page.document.resolver
        raw_annotations = capture.annotations or ()
        resolved_annotation_dicts = tuple(record.dict for record in raw_annotations)

        def annotation_record(
            _index: int, record: RawAnnotation
        ) -> tuple[RawAnnotation, Annotation]:
            return record, Annotation(
                subtype=record.subtype,
                bbox=record.rect,
                contents=record.contents,
                destination=plain_pdf_value(
                    resolve_destination_references(resolver, record.dest or record.action)
                ),
            )

        built = collected_records(lambda: raw_annotations, annotation_record)
        annotations = link_annotations(built, resolver)
        links = collected_records(
            lambda: self.page.get_links(resolved_annotation_dicts),
            lambda _index, record: Link(
                bbox=record.bbox, url=record.url, link_type=record.link_type, text=""
            ),
        )
        source_fields = capture.fields
        fetch_fields = self.page.get_fields if source_fields is None else lambda: source_fields
        document = self.page.document
        # The interactive form dictionary, read once for the fields that need it.
        acroform = cache(lambda: document.acroform)

        def field_record(index: int, record: RawFormField) -> FormField:
            return FormField(
                name=record.name,
                field_type=record.type,
                value_text=record.value_text,
                bbox=record.rect,
                field_index=index,
                required=record.is_required,
                read_only=record.is_read_only,
                no_export=record.no_export,
                options=record.options,
                alignment=(
                    field_quadding(record.quadding, acroform(), resolver.resolve)
                    if record.type in VARIABLE_TEXT_FIELD_TYPES
                    else None
                ),
            )

        field_records = collected_records(fetch_fields, field_record)
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
