"""Concrete v0 document, page, and editor capability objects."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.text import collapse_ws, search_key

from . import inspection
from . import structured as structured_ir
from .editor import PdfEditor
from .errors import DocumentClosed, InvalidRequest
from .models import (
    AnnotationRecord,
    AttachmentInfo,
    ChunkRecord,
    ContentEvent,
    ContentEventKind,
    CoordinateOrigin,
    CoordinateSpace,
    Drawing,
    DrawingItem,
    ElementRecord,
    FormFieldRecord,
    GeometryIssue,
    GeometrySummary,
    ImageInfo,
    LinkRecord,
    OutlineItem,
    PageInfo,
    Raster,
    ReadingOrderItem,
    Rect,
    SearchHit,
    Severity,
    SourceRef,
    TableCell,
    TableRecord,
    TextBlock,
    TextCharacter,
    TextDiagnosticRun,
    TextLine,
    TextSpan,
    TextWord,
)
from .types import PageSelection, PdfInput

if TYPE_CHECKING:
    from core_pdf.impl.engine.document import (
        PdfDocument as EngineDocument,
    )
    from core_pdf.impl.engine.layout import LayoutGeometryIssue, LayoutGeometrySummary
    from core_pdf.impl.engine.page import PdfPage as EnginePage
    from core_pdf.impl.engine.structured import ChunkRecord as EngineChunkRecord

_SEARCH_MODES = frozenset({"exact", "normalized", "regex", "fuzzy"})


def page_space(page: EnginePage) -> CoordinateSpace:
    return CoordinateSpace(
        name="pdf-page",
        origin=CoordinateOrigin.BOTTOM_LEFT,
        width=float(page.width),
        height=float(page.height),
    )


def to_rect(value: object, space: CoordinateSpace) -> Rect | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    raw_value = cast(Any, value)
    x0, y0, x1, y1 = (float(item) for item in raw_value)
    return Rect(x0, y0, x1, y1, space)


def item_bbox(value: object, space: CoordinateSpace) -> Rect | None:
    if isinstance(value, (tuple, list)) and len(value) == 4:
        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in value):
            raw_value = cast(Any, value)
            points = [(float(item[0]), float(item[1])) for item in raw_value]
            return Rect(
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
                space,
            )
        try:
            return to_rect(value, space)
        except (TypeError, ValueError):
            return None
    candidate = value
    if all(hasattr(candidate, name) for name in ("x0", "y0", "x1", "y1")):
        candidate = cast(Any, candidate)
        return Rect(
            float(candidate.x0),
            float(candidate.y0),
            float(candidate.x1),
            float(candidate.y1),
            space,
        )
    return None


def source_ref(
    page: EnginePage, sequence: int | None = None, stage: str | None = None
) -> SourceRef:
    page_number = getattr(page, "page_number", page.structured_view.page_number)
    page_label = getattr(page, "label", page.structured_view.page_label)
    return SourceRef(
        page_index=page_number - 1,
        page_number=page_number,
        page_label=page_label,
        sequence=sequence,
        stage=stage,
    )


def to_geometry_issue(issue: LayoutGeometryIssue, space: CoordinateSpace) -> GeometryIssue:
    return GeometryIssue(
        code=issue.code,
        severity=Severity(issue.severity),
        subject=issue.subject,
        bbox=to_rect(issue.bbox, space),
        message=issue.message,
        details=dict(issue.details),
        repairable=issue.repairable,
    )


def to_geometry_summary(summary: LayoutGeometrySummary) -> GeometrySummary:
    return GeometrySummary(
        issue_count=summary.issue_count,
        error_count=summary.error_count,
        warning_count=summary.warning_count,
        repairable_count=summary.repairable_count,
        text_run_count=summary.text_run_count,
        line_count=summary.line_count,
        issue_codes=summary.issue_codes,
        suspicion_score=summary.suspicion_score,
    )


def to_chunk_record(chunk: EngineChunkRecord, spaces: Mapping[int, CoordinateSpace]) -> ChunkRecord:
    element_bboxes = tuple(
        Rect(float(x0), float(y0), float(x1), float(y1), spaces[page_number])
        for page_number, (x0, y0, x1, y1) in chunk.element_geometry
        if page_number in spaces
    )
    return ChunkRecord(
        text=chunk.text,
        page_numbers=chunk.page_numbers,
        element_ids=chunk.element_ids,
        element_types=chunk.element_types,
        section_path=chunk.section_path,
        metadata={
            "element_types": chunk.element_types,
            "element_geometry": chunk.element_geometry,
        },
        sources=tuple(
            SourceRef(page_number=page_number, stage="retrieval-chunk")
            for page_number in chunk.page_numbers
        ),
        element_bboxes=element_bboxes,
    )


@dataclass(slots=True)
class PdfPage:
    page: EnginePage

    @property
    def structured_view(self) -> structured_ir.Page:
        return self.page.structured_view

    @property
    def info(self) -> PageInfo:
        space = page_space(self.page)
        return PageInfo(
            index=self.page.page_number - 1,
            number=self.page.page_number,
            label=self.page.label,
            width=float(self.page.width),
            height=float(self.page.height),
            rotation=int(self.page.rotation),
            space=space,
        )

    def text_spans(self) -> Iterable[TextSpan]:
        space = page_space(self.page)
        for run in self.page.chars:
            bbox = to_rect((run.x0, run.y0, run.x1, run.y1), space)
            if bbox is None:
                continue
            characters = tuple(
                TextCharacter(
                    text=glyph.text,
                    bbox=to_rect(glyph.ink_bbox, space) or bbox,
                    font_name=glyph.font_name or run.font_name,
                    font_size=glyph.font_size or run.font_size,
                    color=glyph.fill or run.fill_color,
                    sequence=glyph.seqno,
                    visible=glyph.visible,
                    source=source_ref(self.page, glyph.seqno, "content-character"),
                    rotation_angle=int(getattr(run, "rotation_angle", 0)),
                )
                for cluster in (run.glyph_clusters or ())
                for glyph in cluster.glyphs
                if glyph.text
            )
            yield TextSpan(
                text=run.text,
                bbox=bbox,
                font_name=run.font_name,
                font_size=run.font_size,
                color=run.fill_color,
                sequence=run.seqno,
                characters=characters,
                source=source_ref(self.page, run.seqno, "content-text"),
            )

    def text(self) -> str:
        return self.page.structured_view.text

    def text_lines(self) -> Iterable[TextLine]:
        space = page_space(self.page)
        for block in self.page.structured_view.blocks:
            for line in block.lines:
                yield TextLine(text=line.text, bbox=to_rect(line.bbox, space))

    def text_blocks(self) -> Iterable[TextBlock]:
        space = page_space(self.page)
        for block in self.page.structured_view.blocks:
            lines = tuple(
                TextLine(text=line.text, bbox=to_rect(line.bbox, space)) for line in block.lines
            )
            yield TextBlock(text=block.text, bbox=to_rect(block.bbox, space), lines=lines)

    def text_characters(self) -> Iterable[TextCharacter]:
        for span in self.text_spans():
            yield from span.characters

    def text_diagnostics(self, *, include_invisible: bool = True) -> Iterable[TextDiagnosticRun]:
        space = page_space(self.page)
        diagnostics = self.page.text_diagnostics(include_invisible=include_invisible)
        for run in diagnostics.runs:
            issues = tuple(
                to_geometry_issue(cast("LayoutGeometryIssue", issue), space)
                for issue in run.geometry_issues
            )
            yield TextDiagnosticRun(
                text=run.text,
                bbox=to_rect(run.bbox, space) or Rect(0.0, 0.0, 0.0, 0.0, space),
                font_name=run.font_name,
                font_size=run.font_size,
                is_vertical=run.is_vertical,
                visible=run.visible,
                rotation=run.rotation,
                sequence=run.seqno,
                geometry_issues=issues,
            )

    def words(self) -> Iterable[TextWord]:
        space = page_space(self.page)
        for word in self.page.structured_view.words:
            yield TextWord(
                text=word.text,
                bbox=to_rect(word.bbox, space),
                block_index=word.block_index,
                line_index=word.line_index,
                word_index=word.word_index,
                source=source_ref(self.page, None, "content-word"),
            )

    def drawings(self) -> Iterable[Drawing]:
        space = page_space(self.page)
        for drawing in self.page.get_drawings():
            items = tuple(
                DrawingItem(
                    kind=str(item[0]),
                    bbox=item_bbox(item[1], space) if len(item) > 1 else None,
                    data={"raw": item},
                )
                for item in drawing.items
                if isinstance(item, tuple) and item
            )
            yield Drawing(
                kind=drawing.kind,
                bbox=to_rect(drawing.rect, space),
                sequence=drawing.seqno,
                items=items,
                fill=drawing.fill,
                stroke=drawing.stroke_color,
                fill_opacity=drawing.fill_opacity,
                stroke_opacity=drawing.stroke_opacity,
                source=source_ref(self.page, drawing.seqno, "content-drawing"),
                data={
                    "fill_pattern": drawing.fill_pattern,
                    "stroke_pattern": drawing.stroke_pattern,
                },
            )

    def geometry_issues(self) -> Iterable[GeometryIssue]:
        space = page_space(self.page)
        for issue in self.page.extract_geometry_issues():
            yield to_geometry_issue(cast("LayoutGeometryIssue", issue), space)

    def geometry_summary(self) -> GeometrySummary:
        return to_geometry_summary(self.page.extract_geometry_summary())

    def tables(self) -> Iterable[TableRecord]:
        space = page_space(self.page)
        for table in self.page.structured_view.tables:
            cells = tuple(
                TableCell(
                    text=cell.text,
                    bbox=to_rect(cell.bbox, space),
                    row=cell.row,
                    column=cell.column,
                )
                for row in table.rows
                for cell in row
            )
            yield TableRecord(
                bbox=to_rect(table.bbox, space),
                cells=cells,
                rows=len(table.rows),
                columns=max((len(row) for row in table.rows), default=0),
                source=source_ref(self.page, None, "table"),
            )

    def annotations(self) -> Iterable[AnnotationRecord]:
        space = page_space(self.page)
        for annotation in self.page.structured_view.annotations:
            yield AnnotationRecord(
                subtype=annotation.subtype or "",
                bbox=to_rect(annotation.bbox, space),
                contents=annotation.contents,
                destination=annotation.destination,
                source=source_ref(self.page, None, "annotation"),
            )

    def links(self) -> Iterable[LinkRecord]:
        space = page_space(self.page)
        for link in self.page.structured_view.links:
            yield LinkRecord(
                bbox=to_rect(link.bbox, space),
                url=link.url,
                link_type=link.link_type or "uri",
                source=source_ref(self.page, None, "link"),
            )

    def form_fields(self) -> Iterable[FormFieldRecord]:
        space = page_space(self.page)
        for field in self.page.structured_view.form_fields:
            yield FormFieldRecord(
                name=field.name,
                field_type=field.field_type,
                value=field.value_text,
                bbox=to_rect(field.bbox, space),
                field_index=field.field_index,
                source=source_ref(self.page, field.field_index, "form-field"),
                required=field.required,
                read_only=field.read_only,
                no_export=field.no_export,
                options=field.options,
            )

    def images(self) -> Iterable[ImageInfo]:
        space = page_space(self.page)
        for image in self.page.extract_images():
            metadata = image.image_metadata
            if metadata is None:
                continue
            yield ImageInfo(
                bbox=to_rect(image.rect, space),
                width=metadata.width,
                height=metadata.height,
                channels=metadata.channels,
                color_model=metadata.color_model,
                alpha=metadata.alpha,
                sequence=image.seqno,
                source=source_ref(self.page, image.seqno, "content-image"),
                data=image.data if isinstance(image.data, (bytes, memoryview)) else None,
            )

    def content_events(self) -> Iterable[ContentEvent]:
        events = [
            ContentEvent(
                kind=ContentEventKind.TEXT,
                sequence=span.sequence or index,
                bbox=span.bbox,
                text=span,
                source=span.source,
            )
            for index, span in enumerate(self.text_spans())
        ]
        events.extend(
            ContentEvent(
                kind=ContentEventKind.DRAWING,
                sequence=drawing.sequence,
                bbox=drawing.bbox,
                drawing=drawing,
                source=drawing.source,
            )
            for drawing in self.drawings()
        )
        events.extend(
            ContentEvent(
                kind=ContentEventKind.IMAGE,
                sequence=image.sequence or index,
                bbox=image.bbox,
                image=image,
                source=image.source,
            )
            for index, image in enumerate(self.images())
        )
        yield from sorted(events, key=lambda event: event.sequence)

    def render(
        self,
        *,
        dpi: float = 72.0,
        crop: tuple[float, float, float, float] | None = None,
    ) -> Raster:
        rendered = self.page.render()
        raster = rendered.rasterize(scale=max(0.01, dpi / 72.0), crop=crop)
        space = CoordinateSpace(
            name="pixel",
            origin=CoordinateOrigin.TOP_LEFT,
            width=float(raster.width),
            height=float(raster.height),
            unit="pixel",
        )
        return Raster(
            data=raster.pixels,
            width=raster.width,
            height=raster.height,
            channels=raster.channels,
            pixel_format="rgba8" if raster.channels == 4 else f"channels{raster.channels}",
            space=space,
            dpi=dpi,
        )

    def to_markdown(self) -> str:
        return self.page.extract().to_markdown()

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        from core_pdf.impl.engine.structured.serialization import page_to_json_dict

        return json.dumps(
            page_to_json_dict(self.page.extract()), indent=indent, sort_keys=sort_keys
        )

    def to_html(self) -> str:
        return self.page.extract().to_html()


@dataclass(slots=True)
class PdfDocument:
    internal_document: EngineDocument
    internal_owned: bool = False
    internal_pages: dict[int, PdfPage] = dataclass_field(default_factory=dict)

    @classmethod
    def open(cls, source: PdfInput, password: str = "") -> "PdfDocument":
        from core_pdf.impl.engine.document import PdfDocument as EngineDocument

        return cls(EngineDocument.open(source, password=password), internal_owned=True)

    @classmethod
    def from_structured(cls, document: structured_ir.Document) -> "PdfDocument":
        from core_pdf.impl.engine.document import PdfDocument as EngineDocument

        return cls(EngineDocument.from_structured(document), internal_owned=True)

    @classmethod
    def internal_from_engine(cls, document: EngineDocument) -> "PdfDocument":
        """Project an engine document without taking ownership of its lifecycle."""
        return cls(document)

    def _doc(self) -> EngineDocument:
        """Return the engine document, raising once if it has been closed."""
        if self.internal_document.closed:
            raise DocumentClosed("PDF document is closed")
        return self.internal_document

    @property
    def structured(self) -> structured_ir.Document:
        return self.internal_document.structured_document

    def edit(self) -> "PdfEditor":
        return PdfEditor(self._doc())

    @property
    def closed(self) -> bool:
        return bool(self.internal_document.closed)

    def close(self) -> None:
        self.internal_document.close()

    @property
    def page_count(self) -> int:
        return int(self.internal_document.page_count())

    def page(self, index: int) -> PdfPage:
        document = self._doc()
        if index < 0 or index >= self.page_count:
            raise IndexError(index)
        if index not in self.internal_pages:
            self.internal_pages[index] = PdfPage(document.pages[index])
        return self.internal_pages[index]

    def pages(self, selection: PageSelection | None = None) -> Iterable[PdfPage]:
        document = self._doc()
        indexes = document.selected_page_indexes(selection)
        return (self.page(index) for index in indexes)

    @property
    def metadata(self) -> Mapping[str, object]:
        metadata = self._doc().get_metadata()
        if isinstance(metadata, dict):
            return dict(metadata)
        return dict(vars(metadata)) if hasattr(metadata, "__dict__") else {"value": metadata}

    inventory = inspection.document_inventory
    analysis_snapshot = inspection.analysis_snapshot
    native_features = inspection.native_features
    action_inventory = inspection.action_inventory
    optional_content_layers = inspection.optional_content_layers
    revisions = inspection.revisions
    revision_objects = inspection.revision_objects
    fingerprint = inspection.document_fingerprint
    incremental_plan = inspection.incremental_plan
    archival_manifest = inspection.archival_manifest
    object_graph = inspection.object_graph
    inspect_object = inspection.inspect_object
    verify_object_roundtrip = inspection.verify_object_roundtrip
    verify_object_roundtrips = inspection.verify_object_roundtrips

    def text(self, *, pages: PageSelection | None = None) -> str:
        return "\f".join(page.structured_view.text for page in self.pages(pages)) + "\f"

    def words(self, *, pages: PageSelection | None = None) -> Iterable[TextWord]:
        for page in self.pages(pages):
            yield from page.words()

    def text_spans(self, *, pages: PageSelection | None = None) -> Iterable[TextSpan]:
        for page in self.pages(pages):
            yield from page.text_spans()

    def text_characters(self, *, pages: PageSelection | None = None) -> Iterable[TextCharacter]:
        for page in self.pages(pages):
            yield from page.text_characters()

    def text_lines(self, *, pages: PageSelection | None = None) -> Iterable[TextLine]:
        for page in self.pages(pages):
            yield from page.text_lines()

    def reading_order(self, *, pages: PageSelection | None = None) -> Iterable[ReadingOrderItem]:
        for page in self.pages(pages):
            lines = tuple((line, line.bbox) for line in page.text_lines() if line.bbox is not None)
            ordered = sorted(lines, key=lambda item: (-item[1].y1, item[1].x0))
            for order, (line, bbox) in enumerate(ordered):
                yield ReadingOrderItem(
                    page_number=page.info.number,
                    order=order,
                    text=line.text,
                    bbox=bbox,
                    source=line.source,
                )

    def text_blocks(self, *, pages: PageSelection | None = None) -> Iterable[TextBlock]:
        for page in self.pages(pages):
            yield from page.text_blocks()

    def text_diagnostics(
        self, *, include_invisible: bool = True, pages: PageSelection | None = None
    ) -> Iterable[TextDiagnosticRun]:
        for page in self.pages(pages):
            yield from page.text_diagnostics(include_invisible=include_invisible)

    def search(
        self,
        query: str,
        *,
        mode: str = "exact",
        threshold: float = 0.8,
        region: Rect | None = None,
        pages: PageSelection | None = None,
    ) -> Iterable[SearchHit]:
        """Search words using exact, normalized, regex, or fuzzy matching."""
        if mode not in _SEARCH_MODES:
            raise InvalidRequest(f"unsupported search mode: {mode!r}")
        if not 0.0 <= threshold <= 1.0:
            raise InvalidRequest("search threshold must be between 0 and 1")
        normalized_query = search_key(query)
        pattern = re.compile(query, re.IGNORECASE) if mode == "regex" else None

        def in_region(bbox: Rect) -> bool:
            return (
                region is None
                or bbox.x1 > region.x0
                and bbox.x0 < region.x1
                and bbox.y1 > region.y0
                and bbox.y0 < region.y1
            )

        def match(candidate: str) -> tuple[bool, float]:
            folded = candidate.casefold()
            if mode == "exact":
                return normalized_query in folded, 1.0
            if mode == "normalized":
                return normalized_query in collapse_ws(folded), 1.0
            if mode == "regex":
                return pattern is not None and pattern.search(candidate) is not None, 1.0
            score = SequenceMatcher(None, normalized_query, folded).ratio()
            return score >= threshold, score

        def hits() -> Iterable[SearchHit]:
            for page in self.pages(pages):
                if " " in normalized_query:
                    for line in page.text_lines():
                        if line.bbox is None or not in_region(line.bbox):
                            continue
                        matched, score = match(line.text)
                        if matched:
                            yield SearchHit(
                                page.info.number, line.bbox, line.text, score, mode, line.source
                            )
                    continue
                for word in page.words():
                    if word.bbox is None or not in_region(word.bbox):
                        continue
                    candidate = word.text
                    matched, score = match(candidate)
                    if matched:
                        yield SearchHit(
                            page.info.number, word.bbox, candidate, score, mode, word.source
                        )

        return hits()

    def elements(self, *, pages: PageSelection | None = None) -> Iterable[ElementRecord]:
        from core_pdf.impl.engine.structured import document_elements

        document = self._doc()
        spaces = {page.info.number: page.info.space for page in self.pages(pages)}
        return tuple(
            ElementRecord(
                element_id=record.element_id,
                kind=record.kind,
                text=record.text,
                page_number=record.page_number,
                bbox=to_rect(record.bbox, spaces[record.page_number]),
                order=record.order,
                metadata=dict(record.metadata),
                source=SourceRef(page_number=record.page_number, stage="structured-element"),
            )
            for record in document_elements(
                document.extract(pages=pages),
                document.extract_images(pages=pages),
            )
        )

    def chunks(
        self, *, max_characters: int = 2000, pages: PageSelection | None = None
    ) -> Iterable[ChunkRecord]:
        from core_pdf.impl.engine.structured import (
            chunk_elements,
            document_elements,
            document_section_paths,
        )

        document = self._doc()
        section_paths = document_section_paths(self.structured)
        records = chunk_elements(
            document_elements(
                document.extract(pages=pages),
                document.extract_images(pages=pages),
            ),
            max_characters=max_characters,
            section_paths=section_paths,
        )
        page_spaces = {page.info.number: page.info.space for page in self.pages(pages)}
        for chunk in records:
            yield to_chunk_record(chunk, page_spaces)

    def images(self, *, pages: PageSelection | None = None) -> Iterable[ImageInfo]:
        for page in self.pages(pages):
            yield from page.images()

    resource_inventory = inspection.resource_inventory

    content_dependencies = inspection.content_dependencies

    resource_dependency_graph = inspection.resource_dependency_graph

    evidence_graph = inspection.evidence_graph

    resource_diagnostics = inspection.resource_diagnostics

    content_summaries = inspection.content_summaries

    form_inventory = inspection.form_inventory

    annotation_inventory = inspection.annotation_inventory

    def tables(self, *, pages: PageSelection | None = None) -> Iterable[TableRecord]:
        for page in self.pages(pages):
            yield from page.tables()

    def drawings(self, *, pages: PageSelection | None = None) -> Iterable[Drawing]:
        for page in self.pages(pages):
            yield from page.drawings()

    def geometry_issues(self, *, pages: PageSelection | None = None) -> Iterable[GeometryIssue]:
        for page in self.pages(pages):
            yield from page.geometry_issues()

    def geometry_summaries(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[GeometrySummary]:
        for page in self.pages(pages):
            yield page.geometry_summary()

    structure_elements = inspection.structure_elements

    accessibility_inventory = inspection.accessibility_inventory

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return self._doc().extract().to_json(indent=indent, sort_keys=sort_keys)

    def to_structured_json(
        self,
        *,
        pages: PageSelection | None = None,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        document = self._doc()
        selected = document.selected_page_indexes(pages)
        extracted = document.extract(pages=tuple(index + 1 for index in selected))
        payload = {
            "schema_version": extracted.schema_version,
            "document": extracted.to_json_dict(),
            "metadata": document.get_metadata(),
            "page_count": document.page_count(),
            "summary": {
                "page_count": document.page_count(),
                "selected_page_count": len(extracted.pages),
                "selected_pages": [index + 1 for index in selected],
            },
        }
        return json.dumps(payload, indent=indent, sort_keys=sort_keys)

    def to_html(self) -> str:
        return self._doc().extract().to_html()

    def to_markdown(self) -> str:
        return self._doc().extract().to_markdown()

    def to_csv(self, *, pages: PageSelection | None = None) -> str:
        return self._doc().extract(pages=pages).to_csv()

    def to_tei(self, *, pages: PageSelection | None = None) -> str:
        return self._doc().extract(pages=pages).to_tei()

    def annotations(self, *, pages: PageSelection | None = None) -> Iterable[AnnotationRecord]:
        for page in self.pages(pages):
            yield from page.annotations()

    def links(self, *, pages: PageSelection | None = None) -> Iterable[LinkRecord]:
        for page in self.pages(pages):
            yield from page.links()

    def form_fields(self, *, pages: PageSelection | None = None) -> Iterable[FormFieldRecord]:
        for page in self.pages(pages):
            yield from page.form_fields()

    @property
    def outlines(self) -> Iterable[OutlineItem]:
        return tuple(
            OutlineItem(
                title=str(item.title),
                page_number=(item.page_index + 1) if item.page_index is not None else None,
                level=int(item.level) + 1,
                uri=str(item.dest) if isinstance(item.dest, str) else None,
            )
            for item in self._doc().outlines
        )

    @property
    def attachments(self) -> Iterable[AttachmentInfo]:
        return tuple(
            AttachmentInfo(
                name=str(item.filename),
                size=len(item.data),
                data=item.data,
            )
            for item in self._doc().attachments
        )

    embedded_resources = inspection.embedded_resources

    def __enter__(self) -> "PdfDocument":
        self._doc()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        if self.internal_owned:
            self.internal_document.__exit__(exc_type, exc, cast(Any, traceback))


__all__ = (
    "PdfDocument",
    "PdfPage",
)
