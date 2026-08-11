"""Adapters exposing the existing engine through the v0 contracts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from core_pdf.impl.text import collapse_ws, search_key

from . import inspection, structured, verification
from .convert import (
    item_bbox,
    page_space,
    source_ref,
    to_chunk_record,
    to_geometry_issue,
    to_geometry_summary,
    to_rect,
)
from .errors import DocumentClosed, InvalidRequest
from .models import (
    AccessibilityInventory,
    AccessibilityRepairVerification,
    ActionInventory,
    AnnotationInventory,
    AnnotationRecord,
    ArchivalManifest,
    AttachmentInfo,
    ChunkRecord,
    ContentDependencyRecord,
    ContentEvent,
    ContentEventKind,
    CoordinateOrigin,
    CoordinateSpace,
    DocumentAnalysisSnapshot,
    DocumentFingerprint,
    DocumentInventory,
    Drawing,
    DrawingItem,
    EmbeddedResourceRecord,
    EvidenceGraph,
    FormFieldRecord,
    FormInventory,
    GeometryIssue,
    GeometrySummary,
    ImageInfo,
    IncrementalAnalysisPlan,
    LinkRecord,
    NativeFeatureInventory,
    ObjectGraphReport,
    ObjectInspection,
    ObjectRoundTripManifest,
    ObjectRoundTripVerification,
    OptionalContentLayerRecord,
    OutlineItem,
    PageContentSummary,
    PageInfo,
    PageResourceInventory,
    PreservationManifest,
    Raster,
    ReadingOrderItem,
    Rect,
    RedactionVerification,
    ResourceDependencyGraph,
    ResourceDiagnostic,
    RevisionInventory,
    RevisionObjectRecord,
    SanitizationVerification,
    SearchHit,
    StructureElementRecord,
    TableCell,
    TableRecord,
    TextBlock,
    TextCharacter,
    TextDiagnosticRun,
    TextLine,
    TextSpan,
    TextWord,
)
from .protocols import (
    PageSelection,
    PdfDocumentProtocol,
    PdfPageProtocol,
)

_SEARCH_MODES = frozenset({"exact", "normalized", "regex", "fuzzy"})


@dataclass(slots=True)
class PdfPageAdapter(PdfPageProtocol):
    page: Any

    @property
    def structured_view(self) -> structured.Page:
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
            issues = tuple(to_geometry_issue(issue, space) for issue in run.geometry_issues)
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
            yield to_geometry_issue(issue, space)

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
        return self.page.to_markdown()

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return self.page.to_json(indent=indent, sort_keys=sort_keys)

    def to_html(self) -> str:
        return self.page.to_html()


@dataclass(slots=True)
class PdfObjectResolverAdapter:
    resolver: Any

    def resolve(self, value: object) -> object:
        return self.resolver.resolve(value)


@dataclass(slots=True)
class PdfDocumentAdapter(PdfDocumentProtocol):
    document: Any

    def _doc(self) -> Any:
        """Return the engine document, raising once if it has been closed."""
        if self.document.closed:
            raise DocumentClosed("PDF document is closed")
        return self.document

    @property
    def structured_document(self) -> structured.Document:
        return self.document.structured_document

    @property
    def structured_pages(self) -> tuple[structured.Page, ...]:
        return tuple(self.document.structured_pages)

    def edit(self) -> "PdfEditorAdapter":
        return PdfEditorAdapter(self._doc().edit())

    @property
    def closed(self) -> bool:
        return bool(self.document.closed)

    def close(self) -> None:
        self.document.close()

    @property
    def page_count(self) -> int:
        return int(self.document.page_count())

    @property
    def resolver(self) -> PdfObjectResolverAdapter:
        return PdfObjectResolverAdapter(self._doc().resolver)

    def page(self, index: int) -> PdfPageAdapter:
        document = self._doc()
        if index < 0 or index >= self.page_count:
            raise IndexError(index)
        return PdfPageAdapter(document.pages[index])

    def pages(self, selection: PageSelection | None = None) -> Iterable[PdfPageAdapter]:
        document = self._doc()
        indexes = document.selected_page_indexes(selection)
        return (PdfPageAdapter(document.pages[index]) for index in indexes)

    def get_metadata(self) -> Mapping[str, object]:
        metadata = self._doc().get_metadata()
        if isinstance(metadata, dict):
            return dict(metadata)
        return dict(vars(metadata)) if hasattr(metadata, "__dict__") else {"value": metadata}

    def inventory(self) -> DocumentInventory:
        return inspection.document_inventory(self._doc())

    def analysis_snapshot(self) -> DocumentAnalysisSnapshot:
        return inspection.analysis_snapshot(self)

    def native_features(self) -> NativeFeatureInventory:
        return inspection.native_features(self._doc())

    def action_inventory(self) -> ActionInventory:
        return inspection.action_inventory(self._doc())

    def optional_content_layers(self) -> Iterable[OptionalContentLayerRecord]:
        return inspection.optional_content_layers(self._doc())

    def revisions(self) -> RevisionInventory:
        return inspection.revisions(self._doc())

    def revision_objects(self) -> Iterable[RevisionObjectRecord]:
        return inspection.revision_objects(self._doc())

    def fingerprint(self) -> DocumentFingerprint:
        return inspection.document_fingerprint(self._doc())

    def incremental_plan(self, baseline: DocumentFingerprint) -> IncrementalAnalysisPlan:
        return inspection.incremental_plan(self._doc(), baseline)

    def archival_manifest(self) -> ArchivalManifest:
        return inspection.archival_manifest(self)

    def object_graph(self) -> ObjectGraphReport:
        return inspection.object_graph(self._doc())

    def inspect_object(self, object_number: int) -> ObjectInspection:
        return inspection.inspect_object(self._doc(), object_number)

    def verify_object_roundtrip(self, object_number: int) -> ObjectRoundTripVerification:
        return inspection.verify_object_roundtrip(self._doc(), object_number)

    def verify_object_roundtrips(self, object_numbers: Iterable[int]) -> ObjectRoundTripManifest:
        return inspection.verify_object_roundtrips(self._doc(), object_numbers)

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

    def elements(self, *, pages: PageSelection | None = None) -> Iterable[Mapping[str, object]]:
        return self._doc().extract_elements(pages=pages)

    def chunks(
        self, *, max_characters: int = 2000, pages: PageSelection | None = None
    ) -> Iterable[ChunkRecord]:
        from core_pdf.impl.engine.structured import chunk_elements, document_section_paths

        document = self._doc()
        section_paths = document_section_paths(self.structured_document)
        records = chunk_elements(
            document.extract_element_records(pages=pages),
            max_characters=max_characters,
            section_paths=section_paths,
        )
        page_spaces = {page.info.number: page.info.space for page in self.pages(pages)}
        for chunk in records:
            yield to_chunk_record(chunk, page_spaces)

    def images(self, *, pages: PageSelection | None = None) -> Iterable[ImageInfo]:
        for page in self.pages(pages):
            yield from page.images()

    def resource_inventory(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[PageResourceInventory]:
        return inspection.resource_inventory(self, pages=pages)

    def content_dependencies(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[ContentDependencyRecord]:
        return inspection.content_dependencies(self, pages=pages)

    def resource_dependency_graph(self) -> ResourceDependencyGraph:
        return inspection.resource_dependency_graph(self)

    def evidence_graph(self, *, pages: PageSelection | None = None) -> EvidenceGraph:
        return inspection.evidence_graph(self, pages=pages)

    def resource_diagnostics(self) -> Iterable[ResourceDiagnostic]:
        return inspection.resource_diagnostics(self)

    def content_summaries(
        self, *, pages: PageSelection | None = None
    ) -> Iterable[PageContentSummary]:
        return inspection.content_summaries(self._doc(), pages=pages)

    def form_inventory(self, *, pages: PageSelection | None = None) -> FormInventory:
        return inspection.form_inventory(self, pages=pages)

    def annotation_inventory(self, *, pages: PageSelection | None = None) -> AnnotationInventory:
        return inspection.annotation_inventory(self, pages=pages)

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

    def structure_elements(self) -> Iterable[StructureElementRecord]:
        return inspection.structure_elements(self._doc())

    def accessibility_inventory(
        self, *, pages: PageSelection | None = None
    ) -> AccessibilityInventory:
        return inspection.accessibility_inventory(self, pages=pages)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return self._doc().to_json(indent=indent, sort_keys=sort_keys)

    def to_structured_json(
        self,
        *,
        pages: PageSelection | None = None,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        return self._doc().to_structured_json_string(
            pages=pages,
            indent=indent,
            sort_keys=sort_keys,
        )

    def to_html(self) -> str:
        return self._doc().to_html()

    def to_markdown(self) -> str:
        return self._doc().to_markdown()

    def to_csv(self, *, pages: PageSelection | None = None) -> str:
        return self._doc().to_csv(pages=pages)

    def to_tei(self, *, pages: PageSelection | None = None) -> str:
        return self._doc().to_tei(pages=pages)

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

    def embedded_resources(self) -> Iterable[EmbeddedResourceRecord]:
        return inspection.embedded_resources(self._doc())

    def __enter__(self) -> PdfDocumentAdapter:
        self.document.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.document.__exit__(*args)


def adapt_document(document: Any) -> PdfDocumentAdapter:
    """Expose a current core-pdf document through the v0 protocol surface."""

    return PdfDocumentAdapter(document)


def adapt_structured(document: Any) -> PdfDocumentAdapter:
    """Materialize a structured snapshot and expose it through the v0 contract."""
    from core_pdf.impl.engine.document import PdfDocument

    return adapt_document(PdfDocument.from_structured(document))


@dataclass(slots=True)
class PdfEditorAdapter:
    editor: Any

    def _chain(self, name: str, /, *args: object, **kwargs: object) -> "PdfEditorAdapter":
        """Call `editor.<name>(*args, **kwargs)` and return self for chaining."""
        getattr(self.editor, name)(*args, **kwargs)
        return self

    def set_metadata(self, values: Mapping[str, object]) -> "PdfEditorAdapter":
        return self._chain("set_metadata", dict(values))

    def set_page_geometry(
        self,
        page_number: int,
        *,
        rotation: int | None = None,
        cropbox: tuple[float, float, float, float] | None = None,
    ) -> "PdfEditorAdapter":
        return self._chain("set_page_geometry", page_number, rotation=rotation, cropbox=cropbox)

    def encrypt(
        self, user_password: str, *, owner_password: str | None = None
    ) -> "PdfEditorAdapter":
        return self._chain("encrypt", user_password, owner_password=owner_password)

    def sign(self, provider: Any, *, contents_length: int = 8192) -> "PdfEditorAdapter":
        return self._chain("sign", provider, contents_length=contents_length)

    def replace_page(self, page_number: int, page: structured.Page) -> "PdfEditorAdapter":
        return self._chain("replace_page", page_number, page)

    def insert_page(
        self, position: int, width: float = 595.0, height: float = 842.0
    ) -> "PdfEditorAdapter":
        return self._chain("insert_page", position, width, height)

    def insert_structured_page(self, position: int, page: structured.Page) -> "PdfEditorAdapter":
        return self._chain("insert_structured_page", position, page)

    def update_form_field(self, name: str, value: str) -> "PdfEditorAdapter":
        return self._chain("update_form_field", name, value)

    def remove_form_fields(self, names: Iterable[str]) -> "PdfEditorAdapter":
        return self._chain("remove_form_fields", names)

    def apply_redactions(
        self, redactions: Mapping[int, Iterable[tuple[float, float, float, float]]]
    ) -> "PdfEditorAdapter":
        return self._chain("apply_redactions", redactions)

    def remove_annotations(
        self, page_number: int, indices: Iterable[int] | None = None
    ) -> "PdfEditorAdapter":
        return self._chain("remove_annotations", page_number, indices)

    def remove_links(
        self, page_number: int, indices: Iterable[int] | None = None
    ) -> "PdfEditorAdapter":
        return self._chain("remove_links", page_number, indices)

    def delete_pages(self, selection: PageSelection) -> "PdfEditorAdapter":
        return self._chain("delete_pages", selection)

    def set_attachments(self, values: Mapping[str, bytes]) -> "PdfEditorAdapter":
        return self._chain("set_attachments", dict(values))

    def set_outlines(self, values: Iterable[Iterable[object]]) -> "PdfEditorAdapter":
        return self._chain("set_outlines", values)

    def replace_pages(self, pages: Iterable[structured.Page]) -> "PdfEditorAdapter":
        return self._chain("replace_pages", pages)

    def add_annotation(
        self,
        page_number: int,
        subtype: str,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        contents: str = "",
        destination: object = None,
    ) -> "PdfEditorAdapter":
        return self._chain(
            "add_annotation", page_number, subtype, bbox, contents=contents, destination=destination
        )

    def add_link(
        self,
        page_number: int,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        url: str | None = None,
        link_type: str | None = None,
        text: str = "",
    ) -> "PdfEditorAdapter":
        return self._chain("add_link", page_number, bbox, url=url, link_type=link_type, text=text)

    def commit(self, target: str | Path | Any) -> bytes:
        return self.editor.commit(target)

    def commit_verified(
        self, target: str | Path | Any, *, expected_unchanged_pages: tuple[int, ...] = ()
    ) -> PreservationManifest:
        return verification.commit_verified(
            self.editor, target, expected_unchanged_pages=expected_unchanged_pages
        )

    def commit_redactions_verified(
        self,
        target: str | Path | Any,
        redactions: Mapping[int, Iterable[tuple[float, float, float, float]]],
        *,
        queries: Iterable[str] = (),
    ) -> RedactionVerification:
        return verification.commit_redactions_verified(
            self.editor, target, redactions, queries=queries
        )

    def commit_sanitized_verified(
        self,
        target: str | Path | Any,
        *,
        metadata: bool = True,
        annotations: bool = True,
        links: bool = True,
        forms: bool = True,
        attachments: bool = True,
        outlines: bool = True,
        actions: bool = True,
    ) -> SanitizationVerification:
        return verification.commit_sanitized_verified(
            self.editor,
            target,
            metadata=metadata,
            annotations=annotations,
            links=links,
            forms=forms,
            attachments=attachments,
            outlines=outlines,
            actions=actions,
        )

    def commit_accessibility_repair_verified(
        self,
        target: str | Path | Any,
        *,
        title: str | None = None,
        language: str | None = None,
    ) -> AccessibilityRepairVerification:
        return verification.commit_accessibility_repair_verified(
            self.editor, target, title=title, language=language
        )

    def commit_document(self) -> structured.Document:
        return self.editor.commit_document()

    def rollback(self) -> None:
        self.editor.rollback()


__all__ = (
    "PdfDocumentAdapter",
    "PdfEditorAdapter",
    "PdfObjectResolverAdapter",
    "PdfPageAdapter",
    "adapt_document",
    "adapt_structured",
)
