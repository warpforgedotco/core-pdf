"""High-level local document extraction and editing facade."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from os import PathLike
from typing import Any

from core_pdf import PdfDocument
from core_pdf.api.v0.types import PdfInput

from ._common import (
    Block,
    BlockKind,
    ClosingMixin,
    Document,
    Page,
    TextLine,
    open_source,
    project_document,
    serialize_document_to_pdf,
    structured_elements,
    write_bytes,
)


@dataclass(frozen=True, slots=True)
class Element:
    """Normalized element suitable for indexing or downstream LLM pipelines."""

    element_id: str
    type: str
    text: str
    page_number: int
    bbox: tuple[float, float, float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "type": self.type,
            "text": self.text,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Chunk:
    """A deterministic, page-aware text chunk with source element metadata."""

    text: str
    page_numbers: tuple[int, ...]
    element_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page_numbers": self.page_numbers,
            "element_ids": self.element_ids,
            "metadata": dict(self.metadata),
        }


def elements(document: Document) -> tuple[Element, ...]:
    """Convert the canonical structured element projection to compatibility records."""
    return tuple(
        Element(
            element_id=item.element_id,
            type=item.kind,
            text=item.text,
            page_number=item.page_number,
            bbox=item.bbox,
            metadata=dict(item.metadata),
        )
        for item in structured_elements(document)
    )


def chunk_elements(items: Iterable[Element], *, max_characters: int = 2000) -> tuple[Chunk, ...]:
    """Group adjacent normalized elements without crossing the size limit."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    chunks: list[Chunk] = []
    current: list[Element] = []
    size = 0
    for item in items:
        addition = len(item.text) + (2 if current else 0)
        if current and size + addition > max_characters:
            chunks.append(_make_chunk(current))
            current, size = [], 0
        current.append(item)
        size += len(item.text) + (2 if len(current) > 1 else 0)
    if current:
        chunks.append(_make_chunk(current))
    return tuple(chunks)


def _make_chunk(items: list[Element]) -> Chunk:
    pages = tuple(dict.fromkeys(item.page_number for item in items))
    return Chunk(
        text="\n\n".join(item.text for item in items if item.text),
        page_numbers=pages,
        element_ids=tuple(item.element_id for item in items),
        metadata={"element_types": tuple(item.type for item in items)},
    )


class StructuredState(ClosingMixin):
    """Locally parsed document with extraction, form, annotation, and edit views.

    One state owner handles source-backed and synthetic snapshots.  The named
    subclasses remain as compatibility markers returned by the factories.
    """

    def __init__(self, document: PdfDocument | None, structured: Document) -> None:
        self.pdf = document
        self.structured = structured
        self.internal_capability_document: Any | None = None
        self.internal_capability_pages: dict[int, Any] = {}
        self.internal_owned_document: PdfDocument | None = None

    def _with_snapshot(self, structured: Document) -> StructuredState:
        """Return a same-variant state carrying ``structured`` as its snapshot."""
        if getattr(self, "pdf", None) is None:
            return SyntheticState(None, structured)
        return OpenedState(self.pdf, structured)

    @property
    def source_pdf(self) -> PdfDocument:
        """Return the source-backed engine document, when one exists."""
        source = getattr(self, "pdf", None)
        if source is None:
            raise ValueError("synthetic snapshots do not have a source PDF")
        return source

    @property
    def snapshot(self) -> Document:
        """Return the immutable structured snapshot used by compat projections."""
        return self.structured

    @classmethod
    def open(cls, source: PdfInput, *, pages: Any = None, password: str = "") -> StructuredState:
        document = open_source(source, password=password)
        if pages is None:
            structured = document.structured_document
        else:
            structured = document.extract(pages=pages)
        return OpenedState(document, structured)

    @classmethod
    def from_structured(cls, structured: Document) -> StructuredState:
        """Create a compatibility snapshot through the canonical engine."""
        document = PdfDocument.from_structured(structured)
        return OpenedState(document, document.structured_document)

    @classmethod
    def synthetic(cls, structured: Document) -> StructuredState:
        """Create a snapshot-only state with no source-backed engine document."""
        return SyntheticState(None, structured)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.structured.metadata

    @property
    def pages(self) -> tuple[Page, ...]:
        return self.structured.pages

    @property
    def elements(self) -> tuple[Element, ...]:
        base = list(elements(self.structured))
        for page in self.pages:
            base.extend(
                Element(
                    element_id=f"p{page.page_number}-link-{index}",
                    type="link",
                    text=link.text,
                    page_number=page.page_number,
                    bbox=link.bbox,
                    metadata={"url": link.url, "link_type": link.link_type},
                )
                for index, link in enumerate(page.links)
            )
        base.extend(self.images)
        return tuple(base)

    @property
    def forms(self) -> tuple[Any, ...]:
        return tuple(field for page in self.pages for field in page.form_fields)

    @property
    def form_fields(self) -> tuple[Any, ...]:
        """Alias matching common AcroForm terminology."""
        return self.forms

    @property
    def annotations(self) -> tuple[Any, ...]:
        return tuple(annotation for page in self.pages for annotation in page.annotations)

    @property
    def comments(self) -> tuple[Any, ...]:
        return tuple(
            annotation
            for annotation in self.annotations
            if (annotation.subtype or "").casefold() in {"text", "freetext", "popup"}
        )

    @property
    def highlights(self) -> tuple[Any, ...]:
        return tuple(
            annotation
            for annotation in self.annotations
            if (annotation.subtype or "").casefold() in {"highlight", "underline", "squiggly"}
        )

    @property
    def redactions(self) -> tuple[Any, ...]:
        return tuple(
            annotation
            for annotation in self.annotations
            if (annotation.subtype or "").casefold() == "redact"
        )

    @property
    def links(self) -> tuple[Any, ...]:
        return tuple(link for page in self.pages for link in page.links)

    @property
    def images(self) -> tuple[Element, ...]:
        """Expose decoded image records as normalized, page-owned elements."""
        if getattr(self, "pdf", None) is None:
            return ()
        return tuple(
            Element(
                element_id=f"p{item.source.page_number or 0}-image-{index}",
                type="image",
                text="",
                page_number=item.source.page_number or 0,
                bbox=(
                    (item.bbox.x0, item.bbox.y0, item.bbox.x1, item.bbox.y1)
                    if item.bbox is not None
                    else None
                ),
                metadata={
                    "width": item.width,
                    "height": item.height,
                    "channels": item.channels,
                    "color_model": item.color_model,
                    "alpha": item.alpha,
                    "data": item.data,
                },
            )
            for index, item in enumerate(project_document(self.source_pdf).images())
        )

    def chunks(self, *, max_characters: int = 2000) -> tuple[Chunk, ...]:
        return chunk_elements(self.elements, max_characters=max_characters)

    def to_elements(self) -> tuple[dict[str, Any], ...]:
        """Return normalized elements in an OSS-loader-friendly dictionary shape."""
        return tuple(item.to_dict() for item in self.elements)

    def to_chunks(self, *, max_characters: int = 2000) -> tuple[dict[str, Any], ...]:
        return tuple(item.to_dict() for item in self.chunks(max_characters=max_characters))

    def split(self, ranges: Iterable[tuple[int, int]]) -> tuple[bytes, ...]:
        """Serialize inclusive 1-based page ranges as independent PDFs."""
        outputs: list[bytes] = []
        for start, end in ranges:
            if start < 1 or end < start or end > len(self.pages):
                raise ValueError(f"invalid page range: {start}-{end}")
            outputs.append(
                serialize_document_to_pdf(
                    Document(
                        pages=self.pages[start - 1 : end],
                        metadata=self.structured.metadata,
                        schema_version=self.structured.schema_version,
                    )
                )
            )
        return tuple(outputs)

    @staticmethod
    def merge(documents: Sequence[StructuredState]) -> bytes:
        """Serialize several locally extracted documents as one PDF."""
        if not documents:
            raise ValueError("at least one document is required")
        pages = tuple(page for document in documents for page in document.pages)
        return serialize_document_to_pdf(Document(pages=pages))

    def edit(self) -> Any:
        return self.structured.edit()

    def engine_edit(self) -> Any:
        """Return the canonical engine editor for source-backed state."""
        if getattr(self, "pdf", None) is None:
            raise ValueError("synthetic snapshots do not have an engine editor")
        return self.source_pdf.edit()

    def capability_document(self) -> Any:
        """Return a v0 document adapter, including pending snapshot edits."""
        cached = getattr(self, "internal_capability_document", None)
        if cached is not None:
            return cached
        source = getattr(self, "pdf", None)
        if source is not None and self.structured is source.structured_document:
            document = source
        else:
            document = PdfDocument.from_structured(self.structured)
            self.internal_owned_document = document
        cached = project_document(document)
        self.internal_capability_document = cached
        return cached

    def capability_page(self, page_number: int) -> Any:
        """Return a canonical engine page, including pending snapshot edits."""
        if page_number < 1 or page_number > len(self.pages):
            raise IndexError(page_number)
        pages = getattr(self, "internal_capability_pages", None)
        if pages is None:
            pages = self.internal_capability_pages = {}
        if page_number not in pages:
            pages[page_number] = self.capability_document().page(page_number - 1)
        return pages[page_number]

    def insert_page(self, page: Page, position: int | None = None) -> StructuredState:
        """Return a copy with ``page`` inserted at a 1-based position."""
        if getattr(self, "pdf", None) is not None:
            editor = self.engine_edit()
            editor.insert_structured_page(position or len(self.pages) + 1, page)
            return self._with_snapshot(editor.commit_document())
        editor = self.structured.edit()
        editor.insert_page(position or len(self.pages) + 1, page)
        return self._with_snapshot(editor.commit())

    def delete_page(self, page_number: int) -> StructuredState:
        """Return a copy without the requested 1-based page."""
        editor = self.structured.edit()
        editor.delete_page(page_number)
        return self._with_snapshot(editor.commit())

    def update_metadata(self, values: Mapping[str, Any]) -> StructuredState:
        """Return a copy with document metadata updated."""
        if getattr(self, "pdf", None) is not None:
            editor = self.engine_edit()
            editor.set_metadata(dict(values))
            return self._with_snapshot(editor.commit_document())
        editor = self.structured.edit()
        editor.update_metadata(values)
        return self._with_snapshot(editor.commit())

    def replace_pages(self, pages: Sequence[Page]) -> StructuredState:
        """Return a snapshot with replacement pages through the engine editor."""
        if getattr(self, "pdf", None) is not None:
            editor = self.engine_edit()
            editor.replace_pages(pages)
            return self._with_snapshot(editor.commit_document())
        return self._with_snapshot(replace(self.structured, pages=tuple(pages)))

    def fill_form(self, name: str, value: str) -> StructuredState:
        """Set an AcroForm value in the structured representation."""
        if getattr(self, "pdf", None) is not None:
            editor = self.engine_edit()
            editor.update_form_field(name, value)
            return self._with_snapshot(editor.commit_document())
        changed = False
        pages: list[Page] = []
        for page in self.pages:
            fields = tuple(
                replace(field, value_text=value) if field.name == name else field
                for field in page.form_fields
            )
            changed |= fields != page.form_fields
            pages.append(replace(page, form_fields=fields))
        if not changed:
            raise KeyError(name)
        return self._with_snapshot(replace(self.structured, pages=tuple(pages)))

    def save_form_value(
        self,
        name: str,
        value: str,
        target: str | PathLike[str] | BytesIO,
    ) -> bytes:
        """Persist an AcroForm value with an incremental update."""
        return self.source_pdf.save_form_value(name, value, target)

    def save_annotation(
        self,
        index: int,
        target: str | PathLike[str] | BytesIO,
        *,
        contents: str | None = None,
    ) -> bytes:
        """Persist an annotation content update through an incremental revision."""
        return self.source_pdf.save_annotation(index, target, contents=contents)

    def save_link(
        self,
        index: int,
        target: str | PathLike[str] | BytesIO,
        *,
        destination: str,
    ) -> bytes:
        """Persist a link destination in an incremental revision."""
        return self.source_pdf.save_link(index, target, destination=destination)

    def save_redactions(
        self,
        target: str | PathLike[str] | BytesIO,
    ) -> bytes:
        """Persist redaction annotations; call ``apply_redactions`` for IR removal."""
        if not self.redactions:
            raise ValueError("document has no redaction annotations")
        pdf = self.source_pdf
        updates: dict[int, Any] = {}
        for record in (record for page in pdf.pages for record in page.get_annotations()):
            if (record.subtype or "").casefold() != "redact":
                continue
            reference = pdf.find_object_reference(record.dict)
            if reference is not None:
                updates[reference.object_number] = dict(record.dict)
        if not updates:
            raise ValueError("redaction annotations are not indirect PDF objects")
        return pdf.save_incremental(target, updates)

    def apply_redactions(self) -> StructuredState:
        """Remove structured text blocks whose boxes are covered by redactions."""
        redactions = tuple(annotation.bbox for annotation in self.redactions if annotation.bbox)

        def covered(box: tuple[float, float, float, float] | None) -> bool:
            if box is None:
                return False
            return any(
                box[0] >= redact[0]
                and box[1] >= redact[1]
                and box[2] <= redact[2]
                and box[3] <= redact[3]
                for redact in redactions
            )

        pages = tuple(
            replace(page, blocks=tuple(block for block in page.blocks if not covered(block.bbox)))
            for page in self.pages
        )
        return self.replace_pages(pages)

    def write_redacted(
        self,
        target: str | PathLike[str] | BytesIO,
        *,
        outlines: Sequence[Sequence[object]] | None = None,
        attachments: Mapping[str, bytes] | None = None,
    ) -> bytes:
        """Write a rebuilt PDF whose structured text under redactions is removed."""
        redacted = self.apply_redactions()
        data = serialize_document_to_pdf(
            redacted.structured, outlines=outlines, attachments=attachments
        )
        write_bytes(target, data)
        return data

    def stamp(
        self,
        text: str,
        *,
        page_number: int = 1,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> StructuredState:
        """Add a text stamp to a page and return the edited document."""
        page = self.pages[page_number - 1]
        order = max((element.order for element in page.elements), default=-1) + 1
        stamp = Block(
            order=order,
            kind=BlockKind.UNKNOWN,
            lines=(TextLine(text=text, bbox=bbox),),
            bbox=bbox,
            provenance=("compat.stamp",),
        )
        edited = replace(page, blocks=(*page.blocks, stamp))
        return self.replace_pages(
            tuple(
                edited if index == page_number - 1 else current
                for index, current in enumerate(self.pages)
            )
        )

    def overlay(
        self,
        text: str,
        *,
        page_number: int = 1,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> StructuredState:
        """High-level alias for applying a text overlay to a page."""
        return self.stamp(text, page_number=page_number, bbox=bbox)

    def save_incremental(
        self,
        target: str | PathLike[str] | BytesIO,
        objects: Mapping[int, Any],
        *,
        trailer: Mapping[object, object] | None = None,
    ) -> bytes:
        """Append caller-supplied high-level object updates to the source PDF."""
        return self.source_pdf.save_incremental(target, objects, trailer=trailer)

    def write(self, target: str | PathLike[str] | BytesIO | None = None) -> bytes:
        data = serialize_document_to_pdf(self.structured)
        if target is not None:
            write_bytes(target, data)
        return data

    def close(self) -> None:
        owned = getattr(self, "internal_owned_document", None)
        if owned is not None:
            owned.close()
            self.internal_owned_document = None
        source = getattr(self, "pdf", None)
        if source is not None:
            source.close()


class OpenedState(StructuredState):
    """Source-backed compatibility marker."""


class SyntheticState(StructuredState):
    """Snapshot-only compatibility marker."""


__all__ = (
    "Chunk",
    "Element",
    "OpenedState",
    "StructuredState",
    "SyntheticState",
    "chunk_elements",
    "elements",
)
