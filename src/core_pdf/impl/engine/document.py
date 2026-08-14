# SPDX-License-Identifier: AGPL-3.0-only
"""Public PDF document with parse lifecycle and structured views."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import replace
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from core_pdf.impl.engine.execution import RUNTIME, TaskScope
from core_pdf.impl.engine.layout import LayoutGeometryIssue, LayoutGeometrySummary
from core_pdf.impl.engine.parse import parse_document
from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument as SpecPdfDocument
from core_pdf.impl.engine.structured import (
    Annotation,
    Link,
)
from core_pdf.impl.engine.structured import (
    Document as StructuredDocument,
)
from core_pdf.impl.engine.structured import (
    Page as StructuredPage,
)
from core_pdf.impl.engine.writing.api import PdfDocumentWritingMixin
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
from core_pdf.impl.engine.writing.signatures import PdfSignaturePlan
from core_pdf.impl.exceptions import PdfDocumentClosedError, PdfParseError
from core_pdf.impl.models import (
    ImageRecord,
    PageScoped,
)
from core_pdf.impl.objects import PdfName, PdfReference, PdfString
from core_pdf.impl.types import PageSelection, PdfSource

if TYPE_CHECKING:
    from core_pdf.impl.engine.page import PdfPage as PdfPage


class DocumentOperation(AbstractContextManager["DocumentOperation"]):
    __slots__ = ("document", "released")

    def __init__(self, document: PdfDocument) -> None:
        self.document = document
        self.released = False

    @property
    def cancelled(self) -> bool:
        return self.document.internal_operation_cancelled.is_set()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.document.internal_release_operation()

    def __exit__(self, *internal_args: object) -> None:
        self.release()


class PdfDocument(PdfDocumentWritingMixin, SpecPdfDocument["PdfPage"]):
    """A thread-native PDF document backed by the v2 parse pipeline."""

    @classmethod
    def from_structured(cls, document: StructuredDocument) -> "PdfDocument":
        """Create an engine document from the canonical structured representation."""
        data = serialize_document_to_pdf(document)
        return cls.open(BytesIO(data))

    def __init__(
        self,
        source: PdfSource,
        password: str = "",
        *,
        recovery_scan_all_revisions: bool = True,
        legacy_pdfminer_text_operators: bool = False,
    ) -> None:
        self.internal_operation_lock = threading.RLock()
        self.internal_page_locks: dict[int, threading.RLock] = {}
        self.internal_operation_cancelled = threading.Event()
        self.internal_active_operations = 0
        self.internal_closing = False
        self.internal_document_extract_lock = threading.RLock()
        self.internal_extraction_generation = 0
        self.internal_extracted_documents: dict[tuple[int, ...], Any] = {}
        self.internal_extraction_flights: dict[tuple[int, tuple[int, ...]], Future[Any]] = {}
        super().__init__(
            source,
            password=password,
            recovery_scan_all_revisions=recovery_scan_all_revisions,
            legacy_pdfminer_text_operators=legacy_pdfminer_text_operators,
        )
        from core_pdf.impl.engine.page import PdfPage

        self.page_class = PdfPage

    @property
    def closed(self) -> bool:
        return self.internal_closing or super().closed

    @property
    def metadata(self) -> dict[str, object]:
        """Return the document metadata in the canonical high-level shape."""
        value = self.get_metadata()
        return dict(value) if isinstance(value, dict) else {}

    @property
    def outlines(self) -> tuple[Any, ...]:
        """Return resolved outline entries owned by the engine."""
        return tuple(self.iter_outlines())

    @property
    def attachments(self) -> tuple[Any, ...]:
        """Return embedded files without exposing the PDF object model."""
        return tuple(self.embedded_files())

    def _scoped_records(
        self,
        pages: PageSelection | None,
        per_page: Callable[["PdfPage"], Iterable[Any]],
    ) -> tuple[PageScoped[Any], ...]:
        """Fan a per-page extractor out over the selected pages as scoped records."""
        return tuple(
            self.internal_scope(page_index, page, record)
            for page_index, page in self.iter_selected_pages(pages)
            for record in per_page(page)
        )

    def extract_annotations(
        self, *, pages: PageSelection | None = None
    ) -> tuple[PageScoped[Any], ...]:
        return self._scoped_records(pages, lambda page: page.get_annotations())

    def extract_links(self, *, pages: PageSelection | None = None) -> tuple[PageScoped[Any], ...]:
        return self._scoped_records(pages, lambda page: page.get_links())

    def extract_form_fields(
        self, *, pages: PageSelection | None = None
    ) -> tuple[PageScoped[Any], ...]:
        return self._scoped_records(pages, lambda page: page.get_fields())

    def save_form_value(
        self, name: str, value: str, target: str | PathLike[str] | BytesIO
    ) -> bytes:
        records = [record for page in self.pages for record in page.get_fields()]
        record = next((item for item in records if item.name == name), None)
        if record is None:
            raise KeyError(name)
        reference = self.find_object_reference(record.dict)
        if reference is None:
            raise ValueError(f"form field {name!r} is not an indirect PDF object")
        updated = dict(cast(Any, record.dict))
        updated[PdfName.of("V")] = PdfString(value.encode("utf-8"))
        return self.save_incremental(target, {reference.object_number: updated})

    def save_annotation(
        self,
        index: int,
        target: str | PathLike[str] | BytesIO,
        *,
        contents: str | None = None,
    ) -> bytes:
        records = [record for page in self.pages for record in page.get_annotations()]
        try:
            record = records[index]
        except IndexError as exc:
            raise IndexError(index) from exc
        reference = self.find_object_reference(record.dict)
        if reference is None:
            raise ValueError("annotation is not an indirect PDF object")
        updated = dict(record.dict)
        if contents is not None:
            updated[PdfName.of("Contents")] = PdfString(contents.encode("utf-8"))
        return self.save_incremental(target, {reference.object_number: updated})

    def save_link(
        self,
        index: int,
        target: str | PathLike[str] | BytesIO,
        *,
        destination: str,
    ) -> bytes:
        records = [record for page in self.pages for record in page.get_links()]
        try:
            record = records[index]
        except IndexError as exc:
            raise IndexError(index) from exc
        reference = self.find_object_reference(record.dict)
        if reference is None:
            raise ValueError("link is not an indirect PDF object")
        updated = dict(cast(Any, record.dict))
        action = updated.get(PdfName.of("A"))
        action = dict(cast(Any, action)) if isinstance(action, dict) else {}
        action[PdfName.of("D")] = PdfString(destination.encode("utf-8"))
        updated[PdfName.of("A")] = action
        return self.save_incremental(target, {reference.object_number: updated})

    def find_object_reference(self, value: object) -> PdfReference | None:
        for key in self.xref:
            reference = PdfReference(key >> 16, key & 0xFFFF)
            try:
                candidate = self.resolver.resolve(reference)
            except (ValueError, PdfParseError):
                continue
            if candidate is value or candidate == value:
                return reference
        return None

    def acquire_operation(self) -> DocumentOperation:
        with self.internal_operation_lock:
            if self.closed:
                raise PdfDocumentClosedError("PDF document is closed")
            self.internal_active_operations += 1
        return DocumentOperation(self)

    def internal_release_operation(self) -> None:
        should_close = False
        with self.internal_operation_lock:
            self.internal_active_operations = max(0, self.internal_active_operations - 1)
            should_close = self.internal_closing and self.internal_active_operations == 0
        if should_close:
            super().close()

    def page_lock(self, page_number: int) -> threading.RLock:
        with self.internal_cache_lock:
            return self.internal_page_locks.setdefault(page_number, threading.RLock())

    def close(self) -> None:
        with self.internal_operation_lock:
            if self.internal_closing or super().closed:
                return
            self.internal_closing = True
            self.internal_operation_cancelled.set()
            if self.internal_active_operations:
                return
        super().close()

    def invalidate_document_extraction_cache(self) -> None:
        with self.internal_document_extract_lock:
            self.internal_extraction_generation += 1
            self.internal_extracted_documents.clear()
            super().invalidate_document_extraction_cache()

    def extract(
        self,
        *,
        pages: PageSelection | None = None,
        adapters: Iterable[Any] = (),
        context: TaskScope | None = None,
    ) -> Any:
        with self.acquire_operation() as operation:
            leader = False
            flight: Future[Any] | None = None
            selected_pages: tuple[Any, ...] = ()
            with self.internal_document_extract_lock:
                selected = tuple(self.selected_page_indexes(pages))
                generation = self.internal_extraction_generation
                flight_key = generation, selected
                result = self.internal_extracted_documents.get(selected)
                if result is None:
                    flight = self.internal_extraction_flights.get(flight_key)
                    if flight is None:
                        selected_pages = tuple(self.pages[index] for index in selected)
                        flight = Future()
                        self.internal_extraction_flights[flight_key] = flight
                        leader = True
            if result is None:
                assert flight is not None
                active_flight = flight
                if not leader:
                    result = active_flight.result()
                else:
                    try:
                        if context is None:
                            with RUNTIME.task_scope(
                                cancelled=lambda: operation.cancelled,
                                metrics=True,
                            ) as active_context:
                                result = parse_document(self, active_context, selected_pages)
                        else:
                            previous = context.internal_bind_cancelled(lambda: operation.cancelled)
                            try:
                                result = parse_document(self, context, selected_pages)
                            finally:
                                context.internal_restore_cancelled(previous)
                    except BaseException as exc:
                        active_flight.set_exception(exc)
                        raise
                    else:
                        with self.internal_document_extract_lock:
                            if generation == self.internal_extraction_generation:
                                self.internal_extracted_documents[selected] = result
                        active_flight.set_result(result)
                    finally:
                        with self.internal_document_extract_lock:
                            self.internal_extraction_flights.pop(flight_key, None)
        for adapter in adapters:
            result = adapter.apply(result)
        return result

    @property
    def structured_document(self) -> StructuredDocument:
        """Return the cached high-level structured view owned by this document."""
        if self.page_count() == 0:
            return StructuredDocument(metadata=self.metadata)
        return cast(StructuredDocument, self.extract())

    @staticmethod
    def internal_scope(page_index: int, page: Any, record: Any) -> PageScoped[Any]:
        return PageScoped(
            page_index=page_index,
            page_number=page_index + 1,
            page_label=page.label,
            record=record,
        )

    def extract_images(
        self,
        *,
        pages: PageSelection | None = None,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[PageScoped[ImageRecord], ...]:
        return self._scoped_records(
            pages,
            lambda page: page.extract_images(
                include_inline=include_inline,
                include_xobjects=include_xobjects,
            ),
        )

    def extract_geometry_issues(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> tuple[PageScoped[LayoutGeometryIssue], ...]:
        return self._scoped_records(pages, lambda page: page.extract_geometry_issues())

    def extract_geometry_summary(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> tuple[PageScoped[LayoutGeometrySummary], ...]:
        return self._scoped_records(pages, lambda page: (page.extract_geometry_summary(),))

    def edit(self) -> "PdfDocumentEditor":
        return PdfDocumentEditor(self)


class PdfDocumentEditor:
    """Transactional editor backed by the canonical structured document IR."""

    def __init__(self, document: PdfDocument) -> None:
        self.document = document
        self.internal_editor = document.extract().edit()
        self.internal_closed = False
        self.internal_attachments: dict[str, bytes] | None = None
        self.internal_outlines: tuple[tuple[object, ...], ...] | None = None
        self.internal_encryption: StandardPdfEncryption | None = None
        self.internal_signature: PdfSignaturePlan | None = None
        self.internal_geometry_updates: dict[int, tuple[int | None, tuple[float, ...] | None]] = {}
        self.internal_annotation_removals: dict[int, tuple[set[int] | None, set[int] | None]] = {}

    def encrypt(self, user_password: str, *, owner_password: str | None = None) -> Self:
        self.internal_ensure_active()
        if not user_password:
            raise ValueError("user password must not be empty")
        self.internal_encryption = StandardPdfEncryption(user_password, owner_password)
        return self

    def sign(self, provider: Any, *, contents_length: int = 8192) -> Self:
        self.internal_ensure_active()
        self.internal_signature = PdfSignaturePlan(provider, contents_length)
        return self

    def set_metadata(self, values: dict[str, object]) -> Self:
        self.internal_ensure_active()
        self.internal_editor.update_metadata(values)
        return self

    def set_page_geometry(
        self,
        page_number: int,
        *,
        rotation: int | None = None,
        cropbox: tuple[float, float, float, float] | None = None,
    ) -> Self:
        self.internal_ensure_active()
        if rotation is not None and rotation % 90 != 0:
            raise ValueError("page rotation must be a multiple of 90 degrees")
        if cropbox is not None and (cropbox[2] <= cropbox[0] or cropbox[3] <= cropbox[1]):
            raise ValueError("crop box must have positive width and height")
        page = self.internal_editor.internal_pages[page_number - 1]
        self.internal_editor.replace_page(
            page_number,
            replace(
                page,
                rotation=(rotation % 360) if rotation is not None else page.rotation,
                cropbox=cropbox if cropbox is not None else page.cropbox,
            ),
        )
        self.internal_geometry_updates[page_number] = (
            (rotation % 360) if rotation is not None else page.rotation,
            cropbox if cropbox is not None else page.cropbox,
        )
        return self

    def replace_page(self, page_number: int, page: StructuredPage) -> Self:
        self.internal_ensure_active()
        self.internal_editor.replace_page(page_number, page)
        return self

    def replace_pages(self, pages: Iterable[StructuredPage]) -> Self:
        self.internal_ensure_active()
        self.internal_editor.internal_pages = list(pages)
        return self

    def update_form_field(self, name: str, value: str) -> Self:
        self.internal_ensure_active()
        pages = []
        changed = False
        for page in self.internal_editor.internal_pages:
            fields = tuple(
                replace(field, value_text=value) if field.name == name else field
                for field in page.form_fields
            )
            changed |= fields != page.form_fields
            pages.append(replace(page, form_fields=fields))
        if not changed:
            raise KeyError(name)
        self.internal_editor.internal_pages = pages
        return self

    def remove_form_fields(self, names: Iterable[str]) -> Self:
        self.internal_ensure_active()
        selected = set(names)
        self.internal_editor.internal_pages = [
            replace(
                page,
                form_fields=tuple(
                    field for field in page.form_fields if field.name not in selected
                ),
            )
            for page in self.internal_editor.internal_pages
        ]
        return self

    def apply_redactions(
        self, redactions: Mapping[int, Iterable[tuple[float, float, float, float]]]
    ) -> Self:
        self.internal_ensure_active()

        def redact_line(
            line: Any,
            page_redactions: tuple[tuple[float, float, float, float], ...],
        ) -> Any | None:
            box = line.bbox
            if box is None:
                return line
            for redact in page_redactions:
                if (
                    box[0] >= redact[0]
                    and box[1] >= redact[1]
                    and box[2] <= redact[2]
                    and box[3] <= redact[3]
                ):
                    return None
            text = line.text
            if not text or box[2] <= box[0]:
                return line
            keep: list[str] = []
            char_width = (box[2] - box[0]) / len(text)
            changed = False
            for index, character in enumerate(text):
                x0 = box[0] + index * char_width
                x1 = x0 + char_width
                hidden = any(
                    x1 > redact[0] and x0 < redact[2] and box[3] > redact[1] and box[1] < redact[3]
                    for redact in page_redactions
                )
                if hidden:
                    changed = True
                else:
                    keep.append(character)
            if not changed:
                return line
            updated = "".join(keep)
            return replace(line, text=updated, spans=()) if updated else None

        pages = []
        for page in self.internal_editor.internal_pages:
            page_redactions = tuple(redactions.get(page.page_number, ()))
            updated_blocks = []
            for block in page.blocks:
                lines = tuple(
                    updated
                    for line in block.lines
                    if (updated := redact_line(line, page_redactions)) is not None
                )
                if lines:
                    updated_blocks.append(replace(block, lines=lines))
            pages.append(
                replace(
                    page,
                    blocks=tuple(updated_blocks),
                )
            )
        self.internal_editor.internal_pages = pages
        return self

    def remove_annotations(self, page_number: int, indices: Iterable[int] | None = None) -> Self:
        self.internal_ensure_active()
        page = self.internal_editor.internal_pages[page_number - 1]
        selected = set(indices) if indices is not None else None
        annotations = (
            ()
            if selected is None
            else tuple(
                annotation
                for index, annotation in enumerate(page.annotations)
                if index not in selected
            )
        )
        self.internal_editor.replace_page(page_number, replace(page, annotations=annotations))
        self.internal_record_removal(page_number, selected, links=False)
        return self

    def remove_links(self, page_number: int, indices: Iterable[int] | None = None) -> Self:
        self.internal_ensure_active()
        page = self.internal_editor.internal_pages[page_number - 1]
        selected = set(indices) if indices is not None else None
        links = (
            ()
            if selected is None
            else tuple(link for index, link in enumerate(page.links) if index not in selected)
        )
        self.internal_editor.replace_page(page_number, replace(page, links=links))
        self.internal_record_removal(page_number, selected, links=True)
        return self

    def internal_record_removal(
        self, page_number: int, indices: set[int] | None, *, links: bool
    ) -> None:
        annotations, link_indices = self.internal_annotation_removals.get(
            page_number, (set(), set())
        )
        if links:
            link_indices = None if indices is None else (link_indices or set()) | indices
        else:
            annotations = None if indices is None else (annotations or set()) | indices
        self.internal_annotation_removals[page_number] = (annotations, link_indices)

    def add_annotation(
        self,
        page_number: int,
        subtype: str,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        contents: str = "",
        destination: object = None,
    ) -> Self:
        self.internal_ensure_active()
        page = self.internal_editor.internal_pages[page_number - 1]
        self.internal_editor.replace_page(
            page_number,
            replace(
                page,
                annotations=(*page.annotations, Annotation(subtype, bbox, contents, destination)),
            ),
        )
        return self

    def add_link(
        self,
        page_number: int,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        url: str | None = None,
        link_type: str | None = None,
        text: str = "",
    ) -> Self:
        self.internal_ensure_active()
        page = self.internal_editor.internal_pages[page_number - 1]
        self.internal_editor.replace_page(
            page_number,
            replace(
                page,
                links=(*page.links, Link(bbox, url, link_type, text)),
            ),
        )
        return self

    def set_attachments(self, values: dict[str, bytes]) -> Self:
        self.internal_ensure_active()
        self.internal_attachments = dict(values)
        return self

    def set_outlines(self, values: Iterable[Iterable[object]]) -> Self:
        self.internal_ensure_active()
        self.internal_outlines = tuple(tuple(item) for item in values)
        return self

    def insert_page(
        self,
        position: int,
        width: float = 595.0,
        height: float = 842.0,
    ) -> Self:
        self.internal_ensure_active()
        self.internal_editor.insert_page(
            position,
            StructuredPage(page_number=position, width=width, height=height),
        )
        return self

    def insert_structured_page(self, position: int, page: StructuredPage) -> Self:
        self.internal_ensure_active()
        self.internal_editor.insert_page(position, page)
        return self

    def delete_pages(self, selection: PageSelection) -> Self:
        self.internal_ensure_active()
        page_numbers = sorted(
            (index + 1 for index in self.document.selected_page_indexes(selection)),
            reverse=True,
        )
        for page_number in page_numbers:
            self.internal_editor.delete_page(page_number)
        return self

    def commit(self, target: str | Path | BytesIO) -> bytes:
        self.internal_ensure_active()
        if self.internal_is_annotation_cleanup_only():
            updates: dict[int, Any] = {}
            for page_number, (
                annotation_indices,
                link_indices,
            ) in self.internal_annotation_removals.items():
                page = self.document.pages[page_number - 1]
                reference = self.document.find_object_reference(page.page_dict)
                if reference is None:
                    raise ValueError(f"page {page_number} is not an indirect PDF object")
                records = list(page.get_annotations())
                links = list(page.get_links())
                remove: set[int] = set()
                if annotation_indices is None:
                    remove.update(id(record.dict) for record in records)
                else:
                    remove.update(id(records[index].dict) for index in annotation_indices)
                if link_indices is None:
                    remove.update(id(record.dict) for record in links)
                else:
                    remove.update(id(links[index].dict) for index in link_indices)
                page_dict: Any = dict(page.page_dict)
                entries = page_dict.get(PdfName.of("Annots"), ())
                page_dict[PdfName.of("Annots")] = [
                    item
                    for item in entries
                    if id(self.document.resolver.resolve(item)) not in remove
                ]
                updates[reference.object_number] = page_dict
            data = self.document.save_incremental(target, updates)
            self.internal_closed = True
            return data
        if self.internal_is_geometry_only():
            geometry_updates: dict[int, Any] = {}
            for page_number, (rotation, cropbox) in self.internal_geometry_updates.items():
                page = self.document.pages[page_number - 1]
                reference = self.document.find_object_reference(page.page_dict)
                if reference is None:
                    raise ValueError(f"page {page_number} is not an indirect PDF object")
                geometry_page_dict: Any = dict(page.page_dict)
                geometry_page_dict[PdfName.of("Rotate")] = rotation or 0
                if cropbox is not None:
                    geometry_page_dict[PdfName.of("CropBox")] = list(cropbox)
                geometry_updates[reference.object_number] = geometry_page_dict
            data = self.document.save_incremental(target, geometry_updates)
            self.internal_closed = True
            return data
        if self.internal_is_noop():
            data = bytes(self.document.raw_data)
            if isinstance(target, (str, Path)):
                Path(target).write_bytes(data)
            else:
                target.write(data)
            self.internal_closed = True
            return data
        structured = self.commit_document()
        data = serialize_document_to_pdf(
            structured,
            encryption=self.internal_encryption,
            signature=self.internal_signature,
            attachments=self.internal_attachments,
            outlines=self.internal_outlines,
        )
        if isinstance(target, (str, Path)):
            Path(target).write_bytes(data)
        else:
            target.write(data)
        self.internal_closed = True
        return data

    def internal_is_noop(self) -> bool:
        return (
            self.internal_encryption is None
            and self.internal_signature is None
            and self.internal_attachments is None
            and self.internal_outlines is None
            and tuple(self.internal_editor.internal_pages)
            == tuple(self.document.structured_document.pages)
            and self.internal_editor.internal_metadata == self.document.structured_document.metadata
        )

    def internal_is_geometry_only(self) -> bool:
        return bool(self.internal_geometry_updates) and (
            self.internal_encryption is None
            and self.internal_signature is None
            and self.internal_attachments is None
            and self.internal_outlines is None
            and self.internal_editor.internal_metadata == self.document.structured_document.metadata
            and all(
                replace(
                    tuple(self.internal_editor.internal_pages)[page_number - 1],
                    rotation=self.document.structured_document.pages[page_number - 1].rotation,
                    cropbox=self.document.structured_document.pages[page_number - 1].cropbox,
                )
                == self.document.structured_document.pages[page_number - 1]
                for page_number in self.internal_geometry_updates
            )
        )

    def internal_is_annotation_cleanup_only(self) -> bool:
        if not self.internal_annotation_removals or self.internal_geometry_updates:
            return False
        if any(
            value is not None
            for value in (
                self.internal_encryption,
                self.internal_signature,
                self.internal_attachments,
                self.internal_outlines,
            )
        ):
            return False
        original = self.document.structured_document
        if self.internal_editor.internal_metadata != original.metadata:
            return False
        for current, source in zip(
            self.internal_editor.internal_pages, original.pages, strict=True
        ):
            normalized = replace(current, annotations=source.annotations, links=source.links)
            if normalized != source:
                return False
        return True

    def commit_document(self) -> StructuredDocument:
        """Commit the transaction to the canonical structured document IR."""
        self.internal_ensure_active()
        structured = cast(StructuredDocument, self.internal_editor.commit())
        self.internal_closed = True
        return structured

    def rollback(self) -> None:
        self.internal_ensure_active()
        self.internal_editor.rollback()
        self.internal_closed = True

    def internal_ensure_active(self) -> None:
        if self.internal_closed:
            raise RuntimeError("PDF editor is closed")


__all__ = ("DocumentOperation", "PdfDocument", "PdfDocumentEditor")
