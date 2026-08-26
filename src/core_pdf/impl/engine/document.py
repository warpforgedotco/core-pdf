# SPDX-License-Identifier: AGPL-3.0-only
"""Public PDF document with parse lifecycle and structured views."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from contextlib import AbstractContextManager
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from core_pdf.impl.engine.parse import parse_document
from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument as SpecPdfDocument
from core_pdf.impl.engine.structured import (
    Document as StructuredDocument,
)
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryptionContext
from core_pdf.impl.engine.writing.incremental import (
    append_incremental_update,
    find_startxref,
    previous_object_count,
)
from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
from core_pdf.impl.exceptions import (
    PdfDocumentClosedError,
    PdfUnsupportedError,
)
from core_pdf.impl.models import (
    ImageRecord,
    PageScoped,
)
from core_pdf.impl.pages import PageSelection
from core_pdf.impl.runtime.execution import RUNTIME, TaskScope
from core_pdf.impl.types import PdfSource

if TYPE_CHECKING:
    from core_pdf.impl.engine.page import PdfPage as PdfPage
    from core_pdf.impl.engine.spec.s_09_fonts.fallback import RasterFontProviderLike
    from core_pdf.impl.types import PdfObject


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


class PdfDocument(SpecPdfDocument["PdfPage"]):
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
        raster_font_provider: RasterFontProviderLike | None = None,
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
            raster_font_provider=raster_font_provider,
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

    def save_incremental(
        self,
        target: str | PathLike[str] | BinaryIO,
        objects: Mapping[int, PdfObject],
        *,
        trailer: Mapping[object, object] | None = None,
    ) -> bytes:
        """Append an incremental update and write it to ``target``."""
        if self.closed:
            raise PdfDocumentClosedError("PDF document is closed")
        original = bytes(self.raw_data)
        objects_to_write = objects
        if self.decipher is not None:
            handler = getattr(self.decipher, "__self__", None)
            if handler is None:
                raise PdfUnsupportedError("encrypted PDF security handler is unavailable")
            try:
                context = StandardPdfEncryptionContext.from_security_handler(handler)
            except ValueError as exc:
                raise PdfUnsupportedError(str(exc)) from exc
            objects_to_write = {
                number: cast(Any, context.encrypt_object(value, number))
                for number, value in objects.items()
            }
        base_trailer = cast(Mapping[object, object], self.trailer_dict)
        updated = append_incremental_update(
            original,
            objects_to_write,
            trailer={**base_trailer, **dict(trailer or {})},
            previous_xref_offset=find_startxref(original),
            previous_size=previous_object_count(self.xref, base_trailer),
        )
        if isinstance(target, (str, PathLike)):
            Path(cast(str | PathLike[str], target)).write_bytes(updated)
        else:
            target.write(updated)
        return updated

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

    def extract_form_fields(
        self, *, pages: PageSelection | None = None
    ) -> tuple[PageScoped[Any], ...]:
        return self._scoped_records(pages, lambda page: page.get_fields())

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


__all__ = ("DocumentOperation", "PdfDocument")
