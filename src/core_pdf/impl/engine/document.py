# SPDX-License-Identifier: AGPL-3.0-only
"""Public PDF document with parse lifecycle and structured views."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.engine.execution import RUNTIME, TaskScope
from core_pdf.impl.engine.layout import LayoutGeometryIssue, LayoutGeometrySummary
from core_pdf.impl.engine.parse import parse_document
from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument as SpecPdfDocument
from core_pdf.impl.engine.writing.api import PdfDocumentWritingMixin
from core_pdf.impl.models import (
    ImageRecord,
    LineRecord,
    PageScoped,
    TableRecord,
    TextRunRecord,
    WordRecord,
)
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

    def internal_iter_public_pages(
        self, pages: PageSelection | None = None
    ) -> Iterator[tuple[int, Any]]:
        return cast(Iterator[tuple[int, Any]], self.iter_selected_pages(pages))

    def __init__(self, source: PdfSource, password: str = "") -> None:
        self.internal_operation_lock = threading.RLock()
        self.internal_page_locks: dict[int, threading.RLock] = {}
        self.internal_operation_cancelled = threading.Event()
        self.internal_active_operations = 0
        self.internal_closing = False
        self.internal_document_extract_lock = threading.RLock()
        self.internal_extraction_generation = 0
        self.internal_extracted_documents: dict[tuple[int, ...], Any] = {}
        self.internal_extraction_flights: dict[tuple[int, tuple[int, ...]], Future[Any]] = {}
        super().__init__(source, password=password)
        from core_pdf.impl.engine.page import PdfPage

        self.page_class = PdfPage

    @property
    def closed(self) -> bool:
        return self.internal_closing or super().closed

    def acquire_operation(self) -> DocumentOperation:
        with self.internal_operation_lock:
            if self.closed:
                raise ValueError("PDF document is closed")
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
            with self.internal_cache_lock:
                self.page_extraction_caches = None
                for page in tuple(self.pages_cache or ()):
                    if page.extraction_cache is not None:
                        page.extraction_cache.clear()

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

    def extract_structured(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> dict[str, object]:
        selected = self.selected_page_indexes(pages)
        document = self.extract(pages=selected)
        return {
            "schema_version": document.schema_version,
            "document": document.to_json_dict(),
            "metadata": self.get_metadata(),
            "page_count": self.page_count(),
            "summary": {
                "page_count": self.page_count(),
                "selected_page_count": len(document.pages),
                "selected_pages": [index + 1 for index in selected],
            },
        }

    @staticmethod
    def internal_scope(page_index: int, page: Any, record: Any) -> PageScoped[Any]:
        return PageScoped(
            page_index=page_index,
            page_number=page_index + 1,
            page_label=page.label,
            record=record,
        )

    def extract_text_runs(
        self,
        *,
        pages: PageSelection | None = None,
        include_invisible: bool = True,
    ) -> tuple[PageScoped[TextRunRecord], ...]:
        return tuple(
            self.internal_scope(page_index, page, record)
            for page_index, page in self.internal_iter_public_pages(pages)
            for record in page.extract_text_runs(include_invisible=include_invisible)
        )

    def extract_words(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> tuple[PageScoped[WordRecord], ...]:
        return tuple(
            self.internal_scope(page_index, page, record)
            for page_index, page in self.internal_iter_public_pages(pages)
            for record in page.extract_words()
        )

    def extract_lines(
        self,
        *,
        pages: PageSelection | None = None,
        include_words: bool = False,
    ) -> tuple[PageScoped[LineRecord], ...]:
        return tuple(
            self.internal_scope(page_index, page, record)
            for page_index, page in self.internal_iter_public_pages(pages)
            for record in page.extract_lines(include_words=include_words)
        )

    def extract_images(
        self,
        *,
        pages: PageSelection | None = None,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[PageScoped[ImageRecord], ...]:
        return tuple(
            self.internal_scope(page_index, page, record)
            for page_index, page in self.internal_iter_public_pages(pages)
            for record in page.extract_images(
                include_inline=include_inline,
                include_xobjects=include_xobjects,
            )
        )

    def extract_geometry_issues(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> tuple[PageScoped[LayoutGeometryIssue], ...]:
        return tuple(
            self.internal_scope(page_index, page, record)
            for page_index, page in self.internal_iter_public_pages(pages)
            for record in page.extract_geometry_issues()
        )

    def extract_geometry_summary(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> tuple[PageScoped[LayoutGeometrySummary], ...]:
        return tuple(
            self.internal_scope(page_index, page, page.extract_geometry_summary())
            for page_index, page in self.internal_iter_public_pages(pages)
        )

    def extract_tables(
        self,
        *,
        pages: PageSelection | None = None,
    ) -> tuple[PageScoped[TableRecord], ...]:
        records: list[PageScoped[TableRecord]] = []
        for page_index, page in self.internal_iter_public_pages(pages):
            payload = page.table_extraction_payload()
            rows_value = payload.get("tables", [])
            spans_value = payload.get("spans", [])
            bboxes_value = payload.get("bboxes", [])
            rows = rows_value if isinstance(rows_value, list) else []
            spans = spans_value if isinstance(spans_value, list) else []
            bboxes = bboxes_value if isinstance(bboxes_value, list) else []
            for table_index, table_rows in enumerate(rows):
                records.append(
                    self.internal_scope(
                        page_index,
                        page,
                        TableRecord(
                            table_index=table_index,
                            rows=tuple(tuple(cell for cell in row) for row in table_rows),
                            spans=tuple(
                                tuple(
                                    (
                                        int(span.get("row_span", 1)),
                                        int(span.get("col_span", 1)),
                                    )
                                    for span in row
                                )
                                for row in (spans[table_index] if table_index < len(spans) else [])
                            ),
                            bbox=bboxes[table_index] if table_index < len(bboxes) else None,
                        ),
                    )
                )
        return tuple(records)

    def to_structured_json_string(
        self,
        *,
        pages: PageSelection | None = None,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        return json.dumps(
            self.extract_structured(pages=pages),
            indent=indent,
            sort_keys=sort_keys,
        )

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return self.extract().to_json(indent=indent, sort_keys=sort_keys)

    def to_html(self) -> str:
        return self.extract().to_html()

    def to_markdown(self) -> str:
        return self.extract().to_markdown()


__all__ = ("DocumentOperation", "PdfDocument")
