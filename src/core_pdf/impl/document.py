# SPDX-License-Identifier: AGPL-3.0-only
"""Public PDF document and page APIs over the canonical extraction pipeline."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.exceptions import PdfContractError, PdfDocumentClosedError
from core_pdf.impl.extract.ocr.tesseract import internal_prepare_ocr_signals
from core_pdf.impl.extract.pipeline import extract_page
from core_pdf.impl.extract.selection import extract_document
from core_pdf.impl.layout.lines import (
    LayoutGeometrySummary,
    LayoutLine,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issues,
)
from core_pdf.impl.model.geometry import rect_tuple
from core_pdf.impl.output import DiagnosticTextRun, TextDiagnostics
from core_pdf.impl.output import Document as StructuredDocument
from core_pdf.impl.pages import PageSelection
from core_pdf.impl.records import (
    DrawingRecord,
    ImageMetadata,
    ImageRecord,
    PageScoped,
)
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from core_pdf.impl.runtime.cache import ExtractionCache
from core_pdf.impl.runtime.execution import RUNTIME, TaskScope
from core_pdf.impl.spec.s_07_document.document import PdfDocument as SpecPdfDocument
from core_pdf.impl.spec.s_07_document.page import PdfPage as SpecPdfPage
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.types import PdfSource

# Claim process signal ownership when the public document class is imported on
# the application's main thread. Extraction submodules remain side-effect free.
internal_prepare_ocr_signals()

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_09_fonts.fallback import RasterFontProviderLike


class PdfPage(SpecPdfPage):
    document: Any

    @property
    def structured_view(self) -> Any:
        """Return this page's canonical high-level structured representation."""
        return self.extract()

    def internal_cache(self) -> ExtractionCache:
        cache = self.extraction_cache
        if cache is None:
            # The spec page always registers a cache at construction; a missing one
            # would silently escape document-level invalidation, so fail loudly.
            raise PdfContractError("page extraction cache was not initialized")
        return cache

    def extract(self) -> Any:
        with self.document.acquire_operation() as operation:
            with RUNTIME.task_scope(
                cancelled=lambda: operation.cancelled,
            ) as context:
                return extract_page(self, context)

    def text_diagnostics(self, *, include_invisible: bool = True) -> TextDiagnostics:
        with self.internal_page_lock:
            return TextDiagnostics(
                runs=tuple(
                    DiagnosticTextRun(
                        text=run.text,
                        bbox=(run.x0, run.y0, run.x1, run.y1),
                        font_name=run.font_name,
                        font_size=run.font_size,
                        is_vertical=run.is_vertical,
                        visible=run.visible,
                        rotation=run.rotation_angle,
                        seqno=run.seqno,
                        geometry_issues=text_run_geometry_issues(run),
                    )
                    for run in self.get_page_program().runs
                    if include_invisible or run.visible
                )
            )

    def get_text_lines(self) -> list[LayoutLine]:
        with self.internal_page_lock:
            if self.text_lines is None:
                self.text_lines = [LayoutLine([run]) for run in self.chars if run.text]
            return self.text_lines

    def extract_geometry_issues(self) -> tuple[object, ...]:
        with self.internal_page_lock:
            return page_layout_geometry_issues(self.get_text_lines())

    def extract_geometry_summary(self) -> LayoutGeometrySummary:
        with self.internal_page_lock:
            return page_layout_geometry_summary(self.get_text_lines())

    def get_drawings(self) -> tuple[DrawingRecord, ...]:
        cache_key = "page_drawing_records_v2"
        with self.internal_page_lock:
            cache = self.internal_cache()
            cached = cache.get_as(cache_key, tuple)
            if cached is not None:
                return cached
            result = tuple(
                DrawingRecord.from_captured(
                    drawing,
                    raw_data=bytes(drawing.raw_data) if drawing.raw_data is not None else None,
                    image_clip=rect_tuple(drawing.image_clip),
                    items=tuple(drawing.items),
                    rect=rect_tuple(drawing.rect),
                )
                for drawing in self.get_page_program().drawings
            )
            cache[cache_key] = result
            return result

    def extract_images(
        self,
        *,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[ImageRecord, ...]:
        images: list[ImageRecord] = []
        if include_xobjects:
            images.extend(
                ImageRecord.from_captured(drawing)
                for drawing in self.get_drawings()
                if drawing.kind == "image"
            )
        if include_inline:
            images.extend(
                ImageRecord(
                    kind="inline-image",
                    seqno=image.seqno,
                    fill=None,
                    fill_pattern=None,
                    fill_opacity=None,
                    stroke_color=None,
                    stroke_pattern=None,
                    stroke_opacity=None,
                    line_width=1.0,
                    line_cap=0,
                    line_join=0,
                    dash_pattern=None,
                    fill_rule="nonzero",
                    blend_mode=None,
                    soft_mask_alpha=None,
                    raw_data=image.data,
                    dictionary=image.dictionary,
                    image_source=image.image_source,
                    image_clip=image.image_clip,
                    path=None,
                    items=(),
                    rect=None,
                )
                for image in self.get_page_program().inline_images
            )
        for index, image in enumerate(images):
            source = cast(ImageSource | None, image.image_source)
            raster = source.decode() if source is not None else None
            if raster is not None:
                images[index] = replace(
                    image,
                    data=raster,
                    image_metadata=ImageMetadata(
                        width=raster.width,
                        height=raster.height,
                        channels=raster.channels,
                        color_model=raster.color_model,
                        alpha=raster.has_alpha,
                        stride=raster.stride,
                        source_rect=(0.0, 0.0, raster.width, raster.height),
                        transform=None,
                        clipping=rect_tuple(image.image_clip),
                    ),
                )
        return tuple(images)

    def render(self, options: RenderOptions | None = None) -> Any:
        options = options or RenderOptions()
        key = (
            "rendered_page_v2",
            options.rotate,
            options.crop,
            options.include_text,
            options.include_annotations,
            options.include_layers,
        )
        with self.internal_page_lock:
            cache = self.internal_cache()
            cached = cache.get(key)
            if cached is None:
                cached = compose_page(self, options, page_program=self.get_page_program())
                cache[key] = cached
            return cached


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
                            ) as active_context:
                                result = extract_document(self, active_context, selected_pages)
                        else:
                            result = extract_document(
                                self,
                                context.with_cancellation(lambda: operation.cancelled),
                                selected_pages,
                            )
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


__all__ = (
    "DocumentOperation",
    "PdfDocument",
    "PdfPage",
)
