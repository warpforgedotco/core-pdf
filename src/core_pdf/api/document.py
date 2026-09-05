# SPDX-License-Identifier: AGPL-3.0-only
"""Public PDF document and page entry points over the internal processing engine."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.exceptions import PdfDocumentClosedError
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
from core_pdf.impl.model.page_selection import PageSelection
from core_pdf.impl.output.model import DiagnosticTextRun, TextDiagnostics
from core_pdf.impl.output.model import Document as StructuredDocument
from core_pdf.impl.records import (
    DrawingRecord,
    ImageMetadata,
    ImageRecord,
    PageScoped,
)
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from core_pdf.impl.runtime.execution import ExtractionScope
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

    def extract(self) -> Any:
        with self.document.acquire_operation() as operation:
            context = ExtractionScope(cancelled=lambda: operation.cancelled)
            return extract_page(self, context)

    def text_diagnostics(self, *, include_invisible: bool = True) -> TextDiagnostics:
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
        return [LayoutLine([run]) for run in self.chars if run.text]

    def extract_geometry_issues(self) -> tuple[object, ...]:
        return page_layout_geometry_issues(self.get_text_lines())

    def extract_geometry_summary(self) -> LayoutGeometrySummary:
        return page_layout_geometry_summary(self.get_text_lines())

    @staticmethod
    def internal_drawing_records(drawings: Iterable[Any]) -> tuple[DrawingRecord, ...]:
        return tuple(
            DrawingRecord.from_captured(
                drawing,
                raw_data=bytes(drawing.raw_data) if drawing.raw_data is not None else None,
                image_clip=rect_tuple(drawing.image_clip),
                items=tuple(drawing.items),
                rect=rect_tuple(drawing.rect),
            )
            for drawing in drawings
        )

    def get_drawings(self) -> tuple[DrawingRecord, ...]:
        return self.internal_drawing_records(self.get_page_program().drawings)

    def extract_images(
        self,
        *,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[ImageRecord, ...]:
        if not include_inline and not include_xobjects:
            return ()
        program = self.get_page_program()
        images: list[ImageRecord] = []
        if include_xobjects:
            images.extend(
                ImageRecord.from_captured(drawing)
                for drawing in self.internal_drawing_records(program.drawings)
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
                    blend_mode=image.blend_mode,
                    soft_mask_alpha=image.soft_mask_alpha,
                    raw_data=image.data,
                    dictionary=image.dictionary,
                    image_source=image.image_source,
                    image_clip=image.image_clip,
                    path=None,
                    items=(),
                    rect=None,
                )
                for image in program.inline_images
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
        return compose_page(self, options, page_program=self.get_page_program())


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
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> None:
        self.internal_operation_lock = threading.RLock()
        self.internal_operation_cancelled = threading.Event()
        self.internal_active_operations = 0
        self.internal_closing = False
        super().__init__(
            source,
            password=password,
            recovery_scan_all_revisions=recovery_scan_all_revisions,
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

    def internal_scoped_pending[RecordT](
        self,
        pending: Iterable[tuple[int, RecordT]],
    ) -> tuple[PageScoped[RecordT], ...]:
        """Attach page numbers and labels after collecting the selected records."""
        pending = tuple(pending)
        if not pending:
            return ()
        labels = self.page_labels
        return tuple(
            PageScoped(
                page_index=page_index,
                page_number=page_index + 1,
                page_label=labels[page_index] if labels is not None else None,
                record=record,
            )
            for page_index, record in pending
        )

    def extract_form_fields(
        self, *, pages: PageSelection | None = None
    ) -> tuple[PageScoped[Any], ...]:
        selected = tuple(self.iter_selected_pages(pages))
        grouped = self.fields_by_page(tuple(page for _index, page in selected))
        return self.internal_scoped_pending(
            (page_index, record)
            for page_index, _page in selected
            for record in grouped.get(page_index, ())
        )

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

    def extract(
        self,
        *,
        pages: PageSelection | None = None,
        adapters: Iterable[Any] = (),
    ) -> Any:
        with self.acquire_operation() as operation:
            selected_pages = tuple(page for _index, page in self.iter_selected_pages(pages))
            context = ExtractionScope(cancelled=lambda: operation.cancelled)
            result = extract_document(self, context, selected_pages)
        for adapter in adapters:
            result = adapter.apply(result)
        return result

    @property
    def structured_document(self) -> StructuredDocument:
        """Return the high-level structured view of this document."""
        if self.page_count() == 0:
            return StructuredDocument(metadata=self.metadata)
        return cast(StructuredDocument, self.extract())

    def extract_images(
        self,
        *,
        pages: PageSelection | None = None,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[PageScoped[ImageRecord], ...]:
        return self.internal_scoped_pending(
            (page_index, record)
            for page_index, page in self.iter_selected_pages(pages)
            for record in page.extract_images(
                include_inline=include_inline,
                include_xobjects=include_xobjects,
            )
        )


__all__ = (
    "DocumentOperation",
    "PdfDocument",
    "PdfPage",
)
