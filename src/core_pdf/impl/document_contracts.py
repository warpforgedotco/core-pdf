# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol

from core_pdf.impl.pdf_values import ReferenceResolver

if TYPE_CHECKING:
    import mmap
    import threading

    from core_pdf.impl.caches import DocumentCaches
    from core_pdf.impl.capture_recording import TextState
    from core_pdf.impl.document_metadata import MetadataRecord
    from core_pdf.impl.document_navigation import PageLookup
    from core_pdf.impl.document_page import PdfPage
    from core_pdf.impl.document_records import RawFormField
    from core_pdf.impl.document_source import DocumentOperation
    from core_pdf.impl.document_structure import StructureTree
    from core_pdf.impl.fonts_fallback import RasterFontProviderLike, RasterFontRepository
    from core_pdf.impl.recovery_policy import Recovery
    from core_pdf.impl.recovery_resolver import ObjectResolver
    from core_pdf.impl.types import PdfSource
    from core_pdf_spec.s_07_syntax.types import Decipher, PdfDict
    from core_pdf_spec.s_07_syntax.xref import PdfXRefEntry
    from core_pdf_spec.standards import DocumentStandards, SemanticContext


class DocumentState[PageT: PdfPage]:
    __slots__ = (
        "source",
        "raw_data",
        "xref",
        "trailer_dict",
        "decipher",
        "resolver",
        "file_handle",
        "xref_was_recovered",
        "xref_recovery_reason",
        "recovery_scan_all_revisions",
        "raster_font_provider",
        "page_tree_was_recovered",
        "_closed",
        "closing",
        "operation_lock",
        "operation_cancelled",
        "active_operations",
        "_standards",
        "standards_complete",
        "font_decoders",
        "caches",
        "pending_parsed_objects",
    )

    source: PdfSource
    raw_data: bytes | mmap.mmap
    xref: dict[int, PdfXRefEntry]
    trailer_dict: PdfDict
    decipher: Decipher | None
    resolver: ObjectResolver
    file_handle: BinaryIO | None
    xref_was_recovered: bool
    xref_recovery_reason: str | None
    recovery_scan_all_revisions: bool
    raster_font_provider: RasterFontProviderLike | RasterFontRepository | None
    page_tree_was_recovered: bool
    _closed: bool
    closing: bool
    operation_lock: threading.RLock
    operation_cancelled: threading.Event
    active_operations: int
    _standards: DocumentStandards
    standards_complete: bool
    font_decoders: dict[object, object]
    caches: DocumentCaches
    pending_parsed_objects: tuple[SemanticContext | None, dict[int, object]] | None

    if TYPE_CHECKING:

        def catalog(self) -> PdfDict: ...

        def catalog_dict(self, key: str, *, recoverable: bool = False) -> PdfDict | None: ...

        def malformed(self, message: str) -> None: ...

        @property
        def recovery(self) -> Recovery: ...

        @property
        def recovery_enabled(self) -> bool: ...

        @property
        def pages(self) -> tuple[PageT, ...]: ...

        @property
        def page_lookup(self) -> PageLookup[PageT]: ...

        def scan_xref(self) -> None: ...


class ResolverHost(Protocol):
    @property
    def resolver(self) -> ObjectResolver: ...


class CaptureDocument(ResolverHost, ReferenceResolver, Protocol):
    pass


class LayeredDocument(CaptureDocument, Protocol):
    def oc_hidden_layers(self) -> frozenset[str]: ...


class CapturePage(Protocol):
    @property
    def document(self) -> LayeredDocument: ...

    @property
    def resources(self) -> PdfDict: ...

    def annotation_dicts(self) -> list[PdfDict]: ...

    def get_fields(self) -> list[RawFormField]: ...

    def effective_page_clip(self) -> tuple[float, float, float, float] | None: ...

    def consume_contents(self, state: TextState) -> None: ...


class StructureHost(ResolverHost, Protocol):
    @property
    def recovery(self) -> Recovery: ...

    @property
    def structure(self) -> StructureTree | None: ...

    @property
    def pages(self) -> Sequence[PdfPage]: ...

    def page_index_for(self, page_obj: object) -> int | None: ...


class PageHost(LayeredDocument, StructureHost, Protocol):
    @property
    def recovery_enabled(self) -> bool: ...

    def cached_fields_by_page(self) -> dict[int, list[RawFormField]]: ...

    def page_label(self, page_index: int) -> str | None: ...

    def acquire_operation(self) -> DocumentOperation: ...


class ExtractionDocument(LayeredDocument, StructureHost, Protocol):
    def fields_by_page(
        self, pages: Sequence[Any] | None = None
    ) -> dict[int, list[RawFormField]]: ...

    def get_metadata(self) -> MetadataRecord: ...


__all__ = (
    "CaptureDocument",
    "CapturePage",
    "DocumentState",
    "ExtractionDocument",
    "LayeredDocument",
    "PageHost",
    "ResolverHost",
    "StructureHost",
)
