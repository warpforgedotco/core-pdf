# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import mmap
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, BinaryIO, Generic, Protocol, Self, TypeVar, cast

from core_pdf.impl.document_forms import DocumentForms
from core_pdf.impl.document_metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.document_navigation import (
    DocumentNavigation,
    PageLookup,
    first_indexes,
    unresolved_destination,
)
from core_pdf.impl.document_optional_content import OptionalContent
from core_pdf.impl.document_page import PdfPage
from core_pdf.impl.document_page_tree import MAX_PAGE_TREE_DEPTH, PageTreeRecovery
from core_pdf.impl.document_security import SecuritySetupMixin, check_security_aliases
from core_pdf.impl.document_source import DocumentLifecycle, DocumentOperation, load_source
from core_pdf.impl.document_standards import (
    discover_profile_claims,
)
from core_pdf.impl.document_structure import StructureTree
from core_pdf.impl.document_xref_recovery import (
    TRAILER_METADATA_KEYS,
    XRefRecovery,
    object_headers_present,
)
from core_pdf.impl.exceptions import (
    PdfEmptySourceError,
    PdfParseError,
)
from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract_selection import extract_document
from core_pdf.impl.fonts_fallback import RasterFontRepository
from core_pdf.impl.memo import DocumentCaches
from core_pdf.impl.output_model import Document as StructuredDocument
from core_pdf.impl.page_selection import PageSelection, resolve_page_selection
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_policy import LENIENT, STRICT, Recovery
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.recovery_text_strings import parse_text_string
from core_pdf.impl.recovery_trees import iter_number_tree_items
from core_pdf.impl.types import (
    ImageRecord,
    PageScoped,
    PdfName,
    PdfSource,
)
from core_pdf_spec.s_07_document.document_labels import PageLabelStyle
from core_pdf_spec.s_07_document.document_labels import (
    format_page_label as format_spec_page_label,
)
from core_pdf_spec.s_07_document.page import PageNode as PageNode
from core_pdf_spec.s_07_syntax.types import (
    Decipher,
    PdfDict,
)
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
)
from core_pdf_spec.standards import DocumentStandards, SemanticContext

if TYPE_CHECKING:
    from core_pdf.impl.fonts_fallback import RasterFontProviderLike


PageT = TypeVar("PageT", bound=PdfPage, default=PdfPage)


class DocumentAdapter(Protocol):
    def apply(self, document: StructuredDocument, /) -> StructuredDocument: ...


class PdfDocument(
    DocumentLifecycle,
    XRefRecovery,
    PageTreeRecovery,
    DocumentNavigation[PageT],
    DocumentForms[PageT],
    OptionalContent,
    SecuritySetupMixin,
    Generic[PageT],
):
    page_class: type[PdfPage] = PdfPage

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

    def __init__(
        self,
        source: PdfSource,
        password: str = "",
        *,
        recovery_scan_all_revisions: bool = True,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> None:
        self._closed = False
        self.closing = False
        self.operation_lock = threading.RLock()
        self.operation_cancelled = threading.Event()
        self.active_operations = 0
        self.source = source
        self.file_handle = None
        self.raw_data = b""
        self.decipher = None
        self.xref = {}
        self.trailer_dict = {}
        self.xref_was_recovered = False
        self.xref_recovery_reason = None
        self.recovery_scan_all_revisions = recovery_scan_all_revisions
        self.raster_font_provider = RasterFontRepository(raster_font_provider)
        self.page_tree_was_recovered = False
        self._standards = DocumentStandards()
        self.standards_complete = False
        self.font_decoders = {}
        self.pending_parsed_objects = None
        self.caches = DocumentCaches()
        try:
            self.raw_data, self.file_handle = load_source(source)
            if not len(self.raw_data):
                raise PdfEmptySourceError("PDF source is empty")
            self.negotiate_security(password)
        except BaseException:
            self.close()
            raise

    @property
    def font_semantic_context(self) -> SemanticContext | None:
        return self.resolver.semantic_context

    @classmethod
    def open(
        cls,
        source: PdfSource,
        password: str = "",
        *,
        recovery_scan_all_revisions: bool = True,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> Self:
        return cls(
            source,
            password=password,
            recovery_scan_all_revisions=recovery_scan_all_revisions,
            raster_font_provider=raster_font_provider,
        )

    def resolve(self, ref: object) -> object:
        return self.resolver.resolve(ref)

    def catalog(self) -> PdfDict:
        root_ref = self.trailer_dict.get("Root")
        if root_ref is None:
            raise ValueError("missing catalog root")
        root = self.resolver.as_dict(root_ref)
        if root is None:
            raise ValueError("invalid catalog root")
        return root

    def get_metadata(self) -> MetadataRecord:
        return resolve_metadata(
            self.resolver,
            self.trailer_dict,
            recover=self.recovery,
        )

    def get_standards(self) -> DocumentStandards:
        with self.resolver.lock:
            if not self.standards_complete:
                self._standards = discover_profile_claims(
                    self._standards, self.resolver, self.trailer_dict
                )
                self.standards_complete = True
            return self._standards

    def catalog_dict(self, key: str, *, recoverable: bool = False) -> PdfDict | None:
        value = self.resolver.resolve(self.catalog().get(key))
        if value is None or isinstance(value, dict):
            return value
        if recoverable:
            self.malformed(f"invalid {key} dictionary")
            return None
        raise ValueError(f"invalid {key} dictionary")

    @property
    def structure(self) -> StructureTree | None:
        return self.caches.get("structure", self.build_structure)

    def build_structure(self) -> StructureTree | None:
        root = self.catalog_dict("StructTreeRoot")
        return None if root is None else StructureTree(self, root, page_lookup=self.page_lookup)

    @property
    def recovery_enabled(self) -> bool:
        return self.xref_was_recovered or self.page_tree_was_recovered

    @property
    def recovery(self) -> Recovery:
        return LENIENT if self.recovery_enabled else STRICT

    def malformed(self, message: str) -> None:
        self.recovery.malformed(message)

    def page_count(self) -> int:
        if not self.page_tree_was_recovered:
            try:
                count = self.resolver.resolve(self.page_tree_root().get("Count"))
                if type(count) is int and count >= 0:
                    return count
            except PdfParseError, ValueError:
                pass
        return len(self.pages)

    @property
    def metadata(self) -> dict[str, object]:
        value = self.get_metadata()
        return dict(value) if isinstance(value, dict) else {}

    @property
    def standards(self) -> DocumentStandards:
        with self.acquire_operation():
            return self.get_standards()

    def scoped_pending[RecordT](
        self,
        pending: Iterable[tuple[int, RecordT]],
    ) -> tuple[PageScoped[RecordT], ...]:
        pending = tuple(pending)
        if not pending:
            return ()
        labels = self.cached_page_labels()
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
        return self.scoped_pending(
            (page_index, record)
            for page_index, _page in selected
            for record in grouped.get(page_index, ())
        )

    def extract(
        self,
        *,
        pages: PageSelection | None = None,
        adapters: Iterable[DocumentAdapter] = (),
    ) -> StructuredDocument:
        with self.acquire_operation() as operation:
            selected_pages = tuple(page for _index, page in self.iter_selected_pages(pages))
            context = ExtractionScope(cancelled=lambda: operation.cancelled)
            result = self.run_extract_document(context, selected_pages)
        for adapter in adapters:
            result = adapter.apply(result)
        return result

    def run_extract_document(
        self, context: ExtractionScope, pages: Sequence[PdfPage]
    ) -> StructuredDocument:
        return extract_document(self, context, pages)

    @property
    def structured_document(self) -> StructuredDocument:
        if self.page_count() == 0:
            return StructuredDocument(metadata=self.metadata)
        return self.extract()

    def extract_images(
        self,
        *,
        pages: PageSelection | None = None,
        include_inline: bool = True,
        include_xobjects: bool = True,
    ) -> tuple[PageScoped[ImageRecord], ...]:
        return self.scoped_pending(
            (page_index, record)
            for page_index, page in self.iter_selected_pages(pages)
            for record in page.extract_images(
                include_inline=include_inline,
                include_xobjects=include_xobjects,
            )
        )

    @property
    def pages(self) -> tuple[PageT, ...]:
        try:
            return self.caches.content["pages"]
        except KeyError:
            return self.caches.get("pages", self.build_page_tree)

    def build_page_tree(self) -> tuple[PageT, ...]:
        return self.build_pages(self.iter_recovered_page_nodes())

    def build_pages(self, nodes: Iterable[PageNode]) -> tuple[PageT, ...]:
        factory = cast("type[PageT]", self.page_class)
        return tuple(
            factory(
                self,
                page_node.dictionary,
                page_number,
                inherited_values=page_node.inherited_values,
            )
            for page_number, page_node in enumerate(nodes, 1)
        )

    @property
    def page_labels(self) -> list[str] | None:
        labels = self.cached_page_labels()
        return None if labels is None else list(labels)

    def cached_page_labels(self) -> tuple[str, ...] | None:
        return self.caches.get("page_labels", self.build_page_label_tuple)

    def build_page_label_tuple(self) -> tuple[str, ...] | None:
        labels = self.build_page_labels()
        return None if labels is None else tuple(labels)

    def page_label(self, page_index: int) -> str | None:
        labels = self.cached_page_labels()
        if labels is None or page_index < 0 or page_index >= len(labels):
            return None
        return labels[page_index]

    def build_page_labels(self, *, page_count: int | None = None) -> list[str] | None:
        try:
            labels_root = self.resolve(self.catalog().get("PageLabels"))
        except ValueError as error:
            return self.recovery.reject(error, "page-labels", None)
        if labels_root is None:
            return None
        if not isinstance(labels_root, dict):
            raise ValueError("invalid PageLabels number tree")

        specs = [
            (page_index, spec)
            for page_index, spec in iter_number_tree_items(
                labels_root,
                self.resolve,
                on_malformed=self.recovery.malformed,
            )
            if isinstance(spec, dict)
        ]
        if not specs:
            return None
        specs.sort(key=lambda item: item[0])
        if specs[0][0] != 0:
            self.malformed("PageLabels is missing page index 0")
            specs.insert(0, (0, {}))

        if page_count is None:
            page_count = len(self.pages)
        labels: list[str] = []
        spec_pos = 0
        current_index, current_spec = specs[0]
        for page_index in range(page_count):
            while spec_pos + 1 < len(specs) and page_index >= specs[spec_pos + 1][0]:
                spec_pos += 1
                current_index, current_spec = specs[spec_pos]
            labels.append(format_page_label(current_spec, page_index - current_index, self.resolve))
        return labels

    @property
    def page_lookup(self) -> PageLookup[PageT]:
        try:
            return self.caches.content["page_lookup"]
        except KeyError:
            return self.caches.get("page_lookup", self.build_page_lookup)

    def build_page_lookup(self) -> PageLookup[PageT]:
        return PageLookup(self)

    def page_index_for(self, page_obj: object) -> int | None:
        return self.page_lookup.page_index_for(page_obj)

    def iter_selected_pages(
        self, pages: PageSelection | None = None
    ) -> Iterator[tuple[int, PageT]]:
        page_objects = self.pages
        for page_index in resolve_page_selection(pages, len(page_objects)):
            yield page_index, page_objects[page_index]


def format_page_label(spec: PdfDict, page_offset: int, resolve: Callable[[object], object]) -> str:
    style = recover_pdf_name(resolve(spec.get("S")))
    prefix = parse_text_string(resolve(spec.get("P"))) or ""
    start = resolve(spec.get("St"))
    normalized: dict[str, object] = {
        "P": prefix,
        "St": start if type(start) is int and start > 0 else 1,
    }
    if style is not None and style in PageLabelStyle:
        normalized["S"] = PdfName.of(style)
    return format_spec_page_label(normalized, page_offset, lambda value: value)


__all__ = (
    "MAX_PAGE_TREE_DEPTH",
    "TRAILER_METADATA_KEYS",
    "check_security_aliases",
    "DocumentOperation",
    "PageLookup",
    "first_indexes",
    "object_headers_present",
    "unresolved_destination",
)
