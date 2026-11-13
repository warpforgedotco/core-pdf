# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
import mmap
import threading
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Generic,
    Protocol,
    Self,
    TypeVar,
)

from core_pdf.impl.base.caches import DocumentCaches
from core_pdf.impl.base.collector import pauses_collector
from core_pdf.impl.base.exceptions import (
    ExtractionScope,
    PdfDocumentClosedError,
    PdfEmptySourceError,
)
from core_pdf.impl.base.types import (
    PageScoped,
    PdfReference,
    PdfSource,
)
from core_pdf.impl.document.metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.document.pdf_page import PdfPage
from core_pdf.impl.document.standards import (
    discover_profile_claims,
)
from core_pdf.impl.fonts.fallback import RasterFontRepository
from core_pdf.impl.recovery.policy import Recovery, recovery_policy
from core_pdf.impl.recovery.resolver import ObjectResolver
from core_pdf.impl.spec.s_07_security.standard import (
    StandardSecurityHandler,
)
from core_pdf.impl.spec.s_07_syntax.types import (
    PdfDict,
)
from core_pdf.impl.spec.s_07_syntax.xref import (
    PdfXRefEntry,
)
from core_pdf.impl.spec.standards import DocumentStandards, SemanticContext
lazy from core_pdf.impl.base.pages import PageSelection
lazy from core_pdf.impl.document.structure import StructureTree
lazy from core_pdf.impl.extract.selection import extract_document
lazy from core_pdf.impl.output.media import ImageRecord
lazy from core_pdf.impl.output.model import Document

if TYPE_CHECKING:
    from core_pdf.impl.fonts.fallback import RasterFontProviderLike
    from core_pdf.impl.spec.s_07_syntax.types import Decipher


from core_pdf.impl.document import fields as form_fields
from core_pdf.impl.document import navigation, page_tree, security, xref_recovery
from core_pdf.impl.document.operations import (
    DocumentOperation,
)
from core_pdf.impl.document.page_tree import (
    PageLookup,
)
from core_pdf.impl.document.source import (
    load_source,
)

PageT = TypeVar("PageT", bound=PdfPage, default=PdfPage)


class DocumentAdapter(Protocol):
    def apply(self, document: Document, /) -> Document: ...


class PdfDocument(Generic[PageT]):
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
        password: str | bytes = "",
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
        password: str | bytes = "",
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
        return recovery_policy(self.recovery_enabled)

    def malformed(self, message: str) -> None:
        self.recovery.malformed(message)

    def page_count(self) -> int:
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

    @pauses_collector
    def extract(
        self,
        *,
        pages: PageSelection | None = None,
        adapters: Iterable[DocumentAdapter] = (),
    ) -> Document:
        with self.acquire_operation() as operation:
            selected_pages = tuple(page for _index, page in self.iter_selected_pages(pages))
            context = ExtractionScope(cancelled=lambda: operation.cancelled)
            result = self.run_extract_document(context, selected_pages)
        for adapter in adapters:
            result = adapter.apply(result)
        return result

    def run_extract_document(self, context: ExtractionScope, pages: Sequence[PdfPage]) -> Document:
        return extract_document(self, context, pages)

    @property
    def structured_document(self) -> Document:
        if self.page_count() == 0:
            return Document(metadata=self.metadata)
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
        return self.caches.get("pages", self.build_page_tree)

    @property
    def page_labels(self) -> list[str] | None:
        labels = self.cached_page_labels()
        return None if labels is None else list(labels)

    @property
    def page_lookup(self) -> PageLookup[PageT]:
        return self.caches.get("page_lookup", self.build_page_lookup)

    @property
    def acroform(self) -> PdfDict | None:
        return self.catalog_dict("AcroForm", recoverable=True)

    def __enter__(self) -> Self:
        if self.closed:
            raise PdfDocumentClosedError("PDF document is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self.closing or self._closed

    def acquire_operation(self) -> DocumentOperation:
        with self.operation_lock:
            if self.closed:
                raise PdfDocumentClosedError("PDF document is closed")
            self.active_operations += 1
        return DocumentOperation(self)

    def release_operation(self) -> None:
        with self.operation_lock:
            self.active_operations = max(0, self.active_operations - 1)
            should_close = self.closing and self.active_operations == 0
        if should_close:
            self.close_resources()

    def close(self) -> None:
        with self.operation_lock:
            if self.closing or self._closed:
                return
            self.closing = True
            self.operation_cancelled.set()
            if self.active_operations:
                return
        self.close_resources()

    def close_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.font_decoders.clear()
        self.caches.clear()

        resolver = getattr(self, "resolver", None)
        if resolver is not None:
            resolver.close()

        raster_fonts = self.raster_font_provider
        if isinstance(raster_fonts, RasterFontRepository):
            raster_fonts.close()

        raw_data = self.raw_data
        self.raw_data = b""
        if isinstance(raw_data, mmap.mmap):
            with contextlib.suppress(BufferError, OSError, ValueError):
                raw_data.close()

        if self.file_handle is not None:
            with contextlib.suppress(OSError):
                self.file_handle.close()
            self.file_handle = None

    @property
    def xref_context(self) -> SemanticContext | None:
        return self.resolver.semantic_context

    @property
    def outlines(self) -> tuple[Any, ...]:
        return tuple(self.iter_outlines())

    @staticmethod
    def ocg_key(ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    @property
    def security_handler(self) -> StandardSecurityHandler | None:
        """The handler of a file opened with its password, or None for a plain file."""
        handler = getattr(self.decipher, "__self__", None)
        return handler if isinstance(handler, StandardSecurityHandler) else None

    # The interactive form.
    fields = form_fields.fields
    fields_by_page = form_fields.fields_by_page
    cached_fields_by_page = form_fields.cached_fields_by_page
    group_all_fields_by_page = form_fields.group_all_fields_by_page
    group_fields_by_page = form_fields.group_fields_by_page
    discover_widget_field_records = form_fields.discover_widget_field_records
    widget_field_root = form_fields.widget_field_root

    # The page tree.
    build_page_tree = page_tree.build_page_tree
    build_pages = page_tree.build_pages
    cached_page_labels = page_tree.cached_page_labels
    build_page_label_tuple = page_tree.build_page_label_tuple
    page_label = page_tree.page_label
    build_page_labels = page_tree.build_page_labels
    build_page_lookup = page_tree.build_page_lookup
    page_index_for = page_tree.page_index_for
    iter_selected_pages = page_tree.iter_selected_pages
    discover_page_nodes = page_tree.discover_page_nodes
    page_candidate_score = page_tree.page_candidate_score
    pages_candidate_score = page_tree.pages_candidate_score
    recovered_inherited_values = page_tree.recovered_inherited_values
    recovered_page_values = page_tree.recovered_page_values
    collect_inherited_values_from_node = page_tree.collect_inherited_values_from_node
    recovered_page_signature = page_tree.recovered_page_signature
    normalized_reference_signature = page_tree.normalized_reference_signature
    recovered_page_nodes = page_tree.recovered_page_nodes
    page_tree_root = page_tree.page_tree_root
    iter_recovered_page_nodes = page_tree.iter_recovered_page_nodes

    # Outlines, destinations and optional content.
    iter_outlines = navigation.iter_outlines
    walk_outlines = navigation.walk_outlines
    extract_outline_count = navigation.extract_outline_count
    resolve_destination = navigation.resolve_destination
    resolve_named_destination = navigation.resolve_named_destination
    destination_from_list = navigation.destination_from_list
    normalize_destination_value = navigation.normalize_destination_value
    normalize_destination_entry = navigation.normalize_destination_entry
    named_destinations = navigation.named_destinations
    oc_hidden_layers = navigation.oc_hidden_layers
    build_oc_hidden_layers = navigation.build_oc_hidden_layers

    # Encryption.
    negotiate_security = security.negotiate_security
    rescan_xref = security.rescan_xref
    init_security = security.init_security

    # Recovering a damaged cross-reference table.
    strict_xref_validation_error = xref_recovery.strict_xref_validation_error
    read_strict_xref_error = xref_recovery.read_strict_xref_error
    brute_force_xref = xref_recovery.brute_force_xref
    brute_forced_objects = xref_recovery.brute_forced_objects
    scan_xref = xref_recovery.scan_xref
    scan_xref_sections = xref_recovery.scan_xref_sections
    read_xref_revisions = xref_recovery.read_xref_revisions
    repair_stale_xref_offsets = xref_recovery.repair_stale_xref_offsets
    pdf_header_offset = xref_recovery.pdf_header_offset
    find_xref_entry_header = xref_recovery.find_xref_entry_header
    xref_entry_matches_header = xref_recovery.xref_entry_matches_header
    xref_entry_header_nearby = xref_recovery.xref_entry_header_nearby
    is_valid_catalog_root = xref_recovery.is_valid_catalog_root
    infer_catalog_root = xref_recovery.infer_catalog_root
    merge_revision_trailers = xref_recovery.merge_revision_trailers
    merge_recovered_trailer_metadata = xref_recovery.merge_recovered_trailer_metadata
    infer_trailer_metadata = xref_recovery.infer_trailer_metadata
    literal_trailer_dictionaries = xref_recovery.literal_trailer_dictionaries
    collect_literal_trailer_dictionaries = xref_recovery.collect_literal_trailer_dictionaries
    iter_literal_trailer_dictionaries = xref_recovery.iter_literal_trailer_dictionaries
    iter_recoverable_xref_stream_dictionaries = (
        xref_recovery.iter_recoverable_xref_stream_dictionaries
    )
    is_valid_trailer_metadata_value = xref_recovery.is_valid_trailer_metadata_value
