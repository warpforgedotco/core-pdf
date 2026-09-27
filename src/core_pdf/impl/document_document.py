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
from core_pdf.impl.document_records import (
    RawFormField,
)
from core_pdf.impl.document_source import DocumentLifecycle, DocumentOperation, load_source
from core_pdf.impl.document_standards import (
    bootstrap_security_context,
    discover_document_standards,
    discover_header_standards,
    discover_profile_claims,
    preserve_historical_version,
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
    PdfUnsupportedError,
)
from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract_selection import extract_document
from core_pdf.impl.fonts_fallback import RasterFontRepository
from core_pdf.impl.output_model import Document as StructuredDocument
from core_pdf.impl.page_selection import PageSelection, resolve_page_selection
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.recovery_policy import LENIENT, STRICT, Recovery
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.recovery_text_strings import parse_text_string
from core_pdf.impl.recovery_trees import iter_number_tree_items
from core_pdf.impl.types import (
    ImageRecord,
    PageScoped,
    PdfName,
    PdfReference,
    PdfSource,
)
from core_pdf_spec.s_07_document.document_labels import PageLabelStyle
from core_pdf_spec.s_07_document.document_labels import (
    format_page_label as format_spec_page_label,
)
from core_pdf_spec.s_07_document.page import PageNode as PageNode
from core_pdf_spec.s_07_security.document import initialize_document_security
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.types import (
    Decipher,
    PdfDict,
)
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
    key_for,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf_spec.standards import DocumentStandards, PdfVersion, SemanticContext

if TYPE_CHECKING:
    from core_pdf.impl.fonts_fallback import RasterFontProviderLike


PageT = TypeVar("PageT", bound=PdfPage, default=PdfPage)


class DocumentAdapter(Protocol):
    def apply(self, document: StructuredDocument, /) -> StructuredDocument: ...


def legacy_name_context(context: SemanticContext | None) -> bool:
    return context is not None and context.version in {PdfVersion(1, 0), PdfVersion(1, 1)}


def check_security_aliases(trailer: PdfDict, resolver: ObjectResolver) -> None:
    pending: list[tuple[dict, bool]] = [(trailer, True)]
    seen: set[int] = set()
    security_resolver: ObjectResolver | None = None
    try:
        while pending:
            dictionary, top = pending.pop()
            if id(dictionary) in seen:
                continue
            seen.add(id(dictionary))
            names: set[bytes] = set()
            for key, value in dictionary.items():
                raw_name = key.value if isinstance(key, PdfName) else key
                lexer = PdfLexer(b"/" + raw_name.encode("latin-1"))
                try:
                    name = bytes(lexer.read_name())
                finally:
                    lexer.close()
                if top and name not in {b"Encrypt", b"AuthCode", b"ID"}:
                    continue
                if name in names:
                    raise PdfUnsupportedError("Ambiguous security dictionary name aliases")
                names.add(name)
                if top and name in {b"Encrypt", b"AuthCode"}:
                    if security_resolver is None:
                        security_resolver = ObjectResolver(
                            resolver.data, resolver.xref, semantic_context=resolver.semantic_context
                        )
                    value = security_resolver.resolve(value)
                if isinstance(value, dict):
                    pending.append((value, False))
    finally:
        if security_resolver is not None:
            security_resolver.close()


class NotBuilt: ...


NOT_BUILT = NotBuilt()


class PdfDocument(
    DocumentLifecycle,
    XRefRecovery,
    PageTreeRecovery,
    DocumentNavigation[PageT],
    DocumentForms[PageT],
    OptionalContent,
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
        "page_cache",
        "fields_by_page_cache",
        "structure_cache",
        "page_labels_cache",
        "hidden_layers_cache",
        "page_lookup_cache",
        "brute_force_objects",
        "literal_trailers_cache",
        "strict_xref_error_cache",
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
    page_cache: tuple[PageT, ...] | None
    fields_by_page_cache: dict[int, list[RawFormField]] | None
    structure_cache: StructureTree | None | NotBuilt
    page_labels_cache: tuple[str, ...] | None | NotBuilt
    hidden_layers_cache: frozenset[str] | None
    page_lookup_cache: PageLookup[PageT] | None
    brute_force_objects: tuple[SemanticContext | None, dict[int, object]] | None
    literal_trailers_cache: tuple[SemanticContext | None, tuple[PdfDict, ...]] | None
    strict_xref_error_cache: tuple[SemanticContext | None, str | None] | None

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
        self.brute_force_objects = None
        self.literal_trailers_cache = None
        self.strict_xref_error_cache = None
        self.reset_caches()
        try:
            self.raw_data, self.file_handle = load_source(source)
            if not len(self.raw_data):
                raise PdfEmptySourceError("PDF source is empty")
            self._standards = discover_header_standards(self.raw_data)
            header = self._standards
            context = header.context if legacy_name_context(header.context) else None
            self.resolver = ObjectResolver(self.raw_data, self.xref, semantic_context=context)
            self.scan_xref()
            self.resolver.xref = self.xref
            if legacy_name_context(context):
                selected = bootstrap_security_context(
                    header, self.raw_data, self.xref, self.trailer_dict
                )
                if not legacy_name_context(selected):
                    check_security_aliases(self.trailer_dict, self.resolver)
                    self.resolver.semantic_context = selected
                    self.scan_xref()
                    self.resolver.xref = self.xref

            for attempt in range(2):
                self.init_security(password)
                self.resolver.decipher = self.decipher
                self._standards = discover_document_standards(
                    header, self.resolver, self.trailer_dict
                )
                self._standards = preserve_historical_version(
                    self._standards,
                    self.raw_data,
                    self.trailer_dict,
                    self.decipher,
                    recovered=self.xref_was_recovered,
                    trailer_context=self.resolver.semantic_context,
                )
                selected = self._standards.context
                if legacy_name_context(selected) == legacy_name_context(
                    self.resolver.semantic_context
                ):
                    encrypt_ref = self.trailer_dict.get("Encrypt")
                    encrypt_object = (
                        self.resolver.resolve(encrypt_ref)
                        if self.decipher is not None and isinstance(encrypt_ref, PdfReference)
                        else None
                    )
                    self.resolver.semantic_context = selected
                    if encrypt_object is not None and isinstance(encrypt_ref, PdfReference):
                        self.resolver.objects[
                            key_for(encrypt_ref.object_number, encrypt_ref.generation_number)
                        ] = encrypt_object
                    break
                if attempt:
                    raise PdfUnsupportedError("Unstable security dictionary name semantics")
                if not legacy_name_context(selected):
                    check_security_aliases(self.trailer_dict, self.resolver)
                self.resolver.close()
                self.decipher = None
                self.resolver = ObjectResolver(self.raw_data, self.xref, semantic_context=selected)
                self.scan_xref()
                self.resolver.xref = self.xref
        except BaseException:
            self.close()
            raise

    def reset_caches(self) -> None:
        self.page_cache = None
        self.fields_by_page_cache = None
        self.structure_cache = NOT_BUILT
        self.page_labels_cache = NOT_BUILT
        self.hidden_layers_cache = None
        self.page_lookup_cache = None

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
        cached = self.structure_cache
        if not isinstance(cached, NotBuilt):
            return cached
        root = self.catalog_dict("StructTreeRoot")
        tree = None if root is None else StructureTree(self, root, page_lookup=self.page_lookup)
        self.structure_cache = tree
        return tree

    @property
    def recovery_enabled(self) -> bool:
        return self.xref_was_recovered or self.page_tree_was_recovered

    @property
    def recovery(self) -> Recovery:
        return LENIENT if self.recovery_enabled else STRICT

    def malformed(self, message: str) -> None:
        self.recovery.malformed(message)

    def init_security(self, password: str) -> None:
        trailer = self.trailer_dict
        if trailer.get("Encrypt") is not None and trailer.get("ID") is None:
            trailer = dict(trailer)
            trailer["ID"] = [b""]
        self.decipher = initialize_document_security(
            self.raw_data,
            trailer,
            self.resolver,
            password,
            handler_factory=create_recovered_security_handler,
        )

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
        pages = self.page_cache
        if pages is None:
            pages = self.page_cache = self.build_pages(self.iter_recovered_page_nodes())
        return pages

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
        cached = self.page_labels_cache
        if isinstance(cached, NotBuilt):
            labels = self.build_page_labels()
            cached = self.page_labels_cache = None if labels is None else tuple(labels)
        return cached

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
        lookup = self.page_lookup_cache
        if lookup is None:
            lookup = self.page_lookup_cache = PageLookup(self)
        return lookup

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


def normalize_values(
    values: PdfDict, integer_fields: tuple[str, ...], name_fields: tuple[str, ...]
) -> PdfDict:
    normalized = values
    for name in integer_fields:
        value = values.get(name)
        number = parse_int(value)
        if number is not None and type(value) is not int:
            if normalized is values:
                normalized = dict(values)
            normalized[name] = number
    for name in name_fields:
        value = values.get(name)
        decoded = recover_pdf_name(value)
        if decoded is not None and not isinstance(value, PdfName) and decoded != value:
            if normalized is values:
                normalized = dict(values)
            normalized[name] = PdfName.of(decoded)
    return normalized


def create_recovered_security_handler(
    document_id: Sequence[object], params: PdfDict, password: str = ""
) -> StandardSecurityHandler:
    normalized = normalize_values(
        params, ("V", "R", "P", "Length"), ("Filter", "StmF", "StrF", "EFF")
    )
    filters = params.get("CF")
    if isinstance(filters, dict):
        normalized_filters = filters
        for key, value in filters.items():
            name = recover_pdf_name(key)
            normalized_value = value
            if name == "StdCF" and isinstance(value, dict):
                normalized_value = normalize_values(
                    value, ("Length",), ("Type", "CFM", "AuthEvent")
                )
            normalized_key = (
                PdfName.of(name)
                if name is not None and not isinstance(key, PdfName) and name != key
                else key
            )
            if normalized_key != key or normalized_value is not value:
                if normalized_filters is filters:
                    normalized_filters = dict(filters)
                if normalized_key != key:
                    del normalized_filters[key]
                normalized_filters[normalized_key] = normalized_value
        if normalized_filters is not filters:
            if normalized is params:
                normalized = dict(params)
            normalized["CF"] = normalized_filters
    return create_standard_security_handler(document_id, normalized, password)


__all__ = (
    "MAX_PAGE_TREE_DEPTH",
    "TRAILER_METADATA_KEYS",
    "DocumentOperation",
    "PageLookup",
    "first_indexes",
    "object_headers_present",
    "unresolved_destination",
)
