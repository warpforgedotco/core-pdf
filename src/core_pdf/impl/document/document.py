# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
import mmap
import struct
import threading
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from functools import partial
from itertools import compress, repeat
from operator import and_, is_, itemgetter, not_, truth
from os import PathLike
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO, Generic, Protocol, Self, TypeVar

import numpy

from core_pdf.impl.document.fields import collect_field_records
from core_pdf.impl.document.metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.document.page import PAGE_INHERITED_KEYS, PdfPage
from core_pdf.impl.document.page_links import goto_action_destination
from core_pdf.impl.document.page_tree import (
    MAX_PAGE_TREE_DEPTH,
    infer_page_tree_node_type,
    resolve_page_tree_node_type,
)
from core_pdf.impl.document.records import (
    RawEmbeddedFile,
    RawFormField,
    RawNamedDestination,
    RawOutlineItem,
)
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl.document.recovery.text_strings import parse_text_string
from core_pdf.impl.document.recovery.trees import iter_name_tree_items, iter_number_tree_items
from core_pdf.impl.document.recovery.xref import XRefScanner, iter_indirect_object_headers
from core_pdf.impl.document.standards import (
    bootstrap_security_context,
    discover_document_standards,
    discover_header_standards,
    discover_profile_claims,
    find_pdf_header,
    preserve_historical_version,
)
from core_pdf.impl.document.structure import StructureTree
from core_pdf.impl.exceptions import (
    PdfDocumentClosedError,
    PdfParseError,
    PdfSourceError,
    PdfUnsupportedError,
)
from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract.selection import extract_document
from core_pdf.impl.fonts.fallback import RasterFontRepository
from core_pdf.impl.output.model import Document as StructuredDocument
from core_pdf.impl.page_selection import PageSelection, resolve_page_selection
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import (
    ImageRecord,
    PageScoped,
    PdfByteBuffer,
    PdfName,
    PdfReference,
    PdfSource,
)
from core_pdf_cythonized import object_headers_match
from core_pdf_spec.s_07_document.document_labels import PageLabelStyle
from core_pdf_spec.s_07_document.document_labels import (
    format_page_label as format_spec_page_label,
)
from core_pdf_spec.s_07_document.page import PageNode as PageNode
from core_pdf_spec.s_07_document.page import iter_page_nodes
from core_pdf_spec.s_07_security.document import initialize_document_security
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import (
    Decipher,
    InheritedValueMap,
    PdfArray,
    PdfDict,
    ResolvedObjectCache,
)
from core_pdf_spec.s_07_syntax.xref import (
    PdfXRefEntry,
    iter_xref_revisions,
    key_for,
    merge_xref_sections,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf_spec.standards import DocumentStandards, PdfVersion, SemanticContext

if TYPE_CHECKING:
    from core_pdf.impl.fonts.fallback import RasterFontProviderLike


PageT = TypeVar("PageT", bound=PdfPage, default=PdfPage)
LookupPageT = TypeVar("LookupPageT", bound=PdfPage)


class DocumentAdapter(Protocol):
    def apply(self, document: StructuredDocument, /) -> StructuredDocument: ...


class DocumentOperation(AbstractContextManager["DocumentOperation"]):
    __slots__ = ("document", "released")

    def __init__(self, document: PdfDocument[Any]) -> None:
        self.document = document
        self.released = False

    @property
    def cancelled(self) -> bool:
        return self.document.operation_cancelled.is_set()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.document.release_operation()

    def __exit__(self, *args: object) -> None:
        self.release()


def first_indexes(values: Iterable[object]) -> dict[object, int]:
    """Each hashable value's first index in values."""
    indexes: dict[object, int] = {}
    for index, value in enumerate(values):
        with contextlib.suppress(TypeError):
            indexes.setdefault(value, index)
    return indexes


class PageLookup[LookupPageT: PdfPage]:
    __slots__ = (
        "document",
        "iter_nodes",
        "indexes",
        "struct_parents_indexes",
        "signature_indexes",
        "_pages",
        "names",
    )

    def __init__(self, document: PdfDocument[LookupPageT]) -> None:
        self.document = document
        self.iter_nodes: tuple[PageNode, ...] | None = None
        self.indexes: dict[int, int] = {}
        self.struct_parents_indexes: dict[object, int] | None = None
        self.signature_indexes: dict[object, int] | None = None
        self._pages: tuple[LookupPageT, ...] | None = None
        self.names: dict[str, RawNamedDestination] | None = None

    @property
    def nodes(self) -> tuple[PageNode, ...]:
        if self.iter_nodes is None:
            # The document's pages are its page nodes, already walked.
            self.iter_nodes = tuple(
                PageNode(page.page_dict, page.inherited_values) for page in self.document.pages
            )
            for index, node in enumerate(self.iter_nodes):
                self.indexes.setdefault(id(node.dictionary), index)
        return self.iter_nodes

    @property
    def pages(self) -> tuple[LookupPageT, ...]:
        if self._pages is None:
            self._pages = self.document.build_pages(self.nodes)
        return self._pages

    def page_index_for(self, page_obj: object) -> int | None:
        if isinstance(page_obj, PdfPage):
            return page_obj.page_number - 1
        if not isinstance(page_obj, dict):
            return None
        nodes = self.nodes
        index = self.indexes.get(id(page_obj))
        if index is not None:
            return index
        page_struct_parents = page_obj.get("StructParents")
        if page_struct_parents is not None:
            if self.struct_parents_indexes is None:
                self.struct_parents_indexes = first_indexes(
                    node.dictionary.get("StructParents") for node in nodes
                )
            index = self.first_index(
                self.struct_parents_indexes,
                page_struct_parents,
                (node.dictionary.get("StructParents") for node in nodes),
            )
            if index is not None:
                return index
        for index, node in enumerate(nodes):
            if node.dictionary == page_obj:
                return index
        signature_of = self.document.recovered_page_signature
        if self.signature_indexes is None:
            self.signature_indexes = first_indexes(signature_of(node.dictionary) for node in nodes)
        return self.first_index(
            self.signature_indexes,
            signature_of(page_obj),
            (signature_of(node.dictionary) for node in nodes),
        )

    @staticmethod
    def first_index(
        indexes: dict[object, int], value: object, values: Iterable[object]
    ) -> int | None:
        """The first index whose value equals value: from indexes, or by
        scanning values when value cannot be hashed."""
        try:
            return indexes.get(value)
        except TypeError:
            return next((index for index, item in enumerate(values) if item == value), None)

    def resolve_named_destination(self, name: str) -> RawNamedDestination | None:
        if self.names is None:
            self.names = self.document.named_destinations(page_lookup=self)
        return self.names.get(name)


def unresolved_destination(name: str) -> RawNamedDestination:
    return RawNamedDestination(page_index=None, type=None, args=[], raw=name)


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


class NotBuilt:
    """Marks a document cache whose value may legitimately be None."""


NOT_BUILT = NotBuilt()


class PdfDocument(Generic[PageT]):
    page_class: type | None = None

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
        # The document is read-only once open, so its page tree and form fields
        # are built once. Every page.extract() asks for the fields, and building
        # them walks every page, so without this a whole-document pass is
        # quadratic in its page count.
        self.reset_caches()
        try:
            self.raw_data = self.load_data(source)
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
            # Anything built above belongs to a resolver that may since have
            # been replaced.
            self.reset_caches()
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
        should_close = False
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
        self.reset_caches()

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

    def resolve(self, ref: object) -> object:
        return self.resolver.resolve(ref)

    def catalog(self) -> PdfDict:
        root_ref = self.trailer_dict.get("Root")
        if root_ref is None:
            raise ValueError("missing catalog root")
        root = self.resolve(root_ref)
        if not isinstance(root, dict):
            raise ValueError("invalid catalog root")
        return root

    def get_metadata(self) -> MetadataRecord:
        return resolve_metadata(
            self.resolver,
            self.trailer_dict,
            recover=self.recovery_enabled,
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
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if recoverable and self.recovery_enabled:
            return None
        raise ValueError(f"invalid {key} dictionary")

    @property
    def structure(self) -> StructureTree | None:
        # Built once: page.structure asks on every page.extract(), and a tree
        # walks the whole ParentTree to answer -- 6 ms a page on PDF Reference
        # 1.7, whose ParentTree has 33,162 entries.
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

    def load_data(self, source: PdfSource) -> PdfByteBuffer:
        if isinstance(source, (str, PathLike)):
            if isinstance(source, str) and source.startswith("%PDF"):
                return source.encode("latin-1")
            file_handle = open(source, "rb")  # noqa: SIM115
            self.file_handle = file_handle
            try:
                return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError) as exc:
                try:
                    is_empty = file_handle.seek(0, 2) == 0
                except OSError:
                    is_empty = False
                file_handle.close()
                self.file_handle = None
                if is_empty:
                    raise PdfSourceError("PDF source is empty") from exc
                raise PdfSourceError(str(exc)) from exc
        if isinstance(source, bytes):
            return source
        if isinstance(source, (memoryview, bytearray)):
            return bytes(source)

        mapped = self.try_mmap_reader(source)
        if mapped is not None:
            return mapped

        read = getattr(source, "read", None)
        if not callable(read):
            raise PdfSourceError(f"PDF source type {type(source).__name__} is not supported")
        reader = source
        tell = getattr(source, "tell", None)
        seek = getattr(source, "seek", None)
        position: int | None = None
        if callable(tell) and callable(seek):
            try:
                position = tell()
                seek(0)
            except OSError, TypeError, ValueError:
                position = None
        try:
            raw = reader.read()
        except OSError as exc:
            raise PdfSourceError(str(exc)) from exc
        finally:
            if position is not None and callable(seek):
                seek(position)
        return raw if isinstance(raw, bytes) else bytes(raw)

    def try_mmap_reader(self, source: object) -> mmap.mmap | None:
        fileno = getattr(source, "fileno", None)
        if not callable(fileno):
            return None
        try:
            fd = fileno()
        except OSError, TypeError, ValueError:
            return None
        try:
            return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except ValueError as error:
            raise PdfSourceError("PDF source is empty") from error
        except OSError:
            return None

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

    def discover_page_nodes(self) -> Iterator[PageNode]:
        candidates: list[tuple[int, int, int, PdfDict]] = []
        pages_nodes: list[tuple[int, int, int, PdfDict]] = []
        seen_objects: set[int] = set()
        for key, entry in sorted(
            self.xref.items(),
            key=lambda item: (
                item[1].offset if item[1].object_stream is None else 0,
                item[0] >> 16,
            ),
        ):
            if not entry.in_use:
                continue
            try:
                obj = self.resolver.resolve(PdfReference(key >> 16, key & 0xFFFF))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            marker = id(obj)
            if marker in seen_objects:
                continue
            seen_objects.add(marker)
            node_type = resolve_page_tree_node_type(self.resolver, obj)
            pages_score = self.pages_candidate_score(obj, node_type)
            if pages_score > 0:
                pages_nodes.append((pages_score, entry.offset, key >> 16, obj))
            score = self.page_candidate_score(obj, node_type)
            if score > 0:
                candidates.append((score, entry.offset, key >> 16, obj))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        pages_nodes.sort(key=lambda item: (-item[0], item[1], item[2]))
        inherited_sources = [node for _, _, _, node in pages_nodes]
        seen_signatures: set[tuple[object, ...]] = set()
        for _, _, _, page_dict in candidates:
            signature = self.recovered_page_signature(page_dict)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            yield PageNode(
                page_dict,
                self.recovered_page_values(page_dict, inherited_sources),
            )

    def page_candidate_score(self, obj: PdfDict, node_type: str | None) -> int:
        if node_type == "Pages" or node_type not in (None, "Page"):
            return -100

        explicit_type = recover_pdf_name(obj.get("Type"))
        if explicit_type != "Page" and obj.get("Contents") is None and obj.get("Annots") is None:
            return -100

        score = 20 if node_type == "Page" else 0
        if obj.get("Kids") is not None:
            score -= 30
        if obj.get("Contents") is not None:
            score += 12
        if obj.get("MediaBox") is not None:
            score += 8
        if obj.get("Resources") is not None:
            score += 4
        if obj.get("Parent") is not None:
            score += 2
        if obj.get("Annots") is not None:
            score += 1
        return score if score >= 16 else -100

    def pages_candidate_score(self, obj: PdfDict, node_type: str | None) -> int:
        if node_type != "Pages":
            return -100
        score = 20
        try:
            kids = self.resolver.resolve(obj.get("Kids"))
        except Exception:
            kids = None
        if isinstance(kids, list):
            score += min(len(kids), 20)
        try:
            count = self.resolver.resolve(obj.get("Count"))
        except Exception:
            count = None
        if type(count) is int and count >= 0:
            score += min(count, 20)
        if obj.get("Resources") is not None:
            score += 5
        if obj.get("MediaBox") is not None:
            score += 5
        return score

    def recovered_page_values(
        self, page_dict: PdfDict, pages_nodes: list[PdfDict]
    ) -> InheritedValueMap:
        values: InheritedValueMap = {
            key: value for key in PAGE_INHERITED_KEYS if (value := page_dict.get(key)) is not None
        }
        missing = [key for key in PAGE_INHERITED_KEYS if key not in values]
        if not missing:
            return values

        sources: list[PdfDict] = []
        parent = page_dict.get("Parent")
        if parent is not None:
            try:
                parent_obj = self.resolver.resolve(parent)
            except Exception:
                parent_obj = None
            if isinstance(parent_obj, dict):
                sources.append(parent_obj)
        sources.extend(pages_nodes)
        if not sources:
            return values

        for source in sources:
            source_values = self.collect_inherited_values_from_node(source, missing)
            values.update(source_values)
            missing = [key for key in missing if key not in values]
            if not missing:
                break
        return values

    def collect_inherited_values_from_node(
        self, node: PdfDict, keys: list[str]
    ) -> InheritedValueMap:
        def resolve_ref(value: object) -> object:
            try:
                return self.resolver.resolve(value)
            except Exception:
                return None

        return collect_inherited_values(
            node, tuple(keys), resolve_ref, stop_at_malformed_parent=True
        )

    def recovered_page_signature(self, page_dict: PdfDict) -> tuple[object, ...]:
        contents = page_dict.get("Contents")
        normalized_contents = self.normalized_reference_signature(contents)
        if normalized_contents is not None:
            return ("Contents", normalized_contents)
        return (
            "Shape",
            self.normalized_reference_signature(page_dict.get("MediaBox")),
            self.normalized_reference_signature(page_dict.get("Resources")),
            id(page_dict),
        )

    def normalized_reference_signature(self, value: object) -> object:
        if isinstance(value, PdfReference):
            return ("R", value.object_number, value.generation_number)
        if isinstance(value, (list, tuple)):
            return tuple(self.normalized_reference_signature(item) for item in value)
        if isinstance(value, dict):
            return ("D", id(value))
        if isinstance(value, PdfStream):
            return ("S", id(value))
        return value

    def iter_page_dicts(self) -> Iterator[PdfDict]:
        for page_node in self.iter_recovered_page_nodes():
            yield page_node.dictionary

    def recovered_page_nodes(self) -> list[PageNode]:
        discovered = list(self.discover_page_nodes())
        if discovered:
            self.page_tree_was_recovered = True
        return discovered

    def page_tree_root(self) -> PdfDict:
        pages_ref = self.catalog().get("Pages")
        if pages_ref is None:
            raise ValueError("missing page tree root")
        pages_node = self.resolver.resolve(pages_ref)
        if not isinstance(pages_node, dict):
            raise ValueError("invalid page tree root")
        return pages_node

    def iter_recovered_page_nodes(self) -> Iterator[PageNode]:
        try:
            pages_node = self.page_tree_root()
            page_dicts = list(
                iter_page_nodes(
                    pages_node,
                    self.resolver.resolve,
                    inherited_keys=PAGE_INHERITED_KEYS,
                    node_type=lambda node: resolve_page_tree_node_type(self.resolver, node),
                    on_invalid_child=lambda _node: True,
                    max_depth=MAX_PAGE_TREE_DEPTH,
                )
            )
            if page_dicts:
                yield from page_dicts
                return
            discovered = self.recovered_page_nodes()
            if discovered:
                yield from discovered
                return
        except PdfParseError, ValueError:
            discovered = self.recovered_page_nodes()
            if discovered:
                yield from discovered
                return
            return

    def page_count(self) -> int:
        if not self.page_tree_was_recovered:
            try:
                count = self.resolver.resolve(self.page_tree_root().get("Count"))
                if type(count) is int and count >= 0:
                    return count
            except PdfParseError, ValueError:
                pass
        return len(self.pages)

    def build_page_dicts(self) -> list[PdfDict]:
        return list(self.iter_page_dicts())

    @property
    def metadata(self) -> dict[str, object]:
        value = self.get_metadata()
        return dict(value) if isinstance(value, dict) else {}

    @property
    def standards(self) -> DocumentStandards:
        with self.acquire_operation():
            return self.get_standards()

    @property
    def outlines(self) -> tuple[Any, ...]:
        return tuple(self.iter_outlines())

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
        page_class = self.page_class
        if page_class is None:
            page_class = PdfPage
        factory = page_class
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
        # Built once: every page's label indexes these, and building them
        # walks the whole PageLabels tree and page list.
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
        except ValueError:
            if self.recovery_enabled:
                return None
            raise
        if labels_root is None:
            return None
        if not isinstance(labels_root, dict):
            raise ValueError("invalid PageLabels number tree")

        specs = [
            (page_index, spec)
            for page_index, spec in iter_number_tree_items(
                labels_root,
                self.resolve,
                recover=self.recovery_enabled,
            )
            if isinstance(spec, dict)
        ]
        if not specs:
            return None
        specs.sort(key=lambda item: item[0])
        if specs[0][0] != 0:
            if not self.recovery_enabled:
                raise ValueError("PageLabels is missing page index 0")
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
        # One for the document, reset with its pages: each builds its page
        # indexes and named destinations once.
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

    def iter_outlines(self) -> list[RawOutlineItem]:
        outlines = self.catalog_dict("Outlines")
        if outlines is None:
            return []
        first = self.resolver.resolve(outlines.get("First"))
        if first is None:
            return []
        return self.walk_outlines(first, 0)

    def walk_outlines(
        self,
        item: object,
        level: int,
        *,
        page_lookup: PageLookup[PageT] | None = None,
    ) -> list[RawOutlineItem]:
        if page_lookup is None:
            page_lookup = self.page_lookup
        recover_outlines = self.recovery_enabled
        if level > 200:
            raise ValueError("invalid outline depth")
        if not isinstance(item, dict):
            if recover_outlines:
                return []
            raise ValueError("invalid outline item")
        result: list[RawOutlineItem] = []
        current: object | None = item
        seen: set[int] = set()
        while current is not None:
            current = self.resolver.resolve(current)
            if not isinstance(current, dict):
                if recover_outlines:
                    break
                raise ValueError("invalid outline item")
            marker = id(current)
            if marker in seen:
                if recover_outlines:
                    break
                raise ValueError("outline cycle detected")
            seen.add(marker)
            title = self.resolver.resolve_str(current.get("Title"))
            dest = current.get("Dest")
            if dest is None:
                dest = goto_action_destination(
                    self.resolver, self.resolver.resolve(current.get("A"))
                )
            try:
                result.append(
                    RawOutlineItem(
                        title=title or "",
                        level=level,
                        dest=dest,
                        page_index=self.resolve_destination(dest, page_lookup=page_lookup),
                        count=self.extract_outline_count(current),
                    )
                )
            except ValueError:
                if not recover_outlines:
                    raise
            first = current.get("First")
            if first is not None:
                first = self.resolver.resolve(first)
                if not isinstance(first, dict):
                    if recover_outlines:
                        current = current.get("Next")
                        continue
                    raise ValueError("invalid outline child")
                result.extend(self.walk_outlines(first, level + 1, page_lookup=page_lookup))
            current = current.get("Next")
        return result

    @staticmethod
    def validate_outline_count(value: object) -> int:
        if type(value) is not int:
            raise ValueError("invalid outline count")
        return value

    def extract_outline_count(self, current: PdfDict) -> int:
        raw_count = current.get("Count")
        if raw_count is None:
            return 0
        current_count = self.resolver.resolve_int(raw_count)
        if current_count is None:
            if self.recovery_enabled:
                return 0
            raise ValueError("invalid outline count")
        return self.validate_outline_count(current_count)

    def resolve_destination(
        self, dest: object, *, page_lookup: PageLookup[PageT] | None = None
    ) -> int | None:
        if dest is None:
            return None
        normalized = self.normalize_destination_value(dest, page_lookup=page_lookup)
        if (
            normalized.raw is None
            and normalized.page_index is None
            and normalized.type is None
            and not normalized.args
        ):
            raise ValueError("invalid destination")
        return normalized.page_index

    def resolve_named_destination(self, name: str) -> RawNamedDestination | None:
        return self.page_lookup.resolve_named_destination(name)

    def destination_from_list(
        self,
        resolved_list: PdfArray,
        *,
        page_lookup: PageLookup[PageT] | None = None,
    ) -> RawNamedDestination:
        if not resolved_list:
            raise ValueError("invalid destination array")
        page_obj = self.resolver.resolve(resolved_list[0])
        if page_obj is None:
            raise ValueError("invalid destination page reference")
        lookup = self.page_lookup if page_lookup is None else page_lookup
        page_index = lookup.page_index_for(page_obj)
        if page_index is None:
            raise ValueError("invalid destination page reference")
        dest_type = None
        args: PdfArray = []
        if len(resolved_list) >= 2:
            raw_type = resolved_list[1]
            dest_type = self.resolver.resolve_name_or_text(raw_type)
            if dest_type is None:
                raise ValueError("invalid destination type")
            args = list(resolved_list[2:]) if len(resolved_list) > 2 else []
        return RawNamedDestination(
            page_index=page_index, type=dest_type, args=args, raw=resolved_list
        )

    def normalize_destination_value(
        self,
        val: object,
        *,
        page_lookup: PageLookup[PageT] | None = None,
    ) -> RawNamedDestination:
        lookup = self.page_lookup if page_lookup is None else page_lookup
        return self.normalize_destination_entry(val, lookup.resolve_named_destination, lookup)

    def normalize_destination_entry(
        self,
        val: object,
        resolve_name: Callable[[str], RawNamedDestination | None],
        page_lookup: PageLookup[PageT],
    ) -> RawNamedDestination:
        seen: set[int] = set()
        resolved = self.resolver.resolve(val)
        while isinstance(resolved, dict):
            identity = id(resolved)
            if identity in seen:
                raise ValueError("cyclic destination dictionary")
            seen.add(identity)
            dest_value = resolved.get("D")
            if dest_value is None:
                break
            val = dest_value
            resolved = self.resolver.resolve(val)
        resolved_list = val if isinstance(val, list) else resolved
        if isinstance(resolved_list, tuple):
            resolved_list = list(resolved_list)
        if isinstance(resolved_list, list) and resolved_list:
            return self.destination_from_list(resolved_list, page_lookup=page_lookup)
        if isinstance(resolved_list, list):
            raise ValueError("invalid destination array")

        name = self.resolver.resolve_name_like_value(resolved)
        if name is not None:
            nested = resolve_name(name)
            if nested is not None:
                return nested
        raise ValueError("invalid destination")

    def named_destinations(
        self, *, page_lookup: PageLookup[PageT] | None = None
    ) -> dict[str, RawNamedDestination]:
        lookup = self.page_lookup if page_lookup is None else page_lookup
        targets: dict[str, object] = {}
        dests = self.resolver.resolve(self.catalog().get("Dests"))
        if isinstance(dests, dict):
            for name, val in dests.items():
                resolved_name = self.resolver.resolve_name(name)
                if resolved_name is None:
                    raise ValueError("invalid named destination key")
                targets[resolved_name] = self.resolver.resolve(val)
        names = self.resolver.resolve(self.catalog().get("Names"))
        if isinstance(names, dict):
            dests_tree = self.resolver.resolve(names.get("Dests"))
            if isinstance(dests_tree, dict):
                targets.update(
                    iter_name_tree_items(
                        dests_tree,
                        self.resolver.resolve,
                        self.resolver.resolve_str,
                        recover=self.recovery_enabled,
                    )
                )

        normalized: dict[str, RawNamedDestination] = {}
        resolving: set[str] = set()

        def normalize_name(name: str) -> RawNamedDestination:
            cached = normalized.get(name)
            if cached is not None:
                return cached
            if name in resolving:
                return unresolved_destination(name)
            resolving.add(name)
            try:
                target = targets.get(name)
                result = (
                    unresolved_destination(name)
                    if target is None
                    else self.normalize_destination_entry(target, normalize_name, lookup)
                )
                normalized[name] = result
                return result
            finally:
                resolving.discard(name)

        for name in targets:
            try:
                normalize_name(name)
            except PdfParseError, ValueError:
                normalized[name] = unresolved_destination(name)
        return normalized

    @property
    def acroform(self) -> PdfDict | None:
        return self.catalog_dict("AcroForm", recoverable=True)

    def fields(self) -> list[RawFormField]:
        af = self.acroform
        records: list[RawFormField] = []
        if af is not None:
            field_list = af.get("Fields")
            if field_list is None:
                field_list = []
            elif not isinstance(field_list, list):
                if self.recovery_enabled:
                    field_list = []
                else:
                    raise ValueError("invalid AcroForm Fields array")
            for field in field_list:
                field_obj = self.resolver.resolve(field)
                records.extend(
                    collect_field_records(self.resolver, field_obj, recover=self.recovery_enabled)
                )
        if not records or self.recovery_enabled:
            records.extend(self.discover_widget_field_records(records))
        return records

    def fields_by_page(
        self,
        pages: Sequence[PageT] | None = None,
    ) -> dict[int, list[RawFormField]]:
        if pages is not None:
            return self.group_fields_by_page(tuple(pages))
        return {
            page_index: list(fields) for page_index, fields in self.cached_fields_by_page().items()
        }

    def cached_fields_by_page(self) -> dict[int, list[RawFormField]]:
        """The whole document's fields by page, shared: callers must not mutate it."""
        grouped = self.fields_by_page_cache
        if grouped is None:
            grouped = self.fields_by_page_cache = self.group_fields_by_page(self.pages)
        return grouped

    def group_fields_by_page(
        self, page_sequence: tuple[PageT, ...]
    ) -> dict[int, list[RawFormField]]:
        page_indexes_by_dict = {
            id(page.page_dict): page.page_number - 1
            for page in page_sequence
            if isinstance(page, PdfPage)
        }
        grouped: dict[int, list[RawFormField]] = {}
        annot_page_index: dict[int, int] | None = None

        def widget_page_index(widget: object) -> int | None:
            nonlocal annot_page_index
            pg_ref = widget.get("P") if isinstance(widget, dict) else None
            if pg_ref is not None:
                pg_obj = self.resolver.resolve(pg_ref)
                return page_indexes_by_dict.get(id(pg_obj)) if isinstance(pg_obj, dict) else None
            if annot_page_index is None:
                annot_page_index = {
                    id(annot): page.page_number - 1
                    for page in page_sequence
                    if isinstance(page, PdfPage)
                    for annot in page.annotation_dicts()
                }
            return annot_page_index.get(id(widget))

        for field in self.fields():
            page_indexes: set[int] = set()
            if field.widget:
                if not isinstance(field.widget, dict):
                    raise ValueError("invalid field widget entry")
                page_index = widget_page_index(field.widget)
                if page_index is not None:
                    page_indexes.add(page_index)
            elif field.kids:
                if not isinstance(field.kids, list):
                    raise ValueError("invalid field kids array")
                for kid_ref in field.kids:
                    kid = self.resolver.resolve(kid_ref)
                    if (
                        isinstance(kid, dict)
                        and self.resolver.resolve_name(kid.get("Subtype")) == "Widget"
                    ):
                        page_index = widget_page_index(kid)
                        if page_index is not None:
                            page_indexes.add(page_index)
            for page_index in page_indexes:
                grouped.setdefault(page_index, []).append(field)
        return grouped

    def discover_widget_field_records(self, existing: list[RawFormField]) -> list[RawFormField]:
        seen_widgets = {id(record.widget) for record in existing if isinstance(record.widget, dict)}
        records: list[RawFormField] = []
        for page in self.pages:
            for annot in page.annotation_dicts():
                if id(annot) in seen_widgets:
                    continue
                subtype = self.resolver.resolve_name_or_text(annot.get("Subtype")) or ""
                if subtype != "Widget":
                    continue
                root = self.widget_field_root(annot)
                if id(root) in seen_widgets:
                    continue
                seen_widgets.add(id(root))
                seen_widgets.add(id(annot))
                records.extend(
                    collect_field_records(self.resolver, root, recover=self.recovery_enabled)
                )
        return records

    def widget_field_root(self, annot: PdfDict) -> PdfDict:
        node = annot
        seen = {id(node)}
        for _ in range(50):
            parent = self.resolver.resolve(node.get("Parent"))
            if not isinstance(parent, dict) or id(parent) in seen:
                break
            seen.add(id(parent))
            node = parent
        return node

    def embedded_files(self) -> list[RawEmbeddedFile]:
        names = self.resolver.resolve(self.catalog().get("Names"))
        if not isinstance(names, dict):
            return []
        embedded_tree = self.resolver.resolve(names.get("EmbeddedFiles"))
        if embedded_tree is None:
            return []
        if not isinstance(embedded_tree, dict):
            raise ValueError("invalid EmbeddedFiles name tree")

        recover = self.recovery_enabled
        records: list[RawEmbeddedFile] = []
        for name, value in iter_name_tree_items(
            embedded_tree,
            self.resolver.resolve,
            self.resolver.resolve_str,
            recover=recover,
        ):
            try:
                record = self.embedded_file_record(name, value)
            except ValueError:
                if recover:
                    continue
                raise
            records.append(record)
        return records

    def embedded_file_record(self, name: str, value: object) -> RawEmbeddedFile:
        filespec = self.resolver.resolve(value)
        if not isinstance(filespec, dict):
            raise ValueError("invalid embedded file spec")
        ef = self.resolver.resolve(filespec.get("EF"))
        if not isinstance(ef, dict):
            raise ValueError("invalid embedded file stream")
        stream = self.resolver.resolve(ef.get("UF") or ef.get("F"))
        if not isinstance(stream, PdfStream):
            raise ValueError("invalid embedded file stream")
        filename = (
            self.resolver.resolve_str(filespec.get("UF"))
            or self.resolver.resolve_str(filespec.get("F"))
            or name
        )
        return RawEmbeddedFile(name, filename, filespec, stream, stream.data)

    @staticmethod
    def ocg_key(ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    def oc_hidden_layers(self) -> frozenset[str]:
        # Built once: every page's capture asks for it.
        hidden = self.hidden_layers_cache
        if hidden is None:
            hidden = self.hidden_layers_cache = self.build_oc_hidden_layers()
        return hidden

    def build_oc_hidden_layers(self) -> frozenset[str]:
        recover = self.recovery_enabled
        try:
            self.catalog()
        except ValueError:
            return frozenset()
        oc = self.catalog_dict("OCProperties", recoverable=True)
        if oc is None:
            return frozenset()
        ocgs = self.resolver.resolve(oc.get("OCGs"))
        if ocgs is None:
            return frozenset()
        if not isinstance(ocgs, list):
            if recover:
                return frozenset()
            raise ValueError("invalid OCProperties OCGs array")

        on_layers: set[tuple[int, int] | int] = set()
        default_config = self.resolver.resolve(oc.get("D"))
        if default_config is not None and not isinstance(default_config, dict):
            if recover:
                default_config = None
            else:
                raise ValueError("invalid OCProperties D dictionary")
        if default_config is not None:
            base_state_value = default_config.get("BaseState")
            base_state = (
                self.resolver.resolve_name(base_state_value)
                if base_state_value is not None
                else None
            )
            if base_state_value is not None and base_state is None:
                if not recover:
                    raise ValueError("invalid OCProperties BaseState value")
            elif base_state not in (None, "ON", "OFF", "Unchanged"):
                if recover:
                    base_state = None
                else:
                    raise ValueError("invalid OCProperties BaseState value")
            if base_state != "OFF":
                for ocg in ocgs:
                    key = self.ocg_key(ocg, self.resolver.resolve(ocg))
                    if key is not None:
                        on_layers.add(key)

            for override_name, update in (("ON", on_layers.add), ("OFF", on_layers.discard)):
                refs = default_config.get(override_name)
                if not isinstance(refs, list):
                    continue
                for ref in refs:
                    ocg_resolved = self.resolver.resolve(ref)
                    if not isinstance(ocg_resolved, dict):
                        if recover:
                            continue
                        raise ValueError(f"invalid OCProperties {override_name} entry")
                    key = self.ocg_key(ref, ocg_resolved)
                    if key is not None:
                        update(key)

        hidden_layers: set[str] = set()
        for ocg_ref in ocgs:
            ocg_resolved = self.resolver.resolve(ocg_ref)
            if not isinstance(ocg_resolved, dict):
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG entry")
            name = self.resolver.resolve_str(ocg_resolved.get("Name"))
            if not name:
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG name")
            key = self.ocg_key(ocg_ref, ocg_resolved)
            if key is None or key not in on_layers:
                hidden_layers.add(name)
        return frozenset(hidden_layers)

    @property
    def xref_context(self) -> SemanticContext | None:
        resolver: ObjectResolver | None = getattr(self, "resolver", None)
        return None if resolver is None else resolver.semantic_context

    def strict_xref_validation_error(self) -> str | None:
        start = XRefScanner.find_startxref(self.raw_data, semantic_context=self.xref_context)
        if start is None:
            return None
        try:
            read_section = partial(
                XRefScanner.recover_section_at,
                self.raw_data,
                recover_malformed_objects=False,
                semantic_context=self.xref_context,
            )
            for _revision in iter_xref_revisions(start, read_section):
                pass
        except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError) as error:
            return str(error)
        return None

    def brute_force_xref(self) -> dict[int, PdfXRefEntry]:
        return XRefScanner.brute_force_scan(
            self.raw_data,
            stop_at_first_trailer=not self.recovery_scan_all_revisions,
            semantic_context=self.xref_context,
        )

    def scan_xref(self) -> None:
        data = self.raw_data
        try:
            start = XRefScanner.find_startxref(data, semantic_context=self.xref_context)
        except ValueError as exc:
            raise PdfParseError("invalid xref section") from exc
        if start is None and b"startxref" in data:
            raise PdfParseError("missing startxref")
        if start is not None and start < 0:
            raise PdfParseError("invalid xref section")

        # The scan reads only the data and the context, neither of which
        # changes here, so it is made at most once.
        brute_forced: dict[int, PdfXRefEntry] | None = None

        def brute_force() -> dict[int, PdfXRefEntry]:
            nonlocal brute_forced
            if brute_forced is None:
                brute_forced = self.brute_force_xref()
            return brute_forced

        recovery_reason = None
        if start is not None:
            try:
                read_section = partial(
                    XRefScanner.recover_section_at,
                    data,
                    semantic_context=self.xref_context,
                )
                revisions = list(iter_xref_revisions(start, read_section))
                self.xref = merge_xref_sections(revision.entries for revision in revisions)
                self.trailer_dict = revisions[0].trailer
                self.repair_stale_xref_offsets(brute_force)
                self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
                root_ref = self.trailer_dict.get("Root")
                if root_ref is None or not self.is_valid_catalog_root(root_ref):
                    self.xref.update(brute_force())
                    self.xref_was_recovered = True
                    catalog_ref = self.infer_catalog_root()
                    if catalog_ref is not None:
                        self.trailer_dict = dict(self.trailer_dict)
                        self.trailer_dict["Root"] = catalog_ref
                    self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)
            except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError) as error:
                recovery_reason = str(error)
            else:
                return

        self.xref = brute_force()
        self.xref_was_recovered = True
        if recovery_reason is not None:
            self.xref_recovery_reason = recovery_reason
        if not self.xref:
            self.trailer_dict = {}
            return
        catalog_ref = self.infer_catalog_root()
        self.trailer_dict = {"Root": catalog_ref} if catalog_ref is not None else {}
        self.trailer_dict = self.merge_recovered_trailer_metadata(self.trailer_dict)

    def repair_stale_xref_offsets(
        self, brute_force: Callable[[], dict[int, PdfXRefEntry]] | None = None
    ) -> None:
        header_offset = self.pdf_header_offset()
        recovered_xref: dict[int, PdfXRefEntry] | None = None
        repaired = False
        # In use, uncompressed, at a non-negative offset: selected with
        # map() and compress() over the entry tuples, not a loop per entry.
        keys = list(self.xref)
        entries = list(self.xref.values())
        offsets = list(map(ENTRY_OFFSET, entries))
        selected = list(
            map(
                and_,
                map(
                    and_,
                    map(truth, map(ENTRY_IN_USE, entries)),
                    map(is_, map(ENTRY_OBJECT_STREAM, entries), repeat(None), strict=False),
                    strict=True,
                ),
                map((0).__le__, offsets),
                strict=True,
            )
        )
        keys = list(compress(keys, selected))
        entries = list(compress(entries, selected))
        # Every entry's check reads only the data and that entry, and an entry
        # is only changed after its own check, so they are all made up front.
        matched = object_headers_present(self.raw_data, keys, list(compress(offsets, selected)))
        for index in compress(range(len(keys)), map(not_, matched)):
            key = keys[index]
            entry = entries[index]
            if self.xref_entry_header_nearby(key, entry):
                continue
            if header_offset:
                shifted = entry._replace(offset=entry.offset + header_offset)
                if self.xref_entry_matches_header(key, shifted):
                    self.xref[key] = shifted
                    repaired = True
                    continue
                shifted_offset = self.find_xref_entry_header(
                    key,
                    entry.offset + header_offset,
                )
                if shifted_offset is not None:
                    self.xref[key] = entry._replace(offset=shifted_offset)
                    repaired = True
                    continue
            if recovered_xref is None:
                recovered_xref = self.brute_force_xref() if brute_force is None else brute_force()
            replacement = recovered_xref.get(key)
            if (
                replacement is None
                or not replacement.in_use
                or replacement.object_stream is not None
            ):
                continue
            if replacement.offset != entry.offset:
                self.xref[key] = replacement
                repaired = True

        if repaired:
            self.xref_was_recovered = True

    def pdf_header_offset(self) -> int:
        return max(0, find_pdf_header(self.raw_data))

    def find_xref_entry_header(self, key: int, offset: int) -> int | None:
        data = self.raw_data
        expected_object_number = key >> 16
        expected_generation_number = key & 0xFFFF
        search_start = max(0, offset - 1024)
        search_end = min(len(data), offset + 1024)
        for parsed_offset, object_number, generation_number in iter_indirect_object_headers(
            data,
            search_start,
            search_end,
            allow_prefix_before_start=True,
        ):
            if (
                object_number == expected_object_number
                and generation_number == expected_generation_number
            ):
                return parsed_offset
        return None

    def xref_entry_matches_header(self, key: int, entry: PdfXRefEntry) -> bool:
        return object_headers_present(self.raw_data, [key], [entry.offset])[
            0
        ] or self.xref_entry_header_nearby(key, entry)

    def xref_entry_header_nearby(self, key: int, entry: PdfXRefEntry) -> bool:
        """xref_entry_matches_header past its exact check: the first header the
        recovering scan finds within 64 bytes, if it is this entry's, at its offset."""
        data = self.raw_data
        offset = entry.offset
        data_len = len(data)
        if offset < 0 or offset >= data_len:
            return False
        search_end = min(data_len, offset + 64)
        for parsed_offset, object_number, generation_number in iter_indirect_object_headers(
            data,
            offset,
            search_end,
            allow_prefix_before_start=True,
        ):
            return (
                parsed_offset == offset
                and object_number == key >> 16
                and generation_number == key & 0xFFFF
            )
        return False

    def is_valid_catalog_root(self, root_ref: object) -> bool:
        resolver = ObjectResolver(
            self.raw_data,
            self.xref,
            semantic_context=self.xref_context,
        )
        try:
            root = resolver.resolve(root_ref)
            if not isinstance(root, dict):
                return False
            if recover_pdf_name(root.get("Type")) != "Catalog":
                return False
            pages = resolver.resolve(root.get("Pages"))
            if not isinstance(pages, dict):
                return False
            node_type = resolve_page_tree_node_type(resolver, pages)
            if node_type != "Pages":
                return False
            kids = resolver.resolve(pages.get("Kids"))
            count = resolver.resolve(pages.get("Count"))
            return isinstance(kids, list) or (type(count) is int and count >= 0)
        except Exception:
            return False
        finally:
            resolver.close()

    def infer_catalog_root(self) -> PdfReference | None:
        data = self.raw_data
        object_cache: ResolvedObjectCache = {}
        resolver = ObjectResolver(
            self.raw_data,
            self.xref,
            semantic_context=self.xref_context,
        )
        lexer = PdfLexer(data, semantic_context=self.xref_context)
        entries_by_ref = {
            (k >> 16, k & 0xFFFF): entry for k, entry in self.xref.items() if entry.in_use
        }

        def resolve_for_inference(value: object, depth: int = 0) -> object:
            if depth > 12:
                return None
            if not isinstance(value, PdfReference):
                return value
            key = (value.object_number, value.generation_number)
            if key in object_cache:
                return object_cache[key]
            entry = entries_by_ref.get(key)
            if entry is None and value.generation_number != 0:
                key = (value.object_number, 0)
                entry = entries_by_ref.get(key)
            if entry is None:
                return None
            if entry.object_stream is not None:
                try:
                    resolved = resolver.resolve(value)
                except Exception:
                    return None
                object_cache[key] = resolved
                return resolved
            lexer.rewind(entry.offset)
            try:
                resolved = lexer.parse_indirect_object()
            except Exception:
                return None
            object_cache[key] = resolved
            return resolved

        def page_tree_score(node: object, depth: int = 0, seen: set[int] | None = None) -> int:
            if depth > MAX_PAGE_TREE_DEPTH:
                return -1000
            if seen is None:
                seen = set()
            node = resolve_for_inference(node, depth)
            if not isinstance(node, dict):
                return -100
            marker = id(node)
            if marker in seen:
                return -100
            seen.add(marker)
            node_type = recover_pdf_name(resolve_for_inference(node.get("Type"), depth + 1))
            if node_type is None:
                node_type = infer_page_tree_node_type(node)
            if node_type == "Page":
                score = 10
                if node.get("Contents") is not None:
                    score += 3
                if node.get("MediaBox") is not None:
                    score += 2
                return score
            if node_type != "Pages":
                return -50
            kids = resolve_for_inference(node.get("Kids"), depth + 1)
            count = resolve_for_inference(node.get("Count"), depth + 1)
            score = 15
            if type(count) is int and count >= 0:
                score += min(count, 20)
            if not isinstance(kids, list) or not kids:
                return score - 20
            child_scores = [page_tree_score(kid, depth + 1, seen.copy()) for kid in kids[:32]]
            valid_children = [child_score for child_score in child_scores if child_score > 0]
            if not valid_children:
                return score - 30
            return score + sum(valid_children)

        def catalog_score(obj: object) -> int:
            if not isinstance(obj, dict):
                return -1000
            type_name = recover_pdf_name(obj.get("Type"))
            pages = obj.get("Pages")
            score = 0
            if type_name == "Catalog":
                score += 100
            elif pages is not None:
                score += 25
            else:
                return -100
            if pages is not None:
                pages_score = page_tree_score(pages)
                if pages_score <= 0:
                    score -= 150
                else:
                    score += pages_score
            for key in ("Outlines", "Names", "Dests", "AcroForm", "PageLabels"):
                if obj.get(key) is not None:
                    score += 2
            return score

        def select_catalog_root() -> PdfReference | None:
            candidates = sorted(
                {
                    (
                        k >> 16,
                        k & 0xFFFF,
                        entry.offset if entry.object_stream is None else 0,
                        entry.object_stream is not None,
                    )
                    for k, entry in self.xref.items()
                    if entry.in_use
                    and (
                        (entry.object_stream is None and entry.offset >= 0)
                        or entry.object_stream is not None
                    )
                },
                key=lambda item: (item[3], item[2], item[0]),
            )
            scored: list[tuple[int, int, int, int]] = []
            for obj_num, gen_num, offset, compressed in candidates:
                if compressed:
                    try:
                        obj = resolver.resolve(PdfReference(obj_num, gen_num))
                    except Exception:
                        continue
                else:
                    lexer.rewind(offset)
                    try:
                        obj = lexer.parse_indirect_object()
                    except Exception:
                        continue
                object_cache[(obj_num, gen_num)] = obj
                score = catalog_score(obj)
                if score > -100:
                    scored.append((score, -offset, obj_num, gen_num))
            if not scored:
                return None
            scored.sort(reverse=True)
            ignored, ignored, obj_num, gen_num = scored[0]
            return PdfReference(obj_num, gen_num)

        try:
            return select_catalog_root()
        finally:
            object_cache.clear()
            lexer.close()
            resolver.close()

    def merge_recovered_trailer_metadata(self, trailer: PdfDict) -> PdfDict:
        missing_keys = [key for key in TRAILER_METADATA_KEYS if trailer.get(key) is None]
        if not missing_keys:
            return trailer
        if not getattr(self, "xref_was_recovered", False) and not any(
            self.raw_data.find(b"/" + key.encode("ascii")) >= 0 for key in missing_keys
        ):
            return trailer
        if missing_keys == ["Encrypt"] and not getattr(self, "xref_was_recovered", False):
            return trailer
        if missing_keys == ["Encrypt"] and self.raw_data.find(b"Encrypt") < 0:
            return trailer
        recovered = self.infer_trailer_metadata()
        if not recovered:
            return trailer
        merged = dict(trailer)
        for key, value in recovered.items():
            if merged.get(key) is None:
                merged[key] = value
        return merged

    def infer_trailer_metadata(self) -> PdfDict:
        metadata: PdfDict = {}

        for candidate in self.iter_literal_trailer_dictionaries():
            for key in TRAILER_METADATA_KEYS:
                if key not in candidate:
                    continue
                value = candidate[key]
                if self.is_valid_trailer_metadata_value(key, value):
                    metadata[key] = value

        missing_keys = [key for key in TRAILER_METADATA_KEYS if key not in metadata]
        if not missing_keys:
            return metadata
        if missing_keys == ["Encrypt"] and metadata and self.raw_data.find(b"Encrypt") < 0:
            return metadata

        for candidate in self.iter_recoverable_xref_stream_dictionaries():
            for key in missing_keys:
                if key not in candidate:
                    continue
                value = candidate[key]
                if self.is_valid_trailer_metadata_value(key, value):
                    metadata[key] = value
        return metadata

    def iter_literal_trailer_dictionaries(self) -> Iterator[PdfDict]:
        data = self.raw_data
        lexer = PdfLexer(data, semantic_context=self.xref_context)
        try:
            search_from = 0
            while True:
                marker = data.find(b"trailer", search_from)
                if marker < 0:
                    break
                search_from = marker + len(b"trailer")
                dict_start = data.find(b"<<", search_from, search_from + 4096)
                if dict_start < 0:
                    continue
                lexer.rewind(dict_start)
                try:
                    candidate = lexer.parse_dictionary()
                except Exception:
                    continue
                yield candidate
        finally:
            lexer.close()

    def iter_recoverable_xref_stream_dictionaries(self) -> Iterator[PdfDict]:
        data = self.raw_data
        data_len = len(data)
        # Object starts in file order, so each candidate's span can be bounded
        # by the next one rather than by a guess at how long a dictionary runs.
        starts = sorted(
            {
                entry.offset
                for entry in self.xref.values()
                if entry.in_use and entry.object_stream is None and 0 <= entry.offset < data_len
            }
        )
        # Walking the offsets rather than the entries: a damaged xref can point
        # many object numbers at one offset, and the object living there is the
        # same object however many keys reach it. The keys are not otherwise
        # needed here, so distinct offsets in file order are both the shorter
        # loop and the one whose neighbour is already the span boundary.
        last = len(starts) - 1
        lexer = PdfLexer(data, semantic_context=self.xref_context)
        try:
            for index, offset in enumerate(starts):
                end = starts[index + 1] if index < last else data_len
                if not self.may_be_xref_stream(data, offset, end):
                    continue
                lexer.rewind(offset)
                try:
                    obj = lexer.parse_indirect_object()
                except Exception:
                    continue
                if not isinstance(obj, PdfStream):
                    continue
                dictionary = obj.dictionary
                if recover_pdf_name(dictionary.get("Type")) == "XRef" or (
                    dictionary.get("W") is not None and dictionary.get("Size") is not None
                ):
                    yield dictionary
        finally:
            lexer.close()

    def may_be_xref_stream(self, data: bytes | mmap.mmap, offset: int, end: int) -> bool:
        """Rule out an object as an xref stream by reading bytes, not parsing it.

        The caller only wants dictionaries that declare Type /XRef, or that
        carry both W and Size, so a span holding none of those literals cannot
        be one. Parsing every object in the file to discover that is what made
        opening a large document expensive.

        The span runs to the next object rather than over a fixed window,
        because nothing bounds how much dictionary may precede the markers: a
        long Index array or a run of comments can push them arbitrarily far
        into the object. Searching a fixed prefix would skip such a stream and
        lose the trailer metadata it carries.

        A span containing "#" falls through to the parse, since a hex-escaped
        name would not match these literals. Searching a whole object span can
        also match bytes inside stream data, but a false positive only costs
        the parse that used to happen anyway.
        """
        if data.find(b"#", offset, end) >= 0 or data.find(b"XRef", offset, end) >= 0:
            return True
        return data.find(b"/W", offset, end) >= 0 and data.find(b"/Size", offset, end) >= 0

    def is_valid_trailer_metadata_value(self, key: str, value: object) -> bool:
        if key == "Info":
            return isinstance(value, (PdfReference, dict))
        if key == "ID":
            return isinstance(value, (list, tuple)) and len(value) > 0
        if key == "Encrypt":
            return value is not None
        return key == "AuthCode"


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


# Past this an object number does not fit the kernel's 63 bits, and one
# that fits never compares equal to a header run that does not.
HUGE_OBJECT_NUMBER = 1 << 63


ENTRY_OFFSET = itemgetter(0)
ENTRY_IN_USE = itemgetter(2)
ENTRY_OBJECT_STREAM = itemgetter(3)


def object_headers_present(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
    """For each key and offset, whether the object header for key starts at offset.

    The header is "N G obj" with the maximal digit and whitespace runs and a
    delimiter or the end of the data after it, and N and G read as integers;
    object_headers_match checks them all in one pass. Its columns are made
    in numpy when every key and offset fits an int64, which is every real
    table; otherwise an entry at a time.
    """
    data_len = len(data)
    try:
        key_column = numpy.array(keys, dtype=numpy.int64)
        offset_column = numpy.array(offsets, dtype=numpy.int64)
    except OverflowError:
        return object_headers_present_by_entry(data, keys, offsets)
    numbers = key_column >> 16
    generations = key_column & 0xFFFF
    offset_column[(offset_column < 0) | (offset_column >= data_len)] = -1
    states = object_headers_match(data, numbers, generations, offset_column)
    return (states == 1).tolist()


def object_headers_present_by_entry(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
    """object_headers_present for keys or offsets past int64."""
    data_len = len(data)
    numbers = array("q")
    generations = array("q")
    clamped = array("q")
    for key, offset in zip(keys, offsets, strict=True):
        number = key >> 16
        numbers.append(number if number < HUGE_OBJECT_NUMBER else -2)
        generations.append(key & 0xFFFF)
        clamped.append(offset if 0 <= offset < data_len else -1)
    states = object_headers_match(data, numbers, generations, clamped)
    present = [state == 1 for state in states.tolist()]
    for index, state in enumerate(states.tolist()):
        if state == 2:
            # The header is there with a run too long for 63 bits, and so is
            # the number it must equal: compare them as integers.
            offset = clamped[index]
            end = offset
            while end < data_len and 0x30 <= data[end] <= 0x39:
                end += 1
            present[index] = int(bytes(data[offset:end])) == keys[index] >> 16
    return present


TRAILER_METADATA_KEYS = ("Info", "ID", "Encrypt", "AuthCode")


__all__ = ("MAX_PAGE_TREE_DEPTH", "TRAILER_METADATA_KEYS")
