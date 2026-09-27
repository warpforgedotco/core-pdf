# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
import mmap
import struct
import threading
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from functools import partial
from os import PathLike
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy

from core_pdf.impl.caches import DocumentCaches
from core_pdf.impl.document_metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.document_page import PAGE_INHERITED_KEYS, PdfPage, goto_action_destination
from core_pdf.impl.document_records import (
    RawEmbeddedFile,
    RawFormField,
    RawNamedDestination,
    RawOutlineItem,
)
from core_pdf.impl.document_standards import (
    bootstrap_security_context,
    discover_document_standards,
    discover_header_standards,
    discover_profile_claims,
    find_pdf_header,
    preserve_historical_version,
)
from core_pdf.impl.document_structure import StructureTree, resolve_optional_dict
from core_pdf.impl.exceptions import (
    ExtractionScope,
    PdfDocumentClosedError,
    PdfEmptySourceError,
    PdfParseError,
    PdfSourceError,
    PdfUnsupportedError,
)
from core_pdf.impl.extract_selection import extract_document
from core_pdf.impl.fonts_fallback import RasterFontRepository
from core_pdf.impl.output_model import Document
from core_pdf.impl.page_selection import PageSelection, resolve_page_selection
from core_pdf.impl.pdf_values import parse_text_string, recover_pdf_name
from core_pdf.impl.raw_media import ImageRecord
from core_pdf.impl.recovery_lexer import PdfLexer, reader_rules_for
from core_pdf.impl.recovery_policy import MalformedFn, Recovery, recovery_policy
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.recovery_trees import iter_name_tree_items, iter_number_tree_items
from core_pdf.impl.recovery_xref import StrictXRefScanner, XRefScanner, iter_indirect_object_headers
from core_pdf.impl.types import (
    MISSING,
    PageScoped,
    PdfByteBuffer,
    PdfName,
    PdfReference,
    PdfSource,
    PdfString,
)
from core_pdf_cythonized import object_headers_match
from core_pdf_spec.s_07_document import document_labels
from core_pdf_spec.s_07_document.document_labels import PageLabelStyle
from core_pdf_spec.s_07_document.fields import field_children, qualified_field_name
from core_pdf_spec.s_07_document.page import PageNode, iter_page_nodes
from core_pdf_spec.s_07_security.document import initialize_document_security
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.inherited_values import (
    collect_inherited_values,
    inherited_dictionary_value,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import (
    InheritedValueMap,
    PdfArray,
    PdfDict,
    PdfValueResolver,
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
    from core_pdf.impl.fonts_fallback import RasterFontProviderLike
    from core_pdf_spec.s_07_syntax.types import Decipher


def regions_containing(
    data: bytes | mmap.mmap, marker: bytes, starts: Sequence[int], data_len: int
) -> set[int]:
    found: set[int] = set()
    if not starts:
        return found
    last = len(starts) - 1
    position = starts[0]
    while True:
        at = data.find(marker, position)
        if at < 0:
            return found
        index = bisect_right(starts, at) - 1
        end = starts[index + 1] if index < last else data_len
        if at + len(marker) <= end:
            found.add(index)
            position = end
        else:
            position = at + 1


def xref_stream_candidates(
    data: bytes | mmap.mmap, starts: Sequence[int], data_len: int
) -> list[int]:
    """Indexes of the regions a may_be_xref_stream check would accept, in ascending order."""
    candidates = regions_containing(data, b"#", starts, data_len)
    candidates |= regions_containing(data, b"XRef", starts, data_len)
    candidates |= regions_containing(data, b"/W", starts, data_len) & regions_containing(
        data, b"/Size", starts, data_len
    )
    return sorted(candidates)


class FieldResolver(PdfValueResolver, Protocol):
    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None: ...

    def as_dict(self, value: object) -> PdfDict | None: ...


FieldTraversalNode: TypeAlias = tuple[Literal["node"], object, str, str, object, int]


FieldTraversalRecord: TypeAlias = tuple[Literal["record"], RawFormField]


FieldTraversalEntry: TypeAlias = FieldTraversalNode | FieldTraversalRecord


def field_value_text(resolver: FieldResolver, value: object) -> str:
    parts: list[str] = []
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        current = resolver.resolve(current)
        match current:
            case None:
                continue
            case list() | tuple():
                stack.extend(reversed(current))
                continue
            case PdfName(value=item_text):
                pass
            case PdfString() | bytes() | str():
                item_text = (parse_text_string(current) or "").strip()
            case int() | float() if type(current) is not bool:
                item_text = str(current)
            case _:
                continue
        if item_text:
            parts.append(item_text)
    return "\n".join(parts)


def field_record(
    resolver: FieldResolver,
    node: PdfDict,
    parent_name: str,
    parent_type: str,
    parent_value: object,
    malformed: MalformedFn,
    *,
    terminal_widget: bool = False,
) -> RawFormField:
    title = resolver.resolve_str(node.get("T"))
    name = qualified_field_name(parent_name, title)
    field_type = resolver.resolve_name_or_text(node.get("FT"), name_like=True) or parent_type
    value = inherited_dictionary_value(node, "V", parent_value, resolver.resolve)
    value_text = field_value_text(resolver, value)
    try:
        kids = field_children(None if terminal_widget else node.get("Kids"))
    except ValueError as error:
        malformed(str(error))
        kids = []
    is_widget = terminal_widget or resolver.resolve_name_or_text(node.get("Subtype")) == "Widget"
    return RawFormField(
        name,
        field_type,
        value,
        value_text,
        resolver.resolve_box(node.get("Rect")) if is_widget else None,
        node,
        kids=kids,
        widget=node if is_widget else None,
    )


def collect_field_records(
    resolver: FieldResolver,
    node: object,
    malformed: MalformedFn,
) -> list[RawFormField]:
    seen: set[int] = set()
    records: list[RawFormField] = []
    stack: list[FieldTraversalEntry] = [("node", node, "", "", None, 0)]
    while stack:
        entry = stack.pop()
        if entry[0] == "record":
            records.append(entry[1])
            continue
        _, current_node, parent_name, parent_type, parent_value, depth = entry
        if depth > 50:
            malformed("invalid AcroForm depth")
            continue
        current_node = resolver.as_dict(current_node)
        if current_node is None or id(current_node) in seen:
            malformed("invalid AcroForm field entry")
            continue
        seen.add(id(current_node))
        record = field_record(
            resolver,
            current_node,
            parent_name,
            parent_type,
            parent_value,
            malformed,
        )
        records.append(record)
        for kid in reversed(record.kids):
            resolved_kid = resolver.as_dict(kid)
            if resolved_kid is None:
                malformed("invalid AcroForm kid entry")
                continue
            if resolver.resolve_name_or_text(resolved_kid.get("Subtype")) == "Widget":
                stack.append(
                    (
                        "record",
                        field_record(
                            resolver,
                            resolved_kid,
                            record.name,
                            record.type,
                            record.value,
                            malformed,
                            terminal_widget=True,
                        ),
                    )
                )
            else:
                stack.append(
                    ("node", resolved_kid, record.name, record.type, record.value, depth + 1)
                )
    return records


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


def load_source(source: PdfSource) -> tuple[PdfByteBuffer, BinaryIO | None]:
    if isinstance(source, (str, PathLike)):
        if isinstance(source, str) and source.startswith("%PDF"):
            return source.encode("latin-1"), None
        file_handle = open(source, "rb")  # noqa: SIM115
        try:
            return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ), file_handle
        except (OSError, ValueError) as exc:
            try:
                is_empty = file_handle.seek(0, 2) == 0
            except OSError:
                is_empty = False
            file_handle.close()
            if is_empty:
                raise PdfEmptySourceError("PDF source is empty") from exc
            raise PdfSourceError(str(exc)) from exc
    if isinstance(source, bytes):
        return source, None
    if isinstance(source, (memoryview, bytearray)):
        return bytes(source), None

    mapped = try_mmap_reader(source)
    if mapped is not None:
        return mapped, None

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
    return (raw if isinstance(raw, bytes) else bytes(raw)), None


def try_mmap_reader(source: object) -> mmap.mmap | None:
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
        raise PdfEmptySourceError("PDF source is empty") from error
    except OSError:
        return None


MAX_PAGE_TREE_DEPTH = 100


def resolve_page_tree_node_type(resolver: PdfValueResolver, node: PdfDict) -> str | None:
    node_type = resolver.resolve_name(node.get("Type"))
    if node_type is not None:
        return node_type
    inferred = infer_page_tree_node_type(node, include_page_properties=False)
    if inferred is not None:
        return inferred
    if node.get("Parent") is not None:
        return "Page"
    return None


def infer_page_tree_node_type(
    node: PdfDict,
    *,
    include_page_properties: bool = True,
) -> str | None:
    if node.get("Kids") is not None:
        return "Pages"
    if node.get("Count") is not None:
        return "Pages"
    if not include_page_properties:
        return None
    for key in ("Contents", "MediaBox", "Resources", "Parent", "Annots"):
        if node.get(key) is not None:
            return "Page"
    return None


def page_tree_extent(
    resolve: Callable[[object], object], pages: PdfDict
) -> tuple[list[object] | None, int | None]:
    kids = resolve(pages.get("Kids"))
    count = resolve(pages.get("Count"))
    return (
        kids if isinstance(kids, list) else None,
        count if type(count) is int and count >= 0 else None,
    )


HUGE_OBJECT_NUMBER = 1 << 63


def object_headers_present(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
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
            offset = clamped[index]
            end = offset
            while end < data_len and 0x30 <= data[end] <= 0x39:
                end += 1
            present[index] = int(bytes(data[offset:end])) == keys[index] >> 16
    return present


TRAILER_METADATA_KEYS = ("Info", "ID", "Encrypt", "AuthCode")


def first_indexes(values: Iterable[object]) -> dict[object, int]:
    indexes: dict[object, int] = {}
    for index, value in enumerate(values):
        with contextlib.suppress(TypeError):
            indexes.setdefault(value, index)
    return indexes


class PageLookup[LookupPageT: PdfPage]:
    __slots__ = ("document", "indexes", "memo")

    def __init__(self, document: PdfDocument[LookupPageT]) -> None:
        self.document = document
        self.indexes: dict[int, int] = {}
        self.memo = DocumentCaches()

    @property
    def nodes(self) -> tuple[PageNode, ...]:
        return self.memo.get("nodes", self.build_nodes)

    def build_nodes(self) -> tuple[PageNode, ...]:
        nodes = tuple(
            PageNode(page.page_dict, page.inherited_values) for page in self.document.pages
        )
        for index, node in enumerate(nodes):
            self.indexes.setdefault(id(node.dictionary), index)
        return nodes

    @property
    def pages(self) -> tuple[LookupPageT, ...]:
        return self.document.pages

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
            struct_parents_indexes = self.memo.get(
                "struct_parents",
                lambda: first_indexes(node.dictionary.get("StructParents") for node in nodes),
            )
            index = self.first_index(
                struct_parents_indexes,
                page_struct_parents,
                (node.dictionary.get("StructParents") for node in nodes),
            )
            if index is not None:
                return index
        for index, node in enumerate(nodes):
            if node.dictionary == page_obj:
                return index
        signature_of = self.document.recovered_page_signature
        signature_indexes = self.memo.get(
            "signatures",
            lambda: first_indexes(signature_of(node.dictionary) for node in nodes),
        )
        return self.first_index(
            signature_indexes,
            signature_of(page_obj),
            (signature_of(node.dictionary) for node in nodes),
        )

    @staticmethod
    def first_index(
        indexes: dict[object, int], value: object, values: Iterable[object]
    ) -> int | None:
        try:
            return indexes.get(value)
        except TypeError:
            return next((index for index, item in enumerate(values) if item == value), None)

    def resolve_named_destination(self, name: str) -> RawNamedDestination | None:
        names = self.memo.get("names", self.collect_named_destinations)
        return names.get(name)

    def collect_named_destinations(self) -> dict[str, RawNamedDestination]:
        return self.document.named_destinations(page_lookup=self)


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
        return recovery_policy(self.recovery_enabled)

    def malformed(self, message: str) -> None:
        self.recovery.malformed(message)

    def page_count(self) -> int:
        return len(self.pages)

    def declared_page_count(self) -> int:
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
                self.malformed("invalid AcroForm Fields array")
                field_list = []
            for field in field_list:
                field_obj = self.resolver.resolve(field)
                records.extend(
                    collect_field_records(self.resolver, field_obj, self.recovery.malformed)
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
        return self.caches.get("fields_by_page", self.group_all_fields_by_page)

    def group_all_fields_by_page(self) -> dict[int, list[RawFormField]]:
        return self.group_fields_by_page(self.pages)

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
                    kid = self.resolver.as_dict(kid_ref)
                    if kid is not None and self.resolver.name_at(kid, "Subtype") == "Widget":
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
                records.extend(collect_field_records(self.resolver, root, self.recovery.malformed))
        return records

    def widget_field_root(self, annot: PdfDict) -> PdfDict:
        node = annot
        seen = {id(node)}
        for _ in range(50):
            parent = self.resolver.dict_at(node, "Parent")
            if parent is None or id(parent) in seen:
                break
            seen.add(id(parent))
            node = parent
        return node

    def embedded_files(self) -> list[RawEmbeddedFile]:
        names = self.resolver.dict_at(self.catalog(), "Names")
        if names is None:
            return []
        embedded_tree = resolve_optional_dict(
            self.resolver, names.get("EmbeddedFiles"), "invalid EmbeddedFiles name tree"
        )
        if embedded_tree is None:
            return []

        recovery = self.recovery
        records: list[RawEmbeddedFile] = []
        for name, value in iter_name_tree_items(
            embedded_tree,
            self.resolver.resolve,
            self.resolver.resolve_str,
            on_malformed=recovery.malformed,
        ):
            try:
                record = self.embedded_file_record(name, value)
            except ValueError:
                if recovery.enabled:
                    continue
                raise
            records.append(record)
        return records

    def embedded_file_record(self, name: str, value: object) -> RawEmbeddedFile:
        filespec = self.resolver.as_dict(value)
        if filespec is None:
            raise ValueError("invalid embedded file spec")
        ef = self.resolver.dict_at(filespec, "EF")
        if ef is None:
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
            obj = self.resolver.resolve_or_none(PdfReference(key >> 16, key & 0xFFFF))
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
        kids, count = page_tree_extent(self.resolver.resolve_or_none, obj)
        if kids is not None:
            score += min(len(kids), 20)
        if count is not None:
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
            parent_obj = self.resolver.resolve_or_none(parent)
            if isinstance(parent_obj, dict):
                sources.append(parent_obj)
        sources.extend(pages_nodes)
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
        return collect_inherited_values(
            node, tuple(keys), self.resolver.resolve_or_none, stop_at_malformed_parent=True
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

    def recovered_page_nodes(self) -> list[PageNode]:
        discovered = list(self.discover_page_nodes())
        if discovered:
            self.page_tree_was_recovered = True
        return discovered

    def page_tree_root(self) -> PdfDict:
        pages_ref = self.catalog().get("Pages")
        if pages_ref is None:
            raise ValueError("missing page tree root")
        pages_node = self.resolver.as_dict(pages_ref)
        if pages_node is None:
            raise ValueError("invalid page tree root")
        return pages_node

    def iter_recovered_page_nodes(self) -> Iterator[PageNode]:
        try:
            page_nodes = list(
                iter_page_nodes(
                    self.page_tree_root(),
                    self.resolver.resolve,
                    inherited_keys=PAGE_INHERITED_KEYS,
                    node_type=lambda node: resolve_page_tree_node_type(self.resolver, node),
                    on_invalid_child=lambda _node: True,
                    max_depth=MAX_PAGE_TREE_DEPTH,
                )
            )
        except PdfParseError, ValueError:
            page_nodes = []
        yield from page_nodes or self.recovered_page_nodes()

    def build_page_dicts(self) -> list[PdfDict]:
        return [page_node.dictionary for page_node in self.iter_recovered_page_nodes()]

    @property
    def xref_context(self) -> SemanticContext | None:
        return self.resolver.semantic_context

    def strict_xref_validation_error(self) -> str | None:
        return self.caches.get_keyed(
            "strict_xref_error", self.xref_context, self.read_strict_xref_error
        )

    def read_strict_xref_error(self) -> str | None:
        start = XRefScanner.find_startxref(self.raw_data, semantic_context=self.xref_context)
        if start is None:
            return None
        try:
            read_section = partial(
                StrictXRefScanner.recover_section_at,
                self.raw_data,
                semantic_context=self.xref_context,
            )
            for _revision in iter_xref_revisions(start, read_section):
                pass
        except (PdfParseError, PdfUnsupportedError, ValueError, struct.error, OSError) as error:
            return str(error)
        return None

    def brute_force_xref(self) -> dict[int, PdfXRefEntry]:
        parsed: dict[int, object] = {}
        context = self.xref_context
        xref = XRefScanner.brute_force_scan(
            self.raw_data,
            stop_at_first_trailer=not self.recovery_scan_all_revisions,
            semantic_context=context,
            parsed_objects=parsed,
        )
        self.pending_parsed_objects = (context, parsed)
        return xref

    def brute_forced_objects(self) -> dict[int, object]:
        scanned = self.pending_parsed_objects
        if scanned is None or scanned[0] != self.xref_context:
            return {}
        return scanned[1]

    def scan_xref(self) -> None:
        self.pending_parsed_objects = None
        try:
            self.scan_xref_sections()
        finally:
            parsed = self.brute_forced_objects()
            self.pending_parsed_objects = None
            self.resolver.adopt_parsed_objects(parsed, reader_rules_for(self.xref_context))

    def scan_xref_sections(self) -> None:
        data = self.raw_data
        try:
            start = XRefScanner.find_startxref(data, semantic_context=self.xref_context)
        except ValueError as exc:
            raise PdfParseError("invalid xref section") from exc
        if start is None and b"startxref" in data:
            raise PdfParseError("missing startxref")
        if start is not None and start < 0:
            raise PdfParseError("invalid xref section")

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
        selected = [
            (key, entry)
            for key, entry in self.xref.items()
            if entry.in_use and entry.object_stream is None and entry.offset >= 0
        ]
        matched = object_headers_present(
            self.raw_data, [key for key, _ in selected], [entry.offset for _, entry in selected]
        )
        for (key, entry), present in zip(selected, matched, strict=True):
            if present:
                continue
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
            root = resolver.as_dict(root_ref)
            if root is None:
                return False
            if recover_pdf_name(root.get("Type")) != "Catalog":
                return False
            pages = resolver.dict_at(root, "Pages")
            if pages is None:
                return False
            node_type = resolve_page_tree_node_type(resolver, pages)
            if node_type != "Pages":
                return False
            kids, count = page_tree_extent(resolver.resolve, pages)
            return kids is not None or count is not None
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
        unclaimed = dict(self.brute_forced_objects())

        def parse_at(offset: int) -> Any:
            parsed = unclaimed.pop(offset, MISSING)
            if parsed is not MISSING:
                return parsed
            lexer.rewind(offset)
            return lexer.parse_indirect_object()

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
                resolved = resolver.resolve_or_none(value)
                object_cache[key] = resolved
                return resolved
            try:
                resolved = parse_at(entry.offset)
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
                    if entry.in_use and (entry.object_stream is not None or entry.offset >= 0)
                },
                key=lambda item: (item[3], item[2], item[0]),
            )
            scored: list[tuple[int, int, int, int]] = []
            for obj_num, gen_num, offset, compressed in candidates:
                if compressed:
                    obj = resolver.resolve_or_none(PdfReference(obj_num, gen_num))
                else:
                    try:
                        obj = parse_at(offset)
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
        if not self.xref_was_recovered and not any(
            self.raw_data.find(b"/" + key.encode("ascii")) >= 0 for key in missing_keys
        ):
            return trailer
        if missing_keys == ["Encrypt"] and not self.xref_was_recovered:
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

        for candidate in self.literal_trailer_dictionaries():
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

    def literal_trailer_dictionaries(self) -> tuple[PdfDict, ...]:
        return self.caches.get_keyed(
            "literal_trailers", self.xref_context, self.collect_literal_trailer_dictionaries
        )

    def collect_literal_trailer_dictionaries(self) -> tuple[PdfDict, ...]:
        return tuple(self.iter_literal_trailer_dictionaries())

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
        starts = sorted(
            {
                entry.offset
                for entry in self.xref.values()
                if entry.in_use and entry.object_stream is None and 0 <= entry.offset < data_len
            }
        )
        lexer = PdfLexer(data, semantic_context=self.xref_context)
        try:
            for index in xref_stream_candidates(data, starts, data_len):
                lexer.rewind(starts[index])
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

    @property
    def outlines(self) -> tuple[Any, ...]:
        return tuple(self.iter_outlines())

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
        recovery = self.recovery
        malformed = recovery.malformed
        if level > 200:
            raise ValueError("invalid outline depth")
        if not isinstance(item, dict):
            malformed("invalid outline item")
            return []
        result: list[RawOutlineItem] = []
        current: object | None = item
        seen: set[int] = set()
        while current is not None:
            current = self.resolver.as_dict(current)
            if current is None:
                malformed("invalid outline item")
                break
            marker = id(current)
            if marker in seen:
                malformed("outline cycle detected")
                break
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
                if not recovery.enabled:
                    raise
            first = current.get("First")
            if first is not None:
                first = self.resolver.as_dict(first)
                if first is None:
                    malformed("invalid outline child")
                    current = current.get("Next")
                    continue
                result.extend(self.walk_outlines(first, level + 1, page_lookup=page_lookup))
            current = current.get("Next")
        return result

    def extract_outline_count(self, current: PdfDict) -> int:
        raw_count = current.get("Count")
        if raw_count is None:
            return 0
        current_count = self.resolver.resolve_int(raw_count)
        if current_count is None:
            self.malformed("invalid outline count")
            return 0
        return current_count

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
        page_lookup: PageLookup[PageT],
    ) -> RawNamedDestination:
        if not resolved_list:
            raise ValueError("invalid destination array")
        page_obj = self.resolver.resolve(resolved_list[0])
        if page_obj is None:
            raise ValueError("invalid destination page reference")
        page_index = page_lookup.page_index_for(page_obj)
        if page_index is None:
            raise ValueError("invalid destination page reference")
        dest_type = None
        args: PdfArray = []
        if len(resolved_list) >= 2:
            raw_type = resolved_list[1]
            dest_type = self.resolver.resolve_name_or_text(raw_type)
            if dest_type is None:
                raise ValueError("invalid destination type")
            args = list(resolved_list[2:])
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
        dests = self.resolver.dict_at(self.catalog(), "Dests")
        if dests is not None:
            for name, val in dests.items():
                resolved_name = self.resolver.resolve_name(name)
                if resolved_name is None:
                    raise ValueError("invalid named destination key")
                targets[resolved_name] = self.resolver.resolve(val)
        names = self.resolver.dict_at(self.catalog(), "Names")
        if names is not None:
            dests_tree = self.resolver.dict_at(names, "Dests")
            if dests_tree is not None:
                targets.update(
                    iter_name_tree_items(
                        dests_tree,
                        self.resolver.resolve,
                        self.resolver.resolve_str,
                        on_malformed=self.recovery.malformed,
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

    @staticmethod
    def ocg_key(ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    def oc_hidden_layers(self) -> frozenset[str]:
        return self.caches.get("hidden_layers", self.build_oc_hidden_layers)

    def build_oc_hidden_layers(self) -> frozenset[str]:
        malformed = self.recovery.malformed
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
            malformed("invalid OCProperties OCGs array")
            return frozenset()

        on_layers: set[tuple[int, int] | int] = set()
        default_config = self.resolver.resolve(oc.get("D"))
        if default_config is not None and not isinstance(default_config, dict):
            malformed("invalid OCProperties D dictionary")
            default_config = None
        if default_config is not None:
            base_state_value = default_config.get("BaseState")
            base_state = (
                self.resolver.resolve_name(base_state_value)
                if base_state_value is not None
                else None
            )
            if (base_state_value is not None and base_state is None) or base_state not in (
                None,
                "ON",
                "OFF",
                "Unchanged",
            ):
                malformed("invalid OCProperties BaseState value")
                base_state = None
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
                    ocg_resolved = self.resolver.as_dict(ref)
                    if ocg_resolved is None:
                        malformed(f"invalid OCProperties {override_name} entry")
                        continue
                    key = self.ocg_key(ref, ocg_resolved)
                    if key is not None:
                        update(key)

        hidden_layers: set[str] = set()
        for ocg_ref in ocgs:
            ocg_resolved = self.resolver.as_dict(ocg_ref)
            if ocg_resolved is None:
                malformed("invalid OCProperties OCG entry")
                continue
            name = self.resolver.resolve_str(ocg_resolved.get("Name"))
            if not name:
                malformed("invalid OCProperties OCG name")
                continue
            key = self.ocg_key(ocg_ref, ocg_resolved)
            if key is None or key not in on_layers:
                hidden_layers.add(name)
        return frozenset(hidden_layers)

    def negotiate_security(self, password: str) -> None:
        self._standards = discover_header_standards(self.raw_data)
        header = self._standards
        context = header.context if legacy_name_context(header.context) else None
        self.resolver = ObjectResolver(self.raw_data, self.xref, semantic_context=context)
        self.rescan_xref()
        if legacy_name_context(context):
            selected = bootstrap_security_context(
                header, self.raw_data, self.xref, self.trailer_dict
            )
            if not legacy_name_context(selected):
                check_security_aliases(self.trailer_dict, self.resolver)
                self.resolver.semantic_context = selected
                self.rescan_xref()

        for attempt in range(2):
            self.init_security(password)
            self.resolver.decipher = self.decipher
            self._standards = discover_document_standards(header, self.resolver, self.trailer_dict)
            self._standards = preserve_historical_version(
                self._standards,
                self.raw_data,
                self.trailer_dict,
                self.decipher,
                recovered=self.xref_was_recovered,
                trailer_context=self.resolver.semantic_context,
            )
            selected = self._standards.context
            if legacy_name_context(selected) == legacy_name_context(self.resolver.semantic_context):
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
            self.rescan_xref()

    def rescan_xref(self) -> None:
        self.scan_xref()
        self.resolver.xref = self.xref

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
    return document_labels.format_page_label(normalized, page_offset, lambda value: value)
