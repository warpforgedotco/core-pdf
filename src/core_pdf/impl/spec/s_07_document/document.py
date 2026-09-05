# SPDX-License-Identifier: AGPL-3.0-only
"""Spec-level document: catalog, trailer, and security setup."""

from __future__ import annotations

import contextlib
import mmap
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from os import PathLike
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Generic, Self, TypeVar, cast

from core_pdf.impl._impl.model.page_selection import PageSelection, resolve_page_selection
from core_pdf.impl.exceptions import (
    PdfDocumentClosedError,
    PdfParseError,
    PdfSourceError,
    PdfUnsupportedError,
)
from core_pdf.impl.spec.s_07_document.document_labels import (
    MAX_PAGE_TREE_DEPTH,
    format_page_label,
    resolve_page_tree_node_type,
)
from core_pdf.impl.spec.s_07_document.document_xref import DocumentXRefMixin
from core_pdf.impl.spec.s_07_document.fields import collect_field_records
from core_pdf.impl.spec.s_07_document.metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.spec.s_07_document.page import PAGE_INHERITED_KEYS
from core_pdf.impl.spec.s_07_document.records import (
    RawEmbeddedFile,
    RawFormField,
    RawNamedDestination,
    RawOutlineItem,
)
from core_pdf.impl.spec.s_07_security.pdf_mac import (
    validate_pdf_mac_extension,
    validate_pdf_mac_if_present,
)
from core_pdf.impl.spec.s_07_security.standard import create_standard_security_handler
from core_pdf.impl.spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.trees import (
    iter_name_tree_items,
    iter_number_tree_items,
)
from core_pdf.impl.spec.s_07_syntax.types import (
    CachedPdfObject,
    Decipher,
    InheritedValueMap,
    PdfArray,
    PdfDict,
    PdfObject,
)
from core_pdf.impl.spec.s_07_syntax.xref import PdfXRefEntry
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_09_fonts.fallback import internal_RasterFontRepository
from core_pdf.impl.spec.s_14_structure.tree import StructureTree
from core_pdf.impl.types import (
    PathSource,
    PdfByteBuffer,
    PdfReference,
    PdfSource,
    SeekableBinaryReader,
)

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_09_fonts.fallback import (
        RasterFontProviderLike,
        internal_RasterFontRepository,
    )


internal_PageT = TypeVar("internal_PageT")


@dataclass(frozen=True, slots=True)
class internal_PageNode:
    """A source page dictionary and the effective values inherited by that page."""

    dictionary: PdfDict
    inherited_values: InheritedValueMap


class internal_PageLookup(Generic[internal_PageT]):
    """Lazy page snapshot shared only by one navigation or structure operation."""

    __slots__ = (
        "document",
        "internal_nodes",
        "internal_indexes",
        "internal_pages",
        "internal_names",
    )

    def __init__(self, document: PdfDocument[internal_PageT]) -> None:
        self.document = document
        self.internal_nodes: tuple[internal_PageNode, ...] | None = None
        self.internal_indexes: dict[int, int] = {}
        self.internal_pages: tuple[internal_PageT, ...] | None = None
        self.internal_names: dict[str, RawNamedDestination] | None = None

    @property
    def nodes(self) -> tuple[internal_PageNode, ...]:
        if self.internal_nodes is None:
            self.internal_nodes = tuple(self.document.internal_iter_page_nodes())
            for index, node in enumerate(self.internal_nodes):
                self.internal_indexes.setdefault(id(node.dictionary), index)
        return self.internal_nodes

    @property
    def pages(self) -> tuple[internal_PageT, ...]:
        if self.internal_pages is None:
            self.internal_pages = self.document.internal_build_pages(self.nodes)
        return self.internal_pages

    def page_index_for(self, page_obj: object) -> int | None:
        from core_pdf.impl.spec.s_07_document.page import PdfPage

        if isinstance(page_obj, PdfPage):
            return page_obj.page_number - 1
        if not isinstance(page_obj, dict):
            return None
        nodes = self.nodes
        index = self.internal_indexes.get(id(page_obj))
        if index is not None:
            return index
        page_struct_parents = page_obj.get("StructParents")
        if page_struct_parents is not None:
            for index, node in enumerate(nodes):
                if node.dictionary.get("StructParents") == page_struct_parents:
                    return index
        for index, node in enumerate(nodes):
            if node.dictionary == page_obj:
                return index
        signature = self.document.recovered_page_signature(cast(PdfDict, page_obj))
        for index, node in enumerate(nodes):
            if self.document.recovered_page_signature(node.dictionary) == signature:
                return index
        return None

    def resolve_named_destination(self, name: str) -> RawNamedDestination | None:
        if self.internal_names is None:
            self.internal_names = self.document.named_destinations(internal_lookup=self)
        return self.internal_names.get(name)


def internal_unresolved_destination(name: str) -> RawNamedDestination:
    return RawNamedDestination(page_index=None, type=None, args=[], raw=name)


class PdfDocument(
    DocumentXRefMixin,
    Generic[internal_PageT],
):
    # Class-level default; subclasses assign their page factory per instance.
    page_class: type | None = None

    __slots__ = (
        "source",
        "password",
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
        "internal_closed",
    )

    source: PdfSource
    password: str
    raw_data: bytes | mmap.mmap
    xref: dict[int, PdfXRefEntry]
    trailer_dict: PdfDict
    decipher: Decipher | None
    resolver: ObjectResolver
    file_handle: BinaryIO | None
    xref_was_recovered: bool
    xref_recovery_reason: str | None
    recovery_scan_all_revisions: bool
    raster_font_provider: RasterFontProviderLike | internal_RasterFontRepository | None
    page_tree_was_recovered: bool
    internal_closed: bool

    def __init__(
        self,
        source: PdfSource,
        password: str = "",
        *,
        recovery_scan_all_revisions: bool = True,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> None:
        self.internal_closed = False
        self.source = source
        self.password = password
        self.file_handle = None
        self.raw_data = b""
        self.decipher = None
        self.xref = {}
        self.trailer_dict = {}
        self.xref_was_recovered = False
        self.xref_recovery_reason = None
        self.recovery_scan_all_revisions = recovery_scan_all_revisions
        self.raster_font_provider = internal_RasterFontRepository(raster_font_provider)
        self.page_tree_was_recovered = False
        try:
            self.raw_data = self.load_data(source)
            self.scan_xref()

            self.resolver = ObjectResolver(self.raw_data, self.xref, self.trailer_dict)
            self.init_security(password)
            self.resolver.decipher = self.decipher
        except BaseException:
            self.close()
            raise

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
        return self.internal_closed

    def close(self) -> None:
        if self.internal_closed:
            return
        self.internal_closed = True

        resolver = getattr(self, "resolver", None)
        if resolver is not None:
            resolver.close()

        raster_fonts = self.raster_font_provider
        if isinstance(raster_fonts, internal_RasterFontRepository):
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
        return cast(PdfDict, root)

    def get_metadata(self) -> MetadataRecord:
        return resolve_metadata(
            self.resolver,
            self.trailer_dict,
            recover=self.recovery_enabled,
        )

    def internal_catalog_dict(self, key: str, *, recoverable: bool = False) -> PdfDict | None:
        """A catalog entry that must be a dictionary when it is present at all.

        ``recoverable`` drops an entry that is present but not a dictionary,
        instead of raising, once the document has already been reconstructed.
        """
        value = self.resolver.resolve(self.catalog().get(key))
        if value is None:
            return None
        if isinstance(value, dict):
            return cast(PdfDict, value)
        if recoverable and self.recovery_enabled:
            return None
        raise ValueError(f"invalid {key} dictionary")

    @property
    def structure(self) -> StructureTree | None:
        root = self.internal_catalog_dict("StructTreeRoot")
        return (
            None
            if root is None
            else StructureTree(self, root, internal_lookup=internal_PageLookup(self))
        )

    @property
    def mark_info(self) -> PdfDict | None:
        return self.internal_catalog_dict("MarkInfo")

    @property
    def recovery_enabled(self) -> bool:
        """Whether the document was reconstructed and so needs lenient traversal."""
        return self.xref_was_recovered or self.page_tree_was_recovered

    # Source loading and security

    def load_data(self, source: PdfSource) -> PdfByteBuffer:
        if isinstance(source, (str, PathLike)):
            if isinstance(source, str) and source.startswith("%PDF"):
                return source.encode("latin-1")
            file_handle = open(cast(PathSource, source), "rb")  # noqa: SIM115
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
        seekable: SeekableBinaryReader | None = None
        if callable(tell) and callable(seek):
            seekable = cast(SeekableBinaryReader, source)
            try:
                position = seekable.tell()
                seekable.seek(0)
            except (OSError, TypeError, ValueError):
                position = None
                seekable = None
        try:
            raw = reader.read()
        except OSError as exc:
            raise PdfSourceError(str(exc)) from exc
        finally:
            if position is not None and seekable is not None:
                seekable.seek(position)
        return raw if isinstance(raw, bytes) else bytes(raw)

    def try_mmap_reader(self, source: object) -> mmap.mmap | None:
        fileno = getattr(source, "fileno", None)
        if not callable(fileno):
            return None
        try:
            fd = fileno()
        except (OSError, TypeError, ValueError):
            return None
        try:
            return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except ValueError as error:
            raise PdfSourceError("PDF source is empty") from error
        except OSError:
            return None

    def init_security(self, password: str) -> None:
        encrypt_ref = self.trailer_dict.get("Encrypt")
        if encrypt_ref is None:
            # ISO/TS 32004:2024, Table 5 defines AuthCode only for encrypted
            # documents whose Encrypt dictionary has V >= 5.
            if "AuthCode" in self.trailer_dict:
                raise PdfUnsupportedError("AuthCode requires an encrypted document")
            return

        encrypt_dict = self.resolver.resolve_dict(encrypt_ref)
        if not isinstance(encrypt_dict, dict):
            raise PdfUnsupportedError("Invalid Encrypt dictionary")

        docid: object = self.trailer_dict.get("ID")
        if docid is None:
            docid = [b""]
        if isinstance(docid, PdfReference):
            docid = self.resolver.resolve(docid)
        if not isinstance(docid, (list, tuple)) or len(docid) == 0:
            raise PdfUnsupportedError("Invalid trailer ID array")
        docid_list: Sequence[object] = docid

        security_handler = create_standard_security_handler(docid_list, encrypt_dict, password)
        # ISO/TS 32004:2024 integrity validation authenticates the complete
        # serialized file. Perform it before installing the object decipher so
        # no decrypted string, stream, catalog, or page can be exposed first.
        has_pdf_mac = validate_pdf_mac_if_present(
            self.raw_data,
            self.trailer_dict,
            security_handler,
        )
        self.decipher = security_handler.decrypt
        if has_pdf_mac:
            # ISO/TS 32004:2024, clause 4 and Table 1 require this exact
            # declaration. Its text-string fields are encrypted, so resolve it
            # only after authenticating the complete file and installing the
            # decipher, but still before returning the document to the caller.
            self.resolver.decipher = self.decipher
            extensions = self.resolve(self.catalog().get("Extensions"))
            iso_declarations: object = None
            if isinstance(extensions, dict):
                iso_declarations = self.resolve(extensions.get("ISO_"))
            if isinstance(iso_declarations, list):
                iso_declarations = [self.resolve(value) for value in iso_declarations]
            validate_pdf_mac_extension(iso_declarations)

    # Page tree and page labels

    def internal_discover_page_nodes(self) -> Iterator[internal_PageNode]:
        """Recover likely page dictionaries when the declared page tree is unusable."""
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
            obj = cast(PdfDict, obj)
            marker = id(obj)
            if marker in seen_objects:
                continue
            seen_objects.add(marker)
            pages_score = self.pages_candidate_score(obj)
            if pages_score > 0:
                pages_nodes.append((pages_score, entry.offset, key >> 16, obj))
            score = self.page_candidate_score(obj)
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
            yield internal_PageNode(
                page_dict,
                self.internal_recovered_page_values(page_dict, inherited_sources),
            )

    def page_candidate_score(self, obj: PdfDict) -> int:
        node_type = resolve_page_tree_node_type(self.resolver, obj)
        if node_type == "Pages" or node_type not in (None, "Page"):
            return -100

        # A recovered leaf without an explicit /Type needs page content or an
        # annotation to distinguish it from outline destinations and other
        # dictionaries that happen to carry /Parent, /MediaBox, or /Resources.
        explicit_type = normalize_pdf_name(obj.get("Type"))
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

    def pages_candidate_score(self, obj: PdfDict) -> int:
        if resolve_page_tree_node_type(self.resolver, obj) != "Pages":
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

    def internal_recovered_page_values(
        self, page_dict: PdfDict, pages_nodes: list[PdfDict]
    ) -> InheritedValueMap:
        values = {
            key: cast(CachedPdfObject, value)
            for key in PAGE_INHERITED_KEYS
            if (value := page_dict.get(key)) is not None
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
                sources.append(cast(PdfDict, parent_obj))
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

        return collect_inherited_values(node, tuple(keys), resolve_ref)

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
        for page_node in self.internal_iter_page_nodes():
            yield page_node.dictionary

    def internal_recovered_page_nodes(self) -> list[internal_PageNode]:
        discovered = list(self.internal_discover_page_nodes())
        if discovered:
            self.page_tree_was_recovered = True
        return discovered

    def internal_iter_page_nodes(self) -> Iterator[internal_PageNode]:
        def inherited_from_pages_node(
            node: PdfDict, inherited: InheritedValueMap | None
        ) -> InheritedValueMap:
            values = dict(inherited or {})
            for key in PAGE_INHERITED_KEYS:
                value = node.get(key)
                if value is not None:
                    values[key] = cast(CachedPdfObject, value)
            return values

        def traverse(
            node: object,
            depth: int = 0,
            inherited: InheritedValueMap | None = None,
        ) -> Iterator[internal_PageNode]:
            if depth > MAX_PAGE_TREE_DEPTH:
                raise ValueError("invalid page tree depth")
            node = self.resolver.resolve(node)
            if not isinstance(node, dict):
                if depth == 0:
                    raise ValueError("invalid page tree node")
                return
            node = cast(PdfDict, node)
            node_type = resolve_page_tree_node_type(self.resolver, node)
            if node_type == "Pages":
                kids = self.resolver.resolve(node.get("Kids"))
                if kids is None:
                    raise ValueError("invalid page tree Kids array")
                if not isinstance(kids, list):
                    raise ValueError("invalid page tree Kids array")
                node_inherited = inherited_from_pages_node(node, inherited)
                for kid in kids:
                    yield from traverse(kid, depth + 1, node_inherited)
            elif node_type == "Page":
                # Keep the resolver's source identity: widgets, destinations,
                # and structure references point to this exact dictionary.
                # The Kids traversal also supplies inheritance when a damaged
                # leaf has a missing or unusable Parent link.
                yield internal_PageNode(node, inherited_from_pages_node(node, inherited))
            else:
                raise ValueError("invalid page tree node")

        try:
            catalog = self.catalog()
            pages_ref = catalog.get("Pages")
            if pages_ref is None:
                raise ValueError("missing page tree root")
            pages_node = self.resolver.resolve(pages_ref)
            if not isinstance(pages_node, dict):
                raise ValueError("invalid page tree root")
            page_dicts = list(traverse(pages_node))
            if page_dicts:
                yield from page_dicts
                return
            discovered = self.internal_recovered_page_nodes()
            if discovered:
                yield from discovered
                return
        except (PdfParseError, ValueError):
            discovered = self.internal_recovered_page_nodes()
            if discovered:
                yield from discovered
                return
            return

    def page_count(self) -> int:
        if self.page_tree_was_recovered:
            return len(self.build_page_dicts())
        try:
            catalog = self.catalog()
            pages_ref = catalog.get("Pages")
            if pages_ref is None:
                raise ValueError("missing page tree root")
            pages_node = self.resolver.resolve(pages_ref)
            if not isinstance(pages_node, dict):
                raise ValueError("invalid page tree root")
            count = self.resolver.resolve(pages_node.get("Count"))
            if type(count) is int and count >= 0:
                return count
        except (PdfParseError, ValueError):
            return len(self.build_page_dicts())
        return len(self.build_page_dicts())

    def build_page_dicts(self) -> list[PdfDict]:
        return list(self.iter_page_dicts())

    @property
    def pages(self) -> tuple[internal_PageT, ...]:
        return self.internal_build_pages(self.internal_iter_page_nodes())

    def internal_build_pages(
        self, nodes: Iterable[internal_PageNode]
    ) -> tuple[internal_PageT, ...]:
        page_class = self.page_class
        if page_class is None:
            from core_pdf.impl.spec.s_07_document.page import PdfPage

            page_class = PdfPage
        factory = cast(Callable[..., internal_PageT], page_class)
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
        return self.build_page_labels()

    def page_label(self, page_index: int) -> str | None:
        labels = self.page_labels
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
            (page_index, cast(PdfDict, spec))
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
            page_count = len(self.build_page_dicts())
        labels: list[str] = []
        spec_pos = 0
        current_index, current_spec = specs[0]
        for page_index in range(page_count):
            while spec_pos + 1 < len(specs) and page_index >= specs[spec_pos + 1][0]:
                spec_pos += 1
                current_index, current_spec = specs[spec_pos]
            labels.append(format_page_label(current_spec, page_index - current_index, self.resolve))
        return labels

    def page_index_for(self, page_obj: object) -> int | None:
        return internal_PageLookup(self).page_index_for(page_obj)

    def selected_page_indexes(self, pages: PageSelection | None = None) -> list[int]:
        return resolve_page_selection(pages, len(self.pages))

    def iter_selected_pages(
        self, pages: PageSelection | None = None
    ) -> Iterator[tuple[int, internal_PageT]]:
        page_objects = self.pages
        for page_index in resolve_page_selection(pages, len(page_objects)):
            yield page_index, page_objects[page_index]

    # Navigation

    def iter_outlines(self) -> list[RawOutlineItem]:
        outlines = self.resolver.resolve(self.catalog().get("Outlines"))
        if outlines is None:
            return []
        if not isinstance(outlines, dict):
            raise ValueError("invalid Outlines dictionary")
        first = self.resolver.resolve(outlines.get("First"))
        if first is None:
            return []
        return self.walk_outlines(first, 0)

    def walk_outlines(
        self,
        item: object,
        level: int,
        *,
        internal_lookup: internal_PageLookup[internal_PageT] | None = None,
    ) -> list[RawOutlineItem]:
        if internal_lookup is None:
            internal_lookup = internal_PageLookup(self)
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
                # /A is very often an indirect reference, so it has to be
                # resolved before it can be recognised as an action dictionary.
                action = self.resolver.resolve(current.get("A"))
                if (
                    isinstance(action, dict)
                    and self.resolver.resolve_name(action.get("S")) == "GoTo"
                ):
                    dest = action.get("D")
            try:
                result.append(
                    RawOutlineItem(
                        title=title or "",
                        level=level,
                        dest=cast(PdfObject | str | None, dest),
                        page_index=self.resolve_destination(dest, internal_lookup=internal_lookup),
                        count=self.extract_outline_count(cast(PdfDict, current)),
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
                result.extend(self.walk_outlines(first, level + 1, internal_lookup=internal_lookup))
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
        self, dest: object, *, internal_lookup: internal_PageLookup[internal_PageT] | None = None
    ) -> int | None:
        if dest is None:
            return None
        normalized = self.normalize_destination_value(dest, internal_lookup=internal_lookup)
        if (
            normalized.raw is None
            and normalized.page_index is None
            and normalized.type is None
            and not normalized.args
        ):
            raise ValueError("invalid destination")
        return normalized.page_index

    def resolve_named_destination(self, name: str) -> RawNamedDestination | None:
        return self.named_destinations().get(name)

    def destination_from_list(
        self,
        resolved_list: PdfArray,
        *,
        internal_lookup: internal_PageLookup[internal_PageT] | None = None,
    ) -> RawNamedDestination:
        if not resolved_list:
            raise ValueError("invalid destination array")
        page_obj = self.resolver.resolve(resolved_list[0])
        if page_obj is None:
            raise ValueError("invalid destination page reference")
        lookup = internal_PageLookup(self) if internal_lookup is None else internal_lookup
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
        internal_lookup: internal_PageLookup[internal_PageT] | None = None,
    ) -> RawNamedDestination:
        lookup = internal_PageLookup(self) if internal_lookup is None else internal_lookup
        return self.internal_normalize_destination_value(
            val, lookup.resolve_named_destination, lookup
        )

    def internal_normalize_destination_value(
        self,
        val: object,
        resolve_name: Callable[[str], RawNamedDestination | None],
        internal_lookup: internal_PageLookup[internal_PageT],
    ) -> RawNamedDestination:
        resolved = self.resolver.resolve(val)
        if isinstance(resolved, dict):
            dest_value = resolved.get("D")
            if dest_value is not None:
                return self.internal_normalize_destination_value(
                    dest_value, resolve_name, internal_lookup
                )
        resolved_list = val if isinstance(val, list) else resolved
        if isinstance(resolved_list, tuple):
            resolved_list = list(resolved_list)
        if isinstance(resolved_list, list) and resolved_list:
            return self.destination_from_list(
                cast(PdfArray, resolved_list), internal_lookup=internal_lookup
            )
        if isinstance(resolved_list, list):
            raise ValueError("invalid destination array")

        name = self.resolver.resolve_name_like_value(resolved)
        if name is not None:
            nested = resolve_name(name)
            if nested is not None:
                return nested
        raise ValueError("invalid destination")

    def named_destinations(
        self, *, internal_lookup: internal_PageLookup[internal_PageT] | None = None
    ) -> dict[str, RawNamedDestination]:
        lookup = internal_PageLookup(self) if internal_lookup is None else internal_lookup
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
                for name, value in iter_name_tree_items(
                    dests_tree,
                    self.resolver.resolve,
                    self.resolver.resolve_str,
                    recover=self.recovery_enabled,
                ):
                    targets[name] = value

        normalized: dict[str, RawNamedDestination] = {}
        resolving: set[str] = set()

        def normalize_name(name: str) -> RawNamedDestination:
            cached = normalized.get(name)
            if cached is not None:
                return cached
            if name in resolving:
                return internal_unresolved_destination(name)
            resolving.add(name)
            try:
                target = targets.get(name)
                result = (
                    internal_unresolved_destination(name)
                    if target is None
                    else self.internal_normalize_destination_value(target, normalize_name, lookup)
                )
                normalized[name] = result
                return result
            finally:
                resolving.discard(name)

        for name in targets:
            try:
                normalize_name(name)
            except (PdfParseError, ValueError):
                # Entries in a name tree are independent. Keep a damaged or
                # dangling destination unresolved without discarding the rest.
                normalized[name] = internal_unresolved_destination(name)
        return normalized

    # Forms

    @property
    def acroform(self) -> PdfDict | None:
        # Unlike StructTreeRoot and MarkInfo, a recovered document keeps going
        # with no form rather than failing: a damaged AcroForm costs the field
        # list, not the page content every caller came for.
        return self.internal_catalog_dict("AcroForm", recoverable=True)

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
        # 12.5.6.19 lets a field with a single widget merge both dictionaries
        # into one, so a widget carrying /FT is itself a field and a missing or
        # empty catalog field tree does not mean the document has none.
        if not records or self.recovery_enabled:
            records.extend(self.discover_widget_field_records(records))
        return records

    def fields_by_page(
        self,
        pages: Sequence[internal_PageT] | None = None,
    ) -> dict[int, list[RawFormField]]:
        """Group every document field by the page index its widget(s) sit on."""
        from core_pdf.impl.spec.s_07_document.page import PdfPage

        page_sequence = self.pages if pages is None else tuple(pages)
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
        for page_node in self.internal_iter_page_nodes():
            raw_annots = self.resolver.resolve(page_node.inherited_values.get("Annots"))
            if raw_annots is None:
                continue
            annots = raw_annots if isinstance(raw_annots, list) else [raw_annots]
            for annot_ref in annots:
                annot = self.resolver.resolve(annot_ref)
                if not isinstance(annot, dict):
                    continue
                if id(annot) in seen_widgets:
                    continue
                subtype = self.resolver.resolve_name_or_text(annot.get("Subtype")) or ""
                if subtype != "Widget":
                    continue
                # A widget may be merged with its field or hang off one as a
                # kid. Collect from the root of the chain either way, so the
                # /FT, /T and /V a split field keeps on the parent still reach
                # the record.
                root = self.internal_widget_field_root(cast(PdfDict, annot))
                if id(root) in seen_widgets:
                    continue
                seen_widgets.add(id(root))
                seen_widgets.add(id(annot))
                records.extend(
                    collect_field_records(self.resolver, root, recover=self.recovery_enabled)
                )
        return records

    def internal_widget_field_root(self, annot: PdfDict) -> PdfDict:
        node = annot
        seen = {id(node)}
        for _ in range(50):
            parent = self.resolver.resolve(node.get("Parent"))
            if not isinstance(parent, dict) or id(parent) in seen:
                break
            seen.add(id(parent))
            node = cast(PdfDict, parent)
        return node

    # Attachments and optional content

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
            if record is not None:
                records.append(record)
        return records

    def embedded_file_record(self, name: str, value: object) -> RawEmbeddedFile | None:
        filespec = self.resolver.resolve(value)
        if not isinstance(filespec, dict):
            raise ValueError("invalid embedded file spec")
        filespec = cast(PdfDict, filespec)
        ef = self.resolver.resolve(filespec.get("EF"))
        if not isinstance(ef, dict):
            raise ValueError("invalid embedded file stream")
        ef = cast(PdfDict, ef)
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
        recover = self.recovery_enabled
        try:
            catalog = self.catalog()
        except ValueError:
            return frozenset()
        oc = self.resolver.resolve(catalog.get("OCProperties"))
        if oc is None:
            return frozenset()
        if not isinstance(oc, dict):
            if recover:
                return frozenset()
            raise ValueError("invalid OCProperties dictionary")
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

            on_refs = default_config.get("ON")
            if not isinstance(on_refs, list):
                on_refs = []
            for on_ref in on_refs:
                ocg_resolved = self.resolver.resolve(on_ref)
                if not isinstance(ocg_resolved, dict):
                    if recover:
                        continue
                    raise ValueError("invalid OCProperties ON entry")
                key = self.ocg_key(on_ref, ocg_resolved)
                if key is not None:
                    on_layers.add(key)

            off_refs = default_config.get("OFF")
            if not isinstance(off_refs, list):
                off_refs = []
            for off_ref in off_refs:
                ocg_resolved = self.resolver.resolve(off_ref)
                if not isinstance(ocg_resolved, dict):
                    if recover:
                        continue
                    raise ValueError("invalid OCProperties OFF entry")
                key = self.ocg_key(off_ref, ocg_resolved)
                if key is not None:
                    on_layers.discard(key)

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
