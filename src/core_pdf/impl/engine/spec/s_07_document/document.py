# SPDX-License-Identifier: AGPL-3.0-only
"""Spec-level document: catalog, trailer, and security setup."""

from __future__ import annotations

import contextlib
import mmap
import threading
from collections.abc import Iterator, Sequence
from os import PathLike
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Generic, Self, TypeVar, cast

from core_pdf.impl.engine.cache import ExtractionCache
from core_pdf.impl.engine.image_cache import ImageCache
from core_pdf.impl.engine.spec.s_07_document.document_labels import (
    MAX_PAGE_TREE_DEPTH,
    format_page_label,
    resolve_page_tree_node_type,
)
from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
    get_or_compute,
)
from core_pdf.impl.engine.spec.s_07_document.document_pages import (
    PAGE_INHERITED_KEYS,
    LazyPageList,
    PageListItem,
)
from core_pdf.impl.engine.spec.s_07_document.document_xref import DocumentXRefMixin
from core_pdf.impl.engine.spec.s_07_document.fields import (
    FieldTraversalEntry,
    field_value_text,
    field_widget_rect,
)
from core_pdf.impl.engine.spec.s_07_document.metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.engine.spec.s_07_document.records import (
    RawEmbeddedFile,
    RawFormField,
    RawNamedDestination,
    RawOutlineItem,
)
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    InheritedValueMap,
    InheritedValuesCache,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    collect_inherited_values,
    lookup_dict_key,
)
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_objects.trees import (
    iter_name_tree_items,
    iter_number_tree_items,
)
from core_pdf.impl.engine.spec.s_07_security.crypto_handlers import SECURITY_HANDLER_REGISTRY
from core_pdf.impl.engine.spec.s_07_security.errors import (
    PDFEncryptionError,
    PDFPasswordIncorrect,
)
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry
from core_pdf.impl.engine.spec.s_14_structure.tree import StructureTree
from core_pdf.impl.exceptions import (
    PdfDocumentClosedError,
    PdfParseError,
    PdfSourceError,
    PdfUnsupportedError,
)
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.pages import PageSelection, resolve_page_selection
from core_pdf.impl.primitives import MISSING, MissingObject, PdfReference
from core_pdf.impl.types import (
    Decipher,
    PathSource,
    PdfArray,
    PdfByteBuffer,
    PdfDict,
    PdfObject,
    PdfSource,
    SeekableBinaryReader,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
    from core_pdf.impl.engine.spec.s_09_fonts.fallback import RasterFontProviderLike


internal_PageT = TypeVar("internal_PageT", bound=PageListItem)

DOCUMENT_CACHE_FIELDS = (
    "catalog_cache",
    "metadata_cache",
    "structure_cache",
    "mark_info_cache",
    "page_dicts_cache",
    "pages_cache",
    "page_index_cache",
    "named_destinations_cache",
    "embedded_files_cache",
    "oc_layers",
    "acroform_cache",
    "fields_cache",
    "page_labels_cache",
    "page_extraction_caches",
)

OPTIONAL_DOCUMENT_CACHE_FIELDS = (
    "structure_cache",
    "mark_info_cache",
    "acroform_cache",
    "page_labels_cache",
)


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
        "catalog_cache",
        "metadata_cache",
        "structure_cache",
        "mark_info_cache",
        "page_dicts_cache",
        "pages_cache",
        "page_index_cache",
        "named_destinations_cache",
        "embedded_files_cache",
        "oc_layers",
        "acroform_cache",
        "fields_cache",
        "decoder_cache",
        "image_cache",
        "inherited_values_cache",
        "page_labels_cache",
        "page_extraction_caches",
        "internal_cache_lock",
        "xref_was_recovered",
        "xref_recovery_reason",
        "recovery_scan_all_revisions",
        "legacy_pdfminer_text_operators",
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
    catalog_cache: PdfDict | None
    metadata_cache: MetadataRecord | None
    structure_cache: StructureTree | None | MissingObject
    mark_info_cache: PdfDict | None | MissingObject
    page_dicts_cache: list[PdfDict] | None
    pages_cache: LazyPageList[internal_PageT] | None
    page_index_cache: dict[int, int] | None
    named_destinations_cache: dict[str, RawNamedDestination] | None
    embedded_files_cache: list[RawEmbeddedFile] | None
    oc_layers: dict[str, bool] | None
    acroform_cache: PdfDict | None | MissingObject
    fields_cache: list[RawFormField] | None
    decoder_cache: dict[tuple[int, int] | int, FontDecoder]
    image_cache: ImageCache
    inherited_values_cache: InheritedValuesCache
    page_labels_cache: list[str] | None | MissingObject
    page_extraction_caches: dict[int, ExtractionCache] | None
    internal_cache_lock: threading.RLock
    xref_was_recovered: bool
    xref_recovery_reason: str | None
    recovery_scan_all_revisions: bool
    legacy_pdfminer_text_operators: bool
    raster_font_provider: RasterFontProviderLike | None
    page_tree_was_recovered: bool
    internal_closed: bool

    def __init__(
        self,
        source: PdfSource,
        password: str = "",
        *,
        recovery_scan_all_revisions: bool = True,
        legacy_pdfminer_text_operators: bool = False,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> None:
        self.internal_closed = False
        self.internal_cache_lock = threading.RLock()
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
        self.legacy_pdfminer_text_operators = legacy_pdfminer_text_operators
        self.raster_font_provider = raster_font_provider
        self.page_tree_was_recovered = False
        self._initialize_document_caches()

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
        legacy_pdfminer_text_operators: bool = False,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> Self:
        return cls(
            source,
            password=password,
            recovery_scan_all_revisions=recovery_scan_all_revisions,
            legacy_pdfminer_text_operators=legacy_pdfminer_text_operators,
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

        self._clear_document_caches()

        resolver = getattr(self, "resolver", None)
        if resolver is not None:
            resolver.close()

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
        def compute() -> PdfDict:
            root_ref = lookup_dict_key(self.trailer_dict, "Root")
            if root_ref is None:
                raise ValueError("missing catalog root")
            root = self.resolve(root_ref)
            if not isinstance(root, dict):
                raise ValueError("invalid catalog root")
            return cast(PdfDict, root)

        return get_or_compute(self, "catalog_cache", compute)

    def get_metadata(self) -> MetadataRecord:
        return get_or_compute(
            self,
            "metadata_cache",
            lambda: resolve_metadata(
                self.resolver,
                self.trailer_dict,
                recover=document_recovery_enabled(self),
            ),
        )

    @property
    def structure(self) -> StructureTree | None:
        with document_cache_lock(self):
            structure = self.structure_cache
            if structure is not MISSING:
                return cast(StructureTree | None, structure)
            resolved_root = self.resolver.resolve(lookup_dict_key(self.catalog(), "StructTreeRoot"))
            if resolved_root is None:
                self.structure_cache = None
                return None
            if not isinstance(resolved_root, dict):
                raise ValueError("invalid StructTreeRoot dictionary")
            structure = StructureTree(self, cast(PdfDict, resolved_root))
            self.structure_cache = structure
            return structure

    @property
    def mark_info(self) -> PdfDict | None:
        with document_cache_lock(self):
            mark_info = self.mark_info_cache
            if mark_info is not MISSING:
                return cast(PdfDict | None, mark_info)
            resolved_mark_info = self.resolver.resolve(lookup_dict_key(self.catalog(), "MarkInfo"))
            if resolved_mark_info is None:
                self.mark_info_cache = None
                return None
            if not isinstance(resolved_mark_info, dict):
                raise ValueError("invalid MarkInfo dictionary")
            mark_info = cast(PdfDict, resolved_mark_info)
            self.mark_info_cache = mark_info
            return mark_info

    def invalidate_document_extraction_cache(self) -> None:
        """Clear every per-page extraction cache; the single home of page-cache clearing."""
        with document_cache_lock(self):
            if self.page_extraction_caches is not None:
                for cache in self.page_extraction_caches.values():
                    cache.clear()
            self.page_extraction_caches = None
            pages_cache = self.pages_cache
            if pages_cache is not None:
                for page in tuple(pages_cache):
                    page_cache = page.extraction_cache
                    if page_cache is not None:
                        page_cache.clear()

    def _initialize_document_caches(self) -> None:
        for cache_name in DOCUMENT_CACHE_FIELDS:
            setattr(self, cache_name, None)
        for cache_name in OPTIONAL_DOCUMENT_CACHE_FIELDS:
            setattr(self, cache_name, MISSING)
        self.decoder_cache = {}
        self.image_cache = ImageCache()
        self.inherited_values_cache = {}

    def _clear_document_caches(self) -> None:
        for cache_name in DOCUMENT_CACHE_FIELDS:
            setattr(self, cache_name, None)
        for cache_name in OPTIONAL_DOCUMENT_CACHE_FIELDS:
            setattr(self, cache_name, MISSING)
        self.decoder_cache.clear()
        self.image_cache.clear()
        self.inherited_values_cache.clear()

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
        encrypt_ref = lookup_dict_key(self.trailer_dict, "Encrypt")
        if encrypt_ref is None:
            return

        encrypt_dict = self.resolver.resolve_dict(encrypt_ref)
        if not isinstance(encrypt_dict, dict):
            raise PdfUnsupportedError("Invalid Encrypt dictionary")

        filter_name = self.resolver.resolve_name(lookup_dict_key(encrypt_dict, "Filter"))
        if filter_name is None:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        if filter_name in {"Adobe.PubSec", "PubSec"}:
            raise PdfUnsupportedError("Public-key encryption is not supported")
        if filter_name != "Standard":
            raise PdfUnsupportedError(f"Unsupported encryption filter: {filter_name}")

        raw_v = lookup_dict_key(encrypt_dict, "V")
        if raw_v is None:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        v = self.resolver.resolve_int(raw_v)
        if type(v) is not int:
            raise PdfUnsupportedError("Invalid encryption dictionary")
        handler_cls = SECURITY_HANDLER_REGISTRY.get(v)
        if handler_cls is None:
            raise PdfUnsupportedError(f"Unsupported standard encryption algorithm V={v}")

        docid = lookup_dict_key(self.trailer_dict, "ID")
        if docid is None:
            docid = [b""]
        if isinstance(docid, PdfReference):
            docid = self.resolver.resolve(docid)
        if not isinstance(docid, (list, tuple)) or len(docid) == 0:
            raise PdfUnsupportedError("Invalid trailer ID array")
        docid_list: Sequence[object] = docid

        try:
            handler = handler_cls(docid_list, encrypt_dict, password)
        except PDFPasswordIncorrect as exc:
            raise PdfUnsupportedError("Incorrect password") from exc
        except PDFEncryptionError as exc:
            raise PdfUnsupportedError("Invalid encryption dictionary") from exc
        self.decipher = handler.decrypt

    # Page tree and page labels

    def discover_page_dicts(self) -> Iterator[PdfDict]:
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
            repaired_page = self.repair_recovered_page_inherited_values(
                page_dict, inherited_sources
            )
            signature = self.recovered_page_signature(repaired_page)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            yield repaired_page

    def page_candidate_score(self, obj: PdfDict) -> int:
        node_type = resolve_page_tree_node_type(self.resolver, obj)
        if node_type == "Pages" or node_type not in (None, "Page"):
            return -100

        # A recovered leaf without an explicit /Type needs page content or an
        # annotation to distinguish it from outline destinations and other
        # dictionaries that happen to carry /Parent, /MediaBox, or /Resources.
        explicit_type = normalize_pdf_name(lookup_dict_key(obj, "Type"))
        if (
            explicit_type != "Page"
            and lookup_dict_key(obj, "Contents") is None
            and lookup_dict_key(obj, "Annots") is None
        ):
            return -100

        score = 20 if node_type == "Page" else 0
        if lookup_dict_key(obj, "Kids") is not None:
            score -= 30
        if lookup_dict_key(obj, "Contents") is not None:
            score += 12
        if lookup_dict_key(obj, "MediaBox") is not None:
            score += 8
        if lookup_dict_key(obj, "Resources") is not None:
            score += 4
        if lookup_dict_key(obj, "Parent") is not None:
            score += 2
        if lookup_dict_key(obj, "Annots") is not None:
            score += 1
        return score if score >= 16 else -100

    def pages_candidate_score(self, obj: PdfDict) -> int:
        if resolve_page_tree_node_type(self.resolver, obj) != "Pages":
            return -100
        score = 20
        try:
            kids = self.resolver.resolve(lookup_dict_key(obj, "Kids"))
        except Exception:
            kids = None
        if isinstance(kids, list):
            score += min(len(kids), 20)
        try:
            count = self.resolver.resolve(lookup_dict_key(obj, "Count"))
        except Exception:
            count = None
        if type(count) is int and count >= 0:
            score += min(count, 20)
        if lookup_dict_key(obj, "Resources") is not None:
            score += 5
        if lookup_dict_key(obj, "MediaBox") is not None:
            score += 5
        return score

    def repair_recovered_page_inherited_values(
        self, page_dict: PdfDict, pages_nodes: list[PdfDict]
    ) -> PdfDict:
        missing = [key for key in PAGE_INHERITED_KEYS if lookup_dict_key(page_dict, key) is None]
        if not missing:
            return page_dict

        sources: list[PdfDict] = []
        parent = lookup_dict_key(page_dict, "Parent")
        if parent is not None:
            try:
                parent_obj = self.resolver.resolve(parent)
            except Exception:
                parent_obj = None
            if isinstance(parent_obj, dict):
                sources.append(cast(PdfDict, parent_obj))
        sources.extend(pages_nodes)
        if not sources:
            return page_dict

        repaired: PdfDict | None = None
        for source in sources:
            source_values = self.collect_inherited_values_from_node(source, missing)
            if not source_values:
                continue
            if repaired is None:
                repaired = dict(page_dict)
            for key, value in source_values.items():
                if lookup_dict_key(repaired, key) is None:
                    repaired[key] = cast(PdfObject, value)
            missing = [key for key in missing if lookup_dict_key(repaired, key) is None]
            if not missing:
                break
        return repaired if repaired is not None else page_dict

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
        contents = lookup_dict_key(page_dict, "Contents")
        normalized_contents = self.normalized_reference_signature(contents)
        if normalized_contents is not None:
            return ("Contents", normalized_contents)
        return (
            "Shape",
            self.normalized_reference_signature(lookup_dict_key(page_dict, "MediaBox")),
            self.normalized_reference_signature(lookup_dict_key(page_dict, "Resources")),
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
        with document_cache_lock(self):
            if self.page_dicts_cache is not None:
                yield from self.page_dicts_cache
                return

            page_dicts: list[PdfDict] = []
            for page_dict in self.iter_page_dicts_stream():
                page_dicts.append(page_dict)
                yield page_dict
            self.page_dicts_cache = page_dicts

    def internal_recovered_page_dicts(self) -> list[PdfDict]:
        discovered = list(self.discover_page_dicts())
        if discovered:
            self.page_tree_was_recovered = True
            # Publish the recovered page set before invalidating dependent
            # extraction state. A LazyPageList consults page_count while it is
            # cleared; without this cache, that re-enters page-tree recovery.
            self.page_dicts_cache = discovered
            self.invalidate_document_extraction_cache()
        return discovered

    def iter_page_dicts_stream(self) -> Iterator[PdfDict]:
        def inherited_from_pages_node(
            node: PdfDict, inherited: InheritedValueMap | None
        ) -> InheritedValueMap:
            values = dict(inherited or {})
            for key in PAGE_INHERITED_KEYS:
                value = lookup_dict_key(node, key)
                if value is not None:
                    values[key] = cast(CachedPdfObject, value)
            return values

        def apply_inherited_to_page(
            page_dict: PdfDict, inherited: InheritedValueMap | None
        ) -> PdfDict:
            if not inherited:
                return page_dict
            repaired: PdfDict | None = None
            for key, value in inherited.items():
                if lookup_dict_key(page_dict, key) is not None:
                    continue
                if repaired is None:
                    repaired = dict(page_dict)
                repaired[key] = cast(PdfObject, value)
            return repaired if repaired is not None else page_dict

        def traverse(
            node: object,
            depth: int = 0,
            inherited: InheritedValueMap | None = None,
        ) -> Iterator[PdfDict]:
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
                kids = self.resolver.resolve(lookup_dict_key(node, "Kids"))
                if kids is None:
                    raise ValueError("invalid page tree Kids array")
                if not isinstance(kids, list):
                    raise ValueError("invalid page tree Kids array")
                node_inherited = inherited_from_pages_node(node, inherited)
                for kid in kids:
                    yield from traverse(kid, depth + 1, node_inherited)
            elif node_type == "Page":
                yield apply_inherited_to_page(node, inherited)
            else:
                raise ValueError("invalid page tree node")

        try:
            catalog = self.catalog()
            pages_ref = lookup_dict_key(catalog, "Pages")
            if pages_ref is None:
                raise ValueError("missing page tree root")
            pages_node = self.resolver.resolve(pages_ref)
            if not isinstance(pages_node, dict):
                raise ValueError("invalid page tree root")
            page_dicts = list(traverse(pages_node))
            if page_dicts:
                yield from page_dicts
                return
            discovered = self.internal_recovered_page_dicts()
            if discovered:
                yield from discovered
                return
        except (PdfParseError, ValueError):
            discovered = self.internal_recovered_page_dicts()
            if discovered:
                yield from discovered
                return
            return

    def page_count(self) -> int:
        if self.page_dicts_cache is not None:
            return len(self.page_dicts_cache)
        if self.page_tree_was_recovered:
            return len(self.build_page_dicts())
        try:
            catalog = self.catalog()
            pages_ref = lookup_dict_key(catalog, "Pages")
            if pages_ref is None:
                raise ValueError("missing page tree root")
            pages_node = self.resolver.resolve(pages_ref)
            if not isinstance(pages_node, dict):
                raise ValueError("invalid page tree root")
            count = self.resolver.resolve(lookup_dict_key(pages_node, "Count"))
            if type(count) is int and count >= 0:
                return count
        except (PdfParseError, ValueError):
            return len(self.build_page_dicts())
        return len(self.build_page_dicts())

    def build_page_dicts(self) -> list[PdfDict]:
        return list(self.iter_page_dicts_stream())

    @property
    def pages(self) -> LazyPageList[internal_PageT]:
        return get_or_compute(self, "pages_cache", lambda: LazyPageList(self))

    @property
    def page_labels(self) -> list[str] | None:
        with document_cache_lock(self):
            labels = self.page_labels_cache
            if labels is MISSING:
                labels = self.build_page_labels()
                self.page_labels_cache = labels
            return cast(list[str] | None, labels)

    def page_label(self, page_index: int) -> str | None:
        labels = self.page_labels
        if labels is None or page_index < 0 or page_index >= len(labels):
            return None
        return labels[page_index]

    def build_page_labels(self) -> list[str] | None:
        try:
            labels_root = self.resolve(lookup_dict_key(self.catalog(), "PageLabels"))
        except ValueError:
            if document_recovery_enabled(self):
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
                recover=document_recovery_enabled(self),
            )
            if isinstance(spec, dict)
        ]
        if not specs:
            return None
        specs.sort(key=lambda item: item[0])
        if specs[0][0] != 0:
            if not document_recovery_enabled(self):
                raise ValueError("PageLabels is missing page index 0")
            specs.insert(0, (0, {}))

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

    def page_index_for(self, page_obj: object) -> int | None:
        from core_pdf.impl.engine.spec.s_07_document.page import PdfPage

        if isinstance(page_obj, PdfPage):
            return page_obj.page_number - 1
        if not isinstance(page_obj, dict):
            return None
        with document_cache_lock(self):
            if self.page_index_cache is None:
                if self.page_dicts_cache is None:
                    self.page_dicts_cache = self.build_page_dicts()
                self.page_index_cache = {
                    id(page_dict): index for index, page_dict in enumerate(self.page_dicts_cache)
                }
            page_index = self.page_index_cache.get(id(page_obj))
            if page_index is not None:
                return page_index
            page_struct_parents = lookup_dict_key(page_obj, "StructParents")
            if page_struct_parents is not None and self.page_dicts_cache is not None:
                for index, cached_page in enumerate(self.page_dicts_cache):
                    if lookup_dict_key(cached_page, "StructParents") == page_struct_parents:
                        self.page_index_cache[id(page_obj)] = index
                        return index
            if self.page_dicts_cache is None:
                return None
            for index, cached_page in enumerate(self.page_dicts_cache):
                if cached_page == page_obj:
                    self.page_index_cache[id(page_obj)] = index
                    return index
            signature = self.recovered_page_signature(cast(PdfDict, page_obj))
            for index, cached_page in enumerate(self.page_dicts_cache):
                if self.recovered_page_signature(cached_page) == signature:
                    self.page_index_cache[id(page_obj)] = index
                    return index
            return None

    def selected_page_indexes(self, pages: PageSelection | None = None) -> list[int]:
        return resolve_page_selection(pages, len(self.pages))

    def iter_selected_pages(
        self, pages: PageSelection | None = None
    ) -> Iterator[tuple[int, internal_PageT]]:
        for page_index in self.selected_page_indexes(pages):
            yield page_index, self.pages[page_index]

    # Navigation

    def iter_outlines(self) -> list[RawOutlineItem]:
        outlines = self.resolver.resolve(lookup_dict_key(self.catalog(), "Outlines"))
        if outlines is None:
            return []
        if not isinstance(outlines, dict):
            raise ValueError("invalid Outlines dictionary")
        first = self.resolver.resolve(lookup_dict_key(outlines, "First"))
        if first is None:
            return []
        return self.walk_outlines(first, 0)

    def walk_outlines(self, item: object, level: int) -> list[RawOutlineItem]:
        recover_outlines = document_recovery_enabled(self)
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
            title = self.resolver.resolve_str(lookup_dict_key(current, "Title"))
            dest = lookup_dict_key(current, "Dest")
            if dest is None:
                # /A is very often an indirect reference, so it has to be
                # resolved before it can be recognised as an action dictionary.
                action = self.resolver.resolve(lookup_dict_key(current, "A"))
                if (
                    isinstance(action, dict)
                    and self.resolver.resolve_name(lookup_dict_key(action, "S")) == "GoTo"
                ):
                    dest = lookup_dict_key(action, "D")
            try:
                result.append(
                    RawOutlineItem(
                        title=title or "",
                        level=level,
                        dest=cast(PdfObject | str | None, dest),
                        page_index=self.resolve_destination(dest),
                        count=self.extract_outline_count(cast(PdfDict, current)),
                    )
                )
            except ValueError:
                if not recover_outlines:
                    raise
            first = lookup_dict_key(current, "First")
            if first is not None:
                first = self.resolver.resolve(first)
                if not isinstance(first, dict):
                    if recover_outlines:
                        current = lookup_dict_key(current, "Next")
                        continue
                    raise ValueError("invalid outline child")
                result.extend(self.walk_outlines(first, level + 1))
            current = lookup_dict_key(current, "Next")
        return result

    @staticmethod
    def validate_outline_count(value: object) -> int:
        if type(value) is not int:
            raise ValueError("invalid outline count")
        return value

    def extract_outline_count(self, current: PdfDict) -> int:
        raw_count = lookup_dict_key(current, "Count")
        if raw_count is None:
            return 0
        current_count = self.resolver.resolve_int(raw_count)
        if current_count is None:
            if document_recovery_enabled(self):
                return 0
            raise ValueError("invalid outline count")
        return self.validate_outline_count(current_count)

    def resolve_destination(self, dest: object, seen: set[str] | None = None) -> int | None:
        if dest is None:
            return None
        normalized = self.normalize_destination_value(dest, seen)
        if (
            normalized.raw is None
            and normalized.page_index is None
            and normalized.type is None
            and not normalized.args
        ):
            raise ValueError("invalid destination")
        return normalized.page_index

    def named_destinations(
        self,
    ) -> dict[str, RawNamedDestination]:
        with document_cache_lock(self):
            if self.named_destinations_cache is None:
                self.populate_named_destinations()
            return dict(self.named_destinations_cache or {})

    def resolve_named_destination(
        self, name: str, seen: set[str] | None = None
    ) -> RawNamedDestination | None:
        if seen is None:
            seen = set()
        if name in seen:
            return None
        seen.add(name)
        with document_cache_lock(self):
            if self.named_destinations_cache is None:
                self.populate_named_destinations()
            return (self.named_destinations_cache or {}).get(name)

    def destination_from_list(self, resolved_list: PdfArray) -> RawNamedDestination:
        if not resolved_list:
            raise ValueError("invalid destination array")
        page_obj = self.resolver.resolve(resolved_list[0])
        if page_obj is None:
            raise ValueError("invalid destination page reference")
        page_index = self.page_index_for(page_obj)
        if page_index is None:
            raise ValueError("invalid destination page reference")
        dest_type = None
        args: PdfArray = []
        if len(resolved_list) >= 2:
            raw_type = resolved_list[1]
            dest_type = self.resolver.resolve_name(raw_type) or self.resolver.resolve_str(raw_type)
            if dest_type is None:
                raise ValueError("invalid destination type")
            args = list(resolved_list[2:]) if len(resolved_list) > 2 else []
        return RawNamedDestination(
            page_index=page_index, type=dest_type, args=args, raw=resolved_list
        )

    def normalize_destination_value(
        self,
        val: object,
        seen: set[str] | None = None,
        targets: dict[str, object] | None = None,
        normalized: dict[str, RawNamedDestination] | None = None,
        resolving: set[str] | None = None,
    ) -> RawNamedDestination:
        if seen is None:
            seen = set()
        resolved = self.resolver.resolve(val)
        if isinstance(resolved, dict):
            dest_value = lookup_dict_key(resolved, "D")
            if dest_value is not None:
                return self.normalize_destination_value(
                    dest_value,
                    seen,
                    targets=targets,
                    normalized=normalized,
                    resolving=resolving,
                )
        resolved_list = val if isinstance(val, list) else resolved
        if isinstance(resolved_list, tuple):
            resolved_list = list(resolved_list)
        if isinstance(resolved_list, list) and resolved_list:
            return self.destination_from_list(cast(PdfArray, resolved_list))
        if isinstance(resolved_list, list):
            raise ValueError("invalid destination array")

        name = self.resolver.resolve_name_like_value(resolved)
        if name is not None:
            if targets is not None and normalized is not None and resolving is not None:
                cached = normalized.get(name)
                if cached is not None:
                    return cached
                if name in resolving:
                    return RawNamedDestination(page_index=None, type=None, args=[], raw=name)
                resolving.add(name)
                try:
                    target = targets.get(name)
                    result = (
                        RawNamedDestination(page_index=None, type=None, args=[], raw=name)
                        if target is None
                        else self.normalize_destination_value(
                            target,
                            seen,
                            targets=targets,
                            normalized=normalized,
                            resolving=resolving,
                        )
                    )
                    normalized[name] = result
                    return result
                finally:
                    resolving.discard(name)

            if name in seen:
                return RawNamedDestination(page_index=None, type=None, args=[], raw=name)
            nested = self.resolve_named_destination(name, seen)
            if nested is not None:
                return nested
        raise ValueError("invalid destination")

    def populate_named_destinations(self) -> None:
        with document_cache_lock(self):
            if self.named_destinations_cache is not None:
                return
            targets: dict[str, object] = {}
            dests = self.resolver.resolve(lookup_dict_key(self.catalog(), "Dests"))
            if isinstance(dests, dict):
                for name, val in dests.items():
                    resolved_name = self.resolver.resolve_name(name)
                    if resolved_name is None:
                        raise ValueError("invalid named destination key")
                    targets[resolved_name] = self.resolver.resolve(val)
            names = self.resolver.resolve(lookup_dict_key(self.catalog(), "Names"))
            if isinstance(names, dict):
                dests_tree = self.resolver.resolve(lookup_dict_key(names, "Dests"))
                if isinstance(dests_tree, dict):
                    for name, value in iter_name_tree_items(
                        dests_tree,
                        self.resolver.resolve,
                        self.resolver.resolve_str,
                        recover=document_recovery_enabled(self),
                    ):
                        targets[name] = value

            normalized: dict[str, RawNamedDestination] = {}
            resolving: set[str] = set()

            for name in targets:
                try:
                    self.normalize_destination_value(
                        name,
                        targets=targets,
                        normalized=normalized,
                        resolving=resolving,
                    )
                except (PdfParseError, ValueError):
                    # The entries of a name tree are independent, so one that
                    # points at a page the document no longer contains says
                    # nothing about the rest. Record it as unresolved and carry
                    # on: 93 dangling destinations in the PDF 1.7 reference
                    # otherwise took all 70,306 sound ones, and the entire
                    # outline tree, down with them.
                    normalized[name] = RawNamedDestination(
                        page_index=None, type=None, args=[], raw=name
                    )

            object.__setattr__(self, "named_destinations_cache", normalized)

    # Forms

    @property
    def acroform(self) -> PdfDict | None:
        with document_cache_lock(self):
            cached = self.acroform_cache
            if cached is not MISSING:
                return cast(PdfDict | None, cached)
            acroform_val = self.resolver.resolve(lookup_dict_key(self.catalog(), "AcroForm"))
            if acroform_val is None:
                self.acroform_cache = None
                return None
            if isinstance(acroform_val, dict):
                result = cast(PdfDict, acroform_val)
                self.acroform_cache = result
                return result
            if document_recovery_enabled(self):
                self.acroform_cache = None
                return None
            raise ValueError("invalid AcroForm dictionary")

    def collect_field_records(
        self,
        node: object,
        inherited_name: str = "",
        inherited_type: str = "",
        inherited_value: object = None,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[RawFormField]:
        recover = document_recovery_enabled(self)
        if seen is None:
            seen = set()
        records: list[RawFormField] = []
        stack: list[FieldTraversalEntry] = [
            ("node", node, inherited_name, inherited_type, inherited_value, depth)
        ]

        while stack:
            entry = stack.pop()
            if entry[0] == "record":
                records.append(entry[1])
                continue

            (
                ignored,
                current_node,
                parent_name,
                parent_type,
                parent_value,
                current_depth,
            ) = entry
            if current_depth > 50:
                if recover:
                    continue
                raise ValueError("invalid AcroForm depth")
            current_node = self.resolver.resolve(current_node)
            if not isinstance(current_node, dict):
                if recover:
                    continue
                raise ValueError("invalid AcroForm field entry")
            marker = id(current_node)
            if marker in seen:
                if recover:
                    continue
                raise ValueError("invalid AcroForm field entry")
            seen.add(marker)

            title = self.resolver.resolve_str(lookup_dict_key(current_node, "T"))
            current_name = (
                f"{parent_name}.{title}" if parent_name and title else title or parent_name
            )

            type_value = lookup_dict_key(current_node, "FT")
            field_type = (
                self.resolver.resolve_name(type_value)
                or self.resolver.resolve_name_like_value(type_value)
                or self.resolver.resolve_str(type_value)
                or parent_type
            )

            value = lookup_dict_key(current_node, "V")
            if value is None:
                value = parent_value
            value_text = field_value_text(self, value)

            kids = lookup_dict_key(current_node, "Kids")
            if kids is None:
                kids = []
            elif not isinstance(kids, list):
                if recover:
                    kids = []
                else:
                    raise ValueError("invalid AcroForm Kids array")
            kids = cast(list[PdfObject], kids)
            subtype_value = lookup_dict_key(current_node, "Subtype")
            subtype = (
                self.resolver.resolve_name(subtype_value)
                or self.resolver.resolve_str(subtype_value)
                or ""
            )
            current_node = cast(PdfDict, current_node)
            records.append(
                RawFormField(
                    current_name,
                    field_type,
                    cast(PdfObject, value),
                    value_text,
                    field_widget_rect(self, current_node if subtype == "Widget" else None),
                    current_node,
                    kids=kids,
                    widget=current_node if subtype == "Widget" else None,
                )
            )

            for kid in reversed(kids):
                resolved_kid = self.resolver.resolve(kid)
                if not isinstance(resolved_kid, dict):
                    if recover:
                        continue
                    raise ValueError("invalid AcroForm kid entry")
                resolved_kid = cast(PdfDict, resolved_kid)
                subtype_value = lookup_dict_key(resolved_kid, "Subtype")
                subtype = (
                    self.resolver.resolve_name(subtype_value)
                    or self.resolver.resolve_str(subtype_value)
                    or ""
                )
                if subtype == "Widget":
                    widget_type_value = lookup_dict_key(resolved_kid, "FT")
                    widget_type = (
                        self.resolver.resolve_name(widget_type_value)
                        or self.resolver.resolve_name_like_value(widget_type_value)
                        or self.resolver.resolve_str(widget_type_value)
                        or field_type
                    )
                    widget_title = self.resolver.resolve_str(lookup_dict_key(resolved_kid, "T"))
                    widget_name = (
                        f"{current_name}.{widget_title}"
                        if current_name and widget_title
                        else widget_title or current_name
                    )
                    widget_value = lookup_dict_key(resolved_kid, "V")
                    if widget_value is None:
                        widget_value = value
                    stack.append(
                        (
                            "record",
                            RawFormField(
                                widget_name,
                                widget_type,
                                cast(PdfObject, widget_value),
                                field_value_text(self, widget_value),
                                field_widget_rect(self, resolved_kid),
                                resolved_kid,
                                kids=[],
                                widget=resolved_kid,
                            ),
                        )
                    )
                else:
                    stack.append(
                        (
                            "node",
                            resolved_kid,
                            current_name,
                            field_type,
                            value,
                            current_depth + 1,
                        )
                    )
        return records

    def fields(self) -> list[RawFormField]:
        with document_cache_lock(self):
            if self.fields_cache is not None:
                return self.fields_cache
            af = self.acroform
            records: list[RawFormField] = []
            if af is not None:
                field_list = lookup_dict_key(af, "Fields")
                if field_list is None:
                    field_list = []
                elif not isinstance(field_list, list):
                    if document_recovery_enabled(self):
                        field_list = []
                    else:
                        raise ValueError("invalid AcroForm Fields array")
                for field in field_list:
                    field_obj = self.resolver.resolve(field)
                    records.extend(self.collect_field_records(field_obj))
            # 12.5.6.19 lets a field with a single widget merge both
            # dictionaries into one, so a widget carrying /FT is itself a field
            # and a missing or empty catalog field tree does not mean the
            # document has none -- producers do ship filled forms that way.
            # Fall back to the pages when the tree tells us nothing, which also
            # keeps well-formed documents clear of a whole-page scan.
            if not records or document_recovery_enabled(self):
                records.extend(self.discover_widget_field_records(records))
            self.fields_cache = records
            return records

    def discover_widget_field_records(self, existing: list[RawFormField]) -> list[RawFormField]:
        seen_widgets = {id(record.widget) for record in existing if isinstance(record.widget, dict)}
        records: list[RawFormField] = []
        for page_dict in self.iter_page_dicts():
            raw_annots = self.resolver.resolve(lookup_dict_key(page_dict, "Annots"))
            if raw_annots is None:
                continue
            annots = raw_annots if isinstance(raw_annots, list) else [raw_annots]
            for annot_ref in annots:
                annot = self.resolver.resolve(annot_ref)
                if not isinstance(annot, dict):
                    continue
                if id(annot) in seen_widgets:
                    continue
                subtype = (
                    self.resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
                    or self.resolver.resolve_str(lookup_dict_key(annot, "Subtype"))
                    or ""
                )
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
                records.extend(self.collect_field_records(root))
        return records

    def internal_widget_field_root(self, annot: PdfDict) -> PdfDict:
        node = annot
        seen = {id(node)}
        for _ in range(50):
            parent = self.resolver.resolve(lookup_dict_key(node, "Parent"))
            if not isinstance(parent, dict) or id(parent) in seen:
                break
            seen.add(id(parent))
            node = cast(PdfDict, parent)
        return node

    # Attachments and optional content

    def embedded_files(self) -> list[RawEmbeddedFile]:
        return list(get_or_compute(self, "embedded_files_cache", self.build_embedded_files))

    def build_embedded_files(self) -> list[RawEmbeddedFile]:
        names = self.resolver.resolve(lookup_dict_key(self.catalog(), "Names"))
        if not isinstance(names, dict):
            return []
        embedded_tree = self.resolver.resolve(lookup_dict_key(names, "EmbeddedFiles"))
        if embedded_tree is None:
            return []
        if not isinstance(embedded_tree, dict):
            raise ValueError("invalid EmbeddedFiles name tree")

        recover = document_recovery_enabled(self)
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
        ef = self.resolver.resolve(lookup_dict_key(filespec, "EF"))
        if not isinstance(ef, dict):
            raise ValueError("invalid embedded file stream")
        ef = cast(PdfDict, ef)
        stream = self.resolver.resolve(lookup_dict_key(ef, "UF") or lookup_dict_key(ef, "F"))
        if not isinstance(stream, PdfStream):
            raise ValueError("invalid embedded file stream")
        filename = (
            self.resolver.resolve_str(lookup_dict_key(filespec, "UF"))
            or self.resolver.resolve_str(lookup_dict_key(filespec, "F"))
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

    def load_oc_layers(self) -> None:
        with document_cache_lock(self):
            if self.oc_layers is None:
                self.internal_load_oc_layers()

    def internal_load_oc_layers(self) -> None:
        self.oc_layers = {}
        recover = document_recovery_enabled(self)
        try:
            catalog = self.catalog()
        except ValueError:
            return
        oc = self.resolver.resolve(lookup_dict_key(catalog, "OCProperties"))
        if oc is None:
            return
        if not isinstance(oc, dict):
            if recover:
                return
            raise ValueError("invalid OCProperties dictionary")
        ocgs = self.resolver.resolve(lookup_dict_key(oc, "OCGs"))
        if ocgs is None:
            return
        if not isinstance(ocgs, list):
            if recover:
                return
            raise ValueError("invalid OCProperties OCGs array")

        on_layers: set[tuple[int, int] | int] = set()
        default_config = self.resolver.resolve(lookup_dict_key(oc, "D"))
        if default_config is not None and not isinstance(default_config, dict):
            if recover:
                default_config = None
            else:
                raise ValueError("invalid OCProperties D dictionary")
        if default_config is not None:
            base_state_value = lookup_dict_key(default_config, "BaseState")
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

            on_refs = lookup_dict_key(default_config, "ON")
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

            off_refs = lookup_dict_key(default_config, "OFF")
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

        for ocg_ref in ocgs:
            ocg_resolved = self.resolver.resolve(ocg_ref)
            if not isinstance(ocg_resolved, dict):
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG entry")
            name = self.resolver.resolve_str(lookup_dict_key(ocg_resolved, "Name"))
            if not name:
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG name")
            key = self.ocg_key(ocg_ref, ocg_resolved)
            self.oc_layers[name] = key in on_layers if key is not None else False

    def oc_hidden_layers(self) -> frozenset[str]:
        if self.oc_layers is None:
            self.load_oc_layers()
        return frozenset(name for name, on in (self.oc_layers or {}).items() if not on)
