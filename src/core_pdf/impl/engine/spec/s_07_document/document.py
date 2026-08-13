# SPDX-License-Identifier: AGPL-3.0-only
"""Spec-level document: catalog, trailer, and security setup."""

from __future__ import annotations

import contextlib
import mmap
import threading
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Generic, Self, TypeVar, cast

from core_pdf.impl.engine.cache import ExtractionCache
from core_pdf.impl.engine.image_cache import ImageCache
from core_pdf.impl.engine.spec.s_07_document.document_core import DocumentCoreMixin
from core_pdf.impl.engine.spec.s_07_document.document_features import DocumentFeaturesMixin
from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
    get_or_compute,
)
from core_pdf.impl.engine.spec.s_07_document.document_pages import (
    DocumentPagesMixin,
    LazyPageList,
    PageListItem,
)
from core_pdf.impl.engine.spec.s_07_document.metadata import MetadataRecord, resolve_metadata
from core_pdf.impl.engine.spec.s_07_objects.object_cache import InheritedValuesCache
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry
from core_pdf.impl.engine.spec.s_14_structure.tree import StructureTree
from core_pdf.impl.exceptions import PdfDocumentClosedError
from core_pdf.impl.models import RawEmbeddedFile, RawFormField, RawNamedDestination
from core_pdf.impl.types import Decipher, PdfDict, PdfSource

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder


internal_PageT = TypeVar("internal_PageT", bound=PageListItem)

DOCUMENT_CACHE_FIELDS = (
    "catalog_cache",
    "metadata_cache",
    "structure_cache",
    "structure_root_cache",
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


class PdfDocument(
    DocumentCoreMixin,
    DocumentPagesMixin[internal_PageT],
    DocumentFeaturesMixin,
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
        "structure_root_cache",
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
    structure_cache: StructureTree | None
    structure_root_cache: PdfDict | None
    mark_info_cache: PdfDict | None
    page_dicts_cache: list[PdfDict] | None
    pages_cache: LazyPageList[internal_PageT] | None
    page_index_cache: dict[int, int] | None
    named_destinations_cache: dict[str, RawNamedDestination] | None
    embedded_files_cache: list[RawEmbeddedFile] | None
    oc_layers: dict[str, bool] | None
    acroform_cache: PdfDict | None
    fields_cache: list[RawFormField] | None
    decoder_cache: dict[tuple[int, int] | int, FontDecoder]
    image_cache: ImageCache
    inherited_values_cache: InheritedValuesCache
    page_labels_cache: list[str] | None
    page_extraction_caches: dict[int, ExtractionCache] | None
    internal_cache_lock: threading.RLock
    xref_was_recovered: bool
    page_tree_was_recovered: bool
    internal_closed: bool

    def __init__(self, source: PdfSource, password: str = "") -> None:
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
    def open(cls, source: PdfSource, password: str = "") -> Self:
        return cls(source, password=password)

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
            if structure is None:
                struct_root = self.structure_root_cache
                if struct_root is None:
                    resolved_root = self.resolver.resolve(
                        lookup_dict_key(self.catalog(), "StructTreeRoot")
                    )
                    if resolved_root is None:
                        return None
                    if not isinstance(resolved_root, dict):
                        raise ValueError("invalid StructTreeRoot dictionary")
                    struct_root = cast(PdfDict, resolved_root)
                    self.structure_root_cache = struct_root
                structure = StructureTree(self, struct_root)
                self.structure_cache = structure
            return structure

    @property
    def mark_info(self) -> PdfDict | None:
        with document_cache_lock(self):
            mark_info = self.mark_info_cache
            if mark_info is None:
                resolved_mark_info = self.resolver.resolve(
                    lookup_dict_key(self.catalog(), "MarkInfo")
                )
                if resolved_mark_info is None:
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
        self.decoder_cache = {}
        self.image_cache = ImageCache()
        self.inherited_values_cache = {}

    def _clear_document_caches(self) -> None:
        for cache_name in DOCUMENT_CACHE_FIELDS:
            setattr(self, cache_name, None)
        self.decoder_cache.clear()
        self.image_cache.clear()
        self.inherited_values_cache.clear()
