# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import contextlib
import mmap
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.spec.s_07_document.document_catalog import (
    DocumentCatalogMixin,
)
from core_pdf.impl.engine.spec.s_07_document.document_embedded import (
    DocumentEmbeddedMixin,
)
from core_pdf.impl.engine.spec.s_07_document.document_pages import (
    DocumentPagesMixin,
    LazyPageList,
)
from core_pdf.impl.engine.spec.s_07_document.document_security import (
    DocumentSecurityMixin,
)
from core_pdf.impl.engine.spec.s_07_document.document_selection import (
    DocumentSelectionMixin,
)
from core_pdf.impl.engine.spec.s_07_document.document_source import DocumentSourceMixin
from core_pdf.impl.engine.spec.s_07_document.document_xref import (
    DocumentXRefMixin,
)
from core_pdf.impl.engine.spec.s_07_document.forms import FormsMixin
from core_pdf.impl.engine.spec.s_07_document.layers import LayersMixin
from core_pdf.impl.engine.spec.s_07_document.metadata_types import MetadataRecord
from core_pdf.impl.engine.spec.s_07_document.navigation import NavigationMixin
from core_pdf.impl.engine.spec.s_07_objects.object_cache import InheritedValuesCache
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry
from core_pdf.impl.engine.spec.s_14_structure.tree import StructureTree
from core_pdf.impl.models import EmbeddedFileRecord, FieldRecord, NamedDestination
from core_pdf.impl.types import Decipher, PdfDict, PdfSource

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder


class PdfDocument(
    DocumentSourceMixin,
    DocumentXRefMixin,
    DocumentSecurityMixin,
    DocumentPagesMixin,
    DocumentSelectionMixin,
    DocumentEmbeddedMixin,
    DocumentCatalogMixin,
    NavigationMixin,
    FormsMixin,
    LayersMixin,
):
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
        "inherited_values_cache",
        "page_labels_cache",
        "page_extraction_caches",
        "xref_was_recovered",
        "page_tree_was_recovered",
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
    pages_cache: LazyPageList | None
    page_index_cache: dict[int, int] | None
    named_destinations_cache: dict[str, NamedDestination] | None
    embedded_files_cache: list[EmbeddedFileRecord] | None
    oc_layers: dict[str, bool] | None
    acroform_cache: PdfDict | None
    fields_cache: list[FieldRecord] | None
    decoder_cache: dict[tuple[int, int] | int, FontDecoder]
    inherited_values_cache: InheritedValuesCache
    page_labels_cache: list[str] | None
    page_extraction_caches: dict[int, ExtractionCache] | None
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def __init__(self, source: PdfSource, password: str = "") -> None:
        self.source = source
        self.password = password
        self.file_handle = None
        self.decipher = None
        self.xref = {}
        self.trailer_dict = {}
        self.xref_was_recovered = False
        self.page_tree_was_recovered = False

        self.raw_data = self.load_data(source)
        self.scan_xref()

        self.resolver = ObjectResolver(self.raw_data, self.xref, self.trailer_dict)
        self.init_security(password)
        self.resolver.decipher = self.decipher

        self.catalog_cache = None
        self.metadata_cache = None
        self.structure_cache = None
        self.structure_root_cache = None
        self.mark_info_cache = None
        self.page_dicts_cache = None
        self.pages_cache = None
        self.page_index_cache = None
        self.named_destinations_cache = None
        self.embedded_files_cache = None
        self.oc_layers = None
        self.acroform_cache = None
        self.fields_cache = None
        self.decoder_cache = {}
        self.inherited_values_cache = {}
        self.page_labels_cache = None
        self.page_extraction_caches = None

    @classmethod
    def open(cls, source: PdfSource, password: str = "") -> PdfDocument:
        return cls(source, password=password)

    def __enter__(self) -> PdfDocument:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.file_handle is not None:
            with contextlib.suppress(OSError):
                self.file_handle.close()
            self.file_handle = None

    def invalidate_document_extraction_cache(self) -> None:
        self.page_extraction_caches = None
