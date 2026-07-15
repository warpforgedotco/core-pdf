# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from core_pdf.impl.engine.spec.s_07_document.metadata import resolve_metadata
from core_pdf.impl.engine.spec.s_07_document.metadata_types import MetadataRecord
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_14_structure.tree import StructureTree
from core_pdf.impl.types import PdfDict

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document import PdfDocument


class DocumentCatalogResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_dict(self, value: object) -> PdfDict | None: ...


class DocumentCatalogMixin:
    catalog_cache: PdfDict | None
    metadata_cache: MetadataRecord | None
    structure_cache: StructureTree | None
    structure_root_cache: PdfDict | None
    mark_info_cache: PdfDict | None
    resolver: DocumentCatalogResolver
    trailer_dict: PdfDict
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    def resolve(self, ref: object) -> object:
        return self.resolver.resolve(ref)

    def catalog(self) -> PdfDict:
        catalog = self.catalog_cache
        if catalog is None:
            root_ref = lookup_dict_key(self.trailer_dict, "Root")
            if root_ref is None:
                raise ValueError("missing catalog root")
            root = self.resolve(root_ref)
            if not isinstance(root, dict):
                raise ValueError("invalid catalog root")
            catalog = cast(PdfDict, root)
            self.catalog_cache = catalog
        return catalog

    def get_metadata(self) -> MetadataRecord:
        metadata = self.metadata_cache
        if metadata is None:
            metadata = resolve_metadata(
                self.resolver,
                self.trailer_dict,
                recover=self.xref_was_recovered or self.page_tree_was_recovered,
            )
            self.metadata_cache = metadata
            invalidate = getattr(self, "invalidate_document_extraction_cache", None)
            if callable(invalidate):
                invalidate()
        return metadata

    @property
    def structure(self) -> StructureTree | None:
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
            structure = StructureTree(cast("PdfDocument", self), struct_root)
            self.structure_cache = structure
        return structure

    @property
    def mark_info(self) -> PdfDict | None:
        mark_info = self.mark_info_cache
        if mark_info is None:
            resolved_mark_info = self.resolver.resolve(lookup_dict_key(self.catalog(), "MarkInfo"))
            if resolved_mark_info is None:
                return None
            if not isinstance(resolved_mark_info, dict):
                raise ValueError("invalid MarkInfo dictionary")
            mark_info = cast(PdfDict, resolved_mark_info)
            self.mark_info_cache = mark_info
        return mark_info


__all__ = ("DocumentCatalogMixin",)
