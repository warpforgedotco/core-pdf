# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, cast

from core_pdf.impl.engine.spec.s_07_document.document_labels import (
    infer_page_tree_node_type,
)
from core_pdf.impl.engine.spec.s_07_document.document_page_labels import (
    DocumentPageLabelsMixin,
)
from core_pdf.impl.engine.spec.s_07_document.document_page_list import LazyPageList
from core_pdf.impl.engine.spec.s_07_document.document_page_recovery import (
    RECOVERABLE_PAGE_INHERITED_KEYS,
    DocumentPageRecoveryMixin,
)
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    InheritedValueMap,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.types import PdfDict, PdfObject

MAX_PAGE_TREE_DEPTH = 100


class DocumentPagesResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_name(self, value: object) -> str | None: ...


class DocumentPagesMixin(DocumentPageRecoveryMixin, DocumentPageLabelsMixin):
    page_dicts_cache: list[PdfDict] | None
    pages_cache: LazyPageList | None
    page_index_cache: dict[int, int] | None
    page_tree_was_recovered: bool
    resolver: DocumentPagesResolver

    if TYPE_CHECKING:

        def catalog(self) -> PdfDict: ...

        def discover_page_dicts(self) -> Iterator[PdfDict]: ...

        def recovered_page_signature(self, page_dict: PdfDict) -> tuple[object, ...]: ...

    def iter_page_dicts(self) -> Iterator[PdfDict]:
        if self.page_dicts_cache is not None:
            yield from self.page_dicts_cache
            return

        page_dicts: list[PdfDict] = []
        for page_dict in self.iter_page_dicts_stream():
            page_dicts.append(page_dict)
            yield page_dict
        self.page_dicts_cache = page_dicts

    def iter_page_dicts_stream(self) -> Iterator[PdfDict]:
        def inherited_from_pages_node(
            node: PdfDict, inherited: InheritedValueMap | None
        ) -> InheritedValueMap:
            values = dict(inherited or {})
            for key in RECOVERABLE_PAGE_INHERITED_KEYS:
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
            node_type = self.resolver.resolve_name(lookup_dict_key(node, "Type"))
            if node_type is None:
                node_type = infer_page_tree_node_type(node)
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
            discovered = list(self.discover_page_dicts())
            if discovered:
                self.page_tree_was_recovered = True
                invalidate = getattr(self, "invalidate_document_extraction_cache", None)
                if callable(invalidate):
                    invalidate()
                yield from discovered
                return
        except (PdfParseError, ValueError):
            discovered = list(self.discover_page_dicts())
            if discovered:
                self.page_tree_was_recovered = True
                invalidate = getattr(self, "invalidate_document_extraction_cache", None)
                if callable(invalidate):
                    invalidate()
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

    def build_page_cache(self) -> tuple[list[PdfDict], dict[int, int]]:
        page_dicts = self.build_page_dicts()
        page_index_cache = {id(page_dict): index for index, page_dict in enumerate(page_dicts)}
        return page_dicts, page_index_cache

    @property
    def pages(self) -> LazyPageList:
        if self.pages_cache is None:
            self.pages_cache = LazyPageList(self)
        pages = self.pages_cache
        return pages

    def page_index_for(self, page_obj: object) -> int | None:
        if isinstance(page_obj, PdfPage):
            return page_obj.page_number - 1
        if not isinstance(page_obj, dict):
            return None
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


__all__ = ("DocumentPagesMixin", "LazyPageList")
