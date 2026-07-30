# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

from core_pdf.impl.engine.spec.s_07_document.document_labels import (
    MAX_PAGE_TREE_DEPTH,
    format_page_label,
    resolve_page_tree_node_type,
)
from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
)
from core_pdf.impl.engine.spec.s_07_document.name_trees import iter_number_tree_items
from core_pdf.impl.engine.spec.s_07_objects.object_cache import (
    CachedPdfObject,
    InheritedValueMap,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    collect_inherited_values,
    lookup_dict_key,
)
from core_pdf.impl.engine.spec.s_07_objects.resolver_values import PdfValueResolver
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfReference, PdfStream
from core_pdf.impl.types import PageSelection, PdfDict, PdfObject

PAGE_INHERITED_KEYS = (
    "MediaBox",
    "CropBox",
    "BleedBox",
    "TrimBox",
    "ArtBox",
    "Rotate",
    "Resources",
    "Annots",
)


class PageListItem(Protocol):
    page_dict: PdfDict


internal_PageT = TypeVar("internal_PageT", bound=PageListItem)
PageFactory = Callable[[object, PdfDict, int], internal_PageT]


class PageListDocument(Protocol):
    page_dicts_cache: list[PdfDict] | None

    def iter_page_dicts_stream(self) -> Iterator[PdfDict]: ...

    def page_count(self) -> int: ...


class LazyPageList(list[internal_PageT], Generic[internal_PageT]):
    """A sequence that only resolves page dictionaries as they are requested."""

    __slots__ = ("document", "page_dict_iter", "complete")

    document: PageListDocument
    page_dict_iter: Iterator[PdfDict] | None
    complete: bool

    def __init__(self, document: PageListDocument) -> None:
        super().__init__()
        self.document = document
        self.page_dict_iter = None
        self.complete = False

    def next_page_dict(self) -> PdfDict:
        document = self.document
        cached_dicts = document.page_dicts_cache
        current_len = list.__len__(self)
        if cached_dicts is not None:
            if current_len >= len(cached_dicts):
                self.complete = True
                raise IndexError("page index out of range")
            return cached_dicts[current_len]

        if self.page_dict_iter is None:
            self.page_dict_iter = document.iter_page_dicts_stream()
        try:
            return next(self.page_dict_iter)
        except StopIteration:
            self.complete = True
            document.page_dicts_cache = [page.page_dict for page in list.__iter__(self)]
            raise IndexError("page index out of range") from None

    def ensure(self, index: int) -> None:
        while list.__len__(self) <= index:
            page_dict = self.next_page_dict()
            page_class = getattr(self.document, "page_class", None)
            if page_class is None:
                from core_pdf.impl.engine.spec.s_07_document.page import PdfPage

                page_class = PdfPage
            page_class = cast(PageFactory[internal_PageT], page_class)
            list.append(self, page_class(self.document, page_dict, list.__len__(self) + 1))

    def __len__(self) -> int:
        return self.document.page_count()

    def __iter__(self) -> Iterator[internal_PageT]:
        index = 0
        while True:
            try:
                yield self[index]
            except IndexError:
                return
            index += 1

    @overload
    def __getitem__(self, item: SupportsIndex) -> internal_PageT: ...

    @overload
    def __getitem__(self, item: slice[SupportsIndex | None]) -> list[internal_PageT]: ...

    def __getitem__(
        self, item: SupportsIndex | slice[SupportsIndex | None]
    ) -> internal_PageT | list[internal_PageT]:
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return [self[page_index] for page_index in range(start, stop, step)]
        index = item.__index__()
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("page index out of range")
        self.ensure(index)
        return list.__getitem__(self, index)


DocumentPagesResolver = PdfValueResolver


class DocumentPagesMixin(Generic[internal_PageT]):
    internal_cache_lock: Any
    xref: dict[int, Any]
    xref_was_recovered: bool
    page_dicts_cache: list[PdfDict] | None
    pages_cache: LazyPageList[internal_PageT] | None
    page_index_cache: dict[int, int] | None
    page_labels_cache: list[str] | None
    page_tree_was_recovered: bool
    resolver: DocumentPagesResolver

    if TYPE_CHECKING:

        def catalog(self) -> PdfDict: ...

        def resolve(self, ref: object) -> object: ...

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
            invalidate = getattr(self, "invalidate_document_extraction_cache", None)
            if callable(invalidate):
                invalidate()
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
        with document_cache_lock(self):
            if self.pages_cache is None:
                self.pages_cache = LazyPageList(self)
            pages = self.pages_cache
            return pages

    @property
    def page_labels(self) -> list[str] | None:
        with document_cache_lock(self):
            if self.page_labels_cache is None:
                self.page_labels_cache = self.build_page_labels()
            return self.page_labels_cache

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
        page_count = len(self.pages)
        match pages:
            case None:
                selected = list(range(page_count))
            case int() if type(pages) is int:
                selected = [pages - 1]
            case range() as page_range:
                selected = [page_number - 1 for page_number in page_range]
            case str() as page_spec:
                selected = []
                for part in page_spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        parts = part.split("-", 1)
                        if not parts[0].strip() or not parts[1].strip():
                            raise ValueError(f"invalid page selection: {pages!r}")
                        try:
                            start = int(parts[0])
                            end = int(parts[1])
                        except ValueError as exc:
                            raise ValueError(f"invalid page selection: {pages!r}") from exc
                        step = 1 if end >= start else -1
                        selected.extend(range(start - 1, end - 1 + step, step))
                    else:
                        try:
                            selected.append(int(part) - 1)
                        except ValueError as exc:
                            raise ValueError(f"invalid page selection: {pages!r}") from exc
            case Sequence() as page_sequence:
                try:
                    selected = [int(cast(Any, page_number)) - 1 for page_number in page_sequence]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid page selection: {pages!r}") from exc
            case _:
                raise TypeError(f"invalid page selection: {pages!r}")

        if not selected:
            raise ValueError(f"invalid page selection: {pages!r}")

        normalized: list[int] = []
        seen: set[int] = set()
        for page_index in selected:
            if page_index < 0 or page_index >= page_count:
                raise IndexError(f"page selection out of range: {page_index + 1}")
            if page_index not in seen:
                normalized.append(page_index)
                seen.add(page_index)
        return normalized

    def iter_selected_pages(
        self, pages: PageSelection | None = None
    ) -> Iterator[tuple[int, internal_PageT]]:
        for page_index in self.selected_page_indexes(pages):
            yield page_index, self.pages[page_index]


__all__ = (
    "DocumentPagesMixin",
    "LazyPageList",
    "PAGE_INHERITED_KEYS",
    "PageListItem",
)
