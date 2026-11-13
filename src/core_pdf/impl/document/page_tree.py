# SPDX-License-Identifier: AGPL-3.0-only

"""The page tree: pages, page labels and lookups, and the pages recovered from a damaged tree.

The PdfDocument methods here are functions taking the document as self;
PdfDocument lists them as its methods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.base.caches import DocumentCaches
from core_pdf.impl.base.exceptions import PdfParseError
from core_pdf.impl.base.raw_records import RawNamedDestination
from core_pdf.impl.base.types import PdfName, PdfReference
from core_pdf.impl.base.values import parse_text_string, recover_pdf_name
from core_pdf.impl.document.pdf_page import PAGE_INHERITED_KEYS, PdfPage
from core_pdf.impl.document.source import first_indexes
from core_pdf.impl.spec.s_07_document.page import PageNode, iter_page_nodes
from core_pdf.impl.spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import InheritedValueMap, PdfDict, PdfValueResolver
lazy from core_pdf.impl.base.pages import PageSelection, resolve_page_selection
lazy from core_pdf.impl.recovery.trees import iter_number_tree_items
lazy from core_pdf.impl.spec.s_07_document import document_labels
lazy from core_pdf.impl.spec.s_07_document.document_labels import PageLabelStyle

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PageT, PdfDocument


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


def infer_page_tree_node_type(node: PdfDict, *, include_page_properties: bool = True) -> str | None:
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
            "signatures", lambda: first_indexes(signature_of(node.dictionary) for node in nodes)
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


def build_page_tree(self: PdfDocument[PageT]) -> tuple[PageT, ...]:
    return self.build_pages(self.iter_recovered_page_nodes())


def build_pages(self: PdfDocument[PageT], nodes: Iterable[PageNode]) -> tuple[PageT, ...]:
    factory = cast("type[PageT]", self.page_class)
    return tuple(
        factory(
            self, page_node.dictionary, page_number, inherited_values=page_node.inherited_values
        )
        for page_number, page_node in enumerate(nodes, 1)
    )


def cached_page_labels(self: PdfDocument[Any]) -> tuple[str, ...] | None:
    return self.caches.get("page_labels", self.build_page_label_tuple)


def build_page_label_tuple(self: PdfDocument[Any]) -> tuple[str, ...] | None:
    labels = self.build_page_labels()
    return None if labels is None else tuple(labels)


def page_label(self: PdfDocument[Any], page_index: int) -> str | None:
    labels = self.cached_page_labels()
    if labels is None or page_index < 0 or page_index >= len(labels):
        return None
    return labels[page_index]


def build_page_labels(self: PdfDocument[Any], *, page_count: int | None = None) -> list[str] | None:
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
            labels_root, self.resolve, on_malformed=self.recovery.malformed
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


def build_page_lookup(self: PdfDocument[PageT]) -> PageLookup[PageT]:
    return PageLookup(self)


def page_index_for(self: PdfDocument[Any], page_obj: object) -> int | None:
    return self.page_lookup.page_index_for(page_obj)


def iter_selected_pages(
    self: PdfDocument[PageT], pages: PageSelection | None = None
) -> Iterator[tuple[int, PageT]]:
    page_objects = self.pages
    for page_index in resolve_page_selection(pages, len(page_objects)):
        yield page_index, page_objects[page_index]


def discover_page_nodes(self: PdfDocument[Any]) -> Iterator[PageNode]:
    candidates: list[tuple[int, int, int, PdfDict]] = []
    pages_nodes: list[tuple[int, int, int, PdfDict]] = []
    seen_objects: set[int] = set()
    for key, entry in sorted(
        self.xref.items(),
        key=lambda item: (item[1].offset if item[1].object_stream is None else 0, item[0] >> 16),
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
        yield PageNode(page_dict, self.recovered_page_values(page_dict, inherited_sources))


def page_candidate_score(self: PdfDocument[Any], obj: PdfDict, node_type: str | None) -> int:
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


def pages_candidate_score(self: PdfDocument[Any], obj: PdfDict, node_type: str | None) -> int:
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


def recovered_inherited_values(self: PdfDocument[Any], page_dict: PdfDict) -> InheritedValueMap:
    for node in self.recovered_page_nodes():
        if node.dictionary is page_dict:
            return dict(node.inherited_values)
    return self.recovered_page_values(page_dict, [])


def recovered_page_values(
    self: PdfDocument[Any], page_dict: PdfDict, pages_nodes: list[PdfDict]
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
    self: PdfDocument[Any], node: PdfDict, keys: list[str]
) -> InheritedValueMap:
    return collect_inherited_values(
        node, tuple(keys), self.resolver.resolve_or_none, stop_at_malformed_parent=True
    )


def recovered_page_signature(self: PdfDocument[Any], page_dict: PdfDict) -> tuple[object, ...]:
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


def normalized_reference_signature(self: PdfDocument[Any], value: object) -> object:
    if isinstance(value, PdfReference):
        return ("R", value.object_number, value.generation_number)
    if isinstance(value, (list, tuple)):
        return tuple(self.normalized_reference_signature(item) for item in value)
    if isinstance(value, dict):
        return ("D", id(value))
    if isinstance(value, PdfStream):
        return ("S", id(value))
    return value


def recovered_page_nodes(self: PdfDocument[Any]) -> list[PageNode]:
    discovered = list(self.discover_page_nodes())
    if discovered:
        self.page_tree_was_recovered = True
    return discovered


def page_tree_root(self: PdfDocument[Any]) -> PdfDict:
    pages_ref = self.catalog().get("Pages")
    if pages_ref is None:
        raise ValueError("missing page tree root")
    pages_node = self.resolver.as_dict(pages_ref)
    if pages_node is None:
        raise ValueError("invalid page tree root")
    return pages_node


def iter_recovered_page_nodes(self: PdfDocument[Any]) -> Iterator[PageNode]:
    try:
        page_nodes = list(
            iter_page_nodes(
                self.page_tree_root(),
                self.resolver.resolve,
                inherited_keys=PAGE_INHERITED_KEYS,
                node_type=lambda node: resolve_page_tree_node_type(self.resolver, node),
                on_invalid_child=lambda _node: True,
                max_depth=MAX_PAGE_TREE_DEPTH,
                lazy_inherited=True,
            )
        )
    except PdfParseError, ValueError:
        page_nodes = []
    yield from page_nodes or self.recovered_page_nodes()
