# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator

from core_pdf.impl.document_contracts import DocumentState
from core_pdf.impl.document_page import PAGE_INHERITED_KEYS
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_document.page import PageNode, iter_page_nodes
from core_pdf_spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import InheritedValueMap, PdfDict, PdfValueResolver

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


class PageTreeRecovery(DocumentState):
    __slots__ = ()

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
