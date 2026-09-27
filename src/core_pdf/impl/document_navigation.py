# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from core_pdf.impl.caches import DocumentCaches
from core_pdf.impl.document_contracts import DocumentState
from core_pdf.impl.document_page import PdfPage
from core_pdf.impl.document_page_links import goto_action_destination
from core_pdf.impl.document_records import RawNamedDestination, RawOutlineItem
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.recovery_trees import iter_name_tree_items
from core_pdf_spec.s_07_document.page import PageNode
from core_pdf_spec.s_07_syntax.types import PdfArray, PdfDict

if TYPE_CHECKING:
    from core_pdf.impl.document_document import PdfDocument


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


class DocumentNavigation[PageT: PdfPage](DocumentState[PageT]):
    __slots__ = ()

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
