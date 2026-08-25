# SPDX-License-Identifier: AGPL-3.0-only
"""The logical structure tree and queries over its elements."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    coerce_value,
    normalize_pdf_name,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_objects.trees import iter_number_tree_items
from core_pdf.impl.engine.spec.s_14_structure.content import (
    StructureContentItem,
    StructureContentObject,
)
from core_pdf.impl.primitives import MISSING, PdfReference
from core_pdf.impl.types import PdfArray, PdfDict

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.engine.spec.s_07_document.page import PdfPage


MAX_PARENT_TREE_DEPTH = 100
MAX_STRUCTURE_DEPTH = 200

MatchFunc = Callable[["StructureElement"], bool]
StructureDict: TypeAlias = PdfDict
StructureAttributes: TypeAlias = dict[str, Any]
ParentTree: TypeAlias = dict[int, Any]
ParentTreeParents: TypeAlias = PdfArray


def make_match_func(
    matcher: str | MatchFunc | None = None,
) -> MatchFunc:
    if matcher is None:
        return lambda ignored: True
    if isinstance(matcher, str):
        return lambda x: x.role == matcher
    return matcher


def find_all(
    elements: list[StructureElement],
    matcher: str | MatchFunc | None = None,
) -> Iterator[StructureElement]:
    match_func = make_match_func(matcher)
    stack = list(elements)
    stack.reverse()
    while stack:
        el = stack.pop()
        if match_func(el):
            yield el
        for child in reversed(list(el)):
            if isinstance(child, StructureElement):
                stack.append(child)


def find_first(elements: Iterable[StructureElement]) -> StructureElement | None:
    return next(iter(elements), None)


def literal_name(value: Any) -> str | None:
    if isinstance(value, PdfReference):
        return None
    if value is None:
        return None
    return normalize_pdf_name(value)


def structure_key_name(key: Any) -> str:
    return normalize_pdf_name(key) or str(key)


class StructureElement:
    """Logical structure element dictionary from the structure tree."""

    __slots__ = (
        "actual_text_value",
        "alternate_description_value",
        "attributes_value",
        "class_name_value",
        "document",
        "kids_value",
        "language_value",
        "parent_value",
        "props",
        "role_value",
        "title_value",
        "type_value",
    )

    def __init__(self, document: PdfDocument, props: StructureDict) -> None:
        self.document = document
        self.props = props if isinstance(props, dict) else {}
        self.role_value: str | None = None
        self.type_value: Any = MISSING
        self.kids_value: Any = MISSING
        self.title_value: Any = MISSING
        self.language_value: Any = MISSING
        self.alternate_description_value: Any = MISSING
        self.actual_text_value: Any = MISSING
        self.attributes_value: Any = MISSING
        self.class_name_value: Any = MISSING
        self.parent_value: Any = MISSING

    @property
    def type(self) -> str | None:
        if self.type_value is MISSING:
            self.type_value = self.document.resolver.resolve_name_like_value(
                lookup_dict_key(self.props, "S")
            )
        return self.type_value

    @property
    def role(self) -> str:
        if self.role_value is not None:
            return self.role_value
        tree = self.document.structure
        if tree is None:
            return self.type or ""
        return tree.role_map.get(self.type or "", self.type or "")

    @property
    def page_index(self) -> int | None:
        page_ref = lookup_dict_key(self.props, "Pg")
        if page_ref is None:
            return None
        page_obj = self.document.resolver.resolve(page_ref)
        page_index = self.document.page_index_for(page_obj)
        if page_index is None:
            raise ValueError("invalid structure page reference")
        return page_index

    @property
    def page(self) -> PdfPage | None:
        page_index = self.page_index
        if page_index is None:
            return None
        pages = self.document.pages
        if 0 <= page_index < len(pages):
            return pages[page_index]
        return None

    @property
    def title(self) -> str | None:
        if self.title_value is MISSING:
            self.title_value = self.document.resolver.resolve_str(lookup_dict_key(self.props, "T"))
        return self.title_value

    @property
    def language(self) -> str | None:
        if self.language_value is MISSING:
            self.language_value = self.document.resolver.resolve_str(
                lookup_dict_key(self.props, "Lang")
            )
        return self.language_value

    @property
    def alternate_description(self) -> str | None:
        if self.alternate_description_value is MISSING:
            self.alternate_description_value = self.document.resolver.resolve_str(
                lookup_dict_key(self.props, "Alt")
            )
        return self.alternate_description_value

    @property
    def actual_text(self) -> str | None:
        if self.actual_text_value is MISSING:
            self.actual_text_value = self.document.resolver.resolve_str(
                lookup_dict_key(self.props, "ActualText")
            )
        return self.actual_text_value

    @property
    def attributes(self) -> StructureAttributes | None:
        if self.attributes_value is not MISSING:
            return self.attributes_value
        attrs = lookup_dict_key(self.props, "A")
        if isinstance(attrs, dict):
            self.attributes_value = {
                structure_key_name(key): coerce_value(val) for key, val in attrs.items()
            }
            return self.attributes_value
        if isinstance(attrs, list):
            if len(attrs) % 2 != 0:
                raise ValueError("invalid structure attribute array")
            latest: StructureAttributes | None = None
            latest_revision = -1
            for i in range(0, len(attrs), 2):
                attrdict = self.document.resolver.resolve(attrs[i])
                revision = self.document.resolver.resolve_int(attrs[i + 1])
                if not isinstance(attrdict, dict):
                    raise ValueError("invalid structure attribute entry")
                if revision is None:
                    raise ValueError("invalid structure attribute revision")
                if latest is None or revision > latest_revision:
                    latest = {
                        structure_key_name(key): coerce_value(val) for key, val in attrdict.items()
                    }
                    latest_revision = revision
            self.attributes_value = latest
            return latest
        self.attributes_value = None
        return None

    @property
    def class_name(self) -> str | None:
        if self.class_name_value is not MISSING:
            return self.class_name_value
        classes = self.document.resolver.resolve(lookup_dict_key(self.props, "C"))
        if isinstance(classes, list) and classes:
            latest = classes[-2] if len(classes) >= 2 else classes[-1]
            self.class_name_value = literal_name(latest)
            if self.class_name_value is None:
                raise ValueError("invalid structure class name")
            return self.class_name_value
        if classes is None:
            self.class_name_value = None
            return None
        self.class_name_value = literal_name(classes)
        if self.class_name_value is None:
            raise ValueError("invalid structure class name")
        return self.class_name_value

    @property
    def parent(self) -> StructureElement | StructureTree | None:
        if self.parent_value is not MISSING:
            return self.parent_value
        parent = self.document.resolver.resolve(lookup_dict_key(self.props, "P"))
        if parent is None:
            self.parent_value = None
            return None
        if not isinstance(parent, dict):
            raise ValueError("invalid structure parent entry")
        if self.document.resolver.resolve_name(lookup_dict_key(parent, "Type")) == "StructTreeRoot":
            tree = self.document.structure
            self.parent_value = tree
            return tree
        self.parent_value = StructureElement(self.document, cast(PdfDict, parent))
        return self.parent_value

    def __iter__(
        self,
    ) -> Iterator[StructureChild]:
        if self.kids_value is MISSING:
            self.kids_value = tuple(
                make_kids(lookup_dict_key(self.props, "K"), self.page, self.document)
            )
        yield from self.kids_value

    def find_all(self, matcher: str | MatchFunc | None = None) -> Iterator["StructureElement"]:
        elements: list[StructureChild] = list(self)
        filtered = [el for el in elements if isinstance(el, StructureElement)]
        return find_all(filtered, matcher)

    def find(self, matcher: str | MatchFunc | None = None) -> StructureElement | None:
        return find_first(self.find_all(matcher))


class StructureTree(Iterable[StructureElement | StructureContentItem | StructureContentObject]):
    """Document logical structure tree rooted at StructTreeRoot."""

    __slots__ = (
        "document",
        "props",
        "role_map_value",
        "parent_tree_value",
        "kids_value",
    )

    document: PdfDocument
    props: StructureDict
    role_map_value: dict[str, str] | None
    parent_tree_value: ParentTree | None

    def __init__(self, document: PdfDocument, props: StructureDict) -> None:
        self.document = document
        self.props = props if isinstance(props, dict) else {}
        self.role_map_value: dict[str, str] | None = None
        self.parent_tree_value: ParentTree | None = None
        self.kids_value: Any = MISSING

    @property
    def type(self) -> str:
        return "StructTreeRoot"

    @property
    def role(self) -> str:
        return "StructTreeRoot"

    @property
    def role_map(self) -> dict[str, str]:
        if self.role_map_value is not None:
            return self.role_map_value
        resolved = self.document.resolver.resolve(lookup_dict_key(self.props, "RoleMap"))
        role_map: dict[str, str] = {}
        if resolved is None:
            self.role_map_value = role_map
            return role_map
        if not isinstance(resolved, dict):
            raise ValueError("invalid role map dictionary")
        for key, value in resolved.items():
            mapped = self.document.resolver.resolve_name(
                value
            ) or self.document.resolver.resolve_str(value)
            if mapped is None:
                raise ValueError("invalid role map entry")
            role_map[structure_key_name(key)] = mapped
        self.role_map_value = role_map
        return role_map

    @property
    def parent_tree(self) -> ParentTree:
        if self.parent_tree_value is not None:
            return self.parent_tree_value
        resolved = self.document.resolver.resolve(lookup_dict_key(self.props, "ParentTree"))
        results: ParentTree = {}
        if resolved is None:
            self.parent_tree_value = results
            return results
        if not isinstance(resolved, dict):
            raise ValueError("invalid parent tree dictionary")
        recover_parent_tree = (
            self.document.xref_was_recovered or self.document.page_tree_was_recovered
        )

        results.update(
            iter_number_tree_items(
                resolved,
                self.document.resolver.resolve,
                decode_number=self.document.resolver.resolve_int,
                recover_entries=recover_parent_tree,
                resolve_values=False,
                tree_name="parent",
                max_depth=MAX_PARENT_TREE_DEPTH,
            )
        )
        self.parent_tree_value = results
        return results

    def __iter__(
        self,
    ) -> Iterator[StructureChild]:
        if self.kids_value is MISSING:
            self.kids_value = tuple(
                make_kids(lookup_dict_key(self.props, "K"), None, self.document)
            )
        yield from self.kids_value

    def find_all(self, matcher: str | MatchFunc | None = None) -> Iterator[StructureElement]:
        return find_all([item for item in self if isinstance(item, StructureElement)], matcher)

    def find(self, matcher: str | MatchFunc | None = None) -> StructureElement | None:
        return find_first(self.find_all(matcher))

    def page_structure(self, page: PdfPage) -> "PageStructure":
        key = self.document.resolver.resolve_int(lookup_dict_key(page.page_dict, "StructParents"))
        if type(key) is not int:
            raise ValueError("invalid page StructParents value")
        parent_tree = self.parent_tree
        if key not in parent_tree:
            raise ValueError("invalid page structure parent tree entry")
        parents = parent_tree[key]
        if not isinstance(parents, list):
            raise ValueError("invalid page structure parents")
        return PageStructure(page, parents)


class PageStructure(Sequence[StructureElement | None]):
    """Per-page parent-tree slice indexed by marked-content id."""

    __slots__ = ("elements", "page", "parents")

    page: PdfPage
    parents: ParentTreeParents
    elements: dict[int, StructureElement]

    def __init__(self, page: PdfPage, parents: Any) -> None:
        self.page = page
        if isinstance(parents, list):
            self.parents = parents
        elif parents is None:
            self.parents = []
        else:
            raise ValueError("invalid page structure parents")
        self.elements: dict[int, StructureElement] = {}

    def __len__(self) -> int:
        return len(self.parents)

    @overload
    def __getitem__(self, idx: int) -> StructureElement | None: ...

    @overload
    def __getitem__(self, idx: slice) -> "PageStructure": ...

    def __getitem__(self, idx: int | slice) -> PageStructure | StructureElement | None:
        if isinstance(idx, slice):
            return PageStructure(self.page, self.parents[idx])
        obj = self.parents[idx]
        if obj is None:
            return None
        if isinstance(obj, StructureElement):
            return obj
        if isinstance(obj, dict):
            marker = id(obj)
            if marker not in self.elements:
                self.elements[marker] = StructureElement(self.page.document, obj)
            return self.elements[marker]
        if isinstance(obj, PdfReference):
            resolved = self.page.document.resolver.resolve(obj)
            if isinstance(resolved, dict):
                marker = id(resolved)
                if marker not in self.elements:
                    self.elements[marker] = StructureElement(
                        self.page.document, cast(PdfDict, resolved)
                    )
                return self.elements[marker]
            raise ValueError("invalid page structure parent entry")
        raise ValueError("invalid page structure parent entry")

    def find_all(self, matcher: str | MatchFunc | None = None) -> Iterator[StructureElement]:
        seen: set[int] = set()
        match_func = make_match_func(matcher)
        for element in self:
            while isinstance(element, StructureElement):
                if match_func(element):
                    if id(element) not in seen:
                        seen.add(id(element))
                        yield element
                    break
                parent = element.parent
                if not isinstance(parent, StructureElement):
                    break
                element = parent

    def find(self, matcher: str | MatchFunc | None = None) -> StructureElement | None:
        return find_first(self.find_all(matcher))


StructureChild: TypeAlias = StructureElement | StructureContentItem | StructureContentObject


def get_kid_page_index(document: PdfDocument, page: PdfPage | None, kid: PdfDict) -> int | None:
    pg = lookup_dict_key(kid, "Pg")
    if pg is not None:
        page_obj = document.resolver.resolve(pg)
        index = document.page_index_for(page_obj)
        if index is None:
            raise ValueError("invalid structure page reference")
        return index
    if page is not None:
        return page.page_number - 1
    return None


def make_kids(
    kid: Any,
    page: PdfPage | None,
    document: PdfDocument,
    depth: int = 0,
) -> Iterator[StructureChild]:
    recover_structure = document.xref_was_recovered or document.page_tree_was_recovered
    stack: list[tuple[Any, int]] = [(kid, depth)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_STRUCTURE_DEPTH:
            if recover_structure:
                continue
            raise ValueError("invalid structure depth")
        if current is None:
            continue
        if isinstance(current, list):
            for item in reversed(current):
                stack.append((item, depth + 1))
            continue
        if type(current) is bool:
            if recover_structure:
                continue
            raise ValueError("invalid structure content mcid")
        if isinstance(current, int):
            if current < 0:
                if recover_structure:
                    continue
                raise ValueError("invalid structure content mcid")
            yield StructureContentItem(
                page_index=page.page_number - 1 if page is not None else None,
                mcid=current,
            )
            continue
        if isinstance(current, PdfReference):
            stack.append((document.resolver.resolve(current), depth + 1))
            continue
        if isinstance(current, dict):
            ktype_value = lookup_dict_key(current, "Type")
            ktype = document.resolver.resolve_name(ktype_value) or document.resolver.resolve_str(
                ktype_value
            )
            if ktype == "MCR":
                mcid = document.resolver.resolve_int(lookup_dict_key(current, "MCID"))
                if mcid is None:
                    if recover_structure:
                        continue
                    raise ValueError("invalid structure content mcid")
                yield StructureContentItem(
                    page_index=get_kid_page_index(document, page, current),
                    mcid=mcid,
                    stream=lookup_dict_key(current, "Stm"),
                )
                continue
            if ktype == "OBJR":
                obj = lookup_dict_key(current, "Obj")
                if obj is None:
                    if recover_structure:
                        continue
                    raise ValueError("invalid structure object reference")
                if not isinstance(obj, dict):
                    if recover_structure:
                        continue
                    raise ValueError("invalid structure object reference")
                yield StructureContentObject(
                    page_index=get_kid_page_index(document, page, current),
                    props=cast(PdfDict, obj),
                )
                continue
            yield StructureElement(document, current)
            continue
        if recover_structure:
            continue
        raise ValueError("invalid structure kid entry")
