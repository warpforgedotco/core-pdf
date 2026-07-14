# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Callable, overload

from core_pdf.syntax.primitives import MISSING, PdfReference, coerce_value

if TYPE_CHECKING:
    from core_pdf.document.document import PdfDocument
    from core_pdf.document.page import PdfPage


MAX_STRUCTURE_DEPTH = 200
MAX_PARENT_TREE_DEPTH = 100

MatchFunc = Callable[["StructureElement"], bool]


def make_match_func(matcher: str | re.Pattern[str] | MatchFunc | None = None) -> MatchFunc:
    if matcher is None:
        return lambda _: True
    if isinstance(matcher, str):
        return lambda x: x.role == matcher
    if isinstance(matcher, re.Pattern):
        return lambda x: bool(matcher.match(x.role))
    return matcher


def find_all(
    elements: list["StructureElement"],
    matcher: str | re.Pattern[str] | MatchFunc | None = None,
) -> Iterator["StructureElement"]:
    match_func = make_match_func(matcher)
    elements.reverse()
    while elements:
        el = elements.pop()
        if match_func(el):
            yield el
        for child in reversed(list(el)):
            if isinstance(child, StructureElement):
                elements.append(child)


def literal_name(value: Any) -> str | None:
    if isinstance(value, PdfReference):
        return None
    if value is None:
        return None
    text = str(value)
    return text[1:] if text.startswith("/") else text


def coerce_attr_value(value: Any) -> Any:
    return coerce_value(value)


class StructureContentItem:
    __slots__ = ("page_index", "mcid", "stream")

    page_index: int | None
    mcid: int
    stream: Any

    def __init__(self, page_index: int | None, mcid: int, stream: Any = None) -> None:
        if page_index is not None and not isinstance(page_index, int):
            raise ValueError("invalid structure content page index")
        if isinstance(page_index, int) and page_index < 0:
            raise ValueError("invalid structure content page index")
        if not isinstance(mcid, int):
            raise ValueError("invalid structure content mcid")
        if mcid < 0:
            raise ValueError("invalid structure content mcid")
        self.page_index = page_index
        self.mcid = mcid
        self.stream = stream


class StructureContentObject:
    __slots__ = ("page_index", "props")

    page_index: int | None
    props: dict[str, Any]

    def __init__(self, page_index: int | None, props: dict[str, Any]) -> None:
        if page_index is not None and not isinstance(page_index, int):
            raise ValueError("invalid structure content page index")
        if isinstance(page_index, int) and page_index < 0:
            raise ValueError("invalid structure content page index")
        if not isinstance(props, dict):
            raise ValueError("invalid structure content props")
        self.page_index = page_index
        self.props = props


class StructureElement:
    __slots__ = (
        "document",
        "props",
        "role_value",
        "type_value",
        "kids_value",
        "title_value",
        "language_value",
        "alternate_description_value",
        "actual_text_value",
        "attributes_value",
        "class_name_value",
        "parent_value",
    )

    def __init__(self, document: PdfDocument, props: dict[str, Any]) -> None:
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
            self.type_value = self.document.resolver.resolve_name_like_value(self.props.get("S"))
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
        page_ref = self.props.get("Pg")
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
            self.title_value = self.document.resolver.resolve_str(self.props.get("T"))
        return self.title_value

    @property
    def language(self) -> str | None:
        if self.language_value is MISSING:
            self.language_value = self.document.resolver.resolve_str(self.props.get("Lang"))
        return self.language_value

    @property
    def alternate_description(self) -> str | None:
        if self.alternate_description_value is MISSING:
            self.alternate_description_value = self.document.resolver.resolve_str(
                self.props.get("Alt")
            )
        return self.alternate_description_value

    @property
    def actual_text(self) -> str | None:
        if self.actual_text_value is MISSING:
            self.actual_text_value = self.document.resolver.resolve_str(
                self.props.get("ActualText")
            )
        return self.actual_text_value

    @property
    def attributes(self) -> dict[str, Any] | None:
        if self.attributes_value is not MISSING:
            return self.attributes_value
        attrs = self.props.get("A")
        if isinstance(attrs, dict):
            self.attributes_value = {str(key): coerce_attr_value(val) for key, val in attrs.items()}
            return self.attributes_value
        if isinstance(attrs, list):
            if len(attrs) % 2 != 0:
                raise ValueError("invalid structure attribute array")
            latest: dict[str, Any] | None = None
            latest_revision = -1
            for i in range(0, len(attrs), 2):
                attrdict = self.document.resolver.resolve(attrs[i])
                revision = self.document.resolver.resolve_int(attrs[i + 1])
                if not isinstance(attrdict, dict):
                    raise ValueError("invalid structure attribute entry")
                if revision is None:
                    raise ValueError("invalid structure attribute revision")
                if latest is None or revision > latest_revision:
                    latest = {str(key): coerce_attr_value(val) for key, val in attrdict.items()}
                    latest_revision = revision
            self.attributes_value = latest
            return latest
        self.attributes_value = None
        return None

    @property
    def class_name(self) -> str | None:
        if self.class_name_value is not MISSING:
            return self.class_name_value
        classes = self.document.resolver.resolve(self.props.get("C"))
        if isinstance(classes, list) and classes:
            latest = classes[-2] if len(classes) >= 2 else classes[-1]
            if not isinstance(latest, (str, PdfReference)):
                raise ValueError("invalid structure class name")
            self.class_name_value = literal_name(latest)
            if self.class_name_value is None:
                raise ValueError("invalid structure class name")
            return self.class_name_value
        if classes is None:
            self.class_name_value = None
            return None
        if not isinstance(classes, (str, PdfReference)):
            raise ValueError("invalid structure class name")
        self.class_name_value = literal_name(classes)
        return self.class_name_value

    @property
    def parent(self) -> StructureElement | StructureTree | None:
        if self.parent_value is not MISSING:
            return self.parent_value
        parent = self.document.resolver.resolve(self.props.get("P"))
        if parent is None:
            self.parent_value = None
            return None
        if not isinstance(parent, dict):
            raise ValueError("invalid structure parent entry")
        if self.document.resolver.resolve_name(parent.get("Type")) == "StructTreeRoot":
            tree = self.document.structure
            self.parent_value = tree
            return tree
        self.parent_value = StructureElement(self.document, parent)
        return self.parent_value

    def __iter__(
        self,
    ) -> Iterator[StructureElement | StructureContentItem | StructureContentObject]:
        if self.kids_value is MISSING:
            self.kids_value = tuple(make_kids(self.props.get("K"), self.page, self.document))
        yield from self.kids_value

    def find_all(
        self, matcher: str | re.Pattern[str] | MatchFunc | None = None
    ) -> Iterator["StructureElement"]:
        elements: list[StructureElement | StructureContentItem | StructureContentObject] = list(
            self
        )
        filtered = [el for el in elements if isinstance(el, StructureElement)]
        return find_all(filtered, matcher)

    def find(
        self, matcher: str | re.Pattern[str] | MatchFunc | None = None
    ) -> StructureElement | None:
        try:
            return next(self.find_all(matcher))
        except StopIteration:
            return None

    def __hash__(self) -> int:
        return hash((id(self.document), repr(self.props)))


def get_kid_page_index(
    document: PdfDocument, page: PdfPage | None, kid: dict[str, Any]
) -> int | None:
    pg = kid.get("Pg")
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
    _depth: int = 0,
) -> Iterator[StructureElement | StructureContentItem | StructureContentObject]:
    if _depth > MAX_STRUCTURE_DEPTH:
        raise ValueError("invalid structure depth")
    if kid is None:
        return
    if isinstance(kid, list):
        for item in kid:
            yield from make_kids(item, page, document, _depth + 1)
        return
    if isinstance(kid, int):
        if kid < 0:
            raise ValueError("invalid structure content mcid")
        page_index = page.page_number - 1 if page is not None else None
        yield StructureContentItem(page_index=page_index, mcid=kid)
        return
    if isinstance(kid, PdfReference):
        resolved = document.resolver.resolve(kid)
        yield from make_kids(resolved, page, document, _depth + 1)
        return
    if isinstance(kid, dict):
        ktype = document.resolver.resolve_name(kid.get("Type")) or document.resolver.resolve_str(
            kid.get("Type")
        )
        if ktype == "MCR":
            mcid = document.resolver.resolve_int(kid.get("MCID"))
            if mcid is None:
                raise ValueError("invalid structure content mcid")
            yield StructureContentItem(
                page_index=get_kid_page_index(document, page, kid),
                mcid=mcid,
                stream=kid.get("Stm"),
            )
            return
        if ktype == "OBJR":
            obj = kid.get("Obj")
            if obj is None:
                raise ValueError("invalid structure object reference")
            if not isinstance(obj, dict):
                raise ValueError("invalid structure object reference")
            yield StructureContentObject(
                page_index=get_kid_page_index(document, page, kid),
                props=obj,
            )
            return
        yield StructureElement(document, kid)
        return
    raise ValueError("invalid structure kid entry")


class StructureTree(Iterable[StructureElement | StructureContentItem | StructureContentObject]):
    __slots__ = ("document", "props", "role_map_value", "parent_tree_value", "kids_value")

    def __init__(self, document: PdfDocument, props: dict[str, Any]) -> None:
        self.document = document
        self.props = props if isinstance(props, dict) else {}
        self.role_map_value: dict[str, str] | None = None
        self.parent_tree_value: dict[int, Any] | None = None
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
        resolved = self.document.resolver.resolve(self.props.get("RoleMap"))
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
            role_map[str(key)] = mapped
        self.role_map_value = role_map
        return role_map

    @property
    def parent_tree(self) -> dict[int, Any]:
        if self.parent_tree_value is not None:
            return self.parent_tree_value
        resolved = self.document.resolver.resolve(self.props.get("ParentTree"))
        results: dict[int, Any] = {}
        if resolved is None:
            self.parent_tree_value = results
            return results
        if not isinstance(resolved, dict):
            raise ValueError("invalid parent tree dictionary")

        def walk(node: Any, _depth: int = 0) -> None:
            if _depth > MAX_PARENT_TREE_DEPTH:
                raise ValueError("invalid parent tree depth")
            node = self.document.resolver.resolve(node)
            if not isinstance(node, dict):
                raise ValueError("invalid parent tree node")
            nums = node.get("Nums")
            if isinstance(nums, list):
                if len(nums) % 2 != 0:
                    raise ValueError("invalid parent tree Nums array")
                for i in range(0, len(nums), 2):
                    key = self.document.resolver.resolve_int(nums[i])
                    if key is not None:
                        results[key] = nums[i + 1]
            elif nums is not None:
                raise ValueError("invalid parent tree Nums array")
            kids = node.get("Kids")
            if isinstance(kids, list):
                for kid in kids:
                    walk(kid, _depth + 1)
            elif kids is not None:
                raise ValueError("invalid parent tree Kids array")

        walk(resolved)
        self.parent_tree_value = results
        return results

    def __iter__(
        self,
    ) -> Iterator[StructureElement | StructureContentItem | StructureContentObject]:
        if self.kids_value is MISSING:
            self.kids_value = tuple(make_kids(self.props.get("K"), None, self.document))
        yield from self.kids_value

    def find_all(
        self, matcher: str | re.Pattern[str] | MatchFunc | None = None
    ) -> Iterator[StructureElement]:
        return find_all([item for item in self if isinstance(item, StructureElement)], matcher)

    def find(
        self, matcher: str | re.Pattern[str] | MatchFunc | None = None
    ) -> StructureElement | None:
        try:
            return next(self.find_all(matcher))
        except StopIteration:
            return None

    def page_structure(self, page: PdfPage) -> "PageStructure":
        key = self.document.resolver.resolve_int(page.page_dict.get("StructParents"))
        if key is None:
            return PageStructure(page, [])
        parent_tree = self.parent_tree
        if key not in parent_tree:
            raise ValueError("invalid page structure parent tree entry")
        parents = parent_tree[key]
        if not isinstance(parents, list):
            raise ValueError("invalid page structure parents")
        return PageStructure(page, parents)


class PageStructure(Sequence[StructureElement | None]):
    __slots__ = ("page", "parents", "elements")

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
                    self.elements[marker] = StructureElement(self.page.document, resolved)
                return self.elements[marker]
            raise ValueError("invalid page structure parent entry")
        raise ValueError("invalid page structure parent entry")

    def __iter__(self) -> Iterator[StructureElement | None]:
        for i in range(len(self.parents)):
            yield self[i]

    def find_all(
        self, matcher: str | re.Pattern[str] | MatchFunc | None = None
    ) -> Iterator[StructureElement]:
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

    def find(
        self, matcher: str | re.Pattern[str] | MatchFunc | None = None
    ) -> StructureElement | None:
        try:
            return next(self.find_all(matcher))
        except StopIteration:
            return None
