# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.pdf_values import coerce_value
from core_pdf.impl.recovery_trees import iter_number_tree_items
from core_pdf.impl.types import MISSING, PdfReference
from core_pdf_spec.exceptions import PdfError
from core_pdf_spec.s_07_syntax.types import PdfArray, PdfDict, PdfObject
from core_pdf_spec.s_14_structure.dictionaries import (
    attribute_entries,
    marked_content_id,
    parse_role_map,
)
from core_pdf_spec.s_14_structure.roles import StructureRole, resolve_structure_role
from core_pdf_spec.standards import recognized_version

if TYPE_CHECKING:
    from core_pdf.impl.document_document import PageLookup, PdfDocument
    from core_pdf.impl.document_page import PdfPage


class StructureContentItem:
    __slots__ = ("page_index", "mcid", "stream")

    page_index: int | None
    mcid: int
    stream: PdfObject

    def __init__(self, page_index: int | None, mcid: int, stream: Any = None) -> None:
        if page_index is not None and type(page_index) is not int:
            raise ValueError("invalid structure content page index")
        if type(page_index) is int and page_index < 0:
            raise ValueError("invalid structure content page index")
        marked_content_id(mcid)
        self.page_index = page_index
        self.mcid = mcid
        self.stream = stream


class StructureContentObject:
    __slots__ = ("page_index", "props")

    page_index: int | None
    props: PdfDict

    def __init__(self, page_index: int | None, props: PdfDict) -> None:
        if page_index is not None and type(page_index) is not int:
            raise ValueError("invalid structure content page index")
        if type(page_index) is int and page_index < 0:
            raise ValueError("invalid structure content page index")
        if not isinstance(props, dict):
            raise ValueError("invalid structure content props")
        self.page_index = page_index
        self.props = props


MAX_PARENT_TREE_DEPTH = 100
MAX_STRUCTURE_DEPTH = 200

MatchFunc = Callable[["StructureElement"], bool]
StructureAttributes: TypeAlias = dict[str, Any]
ParentTree: TypeAlias = dict[int, Any]


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
        stack.extend(child for child in reversed(list(el)) if isinstance(child, StructureElement))


def structure_key_name(key: Any) -> str:
    return recover_pdf_name(key) or str(key)


def structure_attributes(attrs: PdfDict) -> StructureAttributes:
    return {structure_key_name(key): coerce_value(val) for key, val in attrs.items()}


class StructureNode:
    __slots__ = ("document", "page_lookup", "kids_value", "props")

    def __init__(
        self,
        document: PdfDocument[Any],
        props: PdfDict,
        *,
        page_lookup: PageLookup[Any] | None = None,
    ) -> None:
        self.document = document
        self.page_lookup = page_lookup
        self.props = props if isinstance(props, dict) else {}
        self.kids_value: Any = MISSING

    def kids_page(self) -> PdfPage | None:
        return None

    def __iter__(self) -> Iterator[StructureChild]:
        if self.kids_value is MISSING:
            self.kids_value = tuple(
                make_kids(
                    self.props.get("K"),
                    self.kids_page(),
                    self.document,
                    page_lookup=self.page_lookup,
                )
            )
        yield from self.kids_value

    def find_all(self, matcher: str | MatchFunc | None = None) -> Iterator[StructureElement]:
        return find_all([item for item in self if isinstance(item, StructureElement)], matcher)

    def find(self, matcher: str | MatchFunc | None = None) -> StructureElement | None:
        return next(self.find_all(matcher), None)


class StructureElement(StructureNode):
    __slots__ = (
        "actual_text_value",
        "alternate_description_value",
        "attributes_value",
        "class_name_value",
        "element_cache",
        "language_value",
        "parent_value",
        "role_resolution_value",
        "role_error_value",
        "title_value",
        "type_value",
    )

    def __init__(
        self,
        document: PdfDocument[Any],
        props: PdfDict,
        *,
        page_lookup: PageLookup[Any] | None = None,
        element_cache: dict[int, StructureElement] | None = None,
    ) -> None:
        super().__init__(document, props, page_lookup=page_lookup)
        # Shared with the page structure this element came from, so a parent
        # that many elements share is one object, resolved once.
        self.element_cache = element_cache
        self.role_resolution_value: StructureRole | None | object = MISSING
        self.role_error_value: str | None = None
        self.type_value: Any = MISSING
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
        result = self.role_resolution
        return result.name if result is not None else self.type or ""

    @property
    def role_namespace(self) -> str | None:
        result = self.role_resolution
        return result.namespace if result is not None else None

    @property
    def role_error(self) -> str | None:
        return self.role_error_value if self.role_resolution is None else None

    @property
    def role_resolution(self) -> StructureRole | None:
        cached = self.role_resolution_value
        if cached is None or isinstance(cached, StructureRole):
            return cached
        resolver = self.document.resolver
        context = resolver.semantic_context
        if recognized_version(context) is None:
            context = None
        try:
            tree = self.document.structure
            result = resolve_structure_role(
                self.props.get("S"),
                namespace=self.props.get("NS"),
                role_map=tree.props.get("RoleMap") if tree is not None else None,
                resolve=resolver.resolve,
                resolve_name=resolver.resolve_name_or_text,
                decode_text=resolver.decode_text,
                context=context,
            )
        except (PdfError, ValueError, RecursionError) as exc:
            self.role_error_value = str(exc)
            result = None
        self.role_resolution_value = result
        return result

    @property
    def page_index(self) -> int | None:
        return get_kid_page_index(self.document, None, self.props, self.page_lookup)

    @property
    def page(self) -> PdfPage | None:
        page_index = self.page_index
        if page_index is None:
            return None
        return self.document.pages[page_index]

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
    def attributes(self) -> StructureAttributes | None:
        if self.attributes_value is not MISSING:
            return self.attributes_value
        attrs = self.props.get("A")
        if isinstance(attrs, dict):
            self.attributes_value = structure_attributes(attrs)
            return self.attributes_value
        if isinstance(attrs, list):
            if len(attrs) % 2 != 0:
                raise ValueError("invalid structure attribute array")
            latest: StructureAttributes | None = None
            latest_revision = -1
            for entry in attribute_entries(
                attrs,
                resolve=self.document.resolver.resolve,
                resolve_revision=self.document.resolver.resolve_int,
            ):
                if not isinstance(entry.value, dict):
                    raise ValueError("invalid structure attribute entry")
                if not entry.explicit_revision:
                    raise ValueError("invalid structure attribute revision")
                attrdict, revision = entry.value, entry.revision
                if latest is None or revision > latest_revision:
                    latest = structure_attributes(attrdict)
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
        if classes is None:
            self.class_name_value = None
            return None
        if isinstance(classes, list) and classes:
            classes = classes[-2] if len(classes) >= 2 else classes[-1]
        name = recover_pdf_name(classes)
        if name is None:
            raise ValueError("invalid structure class name")
        self.class_name_value = name
        return name

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
            if tree is not None and self.page_lookup is not None:
                tree.page_lookup = self.page_lookup
            self.parent_value = tree
            return tree
        cache = self.element_cache
        if cache is not None:
            shared = cache.get(id(parent))
            if shared is not None and shared.props is parent:
                self.parent_value = shared
                return shared
        self.parent_value = StructureElement(
            self.document, parent, page_lookup=self.page_lookup, element_cache=cache
        )
        if cache is not None:
            cache[id(parent)] = self.parent_value
        return self.parent_value

    def kids_page(self) -> PdfPage | None:
        return self.page


class StructureTree(StructureNode):
    __slots__ = ("role_map_value", "parent_tree_value")

    role_map_value: dict[str, str] | None
    parent_tree_value: ParentTree | None

    def __init__(
        self,
        document: PdfDocument[Any],
        props: PdfDict,
        *,
        page_lookup: PageLookup[Any] | None = None,
    ) -> None:
        super().__init__(document, props, page_lookup=page_lookup)
        self.role_map_value: dict[str, str] | None = None
        self.parent_tree_value: ParentTree | None = None

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
        role_map = parse_role_map(
            ((structure_key_name(key), value) for key, value in resolved.items()),
            self.document.resolver.resolve_name_or_text,
        )
        self.role_map_value = role_map
        return role_map

    @property
    def parent_tree(self) -> ParentTree:
        if self.parent_tree_value is not None:
            return self.parent_tree_value
        resolved = self.document.resolver.resolve(self.props.get("ParentTree"))
        results: ParentTree = {}
        if resolved is None:
            self.parent_tree_value = results
            return results
        if not isinstance(resolved, dict):
            raise ValueError("invalid parent tree dictionary")
        results.update(
            iter_number_tree_items(
                resolved,
                self.document.resolver.resolve,
                decode_number=self.document.resolver.resolve_int,
                on_malformed_entry=self.document.recovery_policy(),
                resolve_values=False,
                tree_name="parent",
                max_depth=MAX_PARENT_TREE_DEPTH,
            )
        )
        self.parent_tree_value = results
        return results

    def page_structure(self, page: PdfPage) -> PageStructure:
        key = self.document.resolver.resolve_int(page.page_dict.get("StructParents"))
        if key is None:
            raise ValueError("invalid page StructParents value")
        parent_tree = self.parent_tree
        if key not in parent_tree:
            raise ValueError("invalid page structure parent tree entry")
        parents = parent_tree[key]
        if not isinstance(parents, list):
            raise ValueError("invalid page structure parents")
        return PageStructure(page, parents, page_lookup=self.page_lookup)


class PageStructure(Sequence[StructureElement | None]):
    __slots__ = ("elements", "page", "parents", "page_lookup")

    page: PdfPage
    parents: PdfArray
    elements: dict[int, StructureElement]

    def __init__(
        self,
        page: PdfPage,
        parents: Any,
        *,
        page_lookup: PageLookup[Any] | None = None,
    ) -> None:
        self.page = page
        self.page_lookup = page_lookup
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
    def __getitem__(self, idx: slice) -> PageStructure: ...

    def __getitem__(self, idx: int | slice) -> PageStructure | StructureElement | None:
        if isinstance(idx, slice):
            return PageStructure(self.page, self.parents[idx], page_lookup=self.page_lookup)
        obj = self.parents[idx]
        if obj is None:
            return None
        if isinstance(obj, StructureElement):
            return obj
        if isinstance(obj, PdfReference):
            obj = self.page.document.resolver.resolve(obj)
        if not isinstance(obj, dict):
            raise ValueError("invalid page structure parent entry")
        marker = id(obj)
        if marker not in self.elements:
            self.elements[marker] = StructureElement(
                self.page.document,
                obj,
                page_lookup=self.page_lookup,
                element_cache=self.elements,
            )
        return self.elements[marker]

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
        return next(self.find_all(matcher), None)


StructureChild: TypeAlias = StructureElement | StructureContentItem | StructureContentObject


def get_kid_page_index(
    document: PdfDocument[Any],
    page: PdfPage | None,
    kid: PdfDict,
    page_lookup: PageLookup[Any] | None = None,
) -> int | None:
    pg = kid.get("Pg")
    if pg is not None:
        page_obj = document.resolver.resolve(pg)
        index = (page_lookup or document).page_index_for(page_obj)
        if index is None:
            raise ValueError("invalid structure page reference")
        return index
    if page is not None:
        return page.page_number - 1
    return None


def make_kids(
    kid: Any,
    page: PdfPage | None,
    document: PdfDocument[Any],
    depth: int = 0,
    *,
    page_lookup: PageLookup[Any] | None = None,
) -> Iterator[StructureChild]:
    malformed = document.recovery_policy()
    stack: list[tuple[Any, int]] = [(kid, depth)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_STRUCTURE_DEPTH:
            malformed("invalid structure depth")
            continue
        if current is None:
            continue
        if isinstance(current, list):
            stack.extend((item, depth + 1) for item in reversed(current))
            continue
        if type(current) is bool:
            malformed("invalid structure content mcid")
            continue
        if isinstance(current, int):
            if current < 0:
                malformed("invalid structure content mcid")
                continue
            yield StructureContentItem(
                page_index=page.page_number - 1 if page is not None else None,
                mcid=current,
            )
            continue
        if isinstance(current, PdfReference):
            stack.append((document.resolver.resolve(current), depth + 1))
            continue
        if isinstance(current, dict):
            ktype_value = current.get("Type")
            ktype = document.resolver.resolve_name_or_text(ktype_value)
            if ktype == "MCR":
                mcid = document.resolver.resolve_int(current.get("MCID"))
                if mcid is None:
                    malformed("invalid structure content mcid")
                    continue
                yield StructureContentItem(
                    page_index=get_kid_page_index(document, page, current, page_lookup),
                    mcid=mcid,
                    stream=current.get("Stm"),
                )
                continue
            if ktype == "OBJR":
                obj = document.resolver.resolve(current.get("Obj"))
                if not isinstance(obj, dict):
                    malformed("invalid structure object reference")
                    continue
                yield StructureContentObject(
                    page_index=get_kid_page_index(document, page, current, page_lookup),
                    props=obj,
                )
                continue
            yield StructureElement(document, current, page_lookup=page_lookup)
            continue
        malformed("invalid structure kid entry")
