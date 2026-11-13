# Ported from pypdf, whose notice follows.
#
# Copyright (c) 2006, Mathieu Fenniak
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * The name of the author may not be used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""What pypdf's reader and writer share, written over the generic object types.

It reads the catalog through ``root_object`` and never touches a file, so ``PdfWriter`` can
inherit it unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from core_pdf_compat.pypdf._configuration import get_configuration
from core_pdf_compat.pypdf.constants import CheckboxRadioButtonAttributes, Core
from core_pdf_compat.pypdf.constants import FieldDictionaryAttributes as FA  # noqa: N814
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    Destination,
    DictionaryObject,
    EmbeddedFile,
    Field,
    Fit,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
    ViewerPreferences,
    create_string_object,
    is_null_or_none,
)
from core_records import RecordType, ReplaceFields, ReprFields
lazy from core_pdf_compat.pypdf.errors import DeprecationError, LimitReachedError, PdfReadError

from ._generic_trees import _parse_pdf_date

logger = logging.getLogger(__name__)

_UNSET = object()


class TraversalState(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    """pypdf._utils._TraversalState: counts the entries a traversal has seen."""

    entry_count: int = 0
    has_logged: bool = False


FIELD_TYPE_NAMES = {"/Btn": "Button", "/Tx": "Text", "/Ch": "Choice", "/Sig": "Signature"}


class LazyDict(Mapping[Any, Any]):
    """A mapping from each key to a ``(function, argument)`` pair, called on lookup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        function, argument = self._raw_dict[key]
        return function(argument)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)

    def __str__(self) -> str:
        return f"LazyDict(keys={list(self.keys())})"


HEADERS = tuple(f"%PDF-{version}" for version in ("1.3", "1.4", "1.5", "1.6", "1.7", "2.0"))


def highest_header(first: str, second: str) -> str:
    """The later of two PDF headers. A malformed one does not count."""
    known = [HEADERS.index(header) for header in (first, second) if header in HEADERS]
    return HEADERS[max(known)] if known else first


def _info_text(key: str) -> property:
    return property(lambda self: self._get_text(key))


def _info_raw(key: str) -> property:
    return property(lambda self: self.get_resolved(key))


class DocumentInformation(DictionaryObject):
    """The document information dictionary, with a text and a ``_raw`` property per key."""

    def _get_text(self, key: str) -> str | None:
        value = self.get_resolved(key)
        if isinstance(value, TextStringObject):
            return value
        return str(value) if isinstance(value, ByteStringObject) else None

    @property
    def title(self) -> Any:
        title = self.get_resolved("/Title")
        return (self._get_text("/Title") or title) if title else None

    title_raw = _info_raw("/Title")
    author = _info_text("/Author")
    author_raw = _info_raw("/Author")
    subject = _info_text("/Subject")
    subject_raw = _info_raw("/Subject")
    creator = _info_text("/Creator")
    creator_raw = _info_raw("/Creator")
    producer = _info_text("/Producer")
    producer_raw = _info_raw("/Producer")
    keywords = _info_text("/Keywords")
    keywords_raw = _info_raw("/Keywords")
    creation_date_raw = _info_raw("/CreationDate")
    modification_date_raw = _info_raw("/ModDate")

    @property
    def creation_date(self) -> datetime | None:
        return _parse_pdf_date(self._get_text("/CreationDate"))

    @property
    def modification_date(self) -> datetime | None:
        return _parse_pdf_date(self._get_text("/ModDate"))


class _VirtualList(Sequence[Any]):
    """pypdf's list of pages, which looks pages up rather than holding them."""

    def __init__(self, length: Callable[[], int], get: Callable[[int], Any]) -> None:
        self.length_function = length
        self.get_function = get

    def __len__(self) -> int:
        return self.length_function()

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return type(self)(indices.__len__, lambda position: self[indices[position]])
        if not isinstance(index, int):
            raise TypeError("Sequence indices must be integers")
        count = len(self)
        position = index + count if index < 0 else index
        if not 0 <= position < count:
            raise IndexError("Sequence index out of range")
        return self.get_function(position)

    def __delitem__(self, index: Any) -> None:
        if isinstance(index, slice):
            for position in sorted(range(*index.indices(len(self))), reverse=True):
                del self[position]
            return
        if not isinstance(index, int):
            raise TypeError("Index must be integers")
        count = len(self)
        position = index + count if index < 0 else index
        if not 0 <= position < count:
            raise IndexError("Index out of range")
        child = self[position].indirect_reference
        document = child.pdf
        node: Any = child.get_object().get("/Parent")
        remove = True
        first = True
        while node is not None:
            node = node.get_object()
            kids = node["/Kids"]
            if remove and child not in kids:
                if first:
                    raise PdfReadError(f"Page not found in page tree: {child}")
                return
            if remove:
                del kids[kids.index(child)]
            if first and isinstance(getattr(document, "flattened_pages", None), list):
                del document.flattened_pages[position]
            first = False
            if "/Count" in node:
                node[NameObject("/Count")] = NumberObject(node["/Count"] - 1)
            # A branch left empty goes too. Above that, only the counts change.
            remove = len(kids) == 0
            child = node.indirect_reference
            node = node.get("/Parent")

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def __str__(self) -> str:
        return f"[{', '.join(f'PageObject({i})' for i in range(self.length_function()))}]"


class PdfDocCommon:
    strict: bool

    @property
    def pages(self) -> Any:
        raise NotImplementedError

    @property
    def root_object(self) -> Any:
        raise NotImplementedError

    def get_named_dest_root(self) -> ArrayObject:
        output = ArrayObject()
        catalog = self.root_object
        names = catalog.get_resolved("/Names")
        add_object = getattr(self, "_add_object", None)
        if isinstance(names, DictionaryObject):
            dests = names.get_resolved("/Dests")
            if isinstance(dests, DictionaryObject):
                if "/Names" in dests:
                    return dests["/Names"]
                dests[NameObject("/Names")] = output
            elif add_object is not None:
                dests = DictionaryObject()
                names[NameObject("/Dests")] = add_object(dests)
                dests[NameObject("/Names")] = output
        elif add_object is not None:
            names = DictionaryObject()
            catalog[NameObject("/Names")] = add_object(names)
            dests = DictionaryObject()
            names[NameObject("/Dests")] = add_object(dests)
            dests[NameObject("/Names")] = output
        return output

    def _replace_object(self, indirect: IndirectObject, obj: Any) -> Any:
        raise NotImplementedError

    @property
    def xmp_metadata(self) -> Any:
        return self.root_object.xmp_metadata

    def _get_page_number_by_indirect(self, indirect_reference: Any) -> int | None:
        raise NotImplementedError

    def get_destination_page_number(self, destination: Destination) -> int | None:
        return self._get_page_number_by_indirect(destination.page)

    @property
    def viewer_preferences(self) -> ViewerPreferences | None:
        preferences = self.root_object.get_resolved("/ViewerPreferences")
        if preferences is None:
            return None
        if not isinstance(preferences, DictionaryObject):
            logger.warning("Viewer preferences are not a dictionary: %s", preferences)
            return None
        if not isinstance(preferences, ViewerPreferences):
            preferences = ViewerPreferences(preferences)
            if (reference := getattr(preferences, "indirect_reference", None)) is not None:
                self._replace_object(reference, preferences)
            else:
                self.root_object[NameObject("/ViewerPreferences")] = preferences
        return preferences

    @property
    def threads(self) -> ArrayObject | None:
        return self.root_object.get_resolved(Core.THREADS)

    @property
    def open_destination(self) -> Any:
        return self._read_open_destination()

    def _read_open_destination(self) -> Any:
        action = self.root_object.get_resolved("/OpenAction")
        if isinstance(action, (bytes, str)):
            return create_string_object(action)
        if not isinstance(action, ArrayObject):
            return None
        try:
            page, kind, *arguments = list.__iter__(action)
            return Destination("OpenAction", page, Fit(kind, tuple(arguments)))
        except Exception as exc:
            raise Exception(f"Invalid Destination {action}: {exc}") from exc

    # Named destinations.

    @property
    def named_destinations(self) -> dict[str, Destination]:
        return self._get_named_destinations()

    def _get_named_destinations(
        self,
        *,
        tree: Any = None,
        retval: dict[str, Destination] | None = None,
        visited: set[int] | None = None,
        depth: int = 0,
        traversal_state: TraversalState | None = None,
    ) -> dict[str, Destination]:
        if traversal_state is None:
            traversal_state = TraversalState()
        if retval is None:
            retval = {}
            tree = self._destination_tree()
            if tree is not None and not isinstance(tree, DictionaryObject):
                logger.warning("Destination tree is not a dictionary: %s", tree)
                return retval
        if is_null_or_none(tree):
            return retval
        # pypdf 7.0.0 limits the depth and the node count of the destination name tree.
        configuration = get_configuration()
        if depth > configuration.named_destinations_maximum_depth:
            raise LimitReachedError(
                f"Maximum named destination depth exceeded: "
                f"{depth} > {configuration.named_destinations_maximum_depth}."
            )
        visited = set() if visited is None else visited
        if id(tree) in visited:
            logger.warning("Detected cycle in destination tree.")
            return retval
        visited.add(id(tree))
        traversal_state.entry_count += 1
        if traversal_state.entry_count > configuration.named_destinations_maximum_entries:
            raise LimitReachedError(
                f"Maximum named destination entry limit exceeded: "
                f"{traversal_state.entry_count} > "
                f"{configuration.named_destinations_maximum_entries}."
            )

        if "/Kids" in tree:
            kids = tree["/Kids"].get_object()
            if not isinstance(kids, ArrayObject):
                logger.warning("Destination tree kids are not an array: %s", kids)
                return retval
            for kid in kids:
                self._get_named_destinations(
                    tree=kid.get_object(),
                    retval=retval,
                    visited=visited,
                    depth=depth + 1,
                    traversal_state=traversal_state,
                )
        elif "/Names" in tree:
            names = tree["/Names"].get_object()
            if not isinstance(names, ArrayObject):
                logger.warning("Destination tree names are not an array: %s", names)
                return retval
            self._read_name_tree_leaf(names, retval)
        else:
            # PDF 1.1: /Dests in the catalog is a plain dictionary.
            for key, value in tree.items():
                target = self._destination_array(value.get_object())
                if target is not None:
                    retval[key] = self._build_destination(key, target)
        return retval

    def _destination_tree(self) -> Any:
        catalog = self.root_object
        if "/Dests" in catalog:
            return catalog["/Dests"].get_object()
        if "/Names" in catalog:
            names = catalog["/Names"].get_object()
            if isinstance(names, DictionaryObject) and "/Dests" in names:
                return names["/Dests"].get_object()
        return None

    @staticmethod
    def _destination_array(value: Any) -> Any:
        if not isinstance(value, DictionaryObject):
            return value
        return value["/D"].get_object() if "/D" in value else None

    def _read_name_tree_leaf(self, names: Any, retval: dict[str, Destination]) -> None:
        index = 0
        while index < len(names):
            key = names[index].get_object()
            index += 1
            if not isinstance(key, (bytes, str)):
                continue
            if index >= len(names):
                return
            value = names[index].get_object()
            index += 1
            target = self._destination_array(value)
            if isinstance(value, DictionaryObject) and target is None:
                continue
            destination = self._build_destination(key, target)
            retval[cast(str, destination["/Title"])] = destination
            retval[str(key)] = destination

    def _build_destination(self, title: Any, array: Any) -> Destination:
        # A valid destination is an array of at least a page and a fit type.
        if not isinstance(array, ArrayObject) or len(array) < 2:
            return Destination(title, NullObject(), Fit.fit())
        page, kind, *arguments = list.__iter__(array)
        try:
            return Destination(title, page, Fit(fit_type=kind, fit_args=arguments))
        except PdfReadError:
            logger.warning("Unknown destination: %r %s", title, arguments)
            if self.strict:
                raise
            first = self.pages[0].indirect_reference
            return Destination(title, NullObject() if first is None else first, Fit.fit())

    # Outlines.

    @property
    def outline(self) -> list[Any]:
        return self._get_outline()

    @property
    def outlines(self) -> list[Any]:
        return self.outline

    def _get_outline(
        self,
        *,
        node: Any = None,
        outline: list[Any] | None = None,
        visited: set[int] | None = None,
        depth: int = 0,
        traversal_state: TraversalState | None = None,
    ) -> list[Any]:
        if traversal_state is None:
            traversal_state = TraversalState()
        if outline is None:
            outline = []
            catalog = self.root_object
            if Core.OUTLINES in catalog:
                lines = catalog[Core.OUTLINES]
                # A reference to a missing or free object resolves to None.
                if is_null_or_none(lines):
                    return outline
                if not isinstance(lines, DictionaryObject):
                    logger.warning("Outlines are not a dictionary: %s", lines)
                    return outline
                if "/First" in lines:
                    node = lines["/First"]
            self._named_destinations = self._get_named_destinations()
        if node is None:
            return outline

        configuration = get_configuration()
        if depth > configuration.outline_maximum_depth:
            raise LimitReachedError(
                f"Maximum outline depth reached: {depth} > {configuration.outline_maximum_depth}."
            )
        visited = set() if visited is None else visited
        while True:
            if id(node) in visited:
                logger.warning("Detected cycle in outline structure for %s", node)
                break
            visited.add(id(node))
            traversal_state.entry_count += 1
            if traversal_state.entry_count > configuration.outline_maximum_entries:
                raise LimitReachedError(
                    "Maximum outline entry limit reached: "
                    f"{traversal_state.entry_count} > {configuration.outline_maximum_entries}."
                )
            if not isinstance(node, DictionaryObject):
                logger.warning("Outline node is not a dictionary: %s", node)
                break
            item = self._build_outline_item(node)
            if item:
                outline.append(item)
            if "/First" in node:
                children: list[Any] = []
                # A copy, so several entries may share the same children.
                self._get_outline(
                    node=node["/First"],
                    outline=children,
                    visited=visited.copy(),
                    depth=depth + 1,
                    traversal_state=traversal_state,
                )
                if children:
                    outline.append(children)
            if "/Next" not in node:
                break
            node = node["/Next"]
        return outline

    def _outline_target(self, node: Any) -> Any:
        if "/A" in node:
            # Only GoTo actions name a destination (ISO 32000-2 12.6).
            action = node["/A"]
            if "/S" not in action:
                if self.strict:
                    raise PdfReadError(f"Outline Action Missing /S attribute: {node!r}")
                return None
            if action["/S"] != "/GoTo":
                return None
            if "/D" in action:
                return action["/D"]
            if self.strict:
                raise PdfReadError(f"Outline Action Missing /D attribute: {node!r}")
            return None
        if "/Dest" not in node:
            return None
        target = node["/Dest"]
        # An array referenced from elsewhere comes wrapped in a dictionary.
        return target["/D"] if isinstance(target, DictionaryObject) and "/D" in target else target

    def _build_outline_item(self, node: Any) -> Destination:
        if "/Title" in node:
            title = node["/Title"]
        elif self.strict:
            raise PdfReadError(f"Outline Entry Missing /Title attribute: {node!r}")
        else:
            title = ""

        target = self._outline_target(node)
        if isinstance(target, ArrayObject):
            item = self._build_destination(title, target)
        elif isinstance(target, str):
            named = self._named_destinations.get(target)
            array = None if named is None else named.dest_array
            item = self._build_destination(title, array)
        elif target is None:
            item = self._build_destination(title, None)
        elif self.strict:
            raise PdfReadError(f"Unexpected destination {target!r}")
        else:
            logger.warning("Removed unexpected destination %r from destination", target)
            item = self._build_destination(title, None)

        color = node.get_resolved("/C")
        if isinstance(color, list):
            item[NameObject("/C")] = ArrayObject(FloatObject(part) for part in color)
        elif color is not None:
            logger.warning("Ignoring non-array outline color %r", color)
        if "/F" in node:
            item[NameObject("/F")] = node["/F"]
        if "/Count" in node:
            item[NameObject("/Count")] = node["/Count"]
        item[NameObject("/%is_open%")] = BooleanObject(node.get_resolved("/Count", 0) >= 0)
        item.node = node
        if hasattr(node, "indirect_reference"):
            item.indirect_reference = node.indirect_reference
        return item

    # Form fields.

    def get_fields(
        self,
        tree: Any = None,
        retval: dict[str, Any] | None = None,
        fileobj: Any = None,
        stack: Any = _UNSET,
        *,
        _visited: set[int] | None = None,
        _depth: int = 0,
        _traversal_state: TraversalState | None = None,
    ) -> dict[str, Any] | None:
        if stack is not _UNSET:
            raise DeprecationError("stack is deprecated and was removed in pypdf 7.0.0.")
        configuration = get_configuration()
        if _depth >= configuration.form_fields_maximum_depth:
            raise LimitReachedError("Reached maximum allowed form field depth.")
        state = TraversalState() if _traversal_state is None else _traversal_state

        def check_field_limit(increment: int) -> None:
            state.entry_count += increment
            if state.entry_count > configuration.form_fields_maximum_entries:
                raise LimitReachedError("Exceeded maximum number of fields allowed.")

        field_attributes = _field_labels()
        if retval is None:
            retval = {}
            _visited = set()
            catalog = self.root_object
            if "/AcroForm" not in catalog:
                return None
            entry = catalog["/AcroForm"]
            acro_form = None if entry is None else entry.get_object()
            if acro_form is not None and not isinstance(acro_form, DictionaryObject):
                logger.warning("AcroForm is not a dictionary: %s", acro_form)
                return None
            tree = acro_form
        if tree is None:
            return retval
        assert _visited is not None
        if "/Fields" in tree:
            fields = tree["/Fields"].get_object()
            if not isinstance(fields, ArrayObject):
                logger.warning("AcroForm /Fields is not an array: %s", fields)
                return retval
            check_field_limit(len(fields))
            for reference in fields:
                self._build_field(
                    field=reference.get_object(),
                    retval=retval,
                    fileobj=fileobj,
                    field_attributes=field_attributes,
                    visited=_visited,
                    depth=_depth,
                    traversal_state=state,
                )
        elif any(attribute in tree for attribute in field_attributes):
            check_field_limit(1)
            self._build_field(
                field=tree,
                retval=retval,
                fileobj=fileobj,
                field_attributes=field_attributes,
                visited=_visited,
                depth=_depth,
                traversal_state=state,
            )
        return retval

    def _get_qualified_field_name(
        self, *, parent: DictionaryObject, visited: set[int] | None = None
    ) -> str:
        visited = set() if visited is None else visited
        if id(parent) in visited:
            raise LimitReachedError(
                "Detected cycle in /Parent hierarchy when retrieving qualified field name."
            )
        visited.add(id(parent))
        if "/TM" in parent:
            return parent["/TM"]
        if "/Parent" in parent:
            prefix = self._get_qualified_field_name(parent=parent["/Parent"], visited=visited)
            return f"{prefix}.{parent.get_resolved('/T', '')}"
        return parent.get_resolved("/T", "")

    @staticmethod
    def _normal_appearance(appearance: Any) -> DictionaryObject:
        appearance = appearance.get_object()
        if not isinstance(appearance, DictionaryObject):
            raise PdfReadError(f"Expected appearance dictionary, got {appearance!r}")
        normal = appearance.get_resolved("/N")
        if not isinstance(normal, DictionaryObject):
            raise PdfReadError(f"Expected /N appearance dictionary, got {normal!r}")
        return normal

    def _button_states(self, field: Any) -> ArrayObject | None:
        if field.get_resolved(FA.FT, "") != "/Btn":
            return None
        if "/AP" in field:
            states = ArrayObject(self._normal_appearance(field["/AP"]).keys())
            if "/Off" not in states:
                states.append(NameObject("/Off"))
            return states
        flags = field.get_resolved(FA.Ff, 0)
        if not flags & FA.FfBits.Radio:
            return None
        states = ArrayObject()
        for reference in field.get_resolved(FA.Kids, ()):
            kid = reference.get_object()
            if "/AP" not in kid:
                raise PdfReadError(f"Button field kid missing /AP: {kid!r}")
            for state in self._normal_appearance(kid["/AP"]):
                if state not in states:
                    states.append(state)
        if flags & FA.FfBits.NoToggleToOff and "/Off" in states:
            states.remove("/Off")
        return states

    def _build_field(
        self,
        *,
        field: Any,
        retval: dict[str, Any],
        fileobj: Any,
        field_attributes: Any,
        visited: set[int],
        depth: int,
        traversal_state: TraversalState,
    ) -> None:
        del field_attributes
        if not isinstance(field, DictionaryObject):
            logger.warning("Form field is not a dictionary: %s", field)
            return
        if "/T" not in field and "/TM" not in field:
            return
        key = self._get_qualified_field_name(parent=field)
        if fileobj:
            self._write_field(fileobj, field)
            fileobj.write("\n")
        output = retval[key] = Field(field)
        if field.get_resolved(FA.FT, "") == "/Ch" and field.get_resolved(FA.Opt):
            output[NameObject("/_States_")] = field[FA.Opt]
        states = self._button_states(field)
        if states is not None:
            output[NameObject("/_States_")] = states
        # Last, so that a parent comes before its kids.
        self._check_kids(
            tree=field,
            retval=retval,
            fileobj=fileobj,
            visited=visited,
            depth=depth,
            traversal_state=traversal_state,
        )

    def _check_kids(
        self,
        *,
        tree: Any,
        retval: Any,
        fileobj: Any,
        visited: set[int],
        depth: int,
        traversal_state: TraversalState,
    ) -> None:
        if id(tree) in visited:
            logger.warning("%s already parsed", self._get_qualified_field_name(parent=tree))
            return
        visited.add(id(tree))
        if "/Kids" in tree:
            for reference in tree["/Kids"]:
                self.get_fields(
                    tree=reference.get_object(),
                    retval=retval,
                    fileobj=fileobj,
                    _visited=visited,
                    _depth=depth + 1,
                    _traversal_state=traversal_state,
                )

    @staticmethod
    def _write_field(fileobj: Any, field: Any) -> None:
        labels = _field_labels()
        # In pypdf's order of the attributes, which is not that of their labels.
        for attribute in FA.attributes() + CheckboxRadioButtonAttributes.attributes():
            if attribute in {FA.Kids, FA.AA} or attribute not in field:
                continue
            label = labels[attribute]
            value = field[attribute]
            if attribute == FA.FT:
                if value in FIELD_TYPE_NAMES:
                    fileobj.write(f"{label}: {FIELD_TYPE_NAMES[value]}\n")
            elif attribute == FA.Parent:
                name = value.get_resolved(FA.TM) or value.get_resolved(FA.T)
                if name is not None:
                    fileobj.write(f"{label}: {name}\n")
            else:
                fileobj.write(f"{label}: {value}\n")

    def get_form_text_fields(self, full_qualified_name: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, field in (self.get_fields() or {}).items():
            if field.get_resolved("/FT") != "/Tx":
                continue
            key = name if full_qualified_name else field.get_resolved("/T", name)
            if key in output and not full_qualified_name:
                repeats = sum(1 for other in output if other.startswith(f"{key}."))
                key = f"{key}.{repeats + 2}"
            output[key] = field.get_resolved("/V")
        return output

    def _pages_with_widget(self, widget: Any) -> list[Any]:
        if "/P" in widget:
            return [widget["/P"].get_object()]
        reference = widget.indirect_reference
        return [page for page in self.pages if reference in page.get_resolved("/Annots", ())]

    def get_pages_showing_field(self, field: Any) -> list[Any]:
        try:
            field = field.indirect_reference.get_object()
        except Exception as exc:
            raise ValueError("Field type is invalid") from exc
        if is_null_or_none(field.get_inherited(key="/FT", default=None)):
            raise ValueError("Field is not valid")
        if field.get_resolved("/Subtype", "") == "/Widget":
            found = self._pages_with_widget(field)
        else:
            found = []
            for reference in field.get_resolved("/Kids", ()):
                kid = reference.get_object()
                # A kid without /T is a widget of this field, not a field of its own.
                if kid.get_resolved("/Subtype", "") == "/Widget" and "/T" not in kid:
                    found += self._pages_with_widget(kid)
        return [self._page_of(item) for item in found]

    def _page_of(self, item: Any) -> Any:
        if not isinstance(item, DictionaryObject):
            return item
        return self.pages[cast(int, self._get_page_number_by_indirect(item.indirect_reference))]

    @property
    def xfa(self) -> dict[str, bytes] | None:
        form = self.root_object.get_resolved("/AcroForm")
        if not form:
            return None
        output: dict[str, bytes] = {}
        if "/XFA" not in form:
            return output
        fields = form["/XFA"].get_object()
        if not isinstance(fields, ArrayObject):
            logger.warning("XFA entry is not an array: %s", fields)
            return output
        # pypdf 7.0.0 decodes each packet through its filters and limits the total (#4363).
        accumulated_length = 0
        maximum_length = get_configuration().xfa_decoding_work_maximum_length
        entries = list.__iter__(fields)
        for tag in entries:
            # Tag and value pairs; a trailing tag has no value.
            value = next(entries, None)
            if value is None or is_null_or_none(value):
                continue
            stream = value.get_object()
            if is_null_or_none(stream):
                continue
            decoded = stream.get_data()
            accumulated_length += len(decoded)
            if accumulated_length > maximum_length:
                raise LimitReachedError(
                    "XFA decoding exceeded the maximum accumulated "
                    f"decoding work of {maximum_length} bytes."
                )
            output[tag] = decoded
        return output

    # Attachments.

    @property
    def attachment_list(self) -> Iterator[EmbeddedFile]:
        yield from EmbeddedFile._load(self.root_object, strict=self.strict)

    @property
    def attachments(self) -> Mapping[str, list[bytes]]:
        return LazyDict(
            {name: (self._get_attachment_list, name) for name in self._list_attachments()}
        )

    def _list_attachments(self) -> list[str]:
        names: list[str] = []
        for entry in self.attachment_list:
            names.append(entry.name)
            alternative = entry.alternative_name
            if alternative and alternative != entry.name:
                names.append(alternative)
        return names

    def _get_attachment_list(self, name: str) -> list[bytes]:
        content = self._get_attachments(name)[name]
        return content if isinstance(content, list) else [content]

    def _get_attachments(self, filename: str | None = None) -> dict[str, bytes | list[bytes]]:
        attachments: dict[str, bytes | list[bytes]] = {}
        for entry in self.attachment_list:
            alternative = entry.alternative_name
            if filename is None:
                names = {entry.name, alternative}
            elif filename in {entry.name, alternative}:
                names = {filename}
            else:
                continue
            for name in names - {None}:
                known = attachments.get(cast(str, name))
                if known is None:
                    attachments[cast(str, name)] = entry.content
                elif isinstance(known, list):
                    known.append(entry.content)
                else:
                    attachments[cast(str, name)] = [known, entry.content]
        return attachments


def _field_labels() -> dict[str, str]:
    return {**FA.attributes_dict(), **CheckboxRadioButtonAttributes.attributes_dict()}


__all__ = ("DocumentInformation", "LazyDict", "PdfDocCommon", "_VirtualList", "highest_header")
