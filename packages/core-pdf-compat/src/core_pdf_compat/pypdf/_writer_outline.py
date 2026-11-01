# Follows pypdf's outline handling, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""The parts of PdfWriter that build the outline and named destinations."""

from __future__ import annotations

import logging
from typing import Any

from .constants import OutlineFontFlag
from .generic import (
    PAGE_FIT,
    ArrayObject,
    BooleanObject,
    Destination,
    DictionaryObject,
    Fit,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
    TreeObject,
    create_string_object,
    hex_to_rgb,
)

logger = logging.getLogger(__name__)


def _create_outline_item(
    action: IndirectObject | None, title: str, color: Any, italic: bool, bold: bool
) -> TreeObject:
    item = TreeObject()
    if action is not None:
        item[NameObject("/A")] = action
    item[NameObject("/Title")] = create_string_object(title)
    if color:
        rgb = hex_to_rgb(color) if isinstance(color, str) else color
        item[NameObject("/C")] = ArrayObject(FloatObject(part) for part in rgb)
    if italic or bold:
        flags = (OutlineFontFlag.italic if italic else 0) + (OutlineFontFlag.bold if bold else 0)
        item[NameObject("/F")] = NumberObject(flags)
    return item


class OutlineWriter:
    _root_object: DictionaryObject
    pages: Any

    # Provided by the classes PdfWriter is built from.
    get_named_dest_root: Any
    _add_object: Any
    _replace_object: Any

    def get_outline_root(self) -> TreeObject:
        if "/Outlines" not in self._root_object:
            outline = TreeObject()
            self._root_object[NameObject("/Outlines")] = self._add_object(outline)
            return outline
        outline = self._root_object["/Outlines"]
        if not isinstance(outline, TreeObject):
            tree = TreeObject(outline)
            self._replace_object(outline.indirect_reference.idnum, tree)
            outline = tree
        return outline

    def add_outline_item_destination(
        self, page_destination: Any, parent: Any = None, before: Any = None, is_open: bool = True
    ) -> IndirectObject:
        from ._writer import PageObject

        target = page_destination.get_object()
        if isinstance(target, PageObject):
            return self.add_outline_item_destination(
                Destination(f"page #{target.page_number}", target.indirect_reference, Fit.fit())
            )
        parent = self.get_outline_root() if parent is None else parent.get_object()
        target[NameObject("/%is_open%")] = BooleanObject(is_open)
        reference = self._add_object(target)
        parent.insert_child(
            reference,
            None if before is None else before.indirect_reference,
            self,
            target.inc_parent_counter_outline if is_open else (lambda _node, _count: 0),
        )
        if "/Count" not in target:
            target[NameObject("/Count")] = NumberObject(0)
        return reference

    def add_outline_item_dict(
        self, outline_item: Any, parent: Any = None, before: Any = None, is_open: bool = True
    ) -> IndirectObject:
        item = TreeObject()
        for key, value in outline_item.items():
            dict.__setitem__(item, NameObject(key), value)
        return self.add_outline_item_destination(item, parent, before, is_open)

    def _page_reference(self, page_number: Any) -> Any:
        from ._writer import PageObject

        reference: Any = None
        if isinstance(page_number, IndirectObject):
            reference = page_number
        elif isinstance(page_number, PageObject):
            reference = page_number.indirect_reference
        elif isinstance(page_number, int):
            try:
                reference = self.pages[page_number].indirect_reference
            except IndexError:
                reference = NumberObject(page_number)
        if reference is None:
            logger.warning("can not find reference of page %s", page_number)
            return NullObject()
        return reference

    def add_outline_item(
        self,
        title: str,
        page_number: Any,
        parent: Any = None,
        before: Any = None,
        color: Any = None,
        bold: bool = False,
        italic: bool = False,
        fit: Fit = PAGE_FIT,
        is_open: bool = True,
    ) -> IndirectObject:
        if isinstance(italic, Fit):
            # The arguments of earlier versions, which had no `before`.
            old_fit = italic
            return self.add_outline_item(
                title, page_number, parent, None, before, color, bold, old_fit, is_open=is_open
            )
        action = None
        if page_number is not None:
            destination = Destination(
                NameObject(f"/{title} outline item"), self._page_reference(page_number), fit
            )
            action = self._add_object(
                DictionaryObject(
                    {
                        NameObject("/D"): destination.dest_array,
                        NameObject("/S"): NameObject("/GoTo"),
                    }
                )
            )
        item = self._add_object(_create_outline_item(action, title, color, italic, bold))
        parent = self.get_outline_root() if parent is None else parent
        return self.add_outline_item_destination(item, parent, before, is_open)

    def add_outline(self) -> None:
        raise NotImplementedError(
            "This method is not yet implemented. Use :meth:`add_outline_item` instead."
        )

    def find_outline_item(self, outline_item: Any, root: Any = None) -> list[int] | None:
        node: Any = self.get_outline_root() if root is None else root
        index = 0
        while True:
            if node.indirect_reference == outline_item or node.get("/Title") == outline_item:
                return [index]
            if "/First" in node:
                found = self.find_outline_item(outline_item, node["/First"])
                if found:
                    return ([index] if "/Title" in node else []) + found
            if "/Next" not in node:
                return None
            index += 1
            node = node["/Next"]

    def add_named_destination_array(self, title: Any, destination: Any) -> None:
        names = self.get_named_dest_root()
        for index in range(0, len(names), 2):
            if title < names[index]:
                names.insert(index, destination)
                names.insert(index, TextStringObject(title))
                return
        names.extend([TextStringObject(title), destination])

    def add_named_destination_object(self, page_destination: Any) -> IndirectObject:
        reference = self._add_object(page_destination.dest_array)
        self.add_named_destination_array(page_destination["/Title"], reference)
        return reference

    def add_named_destination(self, title: str, page_number: int) -> IndirectObject:
        page = self.pages[page_number].indirect_reference
        target = ArrayObject([page, NameObject("/FitH"), NumberObject(826)])
        reference = self._add_object(
            DictionaryObject({NameObject("/D"): target, NameObject("/S"): NameObject("/GoTo")})
        )
        text = title if isinstance(title, TextStringObject) else TextStringObject(str(title))
        self.add_named_destination_array(text, reference)
        return reference


__all__ = ["OutlineWriter"]
