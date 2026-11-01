# Follows pypdf's merging, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""The parts of PdfWriter that merge another document: pages, outline, links, forms, articles.

Only what the selected pages use comes across. An outline item, a named destination or a link
aimed at a page left out is dropped, so a partial merge carries no titles from those pages.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO, IOBase
from pathlib import Path
from re import Pattern
from typing import Any, cast

from .errors import LimitReachedError
from .generic import (
    ArrayObject,
    Destination,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
    TreeObject,
    is_null_or_none,
)
from .pagerange import PageRange

logger = logging.getLogger(__name__)


class MergeWriter:
    _objects: list[Any]
    _root_object: DictionaryObject
    pages: Any
    threads: Any

    # Provided by the classes PdfWriter is built from.
    get_named_dest_root: Any
    get_outline_root: Any
    add_outline_item: Any
    add_named_destination_array: Any
    _add_object: Any
    _translation: Any
    _import: Any
    add_page: Any
    insert_page: Any

    @staticmethod
    def _create_stream(fileobj: Any) -> BytesIO:
        if isinstance(fileobj, (str, Path)):
            return BytesIO(Path(fileobj).read_bytes())
        if hasattr(fileobj, "seek") and hasattr(fileobj, "read"):
            fileobj.seek(0)
            return BytesIO(fileobj.read())
        raise NotImplementedError(
            "Merging requires an object that PdfReader can parse. "
            "Typically, that is a Path or a string representing a Path, "
            "a file object, or an object implementing .seek and .read. "
            "Passing a PdfReader directly works as well."
        )

    def append(
        self,
        fileobj: Any,
        outline_item: Any = None,
        pages: Any = None,
        import_outline: bool = True,
        excluded_fields: Any = None,
    ) -> None:
        excluded = () if excluded_fields is None else excluded_fields
        if isinstance(outline_item, (tuple, list, PageRange)):
            if isinstance(pages, bool):
                if not isinstance(import_outline, bool):
                    excluded = import_outline
                import_outline = pages
            self.merge(None, fileobj, None, outline_item, import_outline, excluded)
        else:
            self.merge(None, fileobj, outline_item, pages, import_outline, excluded)

    @staticmethod
    def _selected_pages(pages: Any, reader: Any) -> list[Any]:
        count = len(reader.pages)
        if pages is None:
            return list(range(count))
        if isinstance(pages, PageRange):
            return list(range(*pages.indices(count)))
        if isinstance(pages, list):
            return pages
        if isinstance(pages, tuple) and len(pages) <= 3:
            return list(range(*pages))
        raise TypeError('"pages" must be a tuple of (start, stop[, step]) or a list')

    def merge(
        self,
        position: int | None,
        fileobj: Any,
        outline_item: str | None = None,
        pages: Any = None,
        import_outline: bool = True,
        excluded_fields: Any = (),
    ) -> None:
        from . import PdfReader
        from ._doc_common import PdfDocCommon
        from ._writer import PageObject

        if isinstance(fileobj, PdfDocCommon):
            reader: Any = fileobj
        elif isinstance(fileobj, (IOBase, str, Path)) or hasattr(fileobj, "read"):
            reader = PdfReader(self._create_stream(fileobj), strict=False)
        else:
            self._create_stream(fileobj)
        excluded = list(() if excluded_fields is None else excluded_fields)
        source_pages: dict[int, Any] = {}
        for page in self._selected_pages(pages, reader):
            original = page if isinstance(page, PageObject) else reader.pages[page]
            number = cast(IndirectObject, original.indirect_reference).idnum
            cut = [*excluded, "/B", "/Annots"]
            if position is None:
                source_pages[number] = self.add_page(original, cut)
            else:
                source_pages[number] = self.insert_page(original, position, cut)
                position += 1
            source_pages[number].original_page = original

        reader._named_destinations = reader.named_destinations
        for destination in reader._named_destinations.values():
            self._merge__process_named_dests(destination, reader, source_pages)

        root = reader.root_object
        if outline_item is not None and source_pages:
            first = next(iter(source_pages.values())).indirect_reference
            parent = self.add_outline_item(TextStringObject(outline_item), first).get_object()
        else:
            parent = self.get_outline_root()
        if import_outline and "/Outlines" in root:
            filtered = self._get_filtered_outline(
                node=root.get("/Outlines"), pages=source_pages, reader=reader
            )
            self._insert_filtered_outline(filtered, parent, None)

        if "/Annots" not in excluded:
            for added in source_pages.values():
                links = self._insert_filtered_annotations(
                    added.original_page.get("/Annots", []), added, source_pages, reader
                )
                if len(links) > 0:
                    added[NameObject("/Annots")] = links
                self.clean_page(added)
        self._merge_form(reader)
        if "/B" not in excluded:
            self.add_filtered_articles("", source_pages, reader)

    def _merge_form(self, reader: Any) -> None:
        root = reader.root_object
        if "/AcroForm" not in root or is_null_or_none(root["/AcroForm"]):
            return
        form = root["/AcroForm"]
        translated = self._translation(reader)
        if "/AcroForm" not in self._root_object:
            kept = DictionaryObject()
            for key, value in dict.items(form):
                if key != "/Fields":
                    dict.__setitem__(kept, key, value)
            imported = self._import(kept, reader)
            self._root_object[NameObject("/AcroForm")] = self._add_object(imported)
            fields: Any = ArrayObject()
        else:
            fields = self._root_object["/AcroForm"].get_resolved("/Fields", ArrayObject())
        source_fields = form.get_resolved("/Fields")
        if not isinstance(source_fields, ArrayObject):
            source_fields = ArrayObject()
        for field in source_fields:
            number = translated.get(getattr(field, "idnum", None))
            if number is not None and IndirectObject(number, 0, self) not in fields:
                fields.append(IndirectObject(number, 0, self))
        self._root_object["/AcroForm"][NameObject("/Fields")] = fields

    def _merge__process_named_dests(self, dest: Any, reader: Any, pages: dict[int, Any]) -> None:
        array: Any = dest.dest_array
        names = self._root_object.get_resolved("/Names", DictionaryObject())
        known = names.get_resolved("/Dests", DictionaryObject()).get_resolved("/Names", ())
        page = dest["/Page"]
        if dest["/Title"] in known or is_null_or_none(page):
            return
        if isinstance(page, int):
            # A page number instead of a reference is only meant for remote destinations.
            try:
                number = reader.pages[page].indirect_reference.idnum
            except IndexError:
                return
            if number in pages:
                array[NumberObject(0)] = NumberObject(pages[number].page_number)
                self.add_named_destination_array(dest["/Title"], array)
        elif page.indirect_reference.idnum in pages:
            array[NumberObject(0)] = pages[page.indirect_reference.idnum].indirect_reference
            self.add_named_destination_array(dest["/Title"], array)

    def _clone(self, value: Any, reader: Any, ignored: tuple[str, ...] = ()) -> Any:
        """A copy of a source dictionary without some keys, with what it refers to."""
        reference = value.indirect_reference
        if not ignored and reference is not None and reference.pdf is reader:
            # Recorded as a copy, so that /Fields can find a field that came with a page.
            return self._import(reference, reader).get_object()
        kept = DictionaryObject()
        for key, item in dict.items(value):
            if key not in ignored:
                dict.__setitem__(kept, key, item)
        return self._import(kept, reader)

    def _add_articles_thread(self, thread: Any, pages: dict[int, Any], reader: Any) -> Any:
        new_thread = self._clone(thread, reader, ("/F",))
        reference = self._add_object(new_thread)
        self.threads.append(reference)
        if thread.indirect_reference is not None:
            self._translation(reader)[thread.indirect_reference.idnum] = reference.idnum
        first = thread["/F"]
        current: Any = first
        article: Any = None
        new_first: Any = None
        visited: set[int] = set()
        while current is not None:
            if id(current) in visited:
                raise LimitReachedError("Detected cyclic article structure.")
            visited.add(id(current))
            page = self._get_cloned_page(current["/P"], pages, reader)
            if page is not None:
                following = DictionaryObject()
                self._add_object(following)
                if article is None:
                    new_first = following
                    new_thread[NameObject("/F")] = following.indirect_reference
                else:
                    following[NameObject("/V")] = article.indirect_reference
                    article[NameObject("/N")] = following.indirect_reference
                article = following
                article[NameObject("/P")] = page
                article[NameObject("/T")] = reference
                article[NameObject("/R")] = current["/R"]
                page_object = page.get_object()
                if "/B" not in page_object:
                    page_object[NameObject("/B")] = ArrayObject()
                page_object["/B"].append(article.indirect_reference)
            current = current["/N"]
            if current == first:
                if article is not None:
                    article[NameObject("/N")] = new_first.indirect_reference
                    new_first[NameObject("/V")] = article.indirect_reference
                current = None
        return reference

    def add_filtered_articles(self, fltr: Any, pages: dict[int, Any], reader: Any) -> None:
        pattern = re.compile(fltr) if isinstance(fltr, str) else fltr
        if not isinstance(pattern, Pattern):
            pattern = re.compile("")
        translated = self._translation(reader)
        for page in pages.values():
            for item in page.original_page.get("/B", ()):
                article = item.get_object()
                if is_null_or_none(article):
                    continue
                thread = article.get("/T")
                if thread is None:
                    continue
                thread = thread.get_object()
                info = thread.get_resolved("/I", {})
                if thread.indirect_reference.idnum not in translated and pattern.search(
                    info.get("/Title", "")
                ):
                    self._add_articles_thread(thread, pages, reader)

    def _get_cloned_page(self, page: Any, pages: dict[int, Any], reader: Any) -> Any:
        if isinstance(page, NullObject):
            return None
        if isinstance(page, int):
            # A destination can name the page by its index in the source document.
            try:
                page = reader.pages[page].indirect_reference
            except IndexError:
                return None
        if isinstance(page, DictionaryObject) and page.get_resolved("/Type") == "/Page":
            page = page.indirect_reference
        added = pages.get(getattr(page, "idnum", -1))
        return None if added is None else added.indirect_reference

    def _link_with_page(self, page: Any, reader: Any, array: ArrayObject) -> Any:
        return ArrayObject([page, *(self._import(item, reader) for item in array[1:])])

    def _insert_filtered_annotations(
        self, annots: Any, page: Any, pages: dict[int, Any], reader: Any
    ) -> ArrayObject:
        del page
        output = ArrayObject()
        if isinstance(annots, IndirectObject):
            annots = annots.get_object()
        if annots is None:
            return output
        if not isinstance(annots, list):
            logger.warning(
                "Expected list of annotations, got %s of type %s.", annots, type(annots).__name__
            )
            return output
        for item in annots:
            annotation = item.get_object()
            action = annotation.get_resolved("/A")
            go_to = (
                annotation.get_resolved("/Subtype") == "/Link"
                and isinstance(action, DictionaryObject)
                and action.get_resolved("/S") == "/GoTo"
                and "/Dest" not in annotation
            )
            if go_to:
                self._filter_go_to(annotation, action, pages, reader, output)
            elif "/Dest" not in annotation:
                output.append(self._add_object(self._clone(annotation, reader)))
            else:
                self._filter_dest(annotation, pages, reader, output)
        return output

    def _filter_dest(self, annotation: Any, pages: Any, reader: Any, output: ArrayObject) -> None:
        target = annotation["/Dest"]
        if isinstance(target, str):
            if str(target) in self.get_named_dest_root():
                output.append(self._add_object(self._clone(annotation, reader)))
        elif isinstance(target, ArrayObject) and target:
            page = self._get_cloned_page(target[0], pages, reader)
            if page is not None:
                copy = self._clone(annotation, reader, ("/Dest",))
                copy[NameObject("/Dest")] = self._link_with_page(page, reader, target)
                output.append(self._add_object(copy))

    def _filter_go_to(
        self, annotation: Any, action: Any, pages: Any, reader: Any, output: ArrayObject
    ) -> None:
        target = action.get_resolved("/D")
        if is_null_or_none(target):
            return
        if isinstance(target, str):
            if str(target) in self.get_named_dest_root():
                output.append(self._add_object(self._clone(annotation, reader)))
        elif isinstance(target, ArrayObject) and target:
            page = self._get_cloned_page(target[0], pages, reader)
            if page is not None:
                copy = self._clone(annotation, reader, ("/A",))
                new_action = self._clone(action, reader, ("/D",))
                new_action[NameObject("/D")] = self._link_with_page(page, reader, target)
                copy[NameObject("/A")] = new_action
                output.append(self._add_object(copy))

    def clean_page(self, page: Any) -> Any:
        """Turn destinations given as names into strings, which name trees require."""
        page = page.get_object()
        for item in page.get_resolved("/Annots", ()):
            annotation = item.get_object()
            target = annotation.get_resolved("/Dest")
            action = annotation.get_resolved("/A")
            if isinstance(target, NameObject):
                annotation[NameObject("/Dest")] = TextStringObject(target)
            elif action is not None:
                target = action.get_resolved("/D")
                if isinstance(target, NameObject):
                    action[NameObject("/D")] = TextStringObject(target)
        return page

    def _get_filtered_outline(
        self, *, node: Any, pages: dict[int, Any], reader: Any, visited: set[int] | None = None
    ) -> list[Destination]:
        """The outline items that are part of the given pages.

        pypdf keeps an item for a page left out when an item below it stays, with its title.
        Here the item goes and what stays below it moves up, so no such title is carried.
        """
        visited = set() if visited is None else visited
        output: list[Destination] = []
        if node is None:
            return output
        node = node.get_object()
        if is_null_or_none(node):
            node = DictionaryObject()
        if node.get_resolved("/Type", "") == "/Outlines" or "/Title" not in node:
            if id(node) in visited:
                logger.warning("Detected cycle in outlines.")
                return []
            visited.add(id(node))
            first = node.get("/First")
            if first is not None:
                output += self._get_filtered_outline(
                    node=first.get_object(), pages=pages, reader=reader, visited=visited
                )
            return output
        while True:
            node = cast(Any, node.get_object())
            if id(node) in visited:
                logger.warning("Detected cycle in outlines.")
                break
            visited.add(id(node))
            destination = reader._build_outline_item(node)
            target = destination["/Page"]
            page = self._get_cloned_page(target, pages, reader)
            children: list[Destination] = []
            if "/First" in node:
                children = self._get_filtered_outline(
                    node=node["/First"], pages=pages, reader=reader, visited=visited
                )
            destination[NameObject("/Page")] = NullObject() if page is None else page
            destination._filtered_children = children
            if page is not None:
                output.append(destination)
            elif children and is_null_or_none(target):
                # A heading with no destination of its own.
                output.append(destination)
            else:
                output += children
            if "/Next" not in node:
                break
            node = node["/Next"]
        return output

    def _clone_outline(self, destination: Destination) -> TreeObject:
        item = TreeObject()
        self._add_object(item)
        item[NameObject("/Title")] = TextStringObject(destination["/Title"])
        node = destination.node
        if not isinstance(destination["/Page"], NullObject):
            if node is not None and "/A" in node:
                action = node["/A"]
                reader = getattr(node.indirect_reference, "pdf", None)
                item[NameObject("/A")] = self._clone(action, reader)
                if "/D" in action and isinstance(action["/D"], ArrayObject):
                    item["/A"][NameObject("/D")] = destination.dest_array
            else:
                item[NameObject("/Dest")] = destination.dest_array
        if node is not None:
            item[NameObject("/F")] = NumberObject(node.get_resolved("/F", 0))
            color = node.get_resolved("/C")
            if not isinstance(color, list):
                color = [FloatObject(0.0), FloatObject(0.0), FloatObject(0.0)]
            item[NameObject("/C")] = ArrayObject(color)
        return item

    def _insert_filtered_outline(
        self, outlines: list[Destination], parent: Any, before: Any = None
    ) -> None:
        for destination in outlines:
            if destination.get_resolved("/Type", "") == "/Outlines" or "/Title" not in destination:
                item = parent
            else:
                item = self._clone_outline(destination)
                parent.get_object().insert_child(item, before, self)
            self._insert_filtered_outline(destination._filtered_children, item, None)

    def reset_translation(self, reader: Any = None) -> None:
        from . import PdfReader

        copies: dict[int, Any] = getattr(self, "_id_translated")
        if reader is None:
            copies.clear()
        elif isinstance(reader, PdfReader):
            copies.pop(id(reader), None)
        elif isinstance(reader, IndirectObject):
            copies.pop(id(reader.pdf), None)
        else:
            raise Exception("invalid parameter {reader}")


__all__ = ["MergeWriter"]
