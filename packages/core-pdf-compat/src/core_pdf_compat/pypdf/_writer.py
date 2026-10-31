"""pypdf's PdfWriter, over a table of generic objects and core-pdf's serializer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from core_pdf.impl.document_editing import BINARY_MARKER, serialize_body
from core_pdf.impl.types import PdfReference

from ._doc_common import PdfDocCommon
from .generic import (
    ArrayObject,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    RectangleObject,
    StreamObject,
    TextStringObject,
    to_core,
)


class WriterPage(DictionaryObject):
    """A page held by a writer. pypdf's PageObject is a dictionary too."""

    @property
    def mediabox(self) -> RectangleObject:
        return RectangleObject(self.get_resolved("/MediaBox"))

    @property
    def cropbox(self) -> RectangleObject:
        box = self.get_resolved("/CropBox")
        return self.mediabox if box is None else RectangleObject(box)

    @property
    def rotation(self) -> int:
        return int(self.get_resolved("/Rotate", 0))

    def rotate(self, angle: int) -> WriterPage:
        if angle % 90:
            raise ValueError("Rotation angle must be a multiple of 90")
        self[NameObject("/Rotate")] = NumberObject((self.rotation + angle) % 360)
        return self


class WriterPages:
    def __init__(self, writer: PdfWriter) -> None:
        self.writer = writer

    def __len__(self) -> int:
        return len(self.writer._page_list())

    def __iter__(self) -> Iterator[WriterPage]:
        return iter(self.writer._page_list())

    def __getitem__(self, index: Any) -> Any:
        return self.writer._page_list()[index]

    def __delitem__(self, index: Any) -> None:
        self.writer.remove_page(index)


class PdfWriter(PdfDocCommon):
    def __init__(
        self,
        fileobj: Any = "",
        clone_from: Any = None,
        incremental: bool = False,
        full: bool = False,
        strict: bool = False,
    ) -> None:
        del full, strict
        if incremental:
            raise NotImplementedError("core-pdf has no PDF writer: incremental mode")
        self.fileobj = fileobj if isinstance(fileobj, (str, Path)) and fileobj else ""
        source = clone_from
        if source is None and fileobj != "" and not isinstance(fileobj, (str, Path)):
            source = fileobj
        path = Path(fileobj) if isinstance(fileobj, (str, Path)) and fileobj else None
        if source is None and path is not None and path.exists() and path.stat().st_size:
            source = fileobj
        self._objects: list[Any] = []
        self._pages_object = self._new_pages_tree()
        catalog = DictionaryObject()
        catalog[NameObject("/Type")] = NameObject("/Catalog")
        catalog[NameObject("/Pages")] = self._pages_object.indirect_reference
        self._root_object = catalog
        self._add_object(catalog)
        self._info = DictionaryObject()
        self._info[NameObject("/Producer")] = TextStringObject("core-pdf")
        self._add_object(self._info)
        if source is not None:
            self.append(source)

    def _new_pages_tree(self) -> DictionaryObject:
        pages = DictionaryObject()
        pages[NameObject("/Type")] = NameObject("/Pages")
        pages[NameObject("/Count")] = NumberObject(0)
        pages[NameObject("/Kids")] = ArrayObject()
        self._add_object(pages)
        return pages

    def __enter__(self) -> PdfWriter:
        return self

    def __exit__(self, *_: object) -> None:
        if self.fileobj:
            self.write(self.fileobj)

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        # Tests that reach a method not built yet skip rather than fail.
        if name.startswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(f"core-pdf has no PDF writer: {name}")

    def _add_object(self, value: Any) -> IndirectObject:
        reference = getattr(value, "indirect_reference", None)
        if reference is not None and reference.pdf is self:
            return reference
        self._objects.append(value)
        reference = IndirectObject(len(self._objects), 0, self)
        value.indirect_reference = reference
        return reference

    def get_object(self, indirect_reference: Any) -> Any:
        number = getattr(indirect_reference, "idnum", indirect_reference)
        return self._objects[number - 1] if 0 < number <= len(self._objects) else None

    @property
    def root_object(self) -> DictionaryObject:
        return self._root_object

    @property
    def pdf_header(self) -> str:
        return "%PDF-1.7"

    @property
    def metadata(self) -> DictionaryObject:
        return self._info

    def add_metadata(self, infos: dict[str, Any]) -> None:
        for key, value in infos.items():
            text = value if isinstance(value, TextStringObject) else TextStringObject(str(value))
            self._info[NameObject(key)] = text

    @property
    def pages(self) -> WriterPages:  # ty: ignore[invalid-property-type-override]
        return WriterPages(self)

    def _page_list(self) -> list[WriterPage]:
        kids = self._pages_object.get_resolved("/Kids")
        return [list.__getitem__(kids, index).get_object() for index in range(len(kids))]

    def _insert(self, page: WriterPage, index: int) -> WriterPage:
        reference = self._add_object(page)
        page[NameObject("/Parent")] = self._pages_object.indirect_reference
        kids = self._pages_object.get_resolved("/Kids")
        kids.insert(index, reference)
        self._pages_object[NameObject("/Count")] = NumberObject(len(kids))
        return page

    def add_page(self, page: Any, excluded_keys: Iterable[str] = ()) -> WriterPage:
        return self.insert_page(page, len(self.pages), excluded_keys)

    def insert_page(
        self, page: Any, index: int = 0, excluded_keys: Iterable[str] = ()
    ) -> WriterPage:
        return self._insert(self._import_page(page, set(excluded_keys)), index)

    def add_blank_page(self, width: float | None = None, height: float | None = None) -> WriterPage:
        return self.insert_blank_page(width, height, len(self.pages))

    def insert_blank_page(
        self, width: float | None = None, height: float | None = None, index: int = 0
    ) -> WriterPage:
        if width is None or height is None:
            if not len(self.pages):
                raise ValueError("Page size must be given when the document has no pages")
            box = self.pages[max(index - 1, 0)].mediabox
            width, height = box.width, box.height
        page = WriterPage()
        page[NameObject("/Type")] = NameObject("/Page")
        page[NameObject("/MediaBox")] = RectangleObject((0, 0, width, height))
        page[NameObject("/Resources")] = DictionaryObject()
        return self._insert(page, index)

    def remove_page(self, page: Any) -> None:
        kids = self._pages_object.get_resolved("/Kids")
        index = page if isinstance(page, int) else self._page_list().index(page)
        del kids[index]
        self._pages_object[NameObject("/Count")] = NumberObject(len(kids))

    def _import_page(self, page: Any, excluded: set[str]) -> WriterPage:
        source = page if isinstance(page, DictionaryObject) else page._generic_dictionary()
        owner = getattr(source.indirect_reference, "pdf", None)
        copies: dict[int, Any] = {}
        output = WriterPage()
        for key, value in dict.items(source):
            if key in {"/Parent", "/Annots", *excluded}:
                continue
            dict.__setitem__(output, key, self._import(value, owner, copies))
        for key in ("/Resources", "/MediaBox", "/CropBox", "/Rotate"):
            if key not in output and key not in excluded and hasattr(source, "get_inherited"):
                inherited = source.get_inherited(key)
                if inherited is not None:
                    dict.__setitem__(
                        output, NameObject(key), self._import(inherited, owner, copies)
                    )
        return output

    def _import(self, value: Any, owner: Any, copies: dict[int, Any]) -> Any:
        if isinstance(value, IndirectObject):
            if value.pdf is self:
                return value
            if value.idnum in copies:
                return copies[value.idnum]
            target = value.get_object()
            if isinstance(target, DictionaryObject) and target.get_resolved("/Type") == "/Page":
                return NullObject()
            copy = self._empty_copy(target)
            reference = self._add_object(copy)
            copies[value.idnum] = reference
            self._fill_copy(copy, target, owner, copies)
            return reference
        copy = self._empty_copy(value)
        self._fill_copy(copy, value, owner, copies)
        return copy

    @staticmethod
    def _empty_copy(value: Any) -> Any:
        if isinstance(value, StreamObject):
            copy = type(value)()
            # The stored bytes stay encoded, so they keep matching the copied /Filter.
            copy._data = value._stored_data()
            return copy
        if isinstance(value, DictionaryObject):
            return DictionaryObject()
        if isinstance(value, ArrayObject):
            return ArrayObject()
        if isinstance(value, (NumberObject, FloatObject, str, bytes)):
            return value
        return value

    def _fill_copy(self, copy: Any, value: Any, owner: Any, copies: dict[int, Any]) -> None:
        if isinstance(value, DictionaryObject):
            for key, item in dict.items(value):
                dict.__setitem__(copy, key, self._import(item, owner, copies))
        elif isinstance(value, ArrayObject):
            for index in range(len(value)):
                copy.append(self._import(list.__getitem__(value, index), owner, copies))

    def append(self, source: Any, outline_item: Any = None, pages: Any = None, **_: Any) -> None:
        del outline_item
        from . import PdfReader

        reader = source if isinstance(source, PdfReader) else self._reader_for(source)
        indexes = range(len(reader.pages)) if pages is None else self._page_indexes(pages, reader)
        for index in indexes:
            self.add_page(reader.pages[index])

    @staticmethod
    def _page_indexes(pages: Any, reader: Any) -> Iterable[int]:
        if isinstance(pages, (list, tuple)) and not isinstance(pages, range):
            if len(pages) in {2, 3} and all(isinstance(item, int) for item in pages):
                return range(*pages)
            return pages
        if hasattr(pages, "indices"):
            return range(*pages.indices(len(reader.pages)))
        return pages

    @staticmethod
    def _reader_for(source: Any) -> Any:
        from . import PdfReader

        return PdfReader(source)

    def clone_document_from_reader(self, reader: Any, *_: Any) -> None:
        self.append(reader)

    def append_pages_from_reader(self, reader: Any, *_: Any) -> None:
        self.append(reader)

    def write(self, stream: Any) -> tuple[bool, Any]:
        data = self._serialized()
        if isinstance(stream, (str, Path)):
            Path(stream).write_bytes(data)
            return True, stream
        stream.write(data)
        return False, stream

    def _serialized(self) -> bytes:
        objects = {
            PdfReference(index, 0): to_core(item) for index, item in enumerate(self._objects, 1)
        }
        trailer = {
            "Root": to_core(self._root_object.indirect_reference),
            "Info": to_core(self._info.indirect_reference),
        }
        header = b"%PDF-1.7\n" + BINARY_MARKER
        return header + serialize_body(len(header), objects, trailer)

    def write_bytes(self) -> bytes:
        return self._serialized()


__all__ = ("PdfWriter", "WriterPage")
