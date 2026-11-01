"""pypdf's PdfWriter, over a table of generic objects and core-pdf's serializer."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

from core_pdf.impl.document_editing import BINARY_MARKER, serialize_body
from core_pdf.impl.types import PdfReference

from ._doc_common import PdfDocCommon
from ._generic_trees import EmbeddedFile
from ._page_merge import (
    Transformation,
    add_cm,
    as_matrix,
    merge_page,
    merge_resources,
    scale_page,
    transfer_rotation,
    transform_page,
)
from .actions import Action
from .errors import PageSizeNotDefinedError
from .generic import (
    ArrayObject,
    ContentStream,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    RectangleObject,
    StreamObject,
    TextStringObject,
    is_null_or_none,
    to_core,
)


class PageObject(DictionaryObject, ABC):
    """A page as a dictionary, like pypdf's. It may be held by a writer or stand alone."""

    def __init__(self, pdf: Any = None, indirect_reference: Any = None) -> None:
        super().__init__()
        self._pdf = pdf
        self.indirect_reference = indirect_reference

    @staticmethod
    def create_blank_page(
        pdf: Any = None, width: float | None = None, height: float | None = None
    ) -> PageObject:
        if width is None or height is None:
            if pdf is None or not len(pdf.pages):
                raise PageSizeNotDefinedError
            box = pdf.pages[len(pdf.pages) - 1].mediabox
            width, height = box.width, box.height
        page = PageObject(pdf)
        page[NameObject("/Type")] = NameObject("/Page")
        page[NameObject("/Parent")] = NullObject()
        page[NameObject("/Resources")] = DictionaryObject()
        page[NameObject("/MediaBox")] = RectangleObject((0, 0, width, height))
        return page

    def _merge_resources(
        self, res1: Any, res2: Any, resource: str, new_res1: bool = True
    ) -> tuple[DictionaryObject, dict[str, NameObject]]:
        return merge_resources(self, res1, res2, resource, new_res1)

    def scale(self, sx: float, sy: float) -> None:
        scale_page(self, sx, sy)

    def scale_by(self, factor: float) -> None:
        self.scale(factor, factor)

    def scale_to(self, width: float, height: float) -> None:
        box = self.mediabox
        self.scale(width / float(box.width), height / float(box.height))

    @property
    def mediabox(self) -> RectangleObject:
        return RectangleObject(self.get_resolved("/MediaBox"))

    @property
    def cropbox(self) -> RectangleObject:
        box = self.get_resolved("/CropBox")
        return self.mediabox if box is None else RectangleObject(box)

    def _box_or_cropbox(self, key: str) -> RectangleObject:
        box = self.get_resolved(key)
        return self.cropbox if box is None else RectangleObject(box)

    @property
    def bleedbox(self) -> RectangleObject:
        return self._box_or_cropbox("/BleedBox")

    @property
    def trimbox(self) -> RectangleObject:
        return self._box_or_cropbox("/TrimBox")

    @property
    def artbox(self) -> RectangleObject:
        return self._box_or_cropbox("/ArtBox")

    @property
    def rotation(self) -> int:
        return int(self.get_resolved("/Rotate", 0))

    @rotation.setter
    def rotation(self, angle: int) -> None:
        # pypdf rounds to the nearest multiple of 90.
        self[NameObject("/Rotate")] = NumberObject(((int(angle) + 45) // 90 * 90) % 360)

    @property
    def pdf(self) -> Any:
        reference = self.indirect_reference
        return self._pdf if reference is None else reference.pdf

    @property
    def writer(self) -> PdfWriter | None:
        pdf = self.pdf
        return pdf if isinstance(pdf, PdfWriter) else None

    @property
    def annotations(self) -> Any:
        return self.get_resolved("/Annots")

    @property
    def page_number(self) -> int:
        pages = self.pdf._page_list()
        return next(index for index, page in enumerate(pages) if page is self)

    def extract_text(self, *args: Any, **kwargs: Any) -> str:
        # Text extraction reads a file, so the page goes through the serialized document.
        from . import PdfReader

        writer = self.writer
        if writer is None:
            writer = PdfWriter()
            writer.add_page(self)
            number = 0
        else:
            number = self.page_number
        reader = PdfReader(BytesIO(writer._serialized()))
        return reader.pages[number].extract_text(*args, **kwargs)

    def get_contents(self) -> ContentStream | None:
        contents = self.get_resolved("/Contents")
        return None if is_null_or_none(contents) else ContentStream(contents, self.pdf)

    def compress_content_streams(self, level: int = -1) -> None:
        """Join this page's content streams into one Flate-encoded stream."""
        if self.writer is None:
            raise ValueError("Page must be part of a PdfWriter")
        content = self.get_contents()
        if content is not None:
            self.replace_contents(content.flate_encode(level))

    def _get_contents_as_bytes(self) -> bytes | None:
        contents = self.get_contents()
        return None if contents is None else contents.get_data()

    def replace_contents(self, content: Any) -> None:
        # Old streams become null objects, so replaced content does not stay in the output.
        # A stream another page still uses is left alone.
        writer = self.writer
        if writer is None:
            # Nothing to number it with yet. Adding the page to a writer numbers the streams.
            if is_null_or_none(content):
                self.pop(NameObject("/Contents"), None)
            else:
                self[NameObject("/Contents")] = content
            return
        current = self.get_resolved("/Contents")
        slot = self.get("/Contents")
        if isinstance(current, ArrayObject):
            replaced = [list.__getitem__(current, index) for index in range(len(current))]
        else:
            replaced = [slot]
        shared = writer._contents_of_other_pages(self)
        for reference in replaced:
            if isinstance(reference, IndirectObject) and reference.idnum not in shared:
                writer._replace_object(reference, NullObject())
        if is_null_or_none(content):
            if isinstance(slot, IndirectObject) and slot.idnum not in shared:
                writer._replace_object(slot, NullObject())
            self.pop(NameObject("/Contents"), None)
            return
        if isinstance(content, ArrayObject):
            content = ArrayObject(writer._add_object(item) for item in content)
        attached = getattr(content, "indirect_reference", None)
        if isinstance(slot, IndirectObject) and slot.idnum not in shared and attached is None:
            # The page keeps its object number, as in pypdf.
            writer._replace_object(slot, content)
        else:
            self[NameObject("/Contents")] = writer._add_object(content)

    def merge_page(self, page2: Any, expand: bool = False, over: bool = True) -> None:
        merge_page(self, page2, over=over, expand=expand)

    def merge_transformed_page(
        self, page2: Any, ctm: Any, over: bool = True, expand: bool = False
    ) -> None:
        matrix = as_matrix(ctm)
        merge_page(self, page2, lambda stream: add_cm(stream, matrix), matrix, over, expand)

    def merge_scaled_page(
        self, page2: Any, scale: float, over: bool = True, expand: bool = False
    ) -> None:
        self.merge_transformed_page(page2, Transformation().scale(scale, scale), over, expand)

    def merge_rotated_page(
        self, page2: Any, rotation: float, over: bool = True, expand: bool = False
    ) -> None:
        self.merge_transformed_page(page2, Transformation().rotate(rotation), over, expand)

    def merge_translated_page(
        self, page2: Any, tx: float, ty: float, over: bool = True, expand: bool = False
    ) -> None:
        self.merge_transformed_page(page2, Transformation().translate(tx, ty), over, expand)

    def add_transformation(self, ctm: Any, expand: bool = False) -> None:
        transform_page(self, as_matrix(ctm), expand)

    def add_action(self, trigger: Any, action: Any) -> None:
        Action._create_new(self, trigger, action)

    def delete_action(self, trigger: Any) -> None:
        Action._delete(self, trigger)

    def transfer_rotation_to_content(self) -> None:
        transfer_rotation(self)

    def rotate(self, angle: int) -> PageObject:
        if angle % 90:
            raise ValueError("Rotation angle must be a multiple of 90")
        self[NameObject("/Rotate")] = NumberObject((self.rotation + angle) % 360)
        return self


class PageObjects:
    def __init__(self, writer: PdfWriter) -> None:
        self.writer = writer

    def __len__(self) -> int:
        return len(self.writer._page_list())

    def __iter__(self) -> Iterator[PageObject]:
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
        del full
        self.strict = strict
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
        self._copies: dict[int, dict[int, Any]] = {}
        self._reserved_pages: set[int] = set()
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

    def _contents_of_other_pages(self, page: PageObject) -> set[int]:
        """Stream numbers other pages use. Pages can share a content stream."""
        numbers: set[int] = set()
        for other in self._page_list():
            # A deep copy of a page keeps the page's reference, and counts as the same page.
            same = other.indirect_reference == page.indirect_reference
            contents = None if same else other.get("/Contents")
            if isinstance(contents, IndirectObject):
                numbers.add(contents.idnum)
                contents = contents.get_object()
            if isinstance(contents, ArrayObject):
                for index in range(len(contents)):
                    item = list.__getitem__(contents, index)
                    if isinstance(item, IndirectObject):
                        numbers.add(item.idnum)
        return numbers

    def _replace_object(self, indirect: Any, obj: Any) -> Any:
        self._objects[indirect.idnum - 1] = obj
        obj.indirect_reference = indirect
        return obj

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
    def pages(self) -> PageObjects:  # ty: ignore[invalid-property-type-override]
        return PageObjects(self)

    def _page_list(self) -> list[PageObject]:
        kids = self._pages_object.get_resolved("/Kids")
        return [list.__getitem__(kids, index).get_object() for index in range(len(kids))]

    def _insert(self, page: PageObject, index: int) -> PageObject:
        reference = self._add_object(page)
        page[NameObject("/Parent")] = self._pages_object.indirect_reference
        kids = self._pages_object.get_resolved("/Kids")
        kids.insert(index, reference)
        self._pages_object[NameObject("/Count")] = NumberObject(len(kids))
        return page

    def add_page(self, page: Any, excluded_keys: Iterable[str] = ()) -> PageObject:
        return self.insert_page(page, len(self.pages), excluded_keys)

    def insert_page(
        self, page: Any, index: int = 0, excluded_keys: Iterable[str] = ()
    ) -> PageObject:
        return self._insert(self._import_page(page, set(excluded_keys)), index)

    def add_blank_page(self, width: float | None = None, height: float | None = None) -> PageObject:
        return self.insert_blank_page(width, height, len(self.pages))

    def insert_blank_page(
        self, width: float | None = None, height: float | None = None, index: int = 0
    ) -> PageObject:
        count = len(self.pages)
        if not -count <= index <= count:
            raise IndexError(f"Index should be in range [-{count}, {count}]")
        # As in pypdf, a missing or negative size comes from the neighbouring page.
        width_unset = width is None or width < 0
        height_unset = height is None or height < 0
        if width_unset or height_unset:
            if not len(self.pages):
                raise PageSizeNotDefinedError
            box = self.pages[min(index, len(self.pages) - 1)].mediabox
            width = box.width if width_unset else width
            height = box.height if height_unset else height
        page = PageObject()
        page[NameObject("/Type")] = NameObject("/Page")
        page[NameObject("/MediaBox")] = RectangleObject((0, 0, width, height))
        page[NameObject("/Resources")] = DictionaryObject()
        return self._insert(page, index)

    def remove_page(self, page: Any) -> None:
        kids = self._pages_object.get_resolved("/Kids")
        index = page if isinstance(page, int) else self._page_list().index(page)
        del kids[index]
        self._pages_object[NameObject("/Count")] = NumberObject(len(kids))

    def _import_page(self, page: Any, excluded: set[str]) -> PageObject:
        source = page if isinstance(page, DictionaryObject) else page._generic_dictionary()
        reference = source.indirect_reference
        owner = getattr(reference, "pdf", None)
        copies = self._copies.setdefault(id(owner), {})
        # A link imported earlier may already have reserved this page's number.
        reserved = None if reference is None else copies.get(reference.idnum)
        output = PageObject() if reserved is None else reserved.get_object()
        if reference is not None:
            copies[reference.idnum] = self._add_object(output)
            self._reserved_pages.discard(reference.idnum)
        for key, value in dict.items(source):
            if key in {"/Parent", *excluded}:
                continue
            dict.__setitem__(output, key, self._import(value, owner, copies))
        for key in ("/Resources", "/MediaBox", "/CropBox", "/Rotate"):
            if key not in output and key not in excluded and hasattr(source, "get_inherited"):
                inherited = source.get_inherited(key)
                if inherited is not None:
                    imported = self._import(inherited, owner, copies)
                    dict.__setitem__(output, NameObject(key), imported)
        return output

    def _import(self, value: Any, owner: Any, copies: dict[int, Any]) -> Any:
        if isinstance(value, IndirectObject):
            if value.pdf is self:
                return value
            if value.pdf is not owner:
                # A page built outside a writer can hold references from several documents.
                owner = value.pdf
                copies = self._copies.setdefault(id(owner), {})
            if value.idnum in copies:
                return copies[value.idnum]
            target = value.get_object()
            if isinstance(target, DictionaryObject) and target.get_resolved("/Type") == "/Page":
                # Hold the number. If this page is never imported, the link goes to null.
                reference = self._add_object(PageObject())
                copies[value.idnum] = reference
                self._reserved_pages.add(value.idnum)
                return reference
            copy = self._empty_copy(target)
            reference = self._add_object(copy)
            copies[value.idnum] = reference
            self._fill_copy(copy, target, owner, copies)
            return reference
        copy = self._empty_copy(value)
        self._fill_copy(copy, value, owner, copies)
        # A page built outside a writer can hold a stream directly, which PDF does not allow.
        return self._add_object(copy) if isinstance(copy, StreamObject) else copy

    @staticmethod
    def _empty_copy(value: Any) -> Any:
        if isinstance(value, StreamObject):
            copy = ContentStream(None, None) if isinstance(value, ContentStream) else type(value)()
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
        selected = list(indexes)
        for index in selected:
            self.add_page(reader.pages[index])
        # Only a whole document keeps its outline: a partial append must not carry titles
        # from pages it left out.
        if len(set(selected)) == len(reader.pages):
            self._import_outline(reader)
        self._drop_unimported_pages(reader)

    def _import_outline(self, reader: Any) -> None:
        root = reader.root_object
        outline = root.raw_get("/Outlines") if "/Outlines" in root else None
        if outline is None or "/Outlines" in self._root_object:
            return
        copies = self._copies.setdefault(id(reader), {})
        imported = self._import(outline, reader, copies)
        self._root_object[NameObject("/Outlines")] = imported

    def _drop_unimported_pages(self, reader: Any) -> None:
        copies = self._copies.setdefault(id(reader), {})
        for number in self._reserved_pages:
            reference = copies.get(number)
            if reference is not None:
                self._objects[reference.idnum - 1] = NullObject()
        self._reserved_pages.clear()

    def add_attachment(self, filename: str, data: str | bytes) -> EmbeddedFile:
        return EmbeddedFile._create_new(self, filename, data)

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


__all__ = ("PdfWriter", "PageObject")
