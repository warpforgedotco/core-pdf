"""pypdf's PdfWriter, over a table of generic objects and core-pdf's serializer."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from core_pdf.impl.document_editing import BINARY_MARKER, serialize_body
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_syntax.serialize import ObjectLayout

from ._doc_common import PdfDocCommon
from ._generic_trees import EmbeddedFile
from ._images import (
    ImageFile,
    VirtualListImages,
    content_stream_images,
    image_ids,
    page_image,
)
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
from ._text import ALL_ORIENTATIONS, TextOptions, extract_layout_text, extract_legacy_text
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

logger = logging.getLogger(__name__)


def _positional_text_arguments(
    args: tuple[Any, ...], orientations: int | tuple[int, ...], space_width: float
) -> tuple[tuple[int, ...], float]:
    # pypdf once took a leading string argument, and still accepts the old order.
    offset = None
    if args:
        if isinstance(args[0], str):
            offset = 2
        elif isinstance(args[0], (tuple, int)):
            offset = 0
        else:
            raise TypeError(f"Invalid positional parameter {args[0]}")
    if offset is not None:
        if len(args) > offset:
            if not isinstance(args[offset], (tuple, int)):
                raise TypeError(f"Invalid positional parameter {args[offset]}")
            orientations = args[offset]
        if len(args) > offset + 1:
            if not isinstance(args[offset + 1], (float, int)):
                raise TypeError(f"Invalid positional parameter {args[offset + 1]}")
            space_width = args[offset + 1]
    return ((orientations,) if isinstance(orientations, int) else orientations), space_width


def _page_box(name: str, *fallbacks: str) -> property:
    # A box read from the page stays in it, so changing the rectangle changes the page.
    def getter(page: PageObject) -> RectangleObject:
        box = page.get_resolved(name)
        if isinstance(box, RectangleObject):
            return box
        for key in (name, *fallbacks):
            box = page.get_resolved(key)
            if not is_null_or_none(box):
                break
        length = len(box)
        if length > 4:
            # pypdf once wrote boxes this way, so files with them exist.
            logger.warning("Expected four values, got %d: %s", length, box)
        elif length < 4:
            raise ValueError(f"Expected four values for {name}, got {length}: {box}")
        rectangle = RectangleObject(tuple(box[:4]))
        page[NameObject(name)] = rectangle
        return rectangle

    def setter(page: PageObject, value: Any) -> None:
        page[NameObject(name)] = value

    def deleter(page: PageObject) -> None:
        del page[name]

    return property(getter, setter, deleter)


class PageObject(DictionaryObject):
    """A page as a dictionary, like pypdf's. It may be held by a writer or stand alone."""

    def __init__(self, pdf: Any = None, indirect_reference: Any = None) -> None:
        super().__init__()
        self._pdf = pdf
        self.indirect_reference = indirect_reference
        self._source_content: tuple[Any, ...] | None = None
        self._content_stream_images: dict[str, ImageFile | None] | None = None

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

    mediabox = _page_box("/MediaBox")
    cropbox = _page_box("/CropBox", "/MediaBox")
    bleedbox = _page_box("/BleedBox", "/CropBox", "/MediaBox")
    trimbox = _page_box("/TrimBox", "/CropBox", "/MediaBox")
    artbox = _page_box("/ArtBox", "/CropBox", "/MediaBox")

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

    @annotations.setter
    def annotations(self, value: Any) -> None:
        if value is not None:
            self[NameObject("/Annots")] = value
        elif "/Annots" in self:
            del self["/Annots"]

    @property
    def page_number(self) -> int | None:
        pages = self.pdf._page_list()
        return next((index for index, page in enumerate(pages) if page is self), None)

    @property
    def user_unit(self) -> float:
        return self.get_resolved("/UserUnit", 1)

    def _parse_images_from_content_stream(self) -> dict[str, ImageFile | None]:
        return content_stream_images(self)

    def _get_ids_image(
        self, obj: Any = None, ancest: list[str] | None = None, call_stack: list[Any] | None = None
    ) -> list[Any]:
        return image_ids(self, obj, ancest, call_stack)

    def _get_image(self, id: Any, obj: Any = None) -> ImageFile:  # noqa: A002
        return page_image(self, id, obj)

    @property
    def images(self) -> VirtualListImages:
        return VirtualListImages(self._get_ids_image, self._get_image)

    @property
    def inline_images(self) -> dict[str, ImageFile] | None:
        warnings.warn(
            "PageObject.inline_images is deprecated and will be removed in pypdf 7.0.0. "
            "Use PageObject.images instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._content_stream_images is None:
            return None
        return {
            name: image
            for name, image in self._content_stream_images.items()
            if image is not None and image.is_inline
        }

    def _content_keys(self) -> tuple[Any, ...]:
        return tuple(self.get(key) for key in ("/Contents", "/Resources", "/Rotate"))

    def extract_text(
        self,
        *args: Any,
        orientations: int | tuple[int, ...] = ALL_ORIENTATIONS,
        space_width: float = 200.0,
        visitor_operand_before: Any = None,
        visitor_operand_after: Any = None,
        visitor_text: Any = None,
        extraction_mode: str = "plain",
        **kwargs: Any,
    ) -> str:
        if extraction_mode not in {"plain", "layout"}:
            raise ValueError(f"Invalid text extraction mode '{extraction_mode}'")
        orientations, _ = _positional_text_arguments(args, orientations, space_width)
        if extraction_mode == "layout":
            visitors = {
                "visitor_operand_before": visitor_operand_before,
                "visitor_operand_after": visitor_operand_after,
                "visitor_text": visitor_text,
            }
            for name, visitor in visitors.items():
                if visitor:
                    logger.warning("Argument %s is ignored in layout mode", name)
        resources = self.get_resolved("/Resources")
        fonts = None if is_null_or_none(resources) else resources.get_resolved("/Font")
        if is_null_or_none(self.get_resolved("/Contents")) and not fonts:
            return ""  # Without contents, only reading a font can still fail.
        # A changed page has no copy in a file, so it goes through a serialized document.
        from . import PdfReader

        pdf = self.pdf
        source = pdf._core_page(self) if isinstance(pdf, PdfReader) else None
        if source is None:
            writer = self.writer
            if writer is None:
                writer = PdfWriter()
                writer.add_page(self)
                number = 0
            else:
                number = cast(int, self.page_number)
            pdf = PdfReader(BytesIO(writer._serialized()))
            page = pdf.pages[number]
            source = pdf._core_page(page)
        if extraction_mode == "layout":
            return extract_layout_text(
                source,
                pdf.text_caches,
                space_vertically=kwargs.get("layout_mode_space_vertically", True),
                scale_weight=kwargs.get("layout_mode_scale_weight", 1.25),
                strip_rotated=kwargs.get("layout_mode_strip_rotated", True),
                font_height_weight=kwargs.get("layout_mode_font_height_weight", 1),
            )
        options = TextOptions(
            orientations,
            visitor_operand_before,
            visitor_operand_after,
            visitor_text,
            pdf,
        )
        return extract_legacy_text(source, pdf.text_caches, options)

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
            if self.pdf is not None:
                warnings.warn(
                    "Calling `PageObject.replace_contents()` for pages not assigned to a writer "
                    "is deprecated and will be removed in pypdf 7.0.0. Attach the page to the "
                    "writer first or use `PdfWriter(clone_from=...)` directly. The existing "
                    "approach has proved being unreliable.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # Nothing to number it with yet. Adding the page to a writer numbers the streams.
            if is_null_or_none(content):
                self.pop(NameObject("/Contents"), None)
            else:
                self[NameObject("/Contents")] = content
            return
        current = self.get_resolved("/Contents")
        slot = self.get("/Contents")
        if isinstance(current, ArrayObject):
            replaced = list(current)
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
        # For each source document, by id, its object numbers' numbers here. The document
        # is kept too, because an id can be reused once its object is freed.
        self._id_translated: dict[int, dict[Any, Any]] = {}
        self._reserved_pages: set[int] = set()
        self.incremental = False
        self._reader = None
        # Objects are numbered as in pypdf: the information dictionary, pages, then catalog.
        self._info = DictionaryObject()
        self._info[NameObject("/Producer")] = TextStringObject("core-pdf")
        self._add_object(self._info)
        self._pages_object = self._new_pages_tree()
        catalog = DictionaryObject()
        catalog[NameObject("/Type")] = NameObject("/Catalog")
        catalog[NameObject("/Pages")] = self._pages_object.indirect_reference
        self._root_object = catalog
        self._add_object(catalog)
        if source is not None:
            self.append(source)
            self._clone_xmp_metadata(source)

    def _clone_xmp_metadata(self, source: Any) -> None:
        from . import PdfReader

        reader = source if isinstance(source, PdfReader) else self._reader_for(source)
        catalog = reader.root_object
        if "/Metadata" in catalog:
            metadata = self._import(dict.__getitem__(catalog, "/Metadata"), reader)
            self._root_object[NameObject("/Metadata")] = metadata

    @property
    def xmp_metadata(self) -> Any:
        return self.root_object.xmp_metadata

    @xmp_metadata.setter
    def xmp_metadata(self, value: Any) -> None:
        catalog = self.root_object
        if value is None:
            catalog.pop("/Metadata", None)
            return
        current = catalog.get("/Metadata")
        if isinstance(current, IndirectObject):
            stream = current.get_object()
        else:
            stream = StreamObject()
            stream[NameObject("/Type")] = NameObject("/Metadata")
            stream[NameObject("/Subtype")] = NameObject("/XML")
            catalog[NameObject("/Metadata")] = self._add_object(stream)
        stream.set_data(value if isinstance(value, bytes) else value.stream.get_data())

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

    def _translation(self, source: Any) -> dict[Any, Any]:
        return self._id_translated.setdefault(id(source), {"PreventGC": source})

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
                for item in contents:
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
    def pages(self) -> PageObjects:
        return PageObjects(self)

    def _page_list(self) -> list[PageObject]:
        kids = self._pages_object.get_resolved("/Kids")
        return [kid.get_object() for kid in kids]

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
        if not isinstance(page, PageObject):
            raise ValueError("Invalid page object")
        source = page
        reference = source.indirect_reference
        owner = getattr(reference, "pdf", None)
        translated = self._translation(owner)
        # A link imported earlier may already have reserved this page's number. Adding a page
        # again makes a separate copy.
        reserved = None
        if reference is not None and reference.idnum in self._reserved_pages:
            reserved = translated.get(reference.idnum)
        output = PageObject() if reserved is None else self.get_object(reserved)
        if reference is not None:
            translated[reference.idnum] = self._add_object(output).idnum
            self._reserved_pages.discard(reference.idnum)
        for key, value in dict.items(source):
            if key in {"/Parent", *excluded}:
                continue
            dict.__setitem__(output, key, self._import(value, owner))
        for key in ("/Resources", "/MediaBox", "/CropBox", "/Rotate"):
            if key not in output and key not in excluded and hasattr(source, "get_inherited"):
                inherited = source.get_inherited(key)
                if inherited is not None:
                    dict.__setitem__(output, NameObject(key), self._import(inherited, owner))
        return output

    def _import(self, value: Any, owner: Any) -> Any:
        if isinstance(value, IndirectObject):
            if value.pdf is self:
                return value
            # A page built outside a writer can hold references from several documents.
            owner = value.pdf
            translated = self._translation(owner)
            if value.idnum in translated:
                return IndirectObject(translated[value.idnum], 0, self)
            target = value.get_object()
            if isinstance(target, DictionaryObject) and target.get_resolved("/Type") == "/Page":
                # Hold the number. If this page is never imported, the link goes to null.
                reference = self._add_object(PageObject())
                translated[value.idnum] = reference.idnum
                self._reserved_pages.add(value.idnum)
                return reference
            copy = self._empty_copy(target)
            reference = self._add_object(copy)
            translated[value.idnum] = reference.idnum
            self._fill_copy(copy, target, owner)
            return reference
        copy = self._empty_copy(value)
        self._fill_copy(copy, value, owner)
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

    def _fill_copy(self, copy: Any, value: Any, owner: Any) -> None:
        if isinstance(value, DictionaryObject):
            for key, item in dict.items(value):
                dict.__setitem__(copy, key, self._import(item, owner))
        elif isinstance(value, ArrayObject):
            copy.extend(self._import(item, owner) for item in value)

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
        self._root_object[NameObject("/Outlines")] = self._import(outline, reader)

    def _drop_unimported_pages(self, reader: Any) -> None:
        translated = self._translation(reader)
        for number in self._reserved_pages:
            if number in translated:
                self._objects[translated[number] - 1] = NullObject()
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
        return header + serialize_body(len(header), objects, trailer, ObjectLayout.SPACED)

    def write_bytes(self) -> bytes:
        return self._serialized()


__all__ = ("PdfWriter", "PageObject")
