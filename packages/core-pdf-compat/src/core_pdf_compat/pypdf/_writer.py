"""pypdf's PdfWriter, over a table of generic objects and core-pdf's serializer."""

from __future__ import annotations

import hashlib
import logging
import secrets
import sys
import uuid
import warnings
from collections.abc import Iterable
from copy import copy
from io import BytesIO, IOBase
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from core_pdf.impl.document_editing import (
    BINARY_MARKER,
    TRAILER_KEYS,
    EditableDocument,
    SavedPdf,
    references_in,
    serialize_body,
)
from core_pdf.impl.types import PdfReference, PdfString
from core_pdf_spec.s_07_syntax.serialize import ObjectLayout, serialize_object

from ._doc_common import DocumentInformation, PdfDocCommon, _VirtualList, highest_header
from ._encryption import EncryptAlgorithm, Encryption
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
from ._text import (
    ALL_ORIENTATIONS,
    TextOptions,
    extract_form_text,
    extract_layout_text,
    extract_legacy_text,
)
from ._writer_merge import MergeWriter
from ._writer_outline import OutlineWriter
from .actions import Action
from .constants import UserAccessPermissions
from .errors import (
    DeprecationError,
    LimitReachedError,
    PageSizeNotDefinedError,
    PdfReadError,
    PyPdfError,
)
from .generic import (
    PAGE_FIT,
    ArrayObject,
    ByteStringObject,
    ContentStream,
    Destination,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    PdfObject,
    RectangleObject,
    StreamObject,
    TextStringObject,
    ViewerPreferences,
    create_string_object,
    is_null_or_none,
    to_core,
)

logger = logging.getLogger(__name__)

LAYOUTS = (
    "/NoLayout",
    "/SinglePage",
    "/OneColumn",
    "/TwoColumnLeft",
    "/TwoColumnRight",
    "/TwoPageLeft",
    "/TwoPageRight",
)
MODES = (
    "/UseNone",
    "/UseOutlines",
    "/UseThumbs",
    "/FullScreen",
    "/UseOC",
    "/UseAttachments",
)


ALL_DOCUMENT_PERMISSIONS = UserAccessPermissions.all()


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
        document = self.pdf
        if document is None:
            return None
        pages = document._page_list()
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
        pdf, source = self._text_source()
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

    def _text_source(self) -> tuple[Any, Any]:
        """The reader and core-pdf page to extract from."""
        # A changed page has no copy in a file, so it goes through a serialized document.
        from . import PdfReader

        pdf = self.pdf
        source = pdf._core_page(self) if isinstance(pdf, PdfReader) else None
        if source is not None:
            return pdf, source
        writer = self.writer
        if writer is None:
            writer = PdfWriter()
            writer.add_page(self)
            number = 0
        else:
            number = cast(int, self.page_number)
        reader = PdfReader(BytesIO(writer._serialized()))
        return reader, reader._core_page(reader.pages[number])

    def extract_xform_text(
        self,
        xform: Any,
        orientations: tuple[int, ...] = ALL_ORIENTATIONS,
        space_width: float = 200.0,
        visitor_operand_before: Any = None,
        visitor_operand_after: Any = None,
        visitor_text: Any = None,
    ) -> str:
        del space_width
        pdf, source = self._text_source()
        options = TextOptions(
            orientations, visitor_operand_before, visitor_operand_after, visitor_text, pdf
        )
        return extract_form_text(source, to_core(xform), pdf.text_caches, options)

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


class SaveReport(NamedTuple):
    """How the last save was made. Every write sets ``PdfWriter.last_save``.

    A full rewrite holds only what the document reaches. An incremental update keeps the
    source's bytes, so ``superseded`` objects stay readable in their earlier version, and so
    do ``removed`` ones, which the document no longer reaches.
    """

    kind: Literal["full", "incremental"]
    objects_written: int
    superseded: tuple[int, ...]
    removed: tuple[int, ...]


def _digest(value: Any) -> bytes | None:
    if value is None:
        return None
    return hashlib.blake2b(serialize_object(to_core(value)), digest_size=16).digest()


class PdfWriter(OutlineWriter, MergeWriter, PdfDocCommon):
    def __init__(
        self,
        fileobj: Any = "",
        clone_from: Any = None,
        incremental: bool = False,
        full: bool = False,
        strict: bool = False,
        *,
        keep_initial_header: bool = False,
        incremental_clone_object_count_limit: int | None = 500_000,
        incremental_clone_object_id_limit: int | None = 1_000_000,
    ) -> None:
        self.strict = strict
        self.incremental = incremental or full
        self._objects: list[Any] = []
        # For each source document, by id, its object numbers' numbers here. The document
        # is kept too, because an id can be reused once its object is freed.
        self._id_translated: dict[int, dict[Any, Any]] = {}
        self._reserved_pages: set[int] = set()
        self._encryption: Encryption | None = None
        self._encrypt_entry: dict[Any, Any] | None = None
        self._file_id: bytes | None = None
        self._cloning = False
        self._idnum_hash: dict[bytes, Any] = {}
        # For an incremental update: a digest of each object as read, and what the file reached.
        self._original_hash: list[bytes | None] = []
        self._original_reachable: set[int] = set()
        self.last_save: SaveReport | None = None
        self._objects_written = 0
        self._info_obj: Any = None
        self._ID: Any = None
        self.flattened_pages: list[PageObject] = []
        self._count_limit = incremental_clone_object_count_limit or sys.maxsize
        self._id_limit = incremental_clone_object_id_limit or sys.maxsize
        self._header = b"%PDF-1.3"
        if self.incremental:
            fileobj = self._incremental_source(fileobj)
            self._reader = fileobj
            self._header = fileobj.pdf_header.encode()
        else:
            self._info_obj = self._add_object(
                DictionaryObject({NameObject("/Producer"): TextStringObject("pypdf")})
            )
        source = self._clone_source(fileobj, clone_from)
        self.temp_fileobj = fileobj
        self.fileobj: Any = fileobj if isinstance(fileobj, (str, Path)) else ""
        self._cloned = False
        if source is None:
            self._pages = self._add_object(
                DictionaryObject(
                    {
                        NameObject("/Type"): NameObject("/Pages"),
                        NameObject("/Count"): NumberObject(0),
                        NameObject("/Kids"): ArrayObject(),
                    }
                )
            )
            self._root_object = DictionaryObject(
                {NameObject("/Type"): NameObject("/Catalog"), NameObject("/Pages"): self._pages}
            )
            self._add_object(self._root_object)
        else:
            reader: Any = source if isinstance(source, PdfDocCommon) else self._reader_for(source)
            self.clone_document_from_reader(reader)
            self._cloned = True
            if keep_initial_header and not self.incremental:
                self._header = reader.pdf_header.encode()
        if full and not incremental:
            self.incremental = False

    @staticmethod
    def _incremental_source(fileobj: Any) -> Any:
        from . import PdfReader

        if isinstance(fileobj, (str, Path)):
            fileobj = BytesIO(Path(fileobj).read_bytes())
        if isinstance(fileobj, BytesIO):
            fileobj = PdfReader(fileobj)
        if not isinstance(fileobj, PdfReader):
            raise PyPdfError("Invalid type for incremental mode")
        return fileobj

    @staticmethod
    def _clone_source(fileobj: Any, clone_from: Any) -> Any:
        """The document to start from: pypdf also takes it in place of the output file."""
        if isinstance(fileobj, (str, Path, BytesIO)) and (fileobj == "" or clone_from is not None):
            return clone_from
        if isinstance(fileobj, (str, Path)):
            path = Path(fileobj)
            return fileobj if path.exists() and path.stat().st_size else clone_from
        if isinstance(fileobj, IOBase):
            position = fileobj.tell()
            empty = fileobj.seek(0, 2) == 0
            fileobj.seek(position)
            return clone_from if empty else fileobj
        return fileobj

    @property
    def _info(self) -> DictionaryObject | None:
        return None if self._info_obj is None else self._info_obj.get_object()

    @_info.setter
    def _info(self, value: Any) -> None:
        if value is None:
            if self._info_obj is not None:
                self._objects[self._info_obj.idnum - 1] = None
            self._info_obj = None
            return
        if self._info_obj is None:
            self._info_obj = self._add_object(DictionaryObject())
        info = self._info_obj.get_object()
        items = list(dict.items(value.get_object()))
        info.clear()
        for key, item in items:
            dict.__setitem__(info, key, item)

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

    @property
    def with_as_usage(self) -> bool:
        raise DeprecationError("with_as_usage is deprecated and was removed in pypdf 5.0.")

    @with_as_usage.setter
    def with_as_usage(self, value: bool) -> None:
        del value
        raise DeprecationError("with_as_usage is deprecated and was removed in pypdf 5.0.")

    def __enter__(self) -> PdfWriter:
        return self

    def __exit__(self, *_: object) -> None:
        # An existing file was the source. pypdf leaves it alone, so it is not overwritten.
        if self.temp_fileobj and not self._cloned:
            self.write(self.temp_fileobj)

    def close(self) -> None:
        return None

    def __deepcopy__(self, memo: Any) -> PdfWriter:
        # A deep copy of a page keeps its document, as a copied reference does.
        return self

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
        if isinstance(indirect, IndirectObject):
            if indirect.pdf is not self:
                raise ValueError("PDF must be self")
            number = indirect.idnum
        else:
            number = indirect
        current = getattr(obj, "indirect_reference", None)
        if current is not None and current.pdf is not self:
            imported: Any = self._import(obj, current.pdf)
            obj = imported.get_object() if isinstance(imported, IndirectObject) else imported
        generation = getattr(indirect, "generation", 0)
        self._objects[number - 1] = obj
        obj.indirect_reference = IndirectObject(number, generation, self)
        return obj

    def get_object(self, indirect_reference: Any) -> Any:
        if isinstance(indirect_reference, IndirectObject) and indirect_reference.pdf is not self:
            raise ValueError("PDF must be self")
        number = getattr(indirect_reference, "idnum", indirect_reference)
        found = self._objects[number - 1] if 0 < number <= len(self._objects) else None
        if found is None:
            raise PdfReadError(f"Object {indirect_reference!r} not found!")
        return found

    def get_reference(self, obj: Any) -> IndirectObject:
        reference = getattr(obj, "indirect_reference", None)
        if reference is not None and reference.pdf is self:
            return reference
        return IndirectObject(self._objects.index(obj) + 1, 0, self)

    @property
    def root_object(self) -> DictionaryObject:
        return self._root_object

    @property
    def pdf_header(self) -> str:
        return self._header.decode()

    @pdf_header.setter
    def pdf_header(self, value: str | bytes) -> None:
        self._header = value.encode() if isinstance(value, str) else value

    @property
    def metadata(self) -> DocumentInformation | None:
        info = self._info
        if info is None:
            return None
        output = DocumentInformation()
        for key, value in dict.items(info):
            dict.__setitem__(output, key, value)
        output.indirect_reference = self._info_obj
        return output

    @metadata.setter
    def metadata(self, value: Any) -> None:
        if value is None:
            self._info = None
            return
        if self._info is not None:
            self._info.clear()
        self.add_metadata(value)

    def add_metadata(self, infos: Any) -> None:
        if isinstance(infos, PdfObject):
            infos = infos.get_object()
        if self._info is None:
            self._info = DictionaryObject()
        info = cast(DictionaryObject, self._info)
        for key, value in list(infos.items()):
            value = value.get_object() if isinstance(value, PdfObject) else value
            info[NameObject(key)] = create_string_object(str(value))

    @property
    def pages(self) -> _VirtualList:  # ty: ignore[invalid-property-type-override]
        return _VirtualList(self.get_num_pages, self.get_page)

    def get_num_pages(self) -> int:
        return len(self.flattened_pages)

    def get_page(self, page_number: int) -> PageObject:
        return self.flattened_pages[page_number]

    def _page_list(self) -> list[PageObject]:
        return self.flattened_pages

    def _get_page_number_by_indirect(self, indirect_reference: Any) -> int | None:
        if is_null_or_none(indirect_reference):
            return None
        if isinstance(indirect_reference, int):
            indirect_reference = IndirectObject(indirect_reference, 0, self)
        target = indirect_reference.get_object()
        return target.page_number if isinstance(target, PageObject) else None

    def get_page_number(self, page: Any) -> int | None:
        return page.page_number if isinstance(page, PageObject) else None

    def _get_page_in_node(self, page_number: int) -> tuple[DictionaryObject, int]:
        """The tree node and /Kids position of a page, or the top node and -1 past the end."""
        top = self.root_object["/Pages"]

        def search(node: DictionaryObject, low: int) -> tuple[Any, int]:
            size = node.get_resolved("/Count", 1)
            if node["/Type"] == "/Page":
                return (node, -1) if page_number == low else (None, low + 1)
            if page_number - low >= size:
                return (top, -1) if node == top else (None, low + size)
            for index, kid in enumerate(node["/Kids"]):
                found, position = search(kid.get_object(), low)
                if found is not None:
                    return (node, index) if position < 0 else (found, position)
                low = position
            raise PyPdfError("Unexpectedly cannot find the node.")

        if page_number < 0:
            raise PyPdfError("Invalid page number")
        return search(top, 0)

    def _add_page(self, page: Any, index: int, excluded_keys: Iterable[Any] = ()) -> PageObject:
        if not isinstance(page, PageObject) or page.get("/Type") != "/Page":
            raise ValueError("Invalid page object")
        annotations = page.get_resolved("/Annots")
        if not (is_null_or_none(annotations) or isinstance(annotations, ArrayObject)):
            logger.warning(
                "Expected annotation arrays: %s %s. Ignoring annotations.",
                annotations,
                ArrayObject(),
            )
        # Numbers in the list mean a limit on the depth in pypdf: only the page's level is cut.
        excluded = {key for key in excluded_keys if isinstance(key, str)}
        output = self._import_page(page, excluded | {"/Parent", "/StructParents"})
        source = page.pdf
        if source is not None:
            self._header = highest_header(self.pdf_header, source.pdf_header).encode()
        node, position = self._get_page_in_node(index)
        output[NameObject("/Parent")] = node.indirect_reference
        reference = cast(IndirectObject, output.indirect_reference)
        if position >= 0:
            node["/Kids"].insert(position, reference)
            self.flattened_pages.insert(index, output)
        else:
            node["/Kids"].append(reference)
            self.flattened_pages.append(output)
        current: Any = node
        for _ in range(1000):
            if is_null_or_none(current):
                return output
            item = current.get_object()
            item[NameObject("/Count")] = NumberObject(item["/Count"] + 1)
            current = item.get("/Parent")
        raise PyPdfError("Too many recursive calls!")

    def add_page(self, page: Any, excluded_keys: Iterable[Any] = ()) -> PageObject:
        return self._add_page(page, len(self.flattened_pages), excluded_keys)

    def insert_page(
        self, page: Any, index: int = 0, excluded_keys: Iterable[Any] = ()
    ) -> PageObject:
        count = len(self.flattened_pages)
        if index < 0:
            index += count
        if index < 0:
            raise ValueError("Invalid index value")
        return self._add_page(page, min(index, count), excluded_keys)

    def add_blank_page(self, width: float | None = None, height: float | None = None) -> PageObject:
        return self.add_page(PageObject.create_blank_page(self, width, height))

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
        return self.insert_page(PageObject.create_blank_page(self, width, height), index)

    def remove_page(self, page: Any, clean: bool = False) -> None:
        if isinstance(page, IndirectObject):
            target = page.get_object()
            if not isinstance(target, PageObject):
                logger.warning("IndirectObject is not referencing a page")
                return
            page = target
        if not isinstance(page, int):
            try:
                page = self.flattened_pages.index(page)
            except ValueError:
                logger.warning("Cannot find page in pages")
                return
        if not 0 <= page < len(self.flattened_pages):
            logger.warning("Page number is out of range")
            return
        reference = self.flattened_pages[page].indirect_reference
        del self.pages[page]
        if clean and reference is not None:
            self._replace_object(reference, NullObject())

    def _import_page(self, page: Any, excluded: set[str]) -> PageObject:
        reference = page.indirect_reference
        owner = getattr(reference, "pdf", None)
        translated = self._translation(owner)
        # A link imported earlier may already have reserved this page's number. Any other
        # page gets a dictionary of its own, as a file may not list one object twice.
        known = None if reference is None else translated.get(reference.idnum)
        output: Any = self.get_object(known) if known in self._reserved_pages else PageObject()
        number = self._add_object(output).idnum
        self._reserved_pages.discard(number)
        if reference is not None:
            translated[reference.idnum] = number
        for key, value in dict.items(page):
            if key not in excluded:
                dict.__setitem__(output, key, self._import(value, owner))
        for key in ("/Resources", "/MediaBox", "/CropBox", "/Rotate"):
            if key not in output and key not in excluded:
                inherited = page.get_inherited(key)
                if inherited is not None:
                    dict.__setitem__(output, NameObject(key), self._import(inherited, owner))
        return output

    def _import(self, value: Any, owner: Any) -> Any:
        """Copy a value from another document, with what it refers to.

        References go on a list rather than through recursion: outline items chain through
        /Next, and a real outline can be longer than Python's stack.
        """
        pending: list[tuple[Any, Any, Any]] = []
        output = self._bring(value, owner, pending)
        while pending:
            duplicate, source, source_owner = pending.pop()
            self._fill_copy(duplicate, source, source_owner, pending)
        return output

    def _bring(self, value: Any, owner: Any, pending: list[Any]) -> Any:
        if isinstance(value, IndirectObject):
            if value.pdf is self:
                return value
            # A page built outside a writer can hold references from several documents.
            owner = value.pdf
            translated = self._translation(owner)
            if value.idnum in translated:
                return IndirectObject(translated[value.idnum], 0, self)
            target = value.get_object()
            if target is None:
                target = NullObject()
            is_page = (
                isinstance(target, DictionaryObject) and target.get_resolved("/Type") == "/Page"
            )
            if is_page and not self._cloning:
                # Hold the number. If this page is never imported, the link goes to null.
                reference = self._add_object(PageObject())
                translated[value.idnum] = reference.idnum
                self._reserved_pages.add(reference.idnum)
                return reference
            duplicate = self._empty_copy(target)
            if duplicate is target:
                # A plain value is shared with the source, which must keep its own reference.
                duplicate = copy(target)
                duplicate.indirect_reference = None
            reference = self._add_object(duplicate)
            translated[value.idnum] = reference.idnum
            pending.append((duplicate, target, owner))
            return reference
        duplicate = self._empty_copy(value)
        pending.append((duplicate, value, owner))
        # A page built outside a writer can hold a stream directly, which PDF does not allow.
        return self._add_object(duplicate) if isinstance(duplicate, StreamObject) else duplicate

    @staticmethod
    def _empty_copy(value: Any) -> Any:
        if isinstance(value, StreamObject):
            duplicate = (
                ContentStream(None, None) if isinstance(value, ContentStream) else type(value)()
            )
            # The stored bytes stay encoded, so they keep matching the copied /Filter.
            duplicate._data = value._stored_data()
            return duplicate
        if isinstance(value, DictionaryObject):
            return DictionaryObject()
        if isinstance(value, ArrayObject):
            return ArrayObject()
        return value

    def _fill_copy(self, duplicate: Any, value: Any, owner: Any, pending: list[Any]) -> None:
        if isinstance(value, DictionaryObject):
            for key, item in dict.items(value):
                dict.__setitem__(duplicate, key, self._bring(item, owner, pending))
        elif isinstance(value, ArrayObject):
            duplicate.extend(self._bring(item, owner, pending) for item in value)

    def clone_reader_document_root(self, reader: Any) -> None:
        self._info_obj = None
        if self.incremental:
            self._clone_in_place(reader)
        else:
            self._objects.clear()
            self._id_translated.clear()
            self._cloning = True
            root = self._import(reader.trailer.raw_get("/Root"), reader)
            self._cloning = False
            self._root_object = root.get_object()
            self._add_object(self._root_object)
            self._pages = self._add_object(self._root_object["/Pages"])
            self._root_object[NameObject("/Pages")] = self._pages
        self._check_trailer_size(reader)
        # Before flattening, which moves inherited entries into the pages as pypdf does.
        if self.incremental:
            self._original_hash = [_digest(item) for item in self._objects]
            self._original_reachable = {ref.object_number for ref in self._reachable()}
        self._flatten()
        for page in self.flattened_pages:
            self._replace_object(page.indirect_reference, page)
            if not self.incremental:
                page[NameObject("/Parent")] = self._pages
        if not self.incremental:
            pages = self._pages.get_object()
            references = [page.indirect_reference for page in self.flattened_pages]
            pages[NameObject("/Kids")] = ArrayObject(references)
            pages[NameObject("/Count")] = NumberObject(len(references))

    def _collect_incremental_clone_object_ids(self, reader: Any) -> list[int]:
        found = {number for entry in reader.xref.values() for number in entry}
        numbers = sorted((found | set(reader.xref_objStm)) - {0})
        if len(numbers) > self._count_limit:
            raise LimitReachedError(
                f"Incremental clone object count {len(numbers)} exceeds "
                f"maximum allowed count {self._count_limit}."
            )
        if numbers and numbers[-1] > self._id_limit:
            raise LimitReachedError(
                f"Incremental clone object ID {numbers[-1]} exceeds "
                f"maximum allowed ID {self._id_limit}."
            )
        return numbers

    def _clone_in_place(self, reader: Any) -> None:
        """Copy every object under its own number, as an update has to refer to them."""
        numbers = self._collect_incremental_clone_object_ids(reader)
        generations = {
            number: generation for generation, entry in reader.xref.items() for number in entry
        }
        self._objects = [None] * (numbers[-1] if numbers else 0)
        for number in numbers:
            generation = generations.get(number, 0)
            target = reader.get_object(IndirectObject(number, generation, reader))
            if target is None or isinstance(target, NullObject):
                continue
            duplicate = self._same_numbers(target)
            duplicate.indirect_reference = IndirectObject(number, generation, self)
            self._objects[number - 1] = duplicate
        reference = reader.trailer.raw_get("/Root")
        self._root_object = self._objects[reference.idnum - 1]
        self._pages = self._same_numbers(self._root_object.raw_get("/Pages"))

    def _same_numbers(self, value: Any) -> Any:
        if isinstance(value, IndirectObject):
            return IndirectObject(value.idnum, value.generation, self)
        duplicate = self._empty_copy(value)
        if isinstance(value, DictionaryObject):
            for key, item in dict.items(value):
                dict.__setitem__(duplicate, key, self._same_numbers(item))
        elif isinstance(value, ArrayObject):
            duplicate.extend(self._same_numbers(item) for item in list.__iter__(value))
        elif duplicate is value:
            duplicate = copy(value)
            duplicate.indirect_reference = None
        return duplicate

    def _check_trailer_size(self, reader: Any) -> None:
        size = reader.trailer.get_resolved("/Size")
        count = len(self._objects)
        if isinstance(size, int) and count > size:
            message = f"Object count {count} exceeds defined trailer size {size}"
            if self.strict:
                raise PdfReadError(message)
            logger.warning(message)

    def _flatten(self) -> None:
        """List the pages, moving what they inherit from the tree into each of them."""
        self.flattened_pages = []
        pages = self.root_object["/Pages"]
        if not isinstance(pages, DictionaryObject):
            raise PdfReadError("Invalid object in /Pages")
        # Node, reference, inherited values, in document order.
        stack: list[tuple[Any, Any, dict[str, Any]]] = [(pages, self._pages, {})]
        visited: set[int] = set()
        while stack:
            node, reference, inherit = stack.pop()
            kind = node.get_resolved("/Type")
            if kind is None:
                kind = "/Page" if "/Kids" not in node else "/Pages"
            if kind == "/Page":
                page = PageObject(self, reference)
                for key, item in dict.items(node):
                    dict.__setitem__(page, key, item)
                for key, item in inherit.items():
                    if key not in page:
                        dict.__setitem__(page, key, item)
                self.flattened_pages.append(page)
                continue
            if kind != "/Pages":
                continue
            inherited = dict(inherit)
            for key in ("/Resources", "/MediaBox", "/CropBox", "/Rotate"):
                if key in node:
                    inherited[NameObject(key)] = node.raw_get(key)
            kids = node.get_resolved("/Kids", ArrayObject())
            if isinstance(kids, NullObject):
                continue
            if not isinstance(kids, ArrayObject):
                raise PdfReadError(f"Expected /Kids to be an array, got {type(kids).__name__}.")
            for index in range(len(kids) - 1, -1, -1):
                kid = list.__getitem__(kids, index)
                item = kid.get_object()
                if not item:
                    continue
                if id(item) in visited:
                    raise PdfReadError("Detected cyclic page references.")
                visited.add(id(item))
                stack.append((item, kid if isinstance(kid, IndirectObject) else None, inherited))

    def clone_document_from_reader(self, reader: Any, after_page_append: Any = None) -> None:
        self.clone_reader_document_root(reader)
        info = reader.trailer.get("/Info") if not isinstance(reader, PdfWriter) else None
        if not is_null_or_none(info):
            target = cast(Any, info).get_object()
            if isinstance(info, IndirectObject) and self.incremental:
                self._info_obj = IndirectObject(info.idnum, info.generation, self)
            else:
                plain = DictionaryObject()
                for key, value in dict.items(target):
                    dict.__setitem__(plain, key, self._import(value, reader))
                self._info_obj = self._add_object(plain)
        identifier = reader.trailer.get_resolved("/ID")
        if isinstance(identifier, ArrayObject):
            self._ID = ArrayObject(
                ByteStringObject(item.get_original_bytes())
                if isinstance(item, TextStringObject)
                else item
                for item in identifier
            )
        if callable(after_page_append):
            for page in self.flattened_pages:
                after_page_append(page)

    def append_pages_from_reader(self, reader: Any, after_page_append: Any = None) -> None:
        for index in range(len(reader.pages)):
            page = self.add_page(reader.pages[index])
            if callable(after_page_append):
                after_page_append(page)

    def get_threads_root(self) -> ArrayObject:
        if "/Threads" not in self._root_object:
            self._root_object[NameObject("/Threads")] = ArrayObject()
        return self._root_object["/Threads"]

    @property
    def threads(self) -> ArrayObject:  # ty: ignore[invalid-property-type-override]
        return self.get_threads_root()

    @property
    def open_destination(self) -> Any:
        return self._read_open_destination()

    @open_destination.setter
    def open_destination(self, dest: Any) -> None:
        name = NameObject("/OpenAction")
        if dest is None:
            self._root_object.pop(name, None)
        elif isinstance(dest, str):
            self._root_object[name] = TextStringObject(dest)
        elif isinstance(dest, Destination):
            self._root_object[name] = dest.dest_array
        elif isinstance(dest, PageObject):
            page = NullObject() if dest.indirect_reference is None else dest.indirect_reference
            self._root_object[name] = Destination("Opening", page, PAGE_FIT).dest_array

    def _set_page_layout(self, layout: Any) -> None:
        if not isinstance(layout, NameObject):
            if layout not in LAYOUTS:
                logger.warning("Layout should be one of: %s", ", ".join(LAYOUTS))
            layout = NameObject(layout)
        self._root_object[NameObject("/PageLayout")] = layout

    def set_page_layout(self, layout: Any) -> None:
        self._set_page_layout(layout)

    def _get_page_layout(self) -> Any:
        return self._root_object.get_resolved("/PageLayout")

    @property
    def page_layout(self) -> Any:
        return self._get_page_layout()

    @page_layout.setter
    def page_layout(self, layout: Any) -> None:
        self._set_page_layout(layout)

    def _get_page_mode(self) -> Any:
        return self._root_object.get_resolved("/PageMode")

    @property
    def page_mode(self) -> Any:
        return self._get_page_mode()

    @page_mode.setter
    def page_mode(self, mode: Any) -> None:
        if not isinstance(mode, NameObject):
            if mode not in MODES:
                logger.warning("Mode should be one of: %s", ", ".join(MODES))
            mode = NameObject(mode)
        self._root_object[NameObject("/PageMode")] = mode

    def create_viewer_preferences(self) -> ViewerPreferences:
        preferences = ViewerPreferences()
        self._root_object[NameObject("/ViewerPreferences")] = self._add_object(preferences)
        return preferences

    def add_js(self, javascript: str) -> None:
        catalog = self._root_object
        if "/Names" not in catalog:
            catalog[NameObject("/Names")] = DictionaryObject()
        names = catalog["/Names"]
        if "/JavaScript" not in names:
            names[NameObject("/JavaScript")] = DictionaryObject(
                {NameObject("/Names"): ArrayObject()}
            )
        scripts = names["/JavaScript"]["/Names"]
        # A name tree needs a key for each entry, and any unique one will do.
        scripts.append(create_string_object(str(uuid.uuid4())))
        action = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Action"),
                NameObject("/S"): NameObject("/JavaScript"),
                NameObject("/JS"): TextStringObject(javascript),
            }
        )
        scripts.append(self._add_object(action))

    def add_attachment(self, filename: str, data: str | bytes) -> EmbeddedFile:
        return EmbeddedFile._create_new(self, filename, data)

    @staticmethod
    def _reader_for(source: Any) -> Any:
        from . import PdfReader

        return PdfReader(source)

    def list_objects_in_increment(self) -> list[IndirectObject]:
        """The new and changed objects an incremental save writes."""
        known = len(self._original_hash)
        return [
            item.indirect_reference
            for index, item in enumerate(self._objects)
            if item is not None and (index >= known or _digest(item) != self._original_hash[index])
        ]

    def write_stream(self, stream: Any) -> None:
        if hasattr(stream, "mode") and "b" not in stream.mode:
            logger.warning(
                "File <%s> to write to is not in binary mode. It may not be written to correctly.",
                stream.name,
            )
        stream.write(self.write_bytes())

    def encrypt(
        self,
        user_password: str,
        owner_password: str | None = None,
        use_128bit: bool = True,
        permissions_flag: UserAccessPermissions = ALL_DOCUMENT_PERMISSIONS,
        *,
        algorithm: str | None = None,
    ) -> None:
        """Encrypt the written file, as pypdf does. Only the key is kept, not the passwords."""
        if self.incremental:
            raise NotImplementedError(
                "Encrypting incremental PDF files is currently not supported."
            )
        if algorithm is None:
            chosen = EncryptAlgorithm.RC4_128 if use_128bit else EncryptAlgorithm.RC4_40
        else:
            try:
                chosen = EncryptAlgorithm[algorithm.replace("-", "_")]
            except KeyError:
                raise ValueError(f"Algorithm '{algorithm}' NOT supported") from None
        if self._file_id is None:
            known = None if self._ID is None else self._ID[0]
            self._file_id = secrets.token_bytes(16) if known is None else bytes(known)
        encryption = Encryption.make(chosen, permissions_flag, self._file_id)
        self._encrypt_entry = encryption.write_entry(
            user_password, owner_password or user_password, strict=self.strict
        )
        self._encryption = encryption

    @property
    def is_encrypted(self) -> bool:
        return self._encryption is not None

    def write(self, stream: Any) -> tuple[bool, Any]:
        if stream == "":
            raise ValueError(f"Output({stream=}) is empty.")
        if isinstance(stream, (str, Path)):
            with Path(stream).open("wb") as handle:
                self.write_stream(handle)
            return True, handle
        self.write_stream(stream)
        stream.flush()
        return False, stream

    def _reachable(self) -> dict[PdfReference, Any]:
        """Core forms of the objects the file reaches.

        A full rewrite leaves out anything else, so removed pages and images are gone from
        the output rather than kept as unused objects.
        """
        trailer = self._trailer()
        output: dict[PdfReference, Any] = {}
        pending = references_in(trailer)
        visited: set[int] = set()
        while pending:
            reference = pending.pop()
            number = reference.object_number
            if number in visited or not 0 < number <= len(self._objects):
                continue
            visited.add(number)
            item = self._objects[number - 1]
            if item is None:
                continue
            core = None if number in self._reserved_pages else to_core(item)
            output[PdfReference(number, 0)] = core
            pending.extend(references_in(core))
        return output

    def _trailer(self) -> dict[str, Any]:
        trailer = {"Root": to_core(self._root_object.indirect_reference)}
        if self._info_obj is not None:
            trailer["Info"] = to_core(self._info_obj)
        if self._ID is not None:
            trailer["ID"] = to_core(self._ID)
        return trailer

    def _serialized(self) -> bytes:
        header = self._header + b"\n" + BINARY_MARKER
        objects = self._reachable()
        self._objects_written = len(objects)
        trailer = self._trailer()
        security = None
        exempt = set[int]()
        if self._encryption is not None:
            security = self._encryption.handler
            # Past every object in the table, as what is left out keeps its number.
            entry = PdfReference(len(self._objects) + 1, 0)
            objects[entry] = self._encrypt_entry
            exempt.add(entry.object_number)
            identifier = PdfString(self._file_id or b"", is_literal=False)
            trailer |= {"ID": [identifier, identifier], "Encrypt": entry}
        return header + serialize_body(
            len(header), objects, trailer, ObjectLayout.SPACED, security=security, exempt=exempt
        )

    def _incremental_update(self) -> SavedPdf | bytes:
        """The source's bytes, unchanged, followed by the objects that changed."""
        changed = self.list_objects_in_increment()
        document = self._reader._pdf
        editable = EditableDocument(document)
        trailer = self._trailer()
        same_trailer = all(editable.trailer.get(key) == trailer.get(key) for key in TRAILER_KEYS)
        if not changed and same_trailer:
            return bytes(document.raw_data)
        for reference in changed:
            core = to_core(reference.get_object())
            editable.set_object(PdfReference(reference.idnum, reference.generation), core)
        editable.next_object_number = max(editable.next_object_number, len(self._objects) + 1)
        for key in TRAILER_KEYS:
            editable.trailer.pop(key, None)
        editable.trailer.update(trailer)
        # An xref stream either way, as in pypdf.
        return editable.save_incremental(xref_stream=True)

    def _report_save(self) -> None:
        if not self.incremental:
            self.last_save = SaveReport("full", self._objects_written, (), ())
            return
        changed = {reference.idnum for reference in self.list_objects_in_increment()}
        present = {
            reference.object_number
            for reference, value in self._reachable().items()
            if value is not None
        }
        removed = tuple(sorted(self._original_reachable - present))
        superseded = tuple(
            number for number in sorted(changed) if number <= len(self._original_hash)
        )
        self.last_save = SaveReport("incremental", len(changed), superseded, removed)
        if removed:
            logger.warning(
                "Saved incrementally, so the %d removed objects stay readable in the earlier "
                "revision of the file. Write a full rewrite to remove them.",
                len(removed),
            )

    def write_bytes(self) -> SavedPdf:
        data = self._incremental_update() if self.incremental else self._serialized()
        self._report_save()
        if isinstance(data, SavedPdf):
            return data
        return SavedPdf(data, incremental=self.incremental, encryption=None, xref_stream=False)

    def _repr_mimebundle_(
        self, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None
    ) -> dict[str, Any]:
        output = BytesIO()
        self.write(output)
        data = {"application/pdf": output}
        if include is not None:
            data = {key: value for key, value in data.items() if key in include}
        if exclude is not None:
            data = {key: value for key, value in data.items() if key not in exclude}
        return data


__all__ = ("PageObject", "PdfWriter", "SaveReport")
