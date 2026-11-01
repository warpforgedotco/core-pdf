from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from copy import replace
from enum import IntEnum, IntFlag, auto
from io import BytesIO
from mmap import mmap
from typing import Any, cast

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.output_model import (
    Document,
    Page,
)
from core_pdf.impl.types import PdfName, PdfReference
from core_pdf_compat._shared import ClosingMixin, PdfInput
from core_pdf_compat.pdfminer.pdftypes import DocumentView
from core_pdf_compat.pypdf._doc_common import DocumentInformation, PdfDocCommon
from core_pdf_compat.pypdf._text import LegacyTextCaches, extract_legacy_text
from core_pdf_compat.pypdf.errors import FileNotDecryptedError, PdfReadError
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    Destination,
    DictionaryObject,
    IndirectObject,
    NameObject,
    RectangleObject,
    TextStringObject,
    is_null_or_none,
    to_generic,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for

logger = logging.getLogger(__name__)

# pypdf Configuration.page_tree_maximum_depth and page_tree_maximum_entries.
PAGE_TREE_MAXIMUM_DEPTH = 100
PAGE_TREE_MAXIMUM_ENTRIES = 100_000


# pypdf._reader.PdfReader._MAX_STARTXREF_RECOVERY_LINES.
STARTXREF_RECOVERY_LINES = 1000
LINE_SEARCH_WINDOW = 4096
TRUNCATED_EOF_MARKERS = (b"%%EO", b"%%E", b"%%", b"%")


def previous_line(data: bytes | mmap, position: int) -> tuple[bytes, int]:
    """pypdf's read_previous_line: the line ending at ``position``, and the
    position after skipping the line breaks above it."""
    if position == 0:
        raise ValueError("stream has ended unexpectedly")
    # Search back in windows so each call stops at the nearest line break: a
    # search to the start of the data for the other break byte would make a file
    # with no marker quadratic.
    line_break = -1
    start = end = position
    while start > 0:
        start = max(0, start - LINE_SEARCH_WINDOW)
        line_break = max(data.rfind(b"\n", start, end), data.rfind(b"\r", start, end))
        if line_break >= 0:
            break
        end = start
    line = data[line_break + 1 : position]
    while line_break >= 0 and data[line_break] in b"\r\n":
        line_break -= 1
    return line, line_break + 1


def is_startxref_value(value: bytes) -> bool:
    return re.fullmatch(rb"[+-]?[0-9]+(?:_[0-9]+)*", value) is not None


def finds_startxref(data: bytes | mmap) -> bool:
    """Whether pypdf (non-strict) locates %%EOF and the startxref offset above it.

    Mirrors PdfReader._find_eof_marker and _find_startxref_pos: the last line
    may carry the marker inline (py-pdf/pypdf#4127) or truncated, duplicate
    markers above it are skipped (#4008), and a corrupt trailing pointer falls
    back to the previous intact startxref (#3238).
    """
    try:
        position = end_position = len(data)
        line = b""
        last = True
        while True:
            marker = line.find(b"%%EOF")
            if marker == 0:
                break
            if marker > 0 and last:
                if line[:marker].lstrip(b" \t\n\r\x00\x0c"):
                    position = end_position - len(line) + marker
                break
            if line and last:
                if line.strip().endswith(TRUNCATED_EOF_MARKERS):
                    break
                last = False
            end_position = position
            line, position = previous_line(data, position)

        line, position = previous_line(data, position)
        for _ in range(STARTXREF_RECOVERY_LINES):
            if not line.startswith(b"%%EOF") or position == 0:
                break
            line, position = previous_line(data, position)
        try:
            int(line)
        except ValueError:
            return line.startswith(b"startxref") and is_startxref_value(line[9:].strip())
        line, position = previous_line(data, position)
        if line.startswith(b"startxref"):
            return True

        offset_found = False
        for _ in range(STARTXREF_RECOVERY_LINES):
            if position <= 0:
                break
            line, position = previous_line(data, position)
            if not line.startswith(b"startxref"):
                try:
                    int(line)
                except ValueError:
                    offset_found = False
                else:
                    offset_found = True
                continue
            if len(line) > 9:
                if is_startxref_value(line[9:].strip()):
                    return True
                continue
            return offset_found
        return False
    except ValueError:
        return False


def validate_pypdf_page_tree(pdf: PdfDocument) -> None:
    literal_trailers = tuple(pdf.iter_literal_trailer_dictionaries())
    if literal_trailers:
        latest_root = literal_trailers[-1].get("Root")
        if isinstance(latest_root, PdfReference):
            root_key = key_for(latest_root.object_number, latest_root.generation_number)
            if root_key not in pdf.xref:
                raise ValueError("catalog root references a missing object generation")
        elif latest_root is not None and not isinstance(latest_root, dict):
            raise ValueError("invalid catalog root")

    root = pdf.resolver.as_dict(pdf.catalog().get("Pages"))
    if root is None:
        raise ValueError("invalid object in /Pages")
    # pypdf walks the tree iteratively. A cycle is a /Pages node reached again
    # while it is still an ancestor; a page shared by two parents is valid.
    # Entries that are not dictionaries, and empty ones, are skipped.
    entries = 0
    ancestors: set[int] = set()
    stack: list[tuple[PdfDict, Iterator[object], int]] = []

    def enter(node: PdfDict, depth: int) -> None:
        if depth > PAGE_TREE_MAXIMUM_DEPTH:
            raise ValueError(f"maximum page tree depth reached: {depth}")
        declared_type = node.get("Type")
        if declared_type is not None:
            node_type = str(declared_type)
        else:
            node_type = "Page" if "Kids" not in node else "Pages"
        if node_type != "Pages":
            return
        kids = pdf.resolver.resolve(node.get("Kids"))
        if kids is None:
            kids = []
        elif not isinstance(kids, (list, tuple)):
            raise ValueError("expected /Kids to be an array")
        ancestors.add(id(node))
        stack.append((node, iter(kids), depth))

    enter(root, 0)
    while stack:
        parent, kids, depth = stack[-1]
        for kid in kids:
            child = pdf.resolver.as_dict(kid)
            if not child:
                continue
            if id(child) in ancestors:
                raise ValueError("detected cyclic page references")
            entries += 1
            if entries > PAGE_TREE_MAXIMUM_ENTRIES:
                raise ValueError(f"maximum page tree entry limit reached: {entries}")
            enter(child, depth + 1)
            break
        else:
            stack.pop()
            ancestors.discard(id(parent))


class StructuredState(ClosingMixin):
    def __init__(self, pdf: PdfDocument | None, structured: Document | None = None) -> None:
        self.pdf = pdf
        self._structured = structured
        self.text_caches = LegacyTextCaches()

    @property
    def structured(self) -> Document:
        if self._structured is None:
            self._structured = self.source_pdf.structured_document
        return self._structured

    @classmethod
    def open(cls, source: PdfInput, *, password: str = "") -> StructuredState:
        pdf = PdfDocument.open(source, password=password)
        try:
            if not finds_startxref(pdf.raw_data):
                raise ValueError("startxref not found")
            validate_pypdf_page_tree(pdf)
        except Exception:
            pdf.close()
            raise
        return cls(pdf)

    @classmethod
    def synthetic(cls, structured: Document) -> StructuredState:
        return cls(None, structured)

    @property
    def source_pdf(self) -> PdfDocument:
        if self.pdf is None:
            raise ValueError("synthetic snapshots do not have a source PDF")
        return self.pdf

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.structured.metadata

    @property
    def pages(self) -> tuple[Page, ...]:
        return self.structured.pages

    def page_count(self) -> int:
        pdf = self.source_pdf
        return len(pdf.pages) if pdf.declared_page_count() else 0

    @property
    def form_fields(self) -> tuple[Any, ...]:
        return tuple(field for page in self.pages for field in page.form_fields)

    def capability_document(self) -> PdfDocument:
        if self.pdf is None:
            raise PdfUnsupportedError("synthetic snapshots do not have PDF capabilities")
        return self.pdf

    def capability_page(self, page_number: int) -> Any:
        return self.capability_document().pages[page_number - 1]

    def close(self) -> None:
        if self.pdf is not None:
            self.pdf.close()


Rectangle = RectangleObject


class PdfPageObject:
    def __init__(
        self, document: StructuredState, page: Page | None, page_number: int | None = None
    ) -> None:
        self._document = document
        self._loaded_page = page
        self._page_number = page.page_number if page is not None else page_number or 0
        self.text_override: str | None = None
        self._reader: Any = None
        if document.pdf is not None and 0 < self._page_number <= len(document.pdf.pages):
            source_page = document.pdf.pages[self._page_number - 1]
            media_box = source_page.media_box or (0.0, 0.0, self._page.width, self._page.height)
            crop_box = source_page.crop_box or media_box
            self.mediabox = Rectangle(media_box)
            self.cropbox = Rectangle(crop_box)
            self.user_unit = source_page.user_unit
            raw_rotation = source_page.inherited_values.get("Rotate")
            self.rotation = (
                int(raw_rotation)
                if isinstance(raw_rotation, (int, float))
                else source_page.rotation
            )
        else:
            page = self._page
            self.mediabox = Rectangle((0, 0, page.width, page.height))
            self.cropbox = Rectangle(page.cropbox or self.mediabox)
            self.user_unit = page.user_unit
            self.rotation = page.rotation

    @property
    def _page(self) -> Page:
        if self._loaded_page is None:
            self._loaded_page = self._document.pages[self._page_number - 1]
        return self._loaded_page

    @_page.setter
    def _page(self, page: Page) -> None:
        self._loaded_page = page

    def _is_source_page(self) -> bool:
        page = self._loaded_page
        return page is None or page is self._document.pages[page.page_number - 1]

    def _capability_view(self) -> Any:
        return self._document.capability_page(self._page.page_number).structured_view

    def extract_text(self, *args: object, **kwargs: object) -> str:  # noqa: C901
        del args, kwargs
        if self.text_override is not None:
            return self.text_override
        if self._is_source_page() and self._document.pdf is not None:
            page = self._document.capability_page(self._page_number)
            return extract_legacy_text(page, self._document.text_caches)
        return "\n".join(line.text for block in self._page.blocks for line in block.lines)

    @property
    def indirect_reference(self) -> IndirectObject | None:
        if self._reader is None:
            return None
        page_ids = self._reader.page_object_numbers()
        if not 0 < self._page_number <= len(page_ids):
            return None
        return IndirectObject(page_ids[self._page_number - 1], 0, self._reader)

    def get_contents(self) -> Any:
        from core_pdf_compat.pypdf.generic import ContentStream

        contents = self._generic_dictionary().get_resolved("/Contents")
        return None if is_null_or_none(contents) else ContentStream(contents, self._reader)

    def _generic_dictionary(self) -> Any:
        pdf = self._document.pdf
        if pdf is None:
            return to_generic({}, self._reader)
        if self._reader is not None:
            page_ids = self._reader.page_object_numbers()
            if self._page_number <= len(page_ids):
                return self._reader.get_object(page_ids[self._page_number - 1])
        return to_generic(pdf.pages[self._page_number - 1].page_dict, self._reader)

    def __getitem__(self, key: str) -> Any:
        return self._generic_dictionary()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._generic_dictionary()

    def get(self, key: str, default: Any = None) -> Any:
        return self._generic_dictionary().get(key, default)

    def keys(self) -> Any:
        return self._generic_dictionary().keys()

    @property
    def page_number(self) -> int:
        return self._page_number - 1

    def compress_content_streams(self, level: int = -1) -> None:
        del level
        raise ValueError("Page must be part of a PdfWriter")

    def rotate(self, angle: int) -> PdfPageObject:
        self._page = replace(self._page, rotation=(self.rotation + angle) % 360)
        self.rotation = self._page.rotation
        return self

    def scale_to(self, width: float, height: float) -> PdfPageObject:
        self._page = replace(self._page, width=width, height=height)
        self.mediabox = Rectangle((0, 0, width, height))
        self.cropbox = self.mediabox
        return self

    def set_cropbox(self, box: tuple[float, float, float, float]) -> PdfPageObject:
        rectangle = Rectangle(box)
        self._page = replace(
            self._page,
            cropbox=(rectangle.left, rectangle.bottom, rectangle.right, rectangle.top),
        )
        self.cropbox = rectangle
        return self

    def transfer_rotation_to_content(self) -> None:
        if self.rotation % 180:
            self.mediabox = Rectangle((0, 0, self.mediabox.height, self.mediabox.width))
            self.cropbox = Rectangle((0, 0, self.cropbox.height, self.cropbox.width))
        self._page = replace(self._page, rotation=0)
        self.rotation = 0

    def merge_page(self, page: PdfPageObject | Page) -> PdfPageObject:
        base_text = self.extract_text()
        overlay_text = page.extract_text() if isinstance(page, PdfPageObject) else page.text
        overlay = page._page if isinstance(page, PdfPageObject) else page
        order = max((item.order for item in self._page.elements), default=-1) + 1
        blocks = tuple(replace(item, order=order + item.order) for item in overlay.blocks)
        tables = tuple(replace(item, order=order + item.order) for item in overlay.tables)
        figures = tuple(replace(item, order=order + item.order) for item in overlay.figures)
        self._page = replace(
            self._page,
            blocks=(*self._page.blocks, *blocks),
            tables=(*self._page.tables, *tables),
            figures=(*self._page.figures, *figures),
            links=(*self._page.links, *overlay.links),
            annotations=(*self._page.annotations, *overlay.annotations),
            form_fields=(*self._page.form_fields, *overlay.form_fields),
        )
        self.text_override = "\n".join(filter(None, (base_text, overlay_text)))
        return self

    @property
    def annotations(self) -> tuple[Any, ...]:
        return self._page.annotations or self._capability_view().annotations

    @property
    def links(self) -> tuple[Any, ...]:
        return self._page.links or self._capability_view().links

    @property
    def form_fields(self) -> tuple[Any, ...]:
        return self._page.form_fields or self._capability_view().form_fields

    @property
    def images(self) -> tuple[Any, ...]:
        return tuple(self._document.capability_page(self._page.page_number).extract_images())


class LockedPages:
    def __iter__(self) -> Any:
        raise PdfUnsupportedError("file has not been decrypted")

    def __len__(self) -> int:
        raise PdfUnsupportedError("file has not been decrypted")

    def __getitem__(self, index: object) -> Any:
        del index
        raise PdfUnsupportedError("file has not been decrypted")


class PdfReader(PdfDocCommon, ClosingMixin):
    pages: Any

    def __init__(
        self,
        stream: PdfInput,
        strict: bool = False,
        password: str | None = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.strict = strict
        self.resolved_objects: dict[tuple[int, int], Any] = {}
        self._named_destinations_cache: dict[str, Destination] | None = None
        self._encrypted_source = stream.getvalue() if isinstance(stream, BytesIO) else stream
        try:
            document = StructuredState.open(self._encrypted_source, password=password or "")
        except PdfUnsupportedError:
            if password:
                raise
            self._document = None
            self.pages = LockedPages()
            self._decryption_pending = True
            return
        self._decryption_pending = False
        self._bind(document)

    def _bind(self, document: StructuredState) -> None:
        self._document = document  # type: ignore[assignment]
        self.__dict__.pop("pages", None)
        self.__dict__.pop("_page_indexes", None)
        self.__dict__.pop("_generic_trailer", None)
        self.resolved_objects = {}
        self._named_destinations_cache = None

    def __getattr__(self, name: str) -> Any:
        if name == "pages" and self.__dict__.get("_document") is not None:
            self._materialize()
            return self.__dict__[name]
        raise AttributeError(name)

    def _materialize(self) -> None:
        document = cast(StructuredState, self._document)
        self.pages = tuple(
            PdfPageObject(document, None, page_number)
            for page_number in range(1, document.page_count() + 1)
        )
        for page in self.pages:
            page._reader = self
        self._page_indexes = {id(page): index for index, page in enumerate(self.pages)}

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def get_page(self, page_number: int) -> PdfPageObject:
        return self.pages[page_number]

    def get_page_number(self, page: PdfPageObject | Page) -> int:
        pages = self.pages
        page_indexes = self.__dict__.get("_page_indexes")
        if page_indexes is not None and isinstance(page, PdfPageObject):
            index = page_indexes.get(id(page))
            if index is not None and pages[index] is page:
                return index
        value = page._page if isinstance(page, PdfPageObject) else page
        candidate: PdfPageObject
        for index, candidate in enumerate(pages):
            if candidate._page is value or candidate._page == value:
                return index
        raise ValueError("page is not part of this reader")

    def _get_named_destinations(
        self,
        *,
        tree: Any = None,
        retval: dict[str, Destination] | None = None,
        visited: set[int] | None = None,
    ) -> dict[str, Destination]:
        if tree is not None or retval is not None or visited is not None:
            return super()._get_named_destinations(tree=tree, retval=retval, visited=visited)
        if self._named_destinations_cache is None:
            self._named_destinations_cache = super()._get_named_destinations()
        return self._named_destinations_cache

    def _get_page_number_by_indirect(self, indirect_reference: Any) -> int | None:
        if is_null_or_none(indirect_reference):
            return None
        number = getattr(indirect_reference, "idnum", indirect_reference)
        numbers = self.__dict__.get("_page_numbers")
        if numbers is None:
            numbers = {idnum: index for index, idnum in enumerate(self.page_object_numbers())}
            self.__dict__["_page_numbers"] = numbers
        return numbers.get(number)

    def get_object(self, indirect_reference: Any) -> Any:
        source = self._source_document()
        if source is None:
            return None
        if isinstance(indirect_reference, IndirectObject):
            number, generation = indirect_reference.idnum, indirect_reference.generation
        else:
            number, generation = int(indirect_reference), 0
        key = (generation, number)
        if key not in self.resolved_objects:
            resolved = source.resolver.resolve(PdfReference(number, generation))
            item = to_generic(resolved, self)
            item.indirect_reference = IndirectObject(number, generation, self)
            self.resolved_objects[key] = item
        return self.resolved_objects[key]

    def _replace_object(self, indirect: IndirectObject, obj: Any) -> Any:
        key = (indirect.generation, indirect.idnum)
        if key not in self.resolved_objects:
            raise ValueError("Cannot replace an object that has not been read")
        self.resolved_objects[key] = obj
        obj.indirect_reference = indirect
        return obj

    def _next_object_number(self) -> int:
        file_numbers = (key >> 16 for key in self._source_document().xref)
        return max((*file_numbers, *(number for _, number in self.resolved_objects))) + 1

    def add_form_topname(self, name: str) -> DictionaryObject | None:
        form = self.root_object.get_resolved("/AcroForm")
        if not isinstance(form, DictionaryObject) or "/Fields" not in form:
            return None
        group = DictionaryObject()
        group[NameObject("/T")] = TextStringObject(name)
        group[NameObject("/Kids")] = form["/Fields"]
        reference = IndirectObject(self._next_object_number(), 0, self)
        group.indirect_reference = reference
        self.resolved_objects[(0, reference.idnum)] = group
        form[NameObject("/Fields")] = ArrayObject([reference])
        for field in group["/Kids"]:
            item = field.get_object()
            if "/Parent" in item:
                logger.warning(
                    "Top Level Form Field %s has a non-expected parent", item.indirect_reference
                )
            item[NameObject("/Parent")] = reference
        return group

    def rename_form_topname(self, name: str) -> DictionaryObject | None:
        form = self.root_object.get_resolved("/AcroForm")
        if not isinstance(form, DictionaryObject) or "/Fields" not in form:
            return None
        group = form["/Fields"][0].get_object()
        group[NameObject("/T")] = TextStringObject(name)
        return group

    def page_object_numbers(self) -> list[int]:
        numbers = self.__dict__.get("_page_object_numbers")
        if numbers is None:
            source = self._source_document()
            numbers = [] if source is None else DocumentView(source).page_ids()
            self.__dict__["_page_object_numbers"] = numbers
        return numbers

    @property
    def trailer(self) -> Any:
        cached = self.__dict__.get("_generic_trailer")
        if cached is None:
            source = self._source_document()
            cached = to_generic(source.trailer_dict if source is not None else {}, self)
            self.__dict__["_generic_trailer"] = cached
        return cached

    @property
    def metadata(self) -> DocumentInformation | None:
        info = self.trailer.get_resolved("/Info")
        if is_null_or_none(info):
            return None
        if not isinstance(info, DictionaryObject):
            raise PdfReadError(
                "Trailer not found or does not point to a document information dictionary"
            )
        metadata = DocumentInformation()
        metadata.update(info)
        return metadata

    @property
    def root_object(self) -> Any:
        if self._decryption_pending:
            raise FileNotDecryptedError("File has not been decrypted")
        return self.trailer["/Root"]

    def _catalog_name(self, key: str) -> str | None:
        source = self._source_document()
        if source is None:
            return None
        value = source.resolver.resolve(source.catalog().get(key))
        return f"/{value.value}" if isinstance(value, PdfName) else None

    def _source_document(self) -> Any:
        document = self.__dict__.get("_document")
        return None if document is None else document.source_pdf

    @property
    def page_layout(self) -> str | None:
        return self._catalog_name("PageLayout")

    @property
    def page_mode(self) -> str | None:
        return self._catalog_name("PageMode")

    @property
    def page_labels(self) -> list[str]:
        source = self._source_document()
        if source is None:
            return []
        count = len(source.pages)
        labels = source.build_page_labels(page_count=count) or [None] * count
        return [label or str(index) for index, label in enumerate(labels, 1)]

    @property
    def pdf_header(self) -> str:
        source = self.__dict__.get("_encrypted_source")
        if not isinstance(source, (bytes, bytearray)):
            source = bytes(self._source_document().raw_data)
        return bytes(source[:8]).decode("latin-1")

    def get_form_xfa(self) -> None:
        return None

    def close(self) -> None:
        if self._document is not None:
            self._document.close()

    @property
    def is_encrypted(self) -> bool:
        return self._decryption_pending or bool(
            getattr(getattr(self._document, "pdf", None), "decipher", None)
        )

    def decrypt(self, password: str) -> bool:
        if not self._decryption_pending:
            return not self.is_encrypted
        try:
            document = StructuredState.open(self._encrypted_source, password=password)
        except PdfUnsupportedError:
            return False
        self._decryption_pending = False
        self._bind(document)
        return True


class PasswordType(IntEnum):
    NOT_DECRYPTED = 0
    USER_PASSWORD = 1
    OWNER_PASSWORD = 2


class ImageType(IntFlag):
    NONE = 0
    XOBJECT_IMAGES = auto()
    INLINE_IMAGES = auto()
    DRAWING_IMAGES = auto()
    ALL = XOBJECT_IMAGES | INLINE_IMAGES | DRAWING_IMAGES
    IMAGES = ALL


class ObjectDeletionFlag(IntFlag):
    NONE = 0
    TEXT = auto()
    LINKS = auto()
    ATTACHMENTS = auto()
    OBJECTS_3D = auto()
    ALL_ANNOTATIONS = auto()
    XOBJECT_IMAGES = auto()
    INLINE_IMAGES = auto()
    DRAWING_IMAGES = auto()
    IMAGES = XOBJECT_IMAGES | INLINE_IMAGES | DRAWING_IMAGES


def mult(m: list[float], n: list[float]) -> list[float]:
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


__all__ = (
    "mult",
    "DocumentInformation",
    "ObjectDeletionFlag",
    "ImageType",
    "PageObject",
    "PasswordType",
    "Transformation",
    "PdfWriter",
    "Destination",
    "PdfPageObject",
    "PdfReader",
    "Rectangle",
)

from ._page_merge import Transformation  # noqa: E402
from ._writer import PageObject, PdfWriter  # noqa: E402

PageObject.register(PdfPageObject)
