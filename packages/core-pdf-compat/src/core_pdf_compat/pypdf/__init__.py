from __future__ import annotations

import logging
import re
from enum import IntEnum, IntFlag, auto
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import Any, NoReturn, cast

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.types import PdfName, PdfReference
from core_pdf_compat._shared import ClosingMixin, PdfInput
from core_pdf_compat.pdfminer.pdftypes import DocumentView
from core_pdf_compat.pypdf._doc_common import DocumentInformation, PdfDocCommon
from core_pdf_compat.pypdf._reader_scan import XrefScan
from core_pdf_compat.pypdf._text import LegacyTextCaches
from core_pdf_compat.pypdf.errors import (
    DeprecationError,
    FileNotDecryptedError,
    LimitReachedError,
    PdfReadError,
    WrongPasswordError,
)
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    Destination,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    RectangleObject,
    TextStringObject,
    is_null_or_none,
    to_generic,
)
from core_pdf_spec.exceptions import PdfParseError, PdfPasswordError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.xref import key_for

logger = logging.getLogger(__name__)

INHERITED_PAGE_ATTRIBUTES = ("/Resources", "/MediaBox", "/CropBox", "/Rotate")

Rectangle = RectangleObject


def _source_bytes(stream: PdfInput) -> tuple[Any, bytes, bool]:
    """The stream to keep, its bytes, and whether this reader opened it."""
    if "b" not in getattr(stream, "mode", "b"):
        logger.warning(
            "PdfReader stream/file object is not in binary mode. It may not be read correctly."
        )
    if isinstance(stream, (str, Path)):
        with open(stream, "rb") as source:
            data = source.read()
        return BytesIO(data), data, True
    if isinstance(stream, (bytes, bytearray, memoryview)):
        return BytesIO(stream), bytes(stream), False
    readable: Any = stream
    readable.seek(0)
    try:
        return readable, bytes(readable.read()), False
    except UnicodeDecodeError as error:
        raise UnsupportedOperation("cannot read header") from error


class PdfReader(PdfDocCommon, ClosingMixin):
    def __init__(
        self,
        stream: PdfInput,
        strict: bool = False,
        password: str | bytes | None = None,
        *,
        root_object_recovery_limit: int | None = 10_000,
    ) -> None:
        self.strict = strict
        self.flattened_pages: list[PageObject] | None = None
        self.resolved_objects: dict[tuple[int, int], Any] = {}
        self.text_caches = LegacyTextCaches()
        self._named_destinations_cache: dict[str, Destination] | None = None
        self._validated_root: Any = None
        self._root_object_recovery_limit = root_object_recovery_limit
        self._pdf: PdfDocument | None = None
        self._decryption_pending = False
        self.stream, self._data, self._stream_opened = _source_bytes(stream)
        scan = XrefScan(self._data, strict, self)
        scan.read()
        self.xref = scan.xref
        self.xref_free_entry = scan.xref_free_entry
        self.xref_objStm = scan.xref_objStm  # noqa: N815
        self._scan_trailer = scan.trailer
        self._open(password)

    def _open(self, password: str | bytes | None) -> None:
        text = password.decode("utf-8", "replace") if isinstance(password, bytes) else password
        try:
            self._pdf = PdfDocument.open(self._data, password=text or "")
        except PdfPasswordError as error:
            if password is not None:
                raise WrongPasswordError("Wrong password") from error
            self._decryption_pending = True
        except PdfUnsupportedError:
            if password:
                raise
            self._decryption_pending = True
        except PdfParseError, ValueError:
            # pypdf finds out when something asks for the catalog, not on opening.
            self._pdf = None

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

    def _flatten(
        self,
        pages: Any = None,
        inherit: dict[str, Any] | None = None,
        indirect_reference: IndirectObject | None = None,
        visited: set[int] | None = None,
    ) -> None:
        inherit = {} if inherit is None else inherit
        visited = set() if visited is None else visited
        if is_null_or_none(pages):
            pages = self.root_object.get_resolved("/Pages")
            if not isinstance(pages, DictionaryObject):
                raise PdfReadError("Invalid object in /Pages")
            self.flattened_pages = []
        if "/Type" in pages:
            node_type = pages["/Type"]
        elif "/Kids" in pages:
            node_type = "/Pages"
        else:
            if self.strict and not any(
                key in pages for key in ("/Contents", "/MediaBox", "/Parent")
            ):
                raise PdfReadError(f"Non-page object reached through /Kids: {pages!r}")
            node_type = "/Page"
        if node_type == "/Pages":
            self._flatten_kids(pages, inherit, visited)
        elif node_type == "/Page":
            for key, value in inherit.items():
                if key not in pages:
                    pages[NameObject(key)] = value
            page = PageObject(self, indirect_reference)
            page.update(pages)
            page._source_content = page._content_keys()
            cast(list[PageObject], self.flattened_pages).append(page)

    def _flatten_kids(self, node: Any, inherit: dict[str, Any], visited: set[int]) -> None:
        for key in INHERITED_PAGE_ATTRIBUTES:
            if key in node:
                inherit[key] = node[key]
        node_reference = getattr(node, "indirect_reference", object())
        kids = node.get_resolved("/Kids", ArrayObject())
        if isinstance(kids, NullObject):
            return
        if not isinstance(kids, ArrayObject):
            raise PdfReadError(f"Expected /Kids to be an array, got {type(kids).__name__}.")
        for kid in kids:
            if getattr(kid, "indirect_reference", object()) == node_reference:
                raise PdfReadError("Detected cyclic page references.")
            child = kid.get_object() if isinstance(kid, IndirectObject) else kid
            if child:
                if id(child) in visited:
                    raise PdfReadError("Detected cyclic page references.")
                visited.add(id(child))
                reference = kid if isinstance(kid, IndirectObject) else None
                self._flatten(child, inherit, reference, visited)

    def _page_list(self) -> list[PageObject]:
        if self.flattened_pages is None:
            self._flatten()
        return cast(list[PageObject], self.flattened_pages)

    @property
    def pages(self) -> Any:
        return _VirtualList(lambda: len(self._page_list()), lambda index: self._page_list()[index])

    @property
    def num_pages(self) -> int:
        return len(self._page_list())

    def get_page(self, page_number: int) -> PageObject:
        return self._page_list()[page_number]

    def get_page_number(self, page: PageObject) -> int | None:
        return self._get_page_number_by_indirect(page.indirect_reference)

    def _get_page_number_by_indirect(self, indirect_reference: Any) -> int | None:
        if is_null_or_none(indirect_reference):
            return None
        number = getattr(indirect_reference, "idnum", indirect_reference)
        return self._page_numbers().get(number)

    def _page_numbers(self) -> dict[int, int]:
        pages = self._page_list()
        numbers = self.__dict__.get("_page_number_map")
        if numbers is None or len(numbers) > len(pages):
            numbers = {
                page.indirect_reference.idnum: index
                for index, page in reversed(list(enumerate(pages)))
                if page.indirect_reference is not None
            }
            self.__dict__["_page_number_map"] = numbers
        return numbers

    def _core_page(self, page: PageObject) -> Any:
        """core-pdf's copy of ``page``, if it still shows what the file holds."""
        if self._pdf is None or page.indirect_reference is None:
            return None
        if page._source_content != page._content_keys():
            return None
        identifiers = self.__dict__.get("_core_page_ids")
        if identifiers is None:
            identifiers = {
                number: index for index, number in enumerate(DocumentView(self._pdf).page_ids())
            }
            self.__dict__["_core_page_ids"] = identifiers
        index = identifiers.get(page.indirect_reference.idnum)
        return None if index is None else self._pdf.pages[index]

    def get_object(self, indirect_reference: Any) -> Any:
        if self._decryption_pending:
            raise FileNotDecryptedError("File has not been decrypted")
        if isinstance(indirect_reference, IndirectObject):
            number, generation = indirect_reference.idnum, indirect_reference.generation
        else:
            number, generation = int(indirect_reference), 0
        key = (generation, number)
        if key not in self.resolved_objects:
            item = self._read_object(number, generation)
            if item is None:
                return None
            item.indirect_reference = IndirectObject(number, generation, self)
            self.resolved_objects[key] = item
        return self.resolved_objects[key]

    def _read_object(self, number: int, generation: int) -> Any:
        source = self._pdf
        if source is not None and key_for(number, generation) in source.xref:
            return to_generic(source.resolver.resolve(PdfReference(number, generation)), self)
        pattern = rf"\s{number}\s+{generation}\s+obj".encode()
        match = re.search(pattern, self._data)
        if match is None:
            logger.warning("Object %d %d not defined.", number, generation)
            if self.strict:
                raise PdfReadError("Could not find object.")
            return None
        logger.warning("Object %d %d found", number, generation)
        lexer = PdfLexer(self._data)
        lexer.rewind(match.start() + 1)
        return to_generic(lexer.parse_indirect_object(), self)

    def _replace_object(self, indirect: IndirectObject, obj: Any) -> Any:
        key = (indirect.generation, indirect.idnum)
        if key not in self.resolved_objects:
            raise ValueError("Cannot replace an object that has not been read")
        self.resolved_objects[key] = obj
        obj.indirect_reference = indirect
        return obj

    def _next_object_number(self) -> int:
        file_numbers = (key >> 16 for key in self._pdf.xref) if self._pdf is not None else ()
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

    @property
    def trailer(self) -> Any:
        cached = self.__dict__.get("_generic_trailer")
        if cached is None:
            source = self._pdf
            cached = self._scan_trailer
            if source is not None and "/Root" not in cached:
                cached = to_generic(source.trailer_dict, self)
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
        if self._validated_root is None:
            self._validated_root = self._find_root()
        return self._validated_root

    def _find_root(self) -> Any:
        root = self.trailer.get("/Root")
        if is_null_or_none(root):
            logger.warning('Cannot find "/Root" key in trailer')
        elif root.get_object().get("/Type") == "/Catalog":
            return root.get_object()
        else:
            logger.warning("Invalid Root object in trailer")
        logger.warning('Searching object with "/Catalog" key')
        for number in range(1, self.trailer.get("/Size", 0) + 1):
            limit = self._root_object_recovery_limit
            if limit is not None and number > limit:
                raise LimitReachedError("Maximum Root object recovery limit reached.")
            try:
                candidate = self.get_object(number)
            except PdfReadError, PdfParseError, ValueError:
                continue
            if isinstance(candidate, DictionaryObject) and candidate.get("/Type") == "/Catalog":
                logger.warning("Root found at %r", candidate.indirect_reference)
                return candidate
        if not is_null_or_none(root) and "/Pages" in root.get_object():
            logger.warning(
                "Possible root found at %r, but missing /Catalog key", root.indirect_reference
            )
            return root.get_object()
        raise PdfReadError("Cannot find Root object in pdf")

    def _catalog_name(self, key: str) -> str | None:
        if self._pdf is None:
            return None
        value = self._pdf.resolver.resolve(self._pdf.catalog().get(key))
        return f"/{value.value}" if isinstance(value, PdfName) else None

    @property
    def page_layout(self) -> str | None:
        return self._catalog_name("PageLayout")

    @property
    def page_mode(self) -> str | None:
        return self._catalog_name("PageMode")

    @property
    def page_labels(self) -> list[str]:
        if self._pdf is None:
            return []
        count = len(self._pdf.pages)
        labels: Any = self._pdf.build_page_labels(page_count=count) or [None] * count
        return [label or str(index) for index, label in enumerate(labels, 1)]

    @property
    def pdf_header(self) -> str:
        return self._data[:8].decode("utf-8", "backslashreplace")

    def get_form_xfa(self) -> None:
        return None

    def decode_permissions(self, permissions_code: int) -> NoReturn:
        del permissions_code
        raise DeprecationError(
            "decode_permissions is deprecated and was removed in pypdf 5.0.0. "
            "Use user_access_permissions instead."
        )

    def close(self) -> None:
        if self._stream_opened:
            self.stream.close()
        if self._pdf is not None:
            self._pdf.close()
        self.flattened_pages = []
        self.resolved_objects = {}
        self.__dict__["_generic_trailer"] = DictionaryObject()

    @property
    def is_encrypted(self) -> bool:
        return self._decryption_pending or bool(getattr(self._pdf, "decipher", None))

    def decrypt(self, password: str | bytes) -> bool:
        if not self.is_encrypted:
            raise PdfReadError("Not encrypted file")
        if not self._decryption_pending:
            return True
        text = password.decode("utf-8", "replace") if isinstance(password, bytes) else password
        try:
            pdf = PdfDocument.open(self._data, password=text)
        except PdfPasswordError, PdfUnsupportedError:
            return False
        self._pdf = pdf
        self._decryption_pending = False
        self.__dict__.pop("_generic_trailer", None)
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
    "PdfReader",
    "Rectangle",
)

from ._page import _VirtualList  # noqa: E402
from ._page_merge import Transformation  # noqa: E402
from ._writer import PageObject, PdfWriter  # noqa: E402
