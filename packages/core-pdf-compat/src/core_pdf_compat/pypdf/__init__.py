from __future__ import annotations

import logging
import re
from enum import IntFlag, auto
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import Any, NoReturn, cast

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfInvalidPermissionsError, PdfUnsupportedError
from core_pdf.impl.types import PdfName, PdfReference
from core_pdf_compat._shared import ClosingMixin, PdfInput
from core_pdf_compat.pdfminer.pdftypes import DocumentView
from core_pdf_compat.pypdf import filters
from core_pdf_compat.pypdf._doc_common import DocumentInformation, PdfDocCommon
from core_pdf_compat.pypdf._encryption import PasswordType
from core_pdf_compat.pypdf._reader_scan import OBJECT_HEADER, XrefScan
from core_pdf_compat.pypdf._text import LegacyTextCaches
from core_pdf_compat.pypdf.constants import UserAccessPermissions
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
from core_pdf_compat.pypdf.pagerange import PageRange, parse_filename_page_ranges
from core_pdf_compat.pypdf.papersizes import PaperSize
from core_pdf_spec.exceptions import PdfParseError, PdfPasswordError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import key_for

__version__ = "6.15.0"

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
        self._checked_object_streams: set[int] = set()
        self._permissions_tampered = False
        self.read(stream)
        self._open(password)

    def read(self, stream: PdfInput) -> None:
        self.stream, self._data, self._stream_opened = _source_bytes(stream)
        scan = XrefScan(self._data, self.strict, self)
        scan.read()
        self.xref = scan.xref
        self.xref_free_entry = scan.xref_free_entry
        self.xref_objStm = scan.xref_objStm  # noqa: N815
        self._scan_trailer = scan.trailer

    def _open(self, password: str | bytes | None) -> None:
        try:
            self._pdf = PdfDocument.open(self._data, password=password or "")
        except PdfInvalidPermissionsError:
            self._decryption_pending = True
            self._permissions_tampered = True
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
        if password is not None and self._pdf is not None and not self.is_encrypted:
            raise PdfReadError("Not an encrypted file")

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

    def get_num_pages(self) -> int:
        return self.num_pages

    def remove_page(self, page: int | PageObject | IndirectObject, clean: bool = False) -> None:
        pages = self._page_list()
        target = page.get_object() if isinstance(page, IndirectObject) else page
        if isinstance(page, IndirectObject) and not isinstance(target, PageObject):
            logger.warning("IndirectObject is not referencing a page")
            return
        if not isinstance(target, int):
            if target not in pages:
                logger.warning("Cannot find page in pages")
                return
            target = pages.index(target)
        if not 0 <= target < len(pages):
            logger.warning("Page number is out of range")
            return
        reference = pages[target].indirect_reference
        del self.pages[target]
        if clean and reference is not None:
            self._replace_object(reference, NullObject())

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
            if number in self.xref_objStm:
                self._check_object_stream(self.xref_objStm[number][0])
            resolved = source.resolver.resolve(PdfReference(number, generation))
            if isinstance(resolved, PdfStream):
                self._check_stream_length(resolved, source.xref[key_for(number, generation)])
            return to_generic(resolved, self)
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

    def _check_stream_length(self, stream: PdfStream, entry: Any) -> None:
        limit = filters.MAX_DECLARED_STREAM_LENGTH
        declared = stream.dictionary.get("Length")
        if isinstance(declared, int):
            if declared > limit:
                raise LimitReachedError(
                    f"Declared stream length of {declared} exceeds maximum allowed length."
                )
            return
        if declared is None and len(stream.raw_data) >= limit:
            # pypdf reads in chunks up to 8192 bytes, and checks each chunk before searching it.
            read, chunk = 0, 16
            while read + chunk < limit:
                read += chunk
                chunk = min(chunk * 2, 8192)
            start = self._data.index(b"stream", entry.offset) + len(b"stream\n")
            read = min(read + chunk, len(self._data) - start)
            raise LimitReachedError(
                f"Read stream length of {read} exceeds maximum allowed length of {limit}."
            )

    def _check_object_stream(self, number: int) -> None:
        if number in self._checked_object_streams:
            return
        self._checked_object_streams.add(number)
        stream = self.get_object(number)
        data = stream.get_data()
        count = stream["/N"]
        maximum = len(data) // 3
        if count <= maximum:
            return
        message = f"Value /N {count} for object {number} exceeds maximum allowed value {maximum}."
        if self.strict:
            raise LimitReachedError(message)
        logger.warning("%s Limiting to %d.", message, maximum)
        # pypdf reads the header pairs as numbers, which warns on anything else.
        tokens = data[: stream["/First"]].split() + [b""] * (2 * maximum)
        for token in tokens[: 2 * maximum]:
            if not token.isdigit():
                logger.warning("NumberObject(b'') invalid; use 0 instead")

    def cache_get_indirect_object(self, generation: int, idnum: int) -> Any:
        return self.resolved_objects.get((generation, idnum))

    def cache_indirect_object(self, generation: int, idnum: int, obj: Any) -> Any:
        if (generation, idnum) in self.resolved_objects:
            message = f"Overwriting cache for {generation} {idnum}"
            if self.strict:
                raise PdfReadError(message)
            logger.warning(message)
        self.resolved_objects[(generation, idnum)] = obj
        if obj is not None:
            obj.indirect_reference = IndirectObject(idnum, generation, self)
        return obj

    def read_object_header(self, stream: Any) -> tuple[int, int]:
        position = stream.tell()
        match = OBJECT_HEADER.match(stream.read())
        if match is None:
            raise ValueError("no object header")
        stream.seek(position + match.end())
        return int(match[1]), int(match[2])

    def _replace_object(self, indirect: IndirectObject, obj: Any) -> Any:
        if indirect.pdf is not self:
            raise ValueError("Cannot update PdfReader with external object")
        key = (indirect.generation, indirect.idnum)
        if key not in self.resolved_objects:
            raise ValueError("Cannot find referenced object")
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

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, Any]:
        data = {"application/pdf": self._data}
        if include is not None:
            data = {key: value for key, value in data.items() if key in include}
        if exclude is not None:
            data = {key: value for key, value in data.items() if key not in exclude}
        return data

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

    @property
    def _security_handler(self) -> Any:
        return None if self._pdf is None else self._pdf.security_handler

    @property
    def user_access_permissions(self) -> UserAccessPermissions | None:
        handler = self._security_handler
        return None if handler is None else UserAccessPermissions(handler.config.permissions)

    @property
    def are_permissions_valid(self) -> bool | None:
        """True or False once decrypted, and None for a plain or a still locked file.

        pypdf goes on to decrypt a file whose /Perms is wrong. core-pdf treats that as
        tampering, so such a reader stays locked, however often the right password is given.
        """
        if self._permissions_tampered:
            return False
        return None if self._security_handler is None else True

    def decrypt(self, password: str | bytes) -> PasswordType:
        if not self.is_encrypted:
            raise PdfReadError("Not encrypted file")
        try:
            pdf = PdfDocument.open(self._data, password=password)
        except PdfInvalidPermissionsError as error:
            self._permissions_tampered = True
            return PASSWORD_TYPES[error.password_kind]
        except PdfPasswordError, PdfUnsupportedError:
            return PasswordType.NOT_DECRYPTED
        handler = pdf.security_handler
        kind = "user" if handler is None else handler.password_kind
        if self._decryption_pending:
            self._pdf = pdf
            self._decryption_pending = False
            self.__dict__.pop("_generic_trailer", None)
        else:
            pdf.close()
        return PASSWORD_TYPES[kind]


PASSWORD_TYPES = {"user": PasswordType.USER_PASSWORD, "owner": PasswordType.OWNER_PASSWORD}


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
    "PageRange",
    "PaperSize",
    "__version__",
    "mult",
    "parse_filename_page_ranges",
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
