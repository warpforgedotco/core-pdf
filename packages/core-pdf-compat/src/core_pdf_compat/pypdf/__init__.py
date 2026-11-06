# The reader's fallbacks for objects core cannot read, and its catalog search, follow
# pypdf._reader.PdfReader, Copyright (c) 2006, Mathieu Fenniak, under its BSD-style license.
from __future__ import annotations

import contextlib
import functools
import logging
import re
import sys
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import Any, NoReturn, cast

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import (
    PdfDecryptionError,
    PdfInvalidPermissionsError,
    PdfUnsupportedError,
)
from core_pdf.impl.recovery_lexer import PdfLexer as RecoveryLexer
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_compat._shared import ClosingMixin, PdfInput
from core_pdf_compat.pdfminer.pdftypes import DocumentView
from core_pdf_compat.pypdf import filters
from core_pdf_compat.pypdf._crypt_providers import CryptAES
from core_pdf_compat.pypdf._doc_common import DocumentInformation, PdfDocCommon
from core_pdf_compat.pypdf._encryption import PasswordType
from core_pdf_compat.pypdf._generic_read import (
    read_non_whitespace,
    read_object,
    read_until_whitespace,
    skip_over_comment,
    skip_over_whitespace,
)
from core_pdf_compat.pypdf._page_labels import index2label
from core_pdf_compat.pypdf._reader_scan import (
    DECODE_KEY_REFERENCE,
    INDIRECT_HEADER,
    NAME_INTO_NUL,
    OBJECT_VALUES_END,
    PYPDF_ONLY_ESCAPE,
    XrefScan,
    pypdf_stream_data,
    stream_keyword,
)
from core_pdf_compat.pypdf._text import LegacyTextCaches
from core_pdf_compat.pypdf.constants import ImageType, ObjectDeletionFlag, UserAccessPermissions
from core_pdf_compat.pypdf.errors import (
    DeprecationError,
    FileNotDecryptedError,
    LimitReachedError,
    PdfReadError,
    WrongPasswordError,
)
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    BareKeywordFound,
    ByteStringObject,
    Destination,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NullObject,
    RectangleObject,
    StreamObject,
    TextStringObject,
    create_string_object,
    is_null_or_none,
    to_generic,
)
from core_pdf_compat.pypdf.pagerange import PageRange, parse_filename_page_ranges
from core_pdf_compat.pypdf.papersizes import PaperSize
from core_pdf_spec.exceptions import PdfParseError, PdfPasswordError
from core_pdf_spec.s_07_security.output import is_signature_contents
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


DECODE_KEYS = frozenset({"Filter", "DecodeParms", "DP", "FFilter", "FDecodeParms"})


def _holds_reference(value: Any) -> bool:
    if isinstance(value, PdfReference):
        return True
    if isinstance(value, dict):
        return any(_holds_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_holds_reference(item) for item in value)
    return False


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
        self._root_object_recovery_limit = (
            sys.maxsize if root_object_recovery_limit is None else root_object_recovery_limit
        )
        self._pdf: PdfDocument | None = None
        self._decryption_pending = False
        self._checked_object_streams: set[int] = set()
        self._known_objects: set[tuple[int, int]] = set()
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
            try:
                resolved = source.resolver.resolve(PdfReference(number, generation))
            except PdfParseError as error:
                return self._repaired_object(number, generation, error)
            entry = source.xref[key_for(number, generation)]
            if isinstance(resolved, PdfStream):
                if resolved.dictionary.get("Length") == PdfReference(number, generation):
                    # pypdf reads /Length while reading the object, so it meets it again.
                    reference = IndirectObject(number, generation, self)
                    raise LimitReachedError(f"Detected loop with self reference for {reference!r}.")
                self._check_stream_length(resolved, entry)
                resolved = self._as_pypdf_reads(resolved, entry, number, generation)
            in_object_stream = getattr(entry, "object_stream", None) is not None
            if not in_object_stream and self._has_pypdf_only_value(entry):
                found = self._read_with_pypdf_rules(entry)
                return self._decrypted_as_pypdf(found, number, generation)
            if source.security_handler is not None:
                resolved = self._with_signatures_decrypted(resolved, number, generation)
            if in_object_stream:
                # In an object stream, read as core reads it: the stream's own bytes are not
                # at hand here for pypdf's parser.
                return to_generic(resolved, self)
            self._bare_keywords_raise = True
            try:
                return to_generic(resolved, self)
            except BareKeywordFound:
                # Core tolerated a word pypdf's parser refuses, and pypdf keeps only what it
                # read before it, so this object is read with pypdf's rules.
                found = self._read_with_pypdf_rules(entry)
                return self._decrypted_as_pypdf(found, number, generation)
            finally:
                self._bare_keywords_raise = False
        # Core cannot read it: pypdf's get_object, from its own xref, then by searching.
        if number in self.xref.get(generation, {}):
            if self.xref_free_entry.get(generation, {}).get(number, False):
                return NullObject()
            return self._repaired_object(number, generation, None)
        return self._found_object(number, generation)

    def _as_pypdf_reads(
        self, stream: PdfStream, entry: Any, number: int, generation: int
    ) -> PdfStream:
        """The stream as pypdf reads it: decoding keys keep their references, and the data is
        what pypdf keeps, which differs only when /Length is wrong.

        In an encrypted file, /Length counts the encrypted bytes, so the rule applies to those,
        and what it keeps is then decrypted, as in pypdf.
        """
        resolved_dictionary = stream.dictionary
        stream = self._with_decode_references(stream, entry)
        declared = stream.dictionary.get("Length")
        if isinstance(declared, PdfReference):
            declared = self._pdf.resolver.resolve(declared) if self._pdf is not None else None
        handler = None if self._pdf is None else self._pdf.security_handler
        is_xref = str(stream.dictionary.get("Type", "")).endswith("XRef")
        encrypted = handler is not None and not is_xref
        if not encrypted and isinstance(declared, int) and declared == len(stream.raw_data):
            return stream
        offset = getattr(entry, "offset", None)
        if not isinstance(offset, int) or getattr(entry, "object_stream", None) is not None:
            return stream
        found = pypdf_stream_data(
            self._data, offset, declared if isinstance(declared, int) else None
        )
        if found is not None and handler is not None and encrypted:
            try:
                found = handler.decrypt(number, generation, found, resolved_dictionary)
            except PdfDecryptionError, ValueError:
                return stream
        if found is None or found == bytes(stream.raw_data):
            return stream
        return stream.replace(raw_data=found)

    def _decrypted_as_pypdf(self, value: Any, number: int, generation: int) -> Any:
        """Decrypt what pypdf's parser read, as pypdf's reader does after parsing."""
        handler = None if self._pdf is None else self._pdf.security_handler
        if handler is None:
            return value
        if isinstance(value, (TextStringObject, ByteStringObject)):
            try:
                data = handler.decrypt(number, generation, value.original_bytes)
            except PdfDecryptionError, ValueError:
                data = self._decrypted_despite_padding(number, generation, value.original_bytes)
            return create_string_object(data)
        if isinstance(value, StreamObject):
            if value.get("/Type") != "/XRef":
                with contextlib.suppress(PdfDecryptionError, ValueError):
                    value._data = handler.decrypt(number, generation, value._data, None)
            for key, item in list(dict.items(value)):
                dict.__setitem__(value, key, self._decrypted_as_pypdf(item, number, generation))
            return value
        if isinstance(value, DictionaryObject):
            for key, item in list(dict.items(value)):
                dict.__setitem__(value, key, self._decrypted_as_pypdf(item, number, generation))
        elif isinstance(value, ArrayObject):
            for index, item in enumerate(value):
                list.__setitem__(value, index, self._decrypted_as_pypdf(item, number, generation))
        return value

    def _with_signatures_decrypted(self, value: Any, number: int, generation: int) -> Any:
        """Decrypt a signature's /Contents too, as pypdf does, though PDF leaves it in the clear."""
        if isinstance(value, PdfStream) or not isinstance(value, (dict, list)):
            return value
        if isinstance(value, list):
            return [self._with_signatures_decrypted(item, number, generation) for item in value]
        output = {
            key: self._with_signatures_decrypted(item, number, generation)
            for key, item in value.items()
        }
        for key, item in value.items():
            if str(getattr(key, "value", key)) == "Contents" and is_signature_contents(value, item):
                handler = cast(Any, self._pdf).security_handler
                try:
                    data = handler.decrypt(number, generation, bytes(item.data))
                except PdfDecryptionError, ValueError:
                    data = self._decrypted_despite_padding(number, generation, bytes(item.data))
                output[key] = PdfString(data, is_literal=False)
        return output

    def _decrypted_despite_padding(self, number: int, generation: int, data: bytes) -> bytes:
        """A string core cannot decrypt, as pypdf decrypts it: AES data whose padding is wrong
        is kept, cut by its last byte, with a warning, and raises in strict mode.
        """
        handler = cast(Any, self._pdf).security_handler
        match handler.crypt_method(None, None):
            case "AESV2":
                key = handler.object_key(number, generation, b"sAlT")
            case "AESV3":
                key = handler.file_key
            case _:
                return data
        try:
            return CryptAES(key).decrypt(data, strict=self.strict)
        except PdfDecryptionError, ValueError:
            return data

    @functools.cached_property
    def _holds_nul(self) -> bool:
        return b"\x00" in self._data

    def _has_pypdf_only_value(self, entry: Any) -> bool:
        """Whether the object holds a value pypdf's parser reads differently from core's: a
        name running into a NUL byte, which pypdf reads as part of the name, or a string
        escape whose backslash pypdf keeps.
        """
        offset = getattr(entry, "offset", None)
        if not isinstance(offset, int):
            return False
        data = self._data
        end = OBJECT_VALUES_END.search(data, offset)
        stop = len(data) if end is None else end.start()
        if self._holds_nul and NAME_INTO_NUL.search(data, offset, stop) is not None:
            return True
        return data.find(b"\\", offset, stop) >= 0 and (
            PYPDF_ONLY_ESCAPE.search(data, offset, stop) is not None
        )

    def _repaired_object(self, number: int, generation: int, error: Exception | None) -> Any:
        """The object as pypdf's get_object reads it, when core cannot parse the one at its
        offset: pypdf's own xref offset, then the first `number generation obj` in the file,
        then wherever its failed header read left off, with its messages in strict mode.
        """
        offset = self.xref.get(generation, {}).get(number)
        if not isinstance(offset, int):
            return self._found_object(number, generation)
        stream = BytesIO(self._data)
        stream.seek(offset)
        try:
            found_number, found_generation = self.read_object_header(stream)
            if (found_number, found_generation) != (number, generation):
                raise PdfReadError("Not matching, we parse the file for it")
        except Exception:  # noqa: BLE001
            match = re.search(rf"\s{number}\s+{generation}\s+obj".encode(), self._data)
            if match is not None:
                logger.warning("Object ID %d,%d ref repaired", number, generation)
                self.xref[generation][number] = match.start(0) + 1
                stream.seek(match.start(0) + 1)
                found_number, found_generation = self.read_object_header(stream)
            else:
                found_number = found_generation = -1
        if found_number != number and getattr(self, "xref_index", False):
            if self.strict:
                raise PdfReadError(
                    f"Expected object ID ({number} {generation}) does not match actual "
                    f"({found_number} {found_generation}); xref table not zero-indexed."
                ) from error
        elif found_number != number and self.strict:
            raise PdfReadError(
                f"Expected object ID ({number} {generation}) does not match actual "
                f"({found_number} {found_generation})."
            ) from error
        current = (number, generation)
        if current in self._known_objects:
            # As in pypdf, an object stays known once its read fails.
            reference = IndirectObject(number, generation, self)
            raise LimitReachedError(f"Detected loop with self reference for {reference!r}.")
        self._known_objects.add(current)
        found = read_object(stream, self)
        self._known_objects.remove(current)
        return self._decrypted_as_pypdf(found, number, generation)

    def _found_object(self, number: int, generation: int) -> Any:
        """pypdf's reading of an object its xref does not list: the first `number generation
        obj` in the file, or nothing.
        """
        match = re.search(rf"\s{number}\s+{generation}\s+obj".encode(), self._data)
        if match is None:
            logger.warning("Object %d %d not defined.", number, generation)
            if self.strict:
                raise PdfReadError("Could not find object.")
            return None
        logger.warning("Object %d %d found", number, generation)
        self.xref.setdefault(generation, {})[number] = match.start(0) + 1
        stream = BytesIO(self._data)
        stream.seek(match.end(0) + 1)
        skip_over_whitespace(stream)
        stream.seek(-1, 1)
        found = read_object(stream, self)
        return self._decrypted_as_pypdf(found, number, generation)

    def _read_with_pypdf_rules(self, entry: Any) -> Any:
        offset = getattr(entry, "offset", None)
        header = INDIRECT_HEADER.match(self._data, offset) if isinstance(offset, int) else None
        if header is None:
            return None
        stream = BytesIO(self._data)
        stream.seek(header.end())
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        return read_object(stream, self)

    def _with_decode_references(self, stream: PdfStream, entry: Any) -> PdfStream:
        """Put back the references core resolves in /Filter and its siblings while reading,
        nested ones too, such as /JBIG2Globals in /DecodeParms.
        """
        offset = getattr(entry, "offset", None)
        if not isinstance(offset, int) or self._pdf is None:
            return stream
        keyword = stream_keyword(self._data, offset)
        if (
            keyword is None
            or DECODE_KEY_REFERENCE.search(self._data, offset, keyword.start()) is None
        ):
            return stream
        header = INDIRECT_HEADER.match(self._data, offset)
        if header is None:
            return stream
        lexer = RecoveryLexer(self._data)
        lexer.rewind(lexer.skip_ignored_at(header.end()))
        try:
            # The dictionary alone, with its references as they are in the file.
            raw = lexer.parse_dictionary()
        except PdfParseError:
            return stream
        if not isinstance(raw, dict):
            return stream
        restored = dict(stream.dictionary)
        keys = {str(getattr(key, "value", key)): key for key in restored}
        changed = False
        for key, value in raw.items():
            name = str(getattr(key, "value", key))
            if name in DECODE_KEYS and name in keys and _holds_reference(value):
                restored[keys[name]] = value
                changed = True
        return stream.replace(dictionary=restored) if changed else stream

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
        # pypdf's: white-space before the header is tolerated, and reported when strict.
        skip_over_comment(stream)
        extra = skip_over_whitespace(stream)
        stream.seek(-1, 1)
        number = read_until_whitespace(stream)
        extra |= skip_over_whitespace(stream)
        stream.seek(-1, 1)
        generation = read_until_whitespace(stream)
        extra |= skip_over_whitespace(stream)
        stream.seek(-1, 1)
        stream.read(3)
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        if extra and self.strict:
            logger.warning(
                "Superfluous whitespace found in object header %r %r", number, generation
            )
        return int(number), int(generation)

    def _replace_object(self, indirect: IndirectObject, obj: Any) -> Any:
        if indirect.pdf is not self:
            raise ValueError("Cannot update PdfReader with external object")
        key = (indirect.generation, indirect.idnum)
        if key not in self.resolved_objects:
            if indirect.idnum not in self.xref.get(indirect.generation, ()):
                raise ValueError("Cannot find referenced object")
            self.get_object(indirect)
        self.resolved_objects[key] = obj
        obj.indirect_reference = indirect
        return obj

    @property
    def _info(self) -> Any:
        info = self.trailer.get_resolved("/Info")
        return info if isinstance(info, DictionaryObject) else None

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
        # pypdf's trailer is what its walk over the cross-reference data found, even where
        # core recovered more.
        return self.__dict__.get("_generic_trailer", self._scan_trailer)

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
            if number > self._root_object_recovery_limit:
                raise LimitReachedError("Maximum Root object recovery limit reached.")
            try:
                candidate = self.get_object(number)
            except Exception:  # noqa: BLE001 - pypdf skips whatever fails here
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
        return [index2label(self, index) for index in range(len(self.pages))]

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
        else:
            pdf.close()
        return PASSWORD_TYPES[kind]


PASSWORD_TYPES = {"user": PasswordType.USER_PASSWORD, "owner": PasswordType.OWNER_PASSWORD}


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
