"""pypdf's object types, converted to and from core-pdf's parsed objects.

Bytes are always written by core_pdf_spec's serializer, so there is one writer.
"""

from __future__ import annotations

import codecs
import hashlib
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Self, cast

from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_compat.pypdf.errors import (
    DeprecationError,
    LimitReachedError,
    PdfReadError,
    PdfStreamError,
)
from core_pdf_spec.s_07_filters.codecs import encode_flate
from core_pdf_spec.s_07_syntax.serialize import (
    ObjectLayout,
    serialize_name,
    serialize_object,
    serialize_real,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream

logger = logging.getLogger(__name__)


# clone, replicate and the hash functions follow pypdf, Copyright (c) 2006, Mathieu Fenniak,
# under its BSD-style license.
class PdfObject:
    hash_func: Callable[..., Any] = hashlib.sha1
    indirect_reference: IndirectObject | None = None

    def hash_bin(self) -> int:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement .hash_bin() so far"
        )

    def hash_value_data(self) -> bytes:
        return f"{self}".encode()

    def hash_value(self) -> bytes:
        digest = self.hash_func(self.hash_value_data()).hexdigest()
        return f"{self.__class__.__name__}:{digest}".encode()

    def replicate(self, pdf_dest: Any) -> Any:
        return self.clone(pdf_dest)

    def clone(
        self,
        pdf_dest: Any,
        force_duplicate: bool = False,
        ignore_fields: Sequence[str | int] | None = (),
    ) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement .clone so far")

    def _own_reference(self) -> Any:
        return self.__dict__.get("indirect_reference", _NEVER_NUMBERED)

    def _reference_clone(self, clone: Any, pdf_dest: Any, force_duplicate: bool = False) -> Any:
        """Number a clone in pdf_dest, if this object has been given a number at all."""
        target = clone.indirect_reference
        if not force_duplicate and target is not None and target.pdf == pdf_dest:
            return clone
        reference = self._own_reference()
        if reference is _NEVER_NUMBERED:
            return clone
        number = len(pdf_dest._objects) + 1
        if reference is not None:
            translated = pdf_dest._translation(reference.pdf)
            if not force_duplicate and reference.idnum in translated:
                return pdf_dest.get_object(translated[reference.idnum])
            translated[reference.idnum] = number
        pdf_dest._objects.append(clone)
        clone.indirect_reference = IndirectObject(number, 0, pdf_dest)
        return clone

    def get_object(self) -> PdfObject | None:
        return self

    def write_to_stream(self, stream: Any, encryption_key: object = None) -> None:
        if encryption_key is not None:
            raise DeprecationError(
                "the encryption_key parameter of write_to_stream is deprecated and was "
                "removed in pypdf 5.0.0."
            )
        stream.write(serialize_object(to_core(self), ObjectLayout.SPACED))


_NEVER_NUMBERED = object()


class _CopiedObject(PdfObject):
    """An object with no children, so a clone is just a copy of its value."""

    def _copy(self) -> Any:
        raise NotImplementedError

    def clone(
        self,
        pdf_dest: Any,
        force_duplicate: bool = False,
        ignore_fields: Sequence[str | int] | None = (),
    ) -> Any:
        del ignore_fields
        return self._reference_clone(self._copy(), pdf_dest, force_duplicate)


class NullObject(_CopiedObject):
    def _copy(self) -> Self:
        return type(self)()

    def hash_bin(self) -> int:
        return hash((self.__class__,))

    @staticmethod
    def read_from_stream(stream: Any) -> NullObject:
        return read_null(stream)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NullObject)

    def __hash__(self) -> int:
        return self.hash_bin()

    def __repr__(self) -> str:
        return "NullObject"


class BooleanObject(_CopiedObject):
    def __init__(self, value: Any) -> None:
        self.value = bool(value)

    def _copy(self) -> Self:
        return type(self)(self.value)

    def hash_bin(self) -> int:
        return hash((self.__class__, self.value))

    @staticmethod
    def read_from_stream(stream: Any) -> BooleanObject:
        return read_boolean(stream)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BooleanObject):
            return self.value == other.value
        return self.value == other if isinstance(other, bool) else False

    def __hash__(self) -> int:
        return self.hash_bin()

    def __bool__(self) -> bool:
        return self.value

    def __repr__(self) -> str:
        return "True" if self.value else "False"


class NumberObject(int, _CopiedObject):
    NumberPattern = re.compile(b"[^+-.0-9]")
    _LENGTH_LIMIT = 64

    def __new__(cls, value: Any) -> Self:
        try:
            return int.__new__(cls, int(value))
        except ValueError:
            logger.warning("NumberObject(%s) invalid; use 0 instead", value)
            return int.__new__(cls, 0)

    def _copy(self) -> Self:
        return type(self)(self)

    def hash_bin(self) -> int:
        return hash((self.__class__, int(self)))

    def as_numeric(self) -> int:
        return int(self)

    @staticmethod
    def read_from_stream(stream: Any) -> NumberObject | FloatObject:
        return read_number(stream)


class FloatObject(float, _CopiedObject):
    def __new__(cls, value: Any = "0.0", context: object = None) -> Self:
        del context
        try:
            return super().__new__(cls, value)
        except (TypeError, ValueError) as error:
            # Malformed files hold values such as a reference where a number belongs.
            logger.warning("%s : FloatObject (%s) invalid; use 0.0 instead", error, value)
            return super().__new__(cls, 0.0)

    def _copy(self) -> Self:
        return type(self)(self)

    def hash_bin(self) -> int:
        return hash((self.__class__, float(self)))

    def myrepr(self) -> str:
        # pypdf rounds to about nine significant digits here. This keeps the exact value, so
        # what is shown is what the serializer writes.
        return "0.0" if self == 0 else serialize_real(float(self)).decode("ascii")

    def __repr__(self) -> str:
        return self.myrepr()

    def as_numeric(self) -> float:
        return float(self)


class _NameMeta(type):
    @property
    def surfix(cls) -> bytes:
        raise DeprecationError(
            "surfix is deprecated and was removed in pypdf 5.0.0. Use prefix instead."
        )


class NameObject(str, _CopiedObject, metaclass=_NameMeta):
    delimiter_pattern = re.compile(rb"\s+|[\(\)<>\[\]{}/%]")
    prefix = b"/"
    renumber_table: ClassVar[dict[str, bytes]] = {
        **{chr(i): f"#{i:02X}".encode() for i in b"#()<>[]{}/%"},
        **{chr(i): f"#{i:02X}".encode() for i in range(33)},
    }
    CHARSETS = ("utf-8", "gbk", "latin1")
    _LENGTH_LIMIT = 4096
    # The bytes a name had in its file. Text decoded by CHARSETS may not encode back to them.
    _file_bytes: bytes | None = None

    def __new__(cls, value: str = "/") -> NameObject:
        return super().__new__(cls, value)

    def _copy(self) -> NameObject:
        copy = NameObject(self)
        copy._file_bytes = self._file_bytes
        return copy

    def hash_bin(self) -> int:
        return hash((self.__class__, self))

    def renumber(self) -> bytes:
        return serialize_name(_name_bytes(self))

    def _sanitize(self) -> NameObject:
        name = str(self).removeprefix("/")
        return NameObject("/" + re.sub(r"[^a-zA-Z0-9_-]", "_", name))

    @staticmethod
    def unnumber(sin: bytes) -> bytes:
        return NUMBERED_BYTE.sub(lambda match: bytes((int(match[1], 16),)), sin)

    @staticmethod
    def read_from_stream(stream: Any, pdf: Any) -> NameObject:
        return read_name(stream, pdf)


NUMBERED_BYTE = re.compile(rb"#([0-9a-fA-F]{2})")


def _name_bytes(name: str) -> bytes:
    """A name's bytes without its slash. Written names are UTF-8, as in pypdf."""
    if not name.startswith("/"):
        raise DeprecationError(
            f"Incorrect first char in NameObject, should start with '/': ({name}) is "
            "deprecated and was removed in pypdf 5.0.0."
        )
    file_bytes = getattr(name, "_file_bytes", None)
    return name[1:].encode("utf-8") if file_bytes is None else file_bytes


class TextStringObject(str, _CopiedObject):
    autodetect_pdfdocencoding: bool
    autodetect_utf16: bool
    utf16_bom: bytes
    _original_bytes: bytes | None

    def __new__(cls, value: Any) -> Self:
        original = value if isinstance(value, bytes) else None
        text = super().__new__(cls, value.decode("charmap") if original is not None else value)
        text._original_bytes = original
        text.autodetect_utf16 = False
        text.autodetect_pdfdocencoding = False
        text.utf16_bom = b""
        if original is not None and original[:2] in {codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE}:
            try:
                decoded = original.decode("utf-16")
            except UnicodeDecodeError as error:
                logger.warning("%s; initial string: %r", error, error.object)
                decoded = error.object[: error.start].decode("utf-16")
            text = super().__new__(cls, decoded)
            text._original_bytes = original
            text.autodetect_pdfdocencoding = False
            text.autodetect_utf16 = True
            text.utf16_bom = original[:2]
            return text
        try:
            encode_pdfdocencoding(text)
        except UnicodeEncodeError:
            text.autodetect_utf16 = True
            text.utf16_bom = codecs.BOM_UTF16_BE
        else:
            text.autodetect_pdfdocencoding = True
        return text

    def _copy(self) -> TextStringObject:
        copy = TextStringObject(str(self))
        copy._original_bytes = self._original_bytes
        copy.autodetect_pdfdocencoding = self.autodetect_pdfdocencoding
        copy.autodetect_utf16 = self.autodetect_utf16
        copy.utf16_bom = self.utf16_bom
        return copy

    def hash_bin(self) -> int:
        return hash((self.__class__, self.original_bytes))

    @property
    def original_bytes(self) -> bytes:
        return self.get_original_bytes() if self._original_bytes is None else self._original_bytes

    def _encode_utf16(self) -> bytes:
        if self.utf16_bom == codecs.BOM_UTF16_LE:
            return codecs.BOM_UTF16_LE + self.encode("utf-16le")
        if self.utf16_bom == codecs.BOM_UTF16_BE:
            return codecs.BOM_UTF16_BE + self.encode("utf-16be")
        return self.encode("utf-16be")

    def get_original_bytes(self) -> bytes:
        return self._encode_utf16() if self.autodetect_utf16 else encode_pdfdocencoding(self)

    def get_encoded_bytes(self) -> bytes:
        if self._original_bytes is not None:
            return self._original_bytes
        if self.autodetect_utf16:
            return self._encode_utf16()
        try:
            return encode_pdfdocencoding(self)
        except UnicodeEncodeError:
            return self._encode_utf16()


class ByteStringObject(bytes, _CopiedObject):
    def _copy(self) -> ByteStringObject:
        return ByteStringObject(bytes(self))

    def hash_bin(self) -> int:
        return hash((self.__class__, bytes(self)))

    @property
    def original_bytes(self) -> bytes:
        return bytes(self)

    def __str__(self) -> str:
        for charset in ("utf-16", *NameObject.CHARSETS):
            try:
                return self.decode(charset)
            except UnicodeDecodeError:
                continue
        raise PdfReadError("Cannot decode ByteStringObject.")


class ArrayObject(list[Any], PdfObject):
    def replicate(self, pdf_dest: Any) -> ArrayObject:
        array = self._reference_clone(ArrayObject(), pdf_dest, False)
        for item in self:
            array.append(item.replicate(pdf_dest) if hasattr(item, "replicate") else item)
        return array

    def clone(
        self,
        pdf_dest: Any,
        force_duplicate: bool = False,
        ignore_fields: Sequence[str | int] | None = (),
    ) -> ArrayObject:
        if _in_destination(self, pdf_dest, force_duplicate):
            return self
        array = self._reference_clone(ArrayObject(), pdf_dest, True)
        for item in self:
            array.append(_cloned_item(item, pdf_dest, force_duplicate, ignore_fields))
        return array

    def hash_bin(self) -> int:
        return hash((self.__class__, tuple(item.hash_bin() for item in self)))

    def items(self) -> Iterable[Any]:
        return enumerate(self)

    @staticmethod
    def _to_lst(items: Any) -> list[Any]:
        if isinstance(items, (list, tuple, set)):
            return list(items)
        if isinstance(items, PdfObject):
            return [items]
        if isinstance(items, str):
            return [NameObject(items) if items[0] == "/" else TextStringObject(items)]
        if isinstance(items, bytes):
            return [ByteStringObject(items)]
        return [items]

    def __add__(self, items: Any) -> ArrayObject:
        output = ArrayObject(self)
        output.extend(self._to_lst(items))
        return output

    def __iadd__(self, items: Any) -> Self:
        self.extend(self._to_lst(items))
        return self

    def __isub__(self, items: Any) -> Self:
        for item in self._to_lst(items):
            if item in self:
                del self[self.index(item)]
        return self

    @staticmethod
    def read_from_stream(stream: Any, pdf: Any, forced_encoding: Any = None) -> ArrayObject:
        return read_array(stream, pdf, forced_encoding)


def _in_destination(value: Any, pdf_dest: Any, force_duplicate: bool) -> bool:
    reference = value.indirect_reference
    return not force_duplicate and reference is not None and reference.pdf == pdf_dest


def _cloned_item(item: Any, pdf_dest: Any, force_duplicate: bool, ignore_fields: Any) -> Any:
    if isinstance(item, StreamObject):
        copy = item.clone(pdf_dest, force_duplicate, ignore_fields)
        return item._reference_clone(copy, pdf_dest, force_duplicate).indirect_reference
    if isinstance(item, IndirectObject) and isinstance(target := item.get_object(), StreamObject):
        # Each array gets its own copy, so replacing one stream leaves the other arrays alone.
        copy = target.clone(pdf_dest, True, ignore_fields)
        return item._reference_clone(copy, pdf_dest, force_duplicate).indirect_reference
    return item.clone(pdf_dest, force_duplicate, ignore_fields) if hasattr(item, "clone") else item


# Keys such as /%is_open% hold pypdf's own state, and are never written.
def _is_internal_key(key: str) -> bool:
    return len(key) > 2 and key[1] == "%" and key[-1] == "%"


def _trimmed_ignore_fields(ignore_fields: Any) -> list[Any]:
    """Count down the entries limited to some levels: [1, "/B"] ignores /B one level down."""
    fields = list(ignore_fields)
    index = 0
    while index < len(fields):
        if isinstance(fields[index], int):
            if fields[index] <= 0:
                del fields[index : index + 2]
                continue
            fields[index] -= 1
        index += 1
    return fields


def _same_type(first: DictionaryObject, second: DictionaryObject) -> bool:
    kinds = first.get("/Type"), second.get("/Type")
    return None in kinds or kinds[0] == kinds[1]


def _is_chain_link(source: DictionaryObject, key: str) -> bool:
    return (
        key in source
        and isinstance(source.raw_get(key), IndirectObject)
        and isinstance(source[key], DictionaryObject)
        and _same_type(source, source[key])
    )


class DictionaryObject(dict[Any, Any], PdfObject):
    def replicate(self, pdf_dest: Any) -> DictionaryObject:
        copy = self._reference_clone(self.__class__(), pdf_dest, False)
        for key, value in dict.items(self):
            copy[key.replicate(pdf_dest)] = (
                value.replicate(pdf_dest) if hasattr(value, "replicate") else value
            )
        return copy

    def clone(
        self,
        pdf_dest: Any,
        force_duplicate: bool = False,
        ignore_fields: Sequence[str | int] | None = (),
    ) -> DictionaryObject:
        if _in_destination(self, pdf_dest, force_duplicate):
            return self
        copy = self._reference_clone(self.__class__(), pdf_dest, force_duplicate)
        if len(copy) == 0:
            copy._clone(self, pdf_dest, force_duplicate, ignore_fields or [], set())
        return copy

    def _clone_chain(
        self,
        source: DictionaryObject,
        key: str,
        pdf_dest: Any,
        force_duplicate: bool,
        ignore_fields: Any,
        visited: set[tuple[int, int]],
    ) -> None:
        """Follow a linked list in a loop, which a long outline would overflow by recursion."""
        pairs: list[tuple[Any, Any]] = []
        current: Any = source[key]
        previous: Any = self
        while current is not None:
            copy = current._reference_clone(current.__class__(), pdf_dest, force_duplicate)
            reference = copy.indirect_reference
            if reference is not None:
                if (reference.idnum, reference.generation) in visited:
                    break
                visited.add((reference.idnum, reference.generation))
            pairs.append((current, copy))
            previous[NameObject(key)] = reference
            previous = copy
            following = None if current == source else current.get_resolved(key)
            current = following if isinstance(following, DictionaryObject) else None
        for original, copy in pairs:
            copy._clone(original, pdf_dest, force_duplicate, ignore_fields, visited)

    def _clone(
        self,
        src: DictionaryObject,
        pdf_dest: Any,
        force_duplicate: bool,
        ignore_fields: Any,
        visited: set[tuple[int, int]],
    ) -> None:
        ignored = _trimmed_ignore_fields(ignore_fields)
        for key in ("/Next", "/Prev", "/N", "/V"):
            if key not in ignored and key not in self and _is_chain_link(src, key):
                self._clone_chain(src, key, pdf_dest, force_duplicate, ignored, visited)
        for key, value in dict.items(src):
            if key in ignored:
                continue
            if isinstance(value, StreamObject):
                value.__dict__.setdefault("indirect_reference", None)
                copy = value.clone(pdf_dest, force_duplicate, ignored)
                self[key.clone(pdf_dest)] = copy.indirect_reference
            elif key not in self:
                self[key._copy() if isinstance(key, NameObject) else NameObject(key)] = (
                    value.clone(pdf_dest, force_duplicate, ignored)
                    if hasattr(value, "clone")
                    else value
                )

    def hash_bin(self) -> int:
        entries = tuple((key, value.hash_bin()) for key, value in dict.items(self))
        return hash((self.__class__, entries))

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        return value.get_object() if isinstance(value, IndirectObject) else value

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(key, PdfObject):
            raise ValueError("Key must be a PdfObject")
        if not isinstance(value, PdfObject):
            raise ValueError("Value must be a PdfObject")
        dict.__setitem__(self, key, value)

    def setdefault(self, key: Any, value: Any = None) -> Any:
        if not isinstance(key, PdfObject):
            raise ValueError("Key must be a PdfObject")
        if not isinstance(value, PdfObject):
            raise ValueError("Value must be a PdfObject")
        return dict.setdefault(self, key, value)

    def raw_get(self, key: Any) -> Any:
        return super().__getitem__(key)

    @classmethod
    def _get_next_object_position(
        cls, position_before: int, position_end: int, generations: list[int], pdf: Any
    ) -> int:
        positions = (
            position
            for generation in generations
            for position in pdf.xref[generation].values()
            if position_before < position <= position_end
        )
        return min(positions, default=position_end)

    @classmethod
    def _read_unsized_from_stream(cls, *, stream: Any, pdf: Any, length: int) -> bytes:
        return read_unsized(stream, pdf, length)

    @staticmethod
    def read_from_stream(stream: Any, pdf: Any, forced_encoding: Any = None) -> DictionaryObject:
        return read_dictionary(stream, pdf, forced_encoding)

    @property
    def xmp_metadata(self) -> Any:
        from .xmp import XmpInformation

        metadata = self.get_resolved("/Metadata")
        return None if is_null_or_none(metadata) else XmpInformation(metadata)

    def get_resolved(self, key: Any, default: Any = None) -> Any:
        # pypdf's get returns references as stored, which a writer needs. Readers that want
        # the object use this.
        return self[key] if key in self else default  # noqa: SIM401

    def get_inherited(self, key: str, default: Any = None) -> Any:
        """A value from this dictionary or the nearest /Parent that has it."""
        visited: set[int] = set()
        node: Any = self
        while True:
            if id(node) in visited:
                raise LimitReachedError(
                    f"Detected cycle in /Parent hierarchy when retrieving value for key {key!r}."
                )
            visited.add(id(node))
            if key in node:
                return node[key]
            if "/Parent" not in node:
                return default
            node = node["/Parent"]
            if not isinstance(node, DictionaryObject):
                return default


# The stream classes' behaviour is ported from pypdf, Copyright (c) 2006-2008, Mathieu
# Fenniak, under its BSD-style license. core_pdf_spec does the decoding and encoding.
class StreamObject(DictionaryObject):
    def __init__(self) -> None:
        super().__init__()
        self._data = b""
        self.decoded_self: DecodedStreamObject | None = None

    def replicate(self, pdf_dest: Any) -> StreamObject:
        copy = self._reference_clone(self.__class__(), pdf_dest, False)
        copy._data = self._data
        self._replicate_decoded(pdf_dest)
        for key, value in dict.items(self):
            copy[key.replicate(pdf_dest)] = (
                value.replicate(pdf_dest) if hasattr(value, "replicate") else value
            )
        return copy

    def _replicate_decoded(self, pdf_dest: Any) -> None:
        decoded = getattr(self, "decoded_self", None)
        self.decoded_self = None if decoded is None else decoded.replicate(pdf_dest)

    def _clone(
        self,
        src: DictionaryObject,
        pdf_dest: Any,
        force_duplicate: bool,
        ignore_fields: Any,
        visited: set[tuple[int, int]],
    ) -> None:
        source = cast("StreamObject", src)
        self._data = source._data
        decoded = getattr(source, "decoded_self", None)
        self.decoded_self = (
            None if decoded is None else decoded.clone(pdf_dest, force_duplicate, ignore_fields)
        )
        super()._clone(src, pdf_dest, force_duplicate, ignore_fields, visited)

    def hash_bin(self) -> int:
        return hash((super().hash_bin(), self._data))

    def hash_value_data(self) -> bytes:
        return super().hash_value_data() + self.get_data()

    def get_data(self) -> bytes:
        return self._data

    def set_data(self, data: bytes) -> None:
        self._data = data

    def _stored_data(self) -> bytes:
        """The bytes a file holds for this stream, still encoded by its /Filter."""
        return self._data

    def decode_as_image(self, pillow_parameters: dict[str, Any] | None = None) -> Any:
        from core_pdf_compat.pypdf._images import xobject_image

        if self.get("/Subtype", "") != "/Image":
            logger.warning("%s does not seem to be an Image", self.indirect_reference)
        return xobject_image(self, pillow_parameters)[2]

    @staticmethod
    def initialize_from_dictionary(
        data: dict[str, Any],
    ) -> EncodedStreamObject | DecodedStreamObject:
        stream = EncodedStreamObject() if "/Filter" in data else DecodedStreamObject()
        stream._data = data.pop("__streamdata__")
        data.pop("/Length", None)
        dict.update(stream, data)
        return stream

    def flate_encode(self, level: int = -1) -> EncodedStreamObject:
        output = EncodedStreamObject()
        dict.update(output, self)
        flate = NameObject("/FlateDecode")
        if "/Filter" in self:
            filters = self["/Filter"]
            names = list(filters) if isinstance(filters, ArrayObject) else [filters]
            parms = self.get("/DecodeParms", NullObject())
            parms = list(parms) if isinstance(parms, ArrayObject) else [parms] * len(names)
            output[NameObject("/Filter")] = ArrayObject([flate, *names])
            output[NameObject("/DecodeParms")] = ArrayObject([NullObject(), *parms])
        else:
            output[NameObject("/Filter")] = flate
        output._data = encode_flate(self._stored_data(), level)
        return output


class DecodedStreamObject(StreamObject):
    pass


class EncodedStreamObject(StreamObject):
    def get_data(self) -> bytes:
        if self.decoded_self is None:
            decoded = DecodedStreamObject()
            decoded.set_data(decode_stream(self))
            for key, value in dict.items(self):
                if key not in {"/Length", "/Filter", "/DecodeParms"}:
                    dict.__setitem__(decoded, key, value)
            self.decoded_self = decoded
        return self.decoded_self.get_data()

    def set_data(self, data: bytes) -> None:
        if self.get("/Filter", "") not in ("/FlateDecode", ["/FlateDecode"]):
            raise PdfReadError(
                "Streams encoded with a filter different from FlateDecode are not supported"
            )
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        self.get_data()
        assert self.decoded_self is not None
        self.decoded_self.set_data(data)
        # The new data has no predictor applied, so the old parameters would corrupt it.
        self.pop("/DecodeParms", None)
        self._data = encode_flate(data)


class IndirectObject(PdfObject):
    def __init__(self, idnum: int, generation: int, pdf: Any) -> None:
        self.idnum = idnum
        self.generation = generation
        self.pdf = pdf

    def _own_reference(self) -> IndirectObject:
        return self

    def hash_bin(self) -> int:
        return hash((self.__class__, self.idnum, self.generation, id(self.pdf)))

    def replicate(self, pdf_dest: Any) -> IndirectObject:
        return IndirectObject(self.idnum, self.generation, pdf_dest)

    def clone(
        self,
        pdf_dest: Any,
        force_duplicate: bool = False,
        ignore_fields: Sequence[str | int] | None = (),
    ) -> IndirectObject:
        if self.pdf == pdf_dest and not force_duplicate:
            return self
        translated = pdf_dest._translation(self.pdf)
        if self.idnum in translated:
            copy = pdf_dest.get_object(translated[self.idnum])
            reference = copy.indirect_reference
            if force_duplicate:
                return IndirectObject(reference.idnum, reference.generation, reference.pdf)
            return reference
        target = self.get_object()
        if target is None:
            # A reference to a missing object is a reference to null (ISO 32000-2 7.3.10).
            target = NullObject()
            target.indirect_reference = self
        copy = target.clone(pdf_dest, force_duplicate, ignore_fields)
        return pdf_dest._add_object(copy)

    def get_object(self) -> Any:
        return self.pdf.get_object(self)

    def _get_object_with_check(self) -> Any:
        target = self.get_object()
        if isinstance(target, IndirectObject):
            raise PdfStreamError(f"{self!r} references an IndirectObject {target!r}")
        return target

    def __deepcopy__(self, memo: Any) -> IndirectObject:
        # A copied reference points at the same document, which is not copied.
        return IndirectObject(self.idnum, self.generation, self.pdf)

    @property
    def indirect_reference(self) -> IndirectObject:  # type: ignore[override]  # ty: ignore[invalid-property-type-override]
        return self

    def __hash__(self) -> int:
        return hash((self.idnum, self.generation, id(self.pdf)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IndirectObject)
            and self.idnum == other.idnum
            and self.generation == other.generation
            and self.pdf is other.pdf
        )

    def __repr__(self) -> str:
        return f"IndirectObject({self.idnum!r}, {self.generation!r}, {id(self.pdf)})"

    def __str__(self) -> str:
        return str(self.get_object())

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._get_object_with_check(), name)
        except AttributeError:
            raise AttributeError(
                f"No attribute {name} found in IndirectObject or pointed object"
            ) from None

    def __float__(self) -> float:
        return float(self.get_object())

    def __int__(self) -> int:
        return int(self.get_object())

    @staticmethod
    def read_from_stream(stream: Any, pdf: Any) -> IndirectObject:
        return read_indirect(stream, pdf)

    def __getitem__(self, key: Any) -> Any:
        return self._get_object_with_check()[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._get_object_with_check()

    def __iter__(self) -> Any:
        return iter(self._get_object_with_check())


def _number(value: Any) -> FloatObject | NumberObject:
    return value if isinstance(value, (FloatObject, NumberObject)) else FloatObject(value)


def _side(index: int) -> property:
    def setter(self: RectangleObject, value: float) -> None:
        self[index] = FloatObject(value)

    return property(lambda self: self[index], setter)


def _corner(horizontal: int, vertical: int) -> property:
    def setter(self: RectangleObject, value: tuple[float, float]) -> None:
        self[horizontal], self[vertical] = _number(value[0]), _number(value[1])

    return property(lambda self: (self[horizontal], self[vertical]), setter)


class RectangleObject(ArrayObject):
    def __init__(self, values: Iterable[Any]) -> None:
        items = list(values)
        if len(items) != 4:
            raise ValueError("a rectangle needs four values")
        super().__init__(_number(item) for item in items)

    def scale(self, sx: float, sy: float) -> RectangleObject:
        return RectangleObject(
            (
                float(self.left) * sx,
                float(self.bottom) * sy,
                float(self.right) * sx,
                float(self.top) * sy,
            )
        )

    def __repr__(self) -> str:
        return f"RectangleObject({list(self)!r})"

    left = _side(0)
    bottom = _side(1)
    right = _side(2)
    top = _side(3)
    lower_left = _corner(0, 1)
    lower_right = _corner(2, 1)
    upper_left = _corner(0, 3)
    upper_right = _corner(2, 3)
    width = property(lambda self: self.right - self.left)
    height = property(lambda self: self.top - self.bottom)


def is_null_or_none(value: object) -> bool:
    if isinstance(value, IndirectObject):
        value = value.get_object()
    return value is None or isinstance(value, NullObject)


def _decoded_text(text: str, original: bytes) -> TextStringObject:
    output = TextStringObject(text)
    output._original_bytes = original
    return output


def _forced_text(data: bytes, encoding: str | list[str] | dict[int, str]) -> Any:
    if encoding == "bytes":
        return ByteStringObject(data)
    if isinstance(encoding, str):
        return _decoded_text(data.decode(encoding), data)

    def character(code: int) -> str:
        try:
            return encoding[code]
        except IndexError, KeyError:
            return bytes((code,)).decode("charmap")

    return _decoded_text("".join(map(character, data)), data)


def create_string_object(
    string: str | bytes, forced_encoding: str | list[str] | dict[int, str] | None = None
) -> TextStringObject | ByteStringObject:
    if isinstance(string, str):
        return TextStringObject(string)
    if not isinstance(string, bytes):
        raise TypeError("create_string_object should have str or unicode arg")
    if forced_encoding is not None:
        return _forced_text(string, forced_encoding)
    try:
        if string.startswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)):
            output = _decoded_text(string.decode("utf-16"), string)
            output.utf16_bom = string[:2]
        elif string.startswith(b"\x00"):
            output = _decoded_text(string.decode("utf-16be"), string)
            output.utf16_bom = codecs.BOM_UTF16_BE
        elif string[1:2] == b"\x00":
            output = _decoded_text(string.decode("utf-16le"), string)
            output.utf16_bom = codecs.BOM_UTF16_LE
        else:
            output = _decoded_text(decode_pdfdocencoding(string), string)
            output.autodetect_pdfdocencoding, output.autodetect_utf16 = True, False
            return output
    except UnicodeDecodeError:
        return ByteStringObject(string)
    output.autodetect_utf16, output.autodetect_pdfdocencoding = True, False
    return output


def to_generic(value: Any, pdf: Any = None) -> Any:
    """Present a core-pdf object as a pypdf one, leaving references unresolved."""
    if value is None:
        return NullObject()
    if isinstance(value, bool):
        return BooleanObject(value)
    if isinstance(value, int):
        return NumberObject(value)
    if isinstance(value, float):
        return FloatObject(value)
    if isinstance(value, (PdfName, str)):
        return _generic_key(value)
    if isinstance(value, PdfString):
        return create_string_object(bytes(value.data))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return create_string_object(bytes(value))
    if isinstance(value, PdfReference):
        return IndirectObject(value.object_number, value.generation_number, pdf)
    if isinstance(value, PdfStream):
        items: dict[str, Any] = {
            _generic_key(key): to_generic(item, pdf) for key, item in value.dictionary.items()
        }
        items["__streamdata__"] = bytes(value.raw_data)
        return StreamObject.initialize_from_dictionary(items)
    if isinstance(value, Mapping):
        output = DictionaryObject()
        for key, item in value.items():
            dict.__setitem__(output, _generic_key(key), to_generic(item, pdf))
        return output
    if isinstance(value, (list, tuple)):
        array = ArrayObject()
        array.extend(to_generic(item, pdf) for item in value)
        return array
    return value


def _generic_key(key: Any) -> NameObject:
    text = getattr(key, "value", key)
    if text.isascii():
        return NameObject(f"/{text}")
    data = key.value_bytes if isinstance(key, PdfName) else text.encode("latin-1")
    decoded = decode_name(data)
    name = NameObject(f"/{data.decode('charmap') if decoded is None else decoded}")
    name._file_bytes = data
    return name


def _core_key(key: str) -> Any:
    name = _name_bytes(key)
    return key[1:] if key.isascii() else PdfName(name)


def _core_dictionary(items: Iterable[tuple[Any, Any]]) -> dict[Any, Any]:
    return {_core_key(key): to_core(item) for key, item in items if not _is_internal_key(key)}


def to_core(value: Any) -> Any:
    """Convert a pypdf object for core-pdf's serializer."""
    if isinstance(value, (NullObject, type(None))):
        return None
    if isinstance(value, IndirectObject):
        return PdfReference(value.idnum, value.generation)
    # Destinations and outline items are written in a different form from the one they hold.
    core_form = getattr(value, "core_form", None)
    if core_form is not None:
        return to_core(core_form())
    if isinstance(value, BooleanObject):
        return value.value
    if isinstance(value, NumberObject):
        return int(value)
    if isinstance(value, FloatObject):
        return float(value)
    if isinstance(value, NameObject):
        return PdfName(_name_bytes(value))
    if isinstance(value, ByteStringObject):
        return PdfString(bytes(value), is_literal=False)
    if isinstance(value, str):
        return PdfString(_encode_text(value))
    if isinstance(value, StreamObject):
        return PdfStream(_core_dictionary(dict.items(value)), value._stored_data())
    if isinstance(value, dict):
        return _core_dictionary(dict.items(value))
    if isinstance(value, Mapping):
        return _core_dictionary(value.items())
    if isinstance(value, list):
        return [to_core(item) for item in list.__iter__(value)]
    return value


def _encode_text(value: str) -> bytes:
    text = value if isinstance(value, TextStringObject) else TextStringObject(value)
    return text.get_encoded_bytes()


from core_pdf_compat.pypdf._generic_content import ContentStream, decode_stream  # noqa: E402
from core_pdf_compat.pypdf._generic_read import (  # noqa: E402
    decode_name,
    read_array,
    read_boolean,
    read_dictionary,
    read_indirect,
    read_name,
    read_null,
    read_number,
    read_object,
    read_unsized,
)

__all__ = (
    "PAGE_FIT",
    "ArrayObject",
    "BooleanObject",
    "ByteStringObject",
    "ContentStream",
    "DecodedStreamObject",
    "Destination",
    "DictionaryObject",
    "EmbeddedFile",
    "EncodedStreamObject",
    "Field",
    "Fit",
    "FloatObject",
    "IndirectObject",
    "NameObject",
    "NullObject",
    "NumberObject",
    "OutlineFontFlag",
    "OutlineItem",
    "PdfObject",
    "RectangleObject",
    "StreamObject",
    "TextStringObject",
    "TreeObject",
    "ViewerPreferences",
    "create_string_object",
    "decode_pdfdocencoding",
    "encode_pdfdocencoding",
    "hex_to_rgb",
    "is_null_or_none",
    "read_hex_string_from_stream",
    "read_object",
    "read_string_from_stream",
    "to_core",
    "to_generic",
)

from ._generic_trees import (  # noqa: E402
    PAGE_FIT,
    Destination,
    EmbeddedFile,
    Field,
    Fit,
    OutlineFontFlag,
    OutlineItem,
    TreeObject,
    ViewerPreferences,
    decode_pdfdocencoding,
    encode_pdfdocencoding,
    hex_to_rgb,
    read_hex_string_from_stream,
    read_string_from_stream,
)
