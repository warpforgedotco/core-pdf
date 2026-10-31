"""pypdf's object types, converted to and from core-pdf's parsed objects.

Bytes are always written by core_pdf_spec's serializer, so there is one writer.
"""

from __future__ import annotations

import codecs
import logging
from collections.abc import Iterable, Mapping
from io import BytesIO
from typing import Any, Self

from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_compat.pypdf.errors import LimitReachedError, PdfReadError
from core_pdf_spec.s_07_filters.codecs import encode_flate
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.serialize import serialize_object
from core_pdf_spec.s_07_syntax.stream import PdfStream

logger = logging.getLogger(__name__)


class PdfObject:
    indirect_reference: IndirectObject | None = None

    def get_object(self) -> PdfObject | None:
        return self

    @classmethod
    def read_from_stream(cls, stream: Any, pdf: Any = None, *_: object) -> Any:
        return read_object(stream, pdf)

    def write_to_stream(self, stream: Any, encryption_key: object = None) -> None:
        del encryption_key
        stream.write(serialize_object(to_core(self)))


class NullObject(PdfObject):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, NullObject)

    def __hash__(self) -> int:
        return hash(None)

    def __repr__(self) -> str:
        return "NullObject"


class BooleanObject(PdfObject):
    def __init__(self, value: Any) -> None:
        self.value = bool(value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BooleanObject):
            return self.value == other.value
        return self.value == other if isinstance(other, bool) else False

    def __hash__(self) -> int:
        return hash(self.value)

    def __bool__(self) -> bool:
        return self.value

    def __repr__(self) -> str:
        return "True" if self.value else "False"


class NumberObject(int, PdfObject):
    def as_numeric(self) -> int:
        return int(self)


class FloatObject(float, PdfObject):
    def __new__(cls, value: Any = "0.0", context: object = None) -> Self:
        del context
        try:
            return super().__new__(cls, value)
        except (TypeError, ValueError) as error:
            # Malformed files hold values such as a reference where a number belongs.
            logger.warning("%s : FloatObject (%s) invalid; use 0.0 instead", error, value)
            return super().__new__(cls, 0.0)

    def as_numeric(self) -> float:
        return float(self)


class NameObject(str, PdfObject):
    __slots__ = ()

    def __new__(cls, value: str = "/") -> NameObject:
        return super().__new__(cls, value)


class TextStringObject(str, PdfObject):
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


class ByteStringObject(bytes, PdfObject):
    @property
    def original_bytes(self) -> bytes:
        return bytes(self)


class ArrayObject(list[Any], PdfObject):
    def __getitem__(self, index: Any) -> Any:
        value = super().__getitem__(index)
        return value.get_object() if isinstance(value, IndirectObject) else value

    def raw_get(self, index: int) -> Any:
        return super().__getitem__(index)


class DictionaryObject(dict[Any, Any], PdfObject):
    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        return value.get_object() if isinstance(value, IndirectObject) else value

    def raw_get(self, key: Any) -> Any:
        return super().__getitem__(key)

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

    def get_data(self) -> bytes:
        return self._data

    def set_data(self, data: bytes) -> None:
        self._data = data

    def _stored_data(self) -> bytes:
        """The bytes a file holds for this stream, still encoded by its /Filter."""
        return self._data

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

    def get_object(self) -> Any:
        return self.pdf.get_object(self)

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

    @classmethod
    def read_from_stream(cls, stream: Any, pdf: Any = None, *_: object) -> Any:
        start = stream.tell()
        fields = stream.read().split(None, 3)
        if len(fields) < 3 or fields[2][:1] != b"R":
            raise ValueError("invalid indirect reference")
        number, generation = int(fields[0]), int(fields[1])
        stream.seek(start)
        data = stream.read()
        stream.seek(start + data.index(b"R") + 1)
        return cls(number, generation, pdf)

    def __getitem__(self, key: Any) -> Any:
        return self.get_object()[key]

    def get_resolved(self, key: Any, default: Any = None) -> Any:
        return self.get_object().get_resolved(key, default)

    def __contains__(self, key: Any) -> bool:
        return key in self.get_object()

    def __iter__(self) -> Any:
        return iter(self.get_object())


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
    if isinstance(value, PdfName):
        return NameObject(f"/{value.value}")
    if isinstance(value, str):
        return NameObject(f"/{value}")
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
    return NameObject(f"/{getattr(key, 'value', key)}")


def to_core(value: Any) -> Any:
    """Convert a pypdf object for core-pdf's serializer."""
    # Destinations and outline items are written in a different form from the one they hold.
    core_form = getattr(value, "core_form", None)
    if core_form is not None:
        return to_core(core_form())
    if isinstance(value, (NullObject, type(None))):
        return None
    if isinstance(value, BooleanObject):
        return value.value
    if isinstance(value, IndirectObject):
        return PdfReference(value.idnum, value.generation)
    if isinstance(value, NumberObject):
        return int(value)
    if isinstance(value, FloatObject):
        return float(value)
    if isinstance(value, NameObject):
        if not value.startswith("/"):
            raise ValueError(f"name must start with /: {value!r}")
        return PdfName(value[1:].encode("latin-1"))
    if isinstance(value, ByteStringObject):
        return PdfString(bytes(value), is_literal=False)
    if isinstance(value, str):
        return PdfString(_encode_text(value))
    if isinstance(value, StreamObject):
        items: dict[str, Any] = dict(dict.items(value))
        dictionary = {key[1:]: to_core(item) for key, item in items.items()}
        return PdfStream(dictionary, value._stored_data())
    if isinstance(value, Mapping):
        plain: dict[str, Any] = (
            dict(value) if not isinstance(value, dict) else dict(dict.items(value))
        )
        return {key[1:]: to_core(item) for key, item in plain.items()}
    if isinstance(value, list):
        return [to_core(list.__getitem__(value, index)) for index in range(len(value))]
    return value


def _encode_text(value: str) -> bytes:
    text = value if isinstance(value, TextStringObject) else TextStringObject(value)
    return text.get_encoded_bytes()


def read_object(stream: Any, pdf: Any = None) -> Any:
    start = stream.tell()
    data = stream.read()
    lexer = PdfLexer(data)
    value = lexer.parse_object()
    stream.seek(start + lexer.pos)
    return to_generic(value, pdf)


def read_buffer(data: bytes, pdf: Any = None) -> Any:
    return read_object(BytesIO(data), pdf)


from core_pdf_compat.pypdf._generic_content import ContentStream, decode_stream  # noqa: E402

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
