"""pypdf's object types, converted to and from core-pdf's parsed objects.

Bytes are always written by core_pdf_spec's serializer, so there is one writer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from io import BytesIO
from typing import Any

from core_pdf.impl.pdf_values import decode_pdf_text_string
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_compat.pypdf.errors import PdfReadError
from core_pdf_spec.s_07_filters.codecs import encode_flate
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.serialize import serialize_object
from core_pdf_spec.s_07_syntax.stream import PdfStream


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
    def as_numeric(self) -> float:
        return float(self)


class NameObject(str, PdfObject):
    __slots__ = ()

    def __new__(cls, value: str = "/") -> NameObject:
        if not str(value).startswith("/"):
            raise ValueError("name must start with /")
        return super().__new__(cls, value)


class TextStringObject(str, PdfObject):
    pass


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

    def get(self, key: Any, default: Any = None) -> Any:
        return self.__getitem__(key) if dict.__contains__(self, key) else default  # noqa: SIM401

    def raw_get(self, key: Any) -> Any:
        return super().__getitem__(key)


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

    def __contains__(self, key: Any) -> bool:
        return key in self.get_object()

    def __iter__(self) -> Any:
        return iter(self.get_object())


class RectangleObject(ArrayObject):
    def __init__(self, values: Iterable[Any]) -> None:
        items = list(values)
        if len(items) != 4:
            raise ValueError("a rectangle needs four values")
        super().__init__(FloatObject(item) for item in items)

    left = property(lambda self: self[0])
    bottom = property(lambda self: self[1])
    right = property(lambda self: self[2])
    top = property(lambda self: self[3])
    width = property(lambda self: self[2] - self[0])
    height = property(lambda self: self[3] - self[1])


def is_null_or_none(value: object) -> bool:
    if isinstance(value, IndirectObject):
        value = value.get_object()
    return value is None or isinstance(value, NullObject)


def create_string_object(value: str | bytes) -> TextStringObject | ByteStringObject:
    if isinstance(value, str):
        return TextStringObject(value)
    try:
        return TextStringObject(decode_pdf_text_string(value))
    except UnicodeDecodeError, ValueError:
        return ByteStringObject(value)


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
        return create_string_object(value.data)
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
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:
        return b"\xfe\xff" + value.encode("utf-16-be")


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
    "ArrayObject",
    "BooleanObject",
    "ByteStringObject",
    "ContentStream",
    "DecodedStreamObject",
    "DictionaryObject",
    "EncodedStreamObject",
    "FloatObject",
    "IndirectObject",
    "NameObject",
    "NullObject",
    "NumberObject",
    "PdfObject",
    "RectangleObject",
    "StreamObject",
    "TextStringObject",
    "create_string_object",
    "is_null_or_none",
    "read_object",
    "to_core",
    "to_generic",
)
