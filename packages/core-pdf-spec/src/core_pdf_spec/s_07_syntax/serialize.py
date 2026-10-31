# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.3: write PDF objects as bytes, the inverse of the lexer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfName, PdfReference, PdfString

# 7.2.3: white-space and delimiter characters. 7.3.5: these, and bytes outside the
# printable range, are written as #xx inside a name.
NAME_ESCAPED = frozenset(b"\x00\t\n\x0c\r ()<>[]{}/%#")
LITERAL_ESCAPES = {
    0x0A: b"\\n",
    0x0D: b"\\r",
    0x09: b"\\t",
    0x08: b"\\b",
    0x0C: b"\\f",
    0x28: b"\\(",
    0x29: b"\\)",
    0x5C: b"\\\\",
}


def serialize_name(name: str | bytes | PdfName) -> bytes:
    if isinstance(name, PdfName):
        raw = name.value_bytes
    elif isinstance(name, str):
        raw = name.encode("latin-1")
    else:
        raw = bytes(name)
    output = bytearray(b"/")
    for byte in raw:
        if byte in NAME_ESCAPED or not 0x21 <= byte <= 0x7E:
            output += b"#%02X" % byte
        else:
            output.append(byte)
    return bytes(output)


def serialize_string(string: PdfString) -> bytes:
    if string.is_literal is False:
        return b"<" + string.data.hex().upper().encode("ascii") + b">"
    output = bytearray(b"(")
    for byte in string.data:
        escape = LITERAL_ESCAPES.get(byte)
        if escape is not None:
            output += escape
        else:
            output.append(byte)
    output += b")"
    return bytes(output)


def serialize_real(value: float) -> bytes:
    if not math.isfinite(value):
        raise ValueError("PDF real numbers must be finite")
    # 7.3.3: exponent notation is not allowed.
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return (text if text not in {"", "-", "-0"} else "0").encode("ascii")


def serialize_object(value: object) -> bytes:
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int):
        return b"%d" % value
    if isinstance(value, float):
        return serialize_real(value)
    # The lexer returns some well-known names, such as /Type values, as plain strings.
    if isinstance(value, (PdfName, str)):
        return serialize_name(value)
    if isinstance(value, PdfString):
        return serialize_string(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return serialize_string(PdfString(bytes(value)))
    if isinstance(value, PdfReference):
        return b"%d %d R" % (value.object_number, value.generation_number)
    if isinstance(value, PdfStream):
        return serialize_stream(value)
    if isinstance(value, Mapping):
        items = (serialize_name(key) + b" " + serialize_object(item) for key, item in value.items())
        return b"<< " + b" ".join(items) + b" >>"
    if isinstance(value, Sequence):
        return b"[" + b" ".join(serialize_object(item) for item in value) + b"]"
    raise TypeError(f"cannot serialize {type(value).__name__} as a PDF object")


def serialize_stream(stream: PdfStream) -> bytes:
    # 7.3.8: the length is rewritten, because the data may no longer match the original.
    dictionary = dict(stream.dictionary)
    data = bytes(stream.raw_data)
    dictionary["Length"] = len(data)
    return serialize_object(dictionary) + b"\nstream\n" + data + b"\nendstream"


def serialize_indirect_object(reference: PdfReference, value: object) -> bytes:
    header = b"%d %d obj\n" % (reference.object_number, reference.generation_number)
    return header + serialize_object(value) + b"\nendobj\n"


__all__ = (
    "serialize_indirect_object",
    "serialize_name",
    "serialize_object",
    "serialize_real",
    "serialize_stream",
    "serialize_string",
)
