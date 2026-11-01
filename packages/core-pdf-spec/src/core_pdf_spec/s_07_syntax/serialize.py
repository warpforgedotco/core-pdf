# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.3: write PDF objects as bytes, the inverse of the lexer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from typing import Any

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


class ObjectLayout(Enum):
    """How objects are spaced. Both read back as the same objects.

    COMPACT is what core-pdf writes to files. SPACED is the layout pypdf gives an object's
    write_to_stream: one dictionary entry per line, spaces inside array brackets, lowercase
    hex, an octal escape for every byte of a literal string that is not a letter, digit or
    space, and 0.0 for zero.
    """

    COMPACT = "compact"
    SPACED = "spaced"


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


def serialize_string(string: PdfString, layout: ObjectLayout = ObjectLayout.COMPACT) -> bytes:
    spaced = layout is ObjectLayout.SPACED
    if string.is_literal is False:
        digits = string.data.hex()
        return b"<" + (digits if spaced else digits.upper()).encode("ascii") + b">"
    output = bytearray(b"(")
    for byte in string.data:
        if spaced:
            is_plain = byte == 0x20 or bytes((byte,)).isalnum()
            output += bytes((byte,)) if is_plain else b"\\%03o" % byte
            continue
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
    # 7.3.3: exponent notation is not allowed. repr gives the shortest text that reads back
    # as exactly this value, so no precision is lost.
    text = format(Decimal(repr(value)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return (text if text not in {"", "-", "-0"} else "0").encode("ascii")


def serialize_object(value: object, layout: ObjectLayout = ObjectLayout.COMPACT) -> bytes:
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int):
        return b"%d" % value
    if isinstance(value, float):
        return b"0.0" if value == 0 and layout is ObjectLayout.SPACED else serialize_real(value)
    # The lexer returns some well-known names, such as /Type values, as plain strings.
    if isinstance(value, (PdfName, str)):
        return serialize_name(value)
    if isinstance(value, PdfString):
        return serialize_string(value, layout)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return serialize_string(PdfString(bytes(value)), layout)
    if isinstance(value, PdfReference):
        return b"%d %d R" % (value.object_number, value.generation_number)
    if isinstance(value, PdfStream):
        return serialize_stream(value, layout)
    if isinstance(value, Mapping):
        return _serialize_dictionary(value, layout)
    if isinstance(value, Sequence):
        items = [serialize_object(item, layout) for item in value]
        if layout is ObjectLayout.SPACED:
            return b"[" + b"".join(b" " + item for item in items) + b" ]"
        return b"[" + b" ".join(items) + b"]"
    raise TypeError(f"cannot serialize {type(value).__name__} as a PDF object")


def _serialize_dictionary(dictionary: Mapping[Any, object], layout: ObjectLayout) -> bytes:
    entries = [
        serialize_name(key) + b" " + serialize_object(item, layout)
        for key, item in dictionary.items()
    ]
    if layout is ObjectLayout.SPACED:
        return b"<<\n" + b"".join(entry + b"\n" for entry in entries) + b">>"
    return b"<< " + b" ".join(entries) + b" >>"


def byte_width(value: int) -> int:
    return max((value.bit_length() + 7) // 8, 1)


def serialize_xref_stream(
    number: int,
    offsets: Mapping[int, tuple[int, int]],
    trailer: Mapping[object, object],
    own_offset: int,
    *,
    compressed: Mapping[int, tuple[int, int]] | None = None,
    complete: bool = False,
) -> bytes:
    """7.5.8: an xref stream object, numbered `number`, that also carries the trailer entries.

    `offsets` maps object numbers to (file offset, generation). The stream's own entry is added.
    `compressed` maps objects inside object streams to (stream number, index). A `complete`
    stream covers every number from 0, with the gaps free, as a file's only section does.
    Otherwise it lists just these objects, as an incremental update does.
    """
    used = {item: (2, stream, index) for item, (stream, index) in (compressed or {}).items()}
    used |= {item: (1, offset, generation) for item, (offset, generation) in offsets.items()}
    used[number] = (1, own_offset, 0)
    total = max(used) + 1
    index: list[int] = []
    if complete:
        rows = [used.get(item, (0, 0, 65535)) for item in range(total)]
        index = [0, total]
    else:
        rows = [used[item] for item in sorted(used)]
        for item in sorted(used):
            if index and index[-2] + index[-1] == item:
                index[-1] += 1
            else:
                index += [item, 1]
    widths = [1, byte_width(max(row[1] for row in rows)), byte_width(max(row[2] for row in rows))]
    data = b"".join(
        b"".join(field.to_bytes(width, "big") for field, width in zip(row, widths)) for row in rows
    )
    dictionary = {
        **trailer,
        "Type": PdfName.of("XRef"),
        "Size": total,
        "W": widths,
        "Index": index,
    }
    return serialize_indirect_object(PdfReference(number, 0), PdfStream(dictionary, data))


def serialize_stream(stream: PdfStream, layout: ObjectLayout = ObjectLayout.COMPACT) -> bytes:
    # 7.3.8: the length is rewritten, because the data may no longer match the original.
    dictionary = dict(stream.dictionary)
    data = bytes(stream.raw_data)
    dictionary["Length"] = len(data)
    return serialize_object(dictionary, layout) + b"\nstream\n" + data + b"\nendstream"


def serialize_indirect_object(
    reference: PdfReference, value: object, layout: ObjectLayout = ObjectLayout.COMPACT
) -> bytes:
    header = b"%d %d obj\n" % (reference.object_number, reference.generation_number)
    return header + serialize_object(value, layout) + b"\nendobj\n"


__all__ = (
    "ObjectLayout",
    "serialize_indirect_object",
    "serialize_name",
    "serialize_object",
    "serialize_real",
    "serialize_stream",
    "serialize_string",
    "serialize_xref_stream",
)
