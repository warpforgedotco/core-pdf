"""Serialize PDF object syntax without resolving indirect references."""

from __future__ import annotations

from decimal import Decimal

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfName, PdfReference, PdfString


def serialize_object(value: object) -> bytes:
    out = bytearray()
    serialize_into(out, value)
    return bytes(out)


def serialize_into(out: bytearray, value: object) -> None:
    """Append ``value``'s object syntax to ``out``.

    Appending rather than returning matters for streams: a stream body is
    already the largest thing in the file, and building the object by
    concatenation copied it once to join it to its dictionary and again to
    reach the caller's buffer.
    """
    if value is None:
        out += b"null"
        return
    if isinstance(value, bool):
        out += b"true" if value else b"false"
        return
    if isinstance(value, PdfReference):
        out += str(value).encode("ascii")
        return
    if isinstance(value, PdfName):
        out += serialize_name(value)
        return
    if isinstance(value, PdfString):
        out += b"<" + value.data.hex().encode("ascii") + b">"
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        out += b"<" + bytes(value).hex().encode("ascii") + b">"
        return
    if isinstance(value, int):
        out += str(value).encode("ascii")
        return
    if isinstance(value, float):
        out += format(Decimal(str(value)), "f").encode("ascii")
        return
    if isinstance(value, str):
        out += value.encode("latin-1")
        return
    if isinstance(value, (list, tuple)):
        out += b"["
        for index, item in enumerate(value):
            if index:
                out += b" "
            serialize_into(out, item)
        out += b"]"
        return
    if isinstance(value, PdfStream):
        raw = value.raw_data
        dictionary = dict(value.dictionary)
        # A memoryview's len() counts elements rather than bytes. These views
        # are always itemsize 1, but /Length has to be right either way.
        dictionary["Length"] = raw.nbytes if isinstance(raw, memoryview) else len(raw)
        serialize_into(out, dictionary)
        out += b"\nstream\n"
        out += raw
        out += b"\nendstream"
        return
    if isinstance(value, dict):
        out += b"<<"
        for index, (key, item) in enumerate(value.items()):
            if index:
                out += b" "
            out += serialize_name(key)
            out += b" "
            serialize_into(out, item)
        out += b">>"
        return
    raise TypeError(f"unsupported PDF object: {type(value).__name__}")


def serialize_name(value: object) -> bytes:
    """Write a name token, escaping per ISO 32000-1 7.3.5."""
    data = (
        value.value_bytes
        if isinstance(value, PdfName)
        else value
        if isinstance(value, bytes)
        else str(value).encode("latin-1")
    )
    return b"/" + "".join(
        chr(c) if 33 <= c <= 126 and c not in b"()<>[]{}/%#" else f"#{c:02X}" for c in data
    ).encode("ascii")


__all__ = ("serialize_object", "serialize_into", "serialize_name")
