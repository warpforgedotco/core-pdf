"""Serialize PDF object syntax without resolving indirect references."""

from __future__ import annotations

from decimal import Decimal

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfName, PdfReference, PdfString


def serialize_object(value: object) -> bytes:
    if value is None:
        return b"null"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, PdfReference):
        return str(value).encode("ascii")
    if isinstance(value, PdfName):
        return serialize_name(value.value_bytes)
    if isinstance(value, PdfString):
        return b"<" + value.data.hex().encode("ascii") + b">"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"<" + bytes(value).hex().encode("ascii") + b">"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, float):
        return format(Decimal(str(value)), "f").encode("ascii")
    if isinstance(value, str):
        return value.encode("latin-1")
    if isinstance(value, (list, tuple)):
        return b"[" + b" ".join(serialize_object(v) for v in value) + b"]"
    if isinstance(value, PdfStream):
        raw = bytes(value.raw_data)
        dictionary = dict(value.dictionary)
        dictionary["Length"] = len(raw)
        return serialize_object(dictionary) + b"\nstream\n" + raw + b"\nendstream"
    if isinstance(value, dict):
        return (
            b"<<"
            + b" ".join(
                serialize_name(
                    k.value_bytes
                    if isinstance(k, PdfName)
                    else k
                    if isinstance(k, bytes)
                    else str(k).encode("latin-1")
                )
                + b" "
                + serialize_object(v)
                for k, v in value.items()
            )
            + b">>"
        )
    raise TypeError(f"unsupported PDF object: {type(value).__name__}")


def serialize_name(value: bytes) -> bytes:
    return b"/" + "".join(
        chr(c) if 33 <= c <= 126 and c not in b"()<>[]{}/%#" else f"#{c:02X}" for c in value
    ).encode("ascii")


__all__ = ("serialize_object", "serialize_name")
