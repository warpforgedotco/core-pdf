# SPDX-License-Identifier: AGPL-3.0-only
"""Serialize parsed PDF objects back to PDF syntax."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from core_pdf.impl.objects import (
    PdfByteRangePlaceholder,
    PdfName,
    PdfReference,
    PdfSignatureContentsPlaceholder,
    PdfStream,
    PdfString,
)


def serialize_pdf_object(value: object) -> bytes:
    """Return valid PDF syntax for one direct or indirect object."""
    if value is None:
        return b"null"
    if isinstance(value, PdfByteRangePlaceholder):
        return b"[0 0000000000 0000000000 0000000000]"
    if isinstance(value, PdfSignatureContentsPlaceholder):
        if value.length <= 0:
            raise ValueError("signature placeholder length must be positive")
        return b"<" + b"0" * (value.length * 2) + b">"
    if type(value) is bool:
        return b"true" if value else b"false"
    if type(value) is int:
        return str(value).encode("ascii")
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("PDF numbers must be finite")
        if value == 0:
            return b"0"
        return format(value, ".15g").encode("ascii")
    if isinstance(value, PdfName):
        return serialize_pdf_name(value)
    if isinstance(value, PdfReference):
        return f"{value.object_number} {value.generation_number} R".encode("ascii")
    if isinstance(value, PdfString):
        return serialize_pdf_string(value.data)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return serialize_pdf_string(bytes(value))
    if isinstance(value, str):
        return serialize_pdf_string(value.encode("latin-1"))
    if isinstance(value, (list, tuple)):
        return b"[" + b" ".join(serialize_pdf_object(item) for item in value) + b"]"
    if isinstance(value, PdfStream):
        return serialize_pdf_stream(value)
    if isinstance(value, Mapping):
        return serialize_pdf_dictionary(cast(Mapping[object, Any], value))
    raise TypeError(f"unsupported PDF object: {type(value).__name__}")


def serialize_pdf_name(value: PdfName) -> bytes:
    encoded = bytearray(b"/")
    for byte in value.value_bytes:
        char = chr(byte)
        if 33 <= byte <= 126 and char not in "()<>[]{}/%#":
            encoded.append(byte)
        else:
            encoded.extend(f"#{byte:02X}".encode("ascii"))
    return bytes(encoded)


def serialize_pdf_string(value: bytes) -> bytes:
    return b"<" + value.hex().upper().encode("ascii") + b">"


def serialize_pdf_dictionary(value: Mapping[object, Any]) -> bytes:
    entries = []
    for key, item in value.items():
        entries.append(serialize_pdf_key(key) + b" " + serialize_pdf_object(item))
    return b"<<" + (b" " + b" ".join(entries) if entries else b"") + b" >>"


def serialize_pdf_key(value: object) -> bytes:
    if isinstance(value, PdfName):
        return serialize_pdf_name(value)
    if isinstance(value, bytes):
        return serialize_pdf_name(PdfName.of(value))
    if isinstance(value, str):
        return serialize_pdf_name(PdfName.of(value))
    raise TypeError(f"unsupported PDF dictionary key: {type(value).__name__}")


def serialize_pdf_stream(value: PdfStream) -> bytes:
    dictionary = dict(value.dictionary)
    dictionary[PdfName.of("Length")] = len(value.raw_data)
    return (
        serialize_pdf_dictionary(dictionary)
        + b"\nstream\n"
        + bytes(value.raw_data)
        + b"\nendstream"
    )


__all__ = (
    "serialize_pdf_dictionary",
    "serialize_pdf_key",
    "serialize_pdf_name",
    "serialize_pdf_object",
    "serialize_pdf_stream",
    "serialize_pdf_string",
)
