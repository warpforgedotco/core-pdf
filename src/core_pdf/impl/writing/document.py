# SPDX-License-Identifier: AGPL-3.0-only
"""Serialize a PDF object graph into a complete classic-xref PDF file."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from core_pdf.impl.primitives import PdfName, PdfReference
from core_pdf.impl.writing.encryption import StandardPdfEncryption
from core_pdf.impl.writing.objects import (
    internal_append_indirect_objects,
    internal_classic_xref_entry,
    serialize_pdf_dictionary,
)


class PdfObjectGraph:
    """Allocate and freeze a deterministic indirect-object graph."""

    def __init__(self, *, first_object_number: int = 1) -> None:
        if first_object_number < 1:
            raise ValueError("first PDF object number must be positive")
        self.internal_next_object_number = first_object_number
        self.internal_objects: dict[int, object] = {}
        self.internal_frozen = False

    @property
    def objects(self) -> Mapping[int, object]:
        return MappingProxyType(self.internal_objects)

    def add(self, value: object) -> PdfReference:
        self.internal_ensure_mutable()
        number = self.internal_next_object_number
        self.internal_next_object_number += 1
        self.internal_objects[number] = value
        return PdfReference(number)

    def replace(self, reference: PdfReference, value: object) -> None:
        self.internal_ensure_mutable()
        if reference.generation_number != 0 or reference.object_number not in self.internal_objects:
            raise KeyError("reference does not belong to this object graph")
        self.internal_objects[reference.object_number] = value

    def freeze(self) -> Mapping[int, object]:
        self.internal_frozen = True
        return self.objects

    def to_pdf(
        self,
        *,
        trailer: Mapping[object, object],
        version: str = "1.7",
    ) -> bytes:
        self.internal_frozen = True
        return serialize_pdf_file(self.internal_objects, trailer=trailer, version=version)

    def internal_ensure_mutable(self) -> None:
        if self.internal_frozen:
            raise RuntimeError("PDF object graph is frozen")


def serialize_pdf_file(
    objects: Mapping[int, object],
    *,
    trailer: Mapping[object, object],
    version: str = "1.7",
) -> bytes:
    """Serialize numbered indirect objects and a trailer into a PDF file."""
    validate_pdf_version(version)
    if not objects:
        raise ValueError("a PDF file must contain at least one indirect object")

    output = bytearray(f"%PDF-{version}\n%\xe2\xe3\xcf\xd3\n".encode("latin-1"))
    offsets = internal_append_indirect_objects(output, objects)

    xref_offset = len(output)
    size = max(offsets) + 1
    output.extend(f"xref\n0 {size}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for number in range(1, size):
        offset = offsets.get(number)
        if offset is None:
            output.extend(b"0000000000 00000 f \n")
        else:
            output.extend(internal_classic_xref_entry(offset))

    trailer_dict = dict(trailer)
    trailer_dict[PdfName.of("Size")] = size
    output.extend(b"trailer\n")
    output.extend(serialize_pdf_dictionary(trailer_dict))
    output.extend(f"\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(output)


def serialize_encrypted_pdf_file(
    objects: Mapping[int, object],
    *,
    trailer: Mapping[object, object],
    encryption: StandardPdfEncryption,
    file_id: bytes,
    version: str = "1.7",
) -> bytes:
    """Serialize a PDF using Standard Security Revision 3 encryption."""
    if not objects:
        raise ValueError("a PDF file must contain at least one indirect object")
    encryption_object_number = max(objects) + 1
    context = encryption.context(file_id)
    encrypted_objects = {
        number: context.encrypt_object(value, number) for number, value in objects.items()
    }
    encrypted_objects[encryption_object_number] = context.encryption_dictionary()
    encrypted_trailer = {
        **dict(trailer),
        PdfName.of("Encrypt"): PdfReference(encryption_object_number),
        PdfName.of("ID"): [file_id, file_id],
    }
    return serialize_pdf_file(encrypted_objects, trailer=encrypted_trailer, version=version)


def validate_pdf_version(version: str) -> None:
    if version not in {"1.4", "1.5", "1.6", "1.7"}:
        raise ValueError(f"unsupported PDF version: {version!r}")


__all__ = (
    "PdfObjectGraph",
    "serialize_encrypted_pdf_file",
    "serialize_pdf_file",
    "validate_pdf_version",
)
