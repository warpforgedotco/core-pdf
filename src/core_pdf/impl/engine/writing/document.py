# SPDX-License-Identifier: AGPL-3.0-only
"""Serialize a PDF object graph into a complete classic-xref PDF file."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.objects import serialize_pdf_dictionary, serialize_pdf_object
from core_pdf.impl.primitives import PdfName, PdfReference


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
    if any(type(number) is not int or number <= 0 for number in objects):
        raise ValueError("PDF object numbers must be positive integers")

    highest_object_number = max(objects)
    output = bytearray(f"%PDF-{version}\n%\xe2\xe3\xcf\xd3\n".encode("latin-1"))
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(serialize_pdf_object(objects[number]))
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    size = highest_object_number + 1
    output.extend(f"xref\n0 {size}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for number in range(1, size):
        offset = offsets.get(number)
        if offset is None:
            output.extend(b"0000000000 00000 f \n")
        elif offset >= 10_000_000_000:
            raise ValueError("PDF file is too large for a classic xref table")
        else:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

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


__all__ = ("serialize_encrypted_pdf_file", "serialize_pdf_file", "validate_pdf_version")
