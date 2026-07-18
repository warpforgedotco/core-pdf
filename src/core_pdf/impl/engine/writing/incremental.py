# SPDX-License-Identifier: AGPL-3.0-only
"""Append classic-xref incremental updates to an existing PDF."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf.impl.engine.writing.objects import serialize_pdf_dictionary, serialize_pdf_object
from core_pdf.impl.primitives import PdfName


def append_incremental_update(
    original: bytes,
    objects: Mapping[int, object],
    *,
    trailer: Mapping[object, object],
    previous_xref_offset: int,
    previous_size: int,
) -> bytes:
    """Return ``original`` plus a classic-xref incremental update."""
    if not original.startswith(b"%PDF-"):
        raise ValueError("original data is not a PDF file")
    if not objects:
        raise ValueError("an incremental update must contain at least one object")
    if previous_xref_offset < 0 or previous_size < 1:
        raise ValueError("invalid previous xref metadata")
    if any(type(number) is not int or number <= 0 for number in objects):
        raise ValueError("PDF object numbers must be positive integers")

    output = bytearray(original)
    if not output.endswith((b"\n", b"\r")):
        output.extend(b"\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(serialize_pdf_object(objects[number]))
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(b"xref\n")
    for number in sorted(offsets):
        output.extend(f"{number} 1\n".encode("ascii"))
        offset = offsets[number]
        if offset >= 10_000_000_000:
            raise ValueError("PDF file is too large for a classic xref table")
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    trailer_dict = dict(trailer)
    trailer_dict[PdfName.of("Size")] = max(previous_size, max(objects) + 1)
    trailer_dict[PdfName.of("Prev")] = previous_xref_offset
    output.extend(b"trailer\n")
    output.extend(serialize_pdf_dictionary(trailer_dict))
    output.extend(f"\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(output)


__all__ = ("append_incremental_update",)
