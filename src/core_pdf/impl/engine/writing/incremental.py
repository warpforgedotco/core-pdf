# SPDX-License-Identifier: AGPL-3.0-only
"""Append classic-xref incremental updates to an existing PDF."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf.impl.engine.writing.objects import (
    internal_append_indirect_objects,
    internal_classic_xref_entry,
    serialize_pdf_dictionary,
)
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_syntax.xref import XRefScanner


def find_startxref(data: bytes) -> int:
    """Return the final classic-xref offset or reject a malformed source PDF."""
    offset = XRefScanner.find_startxref(data)
    if offset is None:
        raise ValueError("PDF does not contain a startxref marker")
    return offset


def previous_object_count(
    xref: Mapping[int, object],
    trailer: Mapping[object, object],
) -> int:
    """Return the next available object number for an incremental update."""
    raw_size = trailer.get("Size")
    if raw_size is None:
        raw_size = trailer.get(b"Size")
    if type(raw_size) is int and raw_size > 0:
        return raw_size
    object_numbers = [key >> 16 for key in xref if key > 0]
    return max(object_numbers, default=0) + 1


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

    output = bytearray(original)
    if not output.endswith((b"\n", b"\r")):
        output.extend(b"\n")
    offsets = internal_append_indirect_objects(output, objects)

    xref_offset = len(output)
    output.extend(b"xref\n")
    for number in sorted(offsets):
        output.extend(f"{number} 1\n".encode("ascii"))
        output.extend(internal_classic_xref_entry(offsets[number]))

    trailer_dict = dict(trailer)
    trailer_dict[PdfName.of("Size")] = max(previous_size, max(objects) + 1)
    trailer_dict[PdfName.of("Prev")] = previous_xref_offset
    output.extend(b"trailer\n")
    output.extend(serialize_pdf_dictionary(trailer_dict))
    output.extend(f"\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(output)


__all__ = ("append_incremental_update", "find_startxref", "previous_object_count")
