# SPDX-License-Identifier: AGPL-3.0-only

"""Copy-on-write editing over a parsed document, saved as a full rewrite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.recovery_xref import XRefScanner
from core_pdf.impl.types import PdfName, PdfReference
from core_pdf_spec.s_07_syntax.serialize import serialize_indirect_object, serialize_object
from core_pdf_spec.s_07_syntax.stream import PdfStream

if TYPE_CHECKING:
    from core_pdf.impl.document_document import PdfDocument

BINARY_MARKER = b"%\xe2\xe3\xcf\xd3\n"
TRAILER_KEYS = ("Root", "Info", "ID")


def mutable_copy(value: Any) -> Any:
    if isinstance(value, PdfStream):
        return value.replace(dictionary=mutable_copy(dict(value.dictionary)))
    if isinstance(value, dict):
        return {key: mutable_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mutable_copy(item) for item in value]
    return value


def references_in(value: Any) -> list[PdfReference]:
    found: list[PdfReference] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, PdfReference):
            found.append(item)
        elif isinstance(item, PdfStream):
            pending.extend(item.dictionary.values())
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return found


class EditableDocument:
    def __init__(self, document: PdfDocument) -> None:
        if "Encrypt" in document.trailer_dict:
            raise PdfUnsupportedError("editing an encrypted document is not supported")
        self.document = document
        self.trailer: dict[Any, Any] = mutable_copy(dict(document.trailer_dict))
        self.overlay: dict[int, Any] = {}
        # /Size is often wrong, so a new number must also clear every object in use.
        size = document.resolver.resolve(self.trailer.get("Size"))
        declared = size if isinstance(size, int) and size > 0 else 1
        self.next_object_number = max(declared, max(self.reachable_references(), default=0) + 1)

    def get_object(self, reference: PdfReference) -> Any:
        if reference.object_number in self.overlay:
            return self.overlay[reference.object_number]
        return self.document.resolver.resolve(reference)

    def edit_object(self, reference: PdfReference) -> Any:
        """Return a copy of the object that can be changed in place and will be saved."""
        number = reference.object_number
        if number not in self.overlay:
            self.overlay[number] = mutable_copy(self.document.resolver.resolve(reference))
        return self.overlay[number]

    def set_object(self, reference: PdfReference, value: Any) -> None:
        self.overlay[reference.object_number] = value

    def add_object(self, value: Any) -> PdfReference:
        reference = PdfReference(self.next_object_number, 0)
        self.next_object_number += 1
        self.overlay[reference.object_number] = value
        return reference

    def reachable_references(self) -> dict[int, PdfReference]:
        reachable: dict[int, PdfReference] = {}
        pending = references_in({key: self.trailer.get(key) for key in TRAILER_KEYS})
        while pending:
            reference = pending.pop()
            if reference.object_number in reachable:
                continue
            reachable[reference.object_number] = reference
            pending.extend(references_in(self.get_object(reference)))
        return reachable

    def save(self, target: str | Path | None = None) -> bytes:
        """Write every reachable object once, dropping old revisions and unused objects."""
        version = self.document.raw_data[:8]
        header = bytes(version) if bytes(version).startswith(b"%PDF-") else b"%PDF-1.7"
        output = bytearray(header + b"\n" + BINARY_MARKER)
        reachable = self.reachable_references()
        objects = {reference: self.get_object(reference) for reference in reachable.values()}
        trailer = {key: self.trailer[key] for key in TRAILER_KEYS if key in self.trailer}
        output += serialize_body(len(output), objects, trailer)
        return write_output(bytes(output), target)

    def save_incremental(self, target: str | Path | None = None) -> bytes:
        """Append changed objects after the original bytes, which stay identical.

        Earlier revisions remain readable in the file, so this is not a way to remove data.
        """
        original = bytes(self.document.raw_data)
        previous = XRefScanner.find_startxref(original)
        if previous is None:
            raise PdfUnsupportedError("an incremental update needs a readable startxref")
        output = bytearray(original)
        if not output.endswith(b"\n"):
            output += b"\n"
        offsets: dict[int, tuple[int, int]] = {}
        reachable = self.reachable_references()
        for number in sorted(self.overlay):
            existing = reachable.get(number)
            generation = 0 if existing is None else existing.generation_number
            offsets[number] = (len(output), generation)
            reference = PdfReference(number, generation)
            output += serialize_indirect_object(reference, self.overlay[number])
        trailer: dict[Any, Any] = {
            key: self.trailer[key] for key in TRAILER_KEYS if key in self.trailer
        }
        trailer["Prev"] = previous
        section_offset = len(output)
        if original[previous : previous + 4] == b"xref":
            trailer["Size"] = max(self.next_object_number, max(offsets, default=0) + 1)
            output += table_section(offsets) + b"trailer\n" + serialize_object(trailer)
        else:
            output += xref_stream_section(offsets, trailer, section_offset, self.next_object_number)
        output += b"\nstartxref\n%d\n%%%%EOF\n" % section_offset
        return write_output(bytes(output), target)


def serialize_body(start: int, objects: dict[PdfReference, Any], trailer: dict[Any, Any]) -> bytes:
    """Write objects, a classic xref table and a trailer, starting at file offset `start`."""
    output = bytearray()
    offsets: dict[int, tuple[int, int]] = {}
    for reference in sorted(objects, key=lambda item: item.object_number):
        offsets[reference.object_number] = (start + len(output), reference.generation_number)
        output += serialize_indirect_object(reference, objects[reference])
    size = max(offsets, default=0) + 1
    xref_offset = start + len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for number in range(1, size):
        offset, generation = offsets.get(number, (0, 65535))
        marker = b"n" if number in offsets else b"f"
        output += b"%010d %05d %s \n" % (offset, generation, marker)
    output += b"trailer\n" + serialize_object({"Size": size, **trailer})
    output += b"\nstartxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(output)


def write_output(data: bytes, target: str | Path | None) -> bytes:
    if target is not None:
        Path(target).write_bytes(data)
    return data


def table_section(offsets: dict[int, tuple[int, int]]) -> bytes:
    section = bytearray(b"xref\n")
    numbers = sorted(offsets)
    start = 0
    while start < len(numbers):
        end = start
        while end + 1 < len(numbers) and numbers[end + 1] == numbers[end] + 1:
            end += 1
        section += b"%d %d\n" % (numbers[start], end - start + 1)
        for number in numbers[start : end + 1]:
            offset, generation = offsets[number]
            section += b"%010d %05d n \n" % (offset, generation)
        start = end + 1
    return bytes(section)


def xref_stream_section(
    offsets: dict[int, tuple[int, int]],
    trailer: dict[Any, Any],
    own_offset: int,
    next_number: int,
) -> bytes:
    # A file indexed by xref streams has to be continued by one (ISO 32000-2 7.5.8.4).
    own_number = max(next_number, max(offsets, default=0) + 1)
    entries = {**offsets, own_number: (own_offset, 0)}
    numbers = sorted(entries)
    index: list[int] = []
    for number in numbers:
        if index and index[-2] + index[-1] == number:
            index[-1] += 1
        else:
            index += [number, 1]
    data = b"".join(
        b"\x01" + entries[number][0].to_bytes(4, "big") + entries[number][1].to_bytes(2, "big")
        for number in numbers
    )
    dictionary = {
        **trailer,
        "Type": PdfName(b"XRef"),
        "Size": own_number + 1,
        "W": [1, 4, 2],
        "Index": index,
    }
    return serialize_indirect_object(
        PdfReference(own_number, 0), PdfStream(dictionary, data)
    ).rstrip()


__all__ = ("BINARY_MARKER", "EditableDocument", "serialize_body")


__all__ = ("BINARY_MARKER", "EditableDocument", "serialize_body")
