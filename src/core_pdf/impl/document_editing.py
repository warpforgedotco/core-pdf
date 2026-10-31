# SPDX-License-Identifier: AGPL-3.0-only

"""Copy-on-write editing over a parsed document, saved as a full rewrite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.types import PdfReference
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
        offsets: dict[int, tuple[int, int]] = {}
        reachable = self.reachable_references()
        for number in sorted(reachable):
            reference = reachable[number]
            offsets[number] = (len(output), reference.generation_number)
            output += serialize_indirect_object(reference, self.get_object(reference))
        size = max(offsets, default=0) + 1
        xref_offset = len(output)
        output += b"xref\n0 %d\n0000000000 65535 f \n" % size
        for number in range(1, size):
            offset, generation = offsets.get(number, (0, 65535))
            marker = b"n" if number in offsets else b"f"
            output += b"%010d %05d %s \n" % (offset, generation, marker)
        trailer: dict[Any, Any] = {"Size": size}
        trailer.update({key: self.trailer[key] for key in TRAILER_KEYS if key in self.trailer})
        output += b"trailer\n" + serialize_object(trailer)
        output += b"\nstartxref\n%d\n%%%%EOF\n" % xref_offset
        result = bytes(output)
        if target is not None:
            Path(target).write_bytes(result)
        return result


__all__ = ("EditableDocument",)
