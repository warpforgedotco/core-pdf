# SPDX-License-Identifier: AGPL-3.0-only
"""Indirect-object allocation for PDF writers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from core_pdf.impl.primitives import PdfReference
from core_pdf.impl.writing.document import serialize_pdf_file


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


__all__ = ("PdfObjectGraph",)
