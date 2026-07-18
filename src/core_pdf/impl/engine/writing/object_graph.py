# SPDX-License-Identifier: AGPL-3.0-only
"""Indirect-object allocation for PDF writers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from core_pdf.impl.engine.writing.document import serialize_pdf_file
from core_pdf.impl.objects import PdfReference


class PdfObjectGraph:
    """Allocate and freeze a deterministic indirect-object graph."""

    def __init__(self, *, first_object_number: int = 1) -> None:
        if first_object_number < 1:
            raise ValueError("first PDF object number must be positive")
        self._next_object_number = first_object_number
        self._objects: dict[int, object] = {}
        self._frozen = False

    @property
    def objects(self) -> Mapping[int, object]:
        return MappingProxyType(self._objects)

    def add(self, value: object) -> PdfReference:
        self._ensure_mutable()
        number = self._next_object_number
        self._next_object_number += 1
        self._objects[number] = value
        return PdfReference(number)

    def replace(self, reference: PdfReference, value: object) -> None:
        self._ensure_mutable()
        if reference.generation_number != 0 or reference.object_number not in self._objects:
            raise KeyError("reference does not belong to this object graph")
        self._objects[reference.object_number] = value

    def freeze(self) -> Mapping[int, object]:
        self._frozen = True
        return self.objects

    def to_pdf(
        self,
        *,
        trailer: Mapping[object, object],
        version: str = "1.7",
    ) -> bytes:
        self._frozen = True
        return serialize_pdf_file(self._objects, trailer=trailer, version=version)

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("PDF object graph is frozen")


__all__ = ("PdfObjectGraph",)
