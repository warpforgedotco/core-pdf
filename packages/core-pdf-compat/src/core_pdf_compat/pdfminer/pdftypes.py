"""pdfminer's object types, presented over core-pdf's parsed objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core_pdf.impl.types import PdfName, PdfReference, PdfString

from .psparser import PSLiteral


class PDFObject:
    pass


class PDFStream(PDFObject):
    def __init__(self, attrs: dict[str, Any], rawdata: bytes | None = None) -> None:
        self.attrs = attrs
        self.rawdata = rawdata

    def get(self, name: str, default: object = None) -> Any:
        return self.attrs.get(name, default)

    def __contains__(self, name: object) -> bool:
        return name in self.attrs

    def __getitem__(self, name: str) -> Any:
        return self.attrs[name]


class PDFObjRef(PDFObject):
    def __init__(self, doc: DocumentView | None, objid: int) -> None:
        self.doc = doc
        self.objid = objid

    def __repr__(self) -> str:
        return f"<PDFObjRef:{self.objid}>"

    def resolve(self, default: object = None) -> Any:
        if self.doc is None:
            return default
        return self.doc.getobj(self.objid)


def resolve1(value: Any, default: object = None) -> Any:
    while isinstance(value, PDFObjRef):
        value = value.resolve(default)
    return value


class TrailerView:
    def __init__(self, trailer: dict[str, Any]) -> None:
        self.trailer = trailer


class DocumentView:
    """Reads a core-pdf document through pdfminer's PDFDocument attributes."""

    def __init__(self, document: Any) -> None:
        self.document = document
        self.objects: dict[int, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.document, name)

    def bridge(self, value: Any) -> Any:
        if isinstance(value, PdfReference):
            return PDFObjRef(self, value.object_number)
        if isinstance(value, PdfName):
            return PSLiteral(value.value)
        if isinstance(value, PdfString):
            return value.data
        if isinstance(value, Mapping):
            return {
                key.value if isinstance(key, PdfName) else key: self.bridge(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.bridge(item) for item in value]
        dictionary = getattr(value, "dictionary", None)
        if dictionary is not None:
            return PDFStream(self.bridge(dictionary), getattr(value, "raw_data", None))
        return value

    def getobj(self, objid: int) -> Any:
        if objid not in self.objects:
            # Cached so that edits to a returned dictionary persist, as in pdfminer.
            self.objects[objid] = self.bridge(self.document.resolver.resolve(PdfReference(objid)))
        return self.objects[objid]

    @property
    def catalog(self) -> dict[str, Any]:
        return resolve1(self.bridge(self.document.trailer_dict).get("Root"))

    @property
    def xrefs(self) -> list[TrailerView]:
        return [TrailerView(self.bridge(self.document.trailer_dict))]

    @property
    def info(self) -> list[dict[str, Any]]:
        info = resolve1(self.bridge(self.document.trailer_dict).get("Info"))
        return [info] if isinstance(info, dict) else []

    def page_ids(self) -> list[int]:
        found: list[int] = []
        seen: set[int] = set()

        def visit(reference: Any) -> None:
            if not isinstance(reference, PDFObjRef) or reference.objid in seen:
                return
            seen.add(reference.objid)
            node = resolve1(reference)
            if not isinstance(node, dict):
                return
            kids = resolve1(node.get("Kids"))
            if kids is None:
                found.append(reference.objid)
                return
            for kid in kids:
                visit(kid)

        visit(self.catalog.get("Pages"))
        return found


__all__ = ("DocumentView", "PDFObjRef", "PDFObject", "PDFStream", "resolve1")
