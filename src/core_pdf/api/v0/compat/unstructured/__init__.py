"""Local Unstructured-style partition and element conversion APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from core_pdf.api.v0.adapters import adapt_document
from core_pdf.api.v0.compat.pypdf import PdfInput

from .._common import open_source


class ElementMetadata(dict[str, Any]):
    """Dictionary-compatible Unstructured metadata with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


@dataclass(frozen=True, slots=True)
class Element:
    text: str
    metadata: ElementMetadata = field(default_factory=ElementMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ElementMetadata):
            object.__setattr__(self, "metadata", ElementMetadata(self.metadata))

    @property
    def category(self) -> str:
        return type(self).__name__

    @property
    def element_id(self) -> str:
        return str(self.metadata.get("element_id", ""))

    @property
    def id(self) -> str:
        return self.element_id

    @property
    def page_number(self) -> int | None:
        value = self.metadata.get("page_number")
        return value if isinstance(value, int) else None

    @property
    def coordinates(self) -> tuple[float, float, float, float] | None:
        value = self.metadata.get("coordinates", self.metadata.get("bbox"))
        return tuple(value) if isinstance(value, (tuple, list)) and len(value) == 4 else None

    @property
    def text_as_html(self) -> str | None:
        value = self.metadata.get("text_as_html")
        return value if isinstance(value, str) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.category,
            "text": self.text,
            "element_id": self.element_id,
            "metadata": dict(self.metadata),
        }


class Title(Element):
    pass


class NarrativeText(Element):
    pass


class Table(Element):
    pass


class Image(Element):
    pass


class PageBreak(Element):
    pass


def partition_pdf(filename: object, **kwargs: object) -> list[Element]:
    include_page_breaks = bool(kwargs.pop("include_page_breaks", False))
    include_metadata = bool(kwargs.pop("include_metadata", True))
    with open_source(cast(PdfInput, filename)) as document:
        adapted = adapt_document(document)
        result: list[Element] = []
        items_by_page: dict[int, list[dict[str, Any]]] = {}
        for item in adapted.elements():
            page_number = item.get("page_number")
            if isinstance(page_number, int):
                items_by_page.setdefault(page_number, []).append(dict(item))
        for page_index, page in enumerate(adapted.structured_pages):
            if include_page_breaks and page_index:
                result.append(PageBreak("", ElementMetadata(page_number=page.page_number)))
            for item in items_by_page.get(page.page_number, []):
                element_type = str(item.get("type", "")).casefold()
                element_class = {
                    "heading": Title,
                    "block": NarrativeText,
                    "table": Table,
                    "image": Image,
                }.get(element_type, Element)
                metadata = ElementMetadata(item) if include_metadata else ElementMetadata()
                result.append(element_class(str(item.get("text", "")), metadata))
        return result


__all__ = (
    "Element",
    "ElementMetadata",
    "Image",
    "NarrativeText",
    "PageBreak",
    "Table",
    "Title",
    "partition_pdf",
)
