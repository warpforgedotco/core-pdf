"""Local Unstructured-style partition and element conversion APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.structured import document_elements

PdfInput: TypeAlias = Any


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


class UncategorizedText(Element):
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
    with PdfDocument.open(cast(PdfInput, filename)) as document:
        structured = document.structured_document
        result: list[Element] = []
        items_by_page: dict[int, list[Any]] = {}
        native_text_pages = {
            page.page_number
            for page in document.pages
            if any(run.visible and run.text for run in page.get_page_program().products.runs)
        }
        for item in document_elements(structured):
            if item.page_number not in native_text_pages:
                continue
            page = structured.pages[item.page_number - 1]
            source_item = next(
                (
                    value
                    for index, value in enumerate(page.elements)
                    if item.element_id == f"p{item.page_number}-e{index}"
                ),
                None,
            )
            lines = tuple(getattr(source_item, "lines", ()))
            if not item.text or (lines and all(line.source == "ocr" for line in lines)):
                continue
            items_by_page.setdefault(item.page_number, []).append(item)
        for page_index, page in enumerate(structured.pages):
            if include_page_breaks and page_index:
                result.append(PageBreak("", ElementMetadata(page_number=page.page_number)))
            for item in items_by_page.get(page.page_number, []):
                element_type = item.kind.casefold()
                text = " ".join(item.text.split())
                element_class = {
                    "heading": Title,
                    "table": Table,
                    "image": Image,
                }.get(element_type)
                if element_class is None:
                    words = text.split()
                    alphabetic = sum(character.isalpha() for character in text)
                    element_class = (
                        Title
                        if words
                        and len(words) <= 12
                        and alphabetic >= len(text) * 0.6
                        and not text.endswith((".", ":", ";", "?", "!"))
                        else (
                            NarrativeText
                            if len(words) >= 3 and alphabetic >= len(text) * 0.6
                            else UncategorizedText
                        )
                    )
                metadata = (
                    ElementMetadata(
                        {
                            "element_id": item.element_id,
                            "type": item.kind,
                            "page_number": item.page_number,
                            "bbox": item.bbox,
                            **item.metadata,
                        }
                    )
                    if include_metadata
                    else ElementMetadata()
                )
                result.append(element_class(text, metadata))
        return result


__all__ = (
    "Element",
    "ElementMetadata",
    "Image",
    "NarrativeText",
    "PageBreak",
    "Table",
    "Title",
    "UncategorizedText",
    "partition_pdf",
)
