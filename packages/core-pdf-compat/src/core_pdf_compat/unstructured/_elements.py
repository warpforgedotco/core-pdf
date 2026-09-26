from __future__ import annotations

from typing import Any, ClassVar

from core_pdf.impl.types import Record

frozen_setattr = object.__setattr__


class ElementMetadata(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


class Element(Record):
    __slots__ = ("text", "metadata")

    text: str
    metadata: ElementMetadata

    __fields__: ClassVar[tuple[str, ...]] = ("text", "metadata")
    __match_args__ = ("text", "metadata")

    def __init__(self, text: str, metadata: ElementMetadata | None = None) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "metadata", ElementMetadata() if metadata is None else metadata)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.text == other.text and self.metadata == other.metadata

    def __hash__(self) -> int:
        return hash((self.text, self.metadata))

    def _post_init(self) -> None:
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


class Header(Element):
    pass


class Footer(Element):
    pass


class ListItem(Element):
    pass


class EmailAddress(Element):
    pass


class Address(Element):
    pass
