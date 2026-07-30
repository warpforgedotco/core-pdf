# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable, format-neutral document records."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

if TYPE_CHECKING:
    from core_pdf.impl.engine.structured.editor import DocumentEditor

BBox: TypeAlias = tuple[float, float, float, float]
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def internal_freeze(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({key: internal_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(internal_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(internal_freeze(item) for item in value)
    return value


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    QUOTE = "quote"
    CODE = "code"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TableCell:
    row: int
    column: int
    text: str
    row_span: int = 1
    column_span: int = 1
    bbox: BBox | None = None


@dataclass(frozen=True, slots=True)
class Table:
    order: int
    rows: tuple[tuple[TableCell, ...], ...] = ()
    bbox: BBox | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class Figure:
    order: int
    bbox: BBox | None = None
    kind: str = "figure"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class Link:
    bbox: BBox | None = None
    url: str | None = None
    link_type: str | None = None
    text: str = ""


@dataclass(frozen=True, slots=True)
class Annotation:
    subtype: str | None = None
    bbox: BBox | None = None
    contents: str = ""
    destination: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", internal_freeze(self.destination))


@dataclass(frozen=True, slots=True)
class FormField:
    name: str
    field_type: str
    value_text: str = ""
    bbox: BBox | None = None
    field_index: int | None = None


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    mark: bool = False
    superscript: bool = False
    subscript: bool = False


@dataclass(frozen=True, slots=True)
class TextLine:
    text: str
    break_before: int = 1
    bbox: BBox | None = None
    advance_bbox: BBox | None = None
    ink_bbox: BBox | None = None
    kind: str = "text-line"
    source: str = "unknown"
    confidence: float | None = None
    baseline: BBox | None = None
    contributing_sources: tuple[str, ...] = ()
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    mark: bool = False
    superscript: bool = False
    subscript: bool = False
    spans: tuple[TextSpan, ...] = ()

    def styled_spans(self) -> tuple[TextSpan, ...]:
        if self.spans:
            if self.underline or self.strikeout:
                return tuple(
                    replace(
                        span,
                        underline=span.underline or self.underline,
                        strikeout=span.strikeout or self.strikeout,
                    )
                    for span in self.spans
                )
            return self.spans
        return (
            TextSpan(
                text=self.text,
                bold=self.bold,
                italic=self.italic,
                underline=self.underline,
                strikeout=self.strikeout,
                mark=self.mark,
                superscript=self.superscript,
                subscript=self.subscript,
            ),
        )


@dataclass(frozen=True, slots=True)
class Block:
    order: int
    kind: BlockKind
    lines: tuple[TextLine, ...] = ()
    bbox: BBox | None = None
    column_index: int | None = None
    rotation: int = 0
    confidence: float | None = None
    level: int | None = None
    provenance: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        parts: list[str] = []
        for line in self.lines:
            if parts:
                parts.append("\n" * max(1, line.break_before))
            parts.append(line.text)
        return "".join(parts)


PageElement: TypeAlias = Block | Table | Figure


@dataclass(frozen=True, slots=True)
class Page:
    page_number: int
    page_label: str | None = None
    width: float = 0.0
    height: float = 0.0
    rotation: int = 0
    blocks: tuple[Block, ...] = ()
    page_class: str = "unknown"
    base_route: str = "unknown"
    confidence: float | None = None
    tables: tuple[Table, ...] = ()
    figures: tuple[Figure, ...] = ()
    links: tuple[Link, ...] = ()
    annotations: tuple[Annotation, ...] = ()
    form_fields: tuple[FormField, ...] = ()
    header: str = ""
    footer: str = ""

    @property
    def elements(self) -> tuple[PageElement, ...]:
        return tuple(
            sorted((*self.blocks, *self.tables, *self.figures), key=lambda item: item.order)
        )

    @property
    def text(self) -> str:
        parts: list[str] = []
        for element in self.elements:
            if isinstance(element, Block):
                text = element.text
            elif isinstance(element, Table):
                text = "\n".join("\t".join(cell.text for cell in row) for row in element.rows)
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        from core_pdf.impl.engine.structured.serialization import page_to_markdown

        return page_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl.engine.structured.serialization import page_to_html

        return page_to_html(self)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "warning"
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class Document:
    pages: tuple[Page, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))

    def edit(self) -> DocumentEditor:
        from core_pdf.impl.engine.structured.editor import DocumentEditor

        return DocumentEditor(self)

    @property
    def text(self) -> str:
        return "\f".join(page.text for page in self.pages) + "\f"

    def to_json_dict(self) -> dict[str, JsonValue]:
        from core_pdf.impl.engine.structured.serialization import document_to_json_dict

        return document_to_json_dict(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        from core_pdf.impl.engine.structured.serialization import document_to_json

        return document_to_json(self, indent=indent, sort_keys=sort_keys)

    def to_markdown(self) -> str:
        from core_pdf.impl.engine.structured.serialization import document_to_markdown

        return document_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl.engine.structured.serialization import document_to_html

        return document_to_html(self)
