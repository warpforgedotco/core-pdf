# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterator, Mapping
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias

from core_pdf.impl._impl.model.geometry import bbox_union
from core_pdf.impl._impl.model.page_selection import PageSelection
from core_pdf.impl._impl.model.text import internal_reconcile_text_words
from core_pdf.impl.types import Rectangle, TextWord

SCHEMA_VERSION = "5.0"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def internal_freeze(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({key: internal_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(internal_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(internal_freeze(item) for item in value)
    return value


UNKNOWN = "unknown"


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
    bbox: Rectangle | None = None


@dataclass(frozen=True, slots=True)
class TableRowBand:
    index: int
    bbox: Rectangle | None = None
    kind: str = "body"
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TableColumnBand:
    index: int
    bbox: Rectangle | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TableAssociatedText:
    text: str
    bbox: Rectangle | None = None
    kind: str = "caption"
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Table:
    order: int
    rows: tuple[tuple[TableCell, ...], ...] = ()
    bbox: Rectangle | None = None
    confidence: float | None = None
    title: TableAssociatedText | None = None
    caption: TableAssociatedText | None = None
    row_bands: tuple[TableRowBand, ...] = ()
    column_bands: tuple[TableColumnBand, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))

    @property
    def layout_bbox(self) -> Rectangle | None:
        boxes = [box for box in (self.bbox, self.title_bbox, self.caption_bbox) if box is not None]
        return bbox_union(boxes)

    @property
    def title_bbox(self) -> Rectangle | None:
        return self.title.bbox if self.title is not None else None

    @property
    def caption_bbox(self) -> Rectangle | None:
        return self.caption.bbox if self.caption is not None else None

    @property
    def content_bbox(self) -> Rectangle | None:
        boxes = [cell.bbox for row in self.rows for cell in row if cell.bbox is not None]
        return bbox_union(boxes) if boxes else self.bbox


@dataclass(frozen=True, slots=True)
class Figure:
    order: int
    bbox: Rectangle | None = None
    kind: str = "figure"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class Link:
    bbox: Rectangle | None = None
    url: str | None = None
    link_type: str | None = None
    text: str = ""


@dataclass(frozen=True, slots=True)
class Annotation:
    subtype: str | None = None
    bbox: Rectangle | None = None
    contents: str = ""
    destination: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", internal_freeze(self.destination))


@dataclass(frozen=True, slots=True)
class FormField:
    name: str
    field_type: str
    value_text: str = ""
    bbox: Rectangle | None = None
    field_index: int | None = None
    required: bool = False
    read_only: bool = False
    no_export: bool = False
    options: tuple[str, ...] = ()


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
    bbox: Rectangle | None = None
    advance_bbox: Rectangle | None = None
    ink_bbox: Rectangle | None = None
    kind: str = "text-line"
    source: str = UNKNOWN
    confidence: float | None = None
    baseline: Rectangle | None = None
    contributing_sources: tuple[str, ...] = ()
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    mark: bool = False
    superscript: bool = False
    subscript: bool = False
    spans: tuple[TextSpan, ...] = ()
    words: tuple[TextWord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "words", internal_reconcile_text_words(self.text, self.words))
        if self.spans and "".join(span.text for span in self.spans) != self.text:
            object.__setattr__(self, "spans", ())

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
    bbox: Rectangle | None = None
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
class ContentNode:
    node_id: int
    kind: str
    payload: PageElement
    page_number: int | None = None

    @property
    def bbox(self) -> Rectangle | None:
        return self.payload.bbox

    @property
    def provenance(self) -> tuple[str, ...]:
        value = getattr(self.payload, "provenance", ())
        if value:
            return tuple(value)
        metadata = getattr(self.payload, "metadata", {})
        source = metadata.get("source") if isinstance(metadata, MappingABC) else None
        return (str(source),) if source else ()


@dataclass(frozen=True, slots=True)
class TextView:
    elements: tuple[PageElement, ...]
    page_number: int | None = None

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return tuple(line for block in self.blocks for line in block.lines)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(element for element in self.elements if isinstance(element, Block))

    @property
    def words(self) -> tuple[TextWord, ...]:
        words: list[TextWord] = []
        line_index = 0
        for block_index, block in enumerate(self.blocks):
            for line in block.lines:
                for word_index, word in enumerate(line.words):
                    words.append(
                        replace(
                            word,
                            line_index=line_index,
                            word_index=word_index,
                            block_index=block_index,
                            page_number=self.page_number,
                            source=line.source if word.source == UNKNOWN else word.source,
                        )
                    )
                line_index += 1
        return tuple(words)

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


@dataclass(frozen=True, slots=True)
class TextLineReference:
    page_number: int
    line_index: int
    line: TextLine


@dataclass(frozen=True, slots=True)
class TableView:
    tables: tuple[Table, ...]
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class TableReference:
    page_number: int
    table_index: int
    table: Table


@dataclass(frozen=True, slots=True)
class DocumentTextView:
    pages: tuple[TextView, ...]

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return tuple(line for page in self.pages for line in page.lines)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(block for page in self.pages for block in page.blocks)

    @property
    def line_references(self) -> tuple[TextLineReference, ...]:
        return tuple(
            TextLineReference(
                page_number=(page.page_number if page.page_number is not None else page_number),
                line_index=index,
                line=line,
            )
            for page_number, page in enumerate(self.pages, start=1)
            for index, line in enumerate(page.lines)
        )

    @property
    def words(self) -> tuple[TextWord, ...]:
        words: list[TextWord] = []
        for page in self.pages:
            words.extend(page.words)
        return tuple(words)

    @property
    def text(self) -> str:
        return "\f".join(page.text for page in self.pages) + "\f"


@dataclass(frozen=True, slots=True)
class DocumentTableView:
    pages: tuple[TableView, ...]

    @property
    def tables(self) -> tuple[Table, ...]:
        return tuple(table for page in self.pages for table in page.tables)

    @property
    def references(self) -> tuple[TableReference, ...]:
        return tuple(
            TableReference(
                page_number=(page.page_number if page.page_number is not None else page_number),
                table_index=table_index,
                table=table,
            )
            for page_number, page in enumerate(self.pages, start=1)
            for table_index, table in enumerate(page.tables)
        )


@dataclass(frozen=True, slots=True)
class Page:
    page_number: int
    page_label: str | None = None
    width: float = 0.0
    height: float = 0.0
    rotation: int = 0
    blocks: tuple[Block, ...] = ()
    page_class: str = UNKNOWN
    base_route: str = UNKNOWN
    confidence: float | None = None
    tables: tuple[Table, ...] = ()
    figures: tuple[Figure, ...] = ()
    links: tuple[Link, ...] = ()
    annotations: tuple[Annotation, ...] = ()
    form_fields: tuple[FormField, ...] = ()
    header: str = ""
    footer: str = ""
    diagnostics: tuple[Diagnostic, ...] = ()
    cropbox: Rectangle | None = None
    user_unit: float = field(default=1.0, kw_only=True)
    internal_elements: tuple[PageElement, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "internal_elements",
            tuple(sorted((*self.blocks, *self.tables, *self.figures), key=lambda item: item.order)),
        )

    @property
    def width_points(self) -> float:
        return self.width * self.user_unit

    @property
    def height_points(self) -> float:
        return self.height * self.user_unit

    @property
    def elements(self) -> tuple[PageElement, ...]:
        return self.internal_elements

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        return tuple(self.internal_nodes())

    def internal_nodes(self, start_id: int = 0) -> Iterator[ContentNode]:
        for index, element in enumerate(self.elements, start=start_id):
            yield ContentNode(
                node_id=index,
                kind=type(element).__name__.casefold(),
                payload=element,
                page_number=self.page_number,
            )

    @property
    def text_view(self) -> TextView:
        return TextView(self.elements, page_number=self.page_number)

    @property
    def words(self) -> tuple[TextWord, ...]:
        return self.text_view.words

    @property
    def table_view(self) -> TableView:
        return TableView(self.tables, page_number=self.page_number)

    @property
    def text(self) -> str:
        return self.text_view.text

    def to_markdown(self) -> str:
        from core_pdf.impl._impl.output.serialize import page_to_markdown

        return page_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl._impl.output.serialize import page_to_html

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
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported structured schema version: {self.schema_version}")
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))

    @property
    def text_view(self) -> DocumentTextView:
        return DocumentTextView(tuple(page.text_view for page in self.pages))

    @property
    def table_view(self) -> DocumentTableView:
        return DocumentTableView(tuple(page.table_view for page in self.pages))

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        nodes: list[ContentNode] = []
        for page in self.pages:
            nodes.extend(page.internal_nodes(start_id=len(nodes)))
        return tuple(nodes)

    @property
    def text(self) -> str:
        return self.text_view.text

    @property
    def words(self) -> tuple[TextWord, ...]:
        return self.text_view.words

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return self.text_view.lines

    @property
    def blocks(self) -> tuple[Block, ...]:
        return self.text_view.blocks

    def to_json_dict(self) -> dict[str, JsonValue]:
        from core_pdf.impl._impl.output.serialize import document_to_json_dict

        return document_to_json_dict(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        from core_pdf.impl._impl.output.serialize import document_to_json

        return document_to_json(self, indent=indent, sort_keys=sort_keys)

    def to_markdown(self) -> str:
        from core_pdf.impl._impl.output.serialize import document_to_markdown

        return document_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl._impl.output.serialize import document_to_html

        return document_to_html(self)

    def to_csv(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl._impl.output.serialize import document_to_csv

        return document_to_csv(self, pages=pages)

    def to_tei(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl._impl.output.serialize import document_to_tei

        return document_to_tei(self, pages=pages)
