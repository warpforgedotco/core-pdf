# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable, format-neutral document records."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

from core_pdf.impl.engine.model.geometry import bbox_union
from core_pdf.impl.pages import PageSelection
from core_pdf.impl.types import Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.engine.structured.editor import DocumentEditor

SCHEMA_VERSION = "5.0"
"""Schema version stamped on every structured :class:`Document`."""

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
    internal_content_bbox_cache: tuple[Rectangle | None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

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
        cached = self.internal_content_bbox_cache
        if cached is not None:
            return cached[0]
        boxes = [cell.bbox for row in self.rows for cell in row if cell.bbox is not None]
        result = bbox_union(boxes) if boxes else self.bbox
        object.__setattr__(self, "internal_content_bbox_cache", (result,))
        return result


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
    source: str = "unknown"
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
    internal_text_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def text(self) -> str:
        cached = self.internal_text_cache
        if cached is not None:
            return cached
        parts: list[str] = []
        for line in self.lines:
            if parts:
                parts.append("\n" * max(1, line.break_before))
            parts.append(line.text)
        joined = "".join(parts)
        object.__setattr__(self, "internal_text_cache", joined)
        return joined


PageElement: TypeAlias = Block | Table | Figure


@dataclass(frozen=True, slots=True)
class ContentNode:
    """A shared ordered graph node with a typed payload."""

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
    """Reading-order projection over the page's text nodes."""

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
                for word_index, word in enumerate(line.text.split()):
                    words.append(
                        TextWord(
                            text=word,
                            bbox=line.bbox,
                            line_index=line_index,
                            word_index=word_index,
                            block_index=block_index,
                            page_number=self.page_number,
                            source=line.source,
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
class TextWord:
    """A word projection derived from a normalized text line."""

    text: str
    bbox: Rectangle | None = None
    line_index: int = 0
    word_index: int = 0
    block_index: int = 0
    page_number: int | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class TextLineReference:
    """A document-owned reference to a normalized line and its source page."""

    page_number: int
    line_index: int
    line: TextLine


@dataclass(frozen=True, slots=True)
class TextRun:
    """Raw font-level text evidence exposed only through diagnostics."""

    text: str
    bbox: Rectangle
    font_name: str | None
    font_size: float
    is_vertical: bool
    visible: bool
    rotation: int
    seqno: int
    geometry_issues: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class TextDiagnostics:
    """Low-level text evidence separate from semantic text projections."""

    runs: tuple[TextRun, ...]


@dataclass(frozen=True, slots=True)
class TableView:
    """Structured table projection independent of reading-order text."""

    tables: tuple[Table, ...]
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class TableReference:
    """A document-owned reference to a table and its source page."""

    page_number: int
    table_index: int
    table: Table


@dataclass(frozen=True, slots=True)
class DocumentTextView:
    """Document-wide text projection retaining page boundaries."""

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
            words.extend(replace(word, page_number=page.page_number) for word in page.words)
        return tuple(words)

    @property
    def text(self) -> str:
        return "\f".join(page.text for page in self.pages) + "\f"


@dataclass(frozen=True, slots=True)
class DocumentTableView:
    """Document-wide structured table projection retaining page ownership."""

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
    diagnostics: tuple[Diagnostic, ...] = ()
    cropbox: Rectangle | None = None
    internal_elements_cache: tuple[PageElement, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def elements(self) -> tuple[PageElement, ...]:
        cached = self.internal_elements_cache
        if cached is not None:
            return cached
        ordered: list[PageElement] = sorted(
            (*self.blocks, *self.tables, *self.figures), key=lambda item: item.order
        )
        result = tuple(ordered)
        object.__setattr__(self, "internal_elements_cache", result)
        return result

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        return tuple(
            ContentNode(
                node_id=index,
                kind=type(element).__name__.casefold(),
                payload=element,
                page_number=self.page_number,
            )
            for index, element in enumerate(self.elements)
        )

    @property
    def text_view(self) -> TextView:
        return TextView(self.elements, page_number=self.page_number)

    @property
    def words(self) -> tuple[TextWord, ...]:
        """Return the canonical reading-order word projection for this page."""
        return self.text_view.words

    @property
    def table_view(self) -> TableView:
        return TableView(self.tables, page_number=self.page_number)

    @property
    def text(self) -> str:
        return self.text_view.text

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
        """Return one reading-order node stream with page ownership preserved."""
        nodes: list[ContentNode] = []
        for page in self.pages:
            offset = len(nodes)
            nodes.extend(replace(node, node_id=offset + node.node_id) for node in page.nodes)
        return tuple(nodes)

    def edit(self) -> DocumentEditor:
        from core_pdf.impl.engine.structured.editor import DocumentEditor

        return DocumentEditor(self)

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

    def to_csv(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl.engine.structured.serialization import document_to_csv

        return document_to_csv(self, pages=pages)

    def to_tei(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl.engine.structured.serialization import document_to_tei

        return document_to_tei(self, pages=pages)
