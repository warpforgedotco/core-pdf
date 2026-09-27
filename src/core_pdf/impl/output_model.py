# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import replace
from enum import StrEnum
from operator import attrgetter
from typing import Any, ClassVar, NoReturn, Protocol, Self, TypeAlias

from core_pdf.impl.geometry import bbox_union
from core_pdf.impl.page_selection import PageSelection
from core_pdf.impl.text import TextWord, reconcile_text_words
from core_pdf.impl.types import GeneratedRecord, Rectangle, frozen_setattr

SCHEMA_VERSION = "5.0"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class FrozenDict(dict[Any, Any]):
    __slots__ = ()

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[FrozenDict], tuple[dict[Any, Any]]]:
        return (FrozenDict, (dict(self),))

    def _refuse(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


class FrozenMetadataFields:
    __slots__ = ()

    __frozen_fields__: ClassVar[tuple[str, ...]] = ("metadata",)

    def __post_init__(self) -> None:
        for name in self.__frozen_fields__:
            object.__setattr__(self, name, freeze(getattr(self, name)))


TEXT_STYLE_FIELDS = ("bold", "italic", "underline", "strikeout", "mark", "superscript", "subscript")
text_style_values = attrgetter(*TEXT_STYLE_FIELDS)


class TextStyleFields:
    __slots__ = ()

    bold: bool
    italic: bool
    underline: bool
    strikeout: bool
    mark: bool
    superscript: bool
    subscript: bool

    def style_values(self) -> tuple[bool, ...]:
        values: tuple[bool, ...] = text_style_values(self)
        return values

    def style_dict(self) -> dict[str, bool]:
        return dict(zip(TEXT_STYLE_FIELDS, self.style_values(), strict=True))


class PageElementLike(Protocol):
    @property
    def order(self) -> int: ...

    @property
    def bbox(self) -> Rectangle | None: ...

    @property
    def node_kind(self) -> str: ...

    @property
    def provenance(self) -> tuple[str, ...]: ...


def metadata_provenance(metadata: object) -> tuple[str, ...]:
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    return (str(source),) if source else ()


class ViewCache:
    __slots__ = ("_views",)

    _views: dict[str, Any]

    def cached_view[T](self, name: str, build: Callable[[], T]) -> T:
        try:
            views = self._views
        except AttributeError:
            views = {}
            object.__setattr__(self, "_views", views)
        try:
            return views[name]
        except KeyError:
            value = views[name] = build()
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


class TableCell(GeneratedRecord):
    row: int
    column: int
    text: str
    row_span: int = 1
    column_span: int = 1
    bbox: Rectangle | None = None


class TableRowBand(GeneratedRecord):
    index: int
    bbox: Rectangle | None = None
    kind: str = "body"
    confidence: float | None = None


class TableColumnBand(GeneratedRecord):
    index: int
    bbox: Rectangle | None = None
    confidence: float | None = None


class TableAssociatedText(GeneratedRecord):
    text: str
    bbox: Rectangle | None = None
    kind: str = "caption"
    confidence: float | None = None


class Table(FrozenMetadataFields, GeneratedRecord):
    order: int
    rows: tuple[tuple[TableCell, ...], ...]
    bbox: Rectangle | None
    confidence: float | None
    title: TableAssociatedText | None
    caption: TableAssociatedText | None
    row_bands: tuple[TableRowBand, ...]
    column_bands: tuple[TableColumnBand, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        order: int,
        rows: tuple[tuple[TableCell, ...], ...] = (),
        bbox: Rectangle | None = None,
        confidence: float | None = None,
        title: TableAssociatedText | None = None,
        caption: TableAssociatedText | None = None,
        row_bands: tuple[TableRowBand, ...] = (),
        column_bands: tuple[TableColumnBand, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        frozen_setattr(self, "order", order)
        frozen_setattr(self, "rows", rows)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "confidence", confidence)
        frozen_setattr(self, "title", title)
        frozen_setattr(self, "caption", caption)
        frozen_setattr(self, "row_bands", row_bands)
        frozen_setattr(self, "column_bands", column_bands)
        frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        self.__post_init__()

    node_kind: ClassVar[str] = "table"

    @property
    def provenance(self) -> tuple[str, ...]:
        return metadata_provenance(self.metadata)

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


class Figure(FrozenMetadataFields, GeneratedRecord):
    order: int
    bbox: Rectangle | None
    kind: str
    metadata: Mapping[str, Any]

    def __init__(
        self,
        order: int,
        bbox: Rectangle | None = None,
        kind: str = "figure",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        frozen_setattr(self, "order", order)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        self.__post_init__()

    node_kind: ClassVar[str] = "figure"

    @property
    def provenance(self) -> tuple[str, ...]:
        return metadata_provenance(self.metadata)


class Link(GeneratedRecord):
    bbox: Rectangle | None = None
    url: str | None = None
    link_type: str | None = None
    text: str = ""


class Annotation(FrozenMetadataFields, GeneratedRecord):
    subtype: str | None = None
    bbox: Rectangle | None = None
    contents: str = ""
    destination: Any = None

    __frozen_fields__: ClassVar[tuple[str, ...]] = ("destination",)


class FormField(GeneratedRecord):
    name: str
    field_type: str
    value_text: str = ""
    bbox: Rectangle | None = None
    field_index: int | None = None
    required: bool = False
    read_only: bool = False
    no_export: bool = False
    options: tuple[str, ...] = ()


class TextSpan(TextStyleFields, GeneratedRecord):
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    mark: bool = False
    superscript: bool = False
    subscript: bool = False


class TextLine(TextStyleFields, GeneratedRecord):
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
        object.__setattr__(self, "words", reconcile_text_words(self.text, self.words))
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
        return (TextSpan(self.text, *self.style_values()),)


class Block(GeneratedRecord):
    order: int
    kind: BlockKind
    lines: tuple[TextLine, ...] = ()
    bbox: Rectangle | None = None
    column_index: int | None = None
    rotation: int = 0
    confidence: float | None = None
    level: int | None = None
    provenance: tuple[str, ...] = ()

    node_kind: ClassVar[str] = "block"

    @property
    def text(self) -> str:
        parts: list[str] = []
        for line in self.lines:
            if parts:
                parts.append("\n" * max(1, line.break_before))
            parts.append(line.text)
        return "".join(parts)


PageElement: TypeAlias = Block | Table | Figure


class ContentNode(GeneratedRecord):
    node_id: int
    kind: str
    payload: PageElement
    page_number: int | None = None

    @property
    def bbox(self) -> Rectangle | None:
        return self.payload.bbox

    @property
    def provenance(self) -> tuple[str, ...]:
        element: PageElementLike = self.payload
        return tuple(element.provenance)


class TextView(ViewCache, GeneratedRecord):
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
        return self.cached_view("words", self.build_words)

    def build_words(self) -> tuple[TextWord, ...]:
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


class TextLineReference(GeneratedRecord):
    page_number: int
    line_index: int
    line: TextLine


class TableView(GeneratedRecord):
    tables: tuple[Table, ...]
    page_number: int | None = None


class TableReference(GeneratedRecord):
    page_number: int
    table_index: int
    table: Table


class DocumentTextView(GeneratedRecord):
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


class DocumentTableView(GeneratedRecord):
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


class Page(ViewCache, GeneratedRecord):
    page_number: int
    page_label: str | None
    width: float
    height: float
    rotation: int
    blocks: tuple[Block, ...]
    page_class: str
    base_route: str
    confidence: float | None
    tables: tuple[Table, ...]
    figures: tuple[Figure, ...]
    links: tuple[Link, ...]
    annotations: tuple[Annotation, ...]
    form_fields: tuple[FormField, ...]
    header: str
    footer: str
    diagnostics: tuple[Diagnostic, ...]
    cropbox: Rectangle | None
    user_unit: float
    _elements: tuple[PageElement, ...]

    __match_args__ = (
        "page_number",
        "page_label",
        "width",
        "height",
        "rotation",
        "blocks",
        "page_class",
        "base_route",
        "confidence",
        "tables",
        "figures",
        "links",
        "annotations",
        "form_fields",
        "header",
        "footer",
        "diagnostics",
        "cropbox",
    )
    __repr_fields__: ClassVar[tuple[str, ...]] = (*__match_args__, "user_unit")

    def __init__(
        self,
        page_number: int,
        page_label: str | None = None,
        width: float = 0.0,
        height: float = 0.0,
        rotation: int = 0,
        blocks: tuple[Block, ...] = (),
        page_class: str = UNKNOWN,
        base_route: str = UNKNOWN,
        confidence: float | None = None,
        tables: tuple[Table, ...] = (),
        figures: tuple[Figure, ...] = (),
        links: tuple[Link, ...] = (),
        annotations: tuple[Annotation, ...] = (),
        form_fields: tuple[FormField, ...] = (),
        header: str = "",
        footer: str = "",
        diagnostics: tuple[Diagnostic, ...] = (),
        cropbox: Rectangle | None = None,
        *,
        user_unit: float = 1.0,
    ) -> None:
        frozen_setattr(self, "page_number", page_number)
        frozen_setattr(self, "page_label", page_label)
        frozen_setattr(self, "width", width)
        frozen_setattr(self, "height", height)
        frozen_setattr(self, "rotation", rotation)
        frozen_setattr(self, "blocks", blocks)
        frozen_setattr(self, "page_class", page_class)
        frozen_setattr(self, "base_route", base_route)
        frozen_setattr(self, "confidence", confidence)
        frozen_setattr(self, "tables", tables)
        frozen_setattr(self, "figures", figures)
        frozen_setattr(self, "links", links)
        frozen_setattr(self, "annotations", annotations)
        frozen_setattr(self, "form_fields", form_fields)
        frozen_setattr(self, "header", header)
        frozen_setattr(self, "footer", footer)
        frozen_setattr(self, "diagnostics", diagnostics)
        frozen_setattr(self, "cropbox", cropbox)
        frozen_setattr(self, "user_unit", user_unit)
        frozen_setattr(self, "_elements", ())
        self.__post_init__()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_number == other.page_number
            and self.page_label == other.page_label
            and self.width == other.width
            and self.height == other.height
            and self.rotation == other.rotation
            and self.blocks == other.blocks
            and self.page_class == other.page_class
            and self.base_route == other.base_route
            and self.confidence == other.confidence
            and self.tables == other.tables
            and self.figures == other.figures
            and self.links == other.links
            and self.annotations == other.annotations
            and self.form_fields == other.form_fields
            and self.header == other.header
            and self.footer == other.footer
            and self.diagnostics == other.diagnostics
            and self.cropbox == other.cropbox
            and self.user_unit == other.user_unit
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.page_number,
                self.page_label,
                self.width,
                self.height,
                self.rotation,
                self.blocks,
                self.page_class,
                self.base_route,
                self.confidence,
                self.tables,
                self.figures,
                self.links,
                self.annotations,
                self.form_fields,
                self.header,
                self.footer,
                self.diagnostics,
                self.cropbox,
                self.user_unit,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        page_number = changes.pop("page_number", self.page_number)
        page_label = changes.pop("page_label", self.page_label)
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        rotation = changes.pop("rotation", self.rotation)
        blocks = changes.pop("blocks", self.blocks)
        page_class = changes.pop("page_class", self.page_class)
        base_route = changes.pop("base_route", self.base_route)
        confidence = changes.pop("confidence", self.confidence)
        tables = changes.pop("tables", self.tables)
        figures = changes.pop("figures", self.figures)
        links = changes.pop("links", self.links)
        annotations = changes.pop("annotations", self.annotations)
        form_fields = changes.pop("form_fields", self.form_fields)
        header = changes.pop("header", self.header)
        footer = changes.pop("footer", self.footer)
        diagnostics = changes.pop("diagnostics", self.diagnostics)
        cropbox = changes.pop("cropbox", self.cropbox)
        user_unit = changes.pop("user_unit", self.user_unit)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            page_number,
            page_label,
            width,
            height,
            rotation,
            blocks,
            page_class,
            base_route,
            confidence,
            tables,
            figures,
            links,
            annotations,
            form_fields,
            header,
            footer,
            diagnostics,
            cropbox,
            user_unit=user_unit,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_elements",
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
        return self._elements

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        return tuple(self.iter_nodes())

    def iter_nodes(self, start_id: int = 0) -> Iterator[ContentNode]:
        for index, element in enumerate(self.elements, start=start_id):
            yield ContentNode(
                node_id=index,
                kind=element.node_kind,
                payload=element,
                page_number=self.page_number,
            )

    @property
    def text_view(self) -> TextView:
        return self.cached_view(
            "text_view", lambda: TextView(self.elements, page_number=self.page_number)
        )

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
        from core_pdf.impl.output_serialize import page_to_markdown

        return page_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl.output_serialize import page_to_html

        return page_to_html(self)


class Diagnostic(GeneratedRecord):
    code: str
    message: str
    severity: str = "warning"
    page_number: int | None = None


class Document(FrozenMetadataFields, ViewCache, GeneratedRecord):
    pages: tuple[Page, ...]
    metadata: Mapping[str, Any]
    diagnostics: tuple[Diagnostic, ...]
    schema_version: str

    def __init__(
        self,
        pages: tuple[Page, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        diagnostics: tuple[Diagnostic, ...] = (),
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        frozen_setattr(self, "pages", pages)
        frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        frozen_setattr(self, "diagnostics", diagnostics)
        frozen_setattr(self, "schema_version", schema_version)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported structured schema version: {self.schema_version}")
        super().__post_init__()

    @property
    def text_view(self) -> DocumentTextView:
        return DocumentTextView(tuple(page.text_view for page in self.pages))

    @property
    def table_view(self) -> DocumentTableView:
        return DocumentTableView(tuple(page.table_view for page in self.pages))

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        return self.cached_view("nodes", self.build_nodes)

    def build_nodes(self) -> tuple[ContentNode, ...]:
        nodes: list[ContentNode] = []
        for page in self.pages:
            nodes.extend(page.iter_nodes(start_id=len(nodes)))
        return tuple(nodes)

    @property
    def text(self) -> str:
        return self.text_view.text

    @property
    def words(self) -> tuple[TextWord, ...]:
        return self.cached_view("words", lambda: self.text_view.words)

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return self.cached_view("lines", lambda: self.text_view.lines)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return self.cached_view("blocks", lambda: self.text_view.blocks)

    def to_json_dict(self) -> dict[str, JsonValue]:
        from core_pdf.impl.output_serialize import document_to_json_dict

        return document_to_json_dict(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        from core_pdf.impl.output_serialize import document_to_json

        return document_to_json(self, indent=indent, sort_keys=sort_keys)

    def to_markdown(self) -> str:
        from core_pdf.impl.output_serialize import document_to_markdown

        return document_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl.output_serialize import document_to_html

        return document_to_html(self)

    def to_csv(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl.output_serialize import document_to_csv

        return document_to_csv(self, pages=pages)

    def to_tei(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl.output_serialize import document_to_tei

        return document_to_tei(self, pages=pages)
