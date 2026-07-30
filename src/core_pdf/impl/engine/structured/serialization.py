# SPDX-License-Identifier: AGPL-3.0-only
"""Serialization views for the core-document IR."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from html import escape
from typing import TypeVar

from core_pdf.impl.engine.structured.model import (
    Annotation,
    Block,
    BlockKind,
    Document,
    Figure,
    FormField,
    JsonValue,
    Link,
    Page,
    PageElement,
    Table,
    TableCell,
    TextLine,
)

ElementResultT = TypeVar("ElementResultT")


def document_to_json_dict(document: Document) -> dict[str, JsonValue]:
    return {
        "schema_version": document.schema_version,
        "metadata": json_safe(document.metadata),
        "pages": [page_to_json_dict(page) for page in document.pages],
        "diagnostics": [
            {
                "code": diagnostic.code,
                "message": diagnostic.message,
                "severity": diagnostic.severity,
                "page_number": diagnostic.page_number,
            }
            for diagnostic in document.diagnostics
        ],
    }


def page_to_json_dict(page: Page) -> dict[str, JsonValue]:
    return {
        "page_number": page.page_number,
        "page_label": page.page_label,
        "width": page.width,
        "height": page.height,
        "rotation": page.rotation,
        "page_class": page.page_class,
        "base_route": page.base_route,
        "confidence": page.confidence,
        "elements": [element_to_json_dict(element) for element in page.elements],
        "blocks": [block_to_json_dict(block) for block in page.blocks],
        "tables": [table_to_json_dict(table) for table in page.tables],
        "figures": [figure_to_json_dict(figure) for figure in page.figures],
        "links": [link_to_json_dict(link) for link in page.links],
        "annotations": [annotation_to_json_dict(annotation) for annotation in page.annotations],
        "form_fields": [field_to_json_dict(field) for field in page.form_fields],
        "header": page.header,
        "footer": page.footer,
    }


def element_to_json_dict(element: Block | Table | Figure) -> dict[str, JsonValue]:
    return internal_map_page_element(
        element,
        block=lambda block: internal_add_element_type(block_to_json_dict(block), "block"),
        table=lambda table: internal_add_element_type(table_to_json_dict(table), "table"),
        figure=lambda figure: internal_add_element_type(figure_to_json_dict(figure), "figure"),
    )


def internal_map_page_element(
    element: PageElement,
    *,
    block: Callable[[Block], ElementResultT],
    table: Callable[[Table], ElementResultT],
    figure: Callable[[Figure], ElementResultT],
) -> ElementResultT:
    match element:
        case Block():
            return block(element)
        case Table():
            return table(element)
        case Figure():
            return figure(element)
    raise TypeError(f"unsupported page element: {type(element).__name__}")


def internal_add_element_type(
    payload: dict[str, JsonValue], element_type: str
) -> dict[str, JsonValue]:
    payload["element_type"] = element_type
    return payload


def block_to_json_dict(block: Block) -> dict[str, JsonValue]:
    return {
        "order": block.order,
        "kind": block.kind.value,
        "bbox": bbox_to_json(block.bbox),
        "column_index": block.column_index,
        "rotation": block.rotation,
        "confidence": block.confidence,
        "level": block.level,
        "provenance": list(block.provenance),
        "lines": [line_to_json_dict(line) for line in block.lines],
    }


def line_to_json_dict(line: TextLine) -> dict[str, JsonValue]:
    return {
        "text": line.text,
        "break_before": line.break_before,
        "bbox": bbox_to_json(line.bbox),
        "advance_bbox": bbox_to_json(line.advance_bbox),
        "ink_bbox": bbox_to_json(line.ink_bbox),
        "kind": line.kind,
        "source": line.source,
        "confidence": line.confidence,
        "bold": line.bold,
        "italic": line.italic,
        "underline": line.underline,
        "strikeout": line.strikeout,
        "mark": line.mark,
        "superscript": line.superscript,
        "subscript": line.subscript,
        "spans": [
            {
                "text": span.text,
                "bold": span.bold,
                "italic": span.italic,
                "underline": span.underline,
                "strikeout": span.strikeout,
                "mark": span.mark,
                "superscript": span.superscript,
                "subscript": span.subscript,
            }
            for span in line.styled_spans()
        ],
        "baseline": bbox_to_json(line.baseline),
        "contributing_sources": list(line.contributing_sources),
    }


def table_to_json_dict(table: Table) -> dict[str, JsonValue]:
    return {
        "order": table.order,
        "bbox": bbox_to_json(table.bbox),
        "confidence": table.confidence,
        "metadata": json_safe(table.metadata),
        "rows": [[table_cell_to_json_dict(cell) for cell in row] for row in table.rows],
    }


def table_cell_to_json_dict(cell: TableCell) -> dict[str, JsonValue]:
    return {
        "row": cell.row,
        "column": cell.column,
        "text": cell.text,
        "row_span": cell.row_span,
        "column_span": cell.column_span,
        "bbox": bbox_to_json(cell.bbox),
    }


def figure_to_json_dict(figure: Figure) -> dict[str, JsonValue]:
    return {
        "order": figure.order,
        "bbox": bbox_to_json(figure.bbox),
        "kind": figure.kind,
        "metadata": json_safe(figure.metadata),
    }


def link_to_json_dict(link: Link) -> dict[str, JsonValue]:
    return {
        "bbox": bbox_to_json(link.bbox),
        "url": link.url,
        "link_type": link.link_type,
        "text": link.text,
    }


def annotation_to_json_dict(annotation: Annotation) -> dict[str, JsonValue]:
    return {
        "subtype": annotation.subtype,
        "bbox": bbox_to_json(annotation.bbox),
        "contents": annotation.contents,
        "destination": json_safe(annotation.destination),
    }


def field_to_json_dict(field: FormField) -> dict[str, JsonValue]:
    return {
        "name": field.name,
        "field_type": field.field_type,
        "value_text": field.value_text,
        "bbox": bbox_to_json(field.bbox),
        "field_index": field.field_index,
    }


def document_to_json(document: Document, *, indent: int | None, sort_keys: bool) -> str:
    return json.dumps(document_to_json_dict(document), indent=indent, sort_keys=sort_keys)


def bbox_to_json(bbox: tuple[float, float, float, float] | None) -> list[JsonValue] | None:
    return list(bbox) if bbox is not None else None


def document_to_markdown(document: Document) -> str:
    return "\f".join(page_to_markdown(page) for page in document.pages) + "\f"


def page_to_markdown(page: Page) -> str:
    parts = [
        internal_map_page_element(
            element,
            block=block_to_markdown,
            table=table_to_markdown,
            figure=lambda figure: f"> [Figure: {figure.kind}]",
        )
        for element in page.elements
    ]
    return "\n\n".join(parts)


def block_to_markdown(block: Block) -> str:
    def format_line(line: TextLine) -> str:
        rendered: list[str] = []
        for span in line.styled_spans():
            text = span.text
            if span.superscript:
                text = f"<sup>{text}</sup>"
            elif span.subscript:
                text = f"<sub>{text}</sub>"
            if span.mark:
                text = f"<mark>{text}</mark>"
            if span.underline:
                text = f"<u>{text}</u>"
            if span.strikeout:
                text = f"~~{text}~~"
            if span.bold:
                text = f"**{text}**"
            if span.italic:
                text = f"*{text}*"
            rendered.append(text)
        return "".join(rendered)

    text = "\n".join(format_line(line) for line in block.lines)
    if block.kind is BlockKind.HEADING:
        return f"{'#' * (block.level or 2)} {text}"
    if block.kind is BlockKind.LIST:
        return "\n".join(
            line.text if internal_has_list_marker(line.text) else f"- {line.text}"
            for line in block.lines
        )
    return text


def document_to_html(document: Document) -> str:
    pages = "\n".join(page_to_html(page) for page in document.pages)
    return f'<article data-schema-version="{escape(document.schema_version)}">{pages}</article>'


def page_to_html(page: Page) -> str:
    parts = [
        internal_map_page_element(
            element,
            block=block_to_html,
            table=table_to_html,
            figure=lambda figure: f'<figure data-kind="{escape(figure.kind)}"></figure>',
        )
        for element in page.elements
    ]
    rendered = "\n".join(parts)
    return f'<section data-page-number="{page.page_number}">{rendered}</section>'


def block_to_html(block: Block) -> str:
    attributes = f' data-block-kind="{escape(block.kind.value)}"'

    def format_line(line: TextLine) -> str:
        rendered: list[str] = []
        for span in line.styled_spans():
            text = escape(span.text)
            if span.superscript:
                text = f"<sup>{text}</sup>"
            elif span.subscript:
                text = f"<sub>{text}</sub>"
            if span.mark:
                text = f"<mark>{text}</mark>"
            if span.underline:
                text = f"<u>{text}</u>"
            if span.strikeout:
                text = f"<del>{text}</del>"
            if span.bold:
                text = f"<strong>{text}</strong>"
            if span.italic:
                text = f"<em>{text}</em>"
            rendered.append(text)
        return "".join(rendered)

    if block.kind is BlockKind.HEADING:
        tag = f"h{block.level or 2}"
        return (
            f"<{tag}{attributes}>{'<br />'.join(format_line(line) for line in block.lines)}</{tag}>"
        )
    if block.kind is BlockKind.LIST:
        items = "".join(f"<li>{escape(line.text)}</li>" for line in block.lines)
        return f"<ul{attributes}>{items}</ul>"
    rendered_lines = []
    for line in block.lines:
        rendered_lines.append(format_line(line))
    text = "<br />".join(rendered_lines)
    return f"<p{attributes}>{text}</p>"


def table_to_markdown(table: Table) -> str:
    """Render a structured table as HTML embedded in Markdown.

    Pipe-table syntax cannot represent merged cells, so using it here discarded
    row and column spans already recovered by the extraction pipeline.  HTML is
    valid inline Markdown and preserves the complete table structure for both
    renderers and downstream document consumers.
    """
    return table_to_html(table)


def table_to_html(table: Table) -> str:
    if not table.rows:
        return "<table></table>"
    header, *body = table.rows
    head = "".join(internal_table_cell_to_html(cell, header=True) for cell in header)
    body_html = "".join(
        f"<tr>{''.join(internal_table_cell_to_html(cell) for cell in row)}</tr>" for row in body
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body_html}</tbody></table>"


def internal_table_cell_to_html(cell: TableCell, *, header: bool = False) -> str:
    tag = "th" if header else "td"
    spans = "".join(
        (
            f' rowspan="{cell.row_span}"' if cell.row_span > 1 else "",
            f' colspan="{cell.column_span}"' if cell.column_span > 1 else "",
        )
    )
    return f"<{tag}{spans}>{escape(cell.text)}</{tag}>"


def internal_has_list_marker(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    return stripped[0] in "-*•▪◦" or (
        len(stripped) > 1 and stripped[0].isalnum() and stripped[1:2] in {".", ")"}
    )


def json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


__all__ = (
    "block_to_html",
    "block_to_json_dict",
    "block_to_markdown",
    "annotation_to_json_dict",
    "field_to_json_dict",
    "figure_to_json_dict",
    "document_to_html",
    "document_to_json",
    "document_to_json_dict",
    "document_to_markdown",
    "element_to_json_dict",
    "json_safe",
    "page_to_html",
    "page_to_json_dict",
    "page_to_markdown",
    "table_cell_to_json_dict",
    "table_to_html",
    "table_to_markdown",
    "table_to_json_dict",
)
