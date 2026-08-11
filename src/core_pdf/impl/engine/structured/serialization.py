# SPDX-License-Identifier: AGPL-3.0-only
"""Serialization views for the core-document IR."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from csv import writer
from html import escape
from io import StringIO
from typing import TypeVar
from xml.etree.ElementTree import Element, SubElement, tostring

from core_pdf.impl.engine.structured.model import (
    Annotation,
    Block,
    BlockKind,
    ContentNode,
    Diagnostic,
    Document,
    Figure,
    FormField,
    JsonValue,
    Link,
    Page,
    PageElement,
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableRowBand,
    TextLine,
)
from core_pdf.impl.pages import resolve_page_selection
from core_pdf.impl.types import PageSelection

ElementResultT = TypeVar("ElementResultT")


def node_to_json_dict(node: ContentNode) -> dict[str, JsonValue]:
    return {
        "node_id": node.node_id,
        "kind": node.kind,
        "page_number": node.page_number,
        "provenance": list(node.provenance),
        "payload": element_to_json_dict(node.payload),
    }


def diagnostic_to_json_dict(diagnostic: Diagnostic) -> dict[str, JsonValue]:
    return {
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "page_number": diagnostic.page_number,
    }


def document_to_json_dict(document: Document) -> dict[str, JsonValue]:
    return {
        "schema_version": document.schema_version,
        "metadata": json_safe(document.metadata),
        "nodes": [node_to_json_dict(node) for node in document.nodes],
        "table_references": [
            {
                "page_number": reference.page_number,
                "table_index": reference.table_index,
                "table": table_to_json_dict(reference.table),
            }
            for reference in document.table_view.references
        ],
        "line_references": [
            {
                "page_number": reference.page_number,
                "line_index": reference.line_index,
                "line": line_to_json_dict(reference.line),
            }
            for reference in document.text_view.line_references
        ],
        "pages": [page_to_json_dict(page) for page in document.pages],
        "diagnostics": [diagnostic_to_json_dict(diagnostic) for diagnostic in document.diagnostics],
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
        "nodes": [node_to_json_dict(node) for node in page.nodes],
        "elements": [element_to_json_dict(element) for element in page.elements],
        "blocks": [block_to_json_dict(block) for block in page.blocks],
        "tables": [table_to_json_dict(table) for table in page.tables],
        "structured_tables": [table_to_json_dict(table) for table in page.structured_tables],
        "figures": [figure_to_json_dict(figure) for figure in page.figures],
        "links": [link_to_json_dict(link) for link in page.links],
        "annotations": [annotation_to_json_dict(annotation) for annotation in page.annotations],
        "form_fields": [field_to_json_dict(field) for field in page.form_fields],
        "header": page.header,
        "footer": page.footer,
        "diagnostics": [diagnostic_to_json_dict(diagnostic) for diagnostic in page.diagnostics],
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
        "layout_bbox": bbox_to_json(table.layout_bbox),
        "content_bbox": bbox_to_json(table.content_bbox),
        "confidence": table.confidence,
        "title": table_associated_text_to_json_dict(table.title),
        "caption": table_associated_text_to_json_dict(table.caption),
        "row_bands": [table_row_band_to_json_dict(band) for band in table.row_bands],
        "column_bands": [table_column_band_to_json_dict(band) for band in table.column_bands],
        "metadata": json_safe(table.metadata),
        "rows": [[table_cell_to_json_dict(cell) for cell in row] for row in table.rows],
    }


def table_associated_text_to_json_dict(
    value: TableAssociatedText | None,
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return {
        "text": value.text,
        "bbox": bbox_to_json(value.bbox),
        "kind": value.kind,
        "confidence": value.confidence,
    }


def table_row_band_to_json_dict(band: TableRowBand) -> dict[str, JsonValue]:
    return {
        "index": band.index,
        "bbox": bbox_to_json(band.bbox),
        "kind": band.kind,
        "confidence": band.confidence,
    }


def table_column_band_to_json_dict(band: TableColumnBand) -> dict[str, JsonValue]:
    return {
        "index": band.index,
        "bbox": bbox_to_json(band.bbox),
        "confidence": band.confidence,
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


def internal_selected_pages(document: Document, pages: PageSelection | None) -> tuple[Page, ...]:
    """Return document pages narrowed by a 1-based page selection, in selection order."""
    if pages is None:
        return document.pages
    indexes = resolve_page_selection(pages, len(document.pages))
    return tuple(document.pages[index] for index in indexes)


def internal_page_lines(page: Page) -> tuple[TextLine, ...]:
    return tuple(line for block in page.blocks for line in block.lines)


def document_to_csv(document: Document, *, pages: PageSelection | None = None) -> str:
    """Export deterministic page text rows with geometry as CSV."""
    output = StringIO()
    rows = writer(output, lineterminator="\n")
    rows.writerow(("page_number", "line_index", "text", "x0", "y0", "x1", "y1"))
    for page in internal_selected_pages(document, pages):
        for index, line in enumerate(internal_page_lines(page)):
            bbox = line.bbox
            rows.writerow(
                (
                    page.page_number,
                    index,
                    line.text,
                    float(bbox[0]) if bbox is not None else "",
                    float(bbox[1]) if bbox is not None else "",
                    float(bbox[2]) if bbox is not None else "",
                    float(bbox[3]) if bbox is not None else "",
                )
            )
    return output.getvalue()


def document_to_tei(document: Document, *, pages: PageSelection | None = None) -> str:
    """Export page text as deterministic TEI-like XML with page boundaries."""
    root = Element("TEI")
    text = SubElement(root, "text")
    body = SubElement(text, "body")
    for page in internal_selected_pages(document, pages):
        SubElement(body, "pb", {"n": str(page.page_number)})
        for line in internal_page_lines(page):
            paragraph = SubElement(body, "p")
            paragraph.text = line.text
    return tostring(root, encoding="unicode", short_empty_elements=True)


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
        for element in internal_serialization_elements(page)
    ]
    return "\n\n".join(parts)


def internal_render_styled_line(
    line: TextLine,
    *,
    escape_text: bool,
    strikeout: tuple[str, str],
    bold: tuple[str, str],
    italic: tuple[str, str],
) -> str:
    """Wrap each styled span, innermost style first.

    Markdown and HTML differ only in escaping and in the three delimiters that
    have a Markdown spelling; `sup`/`sub`/`mark`/`u` have no Markdown equivalent
    and are emitted as inline HTML by both renderers.
    """
    rendered: list[str] = []
    for span in line.styled_spans():
        text = escape(span.text) if escape_text else span.text
        if span.superscript:
            text = f"<sup>{text}</sup>"
        elif span.subscript:
            text = f"<sub>{text}</sub>"
        if span.mark:
            text = f"<mark>{text}</mark>"
        if span.underline:
            text = f"<u>{text}</u>"
        if span.strikeout:
            text = f"{strikeout[0]}{text}{strikeout[1]}"
        if span.bold:
            text = f"{bold[0]}{text}{bold[1]}"
        if span.italic:
            text = f"{italic[0]}{text}{italic[1]}"
        rendered.append(text)
    return "".join(rendered)


def internal_markdown_line(line: TextLine) -> str:
    return internal_render_styled_line(
        line,
        escape_text=False,
        strikeout=("~~", "~~"),
        bold=("**", "**"),
        italic=("*", "*"),
    )


def internal_html_line(line: TextLine) -> str:
    return internal_render_styled_line(
        line,
        escape_text=True,
        strikeout=("<del>", "</del>"),
        bold=("<strong>", "</strong>"),
        italic=("<em>", "</em>"),
    )


def block_to_markdown(block: Block) -> str:
    text = "\n".join(internal_markdown_line(line) for line in block.lines)
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
        for element in internal_serialization_elements(page)
    ]
    rendered = "\n".join(parts)
    return f'<section data-page-number="{page.page_number}">{rendered}</section>'


def internal_serialization_elements(page: Page) -> tuple[PageElement, ...]:
    """Order page elements for rendering.

    The parse pipeline merges the annotated structured tables into
    ``page.tables`` at assembly time (see ``internal_merge_structured_tables``),
    so rendering only sorts the merged elements.  Hand-built pages may populate
    ``structured_tables`` alone, hence the fallback.
    """
    source_elements: tuple[PageElement, ...] = (
        *page.blocks,
        *(page.tables or page.structured_tables),
        *page.figures,
    )
    return tuple(sorted(source_elements, key=lambda item: item.order))


def block_to_html(block: Block) -> str:
    attributes = f' data-block-kind="{escape(block.kind.value)}"'

    if block.kind is BlockKind.HEADING:
        tag = f"h{block.level or 2}"
        heading = "<br />".join(internal_html_line(line) for line in block.lines)
        return f"<{tag}{attributes}>{heading}</{tag}>"
    if block.kind is BlockKind.LIST:
        items = "".join(f"<li>{escape(line.text)}</li>" for line in block.lines)
        return f"<ul{attributes}>{items}</ul>"
    text = "<br />".join(internal_html_line(line) for line in block.lines)
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
    associated_text = tuple(value for value in (table.title, table.caption) if value is not None)
    prefix = "".join(
        f'<div data-table-associated="{escape(value.kind)}">{escape(value.text)}</div>'
        for value in associated_text
    )
    bands = table.row_bands
    if not associated_text and all(band.kind in {"header", "body"} for band in bands):
        header, *body = table.rows
        head = "".join(internal_table_cell_to_html(cell, header=True) for cell in header)
        body_html = "".join(
            f"<tr>{''.join(internal_table_cell_to_html(cell) for cell in row)}</tr>" for row in body
        )
        return f"{prefix}<table><thead><tr>{head}</tr></thead><tbody>{body_html}</tbody></table>"

    header_rows: list[str] = []
    body_rows: list[str] = []
    for row, band in zip(table.rows, bands, strict=True):
        if band.kind in {"title", "caption"}:
            continue
        cells = "".join(
            internal_table_cell_to_html(cell, header=band.kind == "header") for cell in row
        )
        rendered = f'<tr data-row-kind="{escape(band.kind)}">{cells}</tr>'
        (header_rows if band.kind == "header" else body_rows).append(rendered)
    head_html = f"<thead>{''.join(header_rows)}</thead>" if header_rows else ""
    body_html = f"<tbody>{''.join(body_rows)}</tbody>" if body_rows else ""
    return f"{prefix}<table>{head_html}{body_html}</table>"


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
    "document_to_csv",
    "document_to_html",
    "document_to_json",
    "document_to_json_dict",
    "document_to_markdown",
    "document_to_tei",
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
