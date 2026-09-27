# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from csv import writer
from html import escape
from io import StringIO
from typing import Protocol
from xml.etree.ElementTree import Element, SubElement, tostring

from core_pdf.impl.output_model import (
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
from core_pdf.impl.page_selection import PageSelection, resolve_page_selection
from core_pdf.impl.types import GeneratedRecord

LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•▪◦]|(?:\d+|[^\W_])[.)])[ \t]*")


class PageIds:
    __slots__ = ("page_id",)

    def __init__(self, page_number: int) -> None:
        self.page_id = f"p{page_number}"

    def record(self, kind: str, index: int) -> str:
        return f"{self.page_id}:{kind}:{index}"

    def records(self, kind: str, count: int) -> list[JsonValue]:
        return [self.record(kind, index) for index in range(count)]

    def identities(self, kind: str, items: Iterable[object]) -> dict[int, str]:
        unique = dict.fromkeys(id(item) for item in items)
        return {identity: self.record(kind, index) for index, identity in enumerate(unique)}


def node_to_json_dict(
    node: ContentNode,
    *,
    node_id: str,
    page_id: str,
    target_id: str,
) -> dict[str, JsonValue]:
    return {
        "id": node_id,
        "page_id": page_id,
        "kind": node.kind,
        "target_id": target_id,
        "provenance": list(node.provenance),
    }


def diagnostic_to_json_dict(diagnostic: Diagnostic) -> dict[str, JsonValue]:
    return {
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "page_number": diagnostic.page_number,
    }


def document_to_json_dict(document: Document) -> dict[str, JsonValue]:
    pages: list[JsonValue] = []
    nodes: list[JsonValue] = []
    blocks: list[JsonValue] = []
    lines: list[JsonValue] = []
    tables: list[JsonValue] = []
    figures: list[JsonValue] = []
    links: list[JsonValue] = []
    annotations: list[JsonValue] = []
    form_fields: list[JsonValue] = []
    seen_page_ids: set[str] = set()

    for page in document.pages:
        ids = PageIds(page.page_number)
        page_id = ids.page_id
        if page_id in seen_page_ids:
            raise ValueError(f"duplicate structured page id: {page_id}")
        seen_page_ids.add(page_id)

        page_blocks = {id(block): block for block in page.blocks}
        page_tables = {id(table): table for table in page.tables}
        page_figures = {id(figure): figure for figure in page.figures}
        block_ids = ids.identities("block", page_blocks.values())
        table_ids = ids.identities("table", page_tables.values())
        figure_ids = ids.identities("figure", page_figures.values())

        line_ids: dict[int, str] = {}
        for block in page_blocks.values():
            for line in block.lines:
                identity = id(line)
                if identity not in line_ids:
                    line_ids[identity] = ids.record("line", len(line_ids))
                    lines.append(
                        {
                            "id": line_ids[identity],
                            "page_id": page_id,
                            **line_to_json_dict(line),
                        }
                    )

        blocks.extend(
            {
                "id": block_ids[id(block)],
                "page_id": page_id,
                **block_payload(block),
                "line_ids": [line_ids[id(line)] for line in block.lines],
            }
            for block in page_blocks.values()
        )
        tables.extend(
            {
                "id": table_ids[id(table)],
                "page_id": page_id,
                **table_to_json_dict(table),
            }
            for table in page_tables.values()
        )
        figures.extend(
            {
                "id": figure_ids[id(figure)],
                "page_id": page_id,
                **figure_to_json_dict(figure),
            }
            for figure in page_figures.values()
        )

        target_ids = {**block_ids, **table_ids, **figure_ids}
        page_node_ids: list[JsonValue] = []
        for index, node in enumerate(page.nodes):
            node_id = ids.record("node", index)
            page_node_ids.append(node_id)
            nodes.append(
                node_to_json_dict(
                    node,
                    node_id=node_id,
                    page_id=page_id,
                    target_id=target_ids[id(node.payload)],
                )
            )

        page_link_ids = ids.records("link", len(page.links))
        page_annotation_ids = ids.records("annotation", len(page.annotations))
        page_field_ids = ids.records("form-field", len(page.form_fields))
        links.extend(
            {"id": record_id, "page_id": page_id, **link_to_json_dict(link)}
            for record_id, link in zip(page_link_ids, page.links, strict=True)
        )
        annotations.extend(
            {"id": record_id, "page_id": page_id, **annotation_to_json_dict(annotation)}
            for record_id, annotation in zip(page_annotation_ids, page.annotations, strict=True)
        )
        form_fields.extend(
            {"id": record_id, "page_id": page_id, **field_to_json_dict(field)}
            for record_id, field in zip(page_field_ids, page.form_fields, strict=True)
        )
        pages.append(
            {
                "id": page_id,
                "page_number": page.page_number,
                "page_label": page.page_label,
                "width": page.width,
                "height": page.height,
                "user_unit": page.user_unit,
                "rotation": page.rotation,
                "cropbox": bbox_to_json(page.cropbox),
                "page_class": page.page_class,
                "base_route": page.base_route,
                "confidence": page.confidence,
                "node_ids": page_node_ids,
                "link_ids": page_link_ids,
                "annotation_ids": page_annotation_ids,
                "form_field_ids": page_field_ids,
                "header": page.header,
                "footer": page.footer,
                "diagnostics": [
                    diagnostic_to_json_dict(diagnostic) for diagnostic in page.diagnostics
                ],
            }
        )

    return {
        "schema_version": document.schema_version,
        "metadata": json_safe(document.metadata, path="$.metadata"),
        "pages": pages,
        "nodes": nodes,
        "blocks": blocks,
        "lines": lines,
        "tables": tables,
        "figures": figures,
        "links": links,
        "annotations": annotations,
        "form_fields": form_fields,
        "diagnostics": [diagnostic_to_json_dict(diagnostic) for diagnostic in document.diagnostics],
    }


def map_page_element[ElementResultT](
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


def block_payload(block: Block) -> dict[str, JsonValue]:
    return {
        "order": block.order,
        "kind": block.kind.value,
        "bbox": bbox_to_json(block.bbox),
        "column_index": block.column_index,
        "rotation": block.rotation,
        "confidence": block.confidence,
        "level": block.level,
        "provenance": list(block.provenance),
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
        **line.style_dict(),
        "spans": [{"text": span.text, **span.style_dict()} for span in line.styled_spans()],
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
        "required": field.required,
        "read_only": field.read_only,
        "no_export": field.no_export,
        "options": list(field.options),
    }


def document_to_json(document: Document, *, indent: int | None, sort_keys: bool) -> str:
    return json.dumps(document_to_json_dict(document), indent=indent, sort_keys=sort_keys)


def bbox_to_json(bbox: tuple[float, float, float, float] | None) -> list[JsonValue] | None:
    return list(bbox) if bbox is not None else None


def selected_pages(document: Document, pages: PageSelection | None) -> tuple[Page, ...]:
    if pages is None:
        return document.pages
    indexes = resolve_page_selection(pages, len(document.pages))
    return tuple(document.pages[index] for index in indexes)


def iter_selected_lines(
    document: Document, pages: PageSelection | None
) -> Iterator[tuple[Page, tuple[TextLine, ...]]]:
    for page in selected_pages(document, pages):
        yield page, page.text_view.lines


def document_to_csv(document: Document, *, pages: PageSelection | None = None) -> str:
    output = StringIO()
    rows = writer(output, lineterminator="\n")
    rows.writerow(("page_number", "line_index", "text", "x0", "y0", "x1", "y1"))
    for page, lines in iter_selected_lines(document, pages):
        for index, line in enumerate(lines):
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
    root = Element("TEI")
    text = SubElement(root, "text")
    body = SubElement(text, "body")
    for page, lines in iter_selected_lines(document, pages):
        SubElement(body, "pb", {"n": str(page.page_number)})
        for line in lines:
            paragraph = SubElement(body, "p")
            paragraph.text = line.text
    return tostring(root, encoding="unicode", short_empty_elements=True)


class InlineMarkup(GeneratedRecord):
    escape_text: bool
    strikeout: tuple[str, str]
    bold: tuple[str, str]
    italic: tuple[str, str]


MARKDOWN_MARKUP = InlineMarkup(False, ("~~", "~~"), ("**", "**"), ("*", "*"))
HTML_MARKUP = InlineMarkup(True, ("<del>", "</del>"), ("<strong>", "</strong>"), ("<em>", "</em>"))


def render_styled_line(line: TextLine, markup: InlineMarkup, *, start: int = 0) -> str:
    rendered: list[str] = []
    for span in line.styled_spans():
        text = span.text
        if start:
            text = text[start:]
            start = max(0, start - len(span.text))
            if not text:
                continue
        text = escape(text) if markup.escape_text else text
        if span.superscript:
            text = f"<sup>{text}</sup>"
        elif span.subscript:
            text = f"<sub>{text}</sub>"
        if span.mark:
            text = f"<mark>{text}</mark>"
        if span.underline:
            text = f"<u>{text}</u>"
        if span.strikeout:
            text = f"{markup.strikeout[0]}{text}{markup.strikeout[1]}"
        if span.bold:
            text = f"{markup.bold[0]}{text}{markup.bold[1]}"
        if span.italic:
            text = f"{markup.italic[0]}{text}{markup.italic[1]}"
        rendered.append(text)
    return "".join(rendered)


def markdown_line(line: TextLine, *, start: int = 0) -> str:
    return render_styled_line(line, MARKDOWN_MARKUP, start=start)


def html_line(line: TextLine, *, start: int = 0) -> str:
    return render_styled_line(line, HTML_MARKUP, start=start)


class ElementFormatter(Protocol):
    def format_block(self, block: Block) -> str: ...

    def format_table(self, table: Table) -> str: ...

    def format_figure(self, figure: Figure) -> str: ...

    def format_page(self, page: Page) -> str: ...

    def format_document(self, document: Document) -> str: ...


def format_elements(formatter: ElementFormatter, page: Page) -> list[str]:
    return [
        map_page_element(
            element,
            block=formatter.format_block,
            table=formatter.format_table,
            figure=formatter.format_figure,
        )
        for element in page.elements
    ]


class MarkdownFormatter:
    __slots__ = ()

    def format_block(self, block: Block) -> str:
        return block_to_markdown(block)

    def format_table(self, table: Table) -> str:
        return table_to_html(table)

    def format_figure(self, figure: Figure) -> str:
        return f"> [Figure: {figure.kind}]"

    def format_page(self, page: Page) -> str:
        return "\n\n".join(format_elements(self, page))

    def format_document(self, document: Document) -> str:
        return "\f".join(self.format_page(page) for page in document.pages) + "\f"


class HtmlFormatter:
    __slots__ = ()

    def format_block(self, block: Block) -> str:
        return block_to_html(block)

    def format_table(self, table: Table) -> str:
        return table_to_html(table)

    def format_figure(self, figure: Figure) -> str:
        return f'<figure data-kind="{escape(figure.kind)}"></figure>'

    def format_page(self, page: Page) -> str:
        rendered = "\n".join(format_elements(self, page))
        return f'<section data-page-number="{page.page_number}">{rendered}</section>'

    def format_document(self, document: Document) -> str:
        pages = "\n".join(self.format_page(page) for page in document.pages)
        return f'<article data-schema-version="{escape(document.schema_version)}">{pages}</article>'


MARKDOWN_FORMATTER = MarkdownFormatter()
HTML_FORMATTER = HtmlFormatter()


def document_to_markdown(document: Document) -> str:
    return MARKDOWN_FORMATTER.format_document(document)


def page_to_markdown(page: Page) -> str:
    return MARKDOWN_FORMATTER.format_page(page)


def block_to_markdown(block: Block) -> str:
    if block.kind is BlockKind.LIST:
        return "\n".join(
            f"{prefix or '- '}{markdown_line(line, start=len(prefix))}"
            for line in block.lines
            for prefix in (list_prefix(line.text),)
        )
    text = "\n".join(markdown_line(line) for line in block.lines)
    if block.kind is BlockKind.HEADING:
        return f"{'#' * (block.level or 2)} {text}"
    return text


def document_to_html(document: Document) -> str:
    return HTML_FORMATTER.format_document(document)


def page_to_html(page: Page) -> str:
    return HTML_FORMATTER.format_page(page)


def block_to_html(block: Block) -> str:
    attributes = f' data-block-kind="{escape(block.kind.value)}"'

    if block.kind is BlockKind.HEADING:
        tag = f"h{block.level or 2}"
        heading = "<br />".join(html_line(line) for line in block.lines)
        return f"<{tag}{attributes}>{heading}</{tag}>"
    if block.kind is BlockKind.LIST:
        items = "".join(
            f"<li>{html_line(line, start=len(list_prefix(line.text)))}</li>" for line in block.lines
        )
        return f"<ul{attributes}>{items}</ul>"
    text = "<br />".join(html_line(line) for line in block.lines)
    return f"<p{attributes}>{text}</p>"


def table_to_html(table: Table) -> str:
    if not table.rows:
        return "<table></table>"
    associated_text = tuple(value for value in (table.title, table.caption) if value is not None)
    prefix = "".join(
        f'<div data-table-associated="{escape(value.kind)}">{escape(value.text)}</div>'
        for value in associated_text
    )
    bands = table.row_bands
    if not bands or (
        not associated_text
        and bands[0].kind == "header"
        and all(band.kind == "body" for band in bands[1:])
    ):
        header, *body = table.rows
        head = "".join(table_cell_to_html(cell, header=True) for cell in header)
        body_html = "".join(
            f"<tr>{''.join(table_cell_to_html(cell) for cell in row)}</tr>" for row in body
        )
        return f"{prefix}<table><thead><tr>{head}</tr></thead><tbody>{body_html}</tbody></table>"

    header_rows: list[str] = []
    body_rows: list[str] = []
    for row, band in zip(table.rows, bands, strict=True):
        if band.kind in {"title", "caption"}:
            continue
        cells = "".join(table_cell_to_html(cell, header=band.kind == "header") for cell in row)
        rendered = f'<tr data-row-kind="{escape(band.kind)}">{cells}</tr>'
        (header_rows if band.kind == "header" else body_rows).append(rendered)
    head_html = f"<thead>{''.join(header_rows)}</thead>" if header_rows else ""
    body_html = f"<tbody>{''.join(body_rows)}</tbody>" if body_rows else ""
    return f"{prefix}<table>{head_html}{body_html}</table>"


def table_cell_to_html(cell: TableCell, *, header: bool = False) -> str:
    tag = "th" if header else "td"
    spans = "".join(
        (
            f' rowspan="{cell.row_span}"' if cell.row_span > 1 else "",
            f' colspan="{cell.column_span}"' if cell.column_span > 1 else "",
        )
    )
    return f"<{tag}{spans}>{escape(cell.text)}</{tag}>"


def list_prefix(text: str) -> str:
    match = LIST_PREFIX_RE.match(text)
    return match.group(0) if match is not None else ""


def json_safe(value: object, *, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported JSON metadata key at {path}: {type(key).__name__}")
            result[key] = json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported JSON metadata value at {path}: {type(value).__name__}")


__all__ = (
    "block_to_html",
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
    "json_safe",
    "page_to_html",
    "page_to_markdown",
    "table_cell_to_json_dict",
    "table_to_html",
    "table_to_json_dict",
)
