# SPDX-License-Identifier: AGPL-3.0-only
"""Serialization views for the core-document IR."""

from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape

from core_document.model import (
    Annotation,
    Block,
    BlockKind,
    Document,
    Figure,
    FormField,
    JsonValue,
    Link,
    Page,
    Table,
    TableCell,
    TextLine,
)


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
        "blocks": [block_to_json_dict(block) for block in page.blocks],
        "tables": [table_to_json_dict(table) for table in page.tables],
        "figures": [figure_to_json_dict(figure) for figure in page.figures],
        "links": [link_to_json_dict(link) for link in page.links],
        "annotations": [annotation_to_json_dict(annotation) for annotation in page.annotations],
        "form_fields": [field_to_json_dict(field) for field in page.form_fields],
    }


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
        "baseline": bbox_to_json(line.baseline),
        "contributing_sources": list(line.contributing_sources),
    }


def table_to_json_dict(table: Table) -> dict[str, JsonValue]:
    return {
        "order": table.order,
        "bbox": bbox_to_json(table.bbox),
        "confidence": table.confidence,
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


def bbox_to_json(bbox: tuple[float, float, float, float] | None) -> list[float] | None:
    return list(bbox) if bbox is not None else None


def document_to_markdown(document: Document) -> str:
    return "\f".join(page_to_markdown(page) for page in document.pages) + "\f"


def page_to_markdown(page: Page) -> str:
    parts = [block_to_markdown(block) for block in page.blocks]
    parts.extend(table_to_markdown(table) for table in page.tables)
    parts.extend(f"> [Figure: {figure.kind}]" for figure in page.figures)
    return "\n\n".join(parts)


def block_to_markdown(block: Block) -> str:
    text = block.text
    if block.kind is BlockKind.HEADING:
        return f"{'#' * (block.level or 2)} {text}"
    if block.kind is BlockKind.LIST:
        return "\n".join(
            line.text if _has_list_marker(line.text) else f"- {line.text}" for line in block.lines
        )
    return text


def document_to_html(document: Document) -> str:
    pages = "\n".join(page_to_html(page) for page in document.pages)
    return f'<article data-schema-version="{escape(document.schema_version)}">{pages}</article>'


def page_to_html(page: Page) -> str:
    parts = [block_to_html(block) for block in page.blocks]
    parts.extend(table_to_html(table) for table in page.tables)
    parts.extend(f'<figure data-kind="{escape(figure.kind)}"></figure>' for figure in page.figures)
    rendered = "\n".join(parts)
    return f'<section data-page-number="{page.page_number}">{rendered}</section>'


def block_to_html(block: Block) -> str:
    attributes = f' data-block-kind="{escape(block.kind.value)}"'
    if block.kind is BlockKind.HEADING:
        tag = f"h{block.level or 2}"
        return f"<{tag}{attributes}>{escape(block.text)}</{tag}>"
    if block.kind is BlockKind.LIST:
        items = "".join(f"<li>{escape(line.text)}</li>" for line in block.lines)
        return f"<ul{attributes}>{items}</ul>"
    text = "<br />".join(escape(line.text) for line in block.lines)
    return f"<p{attributes}>{text}</p>"


def table_to_markdown(table: Table) -> str:
    if not table.rows:
        return ""
    rows = [[cell.text.replace("|", "\\|") for cell in row] for row in table.rows]
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header, *body = normalized
    output = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    output.extend(f"| {' | '.join(row)} |" for row in body)
    return "\n".join(output)


def table_to_html(table: Table) -> str:
    if not table.rows:
        return "<table></table>"
    header, *body = table.rows
    head = "".join(f"<th>{escape(cell.text)}</th>" for cell in header)
    body_html = "".join(
        f"<tr>{''.join(f'<td>{escape(cell.text)}</td>' for cell in row)}</tr>" for row in body
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body_html}</tbody></table>"


def _has_list_marker(text: str) -> bool:
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
    "json_safe",
    "page_to_html",
    "page_to_json_dict",
    "page_to_markdown",
    "table_cell_to_json_dict",
    "table_to_html",
    "table_to_markdown",
    "table_to_json_dict",
)
