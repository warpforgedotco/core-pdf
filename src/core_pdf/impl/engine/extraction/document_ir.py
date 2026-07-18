# SPDX-License-Identifier: AGPL-3.0-only
"""Adapt PDF-specific extraction records to the core-document IR."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from core_document import (
    Annotation,
    Block,
    BlockKind,
    Diagnostic,
    Document,
    Figure,
    FormField,
    Link,
    Page,
    Table,
    TableCell,
    TextLine,
)

if TYPE_CHECKING:
    from core_pdf.impl.engine.extraction.page_text.engine import (
        DocumentExtractionResult,
        PageExtractionResult,
        ResolvedLineRecord,
    )


def resolved_line_to_document_line(line: ResolvedLineRecord) -> TextLine:
    return TextLine(
        text=line.text,
        break_before=line.break_before,
        bbox=line.bbox,
        advance_bbox=line.advance_bbox,
        ink_bbox=line.ink_bbox,
        kind=line.kind,
        source=line.source,
        confidence=line.confidence,
        baseline=line.baseline,
        contributing_sources=line.contributing_sources,
    )


def block_kind(value: str) -> BlockKind:
    try:
        return BlockKind(value)
    except ValueError:
        return BlockKind.UNKNOWN


def page_result_to_document_page(
    result: PageExtractionResult,
    *,
    width: float | None = None,
    height: float | None = None,
) -> Page:
    return Page(
        page_number=result.page_number,
        page_label=result.page_label,
        width=result.width if width is None else width,
        height=result.height if height is None else height,
        rotation=result.rotation,
        page_class=result.page_class,
        base_route=result.base_route,
        confidence=result.confidence,
        blocks=tuple(
            Block(
                order=block.order,
                kind=block_kind(block.kind),
                lines=tuple(resolved_line_to_document_line(line) for line in block.lines),
                bbox=block.bbox,
                column_index=block.column_index,
                rotation=block.rotation,
                confidence=result.confidence,
            )
            for block in result.blocks
        ),
        tables=tuple(
            table_record_to_document_table(index, record)
            for index, record in enumerate(result.tables)
        ),
        figures=tuple(
            figure_record_to_document_figure(index, record)
            for index, record in enumerate(result.figures)
        ),
        links=tuple(link_record_to_document_link(record) for record in result.links),
        annotations=tuple(
            annotation_record_to_document_annotation(record) for record in result.annotations
        ),
        form_fields=tuple(
            field_record_to_document_field(index, record)
            for index, record in enumerate(result.form_fields)
        ),
    )


def extraction_result_to_document(result: DocumentExtractionResult) -> Document:
    return Document(
        metadata=result.metadata,
        pages=tuple(page_result_to_document_page(page) for page in result.pages),
        diagnostics=tuple(diagnostic_from_record(record) for record in result.diagnostics)
        + tuple(
            diagnostic_from_record(record) for page in result.pages for record in page.diagnostics
        ),
    )


def diagnostic_from_record(record: Mapping[str, object]) -> Diagnostic:
    page_number = record.get("page_number")
    return Diagnostic(
        code=str(record.get("code") or "unknown"),
        message=str(record.get("message") or ""),
        severity=str(record.get("severity") or "warning"),
        page_number=page_number if isinstance(page_number, int) else None,
    )


def table_record_to_document_table(index: int, record: Mapping[str, object]) -> Table:
    raw_rows = record.get("rows", [])
    rows: list[tuple[TableCell, ...]] = []
    if isinstance(raw_rows, list):
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, list):
                continue
            rows.append(
                tuple(
                    TableCell(row_index, column_index, str(value or ""))
                    for column_index, value in enumerate(raw_row)
                )
            )
    return Table(index, tuple(rows), bbox_from_value(record.get("bbox")))


def figure_record_to_document_figure(index: int, record: Mapping[str, object]) -> Figure:
    metadata = dict(record)
    metadata.pop("bbox", None)
    kind = str(metadata.pop("kind", "figure"))
    return Figure(index, bbox_from_value(record.get("bbox")), kind, metadata)


def link_record_to_document_link(record: Mapping[str, object]) -> Link:
    return Link(
        bbox=bbox_from_value(record.get("bbox")),
        url=string_or_none(record.get("url")),
        link_type=string_or_none(record.get("link_type")),
    )


def annotation_record_to_document_annotation(record: Mapping[str, object]) -> Annotation:
    return Annotation(
        subtype=string_or_none(record.get("subtype")),
        bbox=bbox_from_value(record.get("bbox")),
        contents=str(record.get("contents") or ""),
        destination=record.get("destination"),
    )


def field_record_to_document_field(index: int, record: Mapping[str, object]) -> FormField:
    return FormField(
        name=str(record.get("name") or ""),
        field_type=str(record.get("field_type") or ""),
        value_text=str(record.get("value_text") or ""),
        bbox=bbox_from_value(record.get("bbox")),
        field_index=index,
    )


def bbox_from_value(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            items = cast(tuple[Any, Any, Any, Any], tuple(value))
            return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))
        except (TypeError, ValueError):
            return None
    if all(hasattr(value, name) for name in ("x0", "y0", "x1", "y1")):
        try:
            rect = cast(Any, value)
            return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        except (TypeError, ValueError):
            return None
    return None


def string_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


__all__ = (
    "extraction_result_to_document",
    "field_record_to_document_field",
    "figure_record_to_document_figure",
    "table_record_to_document_table",
    "page_result_to_document_page",
    "resolved_line_to_document_line",
)
