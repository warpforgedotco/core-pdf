# SPDX-License-Identifier: AGPL-3.0-only
"""Adapt PDF-specific extraction records to the core-document IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_document import Block, BlockKind, Document, Page, TextLine

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
    )


def extraction_result_to_document(result: DocumentExtractionResult) -> Document:
    return Document(
        metadata=result.metadata,
        pages=tuple(page_result_to_document_page(page) for page in result.pages),
    )


__all__ = (
    "extraction_result_to_document",
    "page_result_to_document_page",
    "resolved_line_to_document_line",
)
