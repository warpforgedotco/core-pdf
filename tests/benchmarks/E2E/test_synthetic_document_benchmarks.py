# SPDX-License-Identifier: AGPL-3.0-only
"""Writing benchmark over a large synthetic, fixture-free document."""

from __future__ import annotations

from core_pdf import serialize_document_to_pdf
from core_pdf.impl.engine.structured import (
    Block,
    BlockKind,
    Document,
    Page,
    Table,
    TableCell,
    TextLine,
)

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0


def build_synthetic_document(
    *, page_count: int, paragraphs_per_page: int = 4, lines_per_paragraph: int = 6
) -> Document:
    pages = []
    for page_index in range(page_count):
        blocks = [
            Block(
                0,
                BlockKind.HEADING,
                lines=(TextLine(f"Section {page_index + 1}"),),
                level=1,
            ),
        ]
        for paragraph_index in range(paragraphs_per_page):
            lines = tuple(
                TextLine(
                    f"Paragraph {paragraph_index} line {line_index} of synthetic "
                    f"page {page_index}: the quick brown fox jumps over the lazy dog."
                )
                for line_index in range(lines_per_paragraph)
            )
            blocks.append(Block(paragraph_index + 1, BlockKind.PARAGRAPH, lines=lines))
        table = Table(
            len(blocks),
            rows=(
                (TableCell(0, 0, "Name"), TableCell(0, 1, "Value")),
                (TableCell(1, 0, "Alpha"), TableCell(1, 1, "1")),
                (TableCell(2, 0, "Beta"), TableCell(2, 1, "2")),
            ),
        )
        pages.append(
            Page(
                page_number=page_index + 1,
                width=PAGE_WIDTH,
                height=PAGE_HEIGHT,
                blocks=tuple(blocks),
                tables=(table,),
            )
        )
    return Document(pages=tuple(pages))


LARGE_DOCUMENT = build_synthetic_document(page_count=50)


def test_synthetic_pdf_serialize_large_benchmark(benchmark) -> None:
    result = benchmark(serialize_document_to_pdf, LARGE_DOCUMENT)
    assert result
