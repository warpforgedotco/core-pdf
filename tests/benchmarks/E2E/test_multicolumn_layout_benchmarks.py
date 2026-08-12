# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end benchmarks for native multi-column reading-order reconstruction.

Each text line below is written as its own independently positioned ``Tm``/``Tj``
run (via an explicit ``TextLine.bbox``), column by column -- exactly how a real
multi-column PDF's content stream is authored, with no column hints in the file
itself. Reopening it forces the native layout analyzer to reconstruct reading
order (column 1 top-to-bottom, then column 2, ...) from raw glyph geometry
alone, which is one of the most expensive parts of native text extraction.
"""

from __future__ import annotations

from core_pdf import PdfDocument, serialize_document_to_pdf
from core_pdf.impl.engine.structured import Block, BlockKind, Document, Page, TextLine

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
MARGIN = 36.0
COLUMN_GAP = 18.0
LINE_HEIGHT = 12.0


def build_multicolumn_page(page_number: int, *, columns: int, lines_per_column: int) -> Page:
    column_width = (PAGE_WIDTH - 2 * MARGIN - (columns - 1) * COLUMN_GAP) / columns
    blocks = []
    for column in range(columns):
        x0 = MARGIN + column * (column_width + COLUMN_GAP)
        lines = []
        y = PAGE_HEIGHT - MARGIN
        for row in range(lines_per_column):
            y -= LINE_HEIGHT
            lines.append(
                TextLine(
                    f"Column {column} row {row} synthetic body text for layout "
                    f"stress on page {page_number}.",
                    bbox=(x0, y, x0 + column_width, y + LINE_HEIGHT - 1),
                )
            )
        blocks.append(Block(column, BlockKind.PARAGRAPH, tuple(lines)))
    return Page(page_number=page_number, width=PAGE_WIDTH, height=PAGE_HEIGHT, blocks=tuple(blocks))


def build_multicolumn_document(*, page_count: int, columns: int, lines_per_column: int) -> Document:
    return Document(
        pages=tuple(
            build_multicolumn_page(page + 1, columns=columns, lines_per_column=lines_per_column)
            for page in range(page_count)
        )
    )


MULTICOLUMN_DOCUMENT = build_multicolumn_document(page_count=5, columns=3, lines_per_column=50)
MULTICOLUMN_PDF_BYTES = serialize_document_to_pdf(MULTICOLUMN_DOCUMENT)


def internal_open_and_extract(pdf_bytes: bytes) -> int:
    with PdfDocument.open(pdf_bytes) as document:
        extracted = document.extract()
        return len(extracted.pages)


def test_multicolumn_layout_reconstruction_benchmark(benchmark) -> None:
    result = benchmark.pedantic(
        internal_open_and_extract,
        args=(MULTICOLUMN_PDF_BYTES,),
        iterations=1,
        rounds=3,
    )
    assert result == 5
