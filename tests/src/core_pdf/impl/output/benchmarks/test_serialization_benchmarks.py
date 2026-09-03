# SPDX-License-Identifier: AGPL-3.0-only
"""Performance coverage for the normalized structured-document serializers."""

from __future__ import annotations

import pytest

from core_pdf.impl.output import (
    Block,
    BlockKind,
    Document,
    Page,
    Table,
    TableCell,
    TableColumnBand,
    TableRowBand,
    TextLine,
)
from core_pdf.impl.serialize import document_to_json_dict


def internal_document(page_count: int = 24) -> Document:
    pages: list[Page] = []
    for page_number in range(1, page_count + 1):
        lines = tuple(
            TextLine(
                f"Page {page_number} line {index} with representative styled text",
                bbox=(36.0, 700.0 - index * 14.0, 420.0, 712.0 - index * 14.0),
            )
            for index in range(20)
        )
        rows = tuple(
            tuple(TableCell(row, column, f"r{row}c{column}") for column in range(6))
            for row in range(12)
        )
        pages.append(
            Page(
                page_number=page_number,
                width=612.0,
                height=792.0,
                blocks=(Block(0, BlockKind.PARAGRAPH, lines),),
                tables=(
                    Table(
                        order=1,
                        rows=rows,
                        row_bands=tuple(
                            TableRowBand(index, kind="header" if index == 0 else "body")
                            for index in range(len(rows))
                        ),
                        column_bands=tuple(TableColumnBand(index) for index in range(6)),
                    ),
                ),
            )
        )
    return Document(pages=tuple(pages), metadata={"Title": "Benchmark document"})


@pytest.mark.benchmark_high_impact
def test_normalized_json_graph_benchmark(benchmark) -> None:
    document = internal_document()

    record = benchmark(document_to_json_dict, document)

    assert record["schema_version"] == "5.0"
    assert len(record["pages"]) == 24
    assert len(record["lines"]) == 480
    assert len(record["tables"]) == 24
