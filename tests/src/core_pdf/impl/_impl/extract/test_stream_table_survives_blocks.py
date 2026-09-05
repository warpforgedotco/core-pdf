# SPDX-License-Identifier: AGPL-3.0-only
"""Emission retains accepted tables regardless of cell vocabulary or column count."""

from __future__ import annotations

import pytest

from core_pdf.impl._impl.extract.table_reconcile import internal_project_text_and_tables
from core_pdf.impl._impl.output.model import Block, BlockKind, Table, TableCell, TextLine
from tests.helpers.structured import cell, stream_table


def grid_cell(row: int, column: int, text: str) -> TableCell:
    width = 120.0
    left = 60.0 + column * width
    top = 700.0 - row * 20.0
    return cell(row, column, text, (left, top - 18.0, left + width, top))


def table_of(rows: tuple[tuple[str, ...], ...], *, source: str = "stream") -> Table:
    built = tuple(
        tuple(grid_cell(row_index, column_index, text) for column_index, text in enumerate(row))
        for row_index, row in enumerate(rows)
    )
    return stream_table(built, source=source, confidence=1.0)


def blocks_covering(table: Table) -> list[Block]:
    """One block repeating the table's text over the same region."""
    text = " ".join(item.text for row in table.rows for item in row if item.text)
    assert table.bbox is not None
    return [
        Block(
            order=0,
            kind=BlockKind.PARAGRAPH,
            lines=(TextLine(text=text, bbox=table.bbox),),
            bbox=table.bbox,
        )
    ]


PROSE_CELLS = (
    ("Context", "Neuroimaging in medical research", "Neuroimaging in clinical treatment"),
    ("Scanned", "Research participants enrolled", "Patients under active care"),
    ("Purpose", "Improve knowledge and health", "Improve patient health outcomes"),
    ("Oversight", "Institutional review board", "Treating clinician and board"),
)


@pytest.mark.parametrize(
    "rows",
    [
        PROSE_CELLS,
        PROSE_CELLS[:2],
        tuple(("Alpha beta gamma delta", "Epsilon zeta eta theta") for _ in range(9)),
        tuple(("I 0 ll 13 8 7 o o", "531o6 10llfo2 relDc c t2 l") for _ in range(9)),
    ],
)
def test_accepted_tables_survive_blocks_repeating_their_complete_text(
    rows: tuple[tuple[str, ...], ...],
) -> None:
    table = table_of(rows)
    blocks, tables = internal_project_text_and_tables(blocks_covering(table), (table,))
    assert tables == (table,)
    assert blocks == []
