# SPDX-License-Identifier: AGPL-3.0-only
"""A table whose cells are sentences is still a table.

Blocks and a stream table covering the same region hold the same glyphs, so
one of them has to go. The ruled path already answers this the right way round
-- the text survives either way, but only the table carries the rows and
columns -- and keeps anything shaped like a grid. The stream path asked instead
whether the cells were numeric, so comparison tables and schedules written in
prose were discarded whole, and the page kept no table at all.
"""

from __future__ import annotations

import pytest

from core_pdf.impl.extract import table_reconcile
from core_pdf.impl.extract.table_reconcile import (
    internal_stream_table_duplicated_by_blocks,
    internal_stream_table_is_tabular,
)
from core_pdf.impl.output import Block, BlockKind, Table, TableCell, TextLine
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


def test_a_prose_comparison_table_survives_the_blocks_that_repeat_it() -> None:
    table = table_of(PROSE_CELLS)
    # The premise: the blocks do repeat it, so the coverage test alone would drop it.
    assert internal_stream_table_is_tabular(table)
    assert not internal_stream_table_duplicated_by_blocks(table, blocks_covering(table))


def test_reconciliation_builds_each_table_profile_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = table_of(PROSE_CELLS)
    built: list[Table] = []
    original_build = table_reconcile.internal_build_table_profile

    def counting_build(candidate: Table) -> table_reconcile.internal_TableProfile:
        built.append(candidate)
        return original_build(candidate)

    monkeypatch.setattr(table_reconcile, "internal_build_table_profile", counting_build)

    table_reconcile.internal_project_text_and_tables(blocks_covering(table), (table,))

    assert built == [table]


def test_two_columns_of_prose_are_not_mistaken_for_a_table() -> None:
    # A page set in two columns divides its rows exactly as a table does, so
    # shape alone cannot keep it; the third column is what it never produces.
    columns = tuple(
        ("Left column line of running text", "Right column line of running text") for _ in range(6)
    )
    table = table_of(columns)

    assert not internal_stream_table_is_tabular(table)
    assert internal_stream_table_duplicated_by_blocks(table, blocks_covering(table))


def test_a_two_row_fragment_is_not_mistaken_for_a_table() -> None:
    # A wrapped caption can divide into three columns once; a table does it again.
    table = table_of(PROSE_CELLS[:2])

    assert not internal_stream_table_is_tabular(table)
