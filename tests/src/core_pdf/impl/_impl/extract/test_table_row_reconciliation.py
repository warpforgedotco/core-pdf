# SPDX-License-Identifier: AGPL-3.0-only
"""Table rows require complete local text evidence before removal."""

from dataclasses import replace

import pytest

from core_pdf.impl._impl.extract.table_reconcile import (
    internal_project_text_and_tables,
    internal_remove_block_duplicate_table_rows,
)
from core_pdf.impl._impl.output.model import (
    Block,
    BlockKind,
    Table,
    TableAssociatedText,
    TableCell,
    TextLine,
)
from core_pdf.impl.types import Rectangle

internal_ROW_BOX: Rectangle = (20, 120, 260, 140)


def internal_table(text: str, *, box: Rectangle | None = internal_ROW_BOX) -> Table:
    return Table(0, rows=((TableCell(0, 0, text, bbox=box),),), bbox=(20, 120, 260, 260))


def internal_block(text: str, *, box: Rectangle | None = internal_ROW_BOX) -> Block:
    return Block(0, BlockKind.PARAGRAPH, (TextLine(text, bbox=box),), bbox=box)


@pytest.mark.parametrize(
    ("line_text", "row_text"),
    [
        ("> 5", "5"),
        ("Alpha beta", "beta Alpha"),
        ("Total 10", "total 10"),
        (
            "red red blue blue blue blue blue blue blue blue",
            "red blue blue blue blue blue blue blue blue blue",
        ),
        ("Alpha beta", "Alpha beta!"),
        ("!", "~"),
    ],
)
def test_row_projection_preserves_distinct_complete_text(line_text: str, row_text: str) -> None:
    # Poppler 26.07.0 pdftotext -layout retains the punctuation, case, order,
    # and repeated words in distinctions.pdf, repetitions.pdf, and symbols.pdf.
    # A competing layout's nearly matching line cannot justify deleting them.
    block = internal_block(line_text)
    table = internal_table(row_text)

    blocks, tables = internal_project_text_and_tables([block], (table,))

    assert blocks == [block]
    assert tables == (table,)


def test_identical_text_in_another_row_region_is_preserved() -> None:
    # Poppler 26.07.0 -bbox confirms disjoint.pdf has Total 10 at both y=130
    # and y=200. Both lie inside the table's broad bounds, but not the same row.
    block = internal_block("Total 10", box=(20, 190, 260, 220))
    table = internal_table("Total 10")

    assert internal_project_text_and_tables([block], (table,)) == ([block], (table,))


def test_matching_words_scattered_between_lines_do_not_delete_a_row() -> None:
    # repetitions.pdf, checked with Poppler 26.07.0 -layout, contains the two
    # separate lines and the complete four-word line as distinct occurrences.
    block = Block(
        0,
        BlockKind.PARAGRAPH,
        (
            TextLine("Alpha beta", bbox=(20, 120, 120, 140)),
            TextLine("Gamma delta", bbox=(140, 120, 260, 140)),
        ),
        bbox=internal_ROW_BOX,
    )
    table = internal_table("Alpha beta Gamma delta")

    assert internal_remove_block_duplicate_table_rows([block], (table,)) == (table,)


@pytest.mark.parametrize("missing", ["row", "line", "multiline"])
def test_missing_local_geometry_keeps_the_row(missing: str) -> None:
    table = internal_table("Total 10", box=None if missing == "row" else internal_ROW_BOX)
    block = internal_block("Total 10", box=None if missing == "line" else internal_ROW_BOX)
    if missing == "multiline":
        block = replace(block, lines=(TextLine("Total 10"), TextLine("Unrelated text")))

    assert internal_remove_block_duplicate_table_rows([block], (table,)) == (table,)


def test_an_oversized_line_box_does_not_establish_row_duplication() -> None:
    table = internal_table("Total 10")
    block = internal_block("Total 10", box=(20, 120, 260, 260))

    assert internal_remove_block_duplicate_table_rows([block], (table,)) == (table,)


def test_known_line_and_cell_positions_suffice_without_container_boxes() -> None:
    table = replace(internal_table("Total 10"), bbox=None)
    block = replace(internal_block("Total 10"), bbox=None)

    assert internal_remove_block_duplicate_table_rows([block], (table,)) == ()


def test_a_fully_duplicated_table_does_not_leave_an_empty_table() -> None:
    table = internal_table("Total 10")

    assert internal_remove_block_duplicate_table_rows([internal_block("Total 10")], (table,)) == ()
    assert table.rows[0][0].text == "Total 10"


@pytest.mark.parametrize("associated", ["title", "caption"])
def test_removing_all_rows_must_not_discard_associated_text(associated: str) -> None:
    # Poppler 26.07.0 -layout retains both the Report heading and Total 10 in
    # associated.pdf. The surviving block covers only the row's text.
    text = TableAssociatedText("Report heading", kind=associated)
    table = replace(
        internal_table("Total 10"),
        title=text if associated == "title" else None,
        caption=text if associated == "caption" else None,
    )

    assert internal_remove_block_duplicate_table_rows([internal_block("Total 10")], (table,)) == (
        table,
    )
