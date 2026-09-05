"""Table row projection keeps coordinates and semantic bands consistent."""

from core_pdf.impl._impl.extract.table_cleanup import internal_table_with_bands
from core_pdf.impl._impl.extract.table_reconcile import (
    internal_remove_block_duplicate_table_rows,
    internal_remove_block_duplicate_tables,
)
from core_pdf.impl._impl.output.model import (
    Block,
    BlockKind,
    Table,
    TableAssociatedText,
    TableCell,
    TextLine,
)
from core_pdf.impl._impl.output.serialize import table_to_html


def test_removing_a_duplicate_title_row_preserves_table_bands_and_serialization() -> None:
    title = "Repeated report heading"
    table = internal_table_with_bands(
        Table(
            order=0,
            bbox=(0, 0, 100, 100),
            title=TableAssociatedText(title, kind="title"),
            rows=(
                (TableCell(0, 0, title, bbox=(0, 80, 100, 100)),),
                (
                    TableCell(1, 0, "Name", bbox=(0, 50, 50, 70)),
                    TableCell(1, 1, "Value", bbox=(50, 50, 100, 70)),
                ),
                (
                    TableCell(2, 0, "Alpha", bbox=(0, 10, 50, 30)),
                    TableCell(2, 1, "100", bbox=(50, 10, 100, 30)),
                ),
            ),
        )
    )
    block = Block(
        order=0,
        kind=BlockKind.HEADING,
        bbox=(0, 80, 100, 100),
        lines=(TextLine(title),),
    )

    projected = internal_remove_block_duplicate_table_rows([block], (table,))[0]

    assert [[cell.row for cell in row] for row in projected.rows] == [[0, 0], [1, 1]]
    assert [(band.index, band.kind) for band in projected.row_bands] == [
        (0, "header"),
        (1, "body"),
    ]
    assert projected.column_bands[0].bbox == (0, 10, 50, 70)
    html = table_to_html(projected)
    assert '<div data-table-associated="title">Repeated report heading</div>' in html
    assert "<th>Name</th><th>Value</th>" in html
    assert "<td>Alpha</td><td>100</td>" in html
    assert len(table.rows) == len(table.row_bands) == 3
    assert table.rows[1][0].row == 1


def test_removing_a_middle_row_shortens_a_surviving_row_span() -> None:
    table = internal_table_with_bands(
        Table(
            order=0,
            bbox=(0, 0, 100, 100),
            rows=(
                (TableCell(0, 0, "Group", row_span=3), TableCell(0, 1, "First")),
                (TableCell(1, 1, "Repeated row label"),),
                (TableCell(2, 1, "Last"),),
            ),
        )
    )
    block = Block(
        order=0,
        kind=BlockKind.PARAGRAPH,
        bbox=(0, 40, 100, 60),
        lines=(TextLine("Repeated row label"),),
    )

    projected = internal_remove_block_duplicate_table_rows([block], (table,))[0]

    assert len(projected.rows) == len(projected.row_bands) == 2
    assert projected.rows[0][0].row_span == 2
    assert projected.rows[1][0].row == 1
    assert projected.row_bands[1].index == 1
    assert table.rows[0][0].row_span == 3


def test_page_wide_duplicate_coverage_includes_text_without_geometry() -> None:
    table = Table(
        order=0,
        bbox=(0, 0, 100, 20),
        rows=(
            (
                TableCell(0, 0, "Monthly enrollment premiums"),
                TableCell(0, 1, "Insurance coverage amount"),
            ),
        ),
    )
    matching_text = Block(
        order=0,
        kind=BlockKind.PARAGRAPH,
        lines=(TextLine("Monthly enrollment premiums Insurance coverage amount"),),
    )
    unrelated_text = Block(
        order=1,
        kind=BlockKind.PARAGRAPH,
        bbox=(0, 100, 100, 110),
        lines=(TextLine("Unrelated note"),),
    )

    assert internal_remove_block_duplicate_tables([matching_text], (table,)) == ()
    assert internal_remove_block_duplicate_tables([matching_text, unrelated_text], (table,)) == ()
