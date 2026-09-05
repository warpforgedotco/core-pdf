"""Candidate cleanup preserves records until final table order is known."""

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.extract.table_detection import (
    internal_clean_table_cells,
    internal_finalize_tables,
    internal_TableAnalysis,
)
from core_pdf.impl._impl.output.model import Table, TableCell


def test_cell_cleanup_changes_only_repaired_text_and_preserves_the_input() -> None:
    table = Table(
        order=9,
        rows=(
            (TableCell(0, 0, "Name"), TableCell(0, 1, "Date")),
            (TableCell(1, 0, "Ada"), TableCell(1, 1, "10 /1 9/21 ...", column_span=2)),
        ),
        bbox=(0, 0, 100, 100),
        metadata={"source": "stream"},
    )

    cleaned = internal_clean_table_cells(table)

    assert cleaned.rows[1][1].text == "10/19/21"
    assert cleaned.rows[1][1].column_span == 2
    assert cleaned.order == 9
    assert cleaned.bbox == table.bbox
    assert cleaned.metadata == table.metadata
    assert cleaned.rows[0] is table.rows[0]
    assert cleaned.rows[1][0] is table.rows[1][0]
    assert table.rows[1][1].text == "10 /1 9/21 ..."
    assert internal_clean_table_cells(cleaned) is cleaned


def test_finalization_sorts_and_numbers_unsorted_candidates_without_mutating_them() -> None:
    lower = Table(order=9, rows=((TableCell(0, 0, "Lower"),),), bbox=(0, 10, 100, 30))
    upper = Table(order=4, rows=((TableCell(0, 0, "Upper"),),), bbox=(0, 70, 100, 90))
    analysis = internal_TableAnalysis.build(ObservationBatch.empty(), 100)

    finalized = internal_finalize_tables((lower, upper), analysis)

    assert [table.rows[0][0].text for table in finalized] == ["Upper", "Lower"]
    assert [table.order for table in finalized] == [0, 1]
    assert all(len(table.row_bands) == len(table.rows) for table in finalized)
    assert (lower.order, upper.order) == (9, 4)
