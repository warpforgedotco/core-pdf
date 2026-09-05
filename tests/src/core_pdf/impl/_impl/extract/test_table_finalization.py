"""Final table order and metadata do not rewrite captured cell text."""

from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.extract.table_detection import (
    internal_finalize_tables,
    internal_TableAnalysis,
)
from core_pdf.impl._impl.output.model import Table, TableCell


def test_finalization_preserves_cell_text_and_source_records() -> None:
    table = Table(
        order=9,
        rows=(
            (TableCell(0, 0, "Name"), TableCell(0, 1, "Date")),
            (TableCell(1, 0, "Ada"), TableCell(1, 1, "10 /1 9/21 ...", column_span=2)),
        ),
        bbox=(0, 0, 100, 100),
        metadata={"source": "stream"},
    )

    analysis = internal_TableAnalysis.build(ObservationBatch.empty(), 100)
    finalized = internal_finalize_tables((table,), analysis)[0]

    # Poppler 26.07.0 verified this literal text in the ruled PDF regression.
    # A numeric/date interpretation is not evidence for deleting spaces or dots.
    assert finalized.rows[1][1].text == "10 /1 9/21 ..."
    assert finalized.rows[1][1].column_span == 2
    assert finalized.order == 0
    assert finalized.bbox == table.bbox
    assert finalized.metadata == table.metadata
    assert finalized.rows is table.rows
    assert table.rows[1][1].text == "10 /1 9/21 ..."
    assert table.order == 9


def test_finalization_sorts_and_numbers_unsorted_candidates_without_mutating_them() -> None:
    lower = Table(order=9, rows=((TableCell(0, 0, "Lower"),),), bbox=(0, 10, 100, 30))
    upper = Table(order=4, rows=((TableCell(0, 0, "Upper"),),), bbox=(0, 70, 100, 90))
    analysis = internal_TableAnalysis.build(ObservationBatch.empty(), 100)

    finalized = internal_finalize_tables((lower, upper), analysis)

    assert [table.rows[0][0].text for table in finalized] == ["Upper", "Lower"]
    assert [table.order for table in finalized] == [0, 1]
    assert all(len(table.row_bands) == len(table.rows) for table in finalized)
    assert (lower.order, upper.order) == (9, 4)
