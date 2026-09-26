from copy import replace

import pytest

from core_pdf.impl.extract_table_cleanup import merge_wrapped_cell_rows
from core_pdf.impl.output_model import Table, TableCell


def make_table(tall=30, *, ragged=False, missing_box=False, blank=False):
    rows = []
    for row, top in enumerate((120, 110, 70, 60)):
        cells = []
        for column in range(4 if ragged and row % 2 else 5):
            height = tall if row % 2 == 0 and column == 0 else 10
            bbox = (column * 20, top - height, column * 20 + 15, top)
            cells.append(
                TableCell(
                    row,
                    column,
                    "" if blank and column == 2 else ("first" if row % 2 == 0 else "second"),
                    column_span=2 if column == 1 and row % 2 else 1,
                    bbox=None if missing_box and column == 3 else bbox,
                )
            )
        rows.append(tuple(cells))
    return Table(
        7,
        tuple(rows),
        (0, 40, 95, 120),
        confidence=0.8,
        metadata={"source": "stream", "custom": "keep"},
    )


@pytest.mark.parametrize("tall", [30, 40])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("missing_box", [False, True])
@pytest.mark.parametrize("blank", [False, True])
def test_wrapped_cells_merge_by_column_and_union_geometry(tall, ragged, missing_box, blank):
    original = make_table(tall, ragged=ragged, missing_box=missing_box, blank=blank)
    merged = merge_wrapped_cell_rows(original)
    assert len(merged.rows) == 2
    for row_index, row in enumerate(merged.rows):
        assert len(row) == 5
        top = 120 if row_index == 0 else 70
        for column, cell in enumerate(row):
            assert (cell.row, cell.column) == (row_index, column)
            expected = (
                ""
                if blank and column == 2
                else "first"
                if ragged and column == 4
                else "first second"
            )
            assert cell.text == expected
            assert cell.column_span == (2 if column == 1 else 1)
            bottom = top - tall if column == 0 else top - (10 if ragged and column == 4 else 20)
            assert cell.bbox == (
                None
                if missing_box and column == 3
                else (column * 20, bottom, column * 20 + 15, top)
            )
    assert merged.order == original.order
    assert merged.bbox == original.bbox
    assert merged.confidence == original.confidence
    assert dict(merged.metadata) == {"source": "stream", "custom": "keep", "logical_rows": True}
    assert len(original.rows) == 4
    assert "logical_rows" not in original.metadata


@pytest.mark.parametrize("reason", ["short", "narrow", "numeric", "no_geometry", "not_tall"])
def test_wrapped_cell_repair_requires_positive_geometry_and_text_evidence(reason):
    table = make_table(tall=29 if reason == "not_tall" else 30)
    if reason == "short":
        table = replace(table, rows=table.rows[:3])
    elif reason == "narrow":
        table = replace(table, rows=tuple(row[:4] for row in table.rows))
    elif reason == "numeric":
        table = replace(
            table,
            rows=tuple(
                tuple(replace(cell, text="123") if cell.column < 2 else cell for cell in row)
                for row in table.rows
            ),
        )
    elif reason == "no_geometry":
        table = replace(
            table, rows=(tuple(replace(cell, bbox=None) for cell in table.rows[0]), *table.rows[1:])
        )
    assert merge_wrapped_cell_rows(table) is table


@pytest.mark.parametrize("one_group", [False, True])
def test_a_single_group_or_only_singleton_groups_does_not_rewrite_the_table(one_group):
    table = make_table()
    rows = []
    for row_index, row in enumerate(table.rows):
        top = 120 - row_index * (5 if one_group else 100)
        rows.append(
            tuple(
                replace(
                    cell,
                    bbox=(
                        cell.column * 20,
                        top - (30 if cell.column == 0 else 10),
                        cell.column * 20 + 15,
                        top,
                    ),
                )
                for cell in row
            )
        )
    table = replace(table, rows=tuple(rows))
    assert merge_wrapped_cell_rows(table) is table
