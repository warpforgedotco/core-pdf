from dataclasses import replace

import pytest

from core_pdf.impl._impl.extract.table_cleanup import internal_merge_adjacent_tables
from core_pdf.impl._impl.output.model import Table, TableAssociatedText, TableCell


def internal_table(
    top: float, *, confidence: float | None = None, x: float = 0, columns: int = 2
) -> Table:
    texts = [("Item", "Value"), ("A", "1")]
    rows = tuple(
        tuple(
            TableCell(
                row,
                column,
                texts[row][column] if column < 2 else "extra",
                bbox=(x + column * 10, top - (row + 1) * 10, x + (column + 1) * 10, top - row * 10),
            )
            for column in range(columns)
        )
        for row in range(2)
    )
    return Table(5, rows, (x, top - 20, x + columns * 10, top), confidence=confidence)


@pytest.mark.parametrize("left", [None, 0, 0.4, 1])
@pytest.mark.parametrize("right", [None, 0, 0.8, 1])
def test_merged_confidence_never_discards_explicit_zero(left, right):
    top = internal_table(60, confidence=left)
    bottom = internal_table(30, confidence=right)
    (merged,) = internal_merge_adjacent_tables([bottom, top])
    assert merged.confidence == min(1 if left is None else left, 1 if right is None else right)
    assert [[c.text for c in row] for row in merged.rows] == [
        ["Item", "Value"],
        ["A", "1"],
        ["A", "1"],
    ]
    assert [[c.row for c in row] for row in merged.rows] == [[0, 0], [1, 1], [2, 2]]
    assert merged.bbox == (0, 10, 20, 60)
    assert len(top.rows) == len(bottom.rows) == 2


def test_merge_preserves_cell_spans_and_selects_title_and_caption():
    top = internal_table(60)
    top = replace(
        top,
        rows=(top.rows[0], (replace(top.rows[1][0], row_span=2), top.rows[1][1])),
        title=TableAssociatedText("Title"),
        caption=TableAssociatedText("old"),
        metadata={"source": "grid"},
    )
    bottom = replace(internal_table(30), caption=TableAssociatedText("Caption"))
    (merged,) = internal_merge_adjacent_tables([top, bottom])
    assert merged.rows[1][0].row_span == 2
    assert merged.rows[1][0].bbox == top.rows[1][0].bbox
    assert merged.title is top.title
    assert merged.caption is bottom.caption
    assert merged.metadata == top.metadata
    assert merged.order == top.order


@pytest.mark.parametrize(
    ("top", "x", "columns"),
    [(3, 0, 2), (46, 0, 2), (30, 20, 2), (30, 0, 1), (30, 0, 3), (30, 0, 17)],
)
def test_incompatible_table_regions_remain_separate(top, x, columns):
    first = internal_table(60)
    second = internal_table(top, x=x, columns=columns)
    result = internal_merge_adjacent_tables([first, second])
    assert result == [first, second]
    assert result[0] is first
    assert result[1] is second


@pytest.mark.parametrize("missing", [0, 1])
def test_tables_without_geometry_remain_separate(missing):
    tables = [internal_table(60), internal_table(30)]
    tables[missing] = replace(tables[missing], bbox=None)
    result = internal_merge_adjacent_tables(tables)
    assert len(result) == 2
    assert all(any(item is original for original in tables) for item in result)


@pytest.mark.parametrize("geometry", [False, True])
def test_wrapped_stream_rows_join_text_and_union_cell_geometry(geometry):
    from core_pdf.impl._impl.extract.table_cleanup import internal_merge_wrapped_stream_rows

    rows = tuple(
        tuple(
            TableCell(
                i,
                j,
                (f"id{i // 2}" if i % 2 == 0 else "")
                if j == 0
                else ("first" if i % 2 == 0 else "second")
                if j == 1
                else "",
                bbox=(j * 10, 80 - i * 10 - 10, j * 10 + 10, 80 - i * 10) if geometry else None,
            )
            for j in range(5)
        )
        for i in range(8)
    )
    original = Table(2, rows, metadata={"source": "stream", "numeric_cells": 0})
    result = internal_merge_wrapped_stream_rows(original)
    assert len(result.rows) == 4
    for i, row in enumerate(result.rows):
        assert [c.text for c in row] == [f"id{i}", "first second", "", "", ""]
        assert all(c.row == i for c in row)
        assert row[1].bbox == ((10, 60 - 20 * i, 20, 80 - 20 * i) if geometry else None)
    assert result.metadata == original.metadata
    assert result.order == original.order
    assert len(original.rows) == 8


@pytest.mark.parametrize(
    ("source", "row_count", "columns", "numeric"),
    [
        ("grid", 8, 5, 0),
        ("stream", 7, 5, 0),
        ("stream", 8, 4, 0),
        ("stream", 8, 5, 3),
        ("stream", 8, 5, 0),
    ],
)
def test_stream_row_merge_preserves_ineligible_or_unwrapped_tables(
    source, row_count, columns, numeric
):
    from core_pdf.impl._impl.extract.table_cleanup import internal_merge_wrapped_stream_rows

    original = Table(
        0,
        tuple(tuple(TableCell(i, j, "value") for j in range(columns)) for i in range(row_count)),
        metadata={"source": source, "numeric_cells": numeric},
    )
    assert internal_merge_wrapped_stream_rows(original) is original
