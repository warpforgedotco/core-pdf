import pytest

from core_pdf.impl.extract.block_layout import (
    column_major_prose,
    interleave_columnar_blocks,
    transpose_numeric_table_blocks,
)
from core_pdf.impl.extract.contracts import ParsedBlock, ParsedLine
from core_pdf.impl.output.model import TextLine


def internal_grid(columns, rows, *, numeric=False, column_major=False, jitter=0):
    grid = [
        [
            ParsedLine(
                TextLine(
                    str(row * columns + column) if numeric else "word",
                    bbox=(
                        column * 60,
                        1000 - row * 12 + (column % 2) * jitter,
                        column * 60 + 40,
                        1010 - row * 12,
                    ),
                ),
                sequence=row * columns + column,
            )
            for column in range(columns)
        ]
        for row in range(rows)
    ]
    lines = (
        tuple(line for column in zip(*grid, strict=True) for line in column)
        if column_major
        else tuple(line for row in grid for line in row)
    )
    return ParsedBlock(
        lines,
        (0, 1000 - (rows - 1) * 12, columns * 60 - 20, 1010),
        column_index=2,
        kind="table" if numeric else "paragraph",
        level=3,
    ), grid


@pytest.mark.parametrize("columns", [20, 25])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("jitter", [0, 1])
def test_numeric_tables_are_read_top_to_bottom_then_left_to_right(columns, column_major, jitter):
    block, grid = internal_grid(columns, 15, numeric=True, column_major=column_major, jitter=jitter)
    result = transpose_numeric_table_blocks([block])[0]
    assert result.lines == tuple(line for row in grid for line in row)
    assert sorted(map(id, result.lines)) == sorted(map(id, block.lines))
    assert (result.bbox, result.column_index, result.kind, result.level) == (
        block.bbox,
        2,
        "table",
        3,
    )


@pytest.mark.parametrize(
    ("columns", "rows", "numeric"), [(20, 14, True), (19, 16, True), (20, 15, False)]
)
def test_numeric_table_repair_requires_enough_lines_columns_and_digits(columns, rows, numeric):
    block, _ = internal_grid(columns, rows, numeric=numeric, column_major=True)
    assert transpose_numeric_table_blocks([block])[0] is block


@pytest.mark.parametrize("columns", [3, 4])
@pytest.mark.parametrize("rows", [27, 40])
def test_interleaved_prose_becomes_column_major_without_losing_line_identity(columns, rows):
    block, grid = internal_grid(columns, rows)
    result = column_major_prose([block])[0]
    assert result.lines == tuple(line for column in zip(*grid, strict=True) for line in column)
    assert sorted(map(id, result.lines)) == sorted(map(id, block.lines))
    assert (result.bbox, result.column_index, result.kind, result.level) == (
        block.bbox,
        2,
        "paragraph",
        3,
    )


@pytest.mark.parametrize(
    ("columns", "rows", "numeric", "column_major"),
    [(3, 26, False, False), (2, 40, False, False), (3, 30, True, False), (3, 30, False, True)],
)
def test_prose_repair_preserves_short_single_column_numeric_and_already_grouped_blocks(
    columns, rows, numeric, column_major
):
    block, _ = internal_grid(columns, rows, numeric=numeric, column_major=column_major)
    assert column_major_prose([block])[0] is block


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_scanned_columns_interleave_by_row_and_retain_other_blocks(reverse):
    _, grid = internal_grid(3, 20)
    columns = [
        ParsedBlock(tuple(row[column] for row in grid), (0, 772, 160, 1010)) for column in range(3)
    ]
    if reverse:
        columns.reverse()
    note = ParsedBlock((ParsedLine(TextLine("note", bbox=(0, 0, 10, 10))),), (0, 0, 10, 10))
    result = interleave_columnar_blocks([columns[0], note, *columns[1:]])
    assert len(result) == 2
    assert result[0].lines == tuple(line for row in grid for line in row)
    assert result[0].bbox == (0, 772, 160, 1010)
    assert result[1] is note


@pytest.mark.parametrize("reason", ["few", "left", "right"])
def test_scanned_column_repair_preserves_unrelated_regions(reason):
    block, _ = internal_grid(1, 20)
    blocks = [block] * (2 if reason == "few" else 3)
    if reason != "few":
        blocks[-1] = ParsedBlock(
            block.lines, (21 if reason == "left" else 0, 772, 80 if reason == "right" else 40, 1010)
        )
    assert interleave_columnar_blocks(blocks) is blocks
