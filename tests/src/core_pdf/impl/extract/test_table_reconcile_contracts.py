from core_pdf.impl.extract.emit import (
    TableIndex,
    line_duplicates_table,
    remove_table_duplicate_blocks,
)
from core_pdf.impl.output.model import Block, BlockKind, Table, TableCell, TextLine


def make_table(
    cells: list[list[tuple[str, tuple[float, float, float, float] | None]]],
) -> Table:
    rows = tuple(
        tuple(
            TableCell(row=row_index, column=column_index, text=text, bbox=bbox)
            for column_index, (text, bbox) in enumerate(row)
        )
        for row_index, row in enumerate(cells)
    )
    boxes = [cell.bbox for row in rows for cell in row if cell.bbox is not None]
    bbox = (
        (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        if boxes
        else (0.0, 0.0, 100.0, 40.0)
    )
    return Table(order=0, rows=rows, bbox=bbox)


def make_block(lines: list[tuple[str, tuple[float, float, float, float]]]) -> Block:
    return Block(
        order=0,
        kind=BlockKind.PARAGRAPH,
        lines=tuple(TextLine(text, bbox=bbox) for text, bbox in lines),
        bbox=lines[0][1],
    )


def test_line_matching_one_cell_is_a_duplicate() -> None:
    table = make_table([[("alpha", (0.0, 20.0, 50.0, 40.0)), ("beta", (50.0, 20.0, 100.0, 40.0))]])
    index = TableIndex.build(table)
    assert line_duplicates_table("alpha", (0.0, 20.0, 50.0, 40.0), index)
    assert line_duplicates_table("alpha beta", (0.0, 20.0, 100.0, 40.0), index)
    assert not line_duplicates_table("gamma", (0.0, 20.0, 50.0, 40.0), index)
    assert not line_duplicates_table("alpha", (0.0, 0.0, 50.0, 10.0), index)


def test_cells_without_geometry_fall_back_to_text_coverage() -> None:
    table = make_table([[("alpha", None), ("beta", None)]])
    index = TableIndex.build(table)
    assert index.frame is None
    assert line_duplicates_table("alpha beta", (0.0, 0.0, 100.0, 40.0), index)
    assert not line_duplicates_table("alpha", (0.0, 0.0, 100.0, 40.0), index)


def test_zero_area_line_box_never_covers_a_cell() -> None:
    table = make_table([[("alpha", (0.0, 20.0, 50.0, 40.0))]])
    index = TableIndex.build(table)
    assert not line_duplicates_table("alpha", (10.0, 30.0, 10.0, 30.0), index)


def test_duplicate_lines_are_removed_from_blocks_and_others_kept() -> None:
    table = make_table([[("alpha", (0.0, 20.0, 50.0, 40.0)), ("beta", (50.0, 20.0, 100.0, 40.0))]])
    block = make_block([("alpha", (0.0, 20.0, 50.0, 40.0)), ("prose", (0.0, 60.0, 50.0, 80.0))])
    deduplicated = remove_table_duplicate_blocks([block], (table,))
    assert [line.text for line in deduplicated[0].lines] == ["prose"]
    assert deduplicated[0].bbox == (0.0, 60.0, 50.0, 80.0)
    untouched = make_block([("prose", (0.0, 60.0, 50.0, 80.0))])
    assert remove_table_duplicate_blocks([untouched], (table,)) == [untouched]
