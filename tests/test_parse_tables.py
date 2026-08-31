from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy

from core_pdf.impl.parse import (
    CapturedPage,
    ObservationBatch,
    ObservationSource,
    PageEvidence,
)
from core_pdf.impl.parse.tables import (
    extract_tables,
    internal_compact_stream_table,
    internal_merge_adjacent_tables,
    internal_merge_grid_cells,
    internal_split_grid_component,
    internal_split_semantic_table,
    internal_stream_table,
    internal_stream_tables,
    internal_table_character_spaced_prose,
)
from core_pdf.impl.structured import Table, TableCell


def page_evidence() -> PageEvidence:
    return PageEvidence(
        page_area=10_000.0,
        native_characters=0,
        visible_native_characters=0,
        suspicious_characters=0,
        image_count=0,
        image_area_ratio=0.0,
        vector_complexity=0,
    )


def test_split_grid_component_separates_vertical_table_regions() -> None:
    horizontal = numpy.asarray(
        [
            (0.0, 100.0, 100.0),
            (0.0, 100.0, 90.0),
            (0.0, 100.0, 80.0),
            (0.0, 100.0, 50.0),
            (0.0, 100.0, 40.0),
            (0.0, 100.0, 30.0),
        ],
        dtype=numpy.float32,
    )
    vertical = numpy.asarray(
        [(0.0, 30.0, 100.0), (50.0, 30.0, 100.0), (100.0, 30.0, 100.0)],
        dtype=numpy.float32,
    )

    regions = internal_split_grid_component(horizontal, vertical)

    assert len(regions) == 2
    assert {
        tuple(sorted(float(line[2]) for line in region_horizontal))
        for region_horizontal, ignored_vertical in regions
    } == {(30.0, 40.0, 50.0), (80.0, 90.0, 100.0)}


def test_merge_grid_cells_infers_horizontal_span_from_missing_rule() -> None:
    rows = [
        [
            TableCell(0, 0, "Header", bbox=(0.0, 10.0, 10.0, 20.0)),
            TableCell(0, 1, "", bbox=(10.0, 10.0, 20.0, 20.0)),
        ],
        [
            TableCell(1, 0, "A", bbox=(0.0, 0.0, 10.0, 10.0)),
            TableCell(1, 1, "B", bbox=(10.0, 0.0, 20.0, 10.0)),
        ],
    ]
    horizontal = numpy.asarray([(0.0, 20.0, 20.0), (0.0, 20.0, 10.0), (0.0, 20.0, 0.0)])
    vertical = numpy.asarray([(0.0, 0.0, 20.0), (10.0, 0.0, 10.0), (20.0, 0.0, 20.0)])

    merged = internal_merge_grid_cells(
        rows,
        horizontal,
        vertical,
        numpy.asarray((0.0, 10.0, 20.0)),
        numpy.asarray((20.0, 10.0, 0.0)),
    )

    assert merged[0][0].text == "Header"
    assert merged[0][0].column_span == 2
    assert [cell.text for cell in merged[1]] == ["A", "B"]


def test_stream_table_accepts_compact_two_row_table() -> None:
    observations = ObservationBatch.from_columns(
        ("Year", "Value", "2024", "10"),
        (
            (0.0, 10.0, 30.0, 20.0),
            (50.0, 10.0, 80.0, 20.0),
            (0.0, 0.0, 30.0, 10.0),
            (50.0, 0.0, 80.0, 10.0),
        ),
        source=ObservationSource.NATIVE,
    )

    table = internal_stream_table(
        0,
        observations,
        [[0, 1], [2, 3]],
        [0, 1],
        [[(0, 0), (1, 2)], [(0, 1), (1, 3)]],
        minimum_rows=2,
    )

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [
        ["Year", "Value"],
        ["2024", "10"],
    ]
    assert table.metadata["source"] == "stream"


def test_compact_stream_table_handles_interleaved_prose_and_split_cells() -> None:
    rows_text = (
        ("MODE PIN", "OUTPUT FORMAT", "CYCLE STABILIZER"),
        ("prose between table rows",),
        ("0", "Offset Binary", "Off"),
        ("prose between table rows",),
        ("1/3V", "DD", "Offset Binary", "On"),
        ("2/3V", "DD", "2's Complement", "On"),
        ("V", "DD", "2's Complement", "Off"),
    )
    rows: list[list[int]] = []
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    x_positions = (67.0, 140.0, 222.0, 250.0)
    for row_number, row_text in enumerate(rows_text):
        indexes: list[int] = []
        y = float((len(rows_text) - row_number) * 12)
        for column, value in enumerate(row_text):
            if len(row_text) == 1:
                x = 315.0
            elif len(row_text) == 4:
                x = (73.0, 89.0, 149.0, 250.0)[column]
            else:
                x = x_positions[column]
            texts.append(value)
            boxes.append((x, y, x + max(8.0, len(value) * 4.0), y + 9.0))
            indexes.append(len(texts) - 1)
        rows.append(indexes)
    observations = ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.NATIVE,
    )

    table = internal_compact_stream_table(0, observations, rows, 612.0)

    assert table is not None
    assert [[cell.text for cell in row] for row in table.rows] == [
        ["MODE PIN", "OUTPUT FORMAT", "CYCLE STABILIZER"],
        ["0", "Offset Binary", "Off"],
        ["1/3V DD", "Offset Binary", "On"],
        ["2/3V DD", "2's Complement", "On"],
        ["V DD", "2's Complement", "Off"],
    ]


def test_stream_table_rejects_dense_prose_columns() -> None:
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    rows: list[list[int]] = []
    columns: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    index = 0
    for row in range(6):
        row_indexes = []
        y0 = float((5 - row) * 12)
        for column, x0 in enumerate((0.0, 70.0, 140.0, 210.0)):
            texts.append(f"Sentence fragment {row}, column {column}.")
            boxes.append((x0, y0, x0 + 55.0, y0 + 10.0))
            row_indexes.append(index)
            columns[column].append((row, index))
            index += 1
        rows.append(row_indexes)
    observations = ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.NATIVE,
    )

    table = internal_stream_table(0, observations, rows, list(range(6)), columns)

    assert table is None


def test_stream_table_rejects_character_spaced_prose() -> None:
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    rows: list[list[int]] = []
    columns: list[list[tuple[int, int]]] = [[] for _ in range(8)]
    index = 0
    for row in range(5):
        row_indexes = []
        y0 = float((4 - row) * 12)
        for column, x0 in enumerate((0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)):
            texts.append("s p a c e d t e x t")
            boxes.append((x0, y0, x0 + 40.0, y0 + 10.0))
            row_indexes.append(index)
            columns[column].append((row, index))
            index += 1
        rows.append(row_indexes)
    observations = ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.NATIVE,
    )

    table = internal_stream_table(0, observations, rows, list(range(5)), columns)

    assert table is None


def test_stream_table_rejects_character_spaced_prose_with_numeric_noise() -> None:
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    rows: list[list[int]] = []
    columns: list[list[tuple[int, int]]] = [[] for _ in range(8)]
    index = 0
    for row in range(5):
        row_indexes = []
        y0 = float((4 - row) * 12)
        for column, x0 in enumerate((0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)):
            texts.append("2024" if row == 0 and column == 0 else "s p a c e d t e x t")
            boxes.append((x0, y0, x0 + 40.0, y0 + 10.0))
            row_indexes.append(index)
            columns[column].append((row, index))
            index += 1
        rows.append(row_indexes)
    observations = ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.NATIVE,
    )

    table = internal_stream_table(0, observations, rows, list(range(5)), columns)

    assert table is None


def test_split_stream_table_segment_rejects_character_spaced_prose() -> None:
    rows = tuple(
        tuple(TableCell(row, column, "s p a c e d t e x t") for column in range(8))
        for row in range(3)
    )
    table = Table(order=0, rows=rows, metadata={"source": "stream"})

    assert internal_table_character_spaced_prose(table)


def test_split_semantic_table_separates_numeric_sections() -> None:
    rows = (
        (TableCell(0, 0, "Section A", column_span=2),),
        (
            TableCell(
                1,
                0,
                "A",
            ),
            TableCell(1, 1, "10"),
        ),
        (TableCell(2, 0, "Section B", column_span=2),),
        (
            TableCell(
                3,
                0,
                "B",
            ),
            TableCell(3, 1, "20"),
        ),
        (TableCell(4, 0, "Section C", column_span=2),),
        (
            TableCell(
                5,
                0,
                "C",
            ),
            TableCell(5, 1, "30"),
        ),
    )

    tables = internal_split_semantic_table(Table(order=0, rows=rows))

    assert len(tables) == 3
    assert [table.rows[0][0].text for table in tables] == [
        "Section A",
        "Section B",
        "Section C",
    ]


def line(x0: float, y0: float, x1: float, y1: float) -> SimpleNamespace:
    return SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1)


def text(value: str, x: float, y: float, sequence: int) -> SimpleNamespace:
    return SimpleNamespace(
        text=value,
        x0=x,
        y0=y,
        x1=x + 5.0,
        y1=y + 5.0,
        seqno=sequence,
        visible=True,
    )


def observations(runs: tuple[Any, ...]) -> ObservationBatch:
    return ObservationBatch.from_columns(
        (run.text for run in runs),
        ((run.x0, run.y0, run.x1, run.y1) for run in runs),
        source=ObservationSource.NATIVE,
        sequence=(run.seqno for run in runs),
        visible=(run.visible for run in runs),
        references=runs,
    )


def test_extract_tables_assigns_runs_to_ruled_cells() -> None:
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=SimpleNamespace(width=100.0, height=100.0),
            evidence=page_evidence(),
            grid_lines=(
                line(10.0, 90.0, 90.0, 90.0),
                line(10.0, 50.0, 90.0, 50.0),
                line(10.0, 10.0, 90.0, 10.0),
                line(10.0, 10.0, 10.0, 90.0),
                line(50.0, 10.0, 50.0, 90.0),
                line(90.0, 10.0, 90.0, 90.0),
            ),
            runs=(
                text("top-left", 20.0, 70.0, 0),
                text("top-right", 60.0, 70.0, 1),
                text("bottom-left", 20.0, 30.0, 2),
                text("bottom-right", 60.0, 30.0, 3),
            ),
        ),
    )

    tables = extract_tables(capture, observations(capture.runs))

    assert len(tables) == 1
    assert [[cell.text for cell in row] for row in tables[0].rows] == [
        ["top-left", "top-right"],
        ["bottom-left", "bottom-right"],
    ]


def test_merge_adjacent_tables_drops_repeated_continuation_header() -> None:
    def table(order: int, y_top: float, rows: tuple[tuple[str, ...], ...]) -> Table:
        return Table(
            order=order,
            rows=tuple(
                tuple(
                    TableCell(
                        row=row_index,
                        column=column_index,
                        text=text,
                        bbox=(
                            column_index * 50.0,
                            y_top - row_index * 10.0 - 10.0,
                            column_index * 50.0 + 40.0,
                            y_top - row_index * 10.0,
                        ),
                    )
                    for column_index, text in enumerate(values)
                )
                for row_index, values in enumerate(rows)
            ),
            bbox=(0.0, y_top - len(rows) * 10.0, 90.0, y_top),
        )

    merged = internal_merge_adjacent_tables(
        [
            table(0, 100.0, (("Name", "Value"), ("A", "1"))),
            table(1, 55.0, (("Name", "Value"), ("B", "2"))),
        ]
    )

    assert len(merged) == 1
    assert [[cell.text for cell in row] for row in merged[0].rows] == [
        ["Name", "Value"],
        ["A", "1"],
        ["B", "2"],
    ]


def test_extract_tables_ignores_many_observations_outside_ruled_component() -> None:
    table_runs = (
        text("top-left", 20.0, 70.0, 0),
        text("top-right", 60.0, 70.0, 1),
        text("bottom-left", 20.0, 30.0, 2),
        text("bottom-right", 60.0, 30.0, 3),
    )
    outside_runs = tuple(
        text(f"outside-{index}", 200.0 + index * 10.0, 200.0, index + 4) for index in range(300)
    )
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=SimpleNamespace(width=4_000.0, height=4_000.0),
            evidence=page_evidence(),
            grid_lines=(
                line(10.0, 90.0, 90.0, 90.0),
                line(10.0, 50.0, 90.0, 50.0),
                line(10.0, 10.0, 90.0, 10.0),
                line(10.0, 10.0, 10.0, 90.0),
                line(50.0, 10.0, 50.0, 90.0),
                line(90.0, 10.0, 90.0, 90.0),
            ),
            runs=(*table_runs, *outside_runs),
        ),
    )

    tables = extract_tables(capture, observations(capture.runs))

    assert len(tables) == 1
    assert [[cell.text for cell in row] for row in tables[0].rows] == [
        ["top-left", "top-right"],
        ["bottom-left", "bottom-right"],
    ]


def test_extract_tables_detects_aligned_borderless_rows() -> None:
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=SimpleNamespace(width=100.0, height=100.0),
            evidence=page_evidence(),
            grid_lines=(),
            runs=tuple(
                run
                for row, y in enumerate((80.0, 65.0, 50.0, 35.0))
                for run in (
                    text(f"label-{row}", 10.0, y, row * 2),
                    text(str(row + 1), 70.0, y, row * 2 + 1),
                )
            ),
        ),
    )

    tables = extract_tables(capture, observations(capture.runs))

    assert len(tables) == 1
    assert [[cell.text for cell in row] for row in tables[0].rows] == [
        ["label-0", "1"],
        ["label-1", "2"],
        ["label-2", "3"],
        ["label-3", "4"],
    ]


def test_stream_tables_prefer_horizontal_observations_over_rotated_noise() -> None:
    text_values = [value for row in range(4) for value in (f"label-{row}", str(row + 1))]
    boxes = [
        (x, y, x + 20.0, y + 6.0)
        for row, y in enumerate((80.0, 65.0, 50.0, 35.0))
        for x in (10.0, 70.0)
    ]
    text_values.extend(("rotated-a", "rotated-b", "rotated-c", "rotated-d"))
    boxes.extend(
        (
            (5.0, 10.0, 50.0, 15.0),
            (5.0, 20.0, 50.0, 25.0),
            (5.0, 30.0, 50.0, 35.0),
            (5.0, 40.0, 50.0, 45.0),
        )
    )
    observations = ObservationBatch.from_columns(
        text_values,
        boxes,
        source=ObservationSource.NATIVE,
        rotation=(0,) * 8 + (90,) * 4,
    )

    tables = internal_stream_tables(
        cast(CapturedPage, SimpleNamespace(page=SimpleNamespace(width=100.0, height=100.0))),
        observations,
        0,
    )

    assert len(tables) == 1
    assert [[cell.text for cell in row] for row in tables[0].rows] == [
        ["label-0", "1"],
        ["label-1", "2"],
        ["label-2", "3"],
        ["label-3", "4"],
    ]


def test_extract_tables_rejects_aligned_bullet_prose() -> None:
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=SimpleNamespace(width=100.0, height=100.0),
            evidence=page_evidence(),
            grid_lines=(),
            runs=tuple(
                run
                for row, y in enumerate((80.0, 65.0, 50.0, 35.0))
                for run in (
                    text("•", 10.0, y, row * 2),
                    text(
                        "This is a long prose list item without tabular values",
                        20.0,
                        y,
                        row * 2 + 1,
                    ),
                )
            ),
        ),
    )

    assert extract_tables(capture, observations(capture.runs)) == ()
