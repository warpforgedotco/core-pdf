# SPDX-License-Identifier: AGPL-3.0-only
from dataclasses import replace

import pytest

from core_pdf.impl._impl.output.model import Block, BlockKind, Table, TableCell, TextLine
from core_pdf_ocr.impl.extract.table_reconcile import internal_project_text_and_tables


def internal_chart(text: str, *, order: int = 0) -> Table:
    return Table(
        order,
        rows=((TableCell(0, 0, text),),),
        bbox=(20.0, 140.0, 260.0, 180.0),
        metadata={"source": "chart-ocr", "synthetic": True},
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("overlaid", [False, True])
def test_identical_chart_tables_require_overlap_and_keep_one_winner(
    reverse: bool, overlaid: bool
) -> None:
    # Poppler 26.07.0 `pdftotext -layout` preserves both copies in disjoint.pdf
    # and one in overlaid.pdf under /private/tmp/core-pdf-repeated-chart-tables.
    first = replace(
        internal_chart("Total 10"),
        rows=(
            (TableCell(0, 0, "Total"), TableCell(0, 1, "10")),
            (TableCell(1, 0, "CCPS"), TableCell(1, 1, "23")),
        ),
    )
    second = replace(first, order=1, bbox=first.bbox if overlaid else (20.0, 220.0, 260.0, 260.0))
    tables = (second, first) if reverse else (first, second)

    _, projected = internal_project_text_and_tables([], tables)

    assert projected == ((first,) if overlaid else tables)
    assert first.rows == second.rows


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("richer_source", ["stream", "chart-ocr"])
def test_overlapping_chart_projection_keeps_the_complete_richer_text(
    reverse: bool, richer_source: str
) -> None:
    short = internal_chart("musculoskeletal diseases 182")
    rich = replace(
        short,
        order=1,
        rows=((TableCell(0, 0, "musculoskeletal diseases 182 Public Health 2022"),),),
        metadata={"source": richer_source, "synthetic": richer_source == "chart-ocr"},
    )
    tables = (rich, short) if reverse else (short, rich)

    _, projected = internal_project_text_and_tables([], tables)

    assert projected == (rich,)


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_chart_can_cover_a_shorter_stream_table(reverse: bool) -> None:
    chart = internal_chart("Total 10 CCPS 23")
    stream = replace(
        chart,
        order=1,
        rows=((TableCell(0, 0, "Total 10"),),),
        metadata={"source": "stream"},
    )
    tables = (stream, chart) if reverse else (chart, stream)

    _, projected = internal_project_text_and_tables([], tables)

    assert projected == (chart,)


@pytest.mark.parametrize("reverse", [False, True])
def test_chart_replacement_survives_partial_block_coverage(reverse: bool) -> None:
    # Poppler 26.07.0 `pdftotext -layout` and its pdftoppm raster preserve all
    # sixteen words in /private/tmp/core-pdf-chart-reference-survival/verified.pdf.
    # A partial block must not erase the table selected to preserve the chart.
    complete_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen"
    )
    chart = internal_chart("one two three")
    stream = replace(
        chart,
        order=1,
        rows=((TableCell(0, 0, complete_text),),),
        metadata={"source": "stream"},
    )
    block = Block(
        0,
        BlockKind.PARAGRAPH,
        (TextLine(complete_text.removeprefix("one two three "), bbox=chart.bbox),),
        bbox=chart.bbox,
    )

    blocks, projected = internal_project_text_and_tables(
        [block], (stream, chart) if reverse else (chart, stream)
    )

    assert blocks == []
    assert projected == (stream,)


@pytest.mark.parametrize(
    ("first_text", "second_text"),
    [("Limit >5", "Limit 5"), ("Limit > 5", "Limit 5"), ("> 5", "5"), ("Total 10", "total 10")],
)
def test_chart_projection_keeps_punctuation_and_case_differences(
    first_text: str, second_text: str
) -> None:
    tables = (internal_chart(first_text), internal_chart(second_text, order=1))

    _, projected = internal_project_text_and_tables([], tables)

    assert projected == tables


@pytest.mark.parametrize("unknown_geometry", [False, True])
def test_chart_needs_complete_spatial_coverage_to_remove_a_copy(unknown_geometry: bool) -> None:
    first = internal_chart("Total 10")
    second = replace(
        first,
        order=1,
        bbox=None if unknown_geometry else (200.0, 140.0, 440.0, 180.0),
    )

    _, projected = internal_project_text_and_tables([], (first, second))

    assert projected == (first, second)


def test_disjoint_chart_survives_text_shared_with_a_block_and_real_table() -> None:
    chart = replace(internal_chart("1 11 14 13 17 23 12 10"), bbox=(20.0, 200.0, 260.0, 230.0))
    table = Table(
        1,
        rows=(
            (TableCell(0, 0, "Unit Determinations"), TableCell(0, 1, "FY18")),
            (TableCell(1, 0, "CCPS"), TableCell(1, 1, "1 11 14 13")),
            (TableCell(2, 0, "SCoB"), TableCell(2, 1, "17 23 12 10")),
        ),
        bbox=(20.0, 120.0, 260.0, 190.0),
    )
    caption = Block(
        0,
        BlockKind.HEADING,
        (TextLine(chart.rows[0][0].text, bbox=(20.0, 300.0, 260.0, 320.0)),),
        bbox=(20.0, 300.0, 260.0, 320.0),
    )

    blocks, projected = internal_project_text_and_tables([caption], (chart, table))

    assert blocks == [caption]
    assert projected == (chart, table)
