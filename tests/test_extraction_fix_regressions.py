"""Pin the extraction defects fixed in the SCORE-Bench improvement pass.

Every test here corresponds to a bug that reached the benchmark, and each
one failed before its fix.  They exist because the benchmark alone is a poor
guard for this class of defect: it reports means over 224 documents, so a
single page losing half its text moves the headline by a thousandth and
hides.  Several of these bugs were silent content deletion -- text extracted
correctly and then discarded -- which no aggregate score makes obvious.

Where a defect is expressible over synthetic input the test builds it
directly; where it only reproduces through a real font program or content
stream it uses the smallest benchmark fixture that exercises it.
"""

from __future__ import annotations

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.extract.block_layout import internal_column_major_prose
from core_pdf.impl.extract.contracts import (
    ParsedBlock,
    ParsedLine,
)
from core_pdf.impl.extract.emit import (
    internal_corrupt_native_block,
    internal_symbol_characters,
)
from core_pdf.impl.extract.pipeline import internal_PageExtraction
from core_pdf.impl.extract.regions import internal_peel_spanning_band
from core_pdf.impl.extract.table_cleanup import (
    internal_merge_wrapped_cell_rows,
    internal_stream_table_reads_like_prose,
)
from core_pdf.impl.output.model import Table, TableCell
from tests.helpers.paths import score_bench_pdf
from tests.helpers.structured import cell, native_block, stream_table


def test_decimal_separators_are_not_symbols() -> None:
    """A number's separators belong to the number, not to punctuation.

    Counting them made a table of measurements read as symbol soup, which is
    how this module recognizes a damaged text layer.
    """
    assert internal_symbol_characters("79.4 105.1 108.9 102.3") == 0
    assert internal_symbol_characters("1,056.1 (58.6) 39.9") > 0  # brackets still count
    # Digits and punctuation interleaved without forming numbers stay visible:
    # the exemption is granted per token, never per character.
    assert internal_symbol_characters("1911*2.1,z,z") >= 4


def test_numeric_table_block_survives_corruption_filter() -> None:
    """Regression: whole numeric tables were deleted as corrupt native text."""
    numeric = native_block(
        "79.4 105.1 108.9 102.3\n5.1 6.5 5.0 4.5\n6.4 6.2 7.3 4.4",
        (20.0, 100.0, 260.0, 160.0),
    )
    assert not internal_corrupt_native_block(numeric)


@pytest.mark.parametrize(
    "text",
    [
        "1–2 Reserved; must be 0.",
        "7–8 Reserved; must be 1.",
        "13–32 (Revision 3 or greater) Reserved; must be 1.",
    ],
)
def test_short_specification_table_rows_survive_corruption_filter(text: str) -> None:
    """Regression: valid permission rows were deleted as corrupt native text."""
    assert not internal_corrupt_native_block(native_block(text, (20.0, 100.0, 260.0, 120.0)))


def test_damaged_native_layer_is_still_rejected() -> None:
    """The corruption filter must keep catching mojibake after the fix."""
    corrupt = native_block(
        "76391*11 IOIIlo6 I * 9 2*0 118)96 '1'1322) '1'19)20 IZZO 1911*2.1,z,z CSM/l\":OST L*O*Io",
        (20.0, 200.0, 260.0, 260.0),
    )
    assert internal_corrupt_native_block(corrupt)


def test_column_major_reorder_partitions_lines_exactly() -> None:
    """Regression: lines near two column rails were emitted twice.

    Membership was re-derived with a fixed-width window after assigning each
    line to its nearest rail, so a line between rails belonged to both and a
    line outside every window belonged to none.
    """
    # Rails must sit inside one window-width of each other for the defect to
    # appear: clustering starts a new rail past 40pt, while membership was
    # re-derived with a 40pt window either side, so anything between two
    # rails closer than 80pt belonged to both.
    rails = (50.0, 105.0, 160.0)
    lines: list[ParsedLine] = []
    for row in range(27):
        top = 700.0 - row * 12.0
        for column_index, left in enumerate(rails):
            lines.append(
                ParsedLine(
                    text=f"rail {column_index} row {row} of running prose text",
                    bbox=(left, top - 10.0, left + 40.0, top),
                    source="native",
                )
            )
    # An indented opener between two rails: the shape that was emitted twice.
    lines.append(
        ParsedLine(
            text="indented opener between two rails",
            bbox=(130.0, 100.0, 170.0, 110.0),
            source="native",
        )
    )
    block = ParsedBlock(lines=tuple(lines), bbox=(50.0, 90.0, 200.0, 700.0), kind="paragraph")

    reordered = internal_column_major_prose([block])[0]

    assert sorted(line.text for line in reordered.lines) == sorted(
        line.text for line in block.lines
    )


def internal_two_column_boxes(spanner_at_bottom: bool) -> numpy.ndarray:
    boxes: list[tuple[float, float, float, float]] = []
    for row in range(20):
        top = 700.0 - row * 14.0
        boxes.append((50.0, top - 12.0, 240.0, top))
        boxes.append((260.0, top - 12.0, 450.0, top))
    banner = (50.0, 380.0, 450.0, 400.0) if spanner_at_bottom else (50.0, 720.0, 450.0, 740.0)
    boxes.append(banner)
    return numpy.asarray(boxes, dtype=numpy.float64)


def test_bottom_spanner_peel_exposes_hidden_gutter() -> None:
    """Regression: a banner under two columns hid their gutter permanently.

    Peeling only ever removed bands from the top, so a spanner below the
    columns left the region unsplittable and the columns interleaved.
    """
    boxes = internal_two_column_boxes(spanner_at_bottom=True)
    indexes = numpy.arange(len(boxes))

    assert internal_peel_spanning_band(indexes, boxes, 12.0) is None
    peeled = internal_peel_spanning_band(indexes, boxes, 12.0, from_bottom=True)
    assert peeled is not None
    band, remainder = peeled
    assert band.tolist() == [len(boxes) - 1]
    assert len(remainder) == len(boxes) - 1


def test_bottom_peel_ignores_small_key_value_panels() -> None:
    """A receipt's totals rows must not read as a column gutter.

    Removing them exposes an aligned label/value gap that is genuinely
    row-major, so the peel demands a paragraph's worth of lines per side.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for row in range(6):
        top = 300.0 - row * 14.0
        boxes.append((50.0, top - 12.0, 180.0, top))
        boxes.append((260.0, top - 12.0, 400.0, top))
    boxes.append((50.0, 180.0, 400.0, 200.0))
    array = numpy.asarray(boxes, dtype=numpy.float64)

    assert (
        internal_peel_spanning_band(numpy.arange(len(array)), array, 12.0, from_bottom=True) is None
    )


WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


def internal_wrapped_cell_table() -> Table:
    """Four logical rows whose cells wrap over several lines each.

    The cell that wraps furthest opens its logical row and runs the row's
    whole height, which is exactly what holds the row open across the
    shorter cells beside it. Cell text carries no digits: a numeric table
    records one datum per line and is deliberately never regrouped.
    """
    rows: list[tuple[TableCell, ...]] = []
    physical = 0
    for logical in range(4):
        top = 700.0 - logical * 100.0
        for line in range(3):
            y1 = top - line * 14.0
            word = WORDS[(logical + line) % len(WORDS)]
            description = (
                (160.0, top - 40.0, 300.0, top) if line == 0 else (160.0, y1 - 12.0, 300.0, y1)
            )
            rows.append(
                (
                    cell(physical, 0, f"label {word}", (50.0, y1 - 12.0, 150.0, y1)),
                    cell(
                        physical,
                        1,
                        "long wrapped description" if line == 0 else f"continued {word}",
                        description,
                    ),
                    cell(physical, 2, f"practice {word}", (310.0, y1 - 12.0, 430.0, y1)),
                    cell(physical, 3, f"resources {word}", (440.0, y1 - 12.0, 560.0, y1)),
                    cell(physical, 4, f"marketing {word}", (570.0, y1 - 12.0, 690.0, y1)),
                )
            )
            physical += 1
    return stream_table(tuple(rows))


def test_wrapped_cells_group_into_logical_rows() -> None:
    """Regression: wrapped table cells were read one line at a time.

    Reading per line interleaves the wrapped fragments of every column, so
    the content is entirely present and entirely in the wrong place.
    """
    merged = internal_merge_wrapped_cell_rows(internal_wrapped_cell_table())
    assert [row[0].text for row in merged.rows] == [
        "label alpha label beta label gamma",
        "label beta label gamma label delta",
        "label gamma label delta label epsilon",
        "label delta label epsilon label zeta",
    ]
    assert all(len(row) == 5 for row in merged.rows)
    assert all(row[1].text.startswith("long wrapped description") for row in merged.rows)


def test_numeric_table_rows_are_never_regrouped() -> None:
    """A statistical table records one datum per line; its lines are rows."""

    def row(index: int) -> tuple[TableCell, ...]:
        top = 712.0 - index * 12.0
        values = (f"Sample {index}", "12.5", "3.75", "0.82", "44.1")
        return tuple(
            cell(
                index,
                column,
                value,
                (50.0 + column * 70.0, top - 12.0, 150.0 + column * 70.0, top),
            )
            for column, value in enumerate(values)
        )

    rows = tuple(row(index) for index in range(10))
    table = stream_table(rows)
    assert internal_merge_wrapped_cell_rows(table).rows == table.rows


def test_word_rails_over_prose_are_not_a_table() -> None:
    """Whitespace alignment over justified prose yields a word grid."""
    rail_words = ("the", "quick", "brown", "fox", "jumps", "over", "lazy", "dogs", "again")

    def rail_row(row: int) -> tuple[TableCell, ...]:
        top = 712.0 - row * 12.0
        return tuple(
            cell(
                row,
                column,
                word,
                (50.0 + column * 60.0, top - 12.0, 100.0 + column * 60.0, top),
            )
            for column, word in enumerate(rail_words)
        )

    rows = tuple(rail_row(row) for row in range(14))
    assert internal_stream_table_reads_like_prose(stream_table(rows))


def test_sparse_long_celled_narrow_table_is_not_a_table() -> None:
    """Side-by-side lists picked up as one table read row-major and interleave."""
    rows: list[tuple[TableCell, ...]] = []
    for index in range(8):
        top = 700.0 - index * 14.0
        cells = [
            cell(
                index,
                0,
                "a fairly long list entry describing something",
                (50.0, top - 12.0, 300.0, top),
            )
        ]
        if index % 3 == 0:
            cells.append(
                cell(
                    index,
                    1,
                    "another long entry in the parallel list",
                    (320.0, top - 12.0, 560.0, top),
                )
            )
        else:
            cells.append(cell(index, 1, "", (320.0, top - 12.0, 560.0, top)))
        rows.append(tuple(cells))
    assert internal_stream_table_reads_like_prose(stream_table(tuple(rows)))


def test_numeric_table_reaches_the_page() -> None:
    """End-to-end guard: the deleted numbers must survive to the output."""
    fixture = score_bench_pdf("Tobacco-Lab-Reproducibility-Tables-p002.pdf")
    with PdfDocument.open(fixture) as document:
        text = document.extract().text
    for token in ("79.4", "105.1", "108.9"):
        assert token in text


def test_cell_background_does_not_paint_over_its_text() -> None:
    """Regression: fills shared a sequence number with the text inside them.

    A drawing was stamped with the current sequence number without consuming
    it, so a table cell's background tied with the run it contains and the
    replay painted the fill over the row's leading glyphs.  Measured as ink:
    the overpainted header lost roughly a third of its pixels.
    """
    fixture = score_bench_pdf("fhhd0346-p009.pdf")
    scale = 2.0
    with PdfDocument.open(fixture) as document:
        capture = internal_PageExtraction(document.pages[0]).capture
        header = next(run for run in capture.program.runs if (run.text or "").strip() == "Material")
        raster = document.pages[0].render().rasterize(scale=scale)

    pixels = raster.array()[:, :, :3].min(axis=2)
    height = pixels.shape[0]
    crop = pixels[
        max(0, int(height - header.y1 * scale) - 4) : int(height - header.y0 * scale) + 4,
        max(0, int(header.x0 * scale) - 4) : int(header.x1 * scale) + 4,
    ]
    # The intact header measures ~460 ink pixels at this scale; the overpainted
    # one lost roughly a third of them.
    assert int((crop < 128).sum()) > 400
