# SPDX-License-Identifier: AGPL-3.0-only
"""Table facts share measurements without flattening distinct acceptance rules."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core_pdf.impl._impl.extract.table_cleanup import (
    internal_stream_table_reads_like_prose,
    internal_table_has_grid_shape,
    internal_table_is_single_column_prose,
    internal_table_quality,
)
from core_pdf.impl._impl.extract.table_facts import internal_TableFacts
from core_pdf.impl._impl.output.model import Table, TableCell


@pytest.mark.parametrize(
    ("widths", "has_grid", "single_column"),
    [
        ((), False, False),
        ((1,), False, False),
        ((2, 2), True, False),
        ((1, 1), False, False),
        ((1, 1, 1), False, True),
        ((1, 1, 2, 2), False, False),
        ((0, 1, 1, 2), False, True),
        ((0, 1, 2, 2), True, False),
    ],
)
def test_shape_rules_preserve_row_minima_ties_and_spans(
    widths: tuple[int, ...], has_grid: bool, single_column: bool
) -> None:
    rows = tuple(
        tuple(
            TableCell(row=row, column=column, text="value", column_span=2 if width == 1 else 1)
            for column in range(width)
        )
        for row, width in enumerate(widths)
    )
    table = Table(order=0, rows=rows)
    facts = internal_TableFacts.from_rows(rows)

    assert internal_table_has_grid_shape(table, facts=facts) is has_grid
    assert internal_table_is_single_column_prose(table, facts=facts) is single_column


def test_table_facts_distinguish_physical_shape_population_and_spans() -> None:
    rows = (
        (),
        (TableCell(row=1, column=0, text="   ", column_span=3),),
        (
            TableCell(row=2, column=0, text=" 123 "),
            TableCell(row=2, column=1, text=" a b c d "),
        ),
    )
    facts = internal_TableFacts.from_rows(rows)

    assert (facts.row_count, facts.nonempty_rows, facts.populated_rows) == (3, 2, 1)
    assert (facts.columns, facts.spanned_columns, facts.cell_count) == (2, 3, 3)
    assert (facts.divided_rows, facts.single_cell_rows) == (1, 1)
    assert facts.filled_texts == ("123", "a b c d")
    assert facts.numeric_cells == 1
    assert facts.numeric_density == 0.5
    assert facts.character_spaced_cells == 1
    assert facts.text_lengths == (3, 7)
    assert internal_table_quality(Table(order=0, rows=rows)) == (1, 2, 1 / 3, 3, -2)


def test_facts_snapshot_does_not_reuse_classifications_after_text_changes() -> None:
    original = (TableCell(row=0, column=0, text="a b c d"),)
    facts = internal_TableFacts.from_rows((original,))
    assert facts.numeric_cells == 0
    assert facts.character_spaced_cells == 1

    changed = (replace(original[0], text="12"),)
    changed_facts = internal_TableFacts.from_rows((changed,))
    assert changed_facts.numeric_cells == 1
    assert changed_facts.character_spaced_cells == 0
    assert facts.filled_texts == ("a b c d",)


def test_prose_gate_keeps_its_short_digit_bearing_rule() -> None:
    # The prose gate gives short identifiers numeric credit even when digits
    # are not a majority. The generic numeric-density fact must not replace it.
    rows = tuple(
        (
            TableCell(row=row, column=0, text="a long descriptive sentence in this table"),
            TableCell(row=row, column=1, text="another long descriptive sentence here"),
            TableCell(row=row, column=2, text="identifier1"),
        )
        for row in range(3)
    )
    facts = internal_TableFacts.from_rows(rows)
    assert facts.numeric_cells == 0
    assert not internal_stream_table_reads_like_prose(Table(order=0, rows=rows))
