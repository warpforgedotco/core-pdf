# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable table-shape and cell-text facts, independent of acceptance policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from core_pdf.impl._impl.output.model import TableCell


def internal_numeric_cell(text: str) -> bool:
    alphanumeric = sum(character.isalnum() for character in text)
    digits = sum(character.isdigit() for character in text)
    return bool(digits and digits * 2 >= max(1, alphanumeric))


def internal_character_spaced_cell(text: str) -> bool:
    tokens = [token for token in text.split() if any(character.isalpha() for character in token)]
    if len(tokens) < 4:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    return single_character / len(tokens) >= 0.50


@dataclass(frozen=True)
class internal_TableFacts:
    """Facts for one row snapshot; rebuild after merging or rewriting its cells.

    Physical columns and spanned columns are intentionally separate. Expensive
    text classifications are lazy so a shape-only gate does not scan every word.
    """

    row_count: int
    nonempty_rows: int
    populated_rows: int
    columns: int
    spanned_columns: int
    cell_count: int
    divided_rows: int
    single_cell_rows: int
    filled_texts: tuple[str, ...]

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[TableCell]]) -> internal_TableFacts:
        nonempty_rows = populated_rows = columns = spanned_columns = cell_count = 0
        divided_rows = single_cell_rows = 0
        filled_texts: list[str] = []
        for row in rows:
            size = len(row)
            nonempty_rows += bool(size)
            columns = max(columns, size)
            cell_count += size
            divided_rows += size >= 2
            single_cell_rows += size == 1
            previous_count = len(filled_texts)
            for cell in row:
                spanned_columns = max(spanned_columns, cell.column + cell.column_span)
                text = cell.text.strip()
                if text:
                    filled_texts.append(text)
            populated_rows += len(filled_texts) > previous_count
        return cls(
            len(rows),
            nonempty_rows,
            populated_rows,
            columns,
            spanned_columns,
            cell_count,
            divided_rows,
            single_cell_rows,
            tuple(filled_texts),
        )

    @cached_property
    def numeric_cells(self) -> int:
        """Populated cells whose digits make up at least half their alphanumerics."""
        return sum(internal_numeric_cell(text) for text in self.filled_texts)

    @property
    def numeric_density(self) -> float:
        return self.numeric_cells / max(1, len(self.filled_texts))

    @cached_property
    def character_spaced_cells(self) -> int:
        return sum(internal_character_spaced_cell(text) for text in self.filled_texts)

    @cached_property
    def text_lengths(self) -> tuple[int, ...]:
        return tuple(map(len, self.filled_texts))

    @property
    def average_cell_length(self) -> float:
        return sum(self.text_lengths) / max(1, len(self.filled_texts))
