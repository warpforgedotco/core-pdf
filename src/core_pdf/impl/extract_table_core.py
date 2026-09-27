# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import ClassVar

from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf.impl.output_model import TableCell
from core_pdf.impl.text import collapse_ws
from core_pdf.impl.types import FrozenFields, ReplaceFields, ReprFields, frozen_setattr


def cell_text(
    observations: ObservationBatch,
    indexes: list[int],
) -> str:
    boxes = observations.bbox[indexes]
    centers = ((boxes[:, 1] + boxes[:, 3]) * 0.5).tolist()
    lefts = boxes[:, 0].tolist()
    sequences = observations.sequence[indexes].tolist()
    ordered = sorted(
        range(len(indexes)),
        key=lambda position: (-centers[position], lefts[position], sequences[position]),
    )
    parts = []
    for position in ordered:
        part = collapse_ws(observations.text[indexes[position]])
        if part:
            parts.append(part)
    return " ".join(parts)


def numeric_cell(text: str) -> bool:
    alphanumeric = sum(character.isalnum() for character in text)
    digits = sum(character.isdigit() for character in text)
    return bool(digits and digits * 2 >= max(1, alphanumeric))


def character_spaced_cell(text: str) -> bool:
    tokens = [token for token in text.split() if any(character.isalpha() for character in token)]
    if len(tokens) < 4:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    return single_character / len(tokens) >= 0.50


class TableFacts(FrozenFields, ReplaceFields, ReprFields):
    row_count: int
    nonempty_rows: int
    populated_rows: int
    columns: int
    spanned_columns: int
    cell_count: int
    single_cell_rows: int
    filled_texts: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "row_count",
        "nonempty_rows",
        "populated_rows",
        "columns",
        "spanned_columns",
        "cell_count",
        "single_cell_rows",
        "filled_texts",
    )
    __match_args__ = (
        "row_count",
        "nonempty_rows",
        "populated_rows",
        "columns",
        "spanned_columns",
        "cell_count",
        "single_cell_rows",
        "filled_texts",
    )

    def __init__(
        self,
        row_count: int,
        nonempty_rows: int,
        populated_rows: int,
        columns: int,
        spanned_columns: int,
        cell_count: int,
        single_cell_rows: int,
        filled_texts: tuple[str, ...],
    ) -> None:
        frozen_setattr(self, "row_count", row_count)
        frozen_setattr(self, "nonempty_rows", nonempty_rows)
        frozen_setattr(self, "populated_rows", populated_rows)
        frozen_setattr(self, "columns", columns)
        frozen_setattr(self, "spanned_columns", spanned_columns)
        frozen_setattr(self, "cell_count", cell_count)
        frozen_setattr(self, "single_cell_rows", single_cell_rows)
        frozen_setattr(self, "filled_texts", filled_texts)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.row_count == other.row_count
            and self.nonempty_rows == other.nonempty_rows
            and self.populated_rows == other.populated_rows
            and self.columns == other.columns
            and self.spanned_columns == other.spanned_columns
            and self.cell_count == other.cell_count
            and self.single_cell_rows == other.single_cell_rows
            and self.filled_texts == other.filled_texts
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.row_count,
                self.nonempty_rows,
                self.populated_rows,
                self.columns,
                self.spanned_columns,
                self.cell_count,
                self.single_cell_rows,
                self.filled_texts,
            )
        )

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[TableCell]]) -> TableFacts:
        nonempty_rows = populated_rows = columns = spanned_columns = cell_count = 0
        single_cell_rows = 0
        filled_texts: list[str] = []
        for row in rows:
            size = len(row)
            nonempty_rows += bool(size)
            columns = max(columns, size)
            cell_count += size
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
            single_cell_rows,
            tuple(filled_texts),
        )

    @cached_property
    def numeric_cells(self) -> int:
        return sum(numeric_cell(text) for text in self.filled_texts)

    @property
    def numeric_density(self) -> float:
        return self.numeric_cells / max(1, len(self.filled_texts))

    @cached_property
    def character_spaced_cells(self) -> int:
        return sum(character_spaced_cell(text) for text in self.filled_texts)

    @cached_property
    def text_lengths(self) -> tuple[int, ...]:
        return tuple(map(len, self.filled_texts))

    @property
    def average_cell_length(self) -> float:
        return sum(self.text_lengths) / max(1, len(self.filled_texts))
