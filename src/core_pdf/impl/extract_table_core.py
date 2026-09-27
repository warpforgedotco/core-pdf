# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import ClassVar, Protocol, TypeAlias

import numpy

from core_pdf.impl.array_views import finite_median
from core_pdf.impl.extract_contracts import ObservationBatch, PageAnalysis
from core_pdf.impl.output_model import Table, TableCell
from core_pdf.impl.spatial import cluster_1d
from core_pdf.impl.text import collapse_ws
from core_pdf.impl.types import FrozenFields, Record, ReplaceFields, ReprFields, frozen_setattr


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


SHORT_DIGIT_CELL_CHARACTERS = 25


def numeric_cell(text: str) -> bool:
    alphanumeric = sum(character.isalnum() for character in text)
    digits = sum(character.isdigit() for character in text)
    return bool(digits and digits * 2 >= max(1, alphanumeric))


def digit_bearing_cell(text: str) -> bool:
    return numeric_cell(text) or any(character.isdigit() for character in text)


def short_digit_cell(text: str) -> bool:
    return len(text) <= SHORT_DIGIT_CELL_CHARACTERS and any(
        character.isdigit() for character in text
    )


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


TableQuality: TypeAlias = tuple[int, int, float, int, int]


class TableCandidate(Record):
    __slots__ = ("table", "cached_facts")

    table: Table
    cached_facts: TableFacts | None

    __fields__: ClassVar[tuple[str, ...]] = ("table", "cached_facts")
    __match_args__ = ("table", "cached_facts")

    def __init__(self, table: Table, cached_facts: TableFacts | None = None) -> None:
        frozen_setattr(self, "table", table)
        frozen_setattr(self, "cached_facts", cached_facts)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.table == other.table

    def __hash__(self) -> int:
        return hash(self.table)

    @property
    def facts(self) -> TableFacts:
        facts = self.cached_facts
        if facts is None:
            facts = TableFacts.from_rows(self.table.rows)
            frozen_setattr(self, "cached_facts", facts)
        return facts

    @property
    def quality(self) -> TableQuality:
        facts = self.facts
        populated = len(facts.filled_texts)
        density = populated / max(1, facts.row_count * facts.columns)
        return (int(2 <= facts.columns <= 16), populated, density, facts.row_count, -facts.columns)

    def with_table(self, table: Table) -> TableCandidate:
        if table is self.table:
            return self
        return TableCandidate(table, self.cached_facts if table.rows is self.table.rows else None)


class ObservationCoordinates(Record):
    __slots__ = ("x0", "y0", "x1", "y1", "y_centers", "widths", "heights", "sequences")

    x0: list[float]
    y0: list[float]
    x1: list[float]
    y1: list[float]
    y_centers: list[float]
    widths: list[float]
    heights: list[float]
    sequences: list[int]

    __fields__: ClassVar[tuple[str, ...]] = (
        "x0",
        "y0",
        "x1",
        "y1",
        "y_centers",
        "widths",
        "heights",
        "sequences",
    )
    __match_args__ = ("x0", "y0", "x1", "y1", "y_centers", "widths", "heights", "sequences")

    def __init__(
        self,
        x0: list[float],
        y0: list[float],
        x1: list[float],
        y1: list[float],
        y_centers: list[float],
        widths: list[float],
        heights: list[float],
        sequences: list[int],
    ) -> None:
        frozen_setattr(self, "x0", x0)
        frozen_setattr(self, "y0", y0)
        frozen_setattr(self, "x1", x1)
        frozen_setattr(self, "y1", y1)
        frozen_setattr(self, "y_centers", y_centers)
        frozen_setattr(self, "widths", widths)
        frozen_setattr(self, "heights", heights)
        frozen_setattr(self, "sequences", sequences)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
            and self.y_centers == other.y_centers
            and self.widths == other.widths
            and self.heights == other.heights
            and self.sequences == other.sequences
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.x0,
                self.y0,
                self.x1,
                self.y1,
                self.y_centers,
                self.widths,
                self.heights,
                self.sequences,
            )
        )

    @classmethod
    def from_observations(cls, observations: ObservationBatch) -> ObservationCoordinates:
        bbox = observations.bbox
        return cls(
            bbox[:, 0].tolist(),
            bbox[:, 1].tolist(),
            bbox[:, 2].tolist(),
            bbox[:, 3].tolist(),
            ((bbox[:, 1] + bbox[:, 3]) * 0.5).tolist(),
            (bbox[:, 2] - bbox[:, 0]).tolist(),
            (bbox[:, 3] - bbox[:, 1]).tolist(),
            observations.sequence.tolist(),
        )


class TableAnalysis(Record):
    __slots__ = ("observations", "coordinates", "text_rows", "row_centers", "candidate_columns")

    observations: ObservationBatch
    coordinates: ObservationCoordinates
    text_rows: list[list[int]]
    row_centers: list[float]
    candidate_columns: list[list[tuple[int, int]]]

    __fields__: ClassVar[tuple[str, ...]] = (
        "observations",
        "coordinates",
        "text_rows",
        "row_centers",
        "candidate_columns",
    )
    __match_args__ = (
        "observations",
        "coordinates",
        "text_rows",
        "row_centers",
        "candidate_columns",
    )

    def __init__(
        self,
        observations: ObservationBatch,
        coordinates: ObservationCoordinates,
        text_rows: list[list[int]],
        row_centers: list[float],
        candidate_columns: list[list[tuple[int, int]]],
    ) -> None:
        frozen_setattr(self, "observations", observations)
        frozen_setattr(self, "coordinates", coordinates)
        frozen_setattr(self, "text_rows", text_rows)
        frozen_setattr(self, "row_centers", row_centers)
        frozen_setattr(self, "candidate_columns", candidate_columns)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.observations == other.observations
            and self.coordinates == other.coordinates
            and self.text_rows == other.text_rows
            and self.row_centers == other.row_centers
            and self.candidate_columns == other.candidate_columns
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.observations,
                self.coordinates,
                self.text_rows,
                self.row_centers,
                self.candidate_columns,
            )
        )

    @classmethod
    def build(cls, observations: ObservationBatch, page_width: float) -> TableAnalysis:
        coordinates = ObservationCoordinates.from_observations(observations)
        text_rows = group_text_rows(observations, coordinates)
        return cls(
            observations,
            coordinates,
            text_rows,
            compute_row_centers(text_rows, coordinates),
            aligned_column_clusters(
                observations,
                text_rows,
                page_width,
                coordinates,
                minimum_rows=2,
            ),
        )


COLUMN_TOLERANCE = 14.0


def group_text_rows(
    observations: ObservationBatch,
    coordinates: ObservationCoordinates,
) -> list[list[int]]:
    visible_flags = observations.visible.tolist()
    rotations = observations.rotation.tolist()
    visible = [
        index
        for index, text in enumerate(observations.text)
        if visible_flags[index] and text.strip() and rotations[index] == 0
    ]
    if not visible:
        return []
    all_centers = coordinates.y_centers
    all_lefts = coordinates.x0
    all_heights = coordinates.heights
    sequences = coordinates.sequences
    ordered = sorted(
        visible,
        key=lambda index: (-all_centers[index], all_lefts[index], sequences[index]),
    )
    rows: list[list[int]] = []
    centers: list[float] = []
    heights: list[float] = []
    for index in ordered:
        center = all_centers[index]
        height = max(1.0, all_heights[index])
        if rows and abs(center - centers[-1]) <= max(2.0, min(height, heights[-1]) * 0.5):
            count = len(rows[-1])
            rows[-1].append(index)
            centers[-1] = (centers[-1] * count + center) / (count + 1)
            heights[-1] = (heights[-1] * count + height) / (count + 1)
        else:
            rows.append([index])
            centers.append(center)
            heights.append(height)
    return [sorted(row, key=lambda index: (all_lefts[index], sequences[index])) for row in rows]


def compute_row_centers(
    rows: list[list[int]],
    coordinates: ObservationCoordinates,
) -> list[float]:
    centers = coordinates.y_centers
    return [sum(centers[index] for index in row) / len(row) for row in rows]


def aligned_column_clusters(
    observations: ObservationBatch,
    rows: list[list[int]],
    page_width: float,
    coordinates: ObservationCoordinates,
    *,
    minimum_rows: int = 2,
) -> list[list[tuple[int, int]]]:
    tolerance = max(COLUMN_TOLERANCE, min(24.0, page_width * 0.04))
    all_lefts = coordinates.x0
    all_widths = coordinates.widths
    sequences = coordinates.sequences
    positions = [
        (all_lefts[index], row_index, index) for row_index, row in enumerate(rows) for index in row
    ]
    positions.sort(key=lambda item: (item[0], item[1], sequences[item[2]]))
    clusters = [
        [positions[position][1:] for position in group]
        for group in cluster_1d([item[0] for item in positions], tolerance, linkage="mean")
    ]
    candidates = []
    for cluster in clusters:
        row_support = {row_index for row_index, _ in cluster}
        widths = [all_widths[index] for _, index in cluster]
        alphanumeric = sum(
            any(character.isalnum() for character in observations.text[index])
            for _, index in cluster
        )
        if (
            len(row_support) >= minimum_rows
            and finite_median(numpy.asarray(widths, dtype=numpy.float32)) <= page_width * 0.48
            and alphanumeric * 2 >= len(cluster)
        ):
            candidates.append(cluster)
    return candidates


class TableContext(Record):
    __slots__ = ("capture", "analysis")

    capture: PageAnalysis
    analysis: TableAnalysis

    __fields__: ClassVar[tuple[str, ...]] = ("capture", "analysis")
    __match_args__ = ("capture", "analysis")

    def __init__(self, capture: PageAnalysis, analysis: TableAnalysis) -> None:
        frozen_setattr(self, "capture", capture)
        frozen_setattr(self, "analysis", analysis)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.capture == other.capture and self.analysis == other.analysis

    def __hash__(self) -> int:
        return hash((self.capture, self.analysis))


class TableSource(Protocol):
    def detect(self, context: TableContext, start_order: int, /) -> tuple[TableCandidate, ...]: ...

    def admit(
        self, accepted: list[TableCandidate], found: tuple[TableCandidate, ...], /
    ) -> list[TableCandidate]: ...


class TableTransform(Protocol):
    def __call__(self, candidates: list[TableCandidate], /) -> list[TableCandidate]: ...


class TableRule(Record):
    __slots__ = ("name", "rejects")

    name: str
    rejects: Callable[[TableCandidate], bool]

    __fields__: ClassVar[tuple[str, ...]] = ("name", "rejects")
    __match_args__ = ("name", "rejects")

    def __init__(self, name: str, rejects: Callable[[TableCandidate], bool]) -> None:
        frozen_setattr(self, "name", name)
        frozen_setattr(self, "rejects", rejects)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.name == other.name and self.rejects == other.rejects

    def __hash__(self) -> int:
        return hash((self.name, self.rejects))
