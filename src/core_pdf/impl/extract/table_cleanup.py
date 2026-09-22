# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence
from copy import replace
from functools import cached_property
from statistics import fmean
from typing import Any, ClassVar, Self

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.model.geometry import (
    bbox_union,
    horizontal_overlap_ratio,
    interval_overlap,
    union_bbox,
)
from core_pdf.impl.model.text import collapse_ws
from core_pdf.impl.output.model import (
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableRowBand,
)
from core_pdf.impl.records import FrozenFields
from core_pdf.impl.runtime.array_views import finite_median

TABLE_MERGE_GAP = 36.0


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


def table_quality(table: Table) -> tuple[int, int, float, int, int]:
    facts = TableFacts.from_rows(table.rows)
    populated = len(facts.filled_texts)
    density = populated / max(1, facts.row_count * facts.columns)
    return (int(2 <= facts.columns <= 16), populated, density, facts.row_count, -facts.columns)


def table_column_bounds(table: Table) -> tuple[tuple[float, float], ...]:
    bounds: list[list[float | None]] = []
    for row in table.rows:
        for cell in row:
            if cell.bbox is None:
                continue
            while len(bounds) <= cell.column:
                bounds.append([None, None])
            left, right = bounds[cell.column]
            bounds[cell.column][0] = cell.bbox[0] if left is None else min(left, cell.bbox[0])
            bounds[cell.column][1] = cell.bbox[2] if right is None else max(right, cell.bbox[2])
    return tuple(
        (float(left), float(right))
        for left, right in bounds
        if left is not None and right is not None
    )


def table_column_alignment(left: Table, right: Table) -> float:
    left_bounds = table_column_bounds(left)
    right_bounds = table_column_bounds(right)
    if len(left_bounds) != len(right_bounds) or not left_bounds:
        return 0.0
    overlaps = []
    for (left_start, left_end), (right_start, right_end) in zip(
        left_bounds, right_bounds, strict=True
    ):
        intersection = interval_overlap(left_start, left_end, right_start, right_end)
        union = max(left_end, right_end) - min(left_start, right_start)
        overlaps.append(intersection / max(1.0, union))
    return fmean(overlaps)


def same_semantic_header(left: tuple[TableCell, ...], right: tuple[TableCell, ...]) -> bool:
    left_text = tuple(cell.text.strip().casefold() for cell in left if cell.text.strip())
    right_text = tuple(cell.text.strip().casefold() for cell in right if cell.text.strip())
    return (
        len(left_text) >= 2
        and left_text == right_text
        and semantic_header_row(left)
        and semantic_header_row(right)
    )


def merge_adjacent_tables(tables: list[Table]) -> list[Table]:
    ordered = sorted(tables, key=lambda table: -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3])
    merged: list[Table] = []
    for table in ordered:
        if not merged:
            merged.append(table)
            continue
        previous = merged[-1]
        previous_bbox = previous.bbox
        table_bbox = table.bbox
        if previous_bbox is None or table_bbox is None:
            merged.append(table)
            continue
        previous_columns = max((len(row) for row in previous.rows), default=0)
        columns = max((len(row) for row in table.rows), default=0)
        vertical_gap = previous_bbox[1] - table_bbox[3]
        if (
            columns != previous_columns
            or not 2 <= columns <= 16
            or horizontal_overlap_ratio(previous_bbox, table_bbox) < 0.6
            or table_column_alignment(previous, table) < 0.55
            or not -5.0 <= vertical_gap <= TABLE_MERGE_GAP
        ):
            merged.append(table)
            continue
        continuation_rows = table.rows
        if previous.rows and table.rows and same_semantic_header(previous.rows[0], table.rows[0]):
            continuation_rows = table.rows[1:]
        combined_rows: list[tuple[TableCell, ...]] = []
        for row in (*previous.rows, *continuation_rows):
            combined_rows.append(
                tuple(
                    TableCell(
                        row=len(combined_rows),
                        column=cell.column,
                        text=cell.text,
                        row_span=cell.row_span,
                        column_span=cell.column_span,
                        bbox=cell.bbox,
                    )
                    for cell in row
                )
            )
        merged[-1] = Table(
            order=previous.order,
            rows=tuple(combined_rows),
            bbox=union_bbox(previous_bbox, table_bbox),
            confidence=min(
                (value for value in (previous.confidence, table.confidence) if value is not None),
                default=1.0,
            ),
            title=previous.title or table.title,
            caption=table.caption or previous.caption,
            metadata=previous.metadata,
        )
    return merged


def semantic_header_row(row: tuple[TableCell, ...]) -> bool:
    populated = [cell for cell in row if cell.text.strip()]
    if not populated:
        return False
    if len(populated) == 1:
        return populated[0].column_span > 1
    numeric = sum(numeric_cell(cell.text) for cell in populated)
    return numeric == 0 and len(populated) >= 2


def split_semantic_table(table: Table) -> tuple[Table, ...]:
    if len(table.rows) < 6 or TableFacts.from_rows(table.rows).numeric_density < 0.3:
        return (table,)
    boundaries = [
        index
        for index, row in enumerate(table.rows[1:], start=1)
        if (
            semantic_header_row(row)
            and index >= 2
            and index + 1 < len(table.rows)
            and any(numeric_cell(cell.text) for cell in table.rows[index + 1])
        )
    ]
    if not boundaries:
        return (table,)
    signatures = {
        tuple(index for index, cell in enumerate(table.rows[index]) if cell.text.strip())
        for index in boundaries
    }
    labels = {
        " ".join(item.text for item in table.rows[index] if item.text.strip()).casefold()
        for index in boundaries
        if any(item.text.strip() for item in table.rows[index])
    }
    if len(table.rows) > 8 and len(boundaries) > 1 and len(signatures) == 1 and len(labels) == 1:
        return (table,)
    starts = [0, *boundaries]
    segments: list[Table] = []
    for segment_index, start in enumerate(starts):
        end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(table.rows)
        rows = table.rows[start:end]
        if len(rows) < 2:
            continue
        boxes = [cell.bbox for row in rows for cell in row if cell.bbox is not None]
        bbox = bbox_union(boxes)
        segments.append(
            Table(
                order=table.order + segment_index,
                rows=rows,
                bbox=bbox,
                confidence=table.confidence,
                title=table.title if segment_index == 0 else None,
                caption=table.caption if end == len(table.rows) else None,
                metadata=table.metadata,
            )
        )
    return tuple(segments) or (table,)


def table_character_spaced_prose(table: Table, *, facts: TableFacts | None = None) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    facts = facts or TableFacts.from_rows(table.rows)
    if facts.columns < 8:
        return False
    filled_texts = facts.filled_texts
    if not filled_texts:
        return False
    return (
        facts.numeric_density < 0.12 and facts.character_spaced_cells / len(filled_texts) >= 0.60
    ) or (
        len(table.rows) >= 40 and facts.average_cell_length < 8.0 and facts.numeric_density < 0.20
    )


def table_is_single_column_prose(table: Table, *, facts: TableFacts | None = None) -> bool:
    facts = facts or TableFacts.from_rows(table.rows)
    if facts.nonempty_rows < 3:
        return False
    if facts.spanned_columns < 2:
        return False
    return facts.single_cell_rows * 2 > facts.nonempty_rows


STREAM_PROSE_LONG_CELL_CHARACTERS = 25
STREAM_PROSE_LONG_CELL_RATIO = 0.6
STREAM_PROSE_NUMERIC_CELL_RATIO = 0.15
STREAM_WORD_GRID_MIN_COLUMNS = 8
STREAM_WORD_GRID_MIN_ROWS = 12
STREAM_WORD_GRID_NUMERIC_RATIO = 0.2
STREAM_WORD_GRID_MEDIAN_CELL_CHARACTERS = 14
STREAM_SPARSE_PROSE_MAX_DENSITY = 0.68
STREAM_SPARSE_PROSE_LONG_RATIO = 0.25
STREAM_SPARSE_PROSE_MAX_COLUMNS = 6


def stream_table_reads_like_prose(table: Table) -> bool:
    facts = TableFacts.from_rows(table.rows)
    filled = facts.filled_texts
    if not filled:
        return True
    long_cells = sum(length > STREAM_PROSE_LONG_CELL_CHARACTERS for length in facts.text_lengths)
    numeric_cells = sum(
        1
        for text in filled
        if len(text) <= STREAM_PROSE_LONG_CELL_CHARACTERS
        and any(character.isdigit() for character in text)
    )
    if (
        long_cells >= len(filled) * STREAM_PROSE_LONG_CELL_RATIO
        and numeric_cells < len(filled) * STREAM_PROSE_NUMERIC_CELL_RATIO
    ):
        return True
    total_cells = facts.cell_count
    narrow = facts.columns <= STREAM_SPARSE_PROSE_MAX_COLUMNS
    if (
        total_cells
        and narrow
        and len(filled) < total_cells * STREAM_SPARSE_PROSE_MAX_DENSITY
        and long_cells >= len(filled) * STREAM_SPARSE_PROSE_LONG_RATIO
    ):
        return True
    if (
        facts.columns >= STREAM_WORD_GRID_MIN_COLUMNS
        and facts.populated_rows >= STREAM_WORD_GRID_MIN_ROWS
        and numeric_cells < len(filled) * STREAM_WORD_GRID_NUMERIC_RATIO
    ):
        lengths = sorted(facts.text_lengths)
        median_length = lengths[len(lengths) // 2]
        if median_length <= STREAM_WORD_GRID_MEDIAN_CELL_CHARACTERS:
            return True
    return False


def merge_stream_text_columns(table: Table) -> Table:
    columns = max((len(row) for row in table.rows), default=0)
    if (
        table.metadata.get("source") != "stream"
        or columns < 6
        or columns % 2
        or len(table.rows) < 4
        or TableFacts.from_rows(table.rows).numeric_density >= 0.25
    ):
        return table
    group_size = columns // 2
    merged_rows: list[tuple[TableCell, ...]] = []
    for row_index, row in enumerate(table.rows):
        merged: list[TableCell] = []
        for group in range(2):
            cells = row[group * group_size : (group + 1) * group_size]
            text = " ".join(cell.text for cell in cells if cell.text).strip()
            boxes = [cell.bbox for cell in cells if cell.bbox is not None]
            if not text and row_index == 0:
                continue
            bbox = bbox_union(boxes)
            merged.append(
                TableCell(
                    row=row_index,
                    column=len(merged),
                    text=text,
                    column_span=group_size if row_index == 0 and len(merged) == 0 else 1,
                    bbox=bbox,
                )
            )
        if merged:
            merged_rows.append(tuple(merged))
    return replace(table, rows=tuple(merged_rows), metadata={**table.metadata, "merged": True})


LOGICAL_ROW_GAP_RATIO = 0.10
LOGICAL_ROW_MIN_GAP = 0.5
LOGICAL_ROW_MIN_ROWS = 4
LOGICAL_ROW_MAX_NUMERIC_RATIO = 0.40
LOGICAL_ROW_MIN_TALL_RATIO = 3.0
LOGICAL_ROW_MIN_COLUMNS = 5


def merge_wrapped_cell_rows(table: Table) -> Table:
    if table.metadata.get("source") != "stream" or len(table.rows) < LOGICAL_ROW_MIN_ROWS:
        return table
    filled = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if filled:
        numeric = sum(
            1
            for text in filled
            if len(text) <= STREAM_PROSE_LONG_CELL_CHARACTERS
            and any(character.isdigit() for character in text)
        )
        if numeric >= len(filled) * LOGICAL_ROW_MAX_NUMERIC_RATIO:
            return table
    if max((len(row) for row in table.rows), default=0) < LOGICAL_ROW_MIN_COLUMNS:
        return table
    extents: list[tuple[float, float]] = []
    heights: list[float] = []
    for row in table.rows:
        boxes = [cell.bbox for cell in row if cell.bbox is not None]
        if not boxes:
            return table
        extents.append((max(box[3] for box in boxes), min(box[1] for box in boxes)))
        heights.extend(box[3] - box[1] for box in boxes)
    median_height = finite_median(numpy.asarray(heights, dtype=numpy.float32))
    if max(heights) < median_height * LOGICAL_ROW_MIN_TALL_RATIO:
        return table
    tolerance = max(
        LOGICAL_ROW_MIN_GAP,
        median_height * LOGICAL_ROW_GAP_RATIO,
    )
    groups: list[list[int]] = []
    running_bottom = 0.0
    for index, (top, bottom) in enumerate(extents):
        if groups and top >= running_bottom - tolerance:
            groups[-1].append(index)
            running_bottom = min(running_bottom, bottom)
        else:
            groups.append([index])
            running_bottom = bottom
    if len(groups) == len(table.rows) or len(groups) < 2:
        return table
    columns = max((len(row) for row in table.rows), default=0)
    merged_rows: list[tuple[TableCell, ...]] = []
    for row_index, group in enumerate(groups):
        cells: list[TableCell] = []
        for column in range(columns):
            parts: list[str] = []
            cell_boxes: list[tuple[float, float, float, float]] = []
            span = 1
            for index in group:
                row = table.rows[index]
                if column >= len(row):
                    continue
                cell = row[column]
                if cell.text.strip():
                    parts.append(cell.text.strip())
                if cell.bbox is not None:
                    cell_boxes.append(cell.bbox)
                span = max(span, cell.column_span)
            cells.append(
                TableCell(
                    row=row_index,
                    column=column,
                    text=" ".join(parts),
                    column_span=span,
                    bbox=bbox_union(cell_boxes),
                )
            )
        merged_rows.append(tuple(cells))
    return replace(
        table,
        rows=tuple(merged_rows),
        metadata={**table.metadata, "logical_rows": True},
    )


def merge_wrapped_stream_rows(table: Table) -> Table:
    if (
        table.metadata.get("source") != "stream"
        or len(table.rows) < 8
        or max((len(row) for row in table.rows), default=0) < 5
        or table.metadata.get("numeric_cells", 0) > 2
    ):
        return table
    merged: list[list[TableCell]] = []
    for row in table.rows:
        cells = list(row)
        if merged and cells and not cells[0].text.strip():
            previous = merged[-1]
            for index, cell in enumerate(cells):
                if index >= len(previous) or not cell.text.strip():
                    continue
                target = previous[index]
                boxes = [box for box in (target.bbox, cell.bbox) if box is not None]
                previous[index] = replace(
                    target,
                    text=" ".join(part for part in (target.text, cell.text) if part).strip(),
                    bbox=bbox_union(boxes),
                )
            continue
        merged.append(cells)
    if len(merged) == len(table.rows):
        return table
    return replace(
        table,
        rows=tuple(
            tuple(replace(cell, row=index) for cell in row) for index, row in enumerate(merged)
        ),
    )


def annotate_table_associations(
    table: Table,
    observations: ObservationBatch,
    text_rows: list[list[int]],
) -> Table:
    title = table.title
    caption = table.caption
    if len(table.rows) >= 2:
        first = tuple(cell for cell in table.rows[0] if cell.text.strip())
        second = tuple(cell for cell in table.rows[1] if cell.text.strip())
        if len(first) == 1 and len(second) >= 2:
            cell = first[0]
            kind = "caption" if ":" in cell.text else "title"
            associated = TableAssociatedText(cell.text, cell.bbox, kind=kind)
            if kind == "caption":
                caption = associated
            else:
                title = associated
    if table.bbox is None or title is not None:
        return replace(table, title=title, caption=caption)
    x0, _y0, x1, y1 = table.bbox
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for row in text_rows:
        boxes = observations.bbox[list(row)]
        row_x0 = float(boxes[:, 0].min())
        row_x1 = float(boxes[:, 2].max())
        row_y0 = float(boxes[:, 1].min())
        overlap = interval_overlap(x0, x1, row_x0, row_x1)
        if overlap / max(1.0, min(x1 - x0, row_x1 - row_x0)) < 0.60:
            continue
        gap = row_y0 - y1
        if 0.0 <= gap <= 36.0:
            candidates.append((gap, tuple(row)))
    if candidates:
        _gap, title_row = min(candidates, key=lambda item: item[0])
        text = cell_text(observations, list(title_row))
        if text:
            title_boxes = observations.bbox[list(title_row)]
            title = TableAssociatedText(
                text,
                (
                    float(title_boxes[:, 0].min()),
                    float(title_boxes[:, 1].min()),
                    float(title_boxes[:, 2].max()),
                    float(title_boxes[:, 3].max()),
                ),
                kind="title",
            )
    if title is table.title and caption is table.caption:
        return table
    return replace(table, title=title, caption=caption)


def table_with_bands(table: Table) -> Table:
    associated = {
        text.kind: text.text.casefold() for text in (table.title, table.caption) if text is not None
    }
    first_grid = next(
        (
            index
            for index, row in enumerate(table.rows)
            if any(cell.text.strip() for cell in row)
            and not any(
                value in associated.values()
                for value in (cell.text.strip().casefold() for cell in row if cell.text.strip())
            )
        ),
        None,
    )
    row_bands: list[TableRowBand] = []
    for index, row in enumerate(table.rows):
        boxes = [cell.bbox for cell in row if cell.bbox is not None]
        texts = tuple(cell.text.strip().casefold() for cell in row if cell.text.strip())
        kind = "blank"
        if texts:
            if any(value in associated.values() for value in texts):
                kind = "title" if texts[0] == associated.get("title") else "caption"
            elif first_grid is not None and index == first_grid and len(texts) >= 2:
                kind = "header"
            else:
                kind = "body"
        row_bands.append(TableRowBand(index=index, bbox=bbox_union(boxes), kind=kind))

    column_count = max(
        (cell.column + cell.column_span for row in table.rows for cell in row),
        default=0,
    )
    column_boxes: list[list[tuple[float, float, float, float]]] = [[] for _ in range(column_count)]
    for row in table.rows:
        for cell in row:
            if cell.bbox is None:
                continue
            for index in range(
                max(cell.column, 0),
                min(cell.column + cell.column_span, column_count),
            ):
                column_boxes[index].append(cell.bbox)
    column_bands = tuple(
        TableColumnBand(index=index, bbox=bbox_union(boxes))
        for index, boxes in enumerate(column_boxes)
    )
    return replace(table, row_bands=tuple(row_bands), column_bands=column_bands)


frozen_setattr = object.__setattr__


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


class TableFacts(FrozenFields):
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"row_count={self.row_count!r}, "
            f"nonempty_rows={self.nonempty_rows!r}, "
            f"populated_rows={self.populated_rows!r}, "
            f"columns={self.columns!r}, "
            f"spanned_columns={self.spanned_columns!r}, "
            f"cell_count={self.cell_count!r}, "
            f"single_cell_rows={self.single_cell_rows!r}, "
            f"filled_texts={self.filled_texts!r}"
            ")"
        )

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

    def __replace__(self, /, **changes: Any) -> Self:
        row_count = changes.pop("row_count", self.row_count)
        nonempty_rows = changes.pop("nonempty_rows", self.nonempty_rows)
        populated_rows = changes.pop("populated_rows", self.populated_rows)
        columns = changes.pop("columns", self.columns)
        spanned_columns = changes.pop("spanned_columns", self.spanned_columns)
        cell_count = changes.pop("cell_count", self.cell_count)
        single_cell_rows = changes.pop("single_cell_rows", self.single_cell_rows)
        filled_texts = changes.pop("filled_texts", self.filled_texts)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            row_count,
            nonempty_rows,
            populated_rows,
            columns,
            spanned_columns,
            cell_count,
            single_cell_rows,
            filled_texts,
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
