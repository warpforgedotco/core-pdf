# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from bisect import bisect_left
from copy import replace
from typing import Any, ClassVar, NoReturn, Self

import numpy

from core_pdf.impl._impl.extract.table_cleanup import internal_table_with_bands
from core_pdf.impl._impl.model.geometry import (
    bbox_area,
    bbox_union,
    overlap_ratio_min,
    overlap_ratio_of,
)
from core_pdf.impl._impl.model.spatial import SpatialFrame
from core_pdf.impl._impl.model.text import collapse_ws, complete_text_covered, content_tokens
from core_pdf.impl._impl.output.model import Block, Table, TableCell
from core_pdf.impl.types import Rectangle

internal_frozen_setattr = object.__setattr__


def internal_remove_block_duplicate_table_rows(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    if not blocks or not tables:
        return tables
    line_boxes_by_text: dict[str, list[Rectangle]] = {}
    for block in blocks:
        for line in block.lines:
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            text = collapse_ws(line.text)
            if box is not None and text:
                line_boxes_by_text.setdefault(text, []).append(box)
    filtered: list[Table] = []
    for table in tables:
        if not table.rows:
            filtered.append(table)
            continue
        kept_row_indexes: list[int] = []
        for row_index, row in enumerate(table.rows):
            cells = [cell for cell in row if cell.text]
            text = " ".join(collapse_ws(cell.text) for cell in cells)
            cell_boxes = tuple(cell.bbox for cell in cells if cell.bbox is not None)
            row_box = bbox_union(cell_boxes)
            if not text or row_box is None or len(cell_boxes) != len(cells):
                kept_row_indexes.append(row_index)
                continue
            duplicated = any(
                overlap_ratio_of(line_box, row_box) >= 0.90
                and all(overlap_ratio_min(line_box, cell_box) > 0 for cell_box in cell_boxes)
                for line_box in line_boxes_by_text.get(text, ())
            )
            if not duplicated:
                kept_row_indexes.append(row_index)
        if len(kept_row_indexes) == len(table.rows):
            filtered.append(table)
            continue
        if not kept_row_indexes:
            if table.title is not None or table.caption is not None:
                filtered.append(table)
            continue
        rows = tuple(
            tuple(
                replace(
                    cell,
                    row=index,
                    row_span=bisect_left(kept_row_indexes, old_index + cell.row_span) - index,
                )
                for cell in table.rows[old_index]
            )
            for index, old_index in enumerate(kept_row_indexes)
        )
        projected = internal_table_with_bands(replace(table, rows=rows))
        if table.row_bands:
            projected = replace(
                projected,
                row_bands=tuple(
                    replace(table.row_bands[old_index], index=index)
                    for index, old_index in enumerate(kept_row_indexes)
                ),
            )
        filtered.append(projected)
    return tuple(filtered)


class internal_IndexedRow:
    __slots__ = ("cells", "texts", "tokens", "frame_indexes")

    cells: tuple[TableCell, ...]
    texts: tuple[str, ...]
    tokens: tuple[tuple[str, ...], ...]
    frame_indexes: tuple[int, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("cells", "texts", "tokens", "frame_indexes")
    __match_args__ = ("cells", "texts", "tokens", "frame_indexes")

    def __init__(
        self,
        cells: tuple[TableCell, ...],
        texts: tuple[str, ...],
        tokens: tuple[tuple[str, ...], ...],
        frame_indexes: tuple[int, ...],
    ) -> None:
        internal_frozen_setattr(self, "cells", cells)
        internal_frozen_setattr(self, "texts", texts)
        internal_frozen_setattr(self, "tokens", tokens)
        internal_frozen_setattr(self, "frame_indexes", frame_indexes)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"cells={self.cells!r}, "
            f"texts={self.texts!r}, "
            f"tokens={self.tokens!r}, "
            f"frame_indexes={self.frame_indexes!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.cells == other.cells
            and self.texts == other.texts
            and self.tokens == other.tokens
            and self.frame_indexes == other.frame_indexes
        )

    def __hash__(self) -> int:
        return hash((self.cells, self.texts, self.tokens, self.frame_indexes))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        cells = changes.pop("cells", self.cells)
        texts = changes.pop("texts", self.texts)
        tokens = changes.pop("tokens", self.tokens)
        frame_indexes = changes.pop("frame_indexes", self.frame_indexes)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(cells, texts, tokens, frame_indexes)


class internal_TableIndex:
    __slots__ = ("table", "rows", "frame")

    table: Table
    rows: tuple[internal_IndexedRow, ...]
    frame: SpatialFrame | None

    __fields__: ClassVar[tuple[str, ...]] = ("table", "rows", "frame")
    __match_args__ = ("table", "rows", "frame")

    def __init__(
        self,
        table: Table,
        rows: tuple[internal_IndexedRow, ...],
        frame: SpatialFrame | None,
    ) -> None:
        internal_frozen_setattr(self, "table", table)
        internal_frozen_setattr(self, "rows", rows)
        internal_frozen_setattr(self, "frame", frame)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"table={self.table!r}, "
            f"rows={self.rows!r}, "
            f"frame={self.frame!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.table == other.table and self.rows == other.rows and self.frame == other.frame

    def __hash__(self) -> int:
        return hash((self.table, self.rows, self.frame))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        table = changes.pop("table", self.table)
        rows = changes.pop("rows", self.rows)
        frame = changes.pop("frame", self.frame)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(table, rows, frame)

    @classmethod
    def build(cls, table: Table) -> internal_TableIndex:
        rows: list[internal_IndexedRow] = []
        boxes: list[Rectangle] = []
        for row in table.rows:
            cells = tuple(cell for cell in row if cell.text)
            texts = tuple(collapse_ws(cell.text) for cell in cells)
            indexes: list[int] = []
            for cell in cells:
                if cell.bbox is None:
                    indexes.append(-1)
                else:
                    indexes.append(len(boxes))
                    boxes.append(cell.bbox)
            rows.append(
                internal_IndexedRow(
                    cells,
                    texts,
                    tuple(content_tokens(text) for text in texts),
                    tuple(indexes),
                )
            )
        frame = SpatialFrame.from_boxes(boxes) if boxes else None
        return cls(table, tuple(rows), frame)


def internal_line_duplicates_table(text: str, box: Rectangle, index: internal_TableIndex) -> bool:
    table = index.table
    normalized = collapse_ws(text)
    if not normalized or table.bbox is None or overlap_ratio_of(box, table.bbox) < 0.90:
        return False
    tokens = content_tokens(normalized)
    if index.frame is None:
        touches = covered = numpy.zeros(0, dtype=bool)
    else:
        intersections = index.frame.intersection_areas(box)
        touches = intersections > 0.0
        box_area = bbox_area(box)
        covered = (
            intersections / box_area >= 0.90
            if box_area > 0.0
            else numpy.zeros(len(intersections), dtype=bool)
        )
    participating_cells: list[TableCell] = []
    total_cells = 0
    for row in index.rows:
        total_cells += len(row.cells)
        kept = [
            position
            for position, frame_index in enumerate(row.frame_indexes)
            if frame_index < 0 or touches[frame_index]
        ]
        if not kept:
            continue
        cells = [row.cells[position] for position in kept]
        participating_cells.extend(cells)
        cell_texts = [row.texts[position] for position in kept]
        for position in kept:
            frame_index = row.frame_indexes[position]
            if (
                frame_index >= 0
                and covered[frame_index]
                and complete_text_covered(tokens, row.tokens[position])
            ):
                return True
        for start in range(len(cells)):
            candidate = ""
            for end in range(start, len(cells)):
                candidate = f"{candidate} {cell_texts[end]}" if candidate else cell_texts[end]
                if not normalized.startswith(candidate):
                    break
                if normalized != candidate:
                    continue
                matched_cells = cells[start : end + 1]
                if all(cell.bbox is not None for cell in matched_cells):
                    row_box = bbox_union(
                        cell.bbox for cell in matched_cells if cell.bbox is not None
                    )
                    if row_box is not None and overlap_ratio_of(box, row_box) >= 0.90:
                        return True
                elif start == 0 and end + 1 == len(cells) == len(row.cells):
                    return True
    known_geometry = all(cell.bbox is not None for cell in participating_cells)
    if not known_geometry and len(participating_cells) != total_cells:
        return False
    covered_box = (
        bbox_union(cell.bbox for cell in participating_cells if cell.bbox is not None)
        if participating_cells and known_geometry
        else table.bbox
    )
    return (
        covered_box is not None
        and overlap_ratio_of(box, covered_box) >= 0.90
        and normalized == " ".join(collapse_ws(cell.text) for cell in participating_cells)
    )


def internal_remove_table_duplicate_blocks(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> list[Block]:
    if not blocks or not tables:
        return blocks
    located_tables = [table for table in tables if table.bbox is not None]
    if not located_tables:
        return blocks
    table_frame = SpatialFrame.from_boxes(
        table.bbox for table in located_tables if table.bbox is not None
    )
    indexes = [internal_TableIndex.build(table) for table in located_tables]
    deduplicated: list[Block] = []
    for block in blocks:
        kept_lines = []
        for line in block.lines:
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            if box is None or not any(
                internal_line_duplicates_table(line.text, box, indexes[int(index)])
                for index in table_frame.matching_overlap_min(box, 0.90)
            ):
                kept_lines.append(line)
        if len(kept_lines) == len(block.lines):
            deduplicated.append(block)
        elif kept_lines:
            box = (
                bbox_union(line.bbox for line in kept_lines if line.bbox is not None)
                if all(line.bbox is not None for line in kept_lines)
                else block.bbox
            )
            deduplicated.append(replace(block, lines=tuple(kept_lines), bbox=box))
    return deduplicated


def internal_project_text_and_tables(
    blocks: list[Block],
    parsed_tables: tuple[Table, ...],
) -> tuple[list[Block], tuple[Table, ...]]:
    text_blocks = internal_remove_table_duplicate_blocks(blocks, parsed_tables)
    projected_tables = internal_remove_block_duplicate_table_rows(text_blocks, parsed_tables)
    return text_blocks, projected_tables
