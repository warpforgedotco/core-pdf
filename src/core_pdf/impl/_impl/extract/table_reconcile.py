# SPDX-License-Identifier: AGPL-3.0-only
"""Reconcile overlapping text and table projections."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace

from core_pdf.impl._impl.extract.table_cleanup import internal_table_with_bands
from core_pdf.impl._impl.model.geometry import bbox_union, overlap_ratio_min, overlap_ratio_of
from core_pdf.impl._impl.model.spatial import SpatialFrame
from core_pdf.impl._impl.model.text import complete_text_covered, content_tokens
from core_pdf.impl._impl.output.model import Block, Table
from core_pdf.impl.types import Rectangle


def internal_remove_block_duplicate_table_rows(
    blocks: list[Block],
    tables: tuple[Table, ...],
) -> tuple[Table, ...]:
    """Drop rows reproduced completely by a surviving line in the same region."""
    if not blocks or not tables:
        return tables
    line_boxes_by_text: dict[str, list[Rectangle]] = {}
    for block in blocks:
        for line in block.lines:
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            text = " ".join(line.text.split())
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
            text = " ".join(" ".join(cell.text.split()) for cell in cells)
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
            # The blocks cover the rows, but need not include associated text.
            # Keep its table until that text can be projected independently.
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
            # A removed row must not shift another row's semantic role. Rebuild
            # column bounds, but retain the surviving row classifications.
            projected = replace(
                projected,
                row_bands=tuple(
                    replace(table.row_bands[old_index], index=index)
                    for index, old_index in enumerate(kept_row_indexes)
                ),
            )
        filtered.append(projected)
    return tuple(filtered)


def internal_line_duplicates_table(text: str, box: Rectangle, table: Table) -> bool:
    """Require complete text in the same cell region before dropping a line."""
    normalized = " ".join(text.split())
    if not normalized or table.bbox is None or overlap_ratio_of(box, table.bbox) < 0.90:
        return False
    tokens = content_tokens(normalized)
    participating_cells = []
    total_cells = 0
    for row in table.rows:
        row_cells = [cell for cell in row if cell.text]
        total_cells += len(row_cells)
        cells = [
            cell for cell in row_cells if cell.bbox is None or overlap_ratio_min(box, cell.bbox) > 0
        ]
        if not cells:
            continue
        participating_cells.extend(cells)
        cell_texts = [" ".join(cell.text.split()) for cell in cells]
        for cell, cell_text in zip(cells, cell_texts, strict=True):
            if (
                cell.bbox is not None
                and overlap_ratio_of(box, cell.bbox) >= 0.90
                and complete_text_covered(tokens, content_tokens(cell_text))
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
                # Inferred column boundaries can graze the line's last glyph.
                # Test exact cell sequences instead of forcing neighboring
                # text into a match because of a small geometric intersection.
                if all(cell.bbox is not None for cell in matched_cells):
                    row_box = bbox_union(
                        cell.bbox for cell in matched_cells if cell.bbox is not None
                    )
                    if row_box is not None and overlap_ratio_of(box, row_box) >= 0.90:
                        return True
                elif start == 0 and end + 1 == len(cells) == len(row_cells):
                    # Without cell geometry, only the complete row can be
                    # compared using the containing table's known bounds.
                    return True
    # A single text line can represent an entire table in a competing layout.
    # Keep case, punctuation, order, and repeated values in this comparison.
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
        and normalized == " ".join(" ".join(cell.text.split()) for cell in participating_cells)
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
    deduplicated: list[Block] = []
    for block in blocks:
        kept_lines = []
        for line in block.lines:
            # A tall block can include a heading well outside the table. Only
            # its individual lines establish which text actually repeats cells.
            box = line.bbox or (block.bbox if len(block.lines) == 1 else None)
            if box is None or not any(
                internal_line_duplicates_table(line.text, box, located_tables[int(index)])
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
    """Resolve overlap once, producing explicit text and table projections."""
    text_blocks = internal_remove_table_duplicate_blocks(blocks, parsed_tables)
    projected_tables = internal_remove_block_duplicate_table_rows(text_blocks, parsed_tables)
    return text_blocks, projected_tables
