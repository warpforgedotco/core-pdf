# SPDX-License-Identifier: AGPL-3.0-only
"""Table normalization, filtering, merging, and output annotation."""

from __future__ import annotations

import re
from dataclasses import replace
from statistics import fmean

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.model.geometry import (
    bbox_union,
    horizontal_overlap_ratio,
    interval_overlap,
    union_bbox,
)
from core_pdf.impl.output import (
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableRowBand,
)
from core_pdf.impl.runtime.array_views import finite_median
from core_pdf.impl.text import (
    collapse_character_spaced,
    collapse_ws,
    is_leader_run,
    strip_edge_leaders,
)

TABLE_MERGE_GAP = 36.0  # further increased to allow modestly wider table merges (conservative)


def internal_cell_text(
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
        if part and not is_leader_run(part):
            parts.append(part)
    return " ".join(parts)


def internal_clean_table_cell_leader_runs(text: str) -> str:
    """Drop leader/fill punctuation runs from a table cell.

    Dot and dash leaders are page furniture (ToC fillers, reference-list
    separators, dashed cell rules) that reference text omits.  A cell made up
    entirely of such characters, or a cell ending in a long run of them, is
    stripped so the cell matches the reference reading order.
    """
    if not text:
        return text
    if is_leader_run(text):
        return ""
    if all(ch in "\u25cf\u25e6" for ch in text if not ch.isspace()):
        return ""
    return collapse_ws(strip_edge_leaders(text))


internal_TABLE_SPACED_DIGIT_SEQUENCE_RE = re.compile(r"[\d/.,]+(?: +[\d/.,]+)+")
internal_TABLE_SPACED_DIGIT_ADJACENCY_RE = re.compile(r"\d +\d")


def internal_repair_table_cell_spaced_digits(text: str) -> str:
    """Rejoin letter-spaced numeric/date runs inside a table cell.

    Tracked (letter-spaced) digits split a value such as ``10/19/21`` into
    ``10 /1 9`` and the split ``1 9`` no longer matches the reference
    ``19``.  Rejoin any space-separated run of digit/slash tokens when a
    space separates two digits, which is the tracking signature.  Plain
    ratios such as ``40 / 20`` rejoin into ``40/20`` without changing the
    token multiset because the slash keeps the digit groups apart.
    """
    if not text:
        return text
    if ":" in text or "," in text:
        return text

    def rejoin(match: re.Match[str]) -> str:
        sequence = match.group(0)
        if "/" not in sequence:
            return sequence
        if sum(ch.isdigit() for ch in sequence) < 3:
            return sequence
        if not internal_TABLE_SPACED_DIGIT_ADJACENCY_RE.search(sequence):
            return sequence
        joined = re.sub(r" +", "", sequence)
        groups = [group for group in re.split(r"[/.]", joined) if group]
        if len(groups) < 2 or any(len(group) > 4 for group in groups):
            return sequence
        return joined

    return internal_TABLE_SPACED_DIGIT_SEQUENCE_RE.sub(rejoin, text)


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


def internal_collapse_character_spaced_cell(text: str) -> str:
    """Collapse glyph-separated prose captured as a table cell."""
    return collapse_character_spaced(text, min_tokens=8, single_char_ratio=0.80)


def internal_table_quality(table: Table) -> tuple[int, int, float, int, int]:
    rows = len(table.rows)
    columns = max((len(row) for row in table.rows), default=0)
    populated = sum(bool(cell.text.strip()) for row in table.rows for cell in row)
    density = populated / max(1, rows * columns)
    # Allow more columns to be considered valid for table quality (up to 16)
    return (int(2 <= columns <= 16), populated, density, rows, -columns)


def internal_table_column_bounds(table: Table) -> tuple[tuple[float, float], ...]:
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


def internal_table_column_alignment(left: Table, right: Table) -> float:
    left_bounds = internal_table_column_bounds(left)
    right_bounds = internal_table_column_bounds(right)
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


def internal_same_semantic_header(
    left: tuple[TableCell, ...], right: tuple[TableCell, ...]
) -> bool:
    left_text = tuple(cell.text.strip().casefold() for cell in left if cell.text.strip())
    right_text = tuple(cell.text.strip().casefold() for cell in right if cell.text.strip())
    return (
        len(left_text) >= 2
        and left_text == right_text
        and internal_semantic_header_row(left)
        and internal_semantic_header_row(right)
    )


def internal_merge_adjacent_tables(tables: list[Table]) -> list[Table]:
    ordered = sorted(tables, key=lambda table: -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3])
    merged: list[Table] = []
    for table in ordered:
        if not merged or table.bbox is None or merged[-1].bbox is None:
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
        # Relax adjacent-table merge conditions slightly to allow merging
        # of tables with minor horizontal overlap or slightly differing column
        # counts. This reduces false splits where a single logical table is
        # broken into two adjacent segments.
        if (
            columns != previous_columns
            or not 2 <= columns <= 16
            or horizontal_overlap_ratio(previous_bbox, table_bbox) < 0.6
            or internal_table_column_alignment(previous, table) < 0.55
            or not -5.0 <= vertical_gap <= TABLE_MERGE_GAP
        ):
            merged.append(table)
            continue
        continuation_rows = table.rows
        if (
            previous.rows
            and table.rows
            and internal_same_semantic_header(previous.rows[0], table.rows[0])
        ):
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
            confidence=min(previous.confidence or 1.0, table.confidence or 1.0),
            title=previous.title or table.title,
            caption=table.caption or previous.caption,
            metadata=previous.metadata,
        )
    return merged


def internal_semantic_header_row(row: tuple[TableCell, ...]) -> bool:
    populated = [cell for cell in row if cell.text.strip()]
    if not populated:
        return False
    if len(populated) == 1:
        return populated[0].column_span > 1
    numeric = sum(internal_numeric_cell(cell.text) for cell in populated)
    return numeric == 0 and len(populated) >= 2


def internal_numeric_density(table: Table) -> float:
    cells = [cell for row in table.rows for cell in row if cell.text.strip()]
    return sum(internal_numeric_cell(cell.text) for cell in cells) / max(1, len(cells))


def internal_structured_stream_table(table: Table) -> bool:
    """Identify stream tables with enough structured values to preserve."""
    if table.metadata.get("source") != "stream":
        return False
    numeric_cells = table.metadata.get("numeric_cells", 0)
    return internal_numeric_density(table) >= 0.10 or (
        isinstance(numeric_cells, int) and numeric_cells >= 2
    )


def internal_split_semantic_table(table: Table) -> tuple[Table, ...]:
    """Split long grid regions at repeated section-header rows."""
    if len(table.rows) < 6 or internal_numeric_density(table) < 0.3:
        return (table,)
    boundaries = [
        index
        for index, row in enumerate(table.rows[1:], start=1)
        if (
            internal_semantic_header_row(row)
            and index >= 2
            and index + 1 < len(table.rows)
            and any(internal_numeric_cell(cell.text) for cell in table.rows[index + 1])
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


def internal_table_character_spaced_prose(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    columns = max((len(row) for row in table.rows), default=0)
    # Raise the column threshold so only wider multi-column prose is filtered.
    if columns < 8:
        return False
    filled_texts = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled_texts:
        return False
    numeric_cells = sum(internal_numeric_cell(text) for text in filled_texts)
    character_spaced_cells = sum(internal_character_spaced_cell(text) for text in filled_texts)
    average_cell_length = sum(len(text) for text in filled_texts) / len(filled_texts)
    # Tighten the character-spaced fraction to 0.6 to avoid filtering legitimate
    # tables that have some character-spaced cells.
    return (
        numeric_cells / len(filled_texts) < 0.12
        and character_spaced_cells / len(filled_texts) >= 0.60
    ) or (
        len(table.rows) >= 40
        and average_cell_length < 8.0
        and numeric_cells / len(filled_texts) < 0.20
    )


def internal_table_has_grid_shape(table: Table) -> bool:
    """Report whether a table's rows actually use its columns.

    This is the positive form of :func:`internal_table_is_single_column_prose`.
    A grid inferred from alignment or from rules can describe either a real
    table or a column of prose, and what separates them is whether the rows
    divide: a table's rows hold several cells because there are several
    columns to fill, while prose yields one cell per row spanning the width.

    Deciding this on shape alone keeps it free of assumptions about what a
    table contains, so it holds for a schedule of numbers and a grid of
    sentences alike.
    """
    rows = [row for row in table.rows if row]
    if len(rows) < 2:
        return False
    columns = max((cell.column + cell.column_span for row in rows for cell in row), default=0)
    if columns < 2:
        return False
    divided_rows = sum(1 for row in rows if len(row) >= 2)
    return divided_rows * 2 > len(rows)


def internal_table_is_single_column_prose(table: Table) -> bool:
    """Report a detected grid that is really a column of flowing text.

    A grid inferred from whitespace alignment can latch onto ordinary prose:
    paragraph lines all start at the same margin, so a column boundary appears
    to run down the page. What gives it away is shape rather than content --
    the rows hold one cell each, spanning most of the inferred width, because
    there was never a second column to divide them.

    Judging this on shape keeps it free of assumptions about what a table
    contains; a genuine table has rows that actually use its columns. It
    applies whichever way the grid was inferred: rules drawn on a page are as
    happy to be underlines and dividers as they are to be a table border.
    """
    rows = [row for row in table.rows if row]
    if len(rows) < 3:
        return False
    columns = max((cell.column + cell.column_span for row in rows for cell in row), default=0)
    if columns < 2:
        return False
    single_cell_rows = sum(1 for row in rows if len(row) == 1)
    return single_cell_rows * 2 > len(rows)


internal_STREAM_PROSE_LONG_CELL_CHARACTERS = 25
internal_STREAM_PROSE_LONG_CELL_RATIO = 0.6
internal_STREAM_PROSE_NUMERIC_CELL_RATIO = 0.15
internal_STREAM_WORD_GRID_MIN_COLUMNS = 8
internal_STREAM_WORD_GRID_MIN_ROWS = 12
internal_STREAM_WORD_GRID_NUMERIC_RATIO = 0.2
internal_STREAM_WORD_GRID_MEDIAN_CELL_CHARACTERS = 14
internal_STREAM_SPARSE_PROSE_MAX_DENSITY = 0.68
internal_STREAM_SPARSE_PROSE_LONG_RATIO = 0.25
internal_STREAM_SPARSE_PROSE_MAX_COLUMNS = 6


def internal_stream_table_reads_like_prose(table: Table) -> bool:
    """Report a stream table whose cells are sentences rather than values.

    Whitespace alignment finds parallel text columns -- two-column papers,
    side-by-side lists, label/description pairs -- as readily as it finds
    tables. Rendering those row-major interleaves the columns and destroys
    the reading order, which costs far more than the table was worth. Real
    borderless tables carry short data cells and a numeric backbone; a
    candidate dominated by long cells with almost no short numeric ones is
    flowing text and should stay in the normal layout.

    Judged after wrapped-row and text-column merging, because the raw
    detection holds word-level fragments whose lengths say nothing about
    the prose that emerges once the cells are assembled.
    """
    filled = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled:
        return True
    long_cells = sum(1 for text in filled if len(text) > internal_STREAM_PROSE_LONG_CELL_CHARACTERS)
    numeric_cells = sum(
        1
        for text in filled
        if len(text) <= internal_STREAM_PROSE_LONG_CELL_CHARACTERS
        and any(character.isdigit() for character in text)
    )
    if (
        long_cells >= len(filled) * internal_STREAM_PROSE_LONG_CELL_RATIO
        and numeric_cells < len(filled) * internal_STREAM_PROSE_NUMERIC_CELL_RATIO
    ):
        return True
    # Side-by-side lists picked up as one table are sparse -- each row only
    # populates the columns its list reaches -- and their cells run long,
    # because list entries are phrases. A genuine table with empty cells
    # (a financial grid) stays short-celled, a dense table with long cells
    # (definition tables) stays dense, and a wide word matrix (roadmaps)
    # carries more columns than side-by-side lists ever produce, so
    # requiring all three signals rejects the parallel lists alone.
    total_cells = sum(len(row) for row in table.rows)
    narrow = (
        max((len(row) for row in table.rows), default=0) <= internal_STREAM_SPARSE_PROSE_MAX_COLUMNS
    )
    if (
        total_cells
        and narrow
        and len(filled) < total_cells * internal_STREAM_SPARSE_PROSE_MAX_DENSITY
        and long_cells >= len(filled) * internal_STREAM_SPARSE_PROSE_LONG_RATIO
    ):
        return True
    # Word-fragment grids: whitespace alignment over justified prose yields a
    # row per text line and a column per word rail. Cell statistics match a
    # genuine word-y table (short alphabetic cells), but a real one is small
    # -- a body of text produces dozens of rows, and a real table that wide
    # and tall carries numbers.
    columns = max((len(row) for row in table.rows), default=0)
    populated_rows = sum(1 for row in table.rows if any(cell.text.strip() for cell in row))
    if (
        columns >= internal_STREAM_WORD_GRID_MIN_COLUMNS
        and populated_rows >= internal_STREAM_WORD_GRID_MIN_ROWS
        and numeric_cells < len(filled) * internal_STREAM_WORD_GRID_NUMERIC_RATIO
    ):
        lengths = sorted(len(text) for text in filled)
        median_length = lengths[len(lengths) // 2]
        if median_length <= internal_STREAM_WORD_GRID_MEDIAN_CELL_CHARACTERS:
            return True
    return False


def internal_merge_stream_text_columns(table: Table) -> Table:
    """Merge word-aligned columns that form two wrapped text columns."""
    columns = max((len(row) for row in table.rows), default=0)
    if (
        table.metadata.get("source") != "stream"
        or columns < 6
        or columns % 2
        or len(table.rows) < 4
        or internal_numeric_density(table) >= 0.25
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


internal_LOGICAL_ROW_GAP_RATIO = 0.10
internal_LOGICAL_ROW_MIN_GAP = 0.5
internal_LOGICAL_ROW_MIN_ROWS = 4
internal_LOGICAL_ROW_MAX_NUMERIC_RATIO = 0.40
internal_LOGICAL_ROW_MIN_TALL_RATIO = 3.0
internal_LOGICAL_ROW_MIN_COLUMNS = 5


def internal_merge_wrapped_cell_rows(table: Table) -> Table:
    """Group per-line stream rows into the logical rows their cells span.

    Whitespace alignment sees one row per line of text, but a borderless
    table's cells wrap independently: a logical row is as tall as its
    longest cell, and the shorter cells beside it leave the rest of that
    height blank. Emitting the per-line rows reads across all columns at
    each line, interleaving the wrapped fragments of every cell -- the
    content is all present and every word is in the wrong place.

    A cell that continues past the next line's top holds its logical row
    open, so accumulate rows while the next row starts at or above the
    running bottom of the group. Wrapped lines inside a cell touch (the
    leading gap is a fraction of a line), while a genuine row boundary
    clears the cell padding, which the ratio distinguishes.
    """
    if table.metadata.get("source") != "stream" or len(table.rows) < internal_LOGICAL_ROW_MIN_ROWS:
        return table
    # A numeric table records one datum per line, so its lines are its rows;
    # only descriptive tables wrap a cell across several of them. Line
    # spacing alone cannot tell the two apart -- a tightly set list of names
    # separates its records by less than a descriptive table's leading.
    filled = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if filled:
        numeric = sum(
            1
            for text in filled
            if len(text) <= internal_STREAM_PROSE_LONG_CELL_CHARACTERS
            and any(character.isdigit() for character in text)
        )
        if numeric >= len(filled) * internal_LOGICAL_ROW_MAX_NUMERIC_RATIO:
            return table
    if max((len(row) for row in table.rows), default=0) < internal_LOGICAL_ROW_MIN_COLUMNS:
        return table
    extents: list[tuple[float, float]] = []
    heights: list[float] = []
    for row in table.rows:
        boxes = [cell.bbox for cell in row if cell.bbox is not None]
        if not boxes:
            return table
        extents.append((max(box[3] for box in boxes), min(box[1] for box in boxes)))
        heights.extend(box[3] - box[1] for box in boxes)
    if not heights:
        return table
    # Merge only where a cell is demonstrably several lines tall. Spacing
    # alone is too weak a signal: a tightly set table of one-line records
    # separates its rows by less than a wrapped cell's leading, so inferring
    # wrapping from gaps regroups records that were already rows.
    median_height = finite_median(numpy.asarray(heights, dtype=numpy.float32))
    if max(heights) < median_height * internal_LOGICAL_ROW_MIN_TALL_RATIO:
        return table
    tolerance = max(
        internal_LOGICAL_ROW_MIN_GAP,
        median_height * internal_LOGICAL_ROW_GAP_RATIO,
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


def internal_merge_wrapped_stream_rows(table: Table) -> Table:
    """Merge continuation lines in dense, text-only stream tables."""
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


def internal_annotate_table_associations(
    table: Table,
    observations: ObservationBatch,
    text_rows: list[list[int]],
) -> Table:
    """Annotate spanning rows and nearby aligned text without changing cells."""
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
        boxes = [observations.bbox[index] for index in row]
        row_x0 = min(float(box[0]) for box in boxes)
        row_x1 = max(float(box[2]) for box in boxes)
        row_y0 = min(float(box[1]) for box in boxes)
        overlap = interval_overlap(x0, x1, row_x0, row_x1)
        if overlap / max(1.0, min(x1 - x0, row_x1 - row_x0)) < 0.60:
            continue
        gap = row_y0 - y1
        if 0.0 <= gap <= 36.0:
            candidates.append((gap, tuple(row)))
    if candidates:
        _gap, title_row = min(candidates, key=lambda item: item[0])
        text = internal_cell_text(observations, list(title_row))
        if text:
            title = TableAssociatedText(
                text,
                (
                    min(float(observations.bbox[index, 0]) for index in title_row),
                    min(float(observations.bbox[index, 1]) for index in title_row),
                    max(float(observations.bbox[index, 2]) for index in title_row),
                    max(float(observations.bbox[index, 3]) for index in title_row),
                ),
                kind="title",
            )
    if title is table.title and caption is table.caption:
        return table
    return replace(table, title=title, caption=caption)


def internal_table_with_bands(table: Table) -> Table:
    """Materialize table row and column semantics at the extraction boundary."""
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
