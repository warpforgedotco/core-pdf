# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations
import array
import re
from collections.abc import Iterator, Mapping, Sequence
from statistics import median
from typing import Any, Literal, overload
from core_pdf.impl.engine.layout.models import TableGrid, TextRun
from core_pdf.impl.engine.extraction.tables.types import (
    TableBBoxes,
    TableCellSpanRecord,
    TableHeaderResult,
    TableHeaderResultWithBBoxes,
    TableHeuristicResult,
    TableRows,
    TableSet,
    TableSetWithBBoxes,
    TableSpanResult,
    TableSpanResultWithBBoxes,
    TableSpanRows,
)
from core_pdf.impl.engine.extraction.tables.stream import DEFAULT_COLUMN_TOLERANCE
from core_pdf.impl.engine.extraction.tables.stream import DEFAULT_ROW_TOLERANCE
from core_pdf.impl.engine.extraction.tables.stream import infer_stream_columns
from core_pdf.impl.engine.extraction.tables.stream import iter_rows
from bisect import bisect_right


def median_font_size(runs: list[TextRun]) -> float:
    sizes = [run.font_size for run in runs if run.font_size > 0]
    return float(median(sizes)) if sizes else 0.0


def format_run_text(
    run: TextRun,
    *,
    flag_size: bool = False,
    median_font_size: float = 0.0,
) -> str:
    if flag_size and median_font_size > 0 and run.font_size < median_font_size * 0.85:
        return f"<s>{run.text}</s>"
    return run.text


def normalize_cell_text(
    text: str,
    *,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
) -> str:
    value = text
    if not value:
        return ""
    if split_text:
        value = " ".join(value.split())
    if isinstance(strip_text, str):
        for ch in strip_text:
            value = value.replace(ch, "")
    elif strip_text is not None:
        for token in strip_text:
            value = value.replace(str(token), "")
    if replace_text:
        for key, replacement in replace_text.items():
            value = value.replace(str(key), str(replacement))
    return value


def copy_spanning_text(rows: list[list[str]], copy_text: Sequence[str] | None) -> None:
    if not copy_text:
        return
    copy_horizontal = "h" in copy_text
    copy_vertical = "v" in copy_text
    if not (copy_horizontal or copy_vertical):
        return
    for ignored in range(max(1, len(rows)) * 2):
        changed = False
        for row in rows:
            for col_idx, cell in enumerate(row):
                if copy_horizontal and cell == "<span>" and col_idx > 0:
                    replacement = row[col_idx - 1]
                    if replacement != cell:
                        row[col_idx] = replacement
                        changed = True
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                if copy_vertical and cell == "<vspan>" and row_idx > 0:
                    replacement = rows[row_idx - 1][col_idx]
                    if replacement != cell:
                        row[col_idx] = replacement
                        changed = True
        if not changed:
            break


def shift_spanning_text(
    rows: list[list[str]], shift_text: Sequence[str] | None
) -> None:
    if not shift_text:
        return
    if "r" in shift_text:
        for row in rows:
            for col_idx in range(len(row) - 1):
                if row[col_idx + 1] == "<span>" and row[col_idx].strip():
                    row[col_idx + 1] = row[col_idx]
                    row[col_idx] = ""
    if "b" in shift_text:
        for row_idx in range(len(rows) - 1):
            for col_idx in range(len(rows[row_idx])):
                if (
                    rows[row_idx + 1][col_idx] == "<vspan>"
                    and rows[row_idx][col_idx].strip()
                ):
                    rows[row_idx + 1][col_idx] = rows[row_idx][col_idx]
                    rows[row_idx][col_idx] = ""


def postprocess_rows(
    rows: list[list[str]],
    *,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
) -> list[list[str]]:
    shift_spanning_text(rows, shift_text)
    copy_spanning_text(rows, copy_text)
    return rows


def dense_row_slice(rows: list[list[str]]) -> tuple[int, int]:
    if not rows:
        return (0, 0)
    n_cols = max((len(row) for row in rows), default=0)
    if n_cols < 4:
        return (0, len(rows))
    min_populated = max(3, n_cols // 2)
    start = 0
    while start < len(rows):
        populated = sum(1 for cell in rows[start] if cell.strip())
        if populated >= min_populated:
            break
        start += 1
    end = len(rows)
    while end > start:
        populated = sum(1 for cell in rows[end - 1] if cell.strip())
        if populated >= min_populated:
            break
        end -= 1
    return (start, end)


def compact_network_rows(
    rows: TableRows,
    span_grid: TableSpanRows | None = None,
) -> tuple[TableRows, TableSpanRows | None]:
    non_empty: list[tuple[list[str], list[TableCellSpanRecord] | None]] = [
        (row, span_grid[index] if span_grid is not None else None)
        for index, row in enumerate(rows)
        if any(cell.strip() for cell in row)
    ]
    if not non_empty:
        return ([], [] if span_grid is not None else None)
    first_dense = next(
        (
            index
            for index, (row, ignored) in enumerate(non_empty)
            if sum(1 for cell in row if cell.strip()) >= 2
        ),
        None,
    )
    if first_dense is None:
        compact_rows = [row for row, ignored in non_empty]
        compact_spans = (
            [span for ignored, span in non_empty if span is not None]
            if span_grid is not None
            else None
        )
        return (compact_rows, compact_spans)
    start = first_dense
    if first_dense > 0:
        previous = non_empty[first_dense - 1][0]
        if sum(1 for cell in previous if cell.strip()) == 1:
            start = first_dense - 1
    compact: list[tuple[list[str], list[TableCellSpanRecord] | None]] = []
    dense_seen = 0
    for index in range(start, len(non_empty)):
        row, span = non_empty[index]
        populated = sum(1 for cell in row if cell.strip())
        if populated >= 2:
            compact.append((row, span))
            dense_seen += 1
            continue
        if index == start:
            compact.append((row, span))
            continue
        if dense_seen >= 2:
            break
    compact_rows = [row for row, ignored in compact]
    compact_spans = (
        [span for ignored, span in compact if span is not None]
        if span_grid is not None
        else None
    )
    return (compact_rows, compact_spans)


def is_mostly_empty(rows: list[list[str]], reject_threshold: float = 0.95) -> bool:
    cell_count = sum(len(row) for row in rows)
    if cell_count == 0:
        return True
    empty = sum(1 for row in rows for cell in row if not cell.strip())
    return empty / cell_count >= reject_threshold


def iter_cells(
    rows_list: list[list[TextRun]],
    column_positions: list[float],
    *,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    median_font_size: float = 0.0,
    column_tolerance: float = 4.0,
) -> Iterator[
    tuple[list[str], list[TableCellSpanRecord], dict[tuple[int, int], TextRun]]
]:
    row_run_map: dict[tuple[int, int], TextRun] = {}
    for row_idx, row_runs in enumerate(rows_list):
        runs_by_column: list[list[TextRun]] = [[] for ignored in column_positions]
        for run in row_runs:
            col_idx = (
                bisect_right(
                    column_positions, run.x0 + max(1.0, column_tolerance * 0.5)
                )
                - 1
            )
            if 0 <= col_idx < len(column_positions):
                runs_by_column[col_idx].append(run)
        row_cells: list[str] = []
        row_info: list[TableCellSpanRecord] = []
        for col_idx, runs_in_col in enumerate(runs_by_column):
            if runs_in_col:
                text = normalize_cell_text(
                    " ".join(
                        format_run_text(
                            r,
                            flag_size=flag_size,
                            median_font_size=median_font_size,
                        )
                        for r in runs_in_col
                    ),
                    split_text=split_text,
                    strip_text=strip_text,
                    replace_text=replace_text,
                )
                row_cells.append(text)
                n_runs = len(runs_in_col)
                avg_fs = sum(r.font_size for r in runs_in_col) / n_runs
                row_info.append(
                    {
                        "text": text,
                        "row_span": 1,
                        "col_span": 1,
                        "font_size": avg_fs,
                    }
                )
                row_run_map[(row_idx, col_idx)] = runs_in_col[0]
            else:
                row_cells.append("")
                row_info.append(
                    {"text": "", "row_span": 1, "col_span": 1, "font_size": 0}
                )
        yield row_cells, row_info, row_run_map


@overload
def extract_grid(
    runs: list[TextRun],
    grid: TableGrid,
    include_span_info: Literal[True],
    *,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    compact_network: bool = False,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> tuple[TableSet, TableSpanRows]: ...


@overload
def extract_grid(
    runs: list[TextRun],
    grid: TableGrid,
    include_span_info: Literal[False],
    *,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    compact_network: bool = False,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSet: ...


@overload
def extract_grid(
    runs: list[TextRun],
    grid: TableGrid,
    include_span_info: bool,
    *,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    compact_network: bool = False,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSet | tuple[TableSet, TableSpanRows]: ...


def extract_grid(
    runs: list[TextRun],
    grid: TableGrid,
    include_span_info: bool,
    *,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    compact_network: bool = False,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSet | tuple[TableSet, TableSpanRows]:
    n_rows = max(0, len(grid.rows) - 1)
    n_cols = max(0, len(grid.cols) - 1)
    if n_rows == 0 or n_cols == 0:
        return ([], []) if (include_span_info) else []
    text_grid: list[list[str]] = [[""] * n_cols for ignored in range(n_rows)]
    span_grid: TableSpanRows | None = None
    if include_span_info:
        span_grid = [
            [
                {"text": "", "row_span": 1, "col_span": 1, "font_size": 0}
                for ignored in range(n_cols)
            ]
            for ignored in range(n_rows)
        ]
    PRECISION = 2
    rows_desc = grid.rows
    cols_asc = grid.cols
    dense_lookup = False
    if rows_desc[-1] >= 0 and cols_asc[0] >= 0:
        max_y = int(rows_desc[0] * PRECISION) + 1
        max_x = int(cols_asc[-1] * PRECISION) + 1
        if max_y + max_x <= len(runs) * 2:
            row_map = array.array("i", [-1] * max_y)
            for i in range(n_rows):
                y_high = int(rows_desc[i] * PRECISION)
                y_low = int(rows_desc[i + 1] * PRECISION)
                for y in range(y_low, y_high + 1):
                    if 0 <= y < max_y:
                        row_map[y] = i
            col_map = array.array("i", [-1] * max_x)
            for j in range(n_cols):
                x_low = int(cols_asc[j] * PRECISION)
                x_high = int(cols_asc[j + 1] * PRECISION)
                for x in range(x_low, x_high + 1):
                    if 0 <= x < max_x:
                        col_map[x] = j
            dense_lookup = True

            def row_index(y: float) -> int:
                y_index = int(y * PRECISION)
                if 0 <= y_index < max_y:
                    return row_map[y_index]
                return -1

            def col_index(x: float) -> int:
                x_index = int(x * PRECISION)
                if 0 <= x_index < max_x:
                    return col_map[x_index]
                return -1

    if not dense_lookup:
        neg_rows = [-value for value in rows_desc]
        row_index_cache: dict[float, int] = {}
        col_index_cache: dict[float, int] = {}

        def row_index(y: float) -> int:
            cached = row_index_cache.get(y)
            if cached is not None:
                return cached
            index = bisect_right(neg_rows, -y) - 1
            if 0 <= index < n_rows and rows_desc[index] >= y >= rows_desc[index + 1]:
                row_index_cache[y] = index
                return index
            row_index_cache[y] = -1
            return -1

        def col_index(x: float) -> int:
            cached = col_index_cache.get(x)
            if cached is not None:
                return cached
            index = bisect_right(cols_asc, x) - 1
            if 0 <= index < n_cols and cols_asc[index] <= x <= cols_asc[index + 1]:
                col_index_cache[x] = index
                return index
            col_index_cache[x] = -1
            return -1

    median_size = median_font_size(runs) if flag_size else 0.0
    for run in runs:
        if dense_lookup:
            y_index = int(run.mid_y_value * PRECISION)
            if 0 <= y_index < max_y:
                row_idx = row_map[y_index]
            else:
                row_idx = -1
        else:
            row_idx = row_index(run.mid_y)
        if row_idx == -1:
            continue
        if flag_size:
            run_text = format_run_text(
                run, flag_size=True, median_font_size=median_size
            )
        else:
            run_text = run.text
        segments: list[tuple[int, str]]
        if split_text and run.x1 > run.x0 and len(run_text) > 1:
            start_col = col_index(run.x0)
            end_col = col_index(run.x1)
            if start_col == -1:
                start_col = col_index(run.mid_x)
            if end_col == -1:
                end_col = start_col
            if start_col == -1:
                continue
        else:
            if dense_lookup:
                x_index = int(run.mid_x_value * PRECISION)
                if 0 <= x_index < max_x:
                    start_col = col_map[x_index]
                else:
                    start_col = -1
            else:
                start_col = col_index(run.mid_x)
            end_col = start_col
            if start_col == -1:
                continue
        if split_text and end_col > start_col and run.x1 > run.x0 and len(run_text) > 1:
            glyph_chars = run_chars.get(run.seqno) if run_chars else None
            if (
                glyph_chars
                and "".join(char for char, *ignored in glyph_chars) != run_text
            ):
                glyph_chars = None
            if glyph_chars:
                segments = []
                current_col = -1
                current_chars: list[str] = []
                for char, char_x0, char_y0, char_x1, char_y1 in glyph_chars:
                    char_col = col_index((char_x0 + char_x1) * 0.5)
                    if char_col == -1:
                        continue
                    if current_col == -1:
                        current_col = char_col
                    if char_col != current_col:
                        if current_chars:
                            segments.append((current_col, "".join(current_chars)))
                        current_col = char_col
                        current_chars = []
                    current_chars.append(char)
                if current_chars and current_col != -1:
                    segments.append((current_col, "".join(current_chars)))
            else:
                padded_groups = [
                    group for group in re.split(r" {2,}", run_text) if group.strip()
                ]
                spanned_cols = end_col - start_col + 1
                if 1 < len(padded_groups) <= spanned_cols:
                    segments = [
                        (start_col + offset, group)
                        for offset, group in enumerate(padded_groups)
                    ]
                else:
                    width = run.x1 - run.x0
                    text_len = len(run_text)
                    segments = []
                    current_col = -1
                    current_chars = []
                    for char_idx, char in enumerate(run_text):
                        char_x = run.x0 + ((char_idx + 0.5) / text_len) * width
                        char_col = col_index(char_x)
                        if char_col == -1:
                            continue
                        if current_col == -1:
                            current_col = char_col
                        if char_col != current_col:
                            if current_chars:
                                segments.append((current_col, "".join(current_chars)))
                            current_col = char_col
                            current_chars = []
                        current_chars.append(char)
                    if current_chars and current_col != -1:
                        segments.append((current_col, "".join(current_chars)))
        else:
            segments = [(start_col, run_text)]
        for col_idx, segment in segments:
            if not (0 <= col_idx < n_cols):
                continue
            if text_grid[row_idx][col_idx]:
                text_grid[row_idx][col_idx] += " " + segment
            else:
                text_grid[row_idx][col_idx] = segment
            if span_grid is not None:
                span_grid[row_idx][col_idx] = {
                    "text": segment,
                    "row_span": 1,
                    "col_span": 1,
                    "font_size": run.font_size,
                }
    if split_text or strip_text is not None or replace_text:
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                text = text_grid[row_idx][col_idx]
                text_grid[row_idx][col_idx] = normalize_cell_text(
                    text,
                    split_text=split_text,
                    strip_text=strip_text,
                    replace_text=replace_text,
                )
                if span_grid is not None:
                    span_grid[row_idx][col_idx]["text"] = text_grid[row_idx][col_idx]
    elif span_grid is not None:
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                span_grid[row_idx][col_idx]["text"] = text_grid[row_idx][col_idx]
    start_row, end_row = dense_row_slice(text_grid)
    if start_row != 0 or end_row != len(text_grid):
        text_grid = text_grid[start_row:end_row]
        if span_grid is not None:
            span_grid = span_grid[start_row:end_row]
    if compact_network:
        text_grid, compacted_spans = compact_network_rows(text_grid, span_grid)
        span_grid = compacted_spans
    postprocess_rows(text_grid, copy_text=copy_text, shift_text=shift_text)
    if is_mostly_empty(text_grid):
        return ([], []) if include_span_info else []
    if include_span_info:
        return ([text_grid], span_grid or [])
    return [text_grid] if text_grid else []


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: bool,
    include_span_info: Literal[True],
    *,
    include_bboxes: Literal[True],
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSpanResultWithBBoxes: ...


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: bool,
    include_span_info: Literal[True],
    *,
    include_bboxes: Literal[False] = False,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSpanResult: ...


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: Literal[True],
    include_span_info: Literal[False],
    *,
    include_bboxes: Literal[True],
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableHeaderResultWithBBoxes: ...


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: Literal[True],
    include_span_info: Literal[False],
    *,
    include_bboxes: Literal[False] = False,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableHeaderResult: ...


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: Literal[False],
    include_span_info: Literal[False],
    *,
    include_bboxes: Literal[True],
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSetWithBBoxes: ...


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: Literal[False],
    include_span_info: Literal[False],
    *,
    include_bboxes: Literal[False] = False,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableSet: ...


@overload
def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: bool,
    include_span_info: bool,
    *,
    include_bboxes: bool = False,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableHeuristicResult: ...


def extract_heuristic(
    visible_runs: list[TextRun],
    detect_header: bool,
    include_span_info: bool,
    *,
    include_bboxes: bool = False,
    row_tolerance: float = DEFAULT_ROW_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    columns: list[float] | None = None,
    split_text: bool = False,
    strip_text: str | Sequence[str] | None = None,
    replace_text: Mapping[str, str] | None = None,
    flag_size: bool = False,
    copy_text: Sequence[str] | None = None,
    shift_text: Sequence[str] | None = None,
    run_chars: Mapping[int, list[tuple[str, float, float, float, float]]] | None = None,
) -> TableHeuristicResult:
    def heuristic_bbox_for_rows(
        row_groups: list[list[TextRun]],
    ) -> tuple[float, float, float, float] | None:
        runs = [run for row in row_groups for run in row]
        if not runs:
            return None
        return (
            min(run.x0 for run in runs),
            min(run.y0 for run in runs),
            max(run.x1 for run in runs),
            max(run.y1 for run in runs),
        )

    rows_list = list(iter_rows(visible_runs))
    median_size = median_font_size(visible_runs)
    if columns:
        min_x = min((run.x0 for row in rows_list for run in row), default=0.0)
        max_x = max((run.x1 for row in rows_list for run in row), default=0.0)
        column_positions = [min_x]
        column_positions.extend(x for x in sorted(set(columns)) if min_x < x < max_x)
        column_positions.append(max_x)
    else:
        column_positions = infer_stream_columns(
            rows_list,
            column_tolerance=column_tolerance,
            row_tolerance=row_tolerance,
        )
    column_positions.sort()
    rows: TableRows = []
    span_info_list: TableSpanRows = []
    row_run_map: dict[tuple[int, int], TextRun] = {}
    for row_cells, row_info, run_map in iter_cells(
        rows_list,
        column_positions,
        split_text=split_text,
        strip_text=strip_text,
        replace_text=replace_text,
        flag_size=flag_size,
        median_font_size=median_size,
        column_tolerance=column_tolerance,
    ):
        rows.append(row_cells)
        span_info_list.append(row_info)
        row_run_map.update(run_map)
    start_row, end_row = dense_row_slice(rows)
    bbox_rows = rows_list[start_row:end_row]
    if start_row != 0 or end_row != len(rows):
        rows = rows[start_row:end_row]
        span_info_list = span_info_list[start_row:end_row]
        row_run_map = {
            (row_idx - start_row, col_idx): run
            for (row_idx, col_idx), run in row_run_map.items()
            if start_row <= row_idx < end_row
        }
    if include_span_info:
        span_result = apply_heuristic_spans(
            rows, span_info_list, rows_list, column_positions, row_run_map
        )
        for table_rows in span_result[0]:
            postprocess_rows(table_rows, copy_text=copy_text, shift_text=shift_text)
        if include_bboxes:
            return span_result, [heuristic_bbox_for_rows(bbox_rows)] * len(
                span_result[0]
            )
        return span_result
    if not is_plausible_table_rows(rows):
        empty_bboxes: TableBBoxes = []
        if detect_header:
            empty_header: TableHeaderResult = ([], [])
            return (empty_header, empty_bboxes) if include_bboxes else empty_header
        return ([], empty_bboxes) if include_bboxes else []
    if detect_header:
        header_result = detect_heuristic_header(rows, visible_runs)
        if include_bboxes:
            return header_result, [heuristic_bbox_for_rows(bbox_rows)] * len(
                header_result[1]
            )
        return header_result
    postprocess_rows(rows, copy_text=copy_text, shift_text=shift_text)
    tables_result = [rows] if rows else []
    if include_bboxes:
        return tables_result, [heuristic_bbox_for_rows(bbox_rows)] * len(tables_result)
    return tables_result


def is_plausible_table_rows(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    column_count = len(rows[0])
    if column_count < 2:
        return False
    populated_rows = 0
    populated_cells = 0
    for row in rows:
        row_populated_cells = sum(1 for cell in row if cell.strip())
        populated_cells += row_populated_cells
        if row_populated_cells >= 2:
            populated_rows += 1
    if populated_rows < 2 or populated_cells < 4:
        return False
    return populated_cells / (len(rows) * column_count) >= 0.35


def apply_heuristic_spans(
    rows: TableRows,
    span_info_list: TableSpanRows,
    rows_list: list[list[TextRun]],
    column_positions: list[float],
    row_run_map: dict[tuple[int, int], TextRun],
) -> tuple[TableSet, TableSpanRows]:
    n_rows = len(rows)
    n_cols = len(column_positions)
    col_max_widths = [0.0] * n_cols
    for r_idx in range(1, len(rows_list)):
        for r in rows_list[r_idx]:
            c_idx = bisect_right(column_positions, r.x0 + 4.0) - 1
            if 0 <= c_idx < n_cols:
                w = r.x1 - r.x0
                if w > col_max_widths[c_idx]:
                    col_max_widths[c_idx] = w
    for (row_idx, col_idx), run in row_run_map.items():
        if run.font_size > 12 and row_idx < n_rows - 1:
            span_info_list[row_idx][col_idx]["row_span"] = 2
            if row_idx + 1 < n_rows:
                rows[row_idx + 1][col_idx] = "<vspan>"
                span_info_list[row_idx + 1][col_idx] = {
                    "text": "<vspan>",
                    "row_span": 0,
                    "col_span": 1,
                    "font_size": 0,
                }
        if row_idx == 0 and col_idx + 1 < n_cols:
            text_width = run.x1 - run.x0
            max_below_width = col_max_widths[col_idx]
            if max_below_width > 0 and text_width > max_below_width * 3:
                rows[row_idx][col_idx + 1] = "<span>"
                span_info_list[row_idx][col_idx] = {
                    "text": run.text,
                    "row_span": 1,
                    "col_span": 2,
                    "font_size": run.font_size,
                }
                span_info_list[row_idx][col_idx + 1] = {
                    "text": "<span>",
                    "row_span": 0,
                    "col_span": 0,
                    "font_size": 0,
                }
    return ([rows], span_info_list)


def detect_heuristic_header(
    rows: list[list[str]], visible_runs: list[TextRun]
) -> tuple[list[list[str]], list[list[list[str]]]]:
    if rows and visible_runs:
        header_font_size = visible_runs[0].font_size
        other_sizes = [r.font_size for r in visible_runs[1:]]
        avg = sum(other_sizes) / len(other_sizes) if other_sizes else header_font_size
        if header_font_size > avg:
            header_cells = [cell for cell in rows[0] if cell.strip()]
            return ([header_cells], [rows[1:]])
    return ([], [rows])
