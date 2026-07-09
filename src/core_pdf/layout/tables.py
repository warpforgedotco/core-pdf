from __future__ import annotations

import array
from bisect import bisect_left, bisect_right
from typing import Any, Iterator

from core_pdf.layout.models import TableGrid, TextRun
from core_pdf.layout.ordering import cluster_runs_into_lines
from core_pdf.layout.traces import CapturedLine

GRID_SNAP_TOLERANCE: float = 2.0
MIN_CELL_WIDTH: float = 4.0
MIN_CELL_HEIGHT: float = 4.0


class TableExtractor:
    """Namespace for table extraction and grid detection algorithms."""

    @staticmethod
    def snap(values: list[float] | array.array[float], tolerance: float) -> list[float]:
        if not values:
            return []

        sorted_vals = sorted(values)
        clusters: list[list[float]] = [[sorted_vals[0]]]
        last_cluster = clusters[0]

        for v in sorted_vals[1:]:
            if v - last_cluster[-1] <= tolerance:
                last_cluster.append(v)
            else:
                last_cluster = [v]
                clusters.append(last_cluster)

        return [sum(c) * (1.0 / len(c)) for c in clusters]

    @staticmethod
    def detect_grid(lines: list[CapturedLine]) -> TableGrid | None:
        if not lines:
            return None
        horizontals = array.array("f")
        verticals = array.array("f")
        for line in lines:
            dx = abs(line.x1 - line.x0)
            dy = abs(line.y1 - line.y0)
            if dx > dy:
                horizontals.append((line.y0 + line.y1) * 0.5)
            elif dy > dx:
                verticals.append((line.x0 + line.x1) * 0.5)
        h_clusters = TableExtractor.snap(horizontals, GRID_SNAP_TOLERANCE)
        v_clusters = TableExtractor.snap(verticals, GRID_SNAP_TOLERANCE)
        if len(h_clusters) < 2 or len(v_clusters) < 2:
            return None
        rows = sorted(set(h_clusters), reverse=True)
        cols = sorted(set(v_clusters))
        if len(rows) < 2 or len(cols) < 2:
            return None
        row_starts = []
        for i in range(len(rows) - 1):
            if rows[i] - rows[i + 1] >= MIN_CELL_HEIGHT:
                row_starts.append(rows[i])
        row_starts.append(rows[-1])
        col_starts = []
        for i in range(len(cols) - 1):
            if cols[i + 1] - cols[i] >= MIN_CELL_WIDTH:
                col_starts.append(cols[i])
        col_starts.append(cols[-1])
        if len(row_starts) < 2 or len(col_starts) < 2:
            return None
        return TableGrid(cols=col_starts, rows=row_starts)

    @staticmethod
    def detect_stream_grid(runs: list[TextRun]) -> TableGrid | None:
        """Detect table grid based on text alignment (borderless tables)."""
        if not runs:
            return None

        rows = cluster_runs_into_lines(
            runs, lookback=1, sort_key=lambda r: (-(r.y0 + r.y1) * 0.5, r.x0)
        )

        if len(rows) < 2:
            return None

        x0_positions = array.array("f")
        for row in rows:
            for run in row:
                x0_positions.append(run.x0)

        snapped_x = array.array("f", TableExtractor.snap(x0_positions, 8.0))
        if len(snapped_x) < 2:
            return None

        snapped_x_list = list(snapped_x)
        col_counts = [0] * len(snapped_x_list)
        for row in rows:
            row_x0s = [run.x0 for run in row]
            for i, cx in enumerate(snapped_x_list):
                idx = bisect_left(row_x0s, cx - 8.0)
                if idx < len(row_x0s) and row_x0s[idx] - cx < 8.0:
                    col_counts[i] += 1

        min_hits = max(2, round(len(rows) * 0.2))
        final_cols = [cx for i, cx in enumerate(snapped_x_list) if col_counts[i] >= min_hits]
        final_cols.sort()

        if len(final_cols) < 2:
            return None

        filtered_cols = [final_cols[0]]
        for i in range(1, len(final_cols)):
            if final_cols[i] - filtered_cols[-1] >= MIN_CELL_WIDTH:
                filtered_cols.append(final_cols[i])

        if len(filtered_cols) < 2:
            return None

        row_bounds = []
        for row in rows:
            row_y0 = min(r.y0 for r in row)
            row_y1 = max(r.y1 for r in row)
            row_bounds.append(row_y1)
            row_bounds.append(row_y0)

        final_rows = TableExtractor.snap(row_bounds, 2.0)
        final_rows.sort(reverse=True)

        if len(final_rows) < 2:
            return None

        return TableGrid(cols=filtered_cols, rows=final_rows)

    @staticmethod
    def iter_rows(visible_runs: list[TextRun]) -> Iterator[list[TextRun]]:
        clusters = cluster_runs_into_lines(
            visible_runs, lookback=1, sort_key=lambda r: (-r.mid_y, r.x0)
        )
        for row in clusters:
            yield row

    @staticmethod
    def iter_cells(
        rows_list: list[list[TextRun]], column_positions: list[float]
    ) -> Iterator[tuple[list[str], list[dict[str, Any]], dict[tuple[int, int], TextRun]]]:
        row_run_map: dict[tuple[int, int], TextRun] = {}

        bl = bisect_left
        br = bisect_right

        for row_idx, row_runs in enumerate(rows_list):
            row_x0s = [r.x0 for r in row_runs]
            row_cells: list[str] = []
            row_info: list[dict[str, Any]] = []

            for col_idx, col_x in enumerate(column_positions):
                left = bl(row_x0s, col_x - 8.0)
                right = br(row_x0s, col_x + 8.0)
                runs_in_col = row_runs[left:right]
                if runs_in_col:
                    text = " ".join(r.text for r in runs_in_col)
                    row_cells.append(text)
                    n_runs = len(runs_in_col)
                    avg_fs = sum(r.font_size for r in runs_in_col) / n_runs
                    row_info.append(
                        {"text": text, "row_span": 1, "col_span": 1, "font_size": avg_fs}
                    )
                    row_run_map[(row_idx, col_idx)] = runs_in_col[0]
                else:
                    row_cells.append("")
                    row_info.append({"text": "", "row_span": 1, "col_span": 1, "font_size": 0})
            yield row_cells, row_info, row_run_map

    @staticmethod
    def extract_grid(
        runs: list[TextRun],
        grid: TableGrid,
        include_span_info: bool,
    ) -> Any:
        n_rows = max(0, len(grid.rows) - 1)
        n_cols = max(0, len(grid.cols) - 1)
        if n_rows == 0 or n_cols == 0:
            return ([], []) if (include_span_info) else []

        text_grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]
        span_grid: list[list[dict]] = [
            [{"text": "", "row_span": 1, "col_span": 1, "font_size": 0} for _ in range(n_cols)]
            for _ in range(n_rows)
        ]

        PRECISION = 2
        rows_desc = grid.rows
        cols_asc = grid.cols

        max_y = int(rows_desc[0] * PRECISION) + 1
        max_x = int(cols_asc[-1] * PRECISION) + 1

        row_map = array.array("i", [-1] * max_y)
        for i in range(n_rows):
            y_high = int(rows_desc[i] * PRECISION)
            y_low = int(rows_desc[i + 1] * PRECISION)
            for y in range(y_low, y_high + 1):
                if y < max_y:
                    row_map[y] = i

        col_map = array.array("i", [-1] * max_x)
        for j in range(n_cols):
            x_low = int(cols_asc[j] * PRECISION)
            x_high = int(cols_asc[j + 1] * PRECISION)
            for x in range(x_low, x_high + 1):
                if x < max_x:
                    col_map[x] = j

        for run in runs:
            cy_idx = int(run.mid_y * PRECISION)
            cx_idx = int(run.mid_x * PRECISION)

            if 0 <= cy_idx < max_y:
                row_idx = row_map[cy_idx]
                if row_idx != -1 and 0 <= cx_idx < max_x:
                    col_idx = col_map[cx_idx]
                    if col_idx != -1:
                        if text_grid[row_idx][col_idx]:
                            text_grid[row_idx][col_idx] += " " + run.text
                        else:
                            text_grid[row_idx][col_idx] = run.text

                        span_grid[row_idx][col_idx] = {
                            "text": run.text,
                            "row_span": 1,
                            "col_span": 1,
                            "font_size": run.font_size,
                        }

        if include_span_info:
            return ([text_grid], span_grid)

        return [text_grid] if text_grid else []

    @staticmethod
    def extract_heuristic(
        visible_runs: list[TextRun],
        detect_header: bool,
        include_span_info: bool,
    ) -> Any:
        rows_list = list(TableExtractor.iter_rows(visible_runs))

        all_x0s = [run.x0 for row in rows_list for run in row]
        column_positions = TableExtractor.snap(all_x0s, 8.0)
        column_positions.sort()

        rows: list[list[str]] = []
        span_info_list: list[list[dict[str, Any]]] = []
        row_run_map: dict[tuple[int, int], TextRun] = {}

        for row_cells, row_info, run_map in TableExtractor.iter_cells(rows_list, column_positions):
            rows.append(row_cells)
            span_info_list.append(row_info)
            row_run_map.update(run_map)

        if include_span_info:
            return TableExtractor.apply_heuristic_spans(
                rows, span_info_list, rows_list, column_positions, row_run_map
            )

        if detect_header:
            return TableExtractor.detect_heuristic_header(rows, visible_runs)

        return [rows] if rows else []

    @staticmethod
    def apply_heuristic_spans(
        rows: list[list[str]],
        span_info_list: list[list[dict[str, Any]]],
        rows_list: list[list[TextRun]],
        column_positions: list[float],
        row_run_map: dict[tuple[int, int], TextRun],
    ) -> tuple[list[list[list[str]]], list[list[dict[str, Any]]]]:
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

    @staticmethod
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
